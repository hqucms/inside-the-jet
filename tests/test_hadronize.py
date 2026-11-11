"""Hadronisation: fragmentation chains, species, pi0 decays and B hadrons."""

import math

import numpy as np
import pytest

from helpers import KS_LIMIT, N_SIGMA, assert_binomial, ks_distance
from itj.toy.config import HadronPart, preset
from itj.toy.hadronize import (
    B_HADRON,
    B_MASS,
    ELECTRON,
    MASSES,
    MUON,
    NEUTRINO,
    PHOTON,
    PI0,
    PI_CH,
    HadronArrays,
    _b_decays,
    _fragment,
    decay_pi0,
    draw_species,
    ff_z,
    hadronize,
    reshuffle_hadrons,
    singlet_boosts,
    species_probabilities,
)
from itj.toy.hardprocess import sample_hard
from itj.toy.kinematics import boost, mass
from itj.toy.shower import PartonArrays, shower_events


def _partons(energies, *, hard_b=False, events=None):
    energies = np.asarray(energies, dtype=np.float64)
    m = len(energies)
    rng = np.random.default_rng(0)
    n = rng.normal(size=(m, 3)) * 0.1 + np.array([1.0, 0.0, 0.0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    event = np.arange(m) if events is None else np.asarray(events)
    return PartonArrays(
        p4=np.concatenate([energies[:, None] * n, energies[:, None]], axis=1),
        pid=np.where(hard_b, 5, 1) * np.ones(m, dtype=np.int64),
        event=event.astype(np.int64),
        prong=np.zeros(m, dtype=np.int64),
        node=np.zeros(m, dtype=np.int64),
        hard_b=np.full(m, hard_b),
    )


def test_species_probabilities():
    cfg = HadronPart()
    p = species_probabilities(cfg)
    assert p.sum() == pytest.approx(1.0)
    assert p[2] == pytest.approx(cfg.pi0_fraction)
    assert p[3] + p[4] == pytest.approx(cfg.neutral_fraction)
    assert p[3] / (p[3] + p[4]) == pytest.approx(cfg.klong_fraction)
    assert p[1] / (p[0] + p[1]) == pytest.approx(cfg.kaon_fraction)


def test_draw_species_frequencies_and_mass_limit():
    rng = np.random.default_rng(1)
    probs = species_probabilities(HadronPart())
    n = 50_000
    species = draw_species(rng.random(n), probs)
    codes = (PI_CH, 1, PI0, 3, 4)
    for code, p in zip(codes, probs, strict=True):
        assert_binomial((species == code).sum(), n, p)
    limited = draw_species(rng.random(n), probs, np.full(n, 0.3))
    assert set(np.unique(limited)) <= {PI_CH, PI0}
    assert np.all(draw_species(rng.random(10), probs, np.full(10, 0.01)) == PI_CH)


@pytest.mark.parametrize("a", [0.0, 0.3, 0.77, 1.0])
def test_field_feynman_z(a):
    z = ff_z(np.random.default_rng(2).random(20_000), a)
    assert np.all((z >= 0) & (z <= 1))

    def cdf(x):
        w = 1.0 - x
        return 1.0 - (1.0 - a) * w - a * w**3

    assert ks_distance(z, cdf) < KS_LIMIT


def test_pi0_decay():
    rng = np.random.default_rng(3)
    p3 = rng.normal(0, 5.0, (1000, 3))
    m = MASSES[PI0]
    pi0 = np.concatenate([p3, np.sqrt(np.sum(p3 * p3, 1) + m * m)[:, None]], axis=1)
    g1, g2 = decay_pi0(pi0, rng.random((2, 1000)))
    np.testing.assert_allclose(g1 + g2, pi0, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(mass(g1 + g2), m, rtol=1e-6)
    np.testing.assert_allclose(mass(g1) / g1[:, 3], 0.0, atol=1e-6)


def test_chains_conserve_energy_and_transverse_momentum():
    partons = _partons(np.linspace(2.0, 300.0, 400))
    stats = {}
    p4, species, _, chain, rank = _fragment(partons, HadronPart(), np.random.default_rng(4), stats)
    energy = np.bincount(chain, weights=p4[:, 3], minlength=len(partons))
    np.testing.assert_allclose(energy, partons.p4[:, 3], rtol=1e-12)
    # hadrons are on shell
    np.testing.assert_allclose(mass(p4), MASSES[species], atol=2e-5)
    # sum of the transverse momenta (about the parton) vanishes for chains without a capped hadron
    n = partons.p4[:, :3] / np.linalg.norm(partons.p4[:, :3], axis=1, keepdims=True)
    along = np.sum(p4[:, :3] * n[chain], axis=1)
    perp = p4[:, :3] - along[:, None] * n[chain]
    p_abs = np.linalg.norm(p4[:, :3], axis=1)
    capped = np.abs(np.linalg.norm(perp, axis=1) - 0.9 * p_abs) < 1e-9 * np.maximum(p_abs, 1.0)
    chain_capped = np.bincount(chain, weights=capped, minlength=len(partons)) > 0
    total = np.zeros((len(partons), 3))
    np.add.at(total, chain, perp)
    assert (~chain_capped).sum() > 150
    np.testing.assert_allclose(total[~chain_capped], 0.0, atol=1e-9)
    assert 0 < stats["hadrons_capped"] < 0.2 * len(p4)
    # ranks are consecutive per chain
    assert np.all(rank[np.r_[True, chain[1:] != chain[:-1]]] == 0)


def test_hadronize_conserves_energy_without_neutrinos():
    partons = _partons(np.linspace(5.0, 400.0, 300), hard_b=True)
    cfg = HadronPart(semileptonic=0.0, semileptonic_cascade=0.0)
    hadrons = hadronize(partons, cfg, np.random.default_rng(5))
    energy = np.bincount(hadrons.event, weights=hadrons.p4[:, 3], minlength=len(partons))
    np.testing.assert_allclose(energy, partons.p4[:, 3], rtol=1e-9)
    assert not np.any(hadrons.species == PI0)
    assert not np.any(hadrons.species == B_HADRON)
    assert hadrons.from_b.any()
    # production order: sorted by event, index counts per event
    assert np.all(np.diff(hadrons.event) >= 0)
    for e in (0, 17, 299):
        np.testing.assert_array_equal(hadrons.index[hadrons.event == e], np.arange((hadrons.event == e).sum()))
    # charges: +-1 for charged hadrons, 0 for photons
    assert set(np.unique(hadrons.charge[hadrons.species == PI_CH])) <= {-1, 1}
    assert np.all(hadrons.charge[hadrons.species == PHOTON] == 0)


def test_neutrinos_carry_energy_away():
    partons = _partons(np.full(300, 200.0), hard_b=True)
    cfg = HadronPart(semileptonic=1.0, semileptonic_cascade=0.0)
    hadrons = hadronize(partons, cfg, np.random.default_rng(6))
    energy = np.bincount(hadrons.event, weights=hadrons.p4[:, 3], minlength=len(partons))
    assert np.all(energy < 200.0)
    assert not np.any(hadrons.species == NEUTRINO)
    assert np.sum(hadrons.species == ELECTRON) >= 300
    assert np.sum(hadrons.species == MUON) >= 300


def test_b_decays():
    rng = np.random.default_rng(7)
    nb = 20_000
    energy = rng.uniform(20.0, 300.0, nb)
    p = np.sqrt(energy**2 - B_MASS**2)
    b = np.stack([p, np.zeros(nb), np.zeros(nb), energy], axis=1)
    cfg = HadronPart()
    d_p4, species, charge, parent, slot, vertex = _b_decays(b, cfg, rng)
    total = np.zeros((nb, 4))
    np.add.at(total, parent, d_p4)
    np.testing.assert_allclose(total, b, rtol=1e-9, atol=1e-9)
    assert np.all(np.bincount(parent) == cfg.b_decay_multiplicity)
    # flight length: L = beta gamma c tau', mean c tau
    flight = np.linalg.norm(vertex[:, :3], axis=1) / (p / B_MASS)
    assert abs(flight.mean() - cfg.b_ctau) < N_SIGMA * cfg.b_ctau / math.sqrt(nb)
    assert ks_distance(flight, lambda x: 1.0 - np.exp(-x / cfg.b_ctau)) < KS_LIMIT
    np.testing.assert_allclose(vertex[:, 3], flight * energy / B_MASS, rtol=1e-9)  # x^mu = c tau' p^mu / m
    # leptons: semileptonic + cascade electrons, each lepton with a neutrino
    electrons = np.bincount(parent[species == ELECTRON], minlength=nb)
    expected = cfg.semileptonic + cfg.semileptonic_cascade
    var = cfg.semileptonic * (1 - cfg.semileptonic) + cfg.semileptonic_cascade * (1 - cfg.semileptonic_cascade)
    assert abs(electrons.mean() - expected) < N_SIGMA * math.sqrt(var / nb)
    leptons = np.isin(species, (ELECTRON, MUON)).sum()
    assert (species == NEUTRINO).sum() == leptons
    hadronic = ~np.isin(species, (ELECTRON, MUON, NEUTRINO))
    assert_binomial((species[hadronic] == PI_CH).sum(), hadronic.sum(), cfg.b_charged_share)
    assert np.all(charge[species == PHOTON] == 0)
    assert np.all(np.abs(charge[species == PI_CH]) == 1)
    assert slot.max() == cfg.b_decay_multiplicity - 1


def test_b_fragmentation_mean():
    partons = _partons(np.full(4000, 300.0), hard_b=True)
    cfg = HadronPart()
    p4, species, _, _, rank = _fragment(partons, cfg, np.random.default_rng(8), {})
    first = (rank == 0) & (species == B_HADRON)
    assert first.sum() == 4000
    z = p4[first, 3] / 300.0
    sd = math.sqrt(np.var(z) / len(z))
    assert abs(z.mean() - cfg.b_frag_mean) < N_SIGMA * sd + 1e-3


def test_light_partons_make_no_b_hadrons_and_from_partons():
    partons = _partons(np.full(50, 100.0))
    _, species, *_ = _fragment(partons, HadronPart(), np.random.default_rng(9), {})
    assert not np.any(species == B_HADRON)
    as_particles = HadronArrays.from_partons(_partons([10.0, 20.0, 30.0], events=[0, 0, 1]))
    np.testing.assert_array_equal(as_particles.index, [0, 1, 0])
    np.testing.assert_array_equal(as_particles.parton, [0, 1, 0])
    assert not as_particles.is_ue.any()


# ----------------------------------------------------------------------------------------------------------------------
# W and top: the singlet frame and the hadron-level reshuffle


def _resonance_partons(code, n=300, seed=11):
    cfg = preset("hadrons")
    rng = np.random.default_rng(seed)
    hard = sample_hard(n, code, cfg, rng)
    partons, _ = shower_events(hard, cfg, rng)
    return partons


def _sum(index, p4, n):
    total = np.zeros((n, 4))
    np.add.at(total, index, p4)
    return total


@pytest.mark.parametrize("singlet", [False, True])
@pytest.mark.parametrize("code", [1, 2])
def test_hadron_reshuffle_restores_the_resonance(code, singlet):
    partons = _resonance_partons(code)
    n = int(partons.event.max()) + 1
    resonance = _sum(partons.event, partons.p4, n)
    # without neutrinos every stable particle is kept: the hadrons sum to the parton-level resonance
    cfg = HadronPart(hadron_reshuffle=True, singlet_frame=singlet, semileptonic=0.0, semileptonic_cascade=0.0)
    stats = {}
    hadrons = hadronize(partons, cfg, np.random.default_rng(12), stats)
    steps = n if code == 1 else 2 * n
    assert stats["hadron_reshuffles"] == steps
    assert stats["hadron_reshuffle_failed"] < (0.3 if code == 1 else 0.1) * steps
    if singlet:
        assert stats["hadron_reshuffle_failed"] <= 0.01 * steps
    total = _sum(hadrons.event, hadrons.p4, n)
    # the events whose systems were too heavy are left unchanged (their total mass is above the resonance mass)
    exact = np.all(np.abs(total - resonance) <= 1e-9 * resonance[:, 3:], axis=1)
    assert n - exact.sum() <= stats["hadron_reshuffle_failed"]
    assert np.all(mass(total[~exact]) > mass(resonance[~exact]))
    assert 0.0 < stats["hadron_reshuffle_residual"] < 1e-8
    # with the B decays' neutrinos the generator still measures the identity (neutrinos included)
    stats = {}
    hadronize(partons, HadronPart(hadron_reshuffle=True, semileptonic=1.0), np.random.default_rng(12), stats)
    assert stats["hadron_reshuffle_residual"] < 1e-8
    # on shell: the chain hadrons were boosted rigidly
    charged = hadrons.species == PI_CH
    np.testing.assert_allclose(mass(hadrons.p4[charged & ~hadrons.from_b]), MASSES[PI_CH], rtol=1e-4)


def test_hadron_reshuffle_keeps_the_w_inside_the_top():
    partons = _resonance_partons(2)
    p4, _, _, chain, _ = _fragment(partons, HadronPart(), np.random.default_rng(13), {})
    stats = {}
    moved, restored = reshuffle_hadrons(partons, p4, chain, stats)
    n = len(restored)
    in_w = partons.prong >= 1
    w_partons = _sum(partons.event[in_w], partons.p4[in_w], n)
    w_hadrons = _sum(partons.event[chain[in_w[chain]]], moved[in_w[chain]], n)
    top = _sum(partons.event[chain], moved, n)
    # the top and the mass of its W are exact; the W momentum takes the recoil of the b system
    np.testing.assert_allclose(top[restored], _sum(partons.event, partons.p4, n)[restored], rtol=1e-10, atol=1e-8)
    w_ok = restored & (np.abs(mass(w_hadrons) - mass(w_partons)) < 1e-6)
    assert w_ok.sum() >= 0.9 * n
    # the systems move rigidly: the mass of every b system is kept
    b = ~in_w[chain]
    np.testing.assert_allclose(
        mass(_sum(partons.event[chain[b]], moved[b], n)), mass(_sum(partons.event[chain[b]], p4[b], n)), rtol=1e-7
    )


def test_hadron_reshuffle_switch_changes_only_the_momenta_of_resonances():
    qcd = _partons(np.linspace(20.0, 300.0, 50))
    for partons in (qcd, _resonance_partons(1, n=100), _resonance_partons(2, n=100)):
        off = hadronize(partons, HadronPart(), np.random.default_rng(14))
        on = hadronize(partons, HadronPart(hadron_reshuffle=True), np.random.default_rng(14))
        # no random numbers are drawn: the same particles, in the same order, with the same truth
        for name in ("species", "charge", "event", "index", "prong", "parton", "from_b", "is_ue"):
            np.testing.assert_array_equal(getattr(on, name), getattr(off, name))
        if partons is qcd:
            np.testing.assert_array_equal(on.p4, off.p4)  # QCD jets are not reshuffled
        else:
            assert not np.allclose(on.p4, off.p4)
            # the B-decay vertices follow the moved B hadrons: still parallel to the momentum of their parent
            np.testing.assert_array_equal(on.vertex == 0, off.vertex == 0)


def test_singlet_frame():
    for code in (1, 2):
        partons = _resonance_partons(code, n=200)
        beta = singlet_boosts(partons)
        in_w = partons.prong >= (code - 1)
        assert np.all(beta[~in_w] == 0)
        assert np.all(np.linalg.norm(beta[in_w], axis=1) > 0.5)
        # in that frame the W quarks sum to zero three-momentum
        rest = boost(partons.p4, -beta)
        w = _sum(partons.event[in_w], rest[in_w], int(partons.event.max()) + 1)
        np.testing.assert_allclose(w[:, :3], 0.0, atol=1e-9 * w[:, 3:].max())
    assert np.all(singlet_boosts(_partons([50.0, 80.0])) == 0)
    # the W quarks fragment as at the W mass: fewer hadrons; the energy is conserved in the W rest frame
    partons = _resonance_partons(1)
    lab = hadronize(partons, HadronPart(), np.random.default_rng(15))
    singlet = hadronize(partons, HadronPart(singlet_frame=True), np.random.default_rng(15))
    assert len(singlet) < 0.8 * len(lab)
    n = int(partons.event.max()) + 1
    w = _sum(partons.event, partons.p4, n)
    beta = w[:, :3] / w[:, 3:]
    energy = _sum(singlet.event, boost(singlet.p4, -beta[singlet.event]), n)[:, 3]
    np.testing.assert_allclose(energy, mass(w), rtol=1e-9)


def test_w_mass_peak_with_the_singlet_frame_and_the_reshuffle():
    from itj.toy import generate

    def median_msd(config):
        data = generate(400, config=config, processes=("wqq",), seed=0)
        return float(np.median(np.asarray(data["jet_sdmass"], dtype=np.float64)))

    shower = median_msd(preset("shower"))
    off = median_msd(preset("hadrons", ue=False))
    lab = median_msd(preset("hadrons", ue=False, hadron_reshuffle=True))
    singlet = median_msd(preset("hadrons", ue=False, hadron_reshuffle=True, singlet_frame=True))
    # the chains raise the W peak by about 4 GeV; restoring the whole W in the generation frame overshoots by about
    # 10 GeV (the extra mass sits in slow hadrons outside the groomed jet); in the W frame the peak returns
    assert off > shower + 2.0
    assert lab < shower - 5.0
    assert abs(singlet - shower) < 1.5 < off - shower
