"""Hadronisation: fragmentation chains, species, pi0 decays and B hadrons."""

import math

import numpy as np
import pytest

from helpers import KS_LIMIT, N_SIGMA, assert_binomial, ks_distance
from itj.toy.config import HadronPart
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
    species_probabilities,
)
from itj.toy.kinematics import mass
from itj.toy.shower import PartonArrays


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
