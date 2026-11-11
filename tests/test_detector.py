"""Detector: tracking efficiency, impact parameters, calorimeter cells, smearing, keyed randomness."""

import math

import numpy as np
import pytest

from helpers import N_SIGMA, assert_binomial
from itj.toy.config import DetectorPart
from itj.toy.detector import ParticleArrays, cell_phi_bins, detector, impact_parameters, track_efficiency
from itj.toy.hadronize import ELECTRON, MASSES, MUON, NEUTRON, PHOTON, PI_CH, HadronArrays

PERFECT = {
    "ecal_res_const": 0.0,
    "ecal_res_stoch": 0.0,
    "ecal_res_noise": 0.0,
    "hcal_res_const": 0.0,
    "hcal_res_stoch": 0.0,
    "fwd_res_const": 0.0,
    "fwd_res_stoch": 0.0,
}


def _hadrons(pt, eta, phi, species, *, event=None, is_ue=None, vertex=None, charge=None):
    pt, eta, phi = (np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in (pt, eta, phi))
    k = max(len(pt), len(eta), len(phi), len(np.atleast_1d(species)))
    pt, eta, phi = (np.broadcast_to(x, (k,)) for x in (pt, eta, phi))
    species = np.broadcast_to(np.asarray(species, dtype=np.int8), (k,)).copy()
    m = MASSES[species]
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m * m)], 1)
    event = np.arange(k) if event is None else np.asarray(event)
    is_ue = np.zeros(k, dtype=bool) if is_ue is None else np.asarray(is_ue)
    if charge is None:
        charge = np.where(np.isin(species, (PI_CH, ELECTRON, MUON)), 1, 0)
    index = np.zeros(k, dtype=np.int64)
    for flag in (False, True):  # index among the hadrons of the same origin of each event
        sel = np.flatnonzero(is_ue == flag)
        index[sel] = np.arange(len(sel)) - np.searchsorted(event[sel], event[sel])
    return HadronArrays(
        p4=p4,
        vertex=np.zeros((k, 4)) if vertex is None else vertex,
        species=species,
        charge=np.asarray(charge, dtype=np.int8),
        event=event.astype(np.int64),
        index=index,
        prong=np.where(is_ue, -1, 0),
        parton=np.where(is_ue, -1, 0),
        from_b=np.zeros(k, dtype=bool),
        is_ue=is_ue,
    )


def _run(hadrons, cfg=None, seed=1):
    n_events = int(hadrons.event.max()) + 1 if len(hadrons) else 0
    return detector(hadrons, cfg or DetectorPart(), (seed, np.arange(n_events)))


@pytest.mark.parametrize(
    ("pt", "eta", "species", "expected"),
    [
        (5.0, 0.0, PI_CH, 0.95),
        (0.5, 0.3, PI_CH, 0.70),
        (5.0, 2.0, PI_CH, 0.85),
        (0.5, -2.0, PI_CH, 0.60),
        (5.0, 0.0, ELECTRON, 0.95),
        (0.5, 2.0, MUON, 0.95),
        (5.0, 2.7, PI_CH, 0.0),
        (0.08, 0.0, PI_CH, 0.0),
    ],
)
def test_track_efficiency(pt, eta, species, expected):
    n = 20_000
    rng = np.random.default_rng(2)
    hadrons = _hadrons(pt, eta, rng.uniform(-np.pi, np.pi, n), species)
    out = _run(hadrons)
    tracks = out.charge != 0
    if expected == 0:
        assert tracks.sum() == 0
    else:
        assert_binomial(tracks.sum(), n, expected)
    lepton = np.array([species in (ELECTRON, MUON)])
    assert float(track_efficiency(np.array([pt]), np.array([eta]), lepton, DetectorPart())[0]) == expected


def test_untracked_charged_hadrons_reach_the_hcal():
    n = 500
    hadrons = _hadrons(5.0, np.linspace(-0.5, 0.5, n), 0.1, PI_CH)
    out = _run(hadrons, DetectorPart(track_eff_high=0.0, **PERFECT))
    assert len(out) == n
    np.testing.assert_array_equal(out.flags[:, 1], 1)  # neutral hadrons
    np.testing.assert_array_equal(out.charge, 0.0)
    np.testing.assert_allclose(out.p4[:, 3], hadrons.p4[:, 3], rtol=1e-12)
    np.testing.assert_array_equal(out.purity, 1.0)


def test_untracked_leptons():
    electrons = _run(_hadrons(5.0, 0.2, 0.1, ELECTRON), DetectorPart(lepton_eff=0.0, **PERFECT))
    assert len(electrons) == 1
    assert electrons.flags[0, 2] == 1  # an ECAL photon
    muons = _run(_hadrons(5.0, 0.2, 0.1, MUON), DetectorPart(lepton_eff=0.0))
    assert len(muons) == 0
    tracked = _run(_hadrons([5.0, 5.0], 0.2, [0.1, 1.0], [ELECTRON, MUON]), DetectorPart(lepton_eff=1.0))
    np.testing.assert_array_equal(tracked.flags[:, 3:], [[1, 0], [0, 1]])


def test_cells_sum_energies_before_smearing():
    # two photons in one ECAL cell and one in another; one photon below the threshold in a third cell
    cell = 0.0174
    eta = np.array([0.5 * cell, 0.6 * cell, 10.5 * cell, -5.5 * cell])
    hadrons = _hadrons([3.0, 1.0, 2.0, 0.3], eta, [0.0005, 0.001, 0.3, -0.4], PHOTON, event=np.zeros(4, dtype=int))
    out = _run(hadrons, DetectorPart(**PERFECT))
    assert len(out) == 2
    order = np.argsort(-out.p4[:, 3])
    e = out.p4[order, 3]
    np.testing.assert_allclose(e, [hadrons.p4[0, 3] + hadrons.p4[1, 3], hadrons.p4[2, 3]], rtol=1e-12)
    np.testing.assert_allclose(out.purity[order][0], hadrons.p4[0, 3] / e[0], rtol=1e-12)
    np.testing.assert_array_equal(out.flags[:, 2], 1)
    np.testing.assert_allclose(out.p4[:, 3] ** 2 - np.sum(out.p4[:, :3] ** 2, 1), 0.0, atol=1e-9)  # massless


def test_cell_positions_centre_and_random():
    n = 3000
    rng = np.random.default_rng(3)
    eta, phi = rng.uniform(-1.4, 1.4, n), rng.uniform(-np.pi, np.pi, n)
    hadrons = _hadrons(20.0, eta, phi, NEUTRON, charge=np.zeros(n))
    for position in ("centre", "random"):
        out = _run(hadrons, DetectorPart(cell_position=position, **PERFECT))
        assert len(out) == n
        out_eta = np.arcsinh(out.p4[:, 2] / np.hypot(out.p4[:, 0], out.p4[:, 1]))
        out_phi = np.arctan2(out.p4[:, 1], out.p4[:, 0])
        i_eta = np.floor(eta / 0.087)
        i_phi = np.floor((phi + np.pi) / (2 * np.pi / 72))
        if position == "centre":
            np.testing.assert_allclose(out_eta, (i_eta + 0.5) * 0.087, atol=1e-9)
            np.testing.assert_allclose(out_phi, -np.pi + (i_phi + 0.5) * 2 * np.pi / 72, atol=1e-9)
        else:
            assert np.all((out_eta >= i_eta * 0.087 - 1e-9) & (out_eta <= (i_eta + 1) * 0.087 + 1e-9))
            frac = (out_eta / 0.087) - i_eta
            assert 0.45 < frac.mean() < 0.55


def test_log_normal_smearing():
    n = 20_000
    cfg = DetectorPart()
    hadrons = _hadrons(10.0, 0.0, np.linspace(-3, 3, n), PHOTON)
    out = _run(hadrons, cfg)
    assert len(out) == n
    e = out.p4[:, 3]
    sigma = math.sqrt((cfg.ecal_res_const * 10) ** 2 + cfg.ecal_res_stoch**2 * 10 + cfg.ecal_res_noise**2)
    assert np.all(e > 0)
    assert abs(e.mean() - 10.0) < N_SIGMA * sigma / math.sqrt(n)
    assert e.std() == pytest.approx(sigma, rel=0.05)
    log_e = np.log(e)
    s2 = math.log1p((sigma / 10.0) ** 2)
    assert log_e.std() == pytest.approx(math.sqrt(s2), rel=0.05)


def test_impact_parameters():
    vertex = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    p4 = np.array([[0.0, 3.0, 4.0, 5.0], [3.0, 0.0, 0.0, 3.0]])
    d0, dz = impact_parameters(vertex, p4)
    np.testing.assert_allclose(d0, [1.0, 0.0])
    np.testing.assert_allclose(dz, [2.0, 0.0])


def test_track_impact_parameter_smearing():
    n = 20_000
    rng = np.random.default_rng(4)
    phi = rng.uniform(-np.pi, np.pi, n)
    vertex = np.stack([np.cos(phi + np.pi / 2), np.sin(phi + np.pi / 2), np.zeros(n), np.zeros(n)], 1) * 0.5
    hadrons = _hadrons(10.0, 0.0, phi, PI_CH, vertex=vertex)
    cfg = DetectorPart(track_eff_high=1.0)
    out = _run(hadrons, cfg)
    assert len(out) == n
    true_d0, _ = impact_parameters(vertex, hadrons.p4)
    pull = (out.d0 - true_d0[out.event]) / out.d0err
    assert abs(pull.mean()) < N_SIGMA / math.sqrt(n)
    assert pull.std() == pytest.approx(1.0, rel=0.05)
    np.testing.assert_allclose(out.d0err, out.dzerr)
    sigma = np.hypot(cfg.ip_res_const, cfg.ip_res_pt / np.hypot(out.p4[:, 0], out.p4[:, 1]))
    np.testing.assert_allclose(out.d0err, sigma, rtol=1e-9)


def test_neutrals_have_no_impact_parameters():
    rng = np.random.default_rng(5)
    n = 300
    species = rng.choice([PI_CH, PHOTON, NEUTRON], n)
    vertex = np.concatenate([rng.normal(0, 1.0, (n, 3)), np.zeros((n, 1))], 1)
    hadrons = _hadrons(rng.uniform(1, 20, n), rng.uniform(-1, 1, n), rng.uniform(-3, 3, n), species, vertex=vertex)
    for out in (_run(hadrons), ParticleArrays.from_hadrons(hadrons)):
        neutral = out.charge == 0
        for name in ("d0", "d0err", "dz", "dzerr"):
            np.testing.assert_array_equal(getattr(out, name)[neutral], 0.0)
        np.testing.assert_array_equal(out.flags.sum(axis=1), 1)


def test_forward_calorimeter():
    hadrons = _hadrons([5.0, 5.0], [3.0, -3.2], [0.0, 1.0], [PHOTON, PI_CH])
    out = _run(hadrons, DetectorPart(**PERFECT))
    assert len(out) == 2
    np.testing.assert_array_equal(out.flags[:, :3], [[0, 0, 1], [0, 1, 0]])


def test_cell_phi_bins():
    assert cell_phi_bins(0.0174) == 360
    assert cell_phi_bins(0.087) == 72


def test_keyed_randomness_jet_tracks_do_not_depend_on_the_ue_or_cell_positions():
    rng = np.random.default_rng(6)
    n_events, k = 50, 40
    event = np.repeat(np.arange(n_events), k)
    jet = _hadrons(
        rng.exponential(5.0, n_events * k) + 0.2,
        rng.normal(0, 0.3, n_events * k),
        rng.normal(0, 0.3, n_events * k),
        rng.choice([PI_CH, PHOTON, NEUTRON], n_events * k),
        event=event,
    )
    ue_event = np.repeat(np.arange(n_events), 10)
    m = len(ue_event)
    ue = _hadrons(
        rng.exponential(1.0, m) + 0.3,
        rng.normal(0, 0.5, m),
        rng.normal(0, 0.5, m),
        rng.choice([PI_CH, PHOTON], m),
        event=ue_event,
        is_ue=np.ones(m, dtype=bool),
    )
    both = HadronArrays.concat([jet, ue])
    both = both.select(np.argsort(both.event, kind="stable"))
    runs = {
        "detector": _run(both),
        "no ue": _run(jet),
        "centre": _run(both, DetectorPart(cell_position="centre")),
    }

    def jet_tracks(out):
        sel = (out.charge != 0) & ~out.is_ue
        return out.p4[sel], out.d0[sel], out.dz[sel], out.event[sel]

    reference = jet_tracks(runs["detector"])
    assert len(reference[0]) > 100
    for name in ("no ue", "centre"):
        for a, b in zip(reference, jet_tracks(runs[name]), strict=True):
            np.testing.assert_array_equal(a, b, err_msg=name)
    # a different seed changes them
    assert not np.array_equal(jet_tracks(_run(both, seed=2))[0], reference[0])
