"""Underlying event: Poisson multiplicity, the disc, the pT spectrum and the species."""

import math

import numpy as np

from helpers import KS_LIMIT, N_SIGMA, ks_distance
from itj.toy import preset
from itj.toy.hadronize import PHOTON, PI_CH, species_probabilities
from itj.toy.ue import underlying_event


def test_poisson_multiplicity_in_the_disc():
    n_jets = 2000
    cfg = preset("hadrons", ue_density=1.5, ue_radius=0.9)
    ue = underlying_event(n_jets, cfg, np.random.default_rng(1))
    photons = ue.species == PHOTON
    # pi0 -> gamma gamma: a pi0 gives two photons
    per_event = np.bincount(ue.event[~photons], minlength=n_jets) + np.bincount(ue.event[photons], minlength=n_jets) / 2
    mean = 1.5 * math.pi * 0.9**2
    assert abs(per_event.mean() - mean) < N_SIGMA * math.sqrt(mean / n_jets)
    assert abs(per_event.var() / mean - 1.0) < 0.1
    # positions: uniform in the disc about the seed (generation frame: y = 0, phi = 0)
    hadrons = ue.p4[~photons]
    pt = np.hypot(hadrons[:, 0], hadrons[:, 1])
    y = 0.5 * np.log((hadrons[:, 3] + hadrons[:, 2]) / (hadrons[:, 3] - hadrons[:, 2]))
    phi = np.arctan2(hadrons[:, 1], hadrons[:, 0])
    r2 = (y * y + phi * phi) / 0.9**2
    assert np.all(r2 <= 1.0 + 1e-9)
    assert ks_distance(r2, lambda x: x) < KS_LIMIT  # uniform in the disc: r^2 uniform
    # pT = ue_pt_min + Exp(ue_pt_mean)
    assert np.all(pt >= 0.3 - 1e-9)
    assert ks_distance(pt - 0.3, lambda x: 1.0 - np.exp(-x / 0.7)) < KS_LIMIT


def test_flags_species_and_index():
    n_jets = 2000
    cfg = preset("hadrons")
    ue = underlying_event(n_jets, cfg, np.random.default_rng(2))
    assert ue.is_ue.all()
    assert np.all(ue.prong == -1)
    assert np.all(ue.parton == -1)
    assert not ue.from_b.any()
    np.testing.assert_array_equal(ue.vertex, 0.0)
    assert np.all(np.diff(ue.event) >= 0)
    starts = np.searchsorted(ue.event, ue.event)
    np.testing.assert_array_equal(ue.index, np.arange(len(ue)) - starts)
    probs = species_probabilities(cfg.hadron)
    photons = (ue.species == PHOTON).sum() / 2
    hadrons = (ue.species != PHOTON).sum()
    p_pi0 = photons / (photons + hadrons)
    n = photons + hadrons
    assert abs(p_pi0 - probs[2]) < N_SIGMA * math.sqrt(probs[2] * (1 - probs[2]) / n)
    assert set(np.unique(ue.charge[ue.species == PI_CH])) == {-1, 1}


def test_no_ue_density_zero():
    ue = underlying_event(100, preset("hadrons", ue_density=0.0), np.random.default_rng(3))
    assert len(ue) == 0
