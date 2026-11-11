"""Parton shower: veto algorithm against the Sudakov form factor, splitting functions, record, reshuffle."""

import itertools
import math

import numpy as np
import pytest

from helpers import KS_LIMIT, N_SIGMA, assert_binomial, ks_distance, sudakov_cdf
from itj.toy import preset, shower_one
from itj.toy.config import ShowerPart
from itj.toy.hardprocess import sample_hard
from itj.toy.kinematics import mass
from itj.toy.shower import CA, CF, Progenitors, alphas, reshuffle, shower, shower_events

FIXED = ShowerPart(alphas_mode="fixed", alphas_value=0.12, g_to_qq=False)


def _progenitors(n, *, energy=600.0, gluon=False, theta_start=1.0, hard_b=False):
    pid = 21 if gluon else (5 if hard_b else 1)
    return Progenitors(
        energy=np.full(n, energy),
        direction=np.tile([1.0, 0.0, 0.0], (n, 1)),
        pid=np.full(n, pid),
        event=np.arange(n),
        prong=np.zeros(n, dtype=np.int64),
        theta_start=np.full(n, theta_start),
        hard_b=np.full(n, hard_b),
    )


@pytest.fixture(scope="module")
def quarks():
    n = 20_000
    partons, record = shower(_progenitors(n), FIXED, np.random.default_rng(1), n_events=n)
    return n, partons, record


@pytest.fixture(scope="module")
def gluons():
    n = 6000
    cfg = ShowerPart(alphas_mode="fixed", alphas_value=0.12)
    partons, record = shower(_progenitors(n, gluon=True), cfg, np.random.default_rng(2), n_events=n)
    return n, partons, record


def _first_angle(n, record):
    theta = np.zeros(n)
    first = record.parent == 0
    theta[record.event[first]] = record.theta[first]
    assert np.bincount(record.event[first], minlength=n).max() <= 1
    return theta


def two_sample_ks(a, b):
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    d = np.max(np.abs(np.searchsorted(a, x, side="right") / len(a) - np.searchsorted(b, x, side="right") / len(b)))
    return math.sqrt(len(a) * len(b) / (len(a) + len(b))) * d


def test_alphas():
    cfg = ShowerPart()
    assert float(alphas(91.1876, cfg)) == pytest.approx(0.118, rel=1e-12)
    assert float(alphas(1.0, cfg)) == pytest.approx(0.34, abs=0.01)
    assert float(alphas(5.0, FIXED)) == 0.12


def test_first_emission_angle_follows_the_sudakov_form_factor(quarks):
    n, _, record = quarks
    theta = _first_angle(n, record)
    kwargs = {"energy": 600.0, "alphas": 0.12, "kt_cut": 1.0, "theta0": 1.0}
    p_none = float(sudakov_cdf(np.array([0.0]), **kwargs)[0])
    assert_binomial((theta == 0).sum(), n, p_none)
    emitted = theta[theta > 0]

    def conditional(x):
        return (sudakov_cdf(x, **kwargs) - p_none) / (1.0 - p_none)

    assert ks_distance(emitted, conditional) < KS_LIMIT


def _pqq_primitive(z):
    return -z - 0.5 * z * z - 2.0 * np.log1p(-z)


def _pgg_primitive(z):
    return -2.0 * z - np.log1p(-z) + np.log(z) + 0.5 * z * z - z**3 / 3.0


def _pqg_primitive(z):
    return z**3 / 3.0 - (1.0 - z) ** 3 / 3.0


def _pit(record, channel, primitive):
    """Probability integral transform of z given (E, theta): uniform for the exact splitting function."""
    rows = record.channel == channel
    e, theta, z = record.energy[rows], record.theta[rows], record.z[rows]
    root = np.sqrt(1.0 - 4.0 * 1.0 / (e * theta))
    lo, hi = 0.5 * (1.0 - root), 0.5 * (1.0 + root)
    u = (primitive(z) - primitive(lo)) / (primitive(hi) - primitive(lo))
    return theta, u


def test_z_spectrum_of_q_to_qg_in_theta_bins(quarks):
    _, _, record = quarks
    theta, u = _pit(record, 0, _pqq_primitive)
    edges = np.quantile(theta, [0.0, 0.25, 0.5, 0.75, 1.0])
    for lo, hi in itertools.pairwise(edges):
        inside = (theta >= lo) & (theta <= hi)
        assert ks_distance(u[inside], lambda x: x) < KS_LIMIT


@pytest.mark.parametrize(("channel", "primitive"), [(1, _pgg_primitive), (2, _pqg_primitive)])
def test_gluon_splitting_functions(gluons, channel, primitive):
    _, _, record = gluons
    _, u = _pit(record, channel, primitive)
    assert len(u) > 500
    assert ks_distance(u, lambda x: x) < KS_LIMIT


def test_channel_ratio_of_gluons(gluons):
    _, _, record = gluons
    first = record.parent == 0
    channel = record.channel[first]
    # the first branching of a 600 GeV gluon: the channel probability is the ratio of the z integrals (fixed coupling)
    gg, qq = (channel == 1).sum(), (channel == 2).sum()
    assert qq / (gg + qq) < 0.1
    assert gg > 10 * qq


def test_energy_is_conserved_and_partons_are_massless(quarks, gluons):
    for n, partons, _ in (quarks, gluons):
        energy = np.bincount(partons.event, weights=partons.p4[:, 3], minlength=n)
        np.testing.assert_allclose(energy, 600.0, rtol=1e-12)
        np.testing.assert_allclose(mass(partons.p4) / partons.p4[:, 3], 0.0, atol=1e-6)
        assert np.all(np.diff(partons.event) >= 0)


def test_emission_record_is_consistent(gluons):
    n, partons, record = gluons
    np.testing.assert_allclose(record.kt, record.z * (1 - record.z) * record.energy * record.theta, rtol=1e-12)
    assert np.all(record.kt >= 1.0)
    # angular ordering: an emission of a daughter happens below the angle at which the daughter was made
    made = {}
    for e, d1, d2, th, depth in zip(
        record.event, record.daughter1, record.daughter2, record.theta, record.depth, strict=True
    ):
        made[(e, d1)] = (th, depth)
        made[(e, d2)] = (th, depth)
    for e, parent, th, depth in zip(record.event, record.parent, record.theta, record.depth, strict=True):
        if parent == 0:
            assert depth == 0
        else:
            assert th <= made[(e, parent)][0]
            assert depth == made[(e, parent)][1] + 1
    # the final partons are the leaves of the tree
    parents = set(zip(record.event.tolist(), record.parent.tolist(), strict=True))
    nodes = set(made) | {(e, 0) for e in range(n)}
    leaves = nodes - parents
    assert set(zip(partons.event.tolist(), partons.node.tolist(), strict=True)) == leaves
    # one primary chain per jet, starting at the progenitor
    primary = record.primary.astype(bool)
    assert np.all(primary[record.parent == 0])
    same = record.event[1:] == record.event[:-1]
    assert np.all(np.diff(record.seq)[same] > 0)


def test_vectorised_shower_agrees_with_shower_one(quarks):
    n, _, record = quarks
    rng = np.random.default_rng(3)
    m = 3000
    runs = [shower_one(600.0, "q", 1.0, FIXED, rng) for _ in range(m)]
    first = np.array([r[0]["theta"] if r else 0.0 for r in runs])
    counts = np.array([len(r) for r in runs])
    vec_first = _first_angle(n, record)
    vec_counts = np.bincount(record.event, minlength=n)
    assert two_sample_ks(first[first > 0], vec_first[vec_first > 0]) < KS_LIMIT
    p = (vec_first == 0).mean()
    assert_binomial((first == 0).sum(), m, p, n_sigma=N_SIGMA * 1.1)
    error = math.hypot(counts.std() / math.sqrt(m), vec_counts.std() / math.sqrt(n))
    assert abs(counts.mean() - vec_counts.mean()) < N_SIGMA * error
    primary = np.array([sum(e["primary"] for e in r) for r in runs])
    vec_primary = np.bincount(record.event[record.primary.astype(bool)], minlength=n)
    error = math.hypot(primary.std() / math.sqrt(m), vec_primary.std() / math.sqrt(n))
    assert abs(primary.mean() - vec_primary.mean()) < N_SIGMA * error
    # the leading-log number of primary emissions (4.3): 1.60
    assert vec_primary.mean() == pytest.approx(1.60, abs=0.08)


def test_shower_one_record():
    emissions = shower_one(300.0, "g", 0.8, ShowerPart(), np.random.default_rng(4))
    assert emissions
    for e in emissions:
        assert set(e) == {"theta", "z", "kt", "phi", "channel", "depth", "primary", "energy"}
        assert e["kt"] >= 1.0
        assert e["theta"] <= 0.8
        assert 0 <= e["phi"] < 2 * math.pi
    assert shower_one(3.0, "q", 1.0, ShowerPart(), np.random.default_rng(4)) == []


def test_dead_cone():
    n = 4000
    cfg = ShowerPart(dead_cone=True)
    _, record = shower(_progenitors(n, energy=100.0, hard_b=True), cfg, np.random.default_rng(5), n_events=n)
    theta = _first_angle(n, record)
    assert np.all((theta == 0) | (theta >= 4.8 / 100.0))
    _, free = shower(_progenitors(n, energy=100.0, hard_b=True), ShowerPart(), np.random.default_rng(5), n_events=n)
    assert np.any((_first_angle(n, free) > 0) & (_first_angle(n, free) < 4.8 / 100.0))


def test_reshuffle_is_exact():
    rng = np.random.default_rng(6)
    r, s = 50, 3
    big_m = rng.uniform(150.0, 190.0, r)
    q = rng.normal(0.0, 40.0, (r, s, 3))
    q -= q.mean(axis=1, keepdims=True)
    # every system: 1 to 4 partons with some invariant mass
    p4, system = [], []
    for i in range(r):
        for j in range(s):
            k = rng.integers(1, 5)
            p = rng.normal(0.0, 3.0, (k, 3)) + q[i, j] / k
            p4.append(np.concatenate([p, np.linalg.norm(p, axis=1, keepdims=True)], axis=1))
            system += [i * s + j] * k
    p4, system = np.concatenate(p4), np.array(system)
    moved, ok = reshuffle(q, p4, system, big_m)
    total = np.zeros((r, 4))
    np.add.at(total, system // s, moved)
    np.testing.assert_allclose(total[ok], np.stack([0 * big_m, 0 * big_m, 0 * big_m, big_m], 1)[ok], atol=1e-9)
    # every system keeps its mass and moves along its decay direction
    sys_old, sys_new = np.zeros((r * s, 4)), np.zeros((r * s, 4))
    np.add.at(sys_old, system, p4)
    np.add.at(sys_new, system, moved)
    np.testing.assert_allclose(mass(sys_new), mass(sys_old), rtol=1e-7, atol=1e-5)  # sqrt of a rounded 0
    direction = sys_new[:, :3] / np.linalg.norm(sys_new[:, :3], axis=1, keepdims=True)
    q_dir = q.reshape(-1, 3) / np.linalg.norm(q.reshape(-1, 3), axis=1, keepdims=True)
    np.testing.assert_allclose(direction[ok.repeat(s)], q_dir[ok.repeat(s)], atol=1e-9)


def test_reshuffle_keeps_the_mass_of_a_single_hadron():
    # two systems of one particle each: a massive hadron keeps its mass, a massless one stays massless
    q = np.array([[[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]])
    pion = np.array([3.0, 1.0, 0.5, np.sqrt(9.0 + 1.0 + 0.25 + 0.13957**2)])
    photon = np.array([-4.0, 0.0, 0.0, 4.0])
    moved, ok = reshuffle(q, np.stack([pion, photon]), np.array([0, 1]), np.array([25.0]))
    assert ok.all()
    np.testing.assert_allclose(mass(moved[:1]), 0.13957, rtol=1e-9)
    np.testing.assert_allclose(moved[1, 3], np.linalg.norm(moved[1, :3]), rtol=1e-12)
    np.testing.assert_allclose(moved.sum(axis=0), [0.0, 0.0, 0.0, 25.0], atol=1e-9)


@pytest.mark.parametrize("code", [1, 2])
def test_resonance_partons_sum_to_the_resonance(code):
    cfg = preset("shower")
    hard = sample_hard(400, code, cfg, np.random.default_rng(7))
    partons, record = shower_events(hard, cfg, np.random.default_rng(8))
    total = np.zeros((len(hard), 4))
    np.add.at(total, partons.event, partons.p4)
    np.testing.assert_allclose(total, hard.res_p4, rtol=1e-9, atol=1e-9 * hard.res_p4[:, 3:].max())
    assert len(record) > len(hard)
    # without the shower the (massive) b quark becomes massless in the reshuffle and the sum still holds
    bare, _ = shower_events(hard, preset("partons"), np.random.default_rng(8))
    total = np.zeros((len(hard), 4))
    np.add.at(total, bare.event, bare.p4)
    np.testing.assert_allclose(total, hard.res_p4, rtol=1e-9, atol=1e-9 * hard.res_p4[:, 3:].max())


def test_qcd_jets_conserve_energy_and_point_along_the_seed():
    cfg = preset("shower")
    hard = sample_hard(2000, 4, cfg, np.random.default_rng(9))  # gluons: the largest virtualities
    partons, _ = shower_events(hard, cfg, np.random.default_rng(10))
    total = np.zeros((len(hard), 4))
    np.add.at(total, partons.event, partons.p4)
    np.testing.assert_allclose(total[:, 3], hard.pt, rtol=1e-12)
    np.testing.assert_allclose(total[:, 1:3] / hard.pt[:, None], 0.0, atol=1e-9)  # no p_perp about the seed (+x)
    assert np.all(total[:, 0] <= hard.pt)
    np.testing.assert_allclose(mass(partons.p4) / partons.p4[:, 3], 0.0, atol=1e-6)


def test_colour_factors():
    assert pytest.approx(4 / 3) == CF
    assert CA == 3.0
