"""Hard process: seed spectrum, Breit-Wigner masses, decays, W helicity, QCD flavours and containment."""

import math

import numpy as np
import pytest

from helpers import KS_LIMIT, N_SIGMA, assert_binomial, ks_distance
from itj.toy import preset
from itj.toy.generate import balanced_counts
from itj.toy.hardprocess import PROCESSES, pt_spectrum_cdf, sample_hard, truncated_bw_cdf
from itj.toy.kinematics import boost, mass, rapidity, wrap_phi

QCD, W, TOP, Q, G = range(5)


def _hard(n, code, seed=0, **overrides):
    return sample_hard(n, code, preset("partons", **overrides), np.random.default_rng(seed))


def test_processes():
    assert PROCESSES == ("qcd", "wqq", "tbqq", "q", "g")


@pytest.mark.parametrize("spectrum", ["power", "flat"])
def test_seed_pt_spectrum(spectrum):
    hard = _hard(20_000, QCD, pt_spectrum=spectrum)
    cfg = preset("partons", pt_spectrum=spectrum).hard
    assert ks_distance(hard.pt, lambda x: pt_spectrum_cdf(x, cfg)) < KS_LIMIT
    assert hard.pt.min() >= 450.0
    assert hard.pt.max() <= 1100.0


def test_fixed_pt_and_rapidity_range():
    hard = _hard(2000, Q, pt_spectrum="fixed", pt_fixed=600.0)
    np.testing.assert_array_equal(hard.pt, 600.0)
    assert np.all(np.abs(hard.y) <= 2.0)
    assert ks_distance((hard.y + 2.0) / 4.0, lambda x: x) < KS_LIMIT
    assert ks_distance((hard.phi + np.pi) / (2 * np.pi), lambda x: x) < KS_LIMIT


@pytest.mark.parametrize(("code", "m", "w", "window"), [(W, 80.4, 2.085, 15.0), (TOP, 172.5, 1.42, 25.0)])
def test_truncated_breit_wigner(code, m, w, window):
    hard = _hard(20_000, code, containment=False)
    assert np.all(np.abs(hard.mass - m) <= window)
    assert ks_distance(hard.mass, lambda x: truncated_bw_cdf(x, m, w, m - window, m + window)) < KS_LIMIT


def _w_mass_of_top(hard):
    return mass(hard.prong_p4[:, 1] + hard.prong_p4[:, 2])


def test_w_helicity_fraction_in_top_decays():
    hard = _hard(20_000, TOP, containment=False)
    c = hard.cos_theta_star
    mt, mw = hard.mass, _w_mass_of_top(hard)
    assert np.all(mw <= mt - 5.0 + 1e-9)
    f0_expected = float(np.mean(mt * mt / (mt * mt + 2 * mw * mw)))
    assert f0_expected == pytest.approx(0.70, abs=0.01)
    f0_hat = 2.0 - 5.0 * float(np.mean(c * c))
    error = 5.0 * float(np.std(c * c)) / math.sqrt(len(c))
    assert abs(f0_hat - f0_expected) < N_SIGMA * error


def test_isotropic_decays():
    w = _hard(20_000, W, containment=False)
    assert ks_distance((w.cos_theta_star + 1) / 2, lambda x: x) < KS_LIMIT
    top = _hard(20_000, TOP, containment=False, w_helicity=False)
    assert ks_distance((top.cos_theta_star + 1) / 2, lambda x: x) < KS_LIMIT


@pytest.mark.parametrize("code", [W, TOP])
def test_prongs_sum_to_the_resonance(code):
    hard = _hard(2000, code)
    k = 2 if code == W else 3
    total = hard.prong_p4[:, :k].sum(axis=1)
    np.testing.assert_allclose(total, hard.res_p4, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(mass(hard.res_p4), hard.mass, rtol=1e-9)
    # the rest-frame momenta: boosting the prongs back gives rest_q (summing to zero)
    beta = hard.res_p4[:, :3] / hard.res_p4[:, 3:4]
    rest = boost(hard.prong_p4[:, :k], -beta[:, None, :])
    np.testing.assert_allclose(rest[..., :3], hard.rest_q[:, :k], atol=1e-8)
    np.testing.assert_allclose(hard.rest_q[:, :k].sum(axis=1), 0.0, atol=1e-8)
    if code == TOP:
        np.testing.assert_allclose(mass(hard.prong_p4[:, 0]), 4.8, rtol=1e-6)
        assert np.all(np.abs(hard.prong_pid[:, 0]) == 5)
        assert np.all(np.sign(hard.prong_pid[:, 0]) == np.sign(hard.res_pid))
    # W quarks ordered by lab energy
    first = 0 if code == W else 1
    lab_e = lambda p: p[..., 3] * np.cosh(hard.y) + p[..., 2] * np.sinh(hard.y)  # noqa: E731
    assert np.all(lab_e(hard.prong_p4[:, first]) >= lab_e(hard.prong_p4[:, first + 1]))
    assert set(np.abs(hard.res_pid)) == {24 if code == W else 6}
    np.testing.assert_array_equal(hard.nprongs, k)


def _y_phi(p):
    return rapidity(p), np.arctan2(p[..., 1], p[..., 0])


@pytest.mark.parametrize("code", [W, TOP])
def test_containment(code):
    from itj.cluster import leading_jet

    hard = _hard(3000, code)
    k = 2 if code == W else 3
    prongs = hard.prong_p4[:, :k]
    y, phi = _y_phi(prongs)
    yr, phir = _y_phi(hard.res_p4)
    dr = np.hypot(y - yr[:, None], wrap_phi(phi - phir[:, None]))
    assert np.all(dr < 0.8)
    members = leading_jet(prongs, np.ones((len(hard), k), dtype=bool), algorithm="antikt", R=0.8)
    assert np.all(members)
    efficiency = hard.n_contained / hard.n_tested
    expected = {W: (0.90, 0.97), TOP: (0.50, 0.72)}[code]
    assert expected[0] < efficiency < expected[1]
    # the stored pairwise Delta R
    pairs = [np.hypot(y[:, i] - y[:, j], wrap_phi(phi[:, i] - phi[:, j])) for i in range(k) for j in range(i + 1, k)]
    np.testing.assert_allclose(hard.dr_min, np.min(pairs, axis=0), rtol=1e-12)
    np.testing.assert_allclose(hard.dr_max, np.max(pairs, axis=0), rtol=1e-12)


def test_without_containment_some_prongs_leave_the_jet():
    hard = _hard(3000, TOP, containment=False)
    assert hard.n_tested == hard.n_contained == 3000
    assert np.nanmax(hard.dr_max) > 0.8


def test_qcd_flavours():
    n = 20_000
    hard = _hard(n, QCD)
    pid = hard.prong_pid[:, 0]
    quark = pid != 21
    assert_binomial(quark.sum(), n, 0.75)
    np.testing.assert_array_equal(hard.init_pid, np.where(quark, 1, 21))
    flavours = np.abs(pid[quark])
    for flavour, weight in zip(range(1, 6), (0.50, 0.30, 0.08, 0.08, 0.04), strict=True):
        assert_binomial((flavours == flavour).sum(), quark.sum(), weight)
    assert_binomial((pid[quark] > 0).sum(), quark.sum(), 0.5)
    np.testing.assert_array_equal(hard.nprongs, 1)
    assert np.all(np.isnan(hard.cos_theta_star))
    np.testing.assert_array_equal(hard.mass, 0.0)


def test_q_and_g_fix_the_initiator():
    q, g = _hard(500, Q), _hard(500, G)
    assert np.all(q.prong_pid[:, 0] != 21)
    assert np.all(g.prong_pid[:, 0] == 21)
    np.testing.assert_array_equal(q.init_pid, 1)
    np.testing.assert_array_equal(g.init_pid, 21)


def test_start_angles_from_colour_coherence():
    for code in (W, TOP):
        hard = _hard(500, code)
        k0 = 0 if code == W else 1
        a, b = hard.prong_p4[:, k0, :3], hard.prong_p4[:, k0 + 1, :3]
        theta_qq = np.arccos(np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1))
        np.testing.assert_allclose(hard.theta_start[:, k0], theta_qq, rtol=1e-9)
        np.testing.assert_allclose(hard.theta_start[:, k0 + 1], theta_qq, rtol=1e-9)
        incoherent = _hard(500, code, coherence=False)
        np.testing.assert_array_equal(incoherent.theta_start[:, : k0 + 2], 1.0)
        # coherence changes the start angles only
        np.testing.assert_array_equal(incoherent.prong_p4, hard.prong_p4)
    qcd = _hard(100, QCD, qcd_start_angle=0.7)
    np.testing.assert_array_equal(qcd.theta_start[:, 0], 0.7)


def test_same_stream_same_events():
    a, b = _hard(300, TOP, seed=5), _hard(300, TOP, seed=5)
    np.testing.assert_array_equal(a.prong_p4, b.prong_p4)
    assert not np.array_equal(a.prong_p4, _hard(300, TOP, seed=6).prong_p4)


def test_balanced_counts():
    np.testing.assert_array_equal(balanced_counts(1000, 3), [334, 333, 333])
    np.testing.assert_array_equal(balanced_counts(5, 3), [2, 2, 1])
    assert balanced_counts(7, 2).sum() == 7
