"""Four-vector helpers: boosts, two-body decays, RAMBO, the lab frame and the keyed random numbers."""

import math

import numpy as np
import pytest

from helpers import KS_LIMIT, N_SIGMA, ks_distance
from itj.toy.kinematics import (
    boost,
    delta_r,
    hash_normal,
    hash_uniform,
    keyed_hash,
    keyed_normal,
    keyed_uniform,
    mass,
    perp_basis,
    pt_eta_phi,
    rambo,
    rapidity,
    to_lab,
    two_body,
    wrap_phi,
)


def _random_p4(rng, n, m=0.0):
    p3 = rng.normal(0.0, 50.0, (n, 3))
    e = np.sqrt(np.sum(p3 * p3, axis=1) + m * m)
    return np.concatenate([p3, e[:, None]], axis=1)


def test_boost_inverts_and_keeps_the_mass():
    rng = np.random.default_rng(1)
    p = _random_p4(rng, 100, m=5.0)
    beta = rng.uniform(-0.5, 0.5, (100, 3))
    q = boost(p, beta)
    np.testing.assert_allclose(mass(q), 5.0, rtol=1e-9)
    np.testing.assert_allclose(boost(q, -beta), p, rtol=1e-10, atol=1e-9)


def test_boost_of_a_particle_at_rest():
    beta = np.array([[0.6, 0.0, 0.0]])
    q = boost(np.array([[0.0, 0.0, 0.0, 2.0]]), beta)
    gamma = 1.0 / math.sqrt(1.0 - 0.36)
    np.testing.assert_allclose(q, [[2.0 * gamma * 0.6, 0.0, 0.0, 2.0 * gamma]], rtol=1e-12)


def test_two_body():
    rng = np.random.default_rng(2)
    m = rng.uniform(80.0, 180.0, 50)
    c, phi = rng.uniform(-1, 1, 50), rng.uniform(0, 2 * np.pi, 50)
    p1, p2 = two_body(m, 4.8, 30.0, c, phi)
    np.testing.assert_allclose(p1 + p2, np.stack([0 * m, 0 * m, 0 * m, m], axis=1), atol=1e-10)
    np.testing.assert_allclose(mass(p1), 4.8, rtol=1e-7)
    np.testing.assert_allclose(mass(p2), 30.0, rtol=1e-9)
    p = np.sqrt((m * m - (34.8) ** 2) * (m * m - (25.2) ** 2)) / (2 * m)
    np.testing.assert_allclose(np.linalg.norm(p1[:, :3], axis=1), p, rtol=1e-12)
    np.testing.assert_allclose(p1[:, 2] / p, c, rtol=1e-12, atol=1e-12)


def test_rambo_conserves_four_momentum():
    rng = np.random.default_rng(3)
    m = np.full(200, 5.279)
    out = rambo(m, rng.random((4, 200, 7)))
    assert out.shape == (200, 7, 4)
    np.testing.assert_allclose(out.sum(axis=1), np.tile([0.0, 0.0, 0.0, 5.279], (200, 1)), atol=1e-12)
    np.testing.assert_allclose(mass(out), 0.0, atol=1e-6)
    assert np.all(out[..., 3] > 0)


def test_to_lab_keeps_pt_delta_y_delta_phi_and_mass():
    rng = np.random.default_rng(4)
    p = _random_p4(rng, 40, m=1.0)
    y_jet, phi_jet = 1.3, -2.0
    q = to_lab(p, y_jet, phi_jet)
    pt_p, _, phi_p = pt_eta_phi(p)
    pt_q, _, phi_q = pt_eta_phi(q)
    np.testing.assert_allclose(pt_q, pt_p, rtol=1e-12)
    np.testing.assert_allclose(rapidity(q) - rapidity(p), y_jet, atol=1e-9)
    np.testing.assert_allclose(wrap_phi(phi_q - phi_p - phi_jet), 0.0, atol=1e-12)
    np.testing.assert_allclose(mass(q), mass(p), rtol=1e-9)
    np.testing.assert_allclose(
        delta_r(rapidity(q[:1]), phi_q[:1], rapidity(q[1:]), phi_q[1:]),
        delta_r(rapidity(p[:1]), phi_p[:1], rapidity(p[1:]), phi_p[1:]),
        rtol=1e-9,
    )


def test_wrap_phi_range():
    x = np.array([-np.pi, np.pi, 3 * np.pi, -3 * np.pi + 1e-9, 0.0, 7.0])
    w = wrap_phi(x)
    assert np.all(w > -np.pi)
    assert np.all(w <= np.pi)
    np.testing.assert_allclose(np.cos(w), np.cos(x), atol=1e-12)
    assert w[0] == pytest.approx(np.pi)


def test_pt_eta_phi_and_rapidity_of_massless():
    p = np.array([[3.0, 4.0, 5.0 * np.sinh(0.7), 5.0 * np.cosh(0.7)]])
    pt, eta, phi = pt_eta_phi(p)
    assert pt[0] == pytest.approx(5.0)
    assert eta[0] == pytest.approx(0.7)
    assert rapidity(p)[0] == pytest.approx(0.7)
    assert phi[0] == pytest.approx(math.atan2(4.0, 3.0))


def test_perp_basis():
    rng = np.random.default_rng(5)
    n = rng.normal(size=(50, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n[0] = [0.0, 0.0, 1.0]
    e1, e2 = perp_basis(n)
    for a, b in ((e1, e2), (e1, n), (e2, n)):
        np.testing.assert_allclose(np.sum(a * b, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(e1, axis=1), 1.0)
    np.testing.assert_allclose(np.cross(e1, e2), n, atol=1e-12)


def test_keyed_uniform_is_deterministic_and_independent_of_the_other_keys():
    keys = np.arange(1000)
    a = keyed_uniform(7, 3, keys)
    b = keyed_uniform(7, 3, keys[::-1])[::-1]
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[17], keyed_uniform(7, 3, 17))
    assert not np.array_equal(a, keyed_uniform(8, 3, keys))
    assert not np.array_equal(a, keyed_uniform(7, 4, keys))
    assert keyed_uniform(7, -1).shape == ()
    assert keyed_uniform(7, np.arange(6).reshape(2, 3), np.arange(3)).shape == (2, 3)


def test_keyed_uniform_and_normal_distributions():
    keys = np.arange(20_000)
    u = keyed_uniform(11, keys, 5)
    assert np.all((u >= 0) & (u < 1))
    assert ks_distance(u, lambda x: x) < KS_LIMIT
    z = keyed_normal(11, keys)
    assert abs(z.mean()) < N_SIGMA / math.sqrt(len(z))
    assert abs(z.std() - 1.0) < N_SIGMA * math.sqrt(0.5 / len(z))
    h = keyed_hash(11, keys)
    assert h.dtype == np.uint64
    u0, u1 = hash_uniform(h, 0), hash_uniform(h, 1)
    assert abs(np.corrcoef(u0, u1)[0, 1]) < N_SIGMA / math.sqrt(len(keys))
    n = hash_normal(h, 3)
    assert abs(n.mean()) < N_SIGMA / math.sqrt(len(n))
