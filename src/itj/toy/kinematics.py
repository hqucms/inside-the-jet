"""Vectorised four-vector helpers of the generator: boosts, two-body decays, RAMBO, frames and keyed random numbers.

Four-vectors are ``(px, py, pz, E)`` on the last axis, float64. The generator builds every jet in the *generation
frame* (seed along +x at y = 0) and moves it to the lab with :func:`to_lab` (a rotation by phi_jet about z, then a
boost along z with rapidity y_jet), which leaves pT, Delta y and Delta phi unchanged.

:func:`keyed_uniform` is a counter-based random number generator (splitmix64 over uint64 keys): the same keys
always give the same number, independent of how many other numbers are drawn. The detector uses it so that a
particle's efficiency and smearing do not change when other particles (the underlying event) are added or removed.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "boost",
    "delta_r",
    "direction",
    "hash_normal",
    "hash_uniform",
    "keyed_hash",
    "keyed_normal",
    "keyed_uniform",
    "mass",
    "massless_from_angles",
    "perp_basis",
    "pt_eta_phi",
    "rambo",
    "rapidity",
    "to_lab",
    "two_body",
    "wrap_phi",
]


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    """Wrap an azimuth (difference) to (-pi, pi]."""
    out = np.mod(np.asarray(phi, dtype=np.float64) + np.pi, 2 * np.pi) - np.pi
    return np.where(out <= -np.pi, out + 2 * np.pi, out)


def pt_eta_phi(p4: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transverse momentum, pseudorapidity and azimuth of ``p4`` (``(..., 4)``)."""
    px, py, pz = p4[..., 0], p4[..., 1], p4[..., 2]
    pt = np.hypot(px, py)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.arcsinh(np.where(pt > 0, pz / np.where(pt > 0, pt, 1.0), np.sign(pz) * 1e10))
    return pt, eta, np.arctan2(py, px)


def rapidity(p4: np.ndarray) -> np.ndarray:
    """Rapidity ``y = 1/2 ln((E + pz) / (E - pz))`` of ``p4``."""
    e, pz = p4[..., 3], p4[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * np.log(np.maximum(e + pz, 1e-300) / np.maximum(e - pz, 1e-300))


def mass(p4: np.ndarray) -> np.ndarray:
    """Invariant mass ``sqrt(max(E^2 - p^2, 0))``."""
    m2 = p4[..., 3] ** 2 - p4[..., 0] ** 2 - p4[..., 1] ** 2 - p4[..., 2] ** 2
    return np.sqrt(np.maximum(m2, 0.0))


def delta_r(y1: np.ndarray, phi1: np.ndarray, y2: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """``sqrt(Delta y^2 + Delta phi^2)`` with Delta phi wrapped."""
    return np.hypot(np.asarray(y1) - y2, wrap_phi(np.asarray(phi1) - phi2))


def boost(p4: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Boost four-vectors by the velocity ``beta`` (``(..., 3)``, broadcast against ``p4``).

    A particle at rest goes to velocity ``beta``; ``boost(boost(p, b), -b) == p``.
    """
    beta = np.asarray(beta, dtype=np.float64)
    b2 = np.sum(beta * beta, axis=-1)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = np.sum(beta * p4[..., :3], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        coef = np.where(b2 > 0, (gamma - 1.0) / np.where(b2 > 0, b2, 1.0), 0.0)
    e = p4[..., 3]
    out = np.empty(np.broadcast_shapes(p4.shape, (*beta.shape[:-1], 4)), dtype=np.float64)
    out[..., :3] = p4[..., :3] + ((coef * bp + gamma * e)[..., None]) * beta
    out[..., 3] = gamma * (e + bp)
    return out


def perp_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors ``e1, e2`` perpendicular to the unit vectors ``n`` (``(k, 3)``), with ``e1 x e2 = n``."""
    a = np.where(np.abs(n[..., 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    return e1, np.cross(n, e1)


def direction(cos_theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Return unit vectors ``(sin t cos phi, sin t sin phi, cos t)``."""
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))
    return np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=-1)


def massless_from_angles(energy: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Massless four-vectors of energy ``energy`` along the unit vectors ``n``."""
    return np.concatenate([energy[..., None] * n, energy[..., None]], axis=-1)


def two_body(
    m: np.ndarray, m1: np.ndarray | float, m2: np.ndarray | float, cos_theta: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Two-body decay ``M -> 1 2`` in the rest frame: daughter 1 along (cos_theta, phi), daughter 2 opposite.

    ``|p| = sqrt((M^2 - (m1 + m2)^2) (M^2 - (m1 - m2)^2)) / (2 M)``.
    """
    m = np.asarray(m, dtype=np.float64)
    m1 = np.broadcast_to(np.asarray(m1, dtype=np.float64), m.shape)
    m2 = np.broadcast_to(np.asarray(m2, dtype=np.float64), m.shape)
    p = np.sqrt(np.maximum((m * m - (m1 + m2) ** 2) * (m * m - (m1 - m2) ** 2), 0.0)) / (2.0 * m)
    d = direction(cos_theta, phi) * p[..., None]
    p1 = np.concatenate([d, np.sqrt(p * p + m1 * m1)[..., None]], axis=-1)
    p2 = np.concatenate([-d, np.sqrt(p * p + m2 * m2)[..., None]], axis=-1)
    return p1, p2


def rambo(m: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Massless n-body phase space (RAMBO) in the rest frame of mass ``m``.

    Parameters
    ----------
    m : numpy.ndarray
        ``(k,)`` masses of the decaying objects.
    u : numpy.ndarray
        ``(4, k, n)`` uniform random numbers in [0, 1).

    Returns
    -------
    numpy.ndarray
        ``(k, n, 4)`` massless four-momenta that sum to ``(0, 0, 0, m)`` (to rounding).
    """
    c = 2.0 * u[0] - 1.0
    phi = 2.0 * np.pi * u[1]
    q0 = -np.log(np.maximum((1.0 - u[2]) * (1.0 - u[3]), 1e-300))
    q = np.concatenate([direction(c, phi) * q0[..., None], q0[..., None]], axis=-1)
    total = q.sum(axis=1)
    mq = np.sqrt(np.maximum(total[:, 3] ** 2 - np.sum(total[:, :3] ** 2, axis=-1), 1e-300))
    b = -total[:, :3] / mq[:, None]
    x = np.asarray(m, dtype=np.float64) / mq
    gamma = total[:, 3] / mq
    a = 1.0 / (1.0 + gamma)
    bq = np.einsum("ki,kni->kn", b, q[..., :3])
    out = np.empty_like(q)
    out[..., 3] = x[:, None] * (gamma[:, None] * q[..., 3] + bq)
    out[..., :3] = x[:, None, None] * (
        q[..., :3] + b[:, None, :] * q[..., 3:4] + (a[:, None] * bq)[..., None] * b[:, None, :]
    )
    return out


def to_lab(p4: np.ndarray, y_jet: np.ndarray, phi_jet: np.ndarray) -> np.ndarray:
    """Move four-vectors from the generation frame (seed along +x at y = 0) to the lab.

    Rotation by ``phi_jet`` about z, then a boost along z with rapidity ``y_jet`` (both broadcast against ``p4``'s
    leading axes). pT, Delta y and Delta phi are unchanged.
    """
    c, s = np.cos(phi_jet), np.sin(phi_jet)
    ch, sh = np.cosh(y_jet), np.sinh(y_jet)
    out = np.empty_like(p4, dtype=np.float64)
    px, py, pz, e = p4[..., 0], p4[..., 1], p4[..., 2], p4[..., 3]
    out[..., 0] = c * px - s * py
    out[..., 1] = s * px + c * py
    out[..., 2] = ch * pz + sh * e
    out[..., 3] = ch * e + sh * pz
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Counter-based random numbers

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _MIX1
        z = (z ^ (z >> np.uint64(27))) * _MIX2
        return z ^ (z >> np.uint64(31))


def _as_key(value: np.ndarray | int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind == "u":
        return arr.astype(np.uint64)
    return arr.astype(np.int64).view(np.uint64) if arr.ndim else np.asarray(int(arr) & 0xFFFFFFFFFFFFFFFF, np.uint64)


def keyed_hash(seed: int, *keys: np.ndarray | int) -> np.ndarray:
    """Hash ``(seed, *keys)`` to 64 bits (splitmix64 chained over the keys, broadcast).

    The hash is the state of :func:`hash_uniform` and :func:`hash_normal`.
    """
    h = _splitmix64(np.asarray(int(seed) & 0xFFFFFFFFFFFFFFFF, dtype=np.uint64))
    for key in keys:
        h = _splitmix64(h ^ _as_key(key))
    return np.asarray(h)


def hash_uniform(h: np.ndarray, purpose: int) -> np.ndarray:
    """Draw uniform numbers in [0, 1) from a :func:`keyed_hash` state and one more key (the purpose of the variate)."""
    z = _splitmix64(h ^ _as_key(purpose))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def hash_normal(h: np.ndarray, purpose: int) -> np.ndarray:
    """Draw standard normal numbers from a :func:`keyed_hash` state.

    Box-Muller of the uniforms of the purposes ``2 purpose`` and ``2 purpose + 1``.
    """
    u1 = hash_uniform(h, 2 * purpose)
    u2 = hash_uniform(h, 2 * purpose + 1)
    return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * math.pi * u2)


def keyed_uniform(seed: int, *keys: np.ndarray | int) -> np.ndarray:
    """Uniform numbers in [0, 1) that depend only on ``(seed, *keys)`` (splitmix64 chained over the keys).

    Parameters
    ----------
    seed : int
        The run seed.
    *keys : numpy.ndarray or int
        Integer keys, broadcast against each other (negative values are allowed).

    Returns
    -------
    numpy.ndarray
        float64 of the broadcast shape of the keys.

    Examples
    --------
    >>> a = keyed_uniform(1, np.arange(3), 7)
    >>> bool(np.all(a == keyed_uniform(1, np.arange(3), 7))), a.shape
    (True, (3,))
    """
    h = keyed_hash(seed, *keys)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def keyed_normal(seed: int, *keys: np.ndarray | int) -> np.ndarray:
    """Draw standard normal numbers that depend only on ``(seed, *keys)`` (Box-Muller of two keyed uniforms)."""
    u1 = keyed_uniform(seed, *keys, 0)
    u2 = keyed_uniform(seed, *keys, 1)
    return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * math.pi * u2)
