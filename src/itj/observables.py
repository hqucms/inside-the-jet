"""Jet observables: mass, soft drop, N-subjettiness, energy correlation functions, the Lund plane and angularities.

Every function takes the padded particles of a batch of jets, ``p4`` of shape ``(B, N, 4)`` with (px, py, pz, E) on
the last axis and ``mask`` of shape ``(B, N)`` (``JetData.padded_p4``), and returns one float64 value per jet unless
stated otherwise. Angles are rapidity y and azimuth phi, ``dR_ij^2 = dy^2 + dphi^2``; ``pt_J`` is the scalar sum of
the constituent pT and ``z_i = pt_i / pt_J``. The definitions (``docs/observables.md``):

.. code-block:: text

    m^2      = (sum_i p_i)^2
    soft drop: C/A tree, decluster j -> j1, j2 until min(pt1, pt2) / (pt1 + pt2) > zcut (dR12 / R0)^beta,
               dropping the softer branch at every failed step
    tau_N    = sum_i pt_i min_a dR_ia^beta / (sum_i pt_i R0^beta)     (axes: exclusive kt with WTA recombination)
    e2       = sum_{i<j} z_i z_j dR_ij^beta        = 1/2 z^T A z         (A_ij = dR_ij^beta, A_ii = 0)
    e3       = sum_{i<j<k} z_i z_j z_k (dR_ij dR_ik dR_jk)^beta = tr((ZA)^3) / 6     (Z = diag(z))
    D2       = e3 / e2^3,    C2 = e3 / e2^2
    lambda   = sum_i z_i^kappa (dR_i,J / R)^beta   (generalised angularity about the jet axis)

The functions that build ``(B, N, N)`` arrays (:func:`softdrop`, :func:`nsubjettiness`, :func:`subjet_axes`,
:func:`ecf`, :func:`d2`, :func:`c2`, :func:`lund_primary`, :func:`jet_branches`) work on at most ``chunk_size`` jets
at a time, each chunk padded to its own longest jet. The peak memory is about 0.15 GB at N = 123 for every
``chunk_size`` (the ECF matrices are built for 256 jets at a time and the clusterings for blocks of 128 jets; measured
0.13 GB for 2048 jets of 123 particles).

:func:`jet_branches` gives the ten JetClass jet branches with the JetClass conventions: soft drop with C/A, beta 0,
z_cut 0.1, R0 0.8 and N-subjettiness with beta 1, R0 0.8 and WTA-kt axes. JetClass itself was made with Delphes,
whose CMS card uses one-pass minimised kt axes (``axes="onepass_kt"``); the toy keeps WTA-kt axes for speed (about
2.6 times faster here), which shifts its tau_N by about 0.5-1.5 % against one-pass axes (``docs/observables.md``).

Examples
--------
>>> import numpy as np
>>> def massless(pt, y, phi):
...     return [pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(y), pt * np.cosh(y)]
>>> p4 = np.array([[massless(400.0, 0.0, 0.0), massless(200.0, 0.4, 0.0)]])
>>> mask = np.ones((1, 2), bool)
>>> round(float(mass(p4, mask)[0]), 2)  # m^2 = 2 pt1 pt2 (cosh dy - cos dphi)
113.89
>>> nsubjettiness(p4, mask, n=(1, 2)).round(4).tolist()  # WTA axis on the harder particle; tau_2 = 0
[[0.1667, 0.0]]
>>> e2, e3 = ecf(p4, mask)
>>> round(float(e2[0]), 4), float(e3[0])
(0.0889, 0.0)
"""

from __future__ import annotations

import difflib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from .cluster import ClusterTree, _abs_dphi, _check_particles, _pt_y_phi_m, _wrap, cluster, decluster_primary
from .data import Ragged

if TYPE_CHECKING:
    from .data import JetData

__all__ = [
    "COMPUTE_NAMES",
    "SoftDrop",
    "angularity",
    "c2",
    "compute",
    "d2",
    "ecf",
    "jet_branches",
    "jet_p4",
    "leading_fraction",
    "lund_density",
    "lund_primary",
    "mass",
    "multiplicity",
    "nsubjettiness",
    "pt_y_phi_m",
    "ptd",
    "softdrop",
    "subjet_axes",
    "width",
]

Axes = Literal["wta_kt", "kt", "onepass_kt"]

#: The names :func:`compute` knows.
COMPUTE_NAMES: tuple[str, ...] = (
    "mass",
    "sdmass",
    "tau1",
    "tau2",
    "tau3",
    "tau4",
    "tau21",
    "tau32",
    "e2",
    "e3",
    "d2",
    "c2",
    "d2_b2",
    "c2_b2",
    "multiplicity",
    "ptd",
    "width",
    "leading_fraction",
)

#: Iterations and accuracy of the axis minimisation of ``axes="onepass_kt"`` (the defaults of fastjet-contrib's
#: Nsubjettiness: up to 1000 Weiszfeld updates, until the axes move by less than 1e-4 in (y, phi)).
_ONEPASS_ITERATIONS = 1000
_ONEPASS_ACCURACY = 1e-4

_AXES: tuple[str, ...] = ("wta_kt", "kt", "onepass_kt")


# ---------------------------------------------------------------------------------------------------------- helpers


def _blocks(p4: np.ndarray, mask: np.ndarray, chunk_size: int) -> Iterator[tuple[slice, np.ndarray, np.ndarray]]:
    """Yield (rows, p4, mask) blocks of at most ``chunk_size`` jets, each cut to its own longest jet."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    for start in range(0, len(p4), chunk_size):
        rows = slice(start, start + chunk_size)
        m = mask[rows]
        used = np.flatnonzero(m.any(axis=0))
        n = int(used[-1]) + 1 if used.size else 1
        yield rows, p4[rows, :n], m[:, :n]


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Return num / den, 0 where den == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, num / np.where(den != 0, den, 1.0), 0.0)


def _check_axes(axes: str) -> None:
    if axes not in _AXES:
        close = difflib.get_close_matches(str(axes), _AXES, n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        raise ValueError(f"unknown axes {axes!r}{hint}: choose from {', '.join(map(repr, _AXES))}")


def _jet_axis(p4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (y, phi) of the summed four-vector of every jet."""
    _, y, phi, _ = _pt_y_phi_m(p4.sum(axis=1))
    return y, phi


def _delta_r(y: np.ndarray, phi: np.ndarray, ay: np.ndarray, aphi: np.ndarray) -> np.ndarray:
    """Return dR between particles (B, P) and axes (B, A) as (B, P, A)."""
    dy = y[:, :, None] - ay[:, None, :]
    dphi = _abs_dphi(phi[:, :, None], aphi[:, None, :])
    return np.sqrt(dy * dy + dphi * dphi)


# ------------------------------------------------------------------------------------------------------ kinematics


def pt_y_phi_m(p4: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pT, rapidity y, azimuth phi in (-pi, pi] and mass of four-vectors (px, py, pz, E) on the last axis.

    Elementwise; the rapidity uses fastjet's numerically stable form, an all-zero four-vector (padding) has y = 0 and
    a negative m^2 from rounding gives m = 0.
    """
    return _pt_y_phi_m(np.asarray(p4, dtype=np.float64))


def jet_p4(p4: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the four-momentum (px, py, pz, E) of every jet, the sum of its particles, float64 ``(B, 4)``."""
    p4, mask = _check_particles(p4, mask)
    return p4.sum(axis=1)


def mass(p4: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the jet mass ``sqrt((sum_i p_i)^2)`` in GeV (0 if m^2 < 0 from rounding)."""
    return _pt_y_phi_m(jet_p4(p4, mask))[3]


def multiplicity(mask: np.ndarray) -> np.ndarray:
    """Return the number of particles of every jet (int64)."""
    return np.asarray(mask, dtype=bool).sum(axis=1).astype(np.int64)


# ------------------------------------------------------------------------------------------------------- soft drop


@dataclass(frozen=True)
class SoftDrop:
    """The result of :func:`softdrop`: the groomed jets, float64 ``(B,)`` each (``n_dropped`` int64).

    Attributes
    ----------
    mass : numpy.ndarray
        Mass of the groomed jet in GeV (the mass of the particle when a single particle is reached, 0 for none).
    pt : numpy.ndarray
        pT of the groomed jet in GeV.
    zg : numpy.ndarray
        Momentum fraction ``min(pt1, pt2) / (pt1 + pt2)`` of the splitting that passed (0 if none passed).
    rg : numpy.ndarray
        Delta R of the splitting that passed (0 if none passed).
    n_dropped : numpy.ndarray
        Number of branches removed.
    """

    mass: np.ndarray
    pt: np.ndarray
    zg: np.ndarray
    rg: np.ndarray
    n_dropped: np.ndarray


def _softdrop_tree(tree: ClusterTree, zcut: float, beta: float, R0: float) -> SoftDrop:
    """Run soft drop on an exclusive (C/A) tree."""
    n_jets = len(tree.n_leaves)
    rows = np.arange(n_jets)
    cur = tree.root.copy()
    groomed = np.full(n_jets, -1, dtype=np.int64)
    zg = np.zeros(n_jets)
    rg = np.zeros(n_jets)
    n_dropped = np.zeros(n_jets, dtype=np.int64)
    pt_all = tree.pt
    todo = cur >= 0
    while todo.any():
        r = rows[todo]
        c = tree.children[r, cur[r]]
        leaf = c[:, 0] < 0
        groomed[r[leaf]] = cur[r[leaf]]
        todo[r[leaf]] = False
        r, c = r[~leaf], c[~leaf]
        if not len(r):
            break
        pt1, pt2 = pt_all[r, c[:, 0]], pt_all[r, c[:, 1]]
        dy = tree.direction[r, c[:, 0], 0] - tree.direction[r, c[:, 1], 0]
        dphi = _abs_dphi(tree.direction[r, c[:, 0], 1], tree.direction[r, c[:, 1], 1])
        delta = np.sqrt(dy * dy + dphi * dphi)
        z = np.minimum(pt1, pt2) / (pt1 + pt2)
        passed = z > zcut * (delta / R0) ** beta if beta != 0 else z > zcut
        stop = r[passed]
        groomed[stop] = cur[stop]
        zg[stop] = z[passed]
        rg[stop] = delta[passed]
        todo[stop] = False
        go = ~passed
        n_dropped[r[go]] += 1
        cur[r[go]] = np.where(pt1 >= pt2, c[:, 0], c[:, 1])[go]
    p4 = np.where((groomed >= 0)[:, None], tree.p4[rows, np.maximum(groomed, 0)], 0.0)
    pt, _, _, m = _pt_y_phi_m(p4)
    return SoftDrop(mass=m, pt=pt, zg=zg, rg=rg, n_dropped=n_dropped)


def softdrop(
    p4: np.ndarray,
    mask: np.ndarray,
    *,
    zcut: float = 0.1,
    beta: float = 0.0,
    R0: float = 0.8,
    chunk_size: int = 2048,
) -> SoftDrop:
    """Groom every jet with soft drop (1402.2657; beta = 0 is the modified mass-drop tagger).

    The jet is reclustered with C/A (E-scheme) and declustered from the top: at j -> j1, j2 the declustering stops
    when ``min(pt1, pt2) / (pt1 + pt2) > zcut (dR12 / R0)^beta``, else the softer branch is dropped and the harder
    one declustered. Reaching a single particle gives that particle (mass 0 for a massless one).

    Parameters
    ----------
    p4, mask : numpy.ndarray
        ``(B, N, 4)`` particles and ``(B, N)`` mask.
    zcut, beta, R0 : float
        Soft-drop parameters (the JetClass values are the defaults).
    chunk_size : int
        Jets per chunk. The clustering works on blocks of 128 jets: the peak memory is about 0.13 GB at N = 123,
        independent of ``chunk_size``.

    Returns
    -------
    SoftDrop
    """
    p4, mask = _check_particles(p4, mask)
    parts = [
        _softdrop_tree(cluster(p, m, algorithm="ca", R=R0), zcut, beta, R0) for _, p, m in _blocks(p4, mask, chunk_size)
    ]
    if not parts:
        empty = np.zeros(0)
        return SoftDrop(empty, empty, empty, empty, np.zeros(0, dtype=np.int64))
    return SoftDrop(*(np.concatenate([getattr(s, k) for s in parts]) for k in ("mass", "pt", "zg", "rg", "n_dropped")))


# ----------------------------------------------------------------------------------------------- N-subjettiness


def _onepass(y: np.ndarray, phi: np.ndarray, pt: np.ndarray, ay: np.ndarray, aphi: np.ndarray, beta: float) -> None:
    """Move the axes (B, A) in place to a local minimum of sum_i pt_i min_a dR_ia^beta (Weiszfeld updates)."""
    power = beta - 2.0
    todo = np.ones(len(ay), dtype=bool)
    n_axes = ay.shape[1]
    for _ in range(_ONEPASS_ITERATIONS):
        r = np.flatnonzero(todo)
        if not r.size:
            break
        dr = _delta_r(y[r], phi[r], ay[r], aphi[r])
        nearest = dr.argmin(axis=2)
        dmin = np.take_along_axis(dr, nearest[..., None], axis=2)[..., 0]
        weight = pt[r] * np.maximum(dmin, 1e-12) ** power  # pt = 0 for padding
        onehot = nearest[..., None] == np.arange(n_axes)
        w = weight[..., None] * onehot  # (b, P, A)
        total = w.sum(axis=1)
        rel_phi = _wrap(phi[r][:, :, None] - aphi[r][:, None, :])
        new_y = _safe_divide((w * y[r][:, :, None]).sum(axis=1), total)
        new_phi = aphi[r] + _safe_divide((w * rel_phi).sum(axis=1), total)
        empty = total == 0
        new_y = np.where(empty, ay[r], new_y)
        new_phi = _wrap(np.where(empty, aphi[r], new_phi))
        change = ((new_y - ay[r]) ** 2 + _wrap(new_phi - aphi[r]) ** 2).sum(axis=1)
        ay[r], aphi[r] = new_y, new_phi
        todo[r[change < _ONEPASS_ACCURACY**2]] = False


def _axes_from_tree(
    tree: ClusterTree, n: int, axes: str, y: np.ndarray, phi: np.ndarray, pt: np.ndarray, beta: float
) -> np.ndarray:
    """Return (B, n, 2) axes from an exclusive kt tree; NaN where a jet has fewer than n particles."""
    nodes = tree.exclusive_nodes(n)
    rows = np.arange(len(nodes))[:, None]
    direction = tree.direction[rows, np.maximum(nodes, 0)].copy()
    if axes == "onepass_kt":
        ay, aphi = direction[..., 0].copy(), direction[..., 1].copy()
        full = np.flatnonzero((nodes >= 0).all(axis=1))
        if full.size:
            sub_y, sub_phi = ay[full], aphi[full]
            _onepass(y[full], phi[full], pt[full], sub_y, sub_phi, beta)
            ay[full], aphi[full] = sub_y, sub_phi
        direction = np.stack([ay, aphi], axis=-1)
    direction[nodes < 0] = np.nan
    return direction


def _kt_tree(p4: np.ndarray, mask: np.ndarray, axes: str) -> ClusterTree:
    return cluster(p4, mask, algorithm="kt", recombination="WTA" if axes == "wta_kt" else "E")


def subjet_axes(
    p4: np.ndarray, mask: np.ndarray, n: int, *, axes: Axes = "wta_kt", chunk_size: int = 2048
) -> np.ndarray:
    """Return the n subjet axes of every jet for N-subjettiness.

    Parameters
    ----------
    p4, mask : numpy.ndarray
        ``(B, N, 4)`` particles and ``(B, N)`` mask.
    n : int
        Number of axes.
    axes : {"wta_kt", "kt", "onepass_kt"}
        ``"wta_kt"``: the n exclusive kt subjets with winner-take-all recombination (1401.2158). ``"kt"``: the same
        with the E-scheme. ``"onepass_kt"``: the E-scheme kt axes moved to the nearest local minimum of tau_N
        (beta = 1) by Weiszfeld updates, as fastjet-contrib's ``OnePass_KT_Axes`` (the Delphes CMS card).
    chunk_size : int
        Jets per chunk. The clustering works on blocks of 128 jets: the peak memory is about 0.13 GB at N = 123,
        independent of ``chunk_size``.

    Returns
    -------
    numpy.ndarray
        float64 ``(B, n, 2)`` (y, phi), ordered by subjet pT (descending); NaN for jets with fewer than n particles.
    """
    _check_axes(axes)
    p4, mask = _check_particles(p4, mask)
    out = np.full((len(p4), n, 2), np.nan)
    for rows, p, m in _blocks(p4, mask, chunk_size):
        pt, y, phi, _ = _pt_y_phi_m(p)
        out[rows] = _axes_from_tree(_kt_tree(p, m, axes), n, axes, y, phi, np.where(m, pt, 0.0), 1.0)
    return out


def _tau_block(
    p: np.ndarray, m: np.ndarray, n: Sequence[int], beta: float, R0: float, axes: str, tree: ClusterTree | None = None
) -> np.ndarray:
    """N-subjettiness of one block of jets, (b, len(n))."""
    pt, y, phi, _ = _pt_y_phi_m(p)
    pt = np.where(m, pt, 0.0)
    if tree is None:
        tree = _kt_tree(p, m, axes)
    norm = pt.sum(axis=1) * R0**beta
    out = np.zeros((len(p), len(n)))
    for k, nk in enumerate(n):
        direction = _axes_from_tree(tree, nk, axes, y, phi, pt, beta)
        ok = tree.n_leaves > nk
        if not ok.any():
            continue
        dr = _delta_r(y[ok], phi[ok], direction[ok, :, 0], direction[ok, :, 1]).min(axis=2)
        out[ok, k] = _safe_divide((pt[ok] * dr**beta).sum(axis=1), norm[ok])
    return out


def nsubjettiness(
    p4: np.ndarray,
    mask: np.ndarray,
    *,
    n: Sequence[int] = (1, 2, 3, 4),
    beta: float = 1.0,
    R0: float = 0.8,
    axes: Axes = "wta_kt",
    chunk_size: int = 2048,
) -> np.ndarray:
    """Return the N-subjettiness ``tau_N = sum_i pt_i min_a dR_ia^beta / sum_i pt_i R0^beta`` (1011.2268).

    One exclusive kt clustering per jet gives the axes for every N.

    Parameters
    ----------
    p4, mask : numpy.ndarray
        ``(B, N, 4)`` particles and ``(B, N)`` mask.
    n : sequence of int
        The values of N.
    beta : float
        Angular exponent (> 0 for IRC safety).
    R0 : float
        Jet radius of the normalisation.
    axes : {"wta_kt", "kt", "onepass_kt"}
        The axes (:func:`subjet_axes`).
    chunk_size : int
        Jets per chunk. The clustering works on blocks of 128 jets: the peak memory is about 0.13 GB at N = 123,
        independent of ``chunk_size``.

    Returns
    -------
    numpy.ndarray
        float64 ``(B, len(n))``; 0 for jets with at most N particles (as JetClass).
    """
    _check_axes(axes)
    p4, mask = _check_particles(p4, mask)
    n = [int(k) for k in n]
    if any(k < 1 for k in n):
        raise ValueError(f"every N must be >= 1, got {n}")
    out = np.zeros((len(p4), len(n)))
    for rows, p, m in _blocks(p4, mask, chunk_size):
        out[rows] = _tau_block(p, m, n, beta, R0, axes)
    return out


# ---------------------------------------------------------------------------------- energy correlation functions


#: Jets per block of the ECF matrices inside a chunk (5 arrays of ``_ECF_BLOCK * N^2`` float64 at the peak).
_ECF_BLOCK = 256


def _ecf_block(p: np.ndarray, m: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (e2, e3) of a chunk of jets, building the (b, N, N) matrices for ``_ECF_BLOCK`` jets at a time."""
    pt, y, phi, _ = _pt_y_phi_m(p)
    pt = np.where(m, pt, 0.0)
    z = _safe_divide(pt, pt.sum(axis=1, keepdims=True))
    e2 = np.zeros(len(p))
    e3 = np.zeros(len(p))
    diag = np.arange(p.shape[1])
    for start in range(0, len(p), _ECF_BLOCK):
        rows = slice(start, start + _ECF_BLOCK)
        zb = z[rows]
        za = y[rows, :, None] - y[rows, None, :]
        za *= za
        dphi = _abs_dphi(phi[rows, :, None], phi[rows, None, :])
        dphi *= dphi
        za += dphi
        del dphi
        if beta != 2:
            za **= beta / 2
        za[:, diag, diag] = 0.0
        e2[rows] = 0.5 * (np.matmul(za, zb[:, :, None])[..., 0] * zb).sum(axis=1)
        za *= zb[:, :, None]  # Z A
        e3[rows] = np.einsum("bij,bji->b", np.matmul(za, za), za) / 6.0
    return e2, e3


def ecf(
    p4: np.ndarray, mask: np.ndarray, *, beta: float = 1.0, chunk_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Return the energy correlation functions e2 and e3 (1305.0007) with angular exponent beta.

    ``e2 = 1/2 z^T A z`` and ``e3 = tr((ZA)^3) / 6`` with ``A_ij = dR_ij^beta`` (A_ii = 0) and ``Z = diag(z)``:
    repeated indices drop out because A_ii = 0, so the matrix products give exactly the sums over i < j (< k).

    Parameters
    ----------
    p4, mask : numpy.ndarray
        ``(B, N, 4)`` particles and ``(B, N)`` mask.
    beta : float
        Angular exponent (beta = 2: e2 is about m^2 / pT^2).
    chunk_size : int
        Jets per chunk. The matrices are built for 256 jets at a time: the peak memory is about ``5 * 256 * N^2 * 8``
        bytes (0.15 GB at N = 123), independent of ``chunk_size``.

    Returns
    -------
    tuple of numpy.ndarray
        (e2, e3), float64 ``(B,)`` each.
    """
    p4, mask = _check_particles(p4, mask)
    e2 = np.zeros(len(p4))
    e3 = np.zeros(len(p4))
    for rows, p, m in _blocks(p4, mask, chunk_size):
        e2[rows], e3[rows] = _ecf_block(p, m, beta)
    return e2, e3


def d2(p4: np.ndarray, mask: np.ndarray, *, beta: float = 1.0, chunk_size: int = 2048) -> np.ndarray:
    """Return ``D2 = e3 / e2^3`` (1409.6298); 0 where e2 = 0 (one particle). Memory as :func:`ecf`."""
    e2, e3 = ecf(p4, mask, beta=beta, chunk_size=chunk_size)
    return _safe_divide(e3, e2**3)


def c2(p4: np.ndarray, mask: np.ndarray, *, beta: float = 1.0, chunk_size: int = 2048) -> np.ndarray:
    """Return ``C2 = e3 / e2^2`` (1305.0007); 0 where e2 = 0 (one particle). Memory as :func:`ecf`."""
    e2, e3 = ecf(p4, mask, beta=beta, chunk_size=chunk_size)
    return _safe_divide(e3, e2**2)


# ------------------------------------------------------------------------------------------------------ Lund plane


def lund_primary(p4: np.ndarray, mask: np.ndarray, *, chunk_size: int = 2048) -> dict[str, Ragged]:
    """Return the primary Lund-plane coordinates of every jet (1807.04758).

    The jet is reclustered with C/A (E-scheme) and the harder branch followed from the top; every declustering
    gives ``ln_inv_delta = ln(1 / Delta)``, ``ln_kt = ln(pt_soft Delta / GeV)``, ``z = pt_soft / (pt_a + pt_b)`` and
    ``psi = atan2(dy, dphi)`` (:func:`itj.cluster.decluster_primary`). Jets are clustered ``chunk_size`` at a time
    (peak memory about 0.13 GB at N = 123).

    Returns
    -------
    dict of str to Ragged
        float64, one row per jet, largest angle first.
    """
    p4, mask = _check_particles(p4, mask)
    parts = []
    for _, p, m in _blocks(p4, mask, chunk_size):
        chain = decluster_primary(cluster(p, m, algorithm="ca", R=1.0))
        with np.errstate(divide="ignore"):
            parts.append(
                {
                    "ln_inv_delta": chain["delta"].map(lambda v: -np.log(v)),
                    "ln_kt": chain["kt"].map(np.log),
                    "z": chain["z"],
                    "psi": chain["psi"],
                }
            )
    keys = ("ln_inv_delta", "ln_kt", "z", "psi")
    if not parts:
        empty = Ragged(np.zeros(0), np.zeros(1, dtype=np.int64))
        return dict.fromkeys(keys, empty)
    first = Ragged.concat([part["z"] for part in parts])
    return {k: Ragged(np.concatenate([part[k].values for part in parts]), first.offsets) for k in keys}


def lund_density(
    lund: dict[str, Ragged], *, n_jets: int, x_edges: np.ndarray, y_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the primary Lund density ``(1 / N_jet) dn / (d ln(1/Delta) d ln kt)`` and its statistical error.

    Parameters
    ----------
    lund : dict of str to Ragged
        The output of :func:`lund_primary` (or the toy's ``truth_lund_*`` columns under the same keys).
    n_jets : int
        Number of jets the emissions come from.
    x_edges, y_edges : numpy.ndarray
        Bin edges in ln(1/Delta) and ln(kt / GeV).

    Returns
    -------
    tuple of numpy.ndarray
        Density and its error ``sqrt(n) / (N_jet * area)``, shape ``(len(x_edges) - 1, len(y_edges) - 1)``.
    """
    if n_jets < 1:
        raise ValueError(f"n_jets must be >= 1, got {n_jets}")
    x = np.asarray(lund["ln_inv_delta"].values, dtype=np.float64)
    y = np.asarray(lund["ln_kt"].values, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    counts, _, _ = np.histogram2d(x[ok], y[ok], bins=[np.asarray(x_edges), np.asarray(y_edges)])
    area = np.outer(np.diff(x_edges), np.diff(y_edges)) * n_jets
    return counts / area, np.sqrt(counts) / area


# ------------------------------------------------------------------------------------------------------ angularities


def angularity(p4: np.ndarray, mask: np.ndarray, *, kappa: float, beta: float, R: float = 0.8) -> np.ndarray:
    """Return the generalised angularity ``sum_i z_i^kappa (dR_i / R)^beta`` about the jet axis (1408.3122).

    The axis is the direction of the summed four-momentum; IRC safe only for kappa = 1 and beta > 0 (kappa = 0,
    beta = 0 counts the particles; kappa = 2, beta = 0 is pT_D^2).
    """
    p4, mask = _check_particles(p4, mask)
    pt, y, phi, _ = _pt_y_phi_m(p4)
    pt = np.where(mask, pt, 0.0)
    z = _safe_divide(pt, pt.sum(axis=1, keepdims=True))
    ay, aphi = _jet_axis(p4)
    dy = y - ay[:, None]
    dphi = _abs_dphi(phi, aphi[:, None])
    dr = np.sqrt(dy * dy + dphi * dphi) / R
    terms = np.power(z, kappa) * np.power(dr, beta)
    return np.where(mask, terms, 0.0).sum(axis=1)


def ptd(p4: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return ``pT_D = sqrt(sum_i pt_i^2) / sum_i pt_i`` (collinear unsafe)."""
    p4, mask = _check_particles(p4, mask)
    pt = np.where(mask, np.hypot(p4[..., 0], p4[..., 1]), 0.0)
    return _safe_divide(np.sqrt((pt * pt).sum(axis=1)), pt.sum(axis=1))


def width(p4: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the jet width (girth) ``sum_i z_i dR_i / R`` with R = 0.8 (angularity kappa = 1, beta = 1)."""
    return angularity(p4, mask, kappa=1.0, beta=1.0)


def leading_fraction(p4: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the pT fraction ``max_i pt_i / sum_i pt_i`` of the hardest particle (collinear unsafe)."""
    p4, mask = _check_particles(p4, mask)
    pt = np.where(mask, np.hypot(p4[..., 0], p4[..., 1]), 0.0)
    return _safe_divide(pt.max(axis=1, initial=0.0), pt.sum(axis=1))


# --------------------------------------------------------------------------------------------- JetClass branches


def jet_branches(
    p4: np.ndarray,
    mask: np.ndarray,
    *,
    zcut: float = 0.1,
    sd_beta: float = 0.0,
    R0: float = 0.8,
    tau_beta: float = 1.0,
    tau_axes: Axes = "wta_kt",
    chunk_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Return the ten JetClass jet branches computed from the constituents (the generator and T4.2 use it).

    ``jet_pt``, ``jet_eta`` (pseudorapidity), ``jet_phi`` and ``jet_energy`` of the summed four-momentum,
    ``jet_nparticles``, ``jet_sdmass`` (:func:`softdrop`: C/A, beta ``sd_beta``, ``zcut``, ``R0``) and
    ``jet_tau1`` .. ``jet_tau4`` (:func:`nsubjettiness`: beta ``tau_beta``, ``R0``, ``tau_axes``; 0 for jets with
    at most N particles). Two clusterings per jet (C/A and kt), ``chunk_size`` jets at a time; peak memory about
    0.13 GB at N = 123.

    Returns
    -------
    dict of str to numpy.ndarray
        float32 ``(B,)`` per branch, in the JetClass order.
    """
    _check_axes(tau_axes)
    p4, mask = _check_particles(p4, mask)
    total = p4.sum(axis=1)
    pt = np.hypot(total[:, 0], total[:, 1])
    eta = np.arcsinh(_safe_divide(total[:, 2], pt))
    phi = np.arctan2(total[:, 1], total[:, 0])
    sdmass = np.zeros(len(p4))
    tau = np.zeros((len(p4), 4))
    for rows, p, m in _blocks(p4, mask, chunk_size):
        sdmass[rows] = _softdrop_tree(cluster(p, m, algorithm="ca", R=R0), zcut, sd_beta, R0).mass
        tau[rows] = _tau_block(p, m, (1, 2, 3, 4), tau_beta, R0, tau_axes)
    out = {
        "jet_pt": pt,
        "jet_eta": eta,
        "jet_phi": phi,
        "jet_energy": total[:, 3],
        "jet_nparticles": mask.sum(axis=1),
        "jet_sdmass": sdmass,
    }
    out.update({f"jet_tau{k + 1}": tau[:, k] for k in range(4)})
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def compute(data: JetData, names: Sequence[str], *, chunk_size: int = 2048) -> dict[str, np.ndarray]:
    """Compute several observables of a data set, chunk by chunk, sharing the clusterings between them.

    Parameters
    ----------
    data : JetData
        The jets (``part_px``, ``part_py``, ``part_pz``, ``part_energy``).
    names : sequence of str
        Any of :data:`COMPUTE_NAMES`: ``mass``, ``sdmass`` (soft drop as JetClass), ``tau1`` .. ``tau4`` (WTA-kt
        axes, beta 1), ``tau21``, ``tau32``, ``e2``, ``e3``, ``d2``, ``c2`` (beta 1), ``d2_b2``, ``c2_b2`` (beta 2),
        ``multiplicity``, ``ptd``, ``width``, ``leading_fraction``.
    chunk_size : int
        Jets per chunk (peak memory as :func:`ecf`).

    Returns
    -------
    dict of str to numpy.ndarray
        float64 ``(len(data),)`` per name (``multiplicity`` int64); ratios are 0 where the denominator is 0.

    Examples
    --------
    >>> import numpy as np
    >>> from itj.data import JetData, Ragged
    >>> px = Ragged.from_counts(np.array([400.0, 200.0]), [2])
    >>> zero = Ragged.from_counts(np.zeros(2), [2])
    >>> energy = Ragged.from_counts(np.array([400.0, 200.0 * np.cosh(0.4)]), [2])
    >>> pz = Ragged.from_counts(np.array([0.0, 200.0 * np.sinh(0.4)]), [2])
    >>> data = JetData({"part_px": px, "part_py": zero, "part_pz": pz, "part_energy": energy})
    >>> {k: round(float(v[0]), 3) for k, v in compute(data, ["mass", "multiplicity", "tau21"]).items()}
    {'mass': 113.893, 'multiplicity': 2.0, 'tau21': 0.0}
    """
    names = list(names)
    for name in names:
        if name not in COMPUTE_NAMES:
            close = difflib.get_close_matches(name, COMPUTE_NAMES, n=3)
            hint = f" (did you mean {', '.join(map(repr, close))}?)" if close else ""
            raise ValueError(f"unknown observable {name!r}{hint}: choose from {', '.join(COMPUTE_NAMES)}")
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in names}
    want_tau = any(name.startswith("tau") for name in names)
    for chunk in data.chunks(chunk_size):
        p4, mask = chunk.padded_p4()
        values: dict[str, np.ndarray] = {}
        if want_tau:
            tau = _tau_block(p4, mask, (1, 2, 3, 4), 1.0, 0.8, "wta_kt")
            for k in range(4):
                values[f"tau{k + 1}"] = tau[:, k]
            values["tau21"] = _safe_divide(tau[:, 1], tau[:, 0])
            values["tau32"] = _safe_divide(tau[:, 2], tau[:, 1])
        if {"e2", "e3", "d2", "c2"} & set(names):
            e2, e3 = _ecf_block(p4, mask, 1.0)
            values.update(e2=e2, e3=e3, d2=_safe_divide(e3, e2**3), c2=_safe_divide(e3, e2**2))
        if {"d2_b2", "c2_b2"} & set(names):
            e2, e3 = _ecf_block(p4, mask, 2.0)
            values.update(d2_b2=_safe_divide(e3, e2**3), c2_b2=_safe_divide(e3, e2**2))
        if "sdmass" in names:
            values["sdmass"] = _softdrop_tree(cluster(p4, mask, algorithm="ca", R=0.8), 0.1, 0.0, 0.8).mass
        simple = {"mass": mass, "ptd": ptd, "width": width, "leading_fraction": leading_fraction}
        for name in names:
            if name in simple:
                values[name] = simple[name](p4, mask)
            elif name == "multiplicity":
                values[name] = multiplicity(mask)
            pieces[name].append(values[name])
    return {
        name: np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64 if name == "multiplicity" else np.float64)
        for name, parts in pieces.items()
    }
