"""Jet clustering in pure NumPy: generalised-kt clustering of many jets at once, cones and declustering.

The generalised-kt family (anti-kt, Cambridge/Aachen and kt; 0802.1189, hep-ph/9707323) merges, at every step, the
pair of pseudojets with the smallest distance

.. code-block:: text

    d_ij = min(pt_i^2p, pt_j^2p) dR_ij^2 / R^2,    d_iB = pt_i^2p,    p = -1 (anti-kt), 0 (C/A), 1 (kt)

with ``dR_ij^2 = dy^2 + dphi^2`` in rapidity and azimuth (as fastjet). In the **exclusive** mode (the default) there
is no beam distance: every jet is clustered to one pseudojet and the full binary tree is kept (soft drop, the Lund
plane and the N-subjettiness axes read it). In the **inclusive** mode a pseudojet whose ``d_iB`` is the smallest
distance becomes a jet (the definition of the jets of an event).

All functions take padded particle arrays ``p4`` of shape ``(B, N, 4)`` with (px, py, pz, E) on the last axis and a
``mask`` of shape ``(B, N)`` (``JetData.padded_p4``). The clusterer works on all jets at once: it keeps the full
``(B, N, N)`` distance matrix of a block of jets and repeats "argmin, merge, update one row" N times. Jets are
sorted into blocks of similar multiplicity (``bucket_size`` jets each, every block padded to its own longest jet),
and the results come back in the input order. The cost is O(N^3) per jet; about 100 us per jet and pass at N = 30
on one core (10 000 jets/s). No fastjet is needed (the tests compare with fastjet when it is installed).

Examples
--------
>>> import numpy as np
>>> def massless(pt, y, phi):
...     return [pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(y), pt * np.cosh(y)]
>>> p4 = np.array([[massless(300, 0.0, 0.0), massless(100, 0.1, 0.0), massless(150, 0.5, 0.0)]])
>>> tree = cluster(p4, np.ones((1, 3), bool), algorithm="ca")
>>> tree.children[0, 3].tolist(), tree.children[0, 4].tolist(), int(tree.root[0])
([0, 1], [3, 2], 4)
>>> tree.exclusive_labels(2).tolist()
[[0, 0, 1]]
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import Ragged

__all__ = ["Algorithm", "ClusterTree", "cluster", "cone", "decluster_primary", "leading_jet"]

Algorithm = Literal["antikt", "ca", "kt"]

#: The exponent p of the generalised-kt distance of every algorithm.
_EXPONENT: dict[str, float] = {"antikt": -1.0, "ca": 0.0, "kt": 1.0}

#: Rapidity given to a particle along the beam (pT = 0), as fastjet's ``MaxRap``.
_MAX_RAP = 1e5

_TWO_PI = 2.0 * np.pi


def _check_choice(kind: str, value: str, choices: tuple[str, ...]) -> None:
    """Raise ValueError with a suggestion when ``value`` is not one of ``choices``."""
    if value not in choices:
        close = difflib.get_close_matches(str(value), choices, n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        raise ValueError(f"unknown {kind} {value!r}{hint}: choose from {', '.join(map(repr, choices))}")


def _check_particles(p4: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``p4`` as float64 ``(B, N, 4)`` and ``mask`` as bool ``(B, N)``, with masked slots set to zero."""
    p4 = np.asarray(p4, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if p4.ndim != 3 or p4.shape[-1] != 4:
        raise ValueError(f"p4 must have shape (B, N, 4) with (px, py, pz, E) on the last axis, got {p4.shape}")
    if mask.shape != p4.shape[:2]:
        raise ValueError(f"mask must have shape {p4.shape[:2]} (B, N), got {mask.shape}")
    if not mask.all():
        p4 = np.where(mask[..., None], p4, 0.0)
    return p4, mask


def _pt_y_phi_m(p4: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pT, rapidity, azimuth in (-pi, pi] and mass of four-vectors (px, py, pz, E) on the last axis.

    The rapidity uses fastjet's stable form ``y = -+ ln((E + |pz|)^2 / (pT^2 + m^2)) / 2``; a particle along the beam
    gets ``+-1e5`` and an all-zero four-vector (padding) gets 0. A negative ``m^2`` (rounding) counts as 0.
    """
    px, py, pz, e = p4[..., 0], p4[..., 1], p4[..., 2], p4[..., 3]
    pt2 = px * px + py * py
    m2 = np.maximum(e * e - pt2 - pz * pz, 0.0)
    num = pt2 + m2
    e_plus = e + np.abs(pz)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 0.5 * np.log(num / (e_plus * e_plus))
    y = np.where(pz > 0, -y, y)
    if not np.all(num > 0):
        beam = np.where(e_plus > 0, np.where(pz >= 0, _MAX_RAP, -_MAX_RAP), 0.0)
        y = np.where(num > 0, y, beam)
    return np.sqrt(pt2), y, np.arctan2(py, px), np.sqrt(m2)


def _from_pt_y_phi_m(pt: np.ndarray, y: np.ndarray, phi: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Return four-vectors (px, py, pz, E) from pT, rapidity, azimuth and mass."""
    mt = np.sqrt(pt * pt + m * m)
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y), mt * np.cosh(y)], axis=-1)


def _abs_dphi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return |a - b| wrapped to [0, pi] for azimuths in [-pi, pi]."""
    d = np.abs(a - b)
    return np.minimum(d, _TWO_PI - d)


def _wrap(dphi: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    out = np.mod(dphi + np.pi, _TWO_PI) - np.pi
    return np.where(out == -np.pi, np.pi, out)


def _weight(pt: np.ndarray, p: float) -> np.ndarray:
    """Return pT^(2p) (1 for C/A)."""
    if p == 0:
        return np.ones_like(pt)
    with np.errstate(divide="ignore"):
        return pt**2 if p == 1 else 1.0 / (pt * pt)


@dataclass(frozen=True)
class ClusterTree:
    """The clustering history of a batch of jets (output of :func:`cluster`).

    Nodes ``0 .. N-1`` are the input slots (the particles; unused for padded slots), nodes ``N, N+1, ...`` are the
    pseudojets made by the merges, in merge order: after k merges of a jet its nodes ``N .. N+k-1`` exist. A jet with
    L particles has L - 1 merges in the exclusive mode.

    Attributes
    ----------
    algorithm : str
        ``"antikt"``, ``"ca"`` or ``"kt"``.
    R : float
        Jet radius of the distance.
    recombination : str
        ``"E"`` (four-vectors add) or ``"WTA"`` (winner takes all: pT adds, the harder sets the direction).
    n_leaves : numpy.ndarray
        int64 ``(B,)``, the number of particles of every jet.
    p4 : numpy.ndarray
        float64 ``(B, 2N-1, 4)``, (px, py, pz, E) of every node (zero for unused nodes).
    direction : numpy.ndarray
        float64 ``(B, 2N-1, 2)``, (y, phi) of every node: the WTA direction for ``recombination="WTA"``.
    children : numpy.ndarray
        int64 ``(B, 2N-1, 2)``, the two nodes merged into a node; -1 for leaves and unused nodes.
    parent : numpy.ndarray
        int64 ``(B, 2N-1)``, the node a node was merged into; -1 for the root, beam-merged (inclusive) and unused
        nodes.
    dij : numpy.ndarray
        float64 ``(B, 2N-1)``, the distance d_ij at which the node was made; NaN for leaves and unused nodes.
    root : numpy.ndarray
        int64 ``(B,)``, the final node of the exclusive mode (the particle itself for one particle, -1 for none); -1
        in the inclusive mode.
    jets : numpy.ndarray
        bool ``(B, 2N-1)``, inclusive mode: the nodes merged with the beam (the jets); all False in the exclusive
        mode.
    mask : numpy.ndarray
        bool ``(B, N)``, which input slots hold a particle.
    """

    algorithm: str
    R: float
    recombination: str
    n_leaves: np.ndarray
    p4: np.ndarray
    direction: np.ndarray
    children: np.ndarray
    parent: np.ndarray
    dij: np.ndarray
    root: np.ndarray
    jets: np.ndarray
    mask: np.ndarray

    @property
    def n_slots(self) -> int:
        """N, the number of input slots per jet."""
        return int(self.mask.shape[1])

    @property
    def inclusive(self) -> bool:
        """True for a tree of the inclusive mode (with beam merges)."""
        return bool(self.jets.any()) or bool(len(self.root) and np.all(self.root < 0) and self.n_leaves.any())

    @property
    def pt(self) -> np.ndarray:
        """Transverse momentum of every node, float64 ``(B, 2N-1)`` (for WTA nodes the sum of the merged pT)."""
        return np.hypot(self.p4[..., 0], self.p4[..., 1])

    def exclusive_nodes(self, n: int) -> np.ndarray:
        """Return the n exclusive subjets of every jet: the nodes alive after ``n_leaves - n`` merges.

        Parameters
        ----------
        n : int
            Number of subjets (>= 1).

        Returns
        -------
        numpy.ndarray
            int64 ``(B, n)`` node ids, ordered by pT (descending); -1 where a jet has fewer than n particles.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if self.inclusive:
            raise ValueError("exclusive subjets need a tree of the exclusive mode (cluster(..., inclusive=False))")
        big = self.n_slots
        n_nodes = self.p4.shape[1]
        k = np.maximum(self.n_leaves - n, 0)[:, None]  # merges done
        node = np.arange(n_nodes)[None, :]
        made = np.zeros(self.parent.shape, dtype=bool)
        made[:, :big] = self.mask
        made |= (node >= big) & (node < big + k)
        alive = made & ((self.parent < 0) | (self.parent >= big + k))
        key = np.where(alive, -self.pt, np.inf)
        order = np.argsort(key, axis=1, kind="stable")[:, :n]
        if order.shape[1] < n:  # fewer nodes than n (tiny N)
            order = np.concatenate([order, np.zeros((len(order), n - order.shape[1]), np.int64)], axis=1)
            key = np.concatenate([key, np.full((len(key), n - key.shape[1]), np.inf)], axis=1)
        valid = np.take_along_axis(alive, np.minimum(order, n_nodes - 1), axis=1)
        valid &= np.arange(n)[None, :] < np.minimum(self.n_leaves, n)[:, None]
        return np.where(valid, order, -1).astype(np.int64)

    def _propagate(self, start: np.ndarray) -> np.ndarray:
        """Copy every non-negative node value of ``start`` (B, 2N-1) down to the descendants of the node."""
        out = start.copy()
        rows = np.arange(len(out))
        for v in range(out.shape[1] - 1, self.n_slots - 1, -1):
            value = out[:, v]
            child = self.children[:, v]
            ok = (value >= 0) & (child[:, 0] >= 0)
            if not ok.any():
                continue
            r = rows[ok]
            out[r, child[ok, 0]] = value[ok]
            out[r, child[ok, 1]] = value[ok]
        return out

    def exclusive_labels(self, n: int) -> np.ndarray:
        """Return the exclusive subjet of every particle for n subjets.

        Parameters
        ----------
        n : int
            Number of subjets (>= 1).

        Returns
        -------
        numpy.ndarray
            int64 ``(B, N)``: the index 0 .. n-1 of the subjet of every particle (subjets ordered by pT,
            descending); -1 for padding.
        """
        nodes = self.exclusive_nodes(n)
        label = np.full(self.parent.shape, -1, dtype=np.int64)
        b, a = np.nonzero(nodes >= 0)
        label[b, nodes[b, a]] = a
        return self._propagate(label)[:, : self.n_slots]

    def leaves_of(self, node: np.ndarray) -> np.ndarray:
        """Return the particles of one node per jet.

        Parameters
        ----------
        node : numpy.ndarray
            int ``(B,)`` node ids; -1 gives no particles.

        Returns
        -------
        numpy.ndarray
            bool ``(B, N)``, True for the particles (leaves) below ``node``.
        """
        node = np.asarray(node, dtype=np.int64).reshape(-1)
        if len(node) != len(self.n_leaves):
            raise ValueError(f"one node per jet: got {len(node)} for {len(self.n_leaves)} jets")
        flag = np.full(self.parent.shape, -1, dtype=np.int64)
        b = np.flatnonzero(node >= 0)
        flag[b, node[b]] = 1
        return self._propagate(flag)[:, : self.n_slots] > 0


def _cluster_block(
    p4: np.ndarray, mask: np.ndarray, p: float, R: float, wta: bool, inclusive: bool
) -> dict[str, np.ndarray]:
    """Cluster one block of jets (padded to its own length n); node ids are local (merges start at n)."""
    nb, n, _ = p4.shape
    n_nodes = 2 * n - 1
    pt, y, phi, _ = _pt_y_phi_m(p4)
    # WTA keeps the mass of the harder; fastjet's PseudoJet.m() is -sqrt(-m^2) for m^2 < 0 (rounding) and its
    # reset_PtYPhiM squares it again, so |m^2| is what it uses
    m = np.sqrt(np.abs(p4[..., 3] ** 2 - (p4[..., :3] ** 2).sum(axis=-1)))
    out_p4 = np.zeros((nb, n_nodes, 4))
    out_p4[:, :n] = p4
    out_dir = np.zeros((nb, n_nodes, 2))
    out_dir[:, :n, 0] = np.where(mask, y, 0.0)
    out_dir[:, :n, 1] = np.where(mask, phi, 0.0)
    children = np.full((nb, n_nodes, 2), -1, dtype=np.int64)
    parent = np.full((nb, n_nodes), -1, dtype=np.int64)
    dij = np.full((nb, n_nodes), np.nan)
    jets = np.zeros((nb, n_nodes), dtype=bool)
    root = np.full(nb, -1, dtype=np.int64)

    inv_r2 = 1.0 / (R * R)
    # state of the current pseudojets, one per slot (rows are compacted when jets finish)
    act = mask.copy()
    cur_p4 = p4.copy()
    cur_pt, cur_y, cur_phi, cur_m = pt.copy(), y.copy(), phi.copy(), m.copy()
    cur_w = np.where(mask, _weight(np.where(mask, pt, 1.0), p), 1.0)
    slot = np.broadcast_to(np.arange(n), (nb, n)).copy()
    nact = act.sum(axis=1)
    step = np.zeros(nb, dtype=np.int64)
    idmap = np.arange(nb)

    dy = y[:, :, None] - y[:, None, :]
    dphi = _abs_dphi(phi[:, :, None], phi[:, None, :])
    d = np.minimum(cur_w[:, :, None], cur_w[:, None, :]) * (dy * dy + dphi * dphi) * inv_r2
    del dy, dphi
    d[~(act[:, :, None] & act[:, None, :])] = np.inf
    diag = np.arange(n)
    d[:, diag, diag] = np.inf
    dib = np.where(act, cur_w, np.inf) if inclusive else None
    limit = 0 if inclusive else 1

    while True:
        live = nact > limit
        n_live = int(live.sum())
        if n_live == 0:
            break
        if n_live < 0.6 * len(idmap) and len(idmap) > 8:  # drop finished jets from the state
            keep = np.flatnonzero(live)
            d = d[keep]
            act, cur_p4, cur_w, slot = act[keep], cur_p4[keep], cur_w[keep], slot[keep]
            cur_pt, cur_y, cur_phi, cur_m = cur_pt[keep], cur_y[keep], cur_phi[keep], cur_m[keep]
            nact, step, idmap = nact[keep], step[keep], idmap[keep]
            if dib is not None:
                dib = dib[keep]
            live = np.ones(len(keep), dtype=bool)
        rows = np.arange(len(idmap))
        flat = d.reshape(len(idmap), n * n)
        k = flat.argmin(axis=1)
        dmin = flat[rows, k]
        merge = live
        if dib is not None:
            kb = dib.argmin(axis=1)
            beam = live & (dib[rows, kb] <= dmin)
            merge = live & ~beam
            if beam.any():
                r = rows[beam]
                a = kb[beam]
                jets[idmap[r], slot[r, a]] = True
                act[r, a] = False
                nact[r] -= 1
                d[r, a, :] = np.inf
                d[r, :, a] = np.inf
                dib[r, a] = np.inf
        if not merge.any():
            continue
        r = rows[merge]
        orig = idmap[r]
        i, j = np.divmod(k[merge], n)
        a, b = np.minimum(i, j), np.maximum(i, j)
        node_a, node_b = slot[r, a], slot[r, b]
        new = n + step[r]
        if wta:
            pt_a, pt_b = cur_pt[r, a], cur_pt[r, b]
            harder = pt_a >= pt_b
            pt_n = pt_a + pt_b
            y_n = np.where(harder, cur_y[r, a], cur_y[r, b])
            phi_n = np.where(harder, cur_phi[r, a], cur_phi[r, b])
            m_n = np.where(harder, cur_m[r, a], cur_m[r, b])
            p4_n = _from_pt_y_phi_m(pt_n, y_n, phi_n, m_n)
            cur_m[r, a] = m_n
        else:
            p4_n = cur_p4[r, a] + cur_p4[r, b]
            pt_n, y_n, phi_n, _ = _pt_y_phi_m(p4_n)
            cur_p4[r, a] = p4_n
        out_p4[orig, new] = p4_n
        out_dir[orig, new, 0] = y_n
        out_dir[orig, new, 1] = phi_n
        children[orig, new, 0] = node_a
        children[orig, new, 1] = node_b
        parent[orig, node_a] = new
        parent[orig, node_b] = new
        dij[orig, new] = dmin[merge]
        cur_pt[r, a], cur_y[r, a], cur_phi[r, a] = pt_n, y_n, phi_n
        w_n = _weight(pt_n, p)
        cur_w[r, a] = w_n
        slot[r, a] = new
        act[r, b] = False
        nact[r] -= 1
        step[r] += 1
        d[r, b, :] = np.inf
        d[r, :, b] = np.inf
        dy_row = y_n[:, None] - cur_y[r]
        dphi_row = _abs_dphi(phi_n[:, None], cur_phi[r])
        row = np.minimum(w_n[:, None], cur_w[r]) * (dy_row * dy_row + dphi_row * dphi_row) * inv_r2
        row[~act[r]] = np.inf
        row[np.arange(len(r)), a] = np.inf
        d[r, a, :] = row
        d[r, :, a] = row
        if dib is not None:
            dib[r, a] = w_n
            dib[r, b] = np.inf

    if not inclusive:
        count = mask.sum(axis=1)
        last = n + count - 2
        root = np.where(count > 1, last, -1)
        single = np.flatnonzero(count == 1)
        if single.size:
            root[single] = np.argmax(mask[single], axis=1)
    return {
        "p4": out_p4,
        "direction": out_dir,
        "children": children,
        "parent": parent,
        "dij": dij,
        "jets": jets,
        "root": root,
    }


def cluster(
    p4: np.ndarray,
    mask: np.ndarray,
    *,
    algorithm: Algorithm,
    R: float = 0.8,
    recombination: Literal["E", "WTA"] = "E",
    inclusive: bool = False,
    bucket_size: int = 128,
) -> ClusterTree:
    """Cluster the particles of every jet with a generalised-kt algorithm and keep the full history.

    Parameters
    ----------
    p4 : numpy.ndarray
        ``(B, N, 4)`` four-momenta (px, py, pz, E) in GeV.
    mask : numpy.ndarray
        ``(B, N)`` bool, True for the particles (any slots; padded slots are ignored).
    algorithm : {"antikt", "ca", "kt"}
        p = -1, 0 or 1 in ``d_ij = min(pt_i^2p, pt_j^2p) dR_ij^2 / R^2`` and ``d_iB = pt_i^2p``.
    R : float
        Jet radius.
    recombination : {"E", "WTA"}
        ``"E"``: four-vectors add. ``"WTA"`` (winner takes all, fastjet's ``WTA_pt_scheme``): the pT add, the
        rapidity, azimuth and mass are those of the harder (higher pT) pseudojet.
    inclusive : bool
        False (exclusive mode): merge with d_ij only, down to one pseudojet per jet. True: a pseudojet whose
        ``d_iB`` is the smallest distance becomes a jet (``ClusterTree.jets``).
    bucket_size : int
        Jets per block; jets are sorted by multiplicity into blocks, each padded to its own longest jet.

    Returns
    -------
    ClusterTree
        The history, in the input order of the jets.

    Notes
    -----
    Peak memory is about ``3 * bucket_size * N^2 * 8`` bytes (0.2 GB at N = 128 for the default block). Ties of
    distances are broken by the lowest slot pair, so exactly equal distances may give another (equally valid)
    history than fastjet.
    """
    _check_choice("algorithm", algorithm, tuple(_EXPONENT))
    _check_choice("recombination", recombination, ("E", "WTA"))
    if not R > 0:
        raise ValueError(f"R must be > 0, got {R}")
    if bucket_size < 1:
        raise ValueError(f"bucket_size must be >= 1, got {bucket_size}")
    p4, mask = _check_particles(p4, mask)
    n_jets, big = mask.shape
    n_nodes = max(2 * big - 1, 0)
    out_p4 = np.zeros((n_jets, n_nodes, 4))
    out_dir = np.zeros((n_jets, n_nodes, 2))
    children = np.full((n_jets, n_nodes, 2), -1, dtype=np.int64)
    parent = np.full((n_jets, n_nodes), -1, dtype=np.int64)
    dij = np.full((n_jets, n_nodes), np.nan)
    jets = np.zeros((n_jets, n_nodes), dtype=bool)
    root = np.full(n_jets, -1, dtype=np.int64)
    n_leaves = mask.sum(axis=1).astype(np.int64)
    # the length a jet needs: its last particle slot + 1 (masks need not be left-aligned)
    length = np.where(n_leaves > 0, big - np.argmax(mask[:, ::-1], axis=1), 0) if big else n_leaves
    order = np.argsort(length, kind="stable")
    p = _EXPONENT[algorithm]

    def remap(ids: np.ndarray, n: int) -> np.ndarray:
        return np.where(ids >= n, ids - n + big, ids)

    for start in range(0, n_jets, bucket_size):
        idx = order[start : start + bucket_size]
        n = int(np.max(length[idx])) if len(idx) else 0
        if n == 0:
            continue
        res = _cluster_block(p4[idx, :n], mask[idx, :n], p, float(R), recombination == "WTA", inclusive)
        merged = slice(big, big + n - 1)
        for name, out in (("p4", out_p4), ("direction", out_dir), ("dij", dij), ("jets", jets)):
            out[idx, :n] = res[name][:, :n]
            out[idx, merged] = res[name][:, n:]
        for name, out in (("children", children), ("parent", parent)):
            ids = remap(res[name], n)
            out[idx, :n] = ids[:, :n]
            out[idx, merged] = ids[:, n:]
        root[idx] = remap(res["root"], n)
    return ClusterTree(
        algorithm=algorithm,
        R=float(R),
        recombination=recombination,
        n_leaves=n_leaves,
        p4=out_p4,
        direction=out_dir,
        children=children,
        parent=parent,
        dij=dij,
        root=root,
        jets=jets,
        mask=mask,
    )


def leading_jet(p4: np.ndarray, mask: np.ndarray, *, algorithm: Algorithm = "antikt", R: float = 0.8) -> np.ndarray:
    """Return the constituents of the leading (highest pT) inclusive jet of every event.

    Parameters
    ----------
    p4 : numpy.ndarray
        ``(B, N, 4)`` four-momenta (px, py, pz, E) of all particles of every event.
    mask : numpy.ndarray
        ``(B, N)`` bool.
    algorithm : {"antikt", "ca", "kt"}
        Jet algorithm (inclusive mode, E-scheme).
    R : float
        Jet radius.

    Returns
    -------
    numpy.ndarray
        bool ``(B, N)``, True for the particles of the leading jet (all False for an event without particles).
    """
    tree = cluster(p4, mask, algorithm=algorithm, R=R, inclusive=True)
    pt = np.where(tree.jets, tree.pt, -1.0)
    best = np.argmax(pt, axis=1) if pt.shape[1] else np.zeros(len(pt), dtype=np.int64)
    has_jet = tree.jets.any(axis=1)
    return tree.leaves_of(np.where(has_jet, best, -1))


def cone(p4: np.ndarray, mask: np.ndarray, axis: np.ndarray, *, R: float = 0.8) -> np.ndarray:
    """Return the particles within ``dR < R`` of an axis (a fixed cone, the fast alternative to anti-kt).

    Parameters
    ----------
    p4 : numpy.ndarray
        ``(B, N, 4)`` four-momenta (px, py, pz, E).
    mask : numpy.ndarray
        ``(B, N)`` bool.
    axis : numpy.ndarray
        ``(B, 2)`` (y, phi) of the cone axis of every event.
    R : float
        Cone radius.

    Returns
    -------
    numpy.ndarray
        bool ``(B, N)``.
    """
    p4, mask = _check_particles(p4, mask)
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (len(p4), 2):
        raise ValueError(f"axis must have shape ({len(p4)}, 2) (y, phi), got {axis.shape}")
    _, y, phi, _ = _pt_y_phi_m(p4)
    dy = y - axis[:, :1]
    dphi = _wrap(phi - axis[:, 1:])
    return mask & (dy * dy + dphi * dphi < R * R)


def decluster_primary(tree: ClusterTree) -> dict[str, Ragged]:
    """Follow the harder branch from the top of the tree and record every declustering (the primary Lund chain).

    At every node j -> a, b with a the harder by pT: ``delta`` = Delta R_ab, ``kt = pt_b * delta``,
    ``z = pt_b / (pt_a + pt_b)``, ``psi = atan2(y_b - y_a, wrap(phi_b - phi_a))``, ``m`` = mass of j, ``harder`` = a
    and ``softer`` = b (node ids); then continue with a. Directions are ``tree.direction``.

    Parameters
    ----------
    tree : ClusterTree
        Usually C/A with the E-scheme. For an inclusive tree the chain starts at the hardest jet.

    Returns
    -------
    dict of str to Ragged
        Keys ``delta``, ``kt``, ``z``, ``psi``, ``m`` (float64) and ``harder``, ``softer`` (int64), one row per jet
        in the order of the declusterings (largest angle first for C/A).
    """
    n_jets = len(tree.n_leaves)
    if tree.inclusive:
        pt = np.where(tree.jets, tree.pt, -1.0)
        cur = np.where(tree.jets.any(axis=1), np.argmax(pt, axis=1), -1)
    else:
        cur = tree.root.copy()
    pt_all = tree.pt
    rows = np.arange(n_jets)
    records: dict[str, list[np.ndarray]] = {k: [] for k in ("jet", "delta", "kt", "z", "psi", "m", "harder", "softer")}
    while True:
        ok = cur >= 0
        child = np.where(ok[:, None], tree.children[rows, np.maximum(cur, 0)], -1)
        go = child[:, 0] >= 0
        if not go.any():
            break
        r = rows[go]
        c1, c2 = child[go, 0], child[go, 1]
        pt1, pt2 = pt_all[r, c1], pt_all[r, c2]
        first = pt1 >= pt2
        hard = np.where(first, c1, c2)
        soft = np.where(first, c2, c1)
        pt_h, pt_s = np.maximum(pt1, pt2), np.minimum(pt1, pt2)
        dy = tree.direction[r, soft, 0] - tree.direction[r, hard, 0]
        dphi = _wrap(tree.direction[r, soft, 1] - tree.direction[r, hard, 1])
        delta = np.hypot(dy, dphi)
        parent_p4 = tree.p4[r, cur[go]]
        m2 = parent_p4[:, 3] ** 2 - (parent_p4[:, :3] ** 2).sum(axis=1)
        records["jet"].append(r)
        records["delta"].append(delta)
        records["kt"].append(pt_s * delta)
        records["z"].append(pt_s / (pt_h + pt_s))
        records["psi"].append(np.arctan2(dy, dphi))
        records["m"].append(np.sqrt(np.maximum(m2, 0.0)))
        records["harder"].append(hard)
        records["softer"].append(soft)
        cur = np.full(n_jets, -1, dtype=np.int64)
        cur[r] = hard
    if records["jet"]:
        jet = np.concatenate(records["jet"])
        order = np.argsort(jet, kind="stable")
        counts = np.bincount(jet, minlength=n_jets)
    else:
        order = np.zeros(0, dtype=np.int64)
        counts = np.zeros(n_jets, dtype=np.int64)
    out: dict[str, Ragged] = {}
    offsets: np.ndarray | None = None
    for key in ("delta", "kt", "z", "psi", "m", "harder", "softer"):
        dtype = np.int64 if key in ("harder", "softer") else np.float64
        values = np.concatenate(records[key]).astype(dtype)[order] if records[key] else np.zeros(0, dtype=dtype)
        rag = Ragged.from_counts(values, counts) if offsets is None else Ragged(values, offsets)
        offsets = rag.offsets
        out[key] = rag
    return out
