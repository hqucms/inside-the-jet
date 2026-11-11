"""Hadronisation: independent (Field-Feynman) fragmentation chains, pi0 decays and B hadrons.

Every final parton starts a chain (gluons like quarks; ``docs/physics.md#hadronisation``). Per rank the species is
drawn first, then the energy fraction z from ``f(z) = (1 - a) + 3 a (1 - z)^2``; the hadron takes
``E_h = max(z E_rem, e_min_hadron, m_h + 0.05 GeV)``. When ``E_rem < 2 e_stop`` (or when a hadron would leave less
than ``e_min_hadron``) the rest of the energy becomes the last hadron, whose species is drawn again among those
lighter than the remainder. Energy is conserved exactly per chain. Transverse momenta ``k_i ~ N(0, kt_sigma^2)``
per break and component are compensated within the chain (hadron i gets ``k_i - k_{i-1}``, the last one
``-k_last``); a transverse momentum above 0.9 |p| is scaled down to 0.9 |p| (counted as capped).

The b quarks of the hard process (top decays, b-initiated QCD jets) start their chain with a B hadron (z from
``z^alpha (1 - z)`` with mean ``b_frag_mean``) that flies an exponential proper length ``c tau`` and decays into
``b_decay_multiplicity`` massless products (RAMBO): charged hadrons, photons, semileptonic and cascade leptons, each
lepton with an invisible neutrino (dropped). pi0 -> gamma gamma is isotropic in the pi0 rest frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from .kinematics import boost, direction, perp_basis, rambo

if TYPE_CHECKING:
    from .config import HadronPart
    from .shower import PartonArrays

__all__ = [
    "B_MASS",
    "MASSES",
    "SPECIES",
    "HadronArrays",
    "decay_pi0",
    "draw_species",
    "ff_z",
    "hadronize",
    "species_probabilities",
]

#: Species codes of :class:`HadronArrays` (``species``): stable final-state particles, then the decaying ones.
SPECIES: tuple[str, ...] = ("pi+-", "K+-", "pi0", "K_L", "n", "photon", "e", "mu", "nu", "B", "parton")
PI_CH, K_CH, PI0, K_L, NEUTRON, PHOTON, ELECTRON, MUON, NEUTRINO, B_HADRON, PARTON = range(len(SPECIES))

#: Masses in GeV by species code (B-decay products are massless whatever their species).
MASSES = np.array([0.13957, 0.493677, 0.134977, 0.497611, 0.939565, 0.0, 0.000511, 0.105658, 0.0, 5.279, 0.0])

#: Mass of the B hadrons in GeV.
B_MASS = 5.279

#: The species a chain can produce, in the order of :func:`species_probabilities`.
_CHAIN_SPECIES = np.array([PI_CH, K_CH, PI0, K_L, NEUTRON])

#: Extra energy above the mass that a hadron of a chain keeps (GeV).
_MASS_MARGIN = 0.05

#: Transverse momenta are capped at this fraction of |p|.
_KT_CAP = 0.9


@dataclass
class HadronArrays:
    """Structure of arrays of the hadrons (or partons) of a group of events.

    Attributes
    ----------
    p4 : numpy.ndarray
        ``(h, 4)`` four-vectors (generation frame until moved to the lab).
    vertex : numpy.ndarray
        ``(h, 4)`` production point ``(x, y, z, t)`` in mm (a four-vector: it moves to the lab like ``p4``).
    species, charge : numpy.ndarray
        ``(h,)`` int8 species code (:data:`SPECIES`) and charge.
    event : numpy.ndarray
        ``(h,)`` int64 event row in the group.
    index : numpy.ndarray
        ``(h,)`` int64 position among the hadrons of the same origin (jet or underlying event) of its event, in
        production order (a key of the detector randomness).
    prong, parton : numpy.ndarray
        ``(h,)`` prong and index of the parent parton among the final partons of its event (-1 for the UE).
    from_b, is_ue : numpy.ndarray
        ``(h,)`` bool flags.
    """

    p4: np.ndarray
    vertex: np.ndarray
    species: np.ndarray
    charge: np.ndarray
    event: np.ndarray
    index: np.ndarray
    prong: np.ndarray
    parton: np.ndarray
    from_b: np.ndarray
    is_ue: np.ndarray

    def __len__(self) -> int:
        return len(self.event)

    def select(self, rows: np.ndarray) -> HadronArrays:
        """Return the rows ``rows``."""
        return HadronArrays(**{f.name: getattr(self, f.name)[rows] for f in fields(self)})

    @staticmethod
    def concat(parts: list[HadronArrays]) -> HadronArrays:
        """Concatenate row-wise."""
        return HadronArrays(
            **{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(HadronArrays)}
        )

    @staticmethod
    def from_partons(partons: PartonArrays) -> HadronArrays:
        """Turn the partons into particles (presets ``partons`` and ``shower``: species ``parton``, no charge)."""
        m = len(partons)
        index = partons.index_in_event()
        return HadronArrays(
            p4=partons.p4.copy(),
            vertex=np.zeros((m, 4)),
            species=np.full(m, PARTON, dtype=np.int8),
            charge=np.zeros(m, dtype=np.int8),
            event=partons.event.astype(np.int64),
            index=index,
            prong=partons.prong.astype(np.int64),
            parton=index.copy(),
            from_b=np.zeros(m, dtype=bool),
            is_ue=np.zeros(m, dtype=bool),
        )


# ----------------------------------------------------------------------------------------------------------------------
# Building blocks


def species_probabilities(cfg: HadronPart) -> np.ndarray:
    """Probabilities of pi+-, K+-, pi0, K_L and n per hadron of a chain (species drawn before the energy)."""
    charged = 1.0 - cfg.pi0_fraction - cfg.neutral_fraction
    return np.array(
        [
            charged * (1.0 - cfg.kaon_fraction),
            charged * cfg.kaon_fraction,
            cfg.pi0_fraction,
            cfg.neutral_fraction * cfg.klong_fraction,
            cfg.neutral_fraction * (1.0 - cfg.klong_fraction),
        ]
    )


def draw_species(u: np.ndarray, probs: np.ndarray, e_max: np.ndarray | None = None) -> np.ndarray:
    """Species codes from uniforms ``u``; with ``e_max`` only species lighter than ``e_max`` (pi+- if none fits).

    Parameters
    ----------
    u : numpy.ndarray
        ``(k,)`` uniforms in [0, 1).
    probs : numpy.ndarray
        ``(5,)`` from :func:`species_probabilities`.
    e_max : numpy.ndarray, optional
        ``(k,)`` energy available for the hadron.
    """
    p = np.broadcast_to(probs, (len(u), len(probs))).copy()
    if e_max is not None:
        p[MASSES[_CHAIN_SPECIES][None, :] >= np.asarray(e_max)[:, None]] = 0.0
    cum = np.cumsum(p, axis=1)
    total = cum[:, -1]
    idx = np.sum(cum <= (u * total)[:, None], axis=1)
    idx = np.where(total > 0, np.minimum(idx, len(probs) - 1), 0)
    return _CHAIN_SPECIES[idx]


def ff_z(u: np.ndarray, a: float) -> np.ndarray:
    """Energy fraction z from ``f(z) = (1 - a) + 3 a (1 - z)^2`` by the inverse CDF (Cardano's formula).

    With ``w = 1 - z`` the CDF is ``1 - (1 - a) w - a w^3``; ``a w^3 + (1 - a) w = 1 - u`` has one real root.
    """
    rhs = 1.0 - np.asarray(u, dtype=np.float64)
    if a < 1e-9:
        return 1.0 - rhs
    p = (1.0 - a) / a
    q = -rhs / a
    disc = np.sqrt(q * q / 4.0 + p**3 / 27.0)
    w = np.cbrt(-q / 2.0 + disc) + np.cbrt(-q / 2.0 - disc)
    return np.clip(1.0 - w, 0.0, 1.0)


def decay_pi0(p4: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pi0 -> gamma gamma, isotropic in the pi0 rest frame; ``u`` ``(2, k)`` uniforms; returns the two photons."""
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - np.sum(p4[:, :3] ** 2, axis=1), 0.0))
    d = direction(2.0 * u[0] - 1.0, 2.0 * math.pi * u[1]) * (0.5 * m)[:, None]
    g1 = np.concatenate([d, (0.5 * m)[:, None]], axis=1)
    g2 = np.concatenate([-d, (0.5 * m)[:, None]], axis=1)
    beta = p4[:, :3] / p4[:, 3:4]
    return boost(g1, beta), boost(g2, beta)


def _b_alpha(mean: float) -> float:
    """Exponent alpha of ``f(z) ~ z^alpha (1 - z)`` with the mean ``(alpha + 1) / (alpha + 3)``."""
    return (3.0 * mean - 1.0) / (1.0 - mean)


# ----------------------------------------------------------------------------------------------------------------------
# Chains


@dataclass
class _Chains:
    """Per-hadron output of the chains, in rank-major order."""

    p4: list[np.ndarray]
    species: list[np.ndarray]
    charge: list[np.ndarray]
    chain: list[np.ndarray]
    rank: list[np.ndarray]


def _fragment(
    partons: PartonArrays, cfg: HadronPart, rng: np.random.Generator, stats: dict[str, int]
) -> tuple[np.ndarray, ...]:
    """Run all chains together, one rank per iteration; returns per-hadron arrays sorted by (chain, rank)."""
    m = len(partons)
    energy = partons.p4[:, 3].copy()
    n = partons.p4[:, :3] / np.maximum(np.linalg.norm(partons.p4[:, :3], axis=1, keepdims=True), 1e-300)
    e1, e2 = perp_basis(n)
    probs = species_probabilities(cfg)
    alpha_b = _b_alpha(cfg.b_frag_mean)
    first_b = partons.hard_b & cfg.b_hadrons
    out = _Chains([], [], [], [], [])
    chain = np.arange(m)
    e_rem = energy
    k_prev = np.zeros((m, 2))
    rank = 0
    n_capped = 0
    while len(chain):
        k = len(chain)
        u = rng.random((3, k))
        kt_new = rng.standard_normal((k, 2)) * cfg.kt_sigma
        species = draw_species(u[0], probs)
        is_b = first_b[chain] & (e_rem > B_MASS + _MASS_MARGIN + cfg.e_min_hadron) if rank == 0 else np.zeros(k, bool)
        species = np.where(is_b, B_HADRON, species)
        z = ff_z(u[1], cfg.ff_a)
        if is_b.any():
            z[is_b] = rng.beta(alpha_b + 1.0, 2.0, int(is_b.sum()))
        m_h = MASSES[species]
        e_h = np.maximum(np.maximum(z * e_rem, cfg.e_min_hadron), m_h + _MASS_MARGIN)
        e_h = np.where(is_b, np.minimum(e_h, e_rem - cfg.e_min_hadron), e_h)  # a B hadron leaves room for one more
        last = ((e_rem < 2.0 * cfg.e_stop) | (e_rem - e_h < cfg.e_min_hadron)) & ~is_b
        if last.any():
            species[last] = draw_species(u[0, last], probs, e_rem[last])
        e_h = np.where(last, e_rem, e_h)
        m_h = MASSES[species]
        # a remainder lighter than a charged pion (a soft parton below the pion mass): a photon
        too_light = last & (e_rem <= m_h)
        species[too_light] = PHOTON
        m_h = np.where(too_light, 0.0, m_h)
        kt = np.where(last[:, None], -k_prev, kt_new - k_prev)
        p_abs = np.sqrt(np.maximum(e_h * e_h - m_h * m_h, 0.0))
        kt_abs = np.hypot(kt[:, 0], kt[:, 1])
        capped = kt_abs > _KT_CAP * p_abs
        n_capped += int(capped.sum())
        scale = np.where(capped, _KT_CAP * p_abs / np.maximum(kt_abs, 1e-300), 1.0)
        kt_used = kt * scale[:, None]
        p_long = np.sqrt(np.maximum(p_abs * p_abs - np.sum(kt_used * kt_used, axis=1), 0.0))
        p3 = p_long[:, None] * n[chain] + kt_used[:, :1] * e1[chain] + kt_used[:, 1:] * e2[chain]
        out.p4.append(np.concatenate([p3, e_h[:, None]], axis=1))
        out.species.append(species.astype(np.int8))
        charged = (species == PI_CH) | (species == K_CH)
        out.charge.append(np.where(charged, np.where(u[2] < 0.5, 1, -1), 0).astype(np.int8))
        out.chain.append(chain)
        out.rank.append(np.full(k, rank))
        keep = ~last
        chain, e_rem, k_prev = chain[keep], (e_rem - e_h)[keep], kt_new[keep]
        rank += 1
    stats["hadrons_capped"] = stats.get("hadrons_capped", 0) + n_capped
    p4 = np.concatenate(out.p4) if out.p4 else np.zeros((0, 4))
    species_all = np.concatenate(out.species) if out.species else np.zeros(0, np.int8)
    charge_all = np.concatenate(out.charge) if out.charge else np.zeros(0, np.int8)
    chain_all = np.concatenate(out.chain) if out.chain else np.zeros(0, np.int64)
    rank_all = np.concatenate(out.rank) if out.rank else np.zeros(0, np.int64)
    order = np.lexsort((rank_all, chain_all))
    return p4[order], species_all[order], charge_all[order], chain_all[order], rank_all[order]


def _b_decays(
    p4: np.ndarray, cfg: HadronPart, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decay B hadrons ``p4`` ``(b, 4)``; returns daughters (p4, species, charge, parent row, slot) and vertices."""
    nb, nd = len(p4), cfg.b_decay_multiplicity
    u = rng.random((8, nb))
    flight = -np.log1p(-u[0]) * cfg.b_ctau  # proper decay length
    vertex = (flight / B_MASS)[:, None] * p4  # x^mu = c tau' p^mu / m: a four-vector
    rest = rambo(np.full(nb, B_MASS), rng.random((4, nb, nd)))
    beta = p4[:, :3] / p4[:, 3:4]
    daughters = boost(rest.reshape(nb * nd, 4), np.repeat(beta, nd, axis=0)).reshape(nb, nd, 4)
    kind = rng.random((nb, nd))
    signs = np.where(rng.random((nb, nd)) < 0.5, 1, -1)
    species = np.where(kind < cfg.b_charged_share, PI_CH, PHOTON).astype(np.int8)
    # semileptonic (e, mu) and cascade (e, mu) leptons, each with a neutrino, in the next free slots
    used = np.zeros(nb, dtype=np.int64)
    rows = np.arange(nb)
    for i, lepton in enumerate((ELECTRON, MUON, ELECTRON, MUON)):
        prob = cfg.semileptonic if i < 2 else cfg.semileptonic_cascade
        put = (u[1 + i] < prob) & (used + 2 <= nd)
        species[rows[put], used[put]] = lepton
        species[rows[put], used[put] + 1] = NEUTRINO
        used = used + 2 * put
    charged = (species == PI_CH) | (species == ELECTRON) | (species == MUON)
    charge = np.where(charged, signs, 0).astype(np.int8)
    parent = np.repeat(rows, nd)
    slot = np.tile(np.arange(nd), nb)
    return daughters.reshape(nb * nd, 4), species.reshape(-1), charge.reshape(-1), parent, slot, vertex


def hadronize(
    partons: PartonArrays, cfg: HadronPart, rng: np.random.Generator, stats: dict[str, int] | None = None
) -> HadronArrays:
    """Turn the final partons of a group into hadrons (generation frame; neutrinos dropped).

    Parameters
    ----------
    partons : PartonArrays
        Final partons, sorted by event (the parton index of a hadron is its parent's position in its event).
    cfg : HadronPart
        Hadronisation parameters.
    rng : numpy.random.Generator
        The hadron stream.
    stats : dict, optional
        Counters updated in place: ``hadrons`` (chain hadrons) and ``hadrons_capped`` (their transverse momentum
        was capped).

    Returns
    -------
    HadronArrays
        Stable particles in production order per event (chain by chain, rank by rank, decay products in place of
        their parent); ``index`` counts them per event.
    """
    stats = {} if stats is None else stats
    p4, species, charge, chain, rank = _fragment(partons, cfg, rng, stats)
    stats["hadrons"] = stats.get("hadrons", 0) + len(p4)
    h = len(p4)
    vertex = np.zeros((h, 4))
    from_b = np.zeros(h, dtype=bool)
    sub = np.zeros(h, dtype=np.int64)
    pieces_p4 = [p4]
    pieces = {"vertex": [vertex], "species": [species], "charge": [charge], "chain": [chain], "rank": [rank]}
    pieces["from_b"], pieces["sub"] = [from_b], [sub]
    # B hadrons
    is_b = species == B_HADRON
    if is_b.any():
        rows = np.flatnonzero(is_b)
        d_p4, d_species, d_charge, parent, slot, b_vertex = _b_decays(p4[rows], cfg, rng)
        pieces_p4.append(d_p4)
        pieces["vertex"].append(b_vertex[parent])
        pieces["species"].append(d_species)
        pieces["charge"].append(d_charge)
        pieces["chain"].append(chain[rows][parent])
        pieces["rank"].append(rank[rows][parent])
        pieces["from_b"].append(np.ones(len(parent), dtype=bool))
        pieces["sub"].append(slot)
    all_p4 = np.concatenate(pieces_p4)
    cat = {key: np.concatenate(val) for key, val in pieces.items()}
    # pi0 -> gamma gamma
    is_pi0 = cat["species"] == PI0
    if is_pi0.any():
        rows = np.flatnonzero(is_pi0)
        g1, g2 = decay_pi0(all_p4[rows], rng.random((2, len(rows))))
        all_p4 = np.concatenate([all_p4, g1, g2])
        for key in ("vertex", "chain", "rank", "from_b"):
            cat[key] = np.concatenate([cat[key], cat[key][rows], cat[key][rows]])
        cat["species"] = np.concatenate([cat["species"], np.full(2 * len(rows), PHOTON, np.int8)])
        cat["charge"] = np.concatenate([cat["charge"], np.zeros(2 * len(rows), np.int8)])
        cat["sub"] = np.concatenate([cat["sub"], 2 * cat["sub"][rows] + 100, 2 * cat["sub"][rows] + 101])
    stable = (cat["species"] != PI0) & (cat["species"] != B_HADRON) & (cat["species"] != NEUTRINO)
    order = np.flatnonzero(stable)
    order = order[np.lexsort((cat["sub"][order], cat["rank"][order], cat["chain"][order]))]
    parton_index = partons.index_in_event()
    chain_o = cat["chain"][order]
    event = partons.event[chain_o].astype(np.int64)
    index = np.arange(len(order), dtype=np.int64) - np.searchsorted(event, event, side="left")
    return HadronArrays(
        p4=all_p4[order],
        vertex=cat["vertex"][order],
        species=cat["species"][order],
        charge=cat["charge"][order],
        event=event,
        index=index,
        prong=partons.prong[chain_o].astype(np.int64),
        parton=parton_index[chain_o],
        from_b=cat["from_b"][order],
        is_ue=np.zeros(len(order), dtype=bool),
    )
