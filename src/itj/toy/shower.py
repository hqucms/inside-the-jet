"""The parton shower: an angular-ordered Sudakov veto algorithm, the resonance reshuffle and the emission record.

Branchings ``a -> b c`` happen with the probability (``docs/physics.md#parton-shower``)

.. code-block:: text

    dP = alpha_s(kt) / (2 pi) dtheta^2 / theta^2 P(z) dz Theta(kt > kt_cut),    kt = z (1 - z) E theta

with z the energy fraction of b. Every emitter runs the veto algorithm in ``t = ln theta^2`` from its start angle
down to ``theta_min = 4 kt_cut / E``: trial angles from a constant overestimate of the rate, then a rejection with
the true splitting function and coupling. The daughters of an accepted branching start at its angle (angular
ordering). Each splitting keeps the parent's energy and direction; the final partons are massless.

:func:`shower` runs all emitters of many jets at once (one veto trial per active emitter per iteration);
:func:`shower_one` is the same algorithm for one parton, written for reading (``docs/physics.md`` prints it).
:func:`shower_events` showers the hard events of one process group and restores the four-momentum of every W and
top with :func:`reshuffle` (the Herwig recipe with the decay momenta of the hard process).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

import numpy as np

from .kinematics import boost, mass, perp_basis

if TYPE_CHECKING:
    from .config import ShowerPart, SimConfig
    from .hardprocess import HardEvents

__all__ = [
    "CA",
    "CF",
    "TR",
    "EmissionRecord",
    "PartonArrays",
    "Progenitors",
    "alphas",
    "progenitors",
    "reshuffle",
    "shower",
    "shower_events",
    "shower_one",
]

#: Colour factors.
CF, CA, TR = 4.0 / 3.0, 3.0, 0.5

#: Z mass that fixes Lambda of the running coupling (alpha_s(M_Z) = ``alphas_value``).
MZ = 91.1876

#: beta_0 of the one-loop running coupling with n_f = 5.
_B0 = (33.0 - 2.0 * 5) / (12.0 * math.pi)

#: Emission channels (``truth_emis_channel``).
CHANNELS: tuple[str, ...] = ("q -> q g", "g -> g g", "g -> q qbar")

#: Attempts to shower a resonance again when its shower systems are heavier than the resonance.
_MAX_RESHOWER = 100


def alphas(kt: np.ndarray | float, cfg: ShowerPart) -> np.ndarray:
    """Strong coupling at the scale ``kt``: one loop with n_f = 5 and alpha_s(M_Z) = ``alphas_value``, or fixed.

    ``Lambda^2 = M_Z^2 exp(-1 / (beta_0 alpha_s(M_Z)))``, Lambda = 88 MeV for 0.118 (alpha_s(1 GeV) = 0.34).
    """
    kt = np.asarray(kt, dtype=np.float64)
    if cfg.alphas_mode == "fixed":
        return np.full(kt.shape, cfg.alphas_value)
    lambda2 = MZ * MZ * math.exp(-1.0 / (_B0 * cfg.alphas_value))
    return 1.0 / (_B0 * np.log(kt * kt / lambda2))


# ----------------------------------------------------------------------------------------------------------------------
# Arrays


@dataclass
class Progenitors:
    """The partons that start the shower (generation frame).

    Attributes
    ----------
    energy : numpy.ndarray
        ``(m,)`` energy (a b quark from a top decay keeps its massive energy: it is massless in the shower).
    direction : numpy.ndarray
        ``(m, 3)`` unit vectors.
    pid : numpy.ndarray
        ``(m,)`` PDG id (+-1..5, 21).
    event : numpy.ndarray
        ``(m,)`` row of the event in its group.
    prong : numpy.ndarray
        ``(m,)`` prong index (0 for QCD).
    theta_start : numpy.ndarray
        ``(m,)`` start angle of the veto algorithm.
    hard_b : numpy.ndarray
        ``(m,)`` bool, a b quark of the hard process (gets the dead cone and a B hadron).
    """

    energy: np.ndarray
    direction: np.ndarray
    pid: np.ndarray
    event: np.ndarray
    prong: np.ndarray
    theta_start: np.ndarray
    hard_b: np.ndarray

    def select(self, rows: np.ndarray) -> Progenitors:
        """Return the rows ``rows``."""
        return Progenitors(**{f.name: getattr(self, f.name)[rows] for f in fields(self)})


@dataclass
class PartonArrays:
    """Final partons of a group of events, sorted by (event, node).

    Attributes
    ----------
    p4 : numpy.ndarray
        ``(m, 4)`` massless four-vectors (generation frame until moved to the lab).
    pid, event, prong, node : numpy.ndarray
        ``(m,)`` PDG id, event row, prong and shower node id (progenitor i has node i).
    hard_b : numpy.ndarray
        ``(m,)`` bool, on the quark line of a b quark of the hard process.
    """

    p4: np.ndarray
    pid: np.ndarray
    event: np.ndarray
    prong: np.ndarray
    node: np.ndarray
    hard_b: np.ndarray

    def __len__(self) -> int:
        return len(self.event)

    def select(self, rows: np.ndarray) -> PartonArrays:
        """Return the rows ``rows``."""
        return PartonArrays(**{f.name: getattr(self, f.name)[rows] for f in fields(self)})

    @staticmethod
    def concat(parts: list[PartonArrays]) -> PartonArrays:
        """Concatenate row-wise."""
        return PartonArrays(
            **{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(PartonArrays)}
        )

    def index_in_event(self) -> np.ndarray:
        """Position of every parton among the partons of its event (the rows must be sorted by event)."""
        return _rank_in_group(self.event)


@dataclass
class EmissionRecord:
    """Every accepted branching of a group of events, as generated (before the resonance reshuffle).

    Attributes
    ----------
    event, parent, daughter1, daughter2 : numpy.ndarray
        ``(e,)`` event row and shower node ids (daughter1 carries the energy fraction z).
    energy, z, theta, phi, kt : numpy.ndarray
        ``(e,)`` parent energy, z, opening angle, azimuth and ``z (1 - z) E theta``.
    channel, prong, primary, depth : numpy.ndarray
        ``(e,)`` 0 q -> qg, 1 g -> gg, 2 g -> qq; prong; on the harder (energy) chain of its prong; branchings
        above it.
    seq : numpy.ndarray
        ``(e,)`` generation order (the record is sorted by (event, seq)).
    """

    event: np.ndarray
    parent: np.ndarray
    daughter1: np.ndarray
    daughter2: np.ndarray
    energy: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    kt: np.ndarray
    channel: np.ndarray
    prong: np.ndarray
    primary: np.ndarray
    depth: np.ndarray
    seq: np.ndarray

    def __len__(self) -> int:
        return len(self.event)

    def select(self, rows: np.ndarray) -> EmissionRecord:
        """Return the rows ``rows``."""
        return EmissionRecord(**{f.name: getattr(self, f.name)[rows] for f in fields(self)})

    @staticmethod
    def concat(parts: list[EmissionRecord]) -> EmissionRecord:
        """Concatenate row-wise."""
        return EmissionRecord(
            **{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(EmissionRecord)}
        )

    @staticmethod
    def empty() -> EmissionRecord:
        """Return a record without emissions."""
        i, f = np.zeros(0, dtype=np.int64), np.zeros(0)
        return EmissionRecord(i, i, i, i, f, f, f, f, f, i, i, np.zeros(0, dtype=bool), i, i)


def _rank_in_group(group: np.ndarray) -> np.ndarray:
    """Rank of every element among the elements with the same (sorted) group value."""
    if len(group) == 0:
        return np.zeros(0, dtype=np.int64)
    start = np.searchsorted(group, group, side="left")
    return np.arange(len(group), dtype=np.int64) - start


def _new_nodes(event: np.ndarray, next_node: np.ndarray) -> np.ndarray:
    """Allocate two consecutive node ids per accepted branching; returns the first id of each pair."""
    order = np.argsort(event, kind="stable")
    rank = np.empty(len(event), dtype=np.int64)
    rank[order] = _rank_in_group(event[order])
    first = next_node[event] + 2 * rank
    np.add.at(next_node, event, 2)
    return first


# ----------------------------------------------------------------------------------------------------------------------
# The vectorised shower


def _daughter_directions(
    n: np.ndarray, z: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Directions of b (energy fraction z) and c at opening angle theta, azimuth phi about the parent ``n``.

    ``tan theta_b = (1 - z) sin theta / (z + (1 - z) cos theta)``, ``theta_c = theta - theta_b``: the transverse
    momenta balance, so the parent's energy and direction are kept.
    """
    theta_b = np.arctan2((1.0 - z) * np.sin(theta), z + (1.0 - z) * np.cos(theta))
    theta_c = theta - theta_b
    e1, e2 = perp_basis(n)
    u = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
    nb = np.cos(theta_b)[:, None] * n + np.sin(theta_b)[:, None] * u
    nc = np.cos(theta_c)[:, None] * n - np.sin(theta_c)[:, None] * u
    nb /= np.linalg.norm(nb, axis=1, keepdims=True)
    nc /= np.linalg.norm(nc, axis=1, keepdims=True)
    return nb, nc


def shower(
    prog: Progenitors, cfg: ShowerPart, rng: np.random.Generator, *, n_events: int, b_mass: float = 4.8
) -> tuple[PartonArrays, EmissionRecord]:
    """Shower many partons at once with the veto algorithm (one trial per active emitter per iteration).

    Parameters
    ----------
    prog : Progenitors
        The partons that start the shower; progenitor j of an event gets node id ``prong``.
    cfg : ShowerPart
        Shower parameters.
    rng : numpy.random.Generator
        The shower stream.
    n_events : int
        Number of events of the group (node ids are counted per event).
    b_mass : float
        b-quark mass of the dead cone.

    Returns
    -------
    tuple of PartonArrays and EmissionRecord
        Final partons (sorted by event and node) and the accepted branchings (sorted by event and generation
        order).
    """
    kt_cut = cfg.kt_cut
    amax = float(alphas(kt_cut, cfg))
    nf = cfg.n_flavours if cfg.g_to_qq else 0
    next_node = np.zeros(n_events, dtype=np.int64)
    np.maximum.at(next_node, prog.event, prog.prong.astype(np.int64) + 1)
    # state of the active emitters
    E = prog.energy.astype(np.float64)
    n = prog.direction.astype(np.float64)
    pid = prog.pid.astype(np.int64)
    ev = prog.event.astype(np.int64)
    prong = prog.prong.astype(np.int64)
    th0 = prog.theta_start.astype(np.float64)
    lt = 2.0 * np.log(th0)
    node = prog.prong.astype(np.int64)
    prim = np.ones(len(E), dtype=bool)
    depth = np.zeros(len(E), dtype=np.int64)
    hard_b = prog.hard_b.astype(bool)
    finals: list[tuple[np.ndarray, ...]] = []
    emissions: list[tuple[np.ndarray, ...]] = []
    seq = 0
    while len(E):
        k = len(E)
        gluon = pid == 21
        th_min = 4.0 * kt_cut / E
        dead = th0 <= th_min  # the guard: no phase space left (then eps0 >= 1/4)
        eps0 = np.where(dead, 0.25, kt_cut / (E * np.where(dead, 1.0, th0)))
        log_ratio = np.log((1.0 - eps0) / eps0)
        i_q = 2.0 * CF * log_ratio
        i_gg = 2.0 * CA * log_ratio
        i_qq = TR * nf * (1.0 - 2.0 * eps0)
        i_tot = np.where(gluon, i_gg + i_qq, i_q)
        rate = amax / (2.0 * math.pi) * np.where(dead, 1.0, i_tot)
        u = rng.random((6, k))
        lt_new = lt + np.log(u[0]) / rate
        theta = np.exp(0.5 * lt_new)
        fin = dead | (theta < th_min)
        # channel and z from the overestimates
        is_gg = gluon & (u[1] * i_tot < i_gg)
        is_qq = gluon & ~is_gg
        z_q = 1.0 - eps0 * np.exp(u[2] * log_ratio)  # 1 - z log-uniform
        z_gg = 1.0 / (1.0 + np.exp(-log_ratio * (2.0 * u[2] - 1.0)))  # logit(z) uniform
        z_qq = eps0 + (1.0 - 2.0 * eps0) * u[2]
        z = np.where(is_gg, z_gg, np.where(is_qq, z_qq, z_q))
        zz = z * (1.0 - z)
        ratio = np.where(is_gg, 1.0 + (zz - 2.0) * zz, np.where(is_qq, z * z + (1.0 - z) ** 2, 0.5 * (1.0 + z * z)))
        kt = zz * E * theta
        ok = ~fin & (kt >= kt_cut)
        if cfg.dead_cone:
            ok &= ~(hard_b & (theta < b_mass / E))
        ratio = ratio * alphas(np.maximum(kt, kt_cut), cfg) / amax
        acc = ok & (u[3] < ratio)
        if fin.any():
            finals.append((E[fin], n[fin], pid[fin], ev[fin], prong[fin], node[fin], hard_b[fin]))
        cont = ~fin & ~acc
        if acc.any():
            a = np.flatnonzero(acc)
            ea, za, tha = E[a], z[a], theta[a]
            phi = 2.0 * math.pi * u[4, a]
            nb, nc = _daughter_directions(n[a], za, tha, phi)
            channel = np.where(is_gg[a], 1, np.where(is_qq[a], 2, 0))
            flavour = 1 + np.minimum((u[5, a] * max(nf, 1)).astype(np.int64), max(nf, 1) - 1)
            pa = pid[a]
            pid_b = np.where(channel == 0, pa, np.where(channel == 1, 21, flavour))
            pid_c = np.where(channel == 2, -flavour, 21)
            hb = hard_b[a] & (channel == 0)
            d1 = _new_nodes(ev[a], next_node)
            harder_b = za >= 0.5
            emissions.append(
                (
                    ev[a], node[a], d1, d1 + 1, ea, za, tha, phi, zz[a] * ea * tha, channel, prong[a], prim[a],
                    depth[a], seq + np.arange(len(a)),
                )
            )  # fmt: skip
            seq += len(a)
            log_th2 = np.log(tha * tha)
            E = np.concatenate([E[cont], za * ea, (1.0 - za) * ea])
            n = np.concatenate([n[cont], nb, nc])
            pid = np.concatenate([pid[cont], pid_b, pid_c])
            ev = np.concatenate([ev[cont], ev[a], ev[a]])
            prong = np.concatenate([prong[cont], prong[a], prong[a]])
            th0 = np.concatenate([th0[cont], tha, tha])
            lt = np.concatenate([lt_new[cont], log_th2, log_th2])
            node = np.concatenate([node[cont], d1, d1 + 1])
            prim = np.concatenate([prim[cont], prim[a] & harder_b, prim[a] & ~harder_b])
            depth = np.concatenate([depth[cont], depth[a] + 1, depth[a] + 1])
            hard_b = np.concatenate([hard_b[cont], hb, np.zeros(len(a), dtype=bool)])
        else:
            E, n, pid, ev, prong, th0, lt = E[cont], n[cont], pid[cont], ev[cont], prong[cont], th0[cont], lt_new[cont]
            node, prim, depth, hard_b = node[cont], prim[cont], depth[cont], hard_b[cont]
    partons = _finals_to_arrays(finals)
    record = _emissions_to_record(emissions)
    return partons, record


def _finals_to_arrays(finals: list[tuple[np.ndarray, ...]]) -> PartonArrays:
    if not finals:
        z = np.zeros(0, dtype=np.int64)
        return PartonArrays(np.zeros((0, 4)), z, z, z, z, np.zeros(0, dtype=bool))
    e, n, pid, ev, prong, node, hard_b = (np.concatenate(x) for x in zip(*finals, strict=True))
    p4 = np.concatenate([e[:, None] * n, e[:, None]], axis=1)
    order = np.lexsort((node, ev))
    return PartonArrays(p4, pid, ev, prong, node, hard_b).select(order)


def _emissions_to_record(emissions: list[tuple[np.ndarray, ...]]) -> EmissionRecord:
    if not emissions:
        return EmissionRecord.empty()
    cols = [np.concatenate(x) for x in zip(*emissions, strict=True)]
    record = EmissionRecord(*cols)
    return record.select(np.lexsort((record.seq, record.event)))


# ----------------------------------------------------------------------------------------------------------------------
# Reshuffle


def reshuffle(q: np.ndarray, p4: np.ndarray, system: np.ndarray, mass_res: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give each shower system of a resonance its decay momentum, rescaled so that four-momentum is conserved.

    The Herwig recipe with the decay momenta of the hard process (``docs/physics.md#resonance-reshuffle``), in the
    resonance rest frame:

    1. take the decay momenta ``q_i`` and re-centre them, ``q_i <- q_i - sum q / n``;
    2. ``m_i`` = invariant mass of shower system i;
    3. solve ``sum_i sqrt(k^2 |q_i|^2 + m_i^2) = M`` for k (Newton from above: the function is convex);
    4. move each system to ``(sqrt(k^2 |q_i|^2 + m_i^2), k q_i)``: boost to its rest frame, then by
       ``k q_i / E_i'``; a single massless parton is set directly to ``(k q_i, k |q_i|)``.

    Parameters
    ----------
    q : numpy.ndarray
        ``(r, s, 3)`` decay momenta of the ``s`` systems of ``r`` resonances in their rest frames.
    p4 : numpy.ndarray
        ``(m, 4)`` partons in the rest frame of their resonance.
    system : numpy.ndarray
        ``(m,)`` int, ``resonance * s + system index`` of every parton.
    mass_res : numpy.ndarray
        ``(r,)`` resonance masses.

    Returns
    -------
    tuple of numpy.ndarray
        The moved partons ``(m, 4)`` (unchanged for failed resonances) and ``ok`` ``(r,)``: False where the
        systems are heavier than the resonance (``sum m_i >= M``; shower again).
    """
    r, s, _ = q.shape
    q = q - q.sum(axis=1, keepdims=True) / s
    total = np.zeros((r * s, 4))
    np.add.at(total, system, p4)
    count = np.bincount(system, minlength=r * s)
    m_sys = np.where(count > 1, mass(total), 0.0).reshape(r, s)
    qn = np.linalg.norm(q, axis=2)
    big_m = np.asarray(mass_res, dtype=np.float64)
    ok = m_sys.sum(axis=1) < big_m
    # Newton from k_hi = M / sum |q| (where f >= 0) converges monotonically for the convex increasing f
    k = big_m / np.maximum(qn.sum(axis=1), 1e-300)
    for _ in range(100):
        e = np.sqrt((k[:, None] * qn) ** 2 + m_sys**2)
        f = e.sum(axis=1) - big_m
        df = (k[:, None] * qn * qn / np.maximum(e, 1e-300)).sum(axis=1)
        step = np.where(ok, f / np.maximum(df, 1e-300), 0.0)
        k = k - step
        if np.all(np.abs(step) <= 1e-15 * np.abs(k)):
            break
    target = np.concatenate([k[:, None, None] * q, np.sqrt((k[:, None] * qn) ** 2 + m_sys**2)[..., None]], axis=2)
    target = target.reshape(r * s, 4)
    beta_rest = -total[:, :3] / np.maximum(total[:, 3:4], 1e-300)
    m_flat = m_sys.reshape(-1)
    beta_new = target[:, :3] / np.maximum(target[:, 3:4], 1e-300)
    moved = boost(boost(p4, beta_rest[system]), beta_new[system])
    single = (count[system] == 1) & (m_flat[system] == 0.0)
    moved[single] = target[system[single]]
    keep = ~ok.repeat(s)[system]
    moved[keep] = p4[keep]
    return moved, ok


# ----------------------------------------------------------------------------------------------------------------------
# Hard events -> final partons


def progenitors(hard: HardEvents, cfg: SimConfig) -> Progenitors:
    """Return the decay partons (or the QCD initiator) of the hard events as shower progenitors."""
    nprongs = hard.nprongs.astype(np.int64)
    ev, prong = np.nonzero(np.arange(3)[None, :] < nprongs[:, None])
    p = hard.prong_p4[ev, prong]
    direction = p[:, :3] / np.linalg.norm(p[:, :3], axis=1, keepdims=True)
    pid = hard.prong_pid[ev, prong].astype(np.int64)
    return Progenitors(
        energy=p[:, 3].copy(),
        direction=direction,
        pid=pid,
        event=ev.astype(np.int64),
        prong=prong.astype(np.int64),
        theta_start=hard.theta_start[ev, prong].astype(np.float64),
        hard_b=np.abs(pid) == 5,
    )


def _unshowered(prog: Progenitors) -> PartonArrays:
    p4 = np.concatenate([prog.energy[:, None] * prog.direction, prog.energy[:, None]], axis=1)
    order = np.lexsort((prog.prong, prog.event))
    return PartonArrays(p4, prog.pid, prog.event, prog.prong, prog.prong.copy(), prog.hard_b).select(order)


def _reshuffle_events(hard: HardEvents, partons: PartonArrays, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reshuffle the resonance events ``events``; returns the new four-vectors of their partons and ``ok``."""
    rows = np.flatnonzero(np.isin(partons.event, events))
    slot = np.searchsorted(events, partons.event[rows])
    s = int(hard.nprongs[events].max()) if len(events) else 0
    res = hard.res_p4[events]
    beta = res[:, :3] / res[:, 3:4]
    rest = boost(partons.p4[rows], -beta[slot])
    moved, ok = reshuffle(hard.rest_q[events, :s], rest, slot * s + partons.prong[rows], hard.mass[events])
    return boost(moved, beta[slot]), ok


def shower_events(hard: HardEvents, cfg: SimConfig, rng: np.random.Generator) -> tuple[PartonArrays, EmissionRecord]:
    """Shower the hard events of a group and reshuffle every W and top (generation frame).

    Without the shower, the decay partons are the final partons (the b quark becomes massless in the reshuffle).
    A resonance whose shower systems are heavier than itself is showered again with the next draws of the shower
    stream.
    """
    prog = progenitors(hard, cfg)
    n = len(hard)
    if cfg.shower.shower:
        partons, record = shower(prog, cfg.shower, rng, n_events=n, b_mass=cfg.hard.b_mass)
    else:
        partons, record = _unshowered(prog), EmissionRecord.empty()
    todo = np.flatnonzero(hard.nprongs >= 2)
    for _ in range(_MAX_RESHOWER):
        if len(todo) == 0:
            break
        rows = np.flatnonzero(np.isin(partons.event, todo))
        moved, ok = _reshuffle_events(hard, partons, todo)
        partons.p4[rows] = moved
        failed = todo[~ok]
        if len(failed) == 0:
            break
        again = prog.select(np.flatnonzero(np.isin(prog.event, failed)))
        new_partons, new_record = shower(again, cfg.shower, rng, n_events=n, b_mass=cfg.hard.b_mass)
        keep_p = ~np.isin(partons.event, failed)
        keep_e = ~np.isin(record.event, failed)
        new_record.seq = new_record.seq + (int(record.seq.max()) + 1 if len(record) else 0)
        partons = PartonArrays.concat([partons.select(np.flatnonzero(keep_p)), new_partons])
        partons = partons.select(np.lexsort((partons.node, partons.event)))
        record = EmissionRecord.concat([record.select(np.flatnonzero(keep_e)), new_record])
        record = record.select(np.lexsort((record.seq, record.event)))
        todo = failed
    else:  # pragma: no cover - needs an absurd configuration
        raise RuntimeError(f"{len(todo)} resonances could not be showered below their mass in {_MAX_RESHOWER} attempts")
    if cfg.shower.shower:
        _align_qcd(hard, partons)
    return partons, record


def _align_qcd(hard: HardEvents, partons: PartonArrays) -> None:
    """Rotate the partons of every QCD event so that their summed 3-momentum points along the seed (in place).

    The shower keeps every parent's energy and direction, but a parton that branches later becomes virtual and its
    3-momentum shrinks, so the summed 3-momentum of a QCD jet tilts away from the seed by a few per cent. A rotation
    of the whole jet (the recoil taken by the jet as a whole) keeps the energy, every mass and every opening angle
    and puts the summed 3-momentum back on the seed axis; its length stays below the seed energy.
    """
    qcd = hard.nprongs[partons.event] == 1
    if not qcd.any():
        return
    rows = np.flatnonzero(qcd)
    ev = partons.event[rows]
    total = np.zeros((len(hard), 3))
    np.add.at(total, ev, partons.p4[rows, :3])
    seed = hard.prong_p4[:, 0, :3]
    a = total / np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-300)
    b = seed / np.maximum(np.linalg.norm(seed, axis=1, keepdims=True), 1e-300)
    axis = np.cross(a, b)
    sin = np.linalg.norm(axis, axis=1)
    cos = np.sum(a * b, axis=1)
    k = axis / np.maximum(sin, 1e-300)[:, None]
    # Rodrigues: v' = v cos + (k x v) sin + k (k . v)(1 - cos)
    v = partons.p4[rows, :3]
    ke, s, c = k[ev], sin[ev, None], cos[ev, None]
    rotated = v * c + np.cross(ke, v) * s + ke * np.sum(ke * v, axis=1, keepdims=True) * (1.0 - c)
    partons.p4[rows, :3] = rotated


# ----------------------------------------------------------------------------------------------------------------------
# The readable scalar version


def shower_one(
    energy: float, flavour: Literal["q", "g"], theta_start: float, config: ShowerPart, rng: np.random.Generator
) -> list[dict[str, float]]:
    """Shower one parton with the veto algorithm, written for reading.

    The vectorised :func:`shower` runs the same algorithm and is tested against this one.

    Parameters
    ----------
    energy : float
        Energy of the parton in GeV.
    flavour : {"q", "g"}
        Quark or gluon.
    theta_start : float
        Start angle (the largest allowed opening angle).
    config : ShowerPart
        Shower parameters (coupling, cutoff, g -> q qbar).
    rng : numpy.random.Generator
        Random numbers.

    Returns
    -------
    list of dict
        One dict per accepted branching, in generation order (depth first): ``theta``, ``z``, ``kt``, ``phi``,
        ``channel`` (0 q -> qg, 1 g -> gg, 2 g -> qq), ``depth``, ``primary`` (1.0 on the harder chain) and
        ``energy`` (of the parent).

    Examples
    --------
    >>> from itj.toy.config import ShowerPart
    >>> emissions = shower_one(600.0, "q", 1.0, ShowerPart(), np.random.default_rng(1))
    >>> all(e["kt"] >= 1.0 for e in emissions)
    True
    """
    kt_cut = config.kt_cut
    alpha_max = float(alphas(kt_cut, config))
    nf = config.n_flavours if config.g_to_qq else 0
    emissions: list[dict[str, float]] = []
    # each stack entry: energy, is_gluon, start angle, depth, primary
    stack = [(float(energy), flavour == "g", float(theta_start), 0, True)]
    while stack:
        e, is_gluon, theta0, depth, primary = stack.pop()
        theta_min = 4.0 * kt_cut / e
        if theta0 <= theta_min:  # no phase space left: the parton is final
            continue
        eps0 = kt_cut / (e * theta0)  # widest z range, valid for every theta < theta0
        log_ratio = math.log((1.0 - eps0) / eps0)
        integrals = [2.0 * CA * log_ratio, TR * nf * (1.0 - 2.0 * eps0)] if is_gluon else [2.0 * CF * log_ratio]
        rate = alpha_max / (2.0 * math.pi) * sum(integrals)
        log_theta2 = 2.0 * math.log(theta0)
        while True:
            log_theta2 += math.log(rng.random()) / rate  # next trial angle
            theta = math.exp(0.5 * log_theta2)
            if theta < theta_min:
                break  # final parton
            channel = 0  # q -> q g
            if is_gluon:  # g -> g g or g -> q qbar, in proportion to their integrals
                channel = 1 if rng.random() * sum(integrals) < integrals[0] else 2
            r = rng.random()
            if channel == 0:
                z = 1.0 - eps0 * math.exp(r * log_ratio)  # 1 - z log-uniform
                accept = 0.5 * (1.0 + z * z)  # P_qq / (2 C_F / (1 - z))
            elif channel == 1:
                z = 1.0 / (1.0 + math.exp(-log_ratio * (2.0 * r - 1.0)))  # logit(z) uniform
                accept = 1.0 + (z * (1.0 - z) - 2.0) * z * (1.0 - z)  # P_gg / (C_A / (z (1 - z)))
            else:
                z = eps0 + (1.0 - 2.0 * eps0) * r  # uniform
                accept = z * z + (1.0 - z) ** 2  # P_qg / (T_R n_f)
            kt = z * (1.0 - z) * e * theta
            if kt < kt_cut:
                continue  # outside the true phase space: veto
            if rng.random() >= accept * float(alphas(kt, config)) / alpha_max:
                continue  # vetoed by the true splitting function and coupling
            phi = 2.0 * math.pi * rng.random()
            emissions.append(
                {
                    "theta": theta,
                    "z": z,
                    "kt": kt,
                    "phi": phi,
                    "channel": float(channel),
                    "depth": float(depth),
                    "primary": float(primary),
                    "energy": e,
                }
            )
            gluon_b = channel == 1
            gluon_c = channel != 2
            # the daughters start at the angle of the branching (angular ordering); c first so b is showered first
            stack.append(((1.0 - z) * e, gluon_c, theta, depth + 1, primary and z < 0.5))
            stack.append((z * e, gluon_b, theta, depth + 1, primary and z >= 0.5))
            break
    return emissions
