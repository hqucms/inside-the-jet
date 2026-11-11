"""Generate jets: the chunk pipeline, worker processes, jet building, the truth record and the provenance.

Jets are made in chunks of :data:`CHUNK_SIZE` = 1000. Chunk k draws its random numbers from
``SeedSequence(seed, spawn_key=(k,)).spawn(4)``, the hard, shower, hadron and ue streams; the hard stream of the
chunk draws the order of the processes (balanced within the chunk), and each process group of the chunk uses the
children ``(k, stage, process code)`` of the four streams, so that a per-process override changes the jets of that
process only. The detector randomness is counter based, keyed by the seed and ``truth_event_id``
(:mod:`itj.toy.detector`). Consequences (``docs/physics.md#reproducibility``):

* the output does not depend on ``workers``; the first k * 1000 jets of a file are the same for every
  ``n_jets >= k * 1000``;
* two runs with the same seed that differ only in the preset or in switches acting after the hard process hold the
  same hard process jet by jet (and the same shower when only hadron, UE or detector settings differ);
* ``ue=False`` and ``cell_position="centre"`` leave every jet-origin track bit-identical;
* a change of the hard process (a mass, the spectrum, the quark fraction, the process list) gives a statistically
  independent sample.

The pipeline of one process group (``docs/physics.md#building-the-jet``): hard process, shower and reshuffle,
hadronisation, underlying event (generation frame: the seed along +x at y = 0), move to the lab, detector, then the
leading anti-kt jet (or the cone about the seed axis) with its constituents sorted by pT, the JetClass branches of
:func:`itj.observables.jet_branches` and the truth record.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import difflib
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .. import __version__
from ..data import COLUMNS, JETCLASS_LABELS, JetData, Ragged
from .config import PRESETS, SimConfig
from .detector import ParticleArrays, detector
from .hadronize import HadronArrays, hadronize
from .hardprocess import PROCESSES, HardEvents, sample_hard
from .kinematics import pt_eta_phi, to_lab, wrap_phi
from .shower import EmissionRecord, PartonArrays, shower_events
from .ue import underlying_event

__all__ = ["CHUNK_SIZE", "STAGES", "balanced_counts", "build_jets", "generate", "nearest_preset", "show_ladder"]

#: Jets per generation chunk (and per Parquet row group).
CHUNK_SIZE = 1000

#: The stages whose wall time is summed in ``provenance["stage_times_s"]``.
STAGES: tuple[str, ...] = ("hard", "shower", "hadronize", "ue", "detector", "jet_building", "observables", "write")

#: Label column and ``aux_truth_match`` of every process.
_LABEL = {"qcd": "label_QCD", "wqq": "label_Wqq", "tbqq": "label_Tbqq", "q": "label_QCD", "g": "label_QCD"}
_TRUTH_MATCH = {"qcd": 50.0, "wqq": 21.0, "tbqq": 11.0, "q": 50.0, "g": 50.0}

#: Switches of the physics ladder (the fields in which the presets differ).
_LADDER = ("shower", "hadronize", "b_hadrons", "ue", "detector")


# ----------------------------------------------------------------------------------------------------------------------
# Jet building


def _ragged(group_offsets: np.ndarray, values: np.ndarray, name: str) -> Ragged:
    return Ragged(np.asarray(values, dtype=COLUMNS[name].dtype), group_offsets)


def _offsets(counts: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


def _padded(p4: np.ndarray, event: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad particles sorted by event to ``(n, N, 4)``; returns the array, the mask and each particle's slot."""
    counts = np.bincount(event, minlength=n)
    slot = np.arange(len(event)) - np.repeat(_offsets(counts)[:-1], counts)
    width = max(int(counts.max()) if n else 0, 1)
    out = np.zeros((n, width, 4))
    out[event, slot] = p4
    mask = np.zeros((n, width), dtype=bool)
    mask[event, slot] = True
    return out, mask, slot


def _relative(eta: np.ndarray, phi: np.ndarray, jet_eta: np.ndarray, jet_phi: np.ndarray) -> tuple[np.ndarray, ...]:
    """``part_deta = s (eta - eta_jet)`` with s = +1 if eta_jet >= 0 else -1, and the wrapped Delta phi."""
    sign = np.where(jet_eta >= 0, 1.0, -1.0)
    return sign * (eta - jet_eta), wrap_phi(phi - jet_phi)


def build_jets(
    code: int,
    hard: HardEvents,
    partons: PartonArrays,
    record: EmissionRecord,
    particles: ParticleArrays,
    cfg: SimConfig,
    event_ids: np.ndarray,
    times: dict[str, float] | None = None,
) -> tuple[JetData, np.ndarray]:
    """Find the jet of every event of a group and write its JetClass branches and truth record.

    Parameters
    ----------
    code : int
        Process code of the group.
    hard : HardEvents
        The hard events (generation frame, with the lab seed rapidity and azimuth).
    partons : PartonArrays
        Final partons in the lab, sorted by (event, node).
    record : EmissionRecord
        The accepted branchings, sorted by (event, generation order).
    particles : ParticleArrays
        Final particles in the lab.
    cfg : SimConfig
        The configuration (jet part).
    event_ids : numpy.ndarray
        ``truth_event_id`` of every event.
    times : dict, optional
        Wall times in seconds, updated in place (``jet_building``, ``observables``).

    Returns
    -------
    tuple of JetData and numpy.ndarray
        The jets (every JetClass and truth column) and the bool mask of the events that pass the optional reco
        pT window (all True without one).
    """
    from ..cluster import cone, leading_jet
    from ..observables import jet_branches

    times = {} if times is None else times
    t0 = time.perf_counter()
    n = len(hard)
    j = cfg.jet
    order = np.argsort(particles.event, kind="stable")
    particles = particles.select(order)
    # the stored constituents are float32: compute everything from the rounded four-vectors
    p4_all = particles.p4.astype(np.float32).astype(np.float64)
    padded, mask, slot = _padded(p4_all, particles.event, n)
    if j.jet_def == "antikt":
        member = leading_jet(padded, mask, algorithm="antikt", R=j.jet_radius)
    else:
        member = cone(padded, mask, np.stack([hard.y, hard.phi], axis=1), R=j.jet_radius)
    keep = np.flatnonzero(member[particles.event, slot])
    pt_all = np.hypot(p4_all[:, 0], p4_all[:, 1])
    keep = keep[np.lexsort((-pt_all[keep], particles.event[keep]))]
    part = particles.select(keep)
    p4 = p4_all[keep]
    jet_padded, jet_mask, _ = _padded(p4, part.event, n)
    t1 = time.perf_counter()
    branches = jet_branches(
        jet_padded,
        jet_mask,
        zcut=j.sd_zcut,
        sd_beta=j.sd_beta,
        R0=j.tau_r0,
        tau_beta=j.tau_beta,
        tau_axes=j.tau_axes,  # type: ignore[arg-type]
    )
    t2 = time.perf_counter()
    total = jet_padded.sum(axis=1)
    _, jet_eta, jet_phi = pt_eta_phi(total)
    counts = np.bincount(part.event, minlength=n)
    offsets = _offsets(counts)
    _, eta, phi = pt_eta_phi(p4)
    deta, dphi = _relative(eta, phi, jet_eta[part.event], jet_phi[part.event])
    cols: dict[str, np.ndarray | Ragged] = {}
    for name, values in zip(("part_px", "part_py", "part_pz", "part_energy"), p4.T, strict=True):
        cols[name] = _ragged(offsets, values, name)
    cols["part_deta"] = _ragged(offsets, deta, "part_deta")
    cols["part_dphi"] = _ragged(offsets, dphi, "part_dphi")
    for name, values in (
        ("part_d0val", part.d0),
        ("part_d0err", part.d0err),
        ("part_dzval", part.dz),
        ("part_dzerr", part.dzerr),
        ("part_charge", part.charge),
    ):
        cols[name] = _ragged(offsets, values, name)
    flag_names = ("part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton", "part_isElectron", "part_isMuon")
    for k, name in enumerate(flag_names):
        cols[name] = _ragged(offsets, part.flags[:, k], name)
    process = PROCESSES[code]
    for name in JETCLASS_LABELS:
        value = 1 if name == _LABEL[process] else 0
        cols[name] = np.full(n, value, dtype=COLUMNS[name].dtype)
    cols.update(branches)
    res_lab = to_lab(hard.res_p4, hard.y, hard.phi)
    res_pt, res_eta, res_phi = pt_eta_phi(res_lab)
    resonance = code in (1, 2)
    cols["aux_genpart_eta"] = np.where(resonance, res_eta, 0.0).astype(np.float32)
    cols["aux_genpart_phi"] = np.where(resonance, res_phi, 0.0).astype(np.float32)
    cols["aux_genpart_pid"] = np.where(resonance, hard.res_pid, 0).astype(np.float32)
    cols["aux_genpart_pt"] = np.where(resonance, res_pt, 0.0).astype(np.float32)
    cols["aux_truth_match"] = np.full(n, _TRUTH_MATCH[process], dtype=np.float32)
    cols.update(_truth(code, hard, partons, record, part, offsets, cfg, event_ids, res_lab, jet_eta, jet_phi))
    passed = np.ones(n, dtype=bool)
    if j.jet_pt_min is not None:
        passed &= branches["jet_pt"] >= j.jet_pt_min
    if j.jet_pt_max is not None:
        passed &= branches["jet_pt"] <= j.jet_pt_max
    times["jet_building"] = times.get("jet_building", 0.0) + (t1 - t0) + (time.perf_counter() - t2)
    times["observables"] = times.get("observables", 0.0) + (t2 - t1)
    return JetData(cols), passed


def _truth(
    code: int,
    hard: HardEvents,
    partons: PartonArrays,
    record: EmissionRecord,
    part: ParticleArrays,
    part_offsets: np.ndarray,
    cfg: SimConfig,
    event_ids: np.ndarray,
    res_lab: np.ndarray,
    jet_eta: np.ndarray,
    jet_phi: np.ndarray,
) -> dict[str, np.ndarray | Ragged]:
    """Build the ``truth_*`` columns of a group."""
    n = len(hard)
    cols: dict[str, np.ndarray | Ragged] = {}
    seed_pt, seed_eta, seed_phi = pt_eta_phi(res_lab)
    lund_kt = np.minimum(record.z, 1.0 - record.z) * record.energy * record.theta
    primary = record.primary.astype(bool)
    n_primary = np.bincount(record.event[primary], minlength=n)
    ln_kt = np.log(np.maximum(lund_kt, 1e-300))
    lnkt_max = np.full(n, -np.inf)
    np.maximum.at(lnkt_max, record.event[primary], ln_kt[primary])
    lnkt_max[n_primary == 0] = np.nan
    per_jet = {
        "truth_event_id": event_ids,
        "truth_process": np.full(n, code),
        "truth_init_pid": hard.init_pid,
        "truth_nprongs": hard.nprongs,
        "truth_seed_pt": seed_pt,
        "truth_seed_eta": seed_eta,
        "truth_seed_phi": seed_phi,
        "truth_mass": hard.mass,
        "truth_cos_theta_star": hard.cos_theta_star,
        "truth_dr_min": hard.dr_min,
        "truth_dr_max": hard.dr_max,
        "truth_n_emissions": np.bincount(record.event, minlength=n),
        "truth_n_primary": n_primary,
        "truth_lnkt_max": lnkt_max,
    }
    for name, values in per_jet.items():
        cols[name] = np.asarray(values).astype(COLUMNS[name].dtype)
    # prongs
    nprongs = hard.nprongs.astype(np.int64)
    ev, k = np.nonzero(np.arange(3)[None, :] < nprongs[:, None])
    prong_lab = to_lab(hard.prong_p4[ev, k], hard.y[ev], hard.phi[ev])
    _, p_eta, p_phi = pt_eta_phi(prong_lab)
    p_deta, p_dphi = _relative(p_eta, p_phi, jet_eta[ev], jet_phi[ev])
    off = _offsets(nprongs)
    for name, values in zip(
        ("truth_prong_px", "truth_prong_py", "truth_prong_pz", "truth_prong_energy"), prong_lab.T, strict=True
    ):
        cols[name] = _ragged(off, values, name)
    cols["truth_prong_deta"] = _ragged(off, p_deta, "truth_prong_deta")
    cols["truth_prong_dphi"] = _ragged(off, p_dphi, "truth_prong_dphi")
    cols["truth_prong_pid"] = _ragged(off, hard.prong_pid[ev, k], "truth_prong_pid")
    # particles
    cols["truth_part_prong"] = _ragged(part_offsets, part.prong, "truth_part_prong")
    cols["truth_part_parton"] = _ragged(part_offsets, part.parton, "truth_part_parton")
    cols["truth_part_is_ue"] = _ragged(part_offsets, part.is_ue, "truth_part_is_ue")
    cols["truth_part_from_b"] = _ragged(part_offsets, part.from_b, "truth_part_from_b")
    cols["truth_part_purity"] = _ragged(part_offsets, part.purity, "truth_part_purity")
    # partons
    off = _offsets(np.bincount(partons.event, minlength=n))
    for name, values in zip(
        ("truth_parton_px", "truth_parton_py", "truth_parton_pz", "truth_parton_energy"), partons.p4.T, strict=True
    ):
        cols[name] = _ragged(off, values, name)
    cols["truth_parton_pid"] = _ragged(off, partons.pid, "truth_parton_pid")
    cols["truth_parton_prong"] = _ragged(off, partons.prong, "truth_parton_prong")
    cols["truth_parton_id"] = _ragged(off, partons.node, "truth_parton_id")
    # emissions
    off = _offsets(np.bincount(record.event, minlength=n))
    for name, values in (
        ("truth_emis_parent", record.parent),
        ("truth_emis_daughter1", record.daughter1),
        ("truth_emis_daughter2", record.daughter2),
        ("truth_emis_energy", record.energy),
        ("truth_emis_z", record.z),
        ("truth_emis_theta", record.theta),
        ("truth_emis_phi", record.phi),
        ("truth_emis_kt", record.kt),
        ("truth_emis_channel", record.channel),
        ("truth_emis_prong", record.prong),
        ("truth_emis_primary", record.primary),
        ("truth_emis_depth", record.depth),
    ):
        cols[name] = _ragged(off, values, name)
    # primary Lund emissions
    rows = np.flatnonzero(primary)
    off = _offsets(n_primary)
    cols["truth_lund_ln_inv_delta"] = _ragged(off, -np.log(record.theta[rows]), "truth_lund_ln_inv_delta")
    cols["truth_lund_ln_kt"] = _ragged(off, ln_kt[rows], "truth_lund_ln_kt")
    cols["truth_lund_z"] = _ragged(off, np.minimum(record.z, 1.0 - record.z)[rows], "truth_lund_z")
    cols["truth_lund_prong"] = _ragged(off, record.prong[rows], "truth_lund_prong")
    return cols


# ----------------------------------------------------------------------------------------------------------------------
# One process group and one chunk


def _simulate_group(
    n: int,
    code: int,
    cfg: SimConfig,
    rngs: Sequence[np.random.Generator],
    seed: int,
    event_ids: np.ndarray,
    times: dict[str, float],
    stats: dict[str, float],
) -> tuple[JetData, np.ndarray, HardEvents]:
    """Simulate ``n`` events of one process with its four streams; returns the jets, the pass mask, the hard events."""
    rng_hard, rng_shower, rng_hadron, rng_ue = rngs
    clock = _Clock(times)
    hard = sample_hard(n, code, cfg, rng_hard)
    clock.tick("hard")
    partons, record = shower_events(hard, cfg, rng_shower)
    clock.tick("shower")
    if cfg.hadron.hadronize:
        hadrons = hadronize(partons, cfg.hadron, rng_hadron, stats)
    else:
        hadrons = HadronArrays.from_partons(partons)
    clock.tick("hadronize")
    if cfg.ue.ue:
        ue = underlying_event(n, cfg, rng_ue)
        hadrons = HadronArrays.concat([hadrons, ue])
        hadrons = hadrons.select(np.argsort(hadrons.event, kind="stable"))
    clock.tick("ue")
    # move to the lab (the production vertex is a four-vector and moves with the same transformation)
    y, phi = hard.y[hadrons.event], hard.phi[hadrons.event]
    hadrons.p4 = to_lab(hadrons.p4, y, phi)
    hadrons.vertex = to_lab(hadrons.vertex, y, phi)
    partons.p4 = to_lab(partons.p4, hard.y[partons.event], hard.phi[partons.event])
    if cfg.detector.detector and cfg.hadron.hadronize:
        particles = detector(hadrons, cfg.detector, (seed, event_ids))
    else:
        particles = ParticleArrays.from_hadrons(hadrons)
    clock.tick("detector")
    data, passed = build_jets(code, hard, partons, record, particles, cfg, event_ids, times)
    return data, passed, hard


class _Clock:
    """Adds the wall time since the previous tick to ``times[stage]``."""

    def __init__(self, times: dict[str, float]) -> None:
        self.times = times
        self.last = time.perf_counter()

    def tick(self, stage: str) -> None:
        now = time.perf_counter()
        self.times[stage] = self.times.get(stage, 0.0) + now - self.last
        self.last = now


@dataclasses.dataclass
class _Task:
    chunk: int
    size: int
    config: SimConfig
    processes: tuple[str, ...]
    seed: int
    first_event_id: int
    process_overrides: dict[str, dict[str, Any]]


@dataclasses.dataclass
class _ChunkResult:
    data: JetData
    times: dict[str, float]
    stats: dict[str, float]
    tested: dict[str, int]
    contained: dict[str, int]


def balanced_counts(n: int, k: int) -> np.ndarray:
    """``n`` items over ``k`` processes, balanced to +-1 with the remainder to the first processes."""
    counts = np.full(k, n // k, dtype=np.int64)
    counts[: n % k] += 1
    return counts


def _simulate_chunk(task: _Task) -> _ChunkResult:
    """Simulate one chunk: draw the process order, simulate each process group, interleave the groups."""
    root = np.random.SeedSequence(task.seed, spawn_key=(task.chunk,))
    stages = root.spawn(4)
    rng_chunk = np.random.default_rng(stages[0])
    codes = np.array([PROCESSES.index(p) for p in task.processes])
    labels = np.repeat(codes, balanced_counts(task.size, len(codes)))
    process_at = labels[rng_chunk.permutation(task.size)]
    times: dict[str, float] = {}
    stats: dict[str, float] = {}
    tested: dict[str, int] = {}
    contained: dict[str, int] = {}
    parts: list[JetData] = []
    positions: list[np.ndarray] = []
    for name, code in zip(task.processes, codes, strict=True):
        where = np.flatnonzero(process_at == code)
        if len(where) == 0:
            continue
        cfg = task.config.replace(**task.process_overrides[name]) if name in task.process_overrides else task.config
        rngs = [
            np.random.default_rng(np.random.SeedSequence(task.seed, spawn_key=(task.chunk, stage, int(code))))
            for stage in range(4)
        ]
        event_ids = task.first_event_id + task.chunk * CHUNK_SIZE + where
        data, passed, hard = _simulate_group(len(where), int(code), cfg, rngs, task.seed, event_ids, times, stats)
        tested[name] = tested.get(name, 0) + hard.n_tested
        contained[name] = contained.get(name, 0) + hard.n_contained
        if not passed.all():
            data, where = data.select(passed), where[passed]
        parts.append(data)
        positions.append(where)
    data = JetData.concat(parts) if len(parts) > 1 else parts[0]
    data = data.select(np.argsort(np.concatenate(positions), kind="stable"))
    return _ChunkResult(data, times, stats, tested, contained)


# ----------------------------------------------------------------------------------------------------------------------
# Provenance


def nearest_preset(config: SimConfig) -> tuple[str, dict[str, Any]]:
    """Return the ladder preset closest to ``config`` (fewest differing fields) and the fields that differ from it."""
    flat = _flat(config)
    best: tuple[str, dict[str, Any]] | None = None
    for name, candidate in PRESETS.items():
        ref = _flat(candidate)
        diff = {key: value for key, value in flat.items() if value != ref[key]}
        if best is None or len(diff) < len(best[1]):
            best = (name, diff)
    assert best is not None
    return best


def _flat(config: SimConfig) -> dict[str, Any]:
    return {key: value for part in config.to_dict().values() for key, value in part.items()}


_GIT: list[str | None] = []


def _git_commit() -> str | None:
    """Commit of the checkout this package runs from (``-dirty`` with uncommitted changes), or None."""
    if _GIT:
        return _GIT[0]
    root = Path(__file__).resolve().parents[3]
    commit: str | None = None
    if (root / ".git").exists():
        try:
            run = {"cwd": root, "capture_output": True, "text": True, "timeout": 10, "check": True}
            commit = subprocess.run(["git", "rev-parse", "HEAD"], **run).stdout.strip() or None  # type: ignore[call-overload]
            if commit is not None:
                status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], **run)  # type: ignore[call-overload]
                if status.stdout.strip():
                    commit += "-dirty"
        except (OSError, subprocess.SubprocessError):
            commit = None
    _GIT.append(commit)
    return commit


def _provenance(
    *,
    config: SimConfig,
    processes: tuple[str, ...],
    seed: int,
    workers: int,
    first_event_id: int,
    process_overrides: dict[str, dict[str, Any]],
    results: list[_ChunkResult],
    wall: float,
) -> dict[str, Any]:
    import pyarrow

    name, overrides = nearest_preset(config)
    times = dict.fromkeys(STAGES, 0.0)
    tested: dict[str, int] = {}
    contained: dict[str, int] = {}
    hadrons = capped = reshuffles = reshuffle_failed = failed_w_in_top = 0.0
    residual = 0.0
    for r in results:
        for key, value in r.times.items():
            times[key] = times.get(key, 0.0) + value
        for key, value in r.tested.items():
            tested[key] = tested.get(key, 0) + value
        for key, value in r.contained.items():
            contained[key] = contained.get(key, 0) + value
        hadrons += r.stats.get("hadrons", 0)
        capped += r.stats.get("hadrons_capped", 0)
        reshuffles += r.stats.get("hadron_reshuffles", 0)
        reshuffle_failed += r.stats.get("hadron_reshuffle_failed", 0)
        failed_w_in_top += r.stats.get("hadron_reshuffle_failed_w_in_top", 0)
        residual = max(residual, r.stats.get("hadron_reshuffle_residual", 0.0))
    return {
        "seed": seed,
        "first_event_id": first_event_id,
        "chunk_size": CHUNK_SIZE,
        "workers": workers,
        "preset": name,
        "processes": list(processes),
        "overrides": overrides,
        "process_overrides": process_overrides,
        "containment_efficiency": {p: contained[p] / tested[p] for p in processes if tested.get(p)},
        "package_version": __version__,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pyarrow_version": pyarrow.__version__,
        "platform": platform.platform(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": round(wall, 3),
        "command_line": None,
        "stage_times_s": {key: round(value, 4) for key, value in times.items()},
        "hadron_kt_capped": capped / hadrons if hadrons else 0.0,
        "hadron_reshuffle": {
            "steps": int(reshuffles),
            "failed": int(reshuffle_failed),
            "failed_w_in_top": int(failed_w_in_top),
            "max_residual": residual,
        },
    }


# ----------------------------------------------------------------------------------------------------------------------
# generate


def _check_processes(processes: Sequence[str]) -> tuple[str, ...]:
    if isinstance(processes, str):
        processes = [p.strip() for p in processes.split(",") if p.strip()]
    out = tuple(processes)
    if not out:
        raise ValueError("no process given; choose from " + ", ".join(PROCESSES))
    for p in out:
        if p not in PROCESSES:
            close = difflib.get_close_matches(str(p), PROCESSES, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"unknown process {p!r}; choose from {', '.join(PROCESSES)}.{hint}")
    if len(set(out)) != len(out):
        raise ValueError(f"processes must be unique, got {list(out)}")
    return out


def _check_workers(workers: int | str) -> int:
    if workers == "auto":
        return min(os.cpu_count() or 1, 8)
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError(f"workers must be a positive integer or 'auto', got {workers!r}")
    return int(workers)


@contextlib.contextmanager
def _single_threaded_children() -> Iterator[None]:
    """Set ``OMP_NUM_THREADS=1`` (and friends) while the worker processes are spawned."""
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {name: os.environ.get(name) for name in names}
    os.environ.update(dict.fromkeys(names, "1"))
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_chunks(tasks: list[_Task], workers: int, progress: bool, total: int) -> Iterator[_ChunkResult]:
    def report(i: int, done: int) -> None:
        if progress:
            print(f"\rchunk {i + 1}/{total}: {done} jets", end="", file=sys.stderr, flush=True)

    done = 0
    if workers == 1 or len(tasks) == 1:
        for task in tasks:
            result = _simulate_chunk(task)
            done += len(result.data)
            report(task.chunk, done)
            yield result
        return
    context = multiprocessing.get_context("spawn")
    with (
        _single_threaded_children(),
        concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=context) as pool,
    ):
        for task, result in zip(tasks, pool.map(_simulate_chunk, tasks), strict=True):
            done += len(result.data)
            report(task.chunk, done)
            yield result


def generate(
    n_jets: int,
    *,
    config: SimConfig | None = None,
    processes: Sequence[str] = ("qcd", "wqq", "tbqq"),
    seed: int = 0,
    workers: int | Literal["auto"] = 1,
    first_event_id: int = 0,
    process_overrides: Mapping[str, Mapping[str, object]] | None = None,
    truth: bool = True,
    progress: bool = False,
) -> JetData:
    """Generate jets with the toy "A Spray of Hadrons".

    Parameters
    ----------
    n_jets : int
        Number of jets (exactly; the processes are balanced to +-1 within each chunk of 1000 jets).
    config : SimConfig, optional
        The configuration (default: the ``"detector"`` preset).
    processes : sequence of str
        Processes from :data:`itj.toy.PROCESSES` (``"qcd"``, ``"wqq"``, ``"tbqq"``, ``"q"``, ``"g"``).
    seed : int
        The run seed (same seed, same events; see the module docstring for what is paired).
    workers : int or "auto"
        Worker processes (spawn context; ``"auto"`` = min(cpu count, 8)). The output does not depend on it. With
        more than one worker, call ``generate`` under ``if __name__ == "__main__":`` in a script (spawned workers
        import the script again); notebooks and ``jetgen`` need nothing.
    first_event_id : int
        ``truth_event_id`` of the first jet (for sharded generation; it also keys the detector randomness).
    process_overrides : Mapping, optional
        Per-process configuration overrides, e.g. ``{"wqq": {"pi0_fraction": 0.4}}``; the other processes are
        unchanged jet by jet.
    truth : bool
        Keep the ``truth_*`` columns.
    progress : bool
        Print the progress to stderr.

    Returns
    -------
    JetData
        The jets in generation order, with ``config`` and ``provenance``.

    Examples
    --------
    >>> data = generate(30, seed=1)
    >>> len(data), sorted(set(data.process_names.tolist()))
    (30, ['qcd', 'tbqq', 'wqq'])
    """
    start = time.perf_counter()
    if isinstance(n_jets, bool) or not isinstance(n_jets, (int, np.integer)) or n_jets < 1:
        raise ValueError(f"n_jets must be a positive integer, got {n_jets!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if isinstance(first_event_id, bool) or not isinstance(first_event_id, (int, np.integer)) or first_event_id < 0:
        raise ValueError(f"first_event_id must be a non-negative integer, got {first_event_id!r}")
    config = SimConfig() if config is None else config
    if not isinstance(config, SimConfig):
        raise TypeError(f"config must be a SimConfig, got {type(config).__name__}")
    procs = _check_processes(processes)
    n_workers = _check_workers(workers)
    overrides: dict[str, dict[str, Any]] = {}
    for name, values in (process_overrides or {}).items():
        if name not in procs:
            raise ValueError(f"process_overrides for {name!r}, which is not among the processes {list(procs)}")
        config.replace(**values)  # validates the names and values
        overrides[name] = dict(values)
    n_jets, seed, first_event_id = int(n_jets), int(seed), int(first_event_id)
    cut = config.jet.jet_pt_min is not None or config.jet.jet_pt_max is not None

    def make(chunk: int, size: int) -> _Task:
        return _Task(chunk, size, config, procs, seed, first_event_id, overrides)

    results: list[_ChunkResult] = []
    if not cut:
        n_chunks = -(-n_jets // CHUNK_SIZE)
        tasks = [make(k, min(CHUNK_SIZE, n_jets - k * CHUNK_SIZE)) for k in range(n_chunks)]
        results = list(_run_chunks(tasks, n_workers, progress, n_chunks))
    else:  # chunks of 1000 until enough jets pass the reco window
        have, k = 0, 0
        while have < n_jets:
            wave = max(1, n_workers)
            tasks = [make(k + i, CHUNK_SIZE) for i in range(wave)]
            for result in _run_chunks(tasks, n_workers, progress, k + wave):
                results.append(result)
                have += len(result.data)
            k += wave
            if k > 1000 + 10 * n_jets // CHUNK_SIZE:
                raise RuntimeError("almost no jet passes jet_pt_min/jet_pt_max: check the window")
    if progress:
        print(file=sys.stderr)
    data = JetData.concat([r.data for r in results]) if len(results) > 1 else results[0].data
    if len(data) > n_jets:
        data = data.head(n_jets)
    provenance = _provenance(
        config=config,
        processes=procs,
        seed=seed,
        workers=n_workers,
        first_event_id=first_event_id,
        process_overrides=overrides,
        results=results,
        wall=time.perf_counter() - start,
    )
    if not truth:
        data = data.without_truth()
    return JetData(data.columns, config=config.to_dict(), provenance=provenance, source="itj")


def show_ladder(
    process: str = "wqq", *, seed: int = 0, jet: int = 0, config: SimConfig | None = None
) -> dict[str, JetData]:
    """Generate the same jet at the four presets of the ladder (one jet each, ``partons`` to ``detector``).

    Parameters
    ----------
    process : str
        One of :data:`itj.toy.PROCESSES`.
    seed : int
        Run seed.
    jet : int
        Index of the jet in the run (``jet + 1`` jets are generated per preset).
    config : SimConfig, optional
        Base configuration; its ladder switches are set by each preset (default: the presets themselves).

    Returns
    -------
    dict of str to JetData
        Preset name to a data set with one jet, in ladder order (input of :func:`itj.plot.ladder`).
    """
    if isinstance(jet, bool) or not isinstance(jet, (int, np.integer)) or jet < 0:
        raise ValueError(f"jet must be a non-negative integer, got {jet!r}")
    _check_processes([process])
    out: dict[str, JetData] = {}
    for name, ladder_config in PRESETS.items():
        cfg = ladder_config
        if config is not None:
            switches = {key: _flat(ladder_config)[key] for key in _LADDER}
            cfg = config.replace(**switches)
        data = generate(int(jet) + 1, config=cfg, processes=(process,), seed=seed)
        out[name] = data.select(np.array([int(jet)]))
    return out
