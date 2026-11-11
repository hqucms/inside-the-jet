"""Small synthetic jets built from four-vectors directly (no generator), for the tests of the analysis tools.

Every jet has the 41 JetClass columns with the JetClass dtypes: exact jet kinematics (sum of the constituents),
particles sorted by pT, ``part_deta``/``part_dphi`` with the JetClass sign convention, and zero substructure
(``jet_sdmass`` and ``jet_tau*`` are 0, impact parameters and PID flags are 0). Particles are massless unless given
otherwise; ``(px, py, pz, E)`` on the last axis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from itj.data import (
    COLUMNS,
    JETCLASS_AUX_BRANCHES,
    JETCLASS_BRANCHES,
    JETCLASS_JET_BRANCHES,
    JETCLASS_LABELS,
    JETCLASS_PART_BRANCHES,
    JetData,
    Ragged,
)

#: aux_truth_match and the resonance PDG id of the classes the tests use.
_TRUTH_MATCH = {"QCD": (50.0, 0.0), "Wqq": (21.0, 24.0), "Tbqq": (11.0, 6.0)}


def massless(pt: float, eta: float, phi: float) -> np.ndarray:
    """Return the four-vector (px, py, pz, E) of a massless particle."""
    return np.array([pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta), pt * math.cosh(eta)])


def _wrap(dphi: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    out = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
    return np.where(out == -np.pi, np.pi, out)


def synthetic_jets(particles: Sequence[ArrayLike], labels: Sequence[str | int] | None = None) -> JetData:
    """Build a JetClass-like data set from the particle four-vectors of every jet.

    Parameters
    ----------
    particles : sequence of array_like
        One ``(n_i, 4)`` array of (px, py, pz, E) per jet.
    labels : sequence of str or int, optional
        Class of every jet: a JetClass short name (``"QCD"``, ``"Wqq"``, ``"Tbqq"``, ...) or an index into
        ``JETCLASS_LABELS``. Default: QCD.
    """
    n = len(particles)
    labels = ["QCD"] * n if labels is None else list(labels)
    if len(labels) != n:
        raise ValueError("one label per jet")
    rows: dict[str, list[np.ndarray]] = {name: [] for name in JETCLASS_PART_BRANCHES}
    jet: dict[str, np.ndarray] = {name: np.zeros(n) for name in (*JETCLASS_JET_BRANCHES, *JETCLASS_AUX_BRANCHES)}
    label_index = np.zeros(n, dtype=np.int64)
    for i, (p4, label) in enumerate(zip(particles, labels, strict=True)):
        p = np.asarray(p4, dtype=np.float64).reshape(-1, 4)
        pt = np.hypot(p[:, 0], p[:, 1])
        p = p[np.argsort(-pt, kind="stable")]
        pt = np.hypot(p[:, 0], p[:, 1])
        total = p.sum(axis=0)
        jet_pt = math.hypot(total[0], total[1])
        jet_eta = math.asinh(total[2] / jet_pt) if jet_pt > 0 else 0.0
        jet_phi = math.atan2(total[1], total[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.arcsinh(np.where(pt > 0, p[:, 2] / np.where(pt > 0, pt, 1.0), 0.0))
        sign = 1.0 if jet_eta >= 0 else -1.0
        values = {
            "part_px": p[:, 0],
            "part_py": p[:, 1],
            "part_pz": p[:, 2],
            "part_energy": p[:, 3],
            "part_deta": sign * (eta - jet_eta),
            "part_dphi": _wrap(np.arctan2(p[:, 1], p[:, 0]) - jet_phi),
        }
        for name in JETCLASS_PART_BRANCHES:
            rows[name].append(values.get(name, np.zeros(len(p))))
        jet["jet_pt"][i], jet["jet_eta"][i], jet["jet_phi"][i] = jet_pt, jet_eta, jet_phi
        jet["jet_energy"][i] = total[3]
        jet["jet_nparticles"][i] = len(p)
        short = label if isinstance(label, str) else JETCLASS_LABELS[label].removeprefix("label_")
        label_index[i] = JETCLASS_LABELS.index(f"label_{short}")
        match, pid = _TRUTH_MATCH.get(short, (0.0, 0.0))
        jet["aux_truth_match"][i] = match
        if pid:
            jet["aux_genpart_pid"][i] = pid
            jet["aux_genpart_pt"][i], jet["aux_genpart_eta"][i], jet["aux_genpart_phi"][i] = jet_pt, jet_eta, jet_phi
    columns: dict[str, np.ndarray | Ragged] = {}
    for name in JETCLASS_BRANCHES:
        dtype = COLUMNS[name].dtype
        if name in rows:
            columns[name] = Ragged.from_rows(rows[name], dtype)
        elif name in JETCLASS_LABELS:
            columns[name] = (label_index == JETCLASS_LABELS.index(name)).astype(dtype)
        else:
            columns[name] = jet[name].astype(dtype)
    return JetData(columns, source="jetclass")


def two_particle_jet(z: float, dr: float, pt: float = 600.0) -> JetData:
    """One jet of two massless particles with pT fractions z and 1 - z, Delta R = dr apart in eta (at phi 0)."""
    p1 = massless(z * pt, (1 - z) * dr, 0.0)
    p2 = massless((1 - z) * pt, -z * dr, 0.0)
    return synthetic_jets([np.stack([p1, p2])])


def three_prong_jet(pt: float = 600.0, dr: float = 0.5) -> JetData:
    """One top-like jet of three massless particles of pT pt / 3 on an equilateral triangle of side ``dr``."""
    radius = dr / math.sqrt(3.0)
    angles = np.deg2rad([90.0, 210.0, 330.0])
    p4 = [massless(pt / 3, radius * math.sin(a), radius * math.cos(a)) for a in angles]
    return synthetic_jets([np.stack(p4)], labels=["Tbqq"])


def collinear_pair_jet(lam: float, delta: float) -> JetData:
    """Two jets: three particles, and the same with the hardest one split into lam and 1 - lam, delta apart.

    The split particles sit at eta offsets ``+-delta / 2`` around the original (which keeps ``lam`` of its pT on the
    first); for ``delta = 0`` the two jets have the same total four-momentum.
    """
    base = [massless(300.0, 0.0, 0.0), massless(200.0, 0.3, 0.2), massless(100.0, -0.2, -0.3)]
    split = [
        massless(lam * 300.0, delta / 2, 0.0),
        massless((1 - lam) * 300.0, -delta / 2, 0.0),
        base[1],
        base[2],
    ]
    return synthetic_jets([np.stack(base), np.stack(split)])


def ca_tree_jet() -> JetData:
    """One jet of four massless particles with a known Cambridge/Aachen tree.

    Particles (pT, eta, phi): a (300, 0, 0), b (100, 0.1, 0), c (150, 0.5, 0), d (50, 0.5, 0.25). C/A merges a+b
    first (Delta R 0.1), then c+d (Delta R 0.25), then (ab)+(cd) (Delta R about 0.48). C/A with R = 0.4 gives the two
    jets {a, b} and {c, d}.
    """
    p4 = [massless(300.0, 0.0, 0.0), massless(100.0, 0.1, 0.0), massless(150.0, 0.5, 0.0), massless(50.0, 0.5, 0.25)]
    return synthetic_jets([np.stack(p4)])


def jetclass_like_file(tmp_path: Path, n: int = 50, *, seed: int = 0) -> Path:
    """Write ``n`` random jets with the exact JetClass schema (41 columns, no itj metadata) and return the path."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    particles = []
    labels = []
    for i in range(n):
        k = int(rng.integers(1, 40))
        pt = np.sort(rng.exponential(20.0, k) + 0.5)[::-1] * 600.0 / 20.0 / max(k, 1)
        eta = rng.normal(0.0, 0.2, k) + 0.5
        phi = rng.normal(0.0, 0.2, k)
        particles.append(np.stack([massless(a, b, c) for a, b, c in zip(pt, eta, phi, strict=True)]))
        labels.append(("QCD", "Wqq", "Tbqq")[i % 3])
    data = synthetic_jets(particles, labels)
    arrays = []
    fields = []
    for name in JETCLASS_BRANCHES:
        col = data[name]
        info = COLUMNS[name]
        if isinstance(col, Ragged):
            item = pa.from_numpy_dtype(np.dtype(info.dtype))
            arrays.append(pa.LargeListArray.from_arrays(pa.array(col.offsets), pa.array(col.values)))
            fields.append(pa.field(name, pa.large_list(pa.field("item", item, nullable=False)), nullable=False))
        else:
            arrays.append(pa.array(col))
            fields.append(pa.field(name, pa.from_numpy_dtype(np.dtype(info.dtype)), nullable=False))
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    path = Path(tmp_path) / "jetclass_like.parquet"
    pq.write_table(table, path)
    return path


#: Rows per jet of the ragged truth groups in :func:`add_truth_columns` (inclusive ranges).
_TRUTH_COUNTS = {"prong": (1, 3), "parton": (1, 12), "emission": (0, 10), "lund": (0, 5)}


def add_truth_columns(data: JetData, *, seed: int = 0) -> JetData:
    """Return ``data`` with every ``truth_*`` column of ``COLUMNS`` filled with random values of the right dtype.

    The per-particle truth columns follow the particles of ``data``; the prong, parton, emission and Lund groups get
    random row lengths. ``truth_event_id`` is ``0 ... N - 1`` and ``truth_process`` follows the JetClass labels
    (qcd 0, wqq 1, tbqq 2). For tests of the file format, not of the physics.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    counts = {"particle": data.counts}
    for group, (low, high) in _TRUTH_COUNTS.items():
        counts[group] = rng.integers(low, high + 1, n)
    offsets = {group: np.concatenate([[0], np.cumsum(c)]).astype(np.int64) for group, c in counts.items()}

    def random(dtype: str, size: int) -> np.ndarray:
        if dtype.startswith("float"):
            return rng.normal(size=size).astype(dtype)
        return rng.integers(-1, 3, size).astype(dtype)

    columns: dict[str, np.ndarray | Ragged] = {}
    for name, info in COLUMNS.items():
        if not name.startswith("truth_"):
            continue
        if info.jagged:
            off = offsets[info.group]
            columns[name] = Ragged(random(info.dtype, int(off[-1])), off)
        else:
            columns[name] = random(info.dtype, n)
    columns["truth_event_id"] = np.arange(n, dtype=np.int64)
    process = {"qcd": 0, "wqq": 1, "tbqq": 2}
    columns["truth_process"] = np.array([process.get(p, 0) for p in data.process_names], dtype=np.int8)
    return data.with_columns(**columns)
