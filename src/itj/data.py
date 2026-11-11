"""Jets in memory: the column store :class:`JetData` that every part of :mod:`itj` reads and writes.

A data set is a table with one row per jet. Its columns carry the JetClass branch names (section "Data format" of
``docs/data_format.md``), so the toy files, the JetClass files, the weaver configs and pandas all use one
vocabulary::

    data["jet_pt"]          float32 [N]                    one value per jet
    data["part_px"]         Ragged with N rows             one value per particle, pT-descending
    data["truth_prong_px"]  Ragged with N rows             toy only: one value per prong (decay parton)

A :class:`Ragged` column stores the values of all jets in one flat array and the row boundaries in ``offsets``: the
values of jet ``i`` are ``values[offsets[i]:offsets[i + 1]]``. The columns of one group (particles, prongs, partons,
emissions, Lund emissions) share their offsets. ``COLUMNS`` lists every column the toy can write with its dtype,
units, description and group; it is the single source for the file writer, the documentation and the tests.

Examples
--------
>>> import numpy as np
>>> r = Ragged.from_counts(np.arange(5.0), [2, 0, 3])
>>> len(r), r.counts().tolist(), r[2].tolist()
(3, [2, 0, 3], [2.0, 3.0, 4.0])
>>> r.padded(4, mode="wrap").tolist()
[[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [2.0, 3.0, 4.0, 2.0]]
"""

from __future__ import annotations

import contextlib
import difflib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    import pandas

__all__ = [
    "COLUMNS",
    "GROUPS",
    "JETCLASS_AUX_BRANCHES",
    "JETCLASS_BRANCHES",
    "JETCLASS_JET_BRANCHES",
    "JETCLASS_LABELS",
    "JETCLASS_PART_BRANCHES",
    "JETS3",
    "ColumnInfo",
    "JetData",
    "JetView",
    "Ragged",
    "column_group",
]

# ------------------------------------------------------------------------------------------------- JetClass schema

#: The ten JetClass label branches in the JetClass class order (QCD 0 ... Wqq 7, Tbqq 8, Tbl 9).
JETCLASS_LABELS: tuple[str, ...] = (
    "label_QCD",
    "label_Hbb",
    "label_Hcc",
    "label_Hgg",
    "label_H4q",
    "label_Hqql",
    "label_Zqq",
    "label_Wqq",
    "label_Tbqq",
    "label_Tbl",
)

#: The three classes of the course: QCD, W -> qq' and t -> bqq'.
JETS3: tuple[str, ...] = ("label_QCD", "label_Wqq", "label_Tbqq")

#: The 16 per-particle JetClass branches in the order of the JetClass files.
JETCLASS_PART_BRANCHES: tuple[str, ...] = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_deta",
    "part_dphi",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
)

#: The 10 per-jet JetClass branches in the order of the JetClass files.
JETCLASS_JET_BRANCHES: tuple[str, ...] = (
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_energy",
    "jet_nparticles",
    "jet_sdmass",
    "jet_tau1",
    "jet_tau2",
    "jet_tau3",
    "jet_tau4",
)

#: The 5 auxiliary JetClass branches (generator-level resonance and truth match) in file order.
JETCLASS_AUX_BRANCHES: tuple[str, ...] = (
    "aux_genpart_eta",
    "aux_genpart_phi",
    "aux_genpart_pid",
    "aux_genpart_pt",
    "aux_truth_match",
)

#: All 41 JetClass branches in file order: particles, labels, jet, aux.
JETCLASS_BRANCHES: tuple[str, ...] = (
    JETCLASS_PART_BRANCHES + JETCLASS_LABELS + JETCLASS_JET_BRANCHES + JETCLASS_AUX_BRANCHES
)

#: Group name -> column prefixes. Used only to guess the group of a column that is not in ``COLUMNS`` (for example a
#: column a student adds with :meth:`JetData.with_columns`); every other column is in the group "jet". No per-jet
#: column starts with one of these prefixes, and every column that does is ragged.
GROUPS: dict[str, tuple[str, ...]] = {
    "particle": ("part_", "truth_part_"),
    "prong": ("truth_prong_",),
    "parton": ("truth_parton_",),
    "emission": ("truth_emis_",),
    "lund": ("truth_lund_",),
}

_RAGGED_PREFIXES: tuple[str, ...] = tuple(prefix for prefixes in GROUPS.values() for prefix in prefixes)

#: The processes of the toy; ``truth_process`` is the index into this tuple (the same tuple as ``itj.toy.PROCESSES``,
#: repeated here so that this module never imports the generator).
_TOY_PROCESSES: tuple[str, ...] = ("qcd", "wqq", "tbqq", "q", "g")


@dataclass(frozen=True)
class ColumnInfo:
    """What a column holds: the entry of one column in :data:`COLUMNS`.

    Attributes
    ----------
    dtype : str
        NumPy dtype name of the values (``"float32"``, ``"int32"``, ``"int8"``, ``"int16"``, ``"int64"``,
        ``"bool"``).
    units : str
        Units of the values (``"GeV"``, ``"rad"``, ``"mm"`` or ``""``).
    description : str
        One sentence; written as Parquet field metadata.
    group : str
        ``"jet"``, ``"particle"``, ``"prong"``, ``"parton"``, ``"emission"`` or ``"lund"``.
    jagged : bool
        True for a :class:`Ragged` column (one value per particle, prong, ...), False for one value per jet.
    """

    dtype: str
    units: str
    description: str
    group: str
    jagged: bool


def _jet(dtype: str, units: str, description: str) -> ColumnInfo:
    return ColumnInfo(dtype, units, description, "jet", False)


def _ragged(group: str, dtype: str, units: str, description: str) -> ColumnInfo:
    return ColumnInfo(dtype, units, description, group, True)


def _particle(dtype: str, units: str, description: str) -> ColumnInfo:
    return _ragged("particle", dtype, units, description)


_FLAG = "1 if the particle is reconstructed as {}, else 0 (all 0 at the partons and shower presets)."

#: Every column the toy can write (the JetClass branches, then the truth branches), section "Data format". The single
#: source for the file writer, the documentation and the tests; ``COLUMNS[name].group`` is the only authority for the
#: group of a known column.
COLUMNS: dict[str, ColumnInfo] = {
    # JetClass particle branches
    "part_px": _particle("float32", "GeV", "x component of the particle momentum."),
    "part_py": _particle("float32", "GeV", "y component of the particle momentum."),
    "part_pz": _particle("float32", "GeV", "z component of the particle momentum."),
    "part_energy": _particle("float32", "GeV", "Energy of the particle."),
    "part_deta": _particle(
        "float32", "", "Pseudorapidity relative to the jet axis, s (eta - eta_jet) with s = +1 if eta_jet >= 0 else -1."
    ),
    "part_dphi": _particle("float32", "rad", "Azimuth relative to the jet axis, wrapped to (-pi, pi]."),
    "part_d0val": _particle("float32", "mm", "Transverse impact parameter of the track (0 for neutral particles)."),
    "part_d0err": _particle("float32", "mm", "Uncertainty of the transverse impact parameter (0 for neutrals)."),
    "part_dzval": _particle("float32", "mm", "Longitudinal impact parameter of the track (0 for neutral particles)."),
    "part_dzerr": _particle("float32", "mm", "Uncertainty of the longitudinal impact parameter (0 for neutrals)."),
    "part_charge": _particle("float32", "", "Electric charge of the particle: -1, 0 or 1."),
    "part_isChargedHadron": _particle("int32", "", _FLAG.format("a charged hadron")),
    "part_isNeutralHadron": _particle("int32", "", _FLAG.format("a neutral hadron")),
    "part_isPhoton": _particle("int32", "", _FLAG.format("a photon")),
    "part_isElectron": _particle("int32", "", _FLAG.format("an electron")),
    "part_isMuon": _particle("int32", "", _FLAG.format("a muon")),
    # JetClass labels (mixed dtypes, as in the JetClass files)
    "label_QCD": _jet("float32", "", "1 for a QCD jet (the toy processes qcd, q and g)."),
    "label_Hbb": _jet("bool", "", "True for H -> bb (always false in the toy)."),
    "label_Hcc": _jet("bool", "", "True for H -> cc (always false in the toy)."),
    "label_Hgg": _jet("bool", "", "True for H -> gg (always false in the toy)."),
    "label_H4q": _jet("bool", "", "True for H -> WW* -> 4q (always false in the toy)."),
    "label_Hqql": _jet("bool", "", "True for H -> WW* -> qq l nu (always false in the toy)."),
    "label_Zqq": _jet("int32", "", "1 for Z -> qq (always 0 in the toy)."),
    "label_Wqq": _jet("int32", "", "1 for W -> qq' (the toy process wqq)."),
    "label_Tbqq": _jet("int32", "", "1 for t -> bqq' (the toy process tbqq)."),
    "label_Tbl": _jet("int32", "", "1 for t -> b l nu (always 0 in the toy)."),
    # JetClass jet branches
    "jet_pt": _jet("float32", "GeV", "Transverse momentum of the sum of the constituents."),
    "jet_eta": _jet("float32", "", "Pseudorapidity of the sum of the constituents."),
    "jet_phi": _jet("float32", "rad", "Azimuth of the sum of the constituents."),
    "jet_energy": _jet("float32", "GeV", "Energy of the sum of the constituents."),
    "jet_nparticles": _jet("float32", "", "Number of constituents."),
    "jet_sdmass": _jet("float32", "GeV", "Soft-drop mass (C/A, beta 0, z_cut 0.1, R0 0.8)."),
    "jet_tau1": _jet("float32", "", "N-subjettiness tau_1 (beta 1, R0 0.8); 0 if the jet has at most 1 particle."),
    "jet_tau2": _jet("float32", "", "N-subjettiness tau_2 (beta 1, R0 0.8); 0 if the jet has at most 2 particles."),
    "jet_tau3": _jet("float32", "", "N-subjettiness tau_3 (beta 1, R0 0.8); 0 if the jet has at most 3 particles."),
    "jet_tau4": _jet("float32", "", "N-subjettiness tau_4 (beta 1, R0 0.8); 0 if the jet has at most 4 particles."),
    # JetClass aux branches
    "aux_genpart_eta": _jet("float32", "", "Pseudorapidity of the resonance (W or top); 0 for QCD."),
    "aux_genpart_phi": _jet("float32", "rad", "Azimuth of the resonance (W or top); 0 for QCD."),
    "aux_genpart_pid": _jet("float32", "", "PDG id of the resonance (+-24 or +-6, random sign); 0 for QCD."),
    "aux_genpart_pt": _jet("float32", "GeV", "Transverse momentum of the resonance (W or top); 0 for QCD."),
    "aux_truth_match": _jet("float32", "", "JetClass truth-match code: 50 QCD, 21 W, 11 top."),
    # toy truth, one value per jet
    "truth_event_id": _jet("int64", "", "Event number: first_event_id + row."),
    "truth_process": _jet("int8", "", "Index of the process in itj.toy.PROCESSES (qcd, wqq, tbqq, q, g)."),
    "truth_init_pid": _jet("int32", "", "Initiator: 1 quark (any flavour), 21 gluon, 24 W, 6 top."),
    "truth_nprongs": _jet("int8", "", "Number of prongs (decay partons): 1, 2 or 3."),
    "truth_seed_pt": _jet("float32", "GeV", "Transverse momentum of the seed parton or resonance (lab)."),
    "truth_seed_eta": _jet("float32", "", "Pseudorapidity of the seed parton or resonance (lab)."),
    "truth_seed_phi": _jet("float32", "rad", "Azimuth of the seed parton or resonance (lab)."),
    "truth_mass": _jet("float32", "GeV", "Breit-Wigner mass of the resonance (0 for QCD)."),
    "truth_cos_theta_star": _jet(
        "float32", "", "W decay angle (W: in the W frame; top: the helicity angle); NaN for QCD."
    ),
    "truth_dr_min": _jet("float32", "", "Smallest pairwise Delta R of the decay quarks; NaN for QCD."),
    "truth_dr_max": _jet("float32", "", "Largest pairwise Delta R of the decay quarks; NaN for QCD."),
    "truth_n_emissions": _jet("int16", "", "Number of accepted shower branchings."),
    "truth_n_primary": _jet("int16", "", "Number of primary branchings, summed over the prongs."),
    "truth_lnkt_max": _jet("float32", "", "ln(kt / GeV) of the hardest primary emission; NaN if there is none."),
    # toy truth, one value per prong
    "truth_prong_px": _ragged("prong", "float32", "GeV", "x component of the prong momentum (lab)."),
    "truth_prong_py": _ragged("prong", "float32", "GeV", "y component of the prong momentum (lab)."),
    "truth_prong_pz": _ragged("prong", "float32", "GeV", "z component of the prong momentum (lab)."),
    "truth_prong_energy": _ragged("prong", "float32", "GeV", "Energy of the prong (lab)."),
    "truth_prong_deta": _ragged("prong", "float32", "", "Pseudorapidity of the prong relative to the jet axis."),
    "truth_prong_dphi": _ragged("prong", "float32", "rad", "Azimuth of the prong relative to the jet axis."),
    "truth_prong_pid": _ragged("prong", "int32", "", "PDG id of the prong (+-1..5, 21)."),
    # toy truth, one value per particle
    "truth_part_prong": _particle("int8", "", "Prong of the particle; -1 for the underlying event."),
    "truth_part_parton": _particle("int16", "", "Index into the truth_parton_* columns; -1 for the underlying event."),
    "truth_part_is_ue": _particle("int8", "", "1 if the particle comes from the underlying event."),
    "truth_part_from_b": _particle("int8", "", "1 if the particle comes from a B-hadron decay."),
    "truth_part_purity": _particle(
        "float32", "", "Energy share of the largest contributor (1 for tracks and below the detector preset)."
    ),
    # toy truth, one value per final parton
    "truth_parton_px": _ragged("parton", "float32", "GeV", "x component of the parton momentum after the shower."),
    "truth_parton_py": _ragged("parton", "float32", "GeV", "y component of the parton momentum after the shower."),
    "truth_parton_pz": _ragged("parton", "float32", "GeV", "z component of the parton momentum after the shower."),
    "truth_parton_energy": _ragged("parton", "float32", "GeV", "Energy of the parton after the shower."),
    "truth_parton_pid": _ragged("parton", "int32", "", "PDG id of the parton."),
    "truth_parton_prong": _ragged("parton", "int8", "", "Prong of the parton."),
    "truth_parton_id": _ragged("parton", "int16", "", "Shower node id of the parton."),
    # toy truth, one value per shower branching
    "truth_emis_parent": _ragged("emission", "int16", "", "Shower node id of the parent."),
    "truth_emis_daughter1": _ragged("emission", "int16", "", "Shower node id of the first daughter."),
    "truth_emis_daughter2": _ragged("emission", "int16", "", "Shower node id of the second daughter."),
    "truth_emis_energy": _ragged("emission", "float32", "GeV", "Energy of the parent, as generated."),
    "truth_emis_z": _ragged("emission", "float32", "", "Energy fraction of the first daughter (as in P(z))."),
    "truth_emis_theta": _ragged("emission", "float32", "rad", "Opening angle of the branching, as generated."),
    "truth_emis_phi": _ragged("emission", "float32", "rad", "Azimuth of the branching, as generated."),
    "truth_emis_kt": _ragged("emission", "float32", "GeV", "Relative transverse momentum z (1 - z) E theta."),
    "truth_emis_channel": _ragged("emission", "int8", "", "Splitting: 0 q -> qg, 1 g -> gg, 2 g -> qq."),
    "truth_emis_prong": _ragged("emission", "int8", "", "Prong of the branching."),
    "truth_emis_primary": _ragged("emission", "int8", "", "1 if the branching is on the harder chain of its prong."),
    "truth_emis_depth": _ragged("emission", "int8", "", "Number of branchings above this one."),
    # toy truth, one value per primary Lund emission
    "truth_lund_ln_inv_delta": _ragged("lund", "float32", "", "ln(1 / theta) of the primary emission."),
    "truth_lund_ln_kt": _ragged("lund", "float32", "", "ln(min(z, 1 - z) E theta / GeV) of the primary emission."),
    "truth_lund_z": _ragged("lund", "float32", "", "min(z, 1 - z) of the primary emission."),
    "truth_lund_prong": _ragged("lund", "int8", "", "Prong of the primary emission."),
}


def column_group(name: str, *, jagged: bool | None = None) -> str:
    """Return the group of a column: ``COLUMNS[name].group`` for a known column, else a guess from :data:`GROUPS`.

    Parameters
    ----------
    name : str
        Column name.
    jagged : bool, optional
        Whether the column is a :class:`Ragged`. An unknown ragged column without one of the prefixes of
        :data:`GROUPS` is taken to be a particle column.

    Returns
    -------
    str
        ``"jet"``, ``"particle"``, ``"prong"``, ``"parton"``, ``"emission"`` or ``"lund"``.

    Examples
    --------
    >>> column_group("part_px"), column_group("truth_lund_z"), column_group("my_score"), column_group("w", jagged=True)
    ('particle', 'lund', 'jet', 'particle')
    """
    if name in COLUMNS:
        return COLUMNS[name].group
    for group, prefixes in GROUPS.items():
        if name.startswith(prefixes):
            return group
    return "particle" if jagged else "jet"


def _suggest(name: str, choices: Sequence[str]) -> str:
    """Return " (did you mean ...?)" with the closest names, or an empty string."""
    close = difflib.get_close_matches(name, list(choices), n=3)
    return f" (did you mean {', '.join(repr(c) for c in close)}?)" if close else ""


def _fmt(n: int) -> str:
    """Format an integer with a space as the thousands separator (``30 000``)."""
    return f"{n:,}".replace(",", " ")


# ------------------------------------------------------------------------------------------------------- Ragged


def _row_selection(rows: ArrayLike, n: int) -> np.ndarray:
    """Normalise a bool mask [n] or an integer index array into an int64 index array (negative indices allowed)."""
    index = np.asarray(rows)
    if index.dtype == np.bool_:
        if index.shape != (n,):
            raise ValueError(f"a boolean row mask must have shape ({n},), got {index.shape}")
        return np.flatnonzero(index).astype(np.int64)
    if index.size == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.issubdtype(index.dtype, np.integer):
        raise TypeError(f"rows must be a boolean mask or an integer index array, got dtype {index.dtype}")
    index = index.astype(np.int64).reshape(-1)
    if index.size and (np.min(index) < -n or np.max(index) >= n):
        raise IndexError(f"row index out of range for {n} rows")
    return np.where(index < 0, index + n, index)


def _contiguous(index: np.ndarray) -> tuple[int, int] | None:
    """Return (start, stop) if ``index`` is a run of consecutive increasing integers, else None."""
    if index.size == 0:
        return None
    start, stop = int(index[0]), int(index[-1]) + 1
    if stop - start == index.size and (index.size == 1 or bool(np.all(np.diff(index) == 1))):
        return start, stop
    return None


@dataclass(frozen=True, eq=False)
class Ragged:
    """A ragged (jagged) array: rows of different lengths stored as flat ``values`` and row ``offsets``.

    The values of row ``i`` are ``values[offsets[i]:offsets[i + 1]]``. ``values`` may have trailing dimensions
    (shape ``(n_values, ...)``); every method works along the first axis.

    Attributes
    ----------
    values : numpy.ndarray
        Flat values of all rows, shape ``(offsets[-1], ...)``.
    offsets : numpy.ndarray
        int64 [N + 1] row boundaries; ``offsets[0] == 0``, non-decreasing.
    """

    values: np.ndarray
    offsets: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        offsets = np.asarray(self.offsets)
        if offsets.ndim != 1 or offsets.size == 0:
            raise ValueError(f"offsets must be a non-empty 1-D array, got shape {offsets.shape}")
        if not np.issubdtype(offsets.dtype, np.integer):
            raise TypeError(f"offsets must be integers, got dtype {offsets.dtype}")
        offsets = offsets.astype(np.int64, copy=False)
        if values.ndim == 0:
            raise ValueError("values must be at least 1-D")
        if offsets[0] != 0:
            raise ValueError(f"offsets[0] must be 0, got {offsets[0]}")
        if offsets.size > 1 and np.any(np.diff(offsets) < 0):
            raise ValueError("offsets must be non-decreasing")
        if offsets[-1] != len(values):
            raise ValueError(f"offsets[-1] = {offsets[-1]} does not match the number of values ({len(values)})")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "offsets", offsets)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        """Return a view of the values of row ``i`` (negative indices count from the end)."""
        n = len(self)
        i = int(i)
        if i < -n or i >= n:
            raise IndexError(f"row {i} out of range for {n} rows")
        i %= n
        return self.values[self.offsets[i] : self.offsets[i + 1]]

    def __repr__(self) -> str:
        return f"Ragged({len(self)} rows, {len(self.values)} values, dtype {self.values.dtype})"

    @property
    def dtype(self) -> np.dtype:
        """Dtype of the values."""
        return self.values.dtype

    def counts(self) -> np.ndarray:
        """Return the number of values in every row (int64 [N])."""
        return np.diff(self.offsets)

    def row_index(self) -> np.ndarray:
        """Return the row of every value (int64 [len(values)])."""
        return np.repeat(np.arange(len(self), dtype=np.int64), self.counts())

    def _position(self) -> np.ndarray:
        """Return the position of every value within its row (int64 [len(values)])."""
        return np.arange(len(self.values), dtype=np.int64) - np.repeat(self.offsets[:-1], self.counts())

    def _take(self, index: np.ndarray) -> tuple[np.ndarray, np.ndarray | slice]:
        """Plan a row selection: (new offsets, value index or slice) for an int64 row index array."""
        span = _contiguous(index)
        if span is not None:
            start, stop = span
            lo, hi = int(self.offsets[start]), int(self.offsets[stop])
            return self.offsets[start : stop + 1] - lo, slice(lo, hi)
        counts = self.counts()[index]
        offsets = np.zeros(len(index) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        starts = self.offsets[:-1][index]
        value_index = np.repeat(starts - offsets[:-1], counts) + np.arange(offsets[-1], dtype=np.int64)
        return offsets, value_index

    def select(self, rows: ArrayLike) -> Ragged:
        """Return the rows given by a bool mask [N] or an integer index array (any order, repeats allowed)."""
        offsets, take = self._take(_row_selection(rows, len(self)))
        return Ragged(self.values[take], offsets)

    def padded(
        self,
        length: int | None = None,
        *,
        mode: Literal["constant", "wrap"] = "constant",
        fill: float = 0.0,
        dtype: str | None = None,
    ) -> np.ndarray:
        """Return the rows as a dense array ``[N, length, ...]``.

        Parameters
        ----------
        length : int, optional
            Padded length; longer rows are truncated. Default: the longest row (at least 1).
        mode : {"constant", "wrap"}
            ``"constant"`` fills the missing entries with ``fill``; ``"wrap"`` repeats the values of each row
            cyclically (``row[j % count]``, the ``wrap`` padding of weaver 0.5.4). Empty rows are ``fill`` in both
            modes.
        fill : float
            Value of the missing entries.
        dtype : str, optional
            Dtype of the result (default: the dtype of the values).

        Returns
        -------
        numpy.ndarray
            ``[N, length, ...]``.
        """
        counts = self.counts()
        if length is None:
            length = max(int(counts.max()) if counts.size else 0, 1)
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        out_dtype = np.dtype(dtype) if dtype is not None else self.values.dtype
        out = np.full((len(self), length, *self.values.shape[1:]), fill, dtype=out_dtype)
        if length == 0 or len(self.values) == 0:
            return out
        if mode == "constant":
            position = self._position()
            keep = position < length
            out[self.row_index()[keep], position[keep]] = self.values[keep]
        elif mode == "wrap":
            nonempty = np.flatnonzero(counts > 0)
            j = np.arange(length, dtype=np.int64)
            index = self.offsets[nonempty, None] + j[None, :] % counts[nonempty, None]
            out[nonempty] = self.values[index]
        else:
            raise ValueError(f"unknown padding mode {mode!r}: use 'constant' or 'wrap'")
        return out

    def map(self, fn: Callable[[np.ndarray], np.ndarray]) -> Ragged:
        """Apply ``fn`` to the flat values (it must keep their number) and keep the offsets."""
        values = np.asarray(fn(self.values))
        if len(values) != len(self.values):
            raise ValueError(f"fn returned {len(values)} values for {len(self.values)}")
        return Ragged(values, self.offsets)

    @staticmethod
    def from_counts(values: ArrayLike, counts: ArrayLike) -> Ragged:
        """Build a ragged array from flat values and the number of values in every row."""
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        if counts.size and np.min(counts) < 0:
            raise ValueError("counts must be >= 0")
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return Ragged(np.asarray(values), offsets)

    @staticmethod
    def from_rows(rows: Sequence[ArrayLike], dtype: str) -> Ragged:
        """Build a ragged array from a sequence of 1-D rows."""
        arrays = [np.asarray(row, dtype=dtype).reshape(-1) for row in rows]
        counts = [len(a) for a in arrays]
        values = np.concatenate(arrays) if arrays else np.zeros(0, dtype=dtype)
        return Ragged.from_counts(values, counts)

    @staticmethod
    def concat(parts: Sequence[Ragged]) -> Ragged:
        """Concatenate ragged arrays row-wise (the rows of ``parts[0]``, then those of ``parts[1]``, ...)."""
        if not parts:
            raise ValueError("nothing to concatenate")
        values = np.concatenate([p.values for p in parts])
        counts = np.concatenate([p.counts() for p in parts])
        return Ragged.from_counts(values, counts)


# ------------------------------------------------------------------------------------------------------ JetData


@dataclass(frozen=True)
class JetView:
    """One jet of a :class:`JetData`, read-only: its per-jet values and the arrays of each ragged group.

    Attributes
    ----------
    index : int
        Row of the jet in the data set.
    jet : dict of str to float
        Every per-jet column (``jet_*``, ``label_*``, ``aux_*``, per-jet ``truth_*``) as a Python scalar.
    particles : dict of str to numpy.ndarray
        ``part_*`` and ``truth_part_*``.
    prongs, partons, emissions, lund : dict of str to numpy.ndarray
        ``truth_prong_*``, ``truth_parton_*``, ``truth_emis_*``, ``truth_lund_*`` (empty for JetClass).
    """

    index: int
    jet: dict[str, float]
    particles: dict[str, np.ndarray]
    prongs: dict[str, np.ndarray] = field(default_factory=dict)
    partons: dict[str, np.ndarray] = field(default_factory=dict)
    emissions: dict[str, np.ndarray] = field(default_factory=dict)
    lund: dict[str, np.ndarray] = field(default_factory=dict)

    def particles_frame(self) -> pandas.DataFrame:
        """Return the particles of the jet as a pandas DataFrame (needs pandas from the ``analysis`` extra)."""
        import pandas as pd

        return pd.DataFrame(self.particles)


#: A column of a :class:`JetData`: a 1-D array (one value per jet) or a :class:`Ragged`.
Column: TypeAlias = "np.ndarray | Ragged"


@dataclass(frozen=True, eq=False)
class JetData:
    """A data set of jets: a column store keyed by the JetClass branch names (and the toy's ``truth_*`` names).

    Per-jet columns are 1-D NumPy arrays of length N; per-particle (and per-prong, ...) columns are :class:`Ragged`
    with N rows. All ragged columns of one group (``COLUMNS[name].group``) share the same offsets.

    Attributes
    ----------
    columns : Mapping of str to numpy.ndarray or Ragged
        The columns by name, in their order.
    config : dict, optional
        ``SimConfig.to_dict()`` of a toy file; None for JetClass.
    provenance : dict, optional
        How the data were made (seed, preset, versions, timing, ...); None for JetClass.
    source : {"itj", "jetclass"}
        Toy data or JetClass data.

    Examples
    --------
    >>> import numpy as np
    >>> data = JetData({"jet_pt": np.array([600.0, 700.0], dtype=np.float32),
    ...                 "part_px": Ragged.from_counts(np.ones(3, dtype=np.float32), [1, 2])})
    >>> len(data), data.counts.tolist(), data.names
    (2, [1, 2], ('jet_pt', 'part_px'))
    """

    columns: Mapping[str, Column]
    config: dict | None = None
    provenance: dict | None = None
    source: Literal["itj", "jetclass"] = "itj"
    _n: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.source not in ("itj", "jetclass"):
            raise ValueError(f"source must be 'itj' or 'jetclass', got {self.source!r}")
        columns: dict[str, Column] = {}
        n: int | None = None
        group_offsets: dict[str, tuple[str, np.ndarray]] = {}
        for name, col in self.columns.items():
            if not isinstance(name, str):
                raise TypeError(f"column names must be strings, got {name!r}")
            if isinstance(col, Ragged):
                group = column_group(name, jagged=True)
                if name in COLUMNS and not COLUMNS[name].jagged:
                    raise TypeError(f"column {name!r} holds one value per jet: pass a 1-D array, not a Ragged")
                if group == "jet":  # pragma: no cover - column_group never returns "jet" for jagged columns
                    raise TypeError(f"column {name!r} is ragged but belongs to the jet group")
                if group in group_offsets:
                    first, offsets = group_offsets[group]
                    if col.offsets is not offsets:
                        if not np.array_equal(col.offsets, offsets):
                            raise ValueError(
                                f"column {name!r} has other row lengths than {first!r}: the {group} columns must "
                                "share their offsets"
                            )
                        col = Ragged(col.values, offsets)  # share one offsets array per group
                else:
                    group_offsets[group] = (name, col.offsets)
                length = len(col)
            else:
                arr = np.asarray(col)
                if arr.ndim != 1:
                    raise ValueError(f"per-jet column {name!r} must be 1-D, got shape {arr.shape}")
                if (name in COLUMNS and COLUMNS[name].jagged) or name.startswith(_RAGGED_PREFIXES):
                    raise TypeError(
                        f"column {name!r} holds one value per {column_group(name)}: pass a Ragged, not a 1-D array"
                    )
                col = arr
                length = len(arr)
            if n is None:
                n = length
            elif length != n:
                raise ValueError(f"column {name!r} has {length} rows, the other columns {n}")
            columns[name] = col
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "_n", 0 if n is None else n)

    # ----------------------------------------------------------------------------------------------- mapping access

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, name: str) -> Column:
        try:
            return self.columns[name]
        except KeyError:
            raise KeyError(f"no column {name!r}{_suggest(name, self.names)}") from None

    def __contains__(self, name: object) -> bool:
        return name in self.columns

    def __iter__(self) -> Iterator[str]:
        raise TypeError(
            "JetData is not iterable: use data.names for the columns, data.jet(i) for one jet and data.chunks() "
            "for blocks of jets"
        )

    @property
    def names(self) -> tuple[str, ...]:
        """The column names, in order."""
        return tuple(self.columns)

    def _group_columns(self) -> dict[str, list[str]]:
        """Return group -> names of its ragged columns (only groups present)."""
        groups: dict[str, list[str]] = {}
        for name, col in self.columns.items():
            if isinstance(col, Ragged):
                groups.setdefault(column_group(name, jagged=True), []).append(name)
        return groups

    @property
    def counts(self) -> np.ndarray:
        """Number of particles per jet (int64 [N]), from ``part_px`` (or another particle column)."""
        if "part_px" in self.columns:
            col = self.columns["part_px"]
            assert isinstance(col, Ragged)
            return col.counts()
        particle = self._group_columns().get("particle")
        if not particle:
            raise ValueError("this data set has no particle columns (part_*)")
        col = self.columns[particle[0]]
        assert isinstance(col, Ragged)
        return col.counts()

    @property
    def has_truth(self) -> bool:
        """True if any ``truth_*`` column is present."""
        return any(name.startswith("truth_") for name in self.columns)

    def _label_codes(self) -> tuple[np.ndarray, tuple[str, ...]] | None:
        """Return (class index per jet, short class names) from the JetClass label columns present, or None."""
        present = [name for name in JETCLASS_LABELS if name in self.columns]
        if not present:
            return None
        matrix = np.stack([np.asarray(self.columns[name], dtype=np.float64) for name in present], axis=1)
        if len(self) and not np.all(matrix.sum(axis=1) == 1):
            raise ValueError(f"the label columns {present} are not one-hot")
        codes = np.argmax(matrix, axis=1) if len(self) else np.zeros(0, dtype=np.int64)
        return codes, tuple(name.removeprefix("label_") for name in present)

    @property
    def process_names(self) -> np.ndarray:
        """Process of every jet (str [N]), one lower-case vocabulary for toy and JetClass data.

        Toy data: ``PROCESSES[truth_process]`` (``"qcd"``, ``"wqq"``, ``"tbqq"``, ``"q"``, ``"g"``). Without
        ``truth_process`` (JetClass, or a toy file written with ``--no-truth``): the lower-case short JetClass class
        name from the label columns (``"qcd"``, ``"hbb"``, ..., ``"wqq"``, ``"tbqq"``, ``"tbl"``).
        """
        if "truth_process" in self.columns:
            codes = np.asarray(self.columns["truth_process"], dtype=np.int64)
            return np.asarray(_TOY_PROCESSES)[codes]
        labels = self._label_codes()
        if labels is None:
            raise ValueError("cannot tell the processes: no truth_process and no label_* columns")
        codes, short = labels
        return np.asarray([s.lower() for s in short])[codes]

    # ---------------------------------------------------------------------------------------------- row selection

    def _with(self, columns: Mapping[str, Column]) -> JetData:
        """Return a copy with other columns (same config, provenance and source)."""
        return JetData(columns, config=self.config, provenance=self.provenance, source=self.source)

    def select(self, rows: ArrayLike) -> JetData:
        """Return the jets given by a bool mask [N] or an integer index array (any order, repeats allowed)."""
        index = _row_selection(rows, len(self))
        columns: dict[str, Column] = {}
        plans: dict[str, tuple[np.ndarray, np.ndarray | slice]] = {}
        span = _contiguous(index)
        for name, col in self.columns.items():
            if isinstance(col, Ragged):
                group = column_group(name, jagged=True)
                if group not in plans:
                    plans[group] = col._take(index)
                offsets, take = plans[group]
                columns[name] = Ragged(col.values[take], offsets)
            else:
                columns[name] = col[span[0] : span[1]] if span is not None else col[index]
        return self._with(columns)

    def head(self, n: int) -> JetData:
        """Return the first ``n`` jets."""
        return self.select(np.arange(min(max(int(n), 0), len(self))))

    def chunks(self, size: int = 2048) -> Iterator[JetData]:
        """Yield consecutive blocks of ``size`` jets (the last one may be shorter), in order."""
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        for start in range(0, len(self), size):
            yield self.select(np.arange(start, min(start + size, len(self))))

    # ---------------------------------------------------------------------------------------------- column changes

    def with_columns(self, **columns: Column) -> JetData:
        """Return a copy with added or replaced columns (validated like the constructor)."""
        merged = dict(self.columns)
        merged.update(columns)
        return self._with(merged)

    def drop(self, *names_or_prefixes: str) -> JetData:
        """Return a copy without the given columns; an argument ending in ``_`` drops every column with that prefix.

        Examples
        --------
        ``data.drop("truth_")`` drops every truth column; ``data.drop("jet_sdmass", "aux_")`` drops one column and
        the aux branches.
        """
        drop: set[str] = set()
        for key in names_or_prefixes:
            matched = {name for name in self.columns if name == key or (key.endswith("_") and name.startswith(key))}
            if not matched:
                raise ValueError(f"no column named or starting with {key!r}{_suggest(key, self.names)}")
            drop |= matched
        return self._with({name: col for name, col in self.columns.items() if name not in drop})

    def without_truth(self) -> JetData:
        """Return a copy without the ``truth_*`` columns (what a tagger may see)."""
        return self._with({name: col for name, col in self.columns.items() if not name.startswith("truth_")})

    # --------------------------------------------------------------------------------------------- views and arrays

    def jet(self, i: int) -> JetView:
        """Return jet ``i`` as a :class:`JetView`."""
        n = len(self)
        i = int(i)
        if i < -n or i >= n:
            raise IndexError(f"jet {i} out of range for {n} jets")
        i %= n
        groups: dict[str, dict[str, np.ndarray]] = {g: {} for g in ("particle", *GROUPS)}
        jet: dict[str, float] = {}
        for name, col in self.columns.items():
            if isinstance(col, Ragged):
                groups[column_group(name, jagged=True)][name] = col[i]
            else:
                jet[name] = col[i].item()
        return JetView(
            index=i,
            jet=jet,
            particles=groups["particle"],
            prongs=groups["prong"],
            partons=groups["parton"],
            emissions=groups["emission"],
            lund=groups["lund"],
        )

    def labels(self, classes: Sequence[str] = JETS3) -> np.ndarray:
        """Return the class index of every jet among the label columns ``classes`` (int64 [N]).

        Raises
        ------
        ValueError
            If a label column is missing or a jet does not have exactly one of the ``classes`` set (for example a
            Higgs jet of JetClass with ``classes=JETS3``).
        """
        missing = [c for c in classes if c not in self.columns]
        if missing:
            raise ValueError(f"no label column {missing[0]!r}{_suggest(missing[0], self.names)}")
        matrix = np.stack([np.asarray(self.columns[c], dtype=np.float64) for c in classes], axis=1)
        if len(self) == 0:
            return np.zeros(0, dtype=np.int64)
        ok = np.all((matrix == 0) | (matrix == 1), axis=1) & (matrix.sum(axis=1) == 1)
        if not np.all(ok):
            bad = int(np.flatnonzero(~ok)[0])
            raise ValueError(
                f"the labels {list(classes)} are not one-hot ({int((~ok).sum())} jets, first row {bad}): select the "
                "jets of these classes first"
            )
        return np.argmax(matrix, axis=1).astype(np.int64)

    def padded_p4(self, length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the particle four-momenta (px, py, pz, E) as float64 ``(N, P, 4)`` and the mask ``(N, P)`` bool.

        Padded slots are zero. ``length`` defaults to the longest jet (at least 1); longer jets are truncated.
        """
        cols = []
        for name in ("part_px", "part_py", "part_pz", "part_energy"):
            col = self[name]
            assert isinstance(col, Ragged)
            cols.append(col)
        counts = cols[0].counts()
        if length is None:
            length = max(int(counts.max()) if counts.size else 0, 1)
        p4 = np.stack([c.padded(length, dtype="float64") for c in cols], axis=-1)
        mask = np.arange(length)[None, :] < counts[:, None]
        return p4, mask

    def to_pandas(self, level: Literal["jet", "particle"] = "jet") -> pandas.DataFrame:
        """Return the per-jet (``level="jet"``) or the particle columns (``level="particle"``) as a DataFrame.

        The particle frame has a (jet, particle) MultiIndex. Needs pandas (``analysis`` extra).
        """
        import pandas as pd

        if level == "jet":
            return pd.DataFrame({n: c for n, c in self.columns.items() if not isinstance(c, Ragged)})
        if level == "particle":
            names = self._group_columns().get("particle", [])
            if not names:
                raise ValueError("this data set has no particle columns (part_*)")
            first = self.columns[names[0]]
            assert isinstance(first, Ragged)
            index = pd.MultiIndex.from_arrays([first.row_index(), first._position()], names=["jet", "particle"])
            return pd.DataFrame({n: self.columns[n].values for n in names}, index=index)  # type: ignore[union-attr]
        raise ValueError(f"unknown level {level!r}: use 'jet' or 'particle'")

    @staticmethod
    def concat(datasets: Sequence[JetData]) -> JetData:
        """Concatenate data sets with the same columns row-wise.

        The config (and the provenance) are kept if they are equal in all parts, else None. The source is that of
        the first part.
        """
        if not datasets:
            raise ValueError("nothing to concatenate")
        first = datasets[0]
        for other in datasets[1:]:
            if set(other.names) != set(first.names):
                missing = sorted(set(first.names) ^ set(other.names))
                raise ValueError(f"the data sets have different columns: {missing[:5]}")
        columns: dict[str, Column] = {}
        for name, col in first.columns.items():
            parts = [d.columns[name] for d in datasets]
            if isinstance(col, Ragged):
                if not all(isinstance(p, Ragged) for p in parts):
                    raise TypeError(f"column {name!r} is ragged in one data set and not in another")
                columns[name] = Ragged.concat(parts)  # type: ignore[arg-type]
            else:
                columns[name] = np.concatenate([np.asarray(p) for p in parts])
        config = first.config if all(d.config == first.config for d in datasets) else None
        provenance = first.provenance if all(d.provenance == first.provenance for d in datasets) else None
        return JetData(columns, config=config, provenance=provenance, source=first.source)

    def __repr__(self) -> str:
        parts: list[str] = []
        try:
            names = self.process_names
        except ValueError:
            names = None
        head = f"{_fmt(len(self))} jets"
        if names is not None and len(self):
            if "truth_process" in self.columns:
                order = [p for p in _TOY_PROCESSES if p in set(names.tolist())]
                display = {p: p for p in order}
            else:
                short = [name.removeprefix("label_") for name in JETCLASS_LABELS]
                order = [s.lower() for s in short if s.lower() in set(names.tolist())]
                display = {s.lower(): s for s in short}
                if self.source == "itj":
                    display = {p: p for p in order}
            values, counts = np.unique(names, return_counts=True)
            count = dict(zip(values.tolist(), counts.tolist(), strict=True))
            head += ": " + ", ".join(f"{display[p]} {_fmt(count[p])}" for p in order)
        parts.append(head)
        if self.source == "jetclass":
            parts.append("JetClass")
        elif self.provenance and self.provenance.get("preset"):
            parts.append(f"preset {self.provenance['preset']}")
        with contextlib.suppress(ValueError):
            parts.append(f"{_fmt(int(self.counts.sum()))} particles")
        if self.has_truth:
            parts.append("with truth")
        return f"JetData({'; '.join(parts)})"
