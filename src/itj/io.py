"""Reading, writing, merging and splitting jet files: Parquet for the toy, Parquet or ROOT for JetClass.

A toy file is a Parquet file with one row per jet (format version 1, ``docs/data_format.md``)::

    file.parquet
    ├── key-value metadata   itj.format_version "1", itj.config (JSON), itj.provenance (JSON)
    ├── row groups of 1000 jets, ZSTD
    ├── part_*    16 JetClass particle branches   large_list<float32 | int32 not null> not null
    ├── label_*   10 JetClass labels               float32, bool, int32 (mixed, as in JetClass)
    ├── jet_*     10 JetClass jet branches         float32
    ├── aux_*      5 JetClass aux branches         float32
    └── truth_*   toy only: per jet, per particle, per prong, per parton, per emission, per Lund emission

The JetClass columns have exactly the names, order, Arrow types and nullability of the JetClass files, so one
reader (:func:`load`) opens both, and the weaver data configs of :mod:`itj.features` run on both. Every column
carries the field metadata ``units`` and ``description`` of :data:`itj.data.COLUMNS`. Files are written to
``<name>.part`` first and renamed when complete, so an interrupted job never leaves a truncated file under the final
name.

Examples
--------
>>> import tempfile
>>> import numpy as np
>>> from itj.data import JetData, Ragged
>>> data = JetData({"part_px": Ragged.from_counts(np.ones(3, np.float32), [1, 2]),
...                 "jet_pt": np.array([1.0, 2.0], np.float32)})
>>> with tempfile.TemporaryDirectory() as tmp:
...     back = load(save(data, f"{tmp}/a.parquet"))
>>> back.names, back["part_px"].counts().tolist(), back.source
(('part_px', 'jet_pt'), [1, 2], 'itj')
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .data import COLUMNS, JETCLASS_BRANCHES, JETCLASS_LABELS, JetData, Ragged, _fmt, _suggest
from .features import split_bounds

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["FORMAT_VERSION", "METADATA_KEYS", "describe", "load", "merge", "read_metadata", "save", "write_split"]

#: Version of the file layout written by :func:`save` (the key-value metadata ``itj.format_version``).
FORMAT_VERSION = "1"

#: The Parquet key-value metadata of a toy file (JSON values).
METADATA_KEYS: tuple[str, ...] = ("itj.format_version", "itj.config", "itj.provenance")

#: The JetClass ROOT files keep their jets in this tree.
_TREE = "tree"

#: Rows per Parquet row group (one generation chunk).
_ROW_GROUP = 1000

Part = Literal["train", "val", "test"]

_PARTS: tuple[str, ...] = ("train", "val", "test")


# ----------------------------------------------------------------------------------------------------------------------
# Public API


def save(
    data: JetData,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    row_group_size: int = _ROW_GROUP,
    compression: str = "zstd",
    truth: bool = True,
) -> Path:
    """Write jets to a Parquet file.

    The JetClass columns come first, in the JetClass order and with the exact JetClass types (``large_list`` of
    ``not null`` values, every field ``not null``), then the truth columns in the order of :data:`itj.data.COLUMNS`,
    then any other column. ``file_index`` (added by :func:`load` for several files) is not written. Toy data
    (``source="itj"``) get the key-value metadata of :data:`METADATA_KEYS`; JetClass data are written without it,
    like the JetClass files themselves, so that they load again as JetClass data.

    Parameters
    ----------
    data : JetData
        The jets.
    path : str or os.PathLike
        Output file; missing parent directories are created.
    overwrite : bool
        Replace an existing file (otherwise raise :class:`FileExistsError`).
    row_group_size : int
        Rows per Parquet row group.
    compression : str
        Parquet compression codec.
    truth : bool
        Write the ``truth_*`` columns (False: what ``jetgen generate --no-truth`` writes).

    Returns
    -------
    pathlib.Path
        The path written.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (use --overwrite to replace it)")
    if row_group_size < 1:
        raise ValueError(f"row_group_size must be >= 1, got {row_group_size}")
    table = _to_table(data, truth=truth)
    _write_atomic(table, path, row_group_size=row_group_size, compression=compression)
    return path


def load(
    path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    columns: Sequence[str] | None = None,
    part: Part | None = None,
    max_jets: int | None = None,
    truth: bool = True,
) -> JetData:
    """Read jets from toy files or JetClass files (Parquet, or ROOT with the ``analysis`` extra).

    Parameters
    ----------
    path : str, os.PathLike or sequence of them
        A file, a glob pattern (``"data/jetclass/val/*.parquet"``, matches in sorted order) or several of them.
        Several files are concatenated in that order and get a per-jet int16 column ``file_index`` (0, 1, ...).
    columns : sequence of str, optional
        Read only these columns (in this order). Default: all.
    part : {"train", "val", "test"}, optional
        Read one part of the idx split of every file: with n rows in the file, test = rows ``[0, int(0.2 n))``, val
        = ``[int(0.2 n), int(0.3 n))``, train = ``[int(0.3 n), n)``. Only the row groups needed are read.
    max_jets : int, optional
        Read at most this many jets per file (the first ones, after ``part``).
    truth : bool
        Read the ``truth_*`` columns.

    Returns
    -------
    JetData
        ``source="itj"`` with the stored config and provenance for toy files; ``source="jetclass"`` (config and
        provenance None) for files without itj metadata.

    Raises
    ------
    FileNotFoundError
        If a file or a glob pattern matches nothing.
    ValueError
        For an unknown column, a toy file without ``itj.format_version`` or with a newer one.
    ImportError
        For a ROOT file without uproot and awkward (``pip install -e ".[analysis]"``).
    """
    if part is not None and part not in _PARTS:
        raise ValueError(f"part must be 'train', 'val' or 'test', got {part!r}")
    if max_jets is not None and max_jets < 0:
        raise ValueError(f"max_jets must be >= 0, got {max_jets}")
    files = _expand(path)
    datasets = [_read(f, columns=columns, part=part, max_jets=max_jets, truth=truth) for f in files]
    if len(datasets) == 1:
        return datasets[0]
    datasets = [d.with_columns(file_index=np.full(len(d), k, dtype=np.int16)) for k, d in enumerate(datasets)]
    return JetData.concat(datasets)


def read_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the metadata of a file without reading its jets.

    Returns
    -------
    dict
        ``format_version`` (str, None for JetClass files), ``config`` and ``provenance`` (dicts or None),
        ``n_jets`` (int) and ``columns`` ({name: dtype}, ``"float32"`` for one value per jet and
        ``"var * float32"`` for a ragged column).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    if path.suffix == ".root":
        tree = _open_tree(path)
        types = {name: _root_dtype(tree[name].typename) for name in tree.keys()}  # noqa: SIM118 - uproot API
        return {
            "format_version": None,
            "config": None,
            "provenance": None,
            "n_jets": int(tree.num_entries),
            "columns": types,
        }
    parquet = _open_parquet(path)
    version, config, provenance = _itj_metadata(parquet.schema_arrow, path)
    return {
        "format_version": version,
        "config": config,
        "provenance": provenance,
        "n_jets": int(parquet.metadata.num_rows),
        "columns": {field.name: _arrow_dtype(field.type) for field in parquet.schema_arrow},
    }


def merge(
    paths: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    shuffle: bool = False,
    seed: int = 0,
    renumber: bool = False,
    overwrite: bool = False,
) -> Path:
    """Concatenate toy files (for example shards generated on a cluster) into one file.

    The inputs must have the same columns and distinct seeds (equal seeds repeat the same random numbers, so the
    jets would repeat). Their ``truth_event_id`` must not overlap (give each shard its own ``--first-event-id``)
    unless ``renumber``, which numbers the jets ``0 ... N - 1`` in the order of the inputs. The configurations may
    differ: the provenance of the result lists every input as ``{"file", "seed", "n_jets", "config",
    "provenance"}`` under ``"merged"``, and the config of the result is None unless all inputs share it.

    Parameters
    ----------
    paths : sequence of str or os.PathLike
        Input files (globs allowed), in the order in which their jets are concatenated.
    out : str or os.PathLike
        Output file.
    shuffle : bool
        Shuffle the jets (with ``numpy.random.default_rng(seed)``) after concatenating (and renumbering).
    seed : int
        Seed of the shuffle.
    renumber : bool
        Renumber ``truth_event_id`` instead of requiring distinct ids.
    overwrite : bool
        Replace an existing output file.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    sources = _expand(paths)
    target = Path(out)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists (use --overwrite to replace it)")
    if target.resolve() in {p.resolve() for p in sources}:
        raise ValueError(f"the output {target} is also an input")
    datasets = [load(p) for p in sources]
    kinds = {d.source for d in datasets}
    if len(kinds) > 1:
        raise ValueError("cannot merge toy files with JetClass files")
    first = datasets[0]
    for path, data in zip(sources[1:], datasets[1:], strict=True):
        if set(data.names) != set(first.names):
            differ = sorted(set(data.names) ^ set(first.names))
            raise ValueError(
                f"{path} and {sources[0]} have different columns ({', '.join(differ[:5])}); generate all inputs "
                "with the same --no-truth option"
            )
    seeds = [s for d in datasets for s in _seeds(d.provenance)]
    repeated = sorted({s for s in seeds if s is not None and seeds.count(s) > 1}, key=str)
    if repeated:
        raise ValueError(
            f"seed(s) {repeated} occur in more than one input: those jets would repeat the same random numbers"
        )
    if "truth_event_id" in first and not renumber:
        ids = np.concatenate([np.asarray(d["truth_event_id"]) for d in datasets])
        if len(np.unique(ids)) != len(ids):
            raise ValueError(
                "the inputs share event ids; give each shard its own range with --first-event-id, or renumber the "
                "jets (renumber=True / --renumber)"
            )
    merged = JetData.concat(datasets)
    if renumber and "truth_event_id" in merged:
        merged = merged.with_columns(truth_event_id=np.arange(len(merged), dtype=np.int64))
    if shuffle:
        merged = merged.select(np.random.default_rng(seed).permutation(len(merged)))
    if first.source == "itj":
        from . import __version__

        config = first.config if all(d.config == first.config for d in datasets) else None
        presets = {(d.provenance or {}).get("preset") for d in datasets}
        provenance: dict[str, Any] = {
            "merged": [
                {
                    "file": path.name,
                    "seed": (d.provenance or {}).get("seed"),
                    "n_jets": len(d),
                    "config": d.config,
                    "provenance": d.provenance,
                }
                for path, d in zip(sources, datasets, strict=True)
            ],
            "preset": presets.pop() if len(presets) == 1 else None,
            "shuffle": shuffle,
            "shuffle_seed": seed if shuffle else None,
            "renumbered": renumber,
            "package_version": __version__,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        merged = JetData(merged.columns, config=config, provenance=provenance, source="itj")
    return save(merged, target, overwrite=overwrite)


def write_split(
    paths: Sequence[str | os.PathLike[str]],
    out_dir: str | os.PathLike[str],
    *,
    test: float = 0.2,
    val: float = 0.1,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Write the idx split of every file as ``<out_dir>/{train,val,test}/<stem>.parquet`` (``jetgen split``).

    Each output holds row for row what ``load(file, part=...)`` returns (test = the first ``int(test * n)`` rows,
    val = the next ones up to ``int((test + val) * n)``, train = the rest), with every column, the same Arrow
    schema and metadata and 1000-row groups. This is how weaver gets the split of :mod:`itj`: pass the files with
    ``--data-train``, ``--data-val`` and ``--data-test``.

    Parameters
    ----------
    paths : sequence of str or os.PathLike
        Input files (Parquet or ROOT; globs allowed). Their names (stems) must differ.
    out_dir : str or os.PathLike
        Output directory.
    test, val : float
        Fractions of the test and validation parts.
    overwrite : bool
        Replace existing outputs.

    Returns
    -------
    dict of str to list of pathlib.Path
        ``{"train": [...], "val": [...], "test": [...]}`` in the order of the inputs.
    """
    sources = _expand(paths)
    split_bounds(1, test, val)  # validates the fractions
    stems = [p.stem for p in sources]
    repeated = sorted({s for s in stems if stems.count(s) > 1})
    if repeated:
        raise ValueError(f"several inputs are called {repeated[0]!r}: the split files would overwrite each other")
    out = Path(out_dir)
    targets = {part: [out / part / f"{p.stem}.parquet" for p in sources] for part in _PARTS}
    existing = [t for part in _PARTS for t in targets[part] if t.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{existing[0]} exists (use --overwrite to replace it)")
    inputs = {p.resolve() for p in sources}
    for part in _PARTS:
        for target in targets[part]:
            if target.resolve() in inputs:
                raise ValueError(f"the output {target} is also an input")
    for k, source in enumerate(sources):
        if source.suffix == ".root":
            data = load(source)
            a, b = split_bounds(len(data), test, val)
            for part, (start, stop) in zip(_PARTS, ((b, len(data)), (a, b), (0, a)), strict=True):
                save(data.select(np.arange(start, stop)), targets[part][k], overwrite=True)
            continue
        parquet = _open_parquet(source)
        _itj_metadata(parquet.schema_arrow, source)  # refuses files of a newer format
        n = int(parquet.metadata.num_rows)
        a, b = split_bounds(n, test, val)
        for part, (start, stop) in zip(_PARTS, ((b, n), (a, b), (0, a)), strict=True):
            table = _read_rows(parquet, start, stop, None)
            _write_atomic(table, targets[part][k], row_group_size=_ROW_GROUP, compression="zstd")
    return targets


def describe(path: str | os.PathLike[str]) -> str:
    """Return a human-readable summary of a file (the text of ``jetgen info``).

    Jets per process (toy) or per class (JetClass) with their mean multiplicity, the preset, seed, wall time and
    package version of a toy file, and the column count.
    """
    path = Path(path)
    info = read_metadata(path)
    names = list(info["columns"])
    n = info["n_jets"]
    lines = [f"file: {path}  ({path.stat().st_size / 1e6:.2f} MB)"]
    n_jetclass = sum(name in JETCLASS_BRANCHES for name in names)
    n_truth = sum(name.startswith("truth_") for name in names)
    n_other = len(names) - n_jetclass - n_truth
    counts = f"{len(names)} columns ({n_jetclass} JetClass, {n_truth} truth" + (
        f", {n_other} other)" if n_other else ")"
    )
    if info["format_version"] is None:
        lines.append(f"JetClass file (no itj metadata): {_fmt(n)} jets, {counts}")
    else:
        lines.append(f"toy file, format version {info['format_version']}: {_fmt(n)} jets, {counts}")
    provenance = info["provenance"] or {}
    if info["format_version"] is not None:
        lines.extend(_provenance_lines(provenance))
    wanted = [c for c in ("truth_process", *JETCLASS_LABELS, "part_px") if c in names]
    if n and wanted:
        data = load(path, columns=wanted)
        try:
            process = data.process_names
        except ValueError:
            process = None
        if process is not None:
            multiplicity = data.counts if "part_px" in data else None
            toy = "truth_process" in data
            lines.append("")
            lines.append(f"{'process' if toy else 'class':<10s} {'jets':>9s} {'<particles>':>12s}")
            short = {label.removeprefix("label_").lower(): label.removeprefix("label_") for label in JETCLASS_LABELS}
            order = list(dict.fromkeys(process.tolist()))
            if not toy:
                order = [s for s in short if s in order]
            for name in order:
                rows = process == name
                mean = f"{multiplicity[rows].mean():12.1f}" if multiplicity is not None else f"{'-':>12s}"
                lines.append(f"{name if toy else short[name]:<10s} {_fmt(int(rows.sum())):>9s} {mean}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------------
# Paths


def _expand(path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]]) -> list[Path]:
    """Return the files named by a path, a glob pattern or a sequence of them (glob matches sorted)."""
    items = [path] if isinstance(path, (str, os.PathLike)) else list(path)
    files: list[Path] = []
    for item in items:
        text = os.fspath(item)
        if any(char in text for char in "*?["):
            matches = sorted(glob.glob(text))
            if not matches:
                raise FileNotFoundError(f"no file matches {text!r}")
            files.extend(Path(m) for m in matches)
        else:
            file = Path(text)
            if not file.is_file():
                raise FileNotFoundError(f"{file} does not exist")
            files.append(file)
    if not files:
        raise ValueError("no input files")
    return files


def _part_range(n: int, part: Part | None, max_jets: int | None) -> tuple[int, int]:
    """Return the rows [start, stop) of ``part`` (whole file if None), truncated to ``max_jets``."""
    start, stop = 0, n
    if part is not None:
        a, b = split_bounds(n)
        start, stop = {"test": (0, a), "val": (a, b), "train": (b, n)}[part]
    if max_jets is not None:
        stop = min(stop, start + max_jets)
    return start, stop


# ----------------------------------------------------------------------------------------------------------------------
# Writing


def _arrow_type(dtype: np.dtype) -> pa.DataType:
    import pyarrow as pa

    return pa.from_numpy_dtype(dtype)


def _ordered_names(data: JetData, truth: bool) -> list[str]:
    """JetClass columns in JetClass order, then truth columns in COLUMNS order, then the rest; no file_index."""
    names = [n for n in data.names if n != "file_index" and (truth or not n.startswith("truth_"))]
    present = set(names)
    jetclass = [n for n in JETCLASS_BRANCHES if n in present]
    known_truth = [n for n in COLUMNS if n.startswith("truth_") and n in present]
    placed = set(jetclass) | set(known_truth)
    return jetclass + known_truth + [n for n in names if n not in placed]


def _to_table(data: JetData, *, truth: bool) -> pa.Table:
    """Convert jets into an Arrow table with the JetClass types and the itj metadata."""
    import pyarrow as pa

    arrays = []
    fields = []
    for name in _ordered_names(data, truth):
        col = data[name]
        info = COLUMNS.get(name)
        meta = {
            b"units": (info.units if info else "").encode(),
            b"description": (info.description if info else "").encode(),
        }
        if isinstance(col, Ragged):
            if col.values.ndim != 1:
                raise TypeError(f"column {name!r} has values of shape {col.values.shape}: only 1-D values can be saved")
            values = np.ascontiguousarray(col.values, dtype=info.dtype if info else None)
            item = pa.field("item", _arrow_type(values.dtype), nullable=False)
            array = pa.LargeListArray.from_arrays(
                pa.array(col.offsets, type=pa.int64()), pa.array(values), type=pa.large_list(item)
            )
        else:
            values = np.ascontiguousarray(col, dtype=info.dtype if info else None)
            array = pa.array(values)
        arrays.append(array)
        fields.append(pa.field(name, array.type, nullable=False, metadata=meta))
    metadata = None
    if data.source == "itj":
        metadata = {
            "itj.format_version": FORMAT_VERSION,
            "itj.config": json.dumps(data.config, default=str),
            "itj.provenance": json.dumps(data.provenance, default=str),
        }
    schema = pa.schema(fields, metadata=metadata)
    if not arrays:
        return schema.empty_table()
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_atomic(table: pa.Table, path: Path, *, row_group_size: int, compression: str) -> None:
    """Write ``table`` to ``<path>.part`` and rename it to ``path`` when complete."""
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        # use_compliant_nested_type=False keeps the list item name "item" of the JetClass files (not "element")
        pq.write_table(
            table, part, row_group_size=row_group_size, compression=compression, use_compliant_nested_type=False
        )
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


# ----------------------------------------------------------------------------------------------------------------------
# Reading


def _open_parquet(path: Path) -> Any:
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        return pq.ParquetFile(path)
    except (pa.ArrowInvalid, OSError) as error:
        if isinstance(error, FileNotFoundError):
            raise
        raise ValueError(f"{path} is not a readable Parquet file ({error})") from error


def _itj_metadata(schema: pa.Schema, path: Path) -> tuple[str | None, dict | None, dict | None]:
    """Return (format version, config, provenance) of a toy file, or (None, None, None) for a JetClass file."""
    metadata = schema.metadata or {}
    if not any(key.startswith(b"itj.") for key in metadata):
        return None, None, None
    raw = metadata.get(b"itj.format_version")
    if raw is None:
        raise ValueError(f"{path} has itj metadata but no itj.format_version: it was not written by itj.io.save")
    version = raw.decode()
    if not version.isdigit() or int(version) > int(FORMAT_VERSION):
        raise ValueError(
            f"{path} has format version {version}; this version of itj reads up to {FORMAT_VERSION} (upgrade itj)"
        )

    def decode(key: bytes) -> dict | None:
        value = metadata.get(key)
        return None if value is None else json.loads(value)

    return version, decode(b"itj.config"), decode(b"itj.provenance")


def _select_names(available: Sequence[str], columns: Sequence[str] | None, truth: bool, path: Path) -> list[str]:
    if columns is None:
        names = list(available)
    else:
        names = list(dict.fromkeys(columns))
        for name in names:
            if name not in available:
                raise ValueError(f"{path} has no column {name!r}{_suggest(name, available)}")
    return [n for n in names if truth or not n.startswith("truth_")]


def _read_rows(parquet: Any, start: int, stop: int, names: Sequence[str] | None) -> pa.Table:
    """Read rows [start, stop) of a Parquet file, reading only the row groups that hold them."""
    metadata = parquet.metadata
    schema = parquet.schema_arrow
    groups = []
    first_row = None
    row = 0
    for g in range(metadata.num_row_groups):
        size = metadata.row_group(g).num_rows
        if row < stop and row + size > start and size > 0:
            groups.append(g)
            if first_row is None:
                first_row = row
        row += size
    if stop <= start or not groups:
        empty = schema.empty_table()
        return empty if names is None else empty.select(list(names))
    table = parquet.read_row_groups(groups, columns=None if names is None else list(names))
    assert first_row is not None
    table = table.slice(start - first_row, stop - start)
    # keep the schema metadata (read_row_groups keeps it; be explicit for the column subset)
    return table.replace_schema_metadata(schema.metadata)


def _from_arrow(array: Any) -> np.ndarray | Ragged:
    """Convert one Arrow column (chunked or not) into a NumPy array or a Ragged."""
    import pyarrow as pa

    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks() if array.num_chunks != 1 else array.chunk(0)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = np.asarray(array.offsets.to_numpy(), dtype=np.int64)
        values = array.flatten().to_numpy(zero_copy_only=False)
        return Ragged(values, offsets - offsets[0])
    return array.to_numpy(zero_copy_only=False)


def _read(
    path: Path, *, columns: Sequence[str] | None, part: Part | None, max_jets: int | None, truth: bool
) -> JetData:
    """Read one file."""
    if path.suffix == ".root":
        return _read_root(path, columns=columns, part=part, max_jets=max_jets, truth=truth)
    parquet = _open_parquet(path)
    schema = parquet.schema_arrow
    version, config, provenance = _itj_metadata(schema, path)
    names = _select_names(schema.names, columns, truth, path)
    start, stop = _part_range(int(parquet.metadata.num_rows), part, max_jets)
    table = _read_rows(parquet, start, stop, names)
    cols = {name: _from_arrow(table.column(name)) for name in names}
    if version is None:
        return JetData(cols, source="jetclass")
    return JetData(cols, config=config, provenance=provenance, source="itj")


def _open_tree(path: Path) -> Any:
    try:
        import uproot
    except ImportError as error:
        raise ImportError(
            f'reading the ROOT file {path} needs uproot and awkward: pip install -e ".[analysis]"'
        ) from error
    file = uproot.open(path)
    if _TREE not in file:
        raise ValueError(f"{path} has no tree {_TREE!r} (JetClass ROOT files keep their jets in it)")
    return file[_TREE]


_ROOT_TYPES = {
    "float": "float32",
    "double": "float64",
    "bool": "bool",
    "int32_t": "int32",
    "int64_t": "int64",
    "int16_t": "int16",
    "int8_t": "int8",
    "int": "int32",
}


def _root_dtype(typename: str) -> str:
    if typename.startswith("std::vector<") and typename.endswith(">"):
        inner = typename[len("std::vector<") : -1]
        return f"var * {_ROOT_TYPES.get(inner, inner)}"
    return _ROOT_TYPES.get(typename, typename)


def _read_root(
    path: Path, *, columns: Sequence[str] | None, part: Part | None, max_jets: int | None, truth: bool
) -> JetData:
    """Read a JetClass ROOT file (tree ``tree``) with uproot."""
    tree = _open_tree(path)
    import awkward as ak

    names = _select_names(list(tree.keys()), columns, truth, path)
    start, stop = _part_range(int(tree.num_entries), part, max_jets)
    arrays = tree.arrays(names, entry_start=start, entry_stop=stop, library="ak")
    cols: dict[str, np.ndarray | Ragged] = {}
    for name in names:
        array = arrays[name]
        dtype = COLUMNS[name].dtype if name in COLUMNS else None
        if array.ndim > 1:
            values = np.asarray(ak.to_numpy(ak.flatten(array)), dtype=dtype)
            cols[name] = Ragged.from_counts(values, np.asarray(ak.num(array), dtype=np.int64))
        else:
            cols[name] = np.asarray(ak.to_numpy(array), dtype=dtype)
    return JetData(cols, source="jetclass")


def _arrow_dtype(type_: Any) -> str:
    """Return ``"float32"`` for a scalar Arrow type and ``"var * float32"`` for a list type."""
    import pyarrow as pa

    if pa.types.is_list(type_) or pa.types.is_large_list(type_):
        return f"var * {_arrow_dtype(type_.value_type)}"
    try:
        return np.dtype(type_.to_pandas_dtype()).name
    except (NotImplementedError, TypeError):
        return str(type_)


# ----------------------------------------------------------------------------------------------------------------------
# Provenance


def _seeds(provenance: dict | None) -> Iterator[Any]:
    """Yield the generation seeds of a file, following merged inputs."""
    if not provenance:
        return
    if "merged" in provenance:
        for entry in provenance["merged"]:
            if entry.get("provenance"):
                yield from _seeds(entry["provenance"])
            else:
                yield entry.get("seed")
    else:
        yield provenance.get("seed")


def _provenance_lines(provenance: dict[str, Any]) -> list[str]:
    lines = []
    if "merged" in provenance:
        merged = provenance["merged"]
        lines.append(f"merged from {len(merged)} files: {', '.join(str(m.get('file')) for m in merged)}")
        lines.append(f"seeds: {list(_seeds(provenance))}")
    preset = provenance.get("preset")
    lines.append(f"preset: {preset if preset else 'modified or mixed'}")
    if "seed" in provenance:
        lines.append(f"seed: {provenance['seed']}")
    wall = provenance.get("wall_time_s")
    if wall is not None:
        workers = provenance.get("workers")
        plural = "" if workers == 1 else "s"
        lines.append(f"wall time: {float(wall):.1f} s" + (f" ({workers} worker{plural})" if workers else ""))
    for key in ("package_version", "created", "command_line"):
        if provenance.get(key) is not None:
            lines.append(f"{key.replace('_', ' ')}: {provenance[key]}")
    return lines
