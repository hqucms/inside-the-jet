"""JetClass: the ten classes, the files of the CERN mirror, the pretrained models of the JetClass paper, downloads.

JetClass (Qu, Li and Qian, arXiv:2202.03772; CC BY 4.0, doi:10.5281/zenodo.6619768) has 100 M training jets in ten
classes. The course uses the per-class Parquet files of the CERN JetClassMini mirror (one file of 100 000 jets per
class and split, 86-175 MB each), byte-identical to the Zenodo files, and the pretrained ParT and ParticleNet
checkpoints of the ``particle_transformer`` repository at commit ``fa63c86``. Every file has a pinned size and MD5::

    jetgen fetch jetclass --split val --classes QCD,Wqq,Tbqq     ->  data/jetclass/val/<file>
    jetgen fetch models ParT_kin                                  ->  models/pretrained/ParT_kin.pt

A teacher can serve a copy (the ``data/jetclass`` folder written by ``jetgen fetch``) from a local share: pass its
location as ``base_url`` (``--base-url``) or in the environment variable ``ITJ_JETCLASS_URL``, for example
``file:///srv/course/jetclass`` or ``http://teacher.local/jetclass``; files are then read from
``<base_url>/<split>/<filename>`` and checked against the same sizes and MD5s.

Examples
--------
>>> [c.short for c in CLASSES][:3], JETS3_INDEX
(['QCD', 'Hbb', 'Hcc'], (0, 7, 8))
>>> MANIFEST["val", "Wqq"].filename
'WToQQ_120.parquet'
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .data import _suggest

if TYPE_CHECKING:
    from .data import JetData

__all__ = [
    "BASE_URL_ENV",
    "CHECKPOINT_URL",
    "CLASSES",
    "EXAMPLE_URL",
    "JETS3_INDEX",
    "MANIFEST",
    "MIRROR_URL",
    "MODELS",
    "PARTICLE_TRANSFORMER_COMMIT",
    "FileInfo",
    "JetClassClass",
    "class_names",
    "fetch",
    "fetch_models",
    "files",
    "load_jetclass",
    "probs_to_jets3",
]


@dataclass(frozen=True)
class JetClassClass:
    """One JetClass class.

    Attributes
    ----------
    short : str
        Short name (``"QCD"``, ``"Wqq"``, ...), as in the label branch ``label_<short>``.
    prefix : str
        File name prefix (``"ZJetsToNuNu"``, ``"WToQQ"``, ...).
    label : str
        Label branch.
    index : int
        Class index (position of the label in ``JETCLASS_LABELS``, the output of the ten-class models).
    aux_pid : int
        ``|aux_genpart_pid|`` of the resonance (0 for QCD).
    truth_match : tuple of int
        Values of ``aux_truth_match``.
    """

    short: str
    prefix: str
    label: str
    index: int
    aux_pid: int
    truth_match: tuple[int, ...]


#: The ten JetClass classes in the JetClass class order.
CLASSES: tuple[JetClassClass, ...] = (
    JetClassClass("QCD", "ZJetsToNuNu", "label_QCD", 0, 0, (50,)),
    JetClassClass("Hbb", "HToBB", "label_Hbb", 1, 25, (41,)),
    JetClassClass("Hcc", "HToCC", "label_Hcc", 2, 25, (42,)),
    JetClassClass("Hgg", "HToGG", "label_Hgg", 3, 25, (44,)),
    JetClassClass("H4q", "HToWW4Q", "label_H4q", 4, 25, (45,)),
    JetClassClass("Hqql", "HToWW2Q1L", "label_Hqql", 5, 25, (46,)),
    JetClassClass("Zqq", "ZToQQ", "label_Zqq", 6, 23, (31, 32, 33)),
    JetClassClass("Wqq", "WToQQ", "label_Wqq", 7, 24, (21, 22)),
    JetClassClass("Tbqq", "TTBar", "label_Tbqq", 8, 6, (11, 12)),
    JetClassClass("Tbl", "TTBarLep", "label_Tbl", 9, 6, (15, 16)),
)

#: Indices of QCD, Wqq and Tbqq among the ten classes (the three classes of the course).
JETS3_INDEX: tuple[int, int, int] = (0, 7, 8)


@dataclass(frozen=True)
class FileInfo:
    """A downloadable file with its pinned size and MD5.

    Attributes
    ----------
    url : str
        Download URL.
    filename : str
        File name on disk.
    size : int
        Size in bytes.
    md5 : str
        MD5 checksum (32 hexadecimal digits).
    """

    url: str
    filename: str
    size: int
    md5: str


#: The CERN JetClassMini mirror of the per-class Parquet files.
MIRROR_URL = "https://hqu.web.cern.ch/datasets/JetClassMini"

#: The JetClass example file: 100 000 jets, 10 000 per class (ROOT, tree ``tree``).
EXAMPLE_URL = "https://hqu.web.cern.ch/datasets/JetClass/example/JetClass_example_100k.root"

#: The ``particle_transformer`` commit of the pretrained checkpoints (and of the stock JetClass data configs).
PARTICLE_TRANSFORMER_COMMIT = "fa63c86d594ef875c2d25c04bd5afbab5c73b611"

#: URL pattern of the pretrained checkpoints (``{name}`` = ``ParT_kin``, ...).
CHECKPOINT_URL = (
    f"https://github.com/jet-universe/particle_transformer/raw/{PARTICLE_TRANSFORMER_COMMIT}/models/{{name}}.pt"
)

#: Environment variable that replaces the mirror for :func:`fetch` (like ``base_url``).
BASE_URL_ENV = "ITJ_JETCLASS_URL"

_MIRROR_DIRS = {"train": "train_1M_parquet", "val": "val_1M_parquet"}
_FILE_NUMBER = {"train": "000", "val": "120"}

# (prefix, split) -> (size in bytes, md5), computed on 2026-11-04 from byte-identical local copies
_SIZES: dict[tuple[str, str], tuple[int, str]] = {
    ("ZJetsToNuNu", "train"): (112059891, "06cf6f9a3a08ec621424168ab3e27bd9"),
    ("ZJetsToNuNu", "val"): (112349663, "89e0a6881109781fe0f574121e7d6ea2"),
    ("WToQQ", "train"): (103566998, "ffc05b343d23e6e38b79c5bdac0bfa82"),
    ("WToQQ", "val"): (103488757, "2e8cd393e8f9c3de2abc8bbb4db41013"),
    ("TTBar", "train"): (158491075, "9fa363de1bb91de234b1e0f12fd0241d"),
    ("TTBar", "val"): (158634524, "c9b6e734e68fe697ce425789d5c7a9ff"),
    ("HToBB", "train"): (132482611, "3ce91fa78761384017470c5310d41bc8"),
    ("HToBB", "val"): (132514332, "a1cf66d0ffcfff563a04cff1cff8472a"),
    ("HToCC", "train"): (121893521, "9db0c0dc69f730b92ce8afabc426996e"),
    ("HToCC", "val"): (121906060, "c35392421235a4bd6c6c52b7e4eb0567"),
    ("HToGG", "train"): (174220246, "1023199eab965d3225200c6d65249ea1"),
    ("HToGG", "val"): (174475774, "4f3f53d95fba8d2a4d3c75344c3f4b46"),
    ("HToWW2Q1L", "train"): (92582248, "9a962e43d45f9773203a7e0b087cd312"),
    ("HToWW2Q1L", "val"): (92626593, "609ad082b4c3660758347dacd673cfbd"),
    ("HToWW4Q", "train"): (151285903, "69664ff61d08426e12dd0329e6ed7dd0"),
    ("HToWW4Q", "val"): (151292945, "fdb27aa38021c97af7510b82e64e4992"),
    ("TTBarLep", "train"): (90891065, "d493737f5f45991822c9929ab86c344c"),
    ("TTBarLep", "val"): (90901665, "183b60b7edc5684a14c8fef08cbb6846"),
    ("ZToQQ", "train"): (108524958, "ce78d85f908be9e82ca2d660376ed140"),
    ("ZToQQ", "val"): (108370837, "d56b96841d97a6c9d379b82090fb3f68"),
}


def _manifest() -> dict[tuple[str, str], FileInfo]:
    manifest: dict[tuple[str, str], FileInfo] = {}
    for split in ("train", "val"):
        for c in CLASSES:
            filename = f"{c.prefix}_{_FILE_NUMBER[split]}.parquet"
            size, md5 = _SIZES[c.prefix, split]
            manifest[split, c.short] = FileInfo(f"{MIRROR_URL}/{_MIRROR_DIRS[split]}/{filename}", filename, size, md5)
    manifest["example", "all"] = FileInfo(
        EXAMPLE_URL, "JetClass_example_100k.root", 136175293, "6890653f40610149a9beff1497e9aa10"
    )
    return manifest


#: Every JetClass file :func:`fetch` can download, keyed by (split, short class name): ``"train"`` (the first
#: training file of each class, ``<prefix>_000.parquet``), ``"val"`` (the first validation file,
#: ``<prefix>_120.parquet``) and ``("example", "all")`` (the ROOT example file with all ten classes).
MANIFEST: dict[tuple[str, str], FileInfo] = _manifest()

#: The pretrained JetClass models (ten classes) of the ``particle_transformer`` repository at commit ``fa63c86``.
MODELS: dict[str, FileInfo] = {
    name: FileInfo(CHECKPOINT_URL.format(name=name), f"{name}.pt", size, md5)
    for name, size, md5 in (
        ("ParT_kin", 8649963, "4779564054b2272bfaf44c1a57096c1f"),
        ("ParT_kinpid", 8653035, "06a63e24892e580beb7aaf2d7206e959"),
        ("ParT_full", 8655467, "719fb783d8eeb27f609b103a82e100f6"),
        ("ParticleNet_kin", 1514546, "8fb1f78a2f90b71b3dd012d3f4387338"),
        ("ParticleNet_kinpid", 1519154, "644a97c84583e0009b4a8ed8b37ca22c"),
        ("ParticleNet_full", 1522482, "581175b6bd8655a4b7021885218db933"),
    )
}

_SPLITS = ("train", "val", "example")
_CHUNK = 1 << 20  # bytes read at a time


# ----------------------------------------------------------------------------------------------------------------------
# Classes


def class_names(classes: Sequence[str] | str) -> list[str]:
    """Return the short names of the classes given by short names or file prefixes (case-insensitive) or ``"all"``.

    Examples
    --------
    >>> class_names(["qcd", "WToQQ", "TBQQ"]), len(class_names("all"))
    (['QCD', 'Wqq', 'Tbqq'], 10)
    """
    items = [classes] if isinstance(classes, str) else list(classes)
    if any(item.lower() == "all" for item in items):
        return [c.short for c in CLASSES]
    lookup = {c.short.lower(): c.short for c in CLASSES} | {c.prefix.lower(): c.short for c in CLASSES}
    names: list[str] = []
    for item in items:
        short = lookup.get(item.lower())
        if short is None:
            choices = [c.short for c in CLASSES]
            raise ValueError(
                f"unknown JetClass class {item!r}{_suggest(item, choices)}; choose from {', '.join(choices)} or all"
            )
        if short not in names:
            names.append(short)
    if not names:
        raise ValueError("no JetClass class given")
    return names


def _check_split(split: str) -> None:
    if split not in _SPLITS:
        raise ValueError(f"unknown split {split!r}{_suggest(split, _SPLITS)}; choose from train, val, example")


def _entries(split: str, classes: Sequence[str] | str) -> list[tuple[str, FileInfo]]:
    """Return (short class name or "all", file) for a split."""
    _check_split(split)
    if split == "example":
        return [("all", MANIFEST["example", "all"])]
    return [(short, MANIFEST[split, short]) for short in class_names(classes)]


def probs_to_jets3(probs10: np.ndarray) -> np.ndarray:
    """Return the three-class probabilities (QCD, W, top) of ten-class scores: columns 0, 7, 8, renormalised.

    Examples
    --------
    >>> probs_to_jets3(np.array([[0.2, 0, 0, 0, 0, 0, 0.5, 0.2, 0.1, 0]])).round(3).tolist()
    [[0.4, 0.4, 0.2]]
    """
    probs = np.asarray(probs10, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(CLASSES):
        raise ValueError(f"expected scores of shape (N, {len(CLASSES)}), got {probs.shape}")
    three = probs[:, list(JETS3_INDEX)]
    total = three.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, three / np.where(total > 0, total, 1.0), 1.0 / 3.0)


# ----------------------------------------------------------------------------------------------------------------------
# Downloads


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches(path: Path, info: FileInfo) -> bool:
    return path.is_file() and path.stat().st_size == info.size and _md5(path) == info.md5


def _mb(size: int) -> str:
    return f"{size / 1e6:.1f} MB"


def _download(url: str, target: Path, info: FileInfo, *, progress: bool) -> None:
    """Stream ``url`` to ``<target>.part`` (resuming with a Range request), check size and MD5, rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    offset = part.stat().st_size if part.is_file() else 0
    if offset >= info.size:
        part.unlink()
        offset = 0
    request = urllib.request.Request(url, headers={"User-Agent": "itj"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code == 416 and offset:  # range not satisfiable: start again
            part.unlink()
            return _download(url, target, info, progress=progress)
        raise OSError(f"could not download {url}: HTTP {error.code} {error.reason}") from error
    except urllib.error.URLError as error:
        raise OSError(f"could not download {url}: {error.reason}") from error
    with response:
        status = getattr(response, "status", None)
        mode = "ab" if offset and status == 206 else "wb"
        done = offset if mode == "ab" else 0
        shown = -1
        # a terminal gets one updating line; a log file (a job, a redirect, a notebook) one line per 25 %
        live = sys.stderr.isatty()
        step = 1 if live else 25
        with open(part, mode) as f:
            for block in iter(lambda: response.read(_CHUNK), b""):
                f.write(block)
                done += len(block)
                if progress and info.size:
                    percent = int(100 * done / info.size) // step * step
                    if percent != shown:
                        shown = percent
                        text = f"  {target.name}: {percent:3d} % of {_mb(info.size)}"
                        if live:
                            print(f"\r{text}", end="", file=sys.stderr, flush=True)
                        else:
                            print(text, file=sys.stderr, flush=True)
        if progress and live:
            print(file=sys.stderr)
    size = part.stat().st_size
    if size != info.size:
        part.unlink()
        raise OSError(f"{url}: got {size} bytes, expected {info.size}; the download was removed, please try again")
    md5 = _md5(part)
    if md5 != info.md5:
        part.unlink()
        raise OSError(f"{url}: MD5 {md5} does not match the expected {info.md5}; the download was removed")
    os.replace(part, target)
    return None


def _fetch_files(
    items: Sequence[tuple[str, FileInfo, Path]], *, title: str, overwrite: bool, progress: bool, dry_run: bool
) -> list[Path]:
    """Download (url, info, target) items, keeping targets whose size and MD5 match."""
    total = sum(info.size for _, info, _ in items)
    print(f"{title}: {len(items)} file{'s' if len(items) != 1 else ''}, {_mb(total)}", flush=True)
    paths = []
    for url, info, target in items:
        if dry_run:
            print(f"  {url} -> {target}", flush=True)
        elif not overwrite and _matches(target, info):
            print(f"  {target} exists with the right size and MD5: kept", flush=True)
        else:
            if progress:
                print(f"  {url} -> {target}", flush=True)
            _download(url, target, info, progress=progress)
        paths.append(target)
    return paths


def fetch(
    split: Literal["train", "val", "example"] = "val",
    classes: Sequence[str] = ("QCD", "Wqq", "Tbqq"),
    *,
    dest: str | os.PathLike[str] = "data/jetclass",
    overwrite: bool = False,
    progress: bool = True,
    dry_run: bool = False,
    base_url: str | None = None,
) -> list[Path]:
    """Download JetClass files (``jetgen fetch jetclass``) to ``<dest>/<split>/<filename>``.

    The total size is printed first. Each file is streamed to ``<file>.part`` (an interrupted download resumes
    where the server allows it), checked for its size and MD5 and then renamed. An existing file with the right size
    and MD5 is kept (re-downloaded with ``overwrite``); one with another size or MD5 is downloaded again.

    Parameters
    ----------
    split : {"train", "val", "example"}
        ``"val"``: ``<prefix>_120.parquet`` (100 000 jets per class); ``"train"``: ``<prefix>_000.parquet``;
        ``"example"``: the ROOT file with 10 000 jets of each of the ten classes (``classes`` is ignored).
    classes : sequence of str
        Short names or file prefixes, case-insensitive, or ``("all",)``.
    dest : str or os.PathLike
        Root folder of the downloads.
    overwrite : bool
        Download again even if a matching file exists.
    progress : bool
        Print a progress line per file (to stderr).
    dry_run : bool
        Print the URLs and destinations only.
    base_url : str, optional
        Read the files from ``<base_url>/<split>/<filename>`` instead of the mirror (a teacher's copy of a
        ``data/jetclass`` folder, ``file://`` URLs included). Default: the environment variable
        ``ITJ_JETCLASS_URL``, else the mirror.

    Returns
    -------
    list of pathlib.Path
        The local files, in the order of ``classes``.
    """
    entries = _entries(split, classes)
    base = base_url or os.environ.get(BASE_URL_ENV) or None
    root = Path(dest) / split
    items = []
    for _, info in entries:
        url = info.url if base is None else f"{base.rstrip('/')}/{split}/{info.filename}"
        items.append((url, info, root / info.filename))
    source = "" if base is None else f" from {base}"
    return _fetch_files(
        items, title=f"JetClass {split}{source}", overwrite=overwrite, progress=progress, dry_run=dry_run
    )


def fetch_models(
    names: Sequence[str] | None = None,
    *,
    dest: str | os.PathLike[str] = "models/pretrained",
    overwrite: bool = False,
    progress: bool = True,
    dry_run: bool = False,
) -> list[Path]:
    """Download pretrained JetClass checkpoints (``jetgen fetch models``) to ``<dest>/<name>.pt``.

    Parameters
    ----------
    names : sequence of str, optional
        Keys of :data:`MODELS` (default: all six).
    dest : str or os.PathLike
        Output folder.
    overwrite, progress, dry_run : bool
        As for :func:`fetch`.

    Returns
    -------
    list of pathlib.Path
        The local checkpoint files.
    """
    selected = list(MODELS) if names is None else list(dict.fromkeys(names))
    for name in selected:
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}{_suggest(name, list(MODELS))}; choose from {', '.join(MODELS)}")
    if not selected:
        raise ValueError("no model given")
    items = [(MODELS[n].url, MODELS[n], Path(dest) / MODELS[n].filename) for n in selected]
    return _fetch_files(items, title="pretrained models", overwrite=overwrite, progress=progress, dry_run=dry_run)


# ----------------------------------------------------------------------------------------------------------------------
# Local files


def files(
    split: str, classes: Sequence[str] = ("QCD", "Wqq", "Tbqq"), *, root: str | os.PathLike[str] = "data/jetclass"
) -> list[Path]:
    """Return the local JetClass files of a split and classes (as written by :func:`fetch`).

    Raises
    ------
    FileNotFoundError
        If a file is missing; the message gives the ``jetgen fetch`` command that downloads it.
    """
    entries = _entries(split, classes)
    paths = [Path(root) / split / info.filename for _, info in entries]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        command = f"jetgen fetch jetclass --split {split}"
        if split != "example":
            command += f" --classes {','.join(short for short, _ in entries)}"
        if Path(root) != Path("data/jetclass"):
            command += f" --dest {root}"
        raise FileNotFoundError(f"{missing[0]} is missing; download it with: {command}")
    return paths


def load_jetclass(
    split: str = "val",
    classes: Sequence[str] = ("QCD", "Wqq", "Tbqq"),
    *,
    part: Literal["train", "val", "test"] | None = None,
    max_jets_per_class: int | None = None,
    columns: Sequence[str] | None = None,
    root: str | os.PathLike[str] = "data/jetclass",
) -> JetData:
    """Load downloaded JetClass jets of some classes (one file per class; ``file_index`` = position in ``classes``).

    Parameters
    ----------
    split : {"val", "train", "example"}
        Which files.
    classes : sequence of str
        Short names or prefixes, or ``("all",)``. For ``"example"`` the jets of these classes are selected from the
        one ROOT file (needs the ``analysis`` extra).
    part : {"train", "val", "test"}, optional
        The idx split of each file (:func:`itj.io.load`); for ``"example"``, whose ROOT file holds the classes in
        blocks of 10 000 jets, the same split within the jets of each class (test: the first 20 % of every class).
    max_jets_per_class : int, optional
        At most this many jets per class (the first ones of the part).
    columns : sequence of str, optional
        Columns to read (default: all 41).
    root : str or os.PathLike
        Download folder of :func:`fetch`.

    Returns
    -------
    JetData
        ``source="jetclass"``.

    Raises
    ------
    ValueError
        For ``"example"``, if none of the classes is in the file.
    """
    from .io import _part_range, load

    paths = files(split, classes, root=root)
    if split != "example":
        return load(paths, columns=columns, part=part, max_jets=max_jets_per_class)
    shorts = class_names(classes)
    wanted = None if columns is None else list(dict.fromkeys([*columns, *(c.label for c in CLASSES)]))
    data = load(paths[0], columns=wanted)
    process = data.process_names
    rows = []
    for short in shorts:
        index = np.flatnonzero(process == short.lower())
        start, stop = _part_range(len(index), part, max_jets_per_class)
        rows.append(index[start:stop])
    selected = np.sort(np.concatenate(rows))
    if len(selected) == 0:
        raise ValueError(f"no jets of the classes {', '.join(shorts)} (part {part!r}) in {paths[0]}")
    data = data.select(selected)
    extra = [] if columns is None else [n for n in data.names if n not in columns]
    return data.drop(*extra) if extra else data
