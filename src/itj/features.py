"""Network inputs: the per-particle features of weaver, jet-level features, labels, splits and torch batches.

The particle features are computed exactly as weaver 0.5.4 computes them from the data configs in
``itj/configs`` (``jets3_<set>.yaml``, ``JetClass_<set>.yaml``): the ``new_variables`` in float32, then
``(x - center) * scale`` clipped to ``[clip_min, clip_max]`` for the standardised variables, NaN set to 0 and
``wrap`` padding (each jet repeats its own particles cyclically) for points, features and vectors::

    pf_points    (N, 2, L)    part_deta, part_dphi
    pf_features  (N, F, L)    FEATURE_SETS[feature_set]  (F = 7 kin, 13 kinpid, 17 full)
    pf_vectors   (N, 4, L)    part_px, part_py, part_pz, part_energy
    pf_mask      (N, 1, L)    1 for a particle, 0 for padding

For training, :class:`JetDataset` keeps the transformed features ragged and pads every batch to its longest jet;
:func:`make_loader` iterates over it in :class:`JetBatch` objects (torch tensors, channel-first like weaver).

Examples
--------
>>> import numpy as np
>>> from itj.data import JetData, Ragged
>>> px = Ragged.from_counts(np.array([300.0, 200.0, 100.0], dtype=np.float32), [2, 1])
>>> zero = px.map(np.zeros_like)
>>> data = JetData({"part_px": px, "part_py": zero, "part_pz": zero, "part_energy": px, "part_deta": zero,
...                 "part_dphi": zero, "jet_pt": np.array([500.0, 100.0], dtype=np.float32),
...                 "jet_energy": np.array([500.0, 100.0], dtype=np.float32)})
>>> x = inputs(data, "kin", length=3)
>>> x["pf_features"].shape, x["pf_mask"][:, 0].tolist()
((2, 7, 3), [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .data import JETCLASS_LABELS, JETS3, JetData, Ragged, _suggest

if TYPE_CHECKING:
    import torch

__all__ = [
    "CONFIG_NAMES",
    "FEATURE_SETS",
    "JET_OBSERVERS",
    "LABEL_SETS",
    "MAX_LENGTH",
    "FeatureVar",
    "JetBatch",
    "JetDataset",
    "config_path",
    "inputs",
    "jet_features",
    "jetclass_selection",
    "labels",
    "make_batch",
    "make_loader",
    "particle_variables",
    "split",
    "split_bounds",
]


@dataclass(frozen=True)
class FeatureVar:
    """One input variable of a weaver data config: ``[name, center, scale, clip_min, clip_max]``.

    Attributes
    ----------
    name : str
        Variable name (a ``part_*`` branch or a weaver ``new_variable``).
    center : float, optional
        Subtracted before scaling; None means the raw value without clipping (weaver ``manual`` semantics).
    scale : float
        Multiplies the centred value.
    clip_min, clip_max : float
        Clipping range of the standardised value.
    """

    name: str
    center: float | None = None
    scale: float = 1.0
    clip_min: float = -5.0
    clip_max: float = 5.0


_KIN_HEAD = (
    FeatureVar("part_pt_log", 1.7, 0.7),
    FeatureVar("part_e_log", 2.0, 0.7),
    FeatureVar("part_logptrel", -4.7, 0.7),
    FeatureVar("part_logerel", -4.7, 0.7),
    FeatureVar("part_deltaR", 0.2, 4.0),
)
_ANGLES = (FeatureVar("part_deta"), FeatureVar("part_dphi"))
_PID = tuple(
    FeatureVar(name)
    for name in (
        "part_charge",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
    )
)
_TRACK = (
    FeatureVar("part_d0"),
    FeatureVar("part_d0err", 0.0, 1.0, 0.0, 1.0),
    FeatureVar("part_dz"),
    FeatureVar("part_dzerr", 0.0, 1.0, 0.0, 1.0),
)

#: The ``pf_features`` of the three weaver configs of the ParT paper, in the order of the YAML files.
FEATURE_SETS: dict[str, tuple[FeatureVar, ...]] = {
    "kin": _KIN_HEAD + _ANGLES,
    "kinpid": _KIN_HEAD + _PID + _ANGLES,
    "full": _KIN_HEAD + _PID + _TRACK + _ANGLES,
}

_POINTS = ("part_deta", "part_dphi")
_VECTORS = ("part_px", "part_py", "part_pz", "part_energy")

#: The ``observers`` of the weaver configs: the JetClass jet branches.
JET_OBSERVERS: tuple[str, ...] = (
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

#: Label sets of :func:`labels`: label columns (class = index of the column set to 1) or ``"truth_init_pid"`` for
#: quark (0) vs gluon (1).
LABEL_SETS: dict[str, tuple[str, ...] | str] = {"jets3": JETS3, "jetclass10": JETCLASS_LABELS, "qg": "truth_init_pid"}

#: Padded length of the weaver configs (and the default maximum of :class:`JetDataset`).
MAX_LENGTH = 128

#: The packaged weaver data configs (``itj/configs/<name>.yaml``).
CONFIG_NAMES: tuple[str, ...] = (
    "jets3_kin",
    "jets3_kinpid",
    "jets3_full",
    "JetClass_kin",
    "JetClass_kinpid",
    "JetClass_full",
)

#: Derived jet features of :func:`jet_features`: name -> (numerator, denominator).
_DERIVED: dict[str, tuple[str, str]] = {
    "tau21": ("jet_tau2", "jet_tau1"),
    "tau32": ("jet_tau3", "jet_tau2"),
    "tau43": ("jet_tau4", "jet_tau3"),
}


def _check_name(kind: str, name: str, choices: Iterable[str]) -> None:
    choices = list(choices)
    if name not in choices:
        raise ValueError(f"unknown {kind} {name!r}{_suggest(name, choices)}; choose from {', '.join(choices)}")


def _ragged(data: JetData, name: str) -> Ragged:
    col = data[name]
    if not isinstance(col, Ragged):
        raise TypeError(f"column {name!r} is not a particle column")
    return col


# ------------------------------------------------------------------------------------------- particle features


def particle_variables(data: JetData) -> dict[str, Ragged]:
    """Return the weaver ``new_variables`` of the data configs, computed in float32 as weaver does.

    ``part_mask`` (1), ``part_pt = hypot(px, py)``, ``part_pt_log``, ``part_e_log``, ``part_logptrel =
    log(part_pt / jet_pt)``, ``part_logerel = log(part_energy / jet_energy)``, ``part_deltaR = hypot(deta, dphi)``
    and, when the impact parameters are present, ``part_d0 = tanh(part_d0val)``, ``part_dz = tanh(part_dzval)``.

    Parameters
    ----------
    data : JetData
        Needs ``part_px``, ``part_py``, ``part_energy``, ``part_deta``, ``part_dphi``, ``jet_pt``, ``jet_energy``.

    Returns
    -------
    dict of str to Ragged
        float32 columns sharing the particle offsets.
    """
    px = _ragged(data, "part_px")
    offsets = px.offsets
    counts = px.counts()

    def f32(name: str) -> np.ndarray:
        return np.asarray(_ragged(data, name).values, dtype=np.float32)

    def per_particle(name: str) -> np.ndarray:
        return np.repeat(np.asarray(data[name], dtype=np.float32), counts)

    energy = f32("part_energy")
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["part_mask"] = np.ones_like(energy)
        out["part_pt"] = np.hypot(f32("part_px"), f32("part_py"))
        out["part_pt_log"] = np.log(out["part_pt"])
        out["part_e_log"] = np.log(energy)
        out["part_logptrel"] = np.log(out["part_pt"] / per_particle("jet_pt"))
        out["part_logerel"] = np.log(energy / per_particle("jet_energy"))
        out["part_deltaR"] = np.hypot(f32("part_deta"), f32("part_dphi"))
        if "part_d0val" in data:
            out["part_d0"] = np.tanh(f32("part_d0val"))
        if "part_dzval" in data:
            out["part_dz"] = np.tanh(f32("part_dzval"))
    return {name: Ragged(values, offsets) for name, values in out.items()}


def _transform(var: FeatureVar, values: np.ndarray) -> np.ndarray:
    """Standardise and clip one variable as weaver does (float32; NaN -> 0; raw variables are not clipped)."""
    x = np.asarray(values, dtype=np.float32)
    if var.center is not None:
        x = (x - np.float32(var.center)) * np.float32(var.scale)
        x = np.clip(x, np.float32(var.clip_min), np.float32(var.clip_max))
    return np.where(np.isnan(x), np.float32(0.0), x)


def _channels(data: JetData, feature_set: str | None) -> tuple[list[FeatureVar], Ragged]:
    """Return the variables (points, features, vectors) and their transformed values as one Ragged (n, C)."""
    variables = [FeatureVar(name) for name in _POINTS]
    if feature_set is not None:
        _check_name("feature set", feature_set, FEATURE_SETS)
        variables += list(FEATURE_SETS[feature_set])
    variables += [FeatureVar(name) for name in _VECTORS]
    if feature_set is None and "part_px" not in data:  # jet-level data only: no particles at all
        return variables, Ragged(np.zeros((0, len(variables)), dtype=np.float32), np.zeros(len(data) + 1, np.int64))
    derived = particle_variables(data) if feature_set is not None else {}
    columns = []
    for var in variables:
        source = derived.get(var.name)
        if source is None:
            if var.name not in data:
                missing = {"part_d0": "part_d0val", "part_dz": "part_dzval"}.get(var.name, var.name)
                raise ValueError(f"feature set {feature_set!r} needs the column {missing!r}, which is missing")
            source = _ragged(data, var.name)
        columns.append(_transform(var, source.values))
    px = _ragged(data, "part_px")
    return variables, Ragged(np.stack(columns, axis=1), px.offsets)


def _pad_channels(table: Ragged, length: int, n_features: int) -> dict[str, np.ndarray]:
    """Wrap-pad the (n, C) channel table to (N, C, L) and split it into points, features, vectors and mask."""
    padded = np.ascontiguousarray(table.padded(length, mode="wrap", dtype="float32").transpose(0, 2, 1))
    counts = table.counts()
    mask = (np.arange(length)[None, :] < counts[:, None]).astype(np.float32)[:, None, :]
    return {
        "pf_points": padded[:, :2],
        "pf_features": padded[:, 2 : 2 + n_features],
        "pf_vectors": padded[:, 2 + n_features :],
        "pf_mask": mask,
    }


def _default_length(counts: np.ndarray, maximum: int = MAX_LENGTH) -> int:
    return max(min(int(counts.max()) if counts.size else 0, maximum), 1)


def inputs(data: JetData, feature_set: str = "kin", *, length: int | None = MAX_LENGTH) -> dict[str, np.ndarray]:
    """Return the network inputs of weaver 0.5.4 for the data config ``jets3_<feature_set>.yaml``.

    Parameters
    ----------
    data : JetData
        Jets with the JetClass particle and jet branches.
    feature_set : {"kin", "kinpid", "full"}
        The ``pf_features`` of :data:`FEATURE_SETS`.
    length : int, optional
        Padded length L (weaver: 128). None: the longest jet, at most 128.

    Returns
    -------
    dict of str to numpy.ndarray
        ``pf_points`` (N, 2, L), ``pf_features`` (N, F, L), ``pf_vectors`` (N, 4, L), ``pf_mask`` (N, 1, L), float32,
        identical to the arrays weaver builds from the same file. Peak memory about 2 N L (6 + F) * 4 bytes.
    """
    _check_name("feature set", feature_set, FEATURE_SETS)
    _, table = _channels(data, feature_set)
    if length is None:
        length = _default_length(table.counts())
    return _pad_channels(table, length, len(FEATURE_SETS[feature_set]))


# ------------------------------------------------------------------------------------------ jets, labels, splits


def jet_features(
    data: JetData, names: Sequence[str] = JET_OBSERVERS, *, derived: Sequence[str] = ("tau21", "tau32")
) -> tuple[np.ndarray, list[str]]:
    """Return jet-level features as a float32 matrix (N, F) and their names.

    Parameters
    ----------
    data : JetData
        Jets.
    names : sequence of str
        Per-jet columns (any of them may also be a derived name).
    derived : sequence of str
        Ratios appended after ``names``: ``tau21 = tau2 / tau1``, ``tau32 = tau3 / tau2``, ``tau43 = tau4 / tau3``,
        0 where the denominator is 0.

    Returns
    -------
    tuple of numpy.ndarray and list of str
        The features (N, F) and their names.
    """
    all_names = list(names) + [d for d in derived if d not in names]
    columns = []
    for name in all_names:
        if name in _DERIVED:
            num_name, den_name = _DERIVED[name]
            num = np.asarray(data[num_name], dtype=np.float64)
            den = np.asarray(data[den_name], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                columns.append(np.where(den != 0, num / np.where(den != 0, den, 1.0), 0.0))
        else:
            col = data[name]
            if isinstance(col, Ragged):
                raise ValueError(f"{name!r} is not a per-jet column")
            columns.append(np.asarray(col, dtype=np.float64))
    if not columns:
        return np.zeros((len(data), 0), dtype=np.float32), []
    return np.stack(columns, axis=1).astype(np.float32), all_names


def labels(data: JetData, label_set: str = "jets3") -> np.ndarray:
    """Return the class of every jet (int64 [N]) for a label set of :data:`LABEL_SETS`.

    ``"jets3"``: 0 QCD, 1 W, 2 top; ``"jetclass10"``: the JetClass class index; ``"qg"``: 0 quark, 1 gluon (from
    ``truth_init_pid``; toy q/g files only).
    """
    _check_name("label set", label_set, LABEL_SETS)
    spec = LABEL_SETS[label_set]
    if isinstance(spec, str):
        if spec not in data:
            raise ValueError(f"label set {label_set!r} needs the truth column {spec!r} (toy files with truth only)")
        pid = np.asarray(data[spec])
        if not np.all((pid == 1) | (pid == 21)):
            raise ValueError(f"label set {label_set!r} needs quark and gluon jets only (truth_init_pid 1 or 21)")
        return (pid == 21).astype(np.int64)
    return data.labels(spec)


def split_bounds(n: int, test: float = 0.2, val: float = 0.1) -> tuple[int, int]:
    """Return the boundaries of the idx split of a file with ``n`` rows: test ``[0, a)``, val ``[a, b)``, train rest.

    ``a = int(test * n)`` and ``b = int((test + val) * n)``, with ``test + val`` rounded to 12 digits so that the
    default gives exactly ``int(0.3 * n)``.

    Examples
    --------
    >>> split_bounds(1000), split_bounds(7)
    ((200, 300), (1, 2))
    """
    if not (test >= 0 and val >= 0 and test + val <= 1):
        raise ValueError(f"need 0 <= test, 0 <= val and test + val <= 1, got test={test}, val={val}")
    return int(test * n), int(round(test + val, 12) * n)


def split(data: JetData, *, test: float = 0.2, val: float = 0.1) -> tuple[JetData, JetData, JetData]:
    """Split jets into (train, val, test) by the idx rule of the course, per file when ``file_index`` is present.

    In each file with n jets the first ``int(test * n)`` jets are the test set, the next ones up to
    ``int((test + val) * n)`` the validation set and the rest the training set; :func:`itj.io.load` with ``part=``
    and ``jetgen split`` give the same rows.
    """
    n = len(data)
    if "file_index" in data:
        file_index = np.asarray(data["file_index"])
        groups = [np.flatnonzero(file_index == f) for f in dict.fromkeys(file_index.tolist())]
    else:
        groups = [np.arange(n)]
    parts: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for rows in groups:
        a, b = split_bounds(len(rows), test, val)
        parts["test"].append(rows[:a])
        parts["val"].append(rows[a:b])
        parts["train"].append(rows[b:])
    train, val_rows, test_rows = (
        np.concatenate(parts[k]) if parts[k] else np.zeros(0, dtype=np.int64) for k in ("train", "val", "test")
    )
    return data.select(train), data.select(val_rows), data.select(test_rows)


def jetclass_selection(data: JetData, *, pt_min: float = 500.0, pt_max: float = 1000.0) -> np.ndarray:
    """Return the JetClass reco selection ``pt_min < jet_pt < pt_max`` (bool [N]) for analysis-time use on toy files."""
    pt = np.asarray(data["jet_pt"])
    return (pt > pt_min) & (pt < pt_max)


def config_path(name: str) -> Path:
    """Return the path of a packaged weaver data config (``"jets3_kin"``, ..., ``"JetClass_full"``).

    ``jets3_<set>.yaml`` has the three classes of the course and no selection (run it on the files of ``jetgen
    split``); ``JetClass_<set>.yaml`` is the stock ten-class config of the JetClass paper (``particle_transformer``
    commit ``fa63c86``), the one the pretrained models were trained with. Both have the features of
    :data:`FEATURE_SETS` in the same order.

    Examples
    --------
    >>> config_path("jets3_kin").name
    'jets3_kin.yaml'
    """
    _check_name("config", name, CONFIG_NAMES)
    path = Path(str(importlib.resources.files("itj") / "configs" / f"{name}.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"the packaged config {path} is missing (broken installation?)")
    return path


# ------------------------------------------------------------------------------------------------ torch batches


@dataclass
class JetBatch:
    """A batch of jets as torch tensors, channel-first like weaver (P = padded length of this batch).

    Attributes
    ----------
    points : torch.Tensor
        (B, 2, P) ``part_deta``, ``part_dphi``.
    features : torch.Tensor
        (B, F, P) the transformed ``pf_features`` (F = 0 without a feature set).
    vectors : torch.Tensor
        (B, 4, P) px, py, pz, E, wrap-padded.
    mask : torch.Tensor
        (B, 1, P) float, 1 for a particle and 0 for padding.
    label : torch.Tensor, optional
        (B,) int64 class.
    weight : torch.Tensor, optional
        (B,) float32 per-jet weight.
    index : torch.Tensor
        (B,) int64 row of each jet in the source :class:`~itj.data.JetData`.
    jet : torch.Tensor, optional
        (B, J) float32 jet-level features (the MLP input).
    feature_set : str, optional
        Name of the feature set of ``features``.
    """

    points: torch.Tensor
    features: torch.Tensor
    vectors: torch.Tensor
    mask: torch.Tensor
    label: torch.Tensor | None
    weight: torch.Tensor | None
    index: torch.Tensor
    jet: torch.Tensor | None
    feature_set: str | None

    def to(self, device: Any) -> JetBatch:
        """Return the batch on ``device`` (a ``torch.device`` or a string such as ``"cpu"``)."""

        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device)

        return JetBatch(
            points=self.points.to(device),
            features=self.features.to(device),
            vectors=self.vectors.to(device),
            mask=self.mask.to(device),
            label=move(self.label),
            weight=move(self.weight),
            index=self.index.to(device),
            jet=move(self.jet),
            feature_set=self.feature_set,
        )

    def __len__(self) -> int:
        return int(self.index.shape[0])

    @property
    def particle_mask(self) -> torch.Tensor:
        """(B, P) bool, True for a particle."""
        return self.mask[:, 0, :] > 0


class JetDataset:
    """Jets prepared for training: the transformed per-particle features kept ragged, padded per batch.

    Memory: about N <n> (6 + F) * 4 bytes for N jets with <n> particles on average (150 000 toy jets with the kin
    features: about 0.3 GB).

    Parameters
    ----------
    data : JetData
        Jets.
    feature_set : {"kin", "kinpid", "full"} or None
        Particle features; None gives ``features`` with 0 channels (points, vectors and mask are always there).
    label_set : str or None
        Label set of :func:`labels`; None gives batches without labels.
    weights : numpy.ndarray, optional
        Per-jet weights [N].
    length : int
        Maximum padded length; longer jets are truncated (weaver: 128).
    jet_feature_names : sequence of str, optional
        Jet-level features of :func:`jet_features` (with no derived ratios unless named) for ``JetBatch.jet``.

    Attributes
    ----------
    data : JetData
        The jets.
    feature_set, label_set : str or None
        As given.
    feature_names : tuple of str
        The names of the ``features`` channels.
    jet_feature_names : tuple of str
        The names of the ``jet`` columns.
    """

    def __init__(
        self,
        data: JetData,
        feature_set: str | None = "kin",
        label_set: str | None = "jets3",
        *,
        weights: np.ndarray | None = None,
        length: int = MAX_LENGTH,
        jet_feature_names: Sequence[str] | None = None,
    ) -> None:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        self.data = data
        self.feature_set = feature_set
        self.label_set = label_set
        self.length = int(length)
        _, self._table = _channels(data, feature_set)
        self._n_features = 0 if feature_set is None else len(FEATURE_SETS[feature_set])
        self.feature_names: tuple[str, ...] = (
            () if feature_set is None else tuple(v.name for v in FEATURE_SETS[feature_set])
        )
        self._labels = None if label_set is None else labels(data, label_set)
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float32)
            if weights.shape != (len(data),):
                raise ValueError(f"weights must have shape ({len(data)},), got {weights.shape}")
        self._weights = weights
        self.jet_feature_names: tuple[str, ...] = ()
        self._jet: np.ndarray | None = None
        if jet_feature_names is not None:
            self._jet, names = jet_features(data, jet_feature_names, derived=())
            self.jet_feature_names = tuple(names)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"JetDataset({len(self)} jets, feature_set={self.feature_set!r}, label_set={self.label_set!r}, "
            f"length={self.length})"
        )

    def batch(self, idx: np.ndarray, *, length: int | None = None) -> JetBatch:
        """Return the jets ``idx`` as a :class:`JetBatch`, padded to the longest of them (at most ``self.length``).

        ``length`` pads to exactly that length instead.
        """
        import torch

        index = np.asarray(idx, dtype=np.int64).reshape(-1)
        table = self._table.select(index)
        if length is None:
            length = _default_length(table.counts(), self.length)
        arrays = _pad_channels(table, length, self._n_features)

        def tensor(a: np.ndarray | None) -> torch.Tensor | None:
            return None if a is None else torch.from_numpy(np.ascontiguousarray(a))

        return JetBatch(
            points=torch.from_numpy(np.ascontiguousarray(arrays["pf_points"])),
            features=torch.from_numpy(np.ascontiguousarray(arrays["pf_features"])),
            vectors=torch.from_numpy(np.ascontiguousarray(arrays["pf_vectors"])),
            mask=torch.from_numpy(arrays["pf_mask"]),
            label=tensor(None if self._labels is None else self._labels[index]),
            weight=tensor(None if self._weights is None else self._weights[index]),
            index=torch.from_numpy(index.copy()),
            jet=tensor(None if self._jet is None else self._jet[index]),
            feature_set=self.feature_set,
        )


def make_batch(
    data: JetData,
    feature_set: str | None = "kin",
    *,
    label_set: str | None = "jets3",
    length: int | None = None,
    jet_feature_names: Sequence[str] | None = None,
    weights: np.ndarray | None = None,
) -> JetBatch:
    """Return all jets of ``data`` as one :class:`JetBatch` (for small data sets and examples).

    ``length`` None pads to the longest jet (at most 128); an integer pads (or truncates) to exactly that length.
    """
    dataset = JetDataset(
        data,
        feature_set,
        label_set,
        weights=weights,
        length=MAX_LENGTH if length is None else length,
        jet_feature_names=jet_feature_names,
    )
    return dataset.batch(np.arange(len(data)), length=length)


class _Loader:
    """Re-iterable single-process loader over a :class:`JetDataset`; epoch ``e`` uses ``default_rng([seed, e])``."""

    def __init__(self, dataset: JetDataset, batch_size: int, shuffle: bool, seed: int, drop_last: bool) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self) -> Iterator[JetBatch]:
        n = len(self.dataset)
        order = np.random.default_rng([self.seed, self.epoch]).permutation(n) if self.shuffle else np.arange(n)
        self.epoch += 1
        for b in range(len(self)):
            yield self.dataset.batch(order[b * self.batch_size : (b + 1) * self.batch_size])


def make_loader(
    dataset: JetDataset, batch_size: int = 256, *, shuffle: bool = True, seed: int = 0, drop_last: bool = False
) -> Iterable[JetBatch]:
    """Return a re-iterable loader of :class:`JetBatch` objects (single process; a new order in every epoch).

    The order of epoch ``e`` (the ``e``-th iteration over the loader) is ``default_rng([seed, e]).permutation(N)``,
    so a run is reproducible for a given seed. ``len(loader)`` is the number of batches per epoch.
    """
    return _Loader(dataset, batch_size, shuffle, seed, drop_last)
