"""What is inside a representation: embedding extraction, probes with controls and linear CKA.

* :func:`extract` runs a tagger over a data set batch by batch and keeps only the per-jet vectors (the embedding,
  the input of the final Linear layer, and optionally the masked mean of every layer), so memory stays flat;
* :func:`fit_probe` fits a linear (or one-hidden-layer) probe from an embedding to a target with standardised inputs
  and cross-validation; :func:`control_task` builds the control targets and :func:`selectivity` the difference
  "real minus control" that makes a probe result honest (Hewitt and Liang, arXiv:1909.03368);
* :func:`linear_cka` and :func:`cka_matrix` compare two representations of the same jets (Kornblith et al.,
  arXiv:1905.00414): invariant under orthogonal maps and isotropic scaling, not under general linear maps.

scikit-learn (``analysis`` extra) is imported inside the probe functions only.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(500, 8))
>>> Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
>>> round(linear_cka(X, 3.0 * X @ Q), 6)
1.0
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .data import JetData
    from .features import JetDataset
    from .models import Tagger

__all__ = [
    "ProbeResult",
    "cka_matrix",
    "control_task",
    "extract",
    "fit_probe",
    "linear_cka",
    "selectivity",
]

_log = logging.getLogger("itj")

#: Integer targets with at most this many distinct values are classification targets for ``task="auto"``.
MAX_AUTO_CLASSES = 10


# ----------------------------------------------------------------------------------------------------------------------
# Extraction


def _iter_data(data: JetData | JetDataset, batch_size: int) -> Any:
    """Yield consecutive :class:`~itj.data.JetData` blocks of a data set (or of the data of a ``JetDataset``)."""
    from .features import JetDataset

    jets = data.data if isinstance(data, JetDataset) else data
    return jets.chunks(batch_size)


def extract(
    model: Tagger,
    data: JetData | JetDataset,
    *,
    layers: Literal["embedding", "all"] = "embedding",
    batch_size: int = 512,
    device: Any = "cpu",
) -> dict[str, np.ndarray]:
    """Return the per-jet representations of a tagger: the embedding and, for ``layers="all"``, every pooled layer.

    The model runs in evaluation mode (its mode is restored afterwards) without gradients, and ``trace`` is called
    with ``attention=False``; the batches are reduced to per-jet vectors one by one, so the peak memory is that of one
    batch plus the float32 outputs (N x D x 4 bytes per entry).

    Parameters
    ----------
    model : itj.models.Tagger
        Any tagger of :mod:`itj.models`.
    data : JetData or JetDataset
        Jets (a ``JetDataset`` contributes its ``data``); the inputs are built with ``itj.models.make_inputs``.
    layers : {"embedding", "all"}
        "embedding": only ``"embedding"`` (N, D), the input of the final Linear layer; "all": also
        ``"pooled/<layer>"`` for every entry of ``Trace.pooled`` (masked means of the per-particle layers; PFN and EFN
        also ``"pooled/latent"``, the sum).
    batch_size : int
        Jets per forward pass.
    device : str or torch.device
        Where the model runs (it is moved there).

    Returns
    -------
    dict of str to numpy.ndarray
        float32 arrays with one row per jet, in the order of ``data``.
    """
    import torch

    from .models import make_inputs

    if not callable(getattr(model, "trace", None)):
        raise TypeError(
            f"extract needs an itj Tagger with a trace method, not a {type(model).__name__}: copy the weights of your "
            "own model into build_model(...) with load_state_dict (the key map of W2.4.2)"
        )
    if layers not in ("embedding", "all"):
        raise ValueError(f"layers must be 'embedding' or 'all', got {layers!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    parts: dict[str, list[np.ndarray]] = {}
    was_training = model.training
    model.eval()
    model.to(device)
    try:
        with torch.no_grad():
            for chunk in _iter_data(data, batch_size):
                batch = make_inputs(model, chunk, label_set=None).to(device)
                trace = model.trace(batch, attention=False)
                parts.setdefault("embedding", []).append(trace.embedding.detach().float().cpu().numpy())
                if layers == "all":
                    for name, value in trace.pooled.items():
                        parts.setdefault(f"pooled/{name}", []).append(value.detach().float().cpu().numpy())
                del trace, batch
    finally:
        model.train(was_training)
    if not parts:
        return {"embedding": np.zeros((0, 0), dtype=np.float32)}
    return {name: np.concatenate(values, axis=0) for name, values in parts.items()}


# ----------------------------------------------------------------------------------------------------------------------
# Probes


@dataclass(frozen=True)
class ProbeResult:
    """Cross-validated score of a probe.

    Attributes
    ----------
    score : float
        Mean score over the folds.
    score_std : float
        Standard deviation of the score over the folds.
    metric : str
        "r2" (regression), "auc" (binary) or "accuracy" (multi-class).
    task : str
        "regression" or "classification".
    kind : str
        "linear" or "mlp".
    """

    score: float
    score_std: float
    metric: str
    task: str
    kind: str

    def __str__(self) -> str:
        return f"{self.kind} probe ({self.task}): {self.metric} = {self.score:.4f} +- {self.score_std:.4f}"


def _task(y: np.ndarray, task: str) -> str:
    if task in ("regression", "classification"):
        return task
    if task != "auto":
        raise ValueError(f"task must be 'auto', 'regression' or 'classification', got {task!r}")
    if y.dtype == bool or y.dtype.kind in "OUS":
        return "classification"
    if y.dtype.kind in "iu" and len(np.unique(y)) <= MAX_AUTO_CLASSES:
        return "classification"
    return "regression"


def fit_probe(
    X: ArrayLike,
    y: ArrayLike,
    *,
    kind: Literal["linear", "mlp"] = "linear",
    task: Literal["auto", "regression", "classification"] = "auto",
    cv: int = 5,
    seed: int = 0,
    alpha: float | None = None,
) -> ProbeResult:
    """Fit a probe from representations ``X`` to a target ``y`` and return its cross-validated score.

    The inputs are standardised inside every fold (a scikit-learn pipeline, so no information leaks from the test
    fold). Linear probes are ridge regression or logistic regression; MLP probes have one hidden layer of 64 ReLU
    units. Folds are shuffled with ``seed`` (stratified for classification).

    Parameters
    ----------
    X : array_like
        (N, D) representations (for example ``extract(...)["embedding"]``).
    y : array_like
        (N,) target. Rows with a non-finite target are an error: select the jets first (for example
        ``truth_dr_min`` is NaN for QCD jets).
    kind : {"linear", "mlp"}
        Probe family.
    task : {"auto", "regression", "classification"}
        "auto" chooses classification for bool and string targets and for integer targets with at most 10
        distinct values, regression otherwise; pass the task for counts such as ``truth_n_primary``.
    cv : int
        Number of folds (>= 2).
    seed : int
        Seed of the fold assignment and of the MLP initialisation.
    alpha : float, optional
        Regularisation strength: ridge ``alpha`` (default 1), ``C = 1 / alpha`` for the logistic regression
        (default C = 1), the L2 penalty of the MLP (default 1e-4).

    Returns
    -------
    ProbeResult
        Mean and standard deviation of R^2 (regression), ROC AUC (binary) or accuracy (multi-class) over the folds.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(X, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    target = np.asarray(y)
    if x.ndim != 2 or len(x) != len(target):
        raise ValueError(f"X must be (N, D) and y (N,), got {x.shape} and {target.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("X has non-finite values")
    if target.dtype.kind == "f" and not np.all(np.isfinite(target)):
        raise ValueError(f"y has {int(np.sum(~np.isfinite(target)))} non-finite values: select the jets first")
    if kind not in ("linear", "mlp"):
        raise ValueError(f"kind must be 'linear' or 'mlp', got {kind!r}")
    if cv < 2:
        raise ValueError(f"cv must be >= 2, got {cv}")
    chosen = _task(target, task)
    estimator: Any
    if chosen == "regression":
        target = target.astype(np.float64)
        metric, scoring = "r2", "r2"
        splitter: Any = KFold(cv, shuffle=True, random_state=seed)
        if kind == "linear":
            estimator = Ridge(alpha=1.0 if alpha is None else alpha)
        else:
            estimator = MLPRegressor(
                hidden_layer_sizes=(64,), alpha=1e-4 if alpha is None else alpha, max_iter=500, random_state=seed
            )
    else:
        classes = np.unique(target)
        if len(classes) < 2:
            raise ValueError("a classification target needs at least two classes")
        metric = "auc" if len(classes) == 2 else "accuracy"
        scoring = "roc_auc" if metric == "auc" else "accuracy"
        splitter = StratifiedKFold(cv, shuffle=True, random_state=seed)
        if kind == "linear":
            estimator = LogisticRegression(C=1.0 if alpha is None else 1.0 / alpha, max_iter=2000)
        else:
            estimator = MLPClassifier(
                hidden_layer_sizes=(64,), alpha=1e-4 if alpha is None else alpha, max_iter=500, random_state=seed
            )
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = cross_val_score(make_pipeline(StandardScaler(), estimator), x, target, cv=splitter, scoring=scoring)
    _log.debug("%s probe, %s: %s per fold %s", kind, chosen, metric, scores)
    return ProbeResult(float(np.mean(scores)), float(np.std(scores)), metric, chosen, kind)


def control_task(
    y: ArrayLike,
    *,
    kind: Literal["shuffle", "random_map"] = "shuffle",
    groups: ArrayLike | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Return control targets: the same marginal distribution as ``y``, no relation to the jets.

    Parameters
    ----------
    y : array_like
        (N,) real targets.
    kind : {"shuffle", "random_map"}
        "shuffle": a random permutation of ``y`` (with ``groups``: permuted within every group, so the control keeps
        what the groups explain). "random_map": every group gets one target drawn at random from ``y`` (a random
        function of the group, the control task of Hewitt and Liang); a probe can only score by memorising the
        groups. Without ``groups`` every jet is its own group (a draw with replacement).
    groups : array_like, optional
        (N,) group of every jet (for example a binned observable or the multiplicity).
    seed : int
        Seed.

    Returns
    -------
    numpy.ndarray
        (N,) control targets with the dtype of ``y``.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.arange(6)
    >>> sorted(control_task(y, seed=1).tolist()) == y.tolist()
    True
    >>> c = control_task(y, kind="random_map", groups=[0, 0, 0, 1, 1, 1], seed=1)
    >>> bool(c[0] == c[1] == c[2] and c[3] == c[4] == c[5])
    True
    """
    target = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(target)
    g = None if groups is None else np.asarray(groups)
    if g is not None and g.shape != (n,):
        raise ValueError(f"groups must have shape ({n},), got {g.shape}")
    if kind == "shuffle":
        if g is None:
            return target[rng.permutation(n)]
        out = target.copy()
        for value in np.unique(g):
            idx = np.flatnonzero(g == value)
            out[idx] = target[idx[rng.permutation(len(idx))]]
        return out
    if kind == "random_map":
        if g is None:
            return target[rng.integers(0, n, n)] if n else target.copy()
        values, inverse = np.unique(g, return_inverse=True)
        draws = target[rng.integers(0, n, len(values))]
        return draws[inverse.reshape(-1)]
    raise ValueError(f"kind must be 'shuffle' or 'random_map', got {kind!r}")


def selectivity(real: ProbeResult, control: ProbeResult) -> float:
    """Return the selectivity ``real.score - control.score`` (both probes must use the same metric)."""
    if real.metric != control.metric:
        raise ValueError(f"the probes use different metrics: {real.metric!r} and {control.metric!r}")
    return real.score - control.score


# ----------------------------------------------------------------------------------------------------------------------
# CKA


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Return the linear centred kernel alignment of two representations of the same N jets.

    ``CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)`` after centring every column (float64). It is 1 for
    representations that differ by an orthogonal map and an isotropic scaling, about 0 for independent ones (with a
    positive bias of order D / N), and in [0, 1] in general; it is NaN if either representation is constant.

    Parameters
    ----------
    X : numpy.ndarray
        (N, D1) representation.
    Y : numpy.ndarray
        (N, D2) representation.

    Returns
    -------
    float
        Linear CKA.
    """
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError(f"X and Y must be (N, D) arrays with the same N, got {x.shape} and {y.shape}")
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    cross = float(np.linalg.norm(y.T @ x) ** 2)
    norm = float(np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y))
    return cross / norm if norm > 0 else float("nan")


def cka_matrix(
    reps_a: Mapping[str, np.ndarray], reps_b: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return the linear CKA between every representation of ``reps_a`` and every one of ``reps_b``.

    Parameters
    ----------
    reps_a, reps_b : Mapping of str to numpy.ndarray
        Name to (N, D) representation (for example the output of :func:`extract` with ``layers="all"``).

    Returns
    -------
    tuple of (numpy.ndarray, list of str, list of str)
        The (len(reps_a), len(reps_b)) matrix and the row and column names (for ``itj.plot.cka_heatmap``).
    """
    names_a, names_b = list(reps_a), list(reps_b)
    m = np.empty((len(names_a), len(names_b)))
    for r, a in enumerate(names_a):
        for c, b in enumerate(names_b):
            m[r, c] = linear_cka(reps_a[a], reps_b[b])
    return m, names_a, names_b
