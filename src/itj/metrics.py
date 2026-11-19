"""Tagging metrics: ROC curves, AUC, background rejection, one-vs-one scores, mass sculpting and the results table.

Conventions (TASKS.md, docs/methods.md): ``y`` is 1 for signal and 0 for background, ``s`` a score that is larger
for signal; ``eff_s`` and ``eff_b`` are the signal and background efficiencies of the cut ``s >= threshold``;
R50 = 1 / eff_b at eff_s = 0.5 (:func:`rejection`). Multi-class models give per-pair scores with
:func:`one_vs_one`: ``s = p_S / (p_S + p_B)`` on the jets of the two classes only (the JetClass convention).

Everything is NumPy; scikit-learn is not needed (the tests compare with it).

Examples
--------
>>> import numpy as np
>>> y = np.array([0, 0, 1, 1])
>>> s = np.array([0.1, 0.4, 0.35, 0.8])
>>> auc(y, s)
0.75
>>> probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
>>> yy, ss, idx = one_vs_one(probs, np.array([0, 1, 2]), signal=1, background=0)
>>> yy.tolist(), np.round(ss, 3).tolist(), idx.tolist()
([0, 1], [0.222, 0.889], [0, 1])
"""

from __future__ import annotations

import csv
import io
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "RESULT_COLUMNS",
    "accuracy",
    "append_result",
    "auc",
    "bootstrap",
    "confusion",
    "jsd",
    "one_vs_one",
    "rejection",
    "roc",
    "sculpting_jsd",
    "threshold_at",
]

#: The columns of ``results_table.csv`` (one row per trained or evaluated model, weeks 2 to 4); the ``*_err`` columns
#: hold the bootstrap errors (:func:`bootstrap`) and stay empty when unknown.
RESULT_COLUMNS: tuple[str, ...] = (
    "model",
    "train_data",
    "test_data",
    "n_train",
    "feature_set",
    "auc_w",
    "r50_w",
    "auc_top",
    "r50_top",
    "r80_top",
    "auc_w_err",
    "r50_w_err",
    "auc_top_err",
    "r50_top_err",
    "r80_top_err",
    "accuracy",
    "n_params",
    "train_time_s",
    "seed",
    "note",
)


def _binary(y: ArrayLike, s: ArrayLike, sample_weight: ArrayLike | None = None) -> tuple[np.ndarray, ...]:
    """Validate (y, s, w) and return them as 1-d arrays (bool, float64, float64)."""
    y_arr = np.asarray(y).reshape(-1)
    s_arr = np.asarray(s, dtype=np.float64).reshape(-1)
    if y_arr.shape != s_arr.shape:
        raise ValueError(f"y and s must have the same length, got {y_arr.shape[0]} and {s_arr.shape[0]}")
    values = np.unique(y_arr)
    if not np.all(np.isin(values, (0, 1))):
        raise ValueError(f"y must hold 0 (background) and 1 (signal) only, got the values {values[:5].tolist()}")
    w = np.ones_like(s_arr) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.shape != s_arr.shape:
        raise ValueError(f"sample_weight must have the length of s ({s_arr.shape[0]}), got {w.shape[0]}")
    if np.isnan(s_arr).any():
        raise ValueError("the scores contain NaN")
    return y_arr.astype(bool), s_arr, w


def roc(
    y: ArrayLike, s: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ROC curve of the cut ``s >= threshold``: (eff_s, eff_b, thresholds).

    Parameters
    ----------
    y : array_like
        1 for signal, 0 for background.
    s : array_like
        Scores (larger = more signal-like).
    sample_weight : array_like, optional
        Per-jet weights.

    Returns
    -------
    tuple of numpy.ndarray
        Signal efficiency, background efficiency and threshold, one point per distinct score in decreasing order
        of the threshold, starting at (0, 0, +inf) and ending at (1, 1, min s); the same points as
        ``sklearn.metrics.roc_curve(..., drop_intermediate=False)``.
    """
    yb, sv, w = _binary(y, s, sample_weight)
    w_sig, w_bkg = w[yb].sum(), w[~yb].sum()
    if w_sig <= 0 or w_bkg <= 0:
        raise ValueError("the ROC curve needs signal (y = 1) and background (y = 0) jets with positive weight")
    order = np.argsort(-sv, kind="mergesort")
    s_sorted = sv[order]
    tp = np.cumsum(np.where(yb[order], w[order], 0.0))
    fp = np.cumsum(np.where(yb[order], 0.0, w[order]))
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), s_sorted.size - 1]  # the last jet of every distinct score
    eff_s = np.r_[0.0, tp[last] / w_sig]
    eff_b = np.r_[0.0, fp[last] / w_bkg]
    thresholds = np.r_[np.inf, s_sorted[last]]
    return eff_s, eff_b, thresholds


def auc(y: ArrayLike, s: ArrayLike, *, sample_weight: ArrayLike | None = None) -> float:
    """Return the area under the ROC curve (trapezoids; ties count one half, as ``sklearn.metrics.roc_auc_score``)."""
    eff_s, eff_b, _ = roc(y, s, sample_weight=sample_weight)
    return float(np.sum(np.diff(eff_b) * 0.5 * (eff_s[1:] + eff_s[:-1])))


def rejection(y: ArrayLike, s: ArrayLike, eff_s: float = 0.5) -> float:
    """Return the background rejection 1 / eff_b at a signal efficiency (R50 for ``eff_s=0.5``).

    eff_b is interpolated linearly in eff_s between the points of :func:`roc`; ``inf`` when no background passes.
    """
    if not 0.0 < eff_s <= 1.0:
        raise ValueError(f"eff_s must be in (0, 1], got {eff_s}")
    es, eb, _ = roc(y, s)
    # eff_s is non-decreasing; at a vertical step take the smallest eff_b that reaches eff_s
    i = int(np.searchsorted(es, eff_s, side="left"))
    if i == 0:
        value = float(eb[0])
    else:
        x0, x1, y0, y1 = es[i - 1], es[i], eb[i - 1], eb[i]
        value = float(y1 if x1 == x0 else y0 + (y1 - y0) * (eff_s - x0) / (x1 - x0))
    return math.inf if value <= 0.0 else 1.0 / value


def threshold_at(y: ArrayLike, s: ArrayLike, eff_s: float) -> float:
    """Return the score threshold whose cut ``s >= threshold`` keeps the fraction ``eff_s`` of the signal.

    The ``1 - eff_s`` quantile of the signal scores (linear interpolation between jets).
    """
    if not 0.0 <= eff_s <= 1.0:
        raise ValueError(f"eff_s must be in [0, 1], got {eff_s}")
    yb, sv, _ = _binary(y, s)
    if not yb.any():
        raise ValueError("no signal jets (y = 1)")
    return float(np.quantile(sv[yb], 1.0 - eff_s))


def one_vs_one(
    probs: np.ndarray, labels: np.ndarray, signal: int, background: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the binary problem "class ``signal`` vs class ``background``" of a multi-class model.

    Parameters
    ----------
    probs : numpy.ndarray
        (N, C) class probabilities.
    labels : numpy.ndarray
        (N,) true classes.
    signal, background : int
        The two class indices (JetClass: QCD 0, Wqq 7, Tbqq 8; jets3: QCD 0, W 1, top 2).

    Returns
    -------
    tuple of numpy.ndarray
        ``y`` (int64, 1 for signal), ``s = p_S / (p_S + p_B)`` (float64; 0.5 where both are 0) and the indices of
        the selected jets in ``probs``, for the jets whose true class is one of the two.
    """
    p = np.asarray(probs, dtype=np.float64)
    lab = np.asarray(labels).reshape(-1)
    if p.ndim != 2 or p.shape[0] != lab.shape[0]:
        raise ValueError(f"probs must be (N, C) with N = len(labels) = {lab.shape[0]}, got {p.shape}")
    for c in (signal, background):
        if not 0 <= c < p.shape[1]:
            raise ValueError(f"class index {c} outside [0, {p.shape[1]})")
    if signal == background:
        raise ValueError("signal and background must be different classes")
    idx = np.flatnonzero((lab == signal) | (lab == background))
    ps, pb = p[idx, signal], p[idx, background]
    den = ps + pb
    s = np.divide(ps, den, out=np.full_like(ps, 0.5), where=den > 0)
    return (lab[idx] == signal).astype(np.int64), s, idx


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Return the fraction of jets whose most probable class is the true class."""
    p = np.asarray(probs)
    lab = np.asarray(labels).reshape(-1)
    if p.ndim != 2 or p.shape[0] != lab.shape[0]:
        raise ValueError(f"probs must be (N, C) with N = len(labels) = {lab.shape[0]}, got {p.shape}")
    if lab.size == 0:
        return math.nan
    return float(np.mean(np.argmax(p, axis=1) == lab))


def confusion(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Return the confusion matrix (n_classes, n_classes) of counts: row = true class, column = predicted class.

    Divide by ``.sum(axis=1, keepdims=True)`` for the fractions per true class.
    """
    p = np.asarray(probs)
    lab = np.asarray(labels).reshape(-1).astype(np.int64)
    if p.ndim != 2 or p.shape[0] != lab.shape[0]:
        raise ValueError(f"probs must be (N, C) with N = len(labels) = {lab.shape[0]}, got {p.shape}")
    pred = np.argmax(p, axis=1).astype(np.int64)
    if lab.size and (np.min(lab) < 0 or np.max(lab) >= n_classes or np.max(pred) >= n_classes):
        raise ValueError(f"classes must be in [0, {n_classes})")
    out = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(out, (lab, pred), 1)
    return out


def jsd(
    a: np.ndarray,
    b: np.ndarray,
    *,
    bins: int | np.ndarray = 40,
    range: tuple[float, float] | None = None,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
) -> float:
    """Return the Jensen-Shannon divergence (base 2, in [0, 1]) of the histograms of two samples.

    ``JSD = KL(P || M) / 2 + KL(Q || M) / 2`` with M = (P + Q) / 2 and P, Q the normalised histograms with the same
    bins (``range`` None: from the smallest to the largest value of both samples). 0 for identical histograms, 1 for
    histograms without a common bin. Entries outside ``range`` are ignored.
    """
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    if av.size == 0 or bv.size == 0:
        raise ValueError("jsd needs two non-empty samples")
    if range is None and np.ndim(bins) == 0:
        both = np.concatenate([av, bv])
        lo, hi = float(both.min()), float(both.max())
        range = (lo, hi if hi > lo else lo + 1.0)
    p, edges = np.histogram(av, bins=bins, range=range, weights=weights_a)
    q, _ = np.histogram(bv, bins=edges, weights=weights_b)
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("a histogram of jsd is empty (check range)")
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl(x: np.ndarray) -> float:
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / m[nz])))

    return float(min(1.0, max(0.0, 0.5 * kl(p) + 0.5 * kl(q))))


def sculpting_jsd(
    mass: np.ndarray, passed: np.ndarray, *, bins: int = 40, range: tuple[float, float] = (0.0, 300.0)
) -> float:
    """Return the mass sculpting of a cut: the JSD (base 2) between the mass histograms of passing and failing jets.

    Apply it to background jets only (QCD m_SD with a W-tagger cut at eff_s = 50 %, W3.2.3): 0 means the cut keeps
    the shape of the background mass distribution; 1/JSD is the usual decorrelation figure of merit.

    Parameters
    ----------
    mass : numpy.ndarray
        Masses (GeV) of the jets.
    passed : numpy.ndarray
        Bool, True for the jets that pass the cut.
    bins : int
        Number of bins.
    range : tuple of float
        Histogram range in GeV.
    """
    m = np.asarray(mass, dtype=np.float64).reshape(-1)
    ok = np.asarray(passed, dtype=bool).reshape(-1)
    if m.shape != ok.shape:
        raise ValueError(f"mass and passed must have the same length, got {m.shape[0]} and {ok.shape[0]}")
    if ok.all() or not ok.any():
        raise ValueError("sculpting_jsd needs jets that pass and jets that fail the cut")
    return jsd(m[ok], m[~ok], bins=bins, range=range)


def bootstrap(
    metric: Callable[[np.ndarray, np.ndarray], float], y: ArrayLike, s: ArrayLike, *, n: int = 100, seed: int = 0
) -> tuple[float, float]:
    """Return the metric on the jets and its bootstrap standard deviation: (value, error).

    Parameters
    ----------
    metric : callable
        ``metric(y, s) -> float``, e.g. :func:`auc` or :func:`rejection`.
    y, s : array_like
        Labels and scores.
    n : int
        Number of resamplings (jets drawn with replacement, ``default_rng(seed)``).
    seed : int
        Seed of the resampling.

    Returns
    -------
    tuple of float
        ``metric(y, s)`` and the standard deviation (ddof 1) of the metric over the resamplings (infinite values,
        e.g. a rejection with no background left, are dropped).
    """
    yv = np.asarray(y).reshape(-1)
    sv = np.asarray(s).reshape(-1)
    if yv.shape != sv.shape:
        raise ValueError(f"y and s must have the same length, got {yv.shape[0]} and {sv.shape[0]}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, yv.size, yv.size)
        try:
            values.append(float(metric(yv[idx], sv[idx])))
        except ValueError:  # a resampling without one of the classes
            continue
    finite = np.asarray([v for v in values if math.isfinite(v)])
    err = float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan
    return float(metric(yv, sv)), err


def append_result(path: str | os.PathLike[str], row: Mapping[str, object]) -> None:
    """Append one row to the results table (CSV with the columns :data:`RESULT_COLUMNS`).

    The header is written when the file is new or empty; missing columns are left empty and floats are written
    with 6 significant digits. Unknown keys raise a ValueError (with a suggestion), so a typo cannot silently drop
    a number.
    """
    import difflib

    unknown = [k for k in row if k not in RESULT_COLUMNS]
    if unknown:
        hints = []
        for k in unknown:
            close = difflib.get_close_matches(k, RESULT_COLUMNS, n=1)
            hints.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(f"unknown result columns {', '.join(hints)}; the columns are {', '.join(RESULT_COLUMNS)}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v: object) -> object:
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.6g}"
        if isinstance(v, np.integer):
            return int(v)
        return v

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=RESULT_COLUMNS)
    writer.writerow({k: fmt(row.get(k)) for k in RESULT_COLUMNS})
    with p.open("a", newline="", encoding="utf-8") as f:
        # decide on the header with the file open and write header and row at once, so that two trainings that
        # finish together (the week-2 box and a student's own run) do not both write a header or interleave their rows
        f.seek(0, os.SEEK_END)
        header = ""
        if f.tell() == 0:
            head = io.StringIO(newline="")
            csv.DictWriter(head, fieldnames=RESULT_COLUMNS).writeheader()
            header = head.getvalue()
        f.write(header + buffer.getvalue())
