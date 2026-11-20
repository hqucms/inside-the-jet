"""Tests of itj.metrics: ROC and AUC against scikit-learn, rejection, one-vs-one scores, JSD and the results table."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np
import pytest

from itj.metrics import (
    RESULT_COLUMNS,
    accuracy,
    append_result,
    auc,
    bootstrap,
    confusion,
    jsd,
    one_vs_one,
    rejection,
    roc,
    sculpting_jsd,
    threshold_at,
)


@pytest.fixture
def scores() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = rng.normal(y * 1.2, 1.0).round(2)  # rounded: many ties
    w = rng.uniform(0.2, 3.0, y.size)
    return y, s, w


def test_roc_and_auc_equal_sklearn(scores: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    metrics = pytest.importorskip("sklearn.metrics")
    y, s, w = scores
    for weight in (None, w):
        eff_s, eff_b, thr = roc(y, s, sample_weight=weight)
        fpr, tpr, sk_thr = metrics.roc_curve(y, s, sample_weight=weight, drop_intermediate=False)
        np.testing.assert_allclose(eff_s, tpr, atol=1e-12)
        np.testing.assert_allclose(eff_b, fpr, atol=1e-12)
        np.testing.assert_array_equal(thr[1:], sk_thr[1:])
        assert thr[0] == np.inf
        assert auc(y, s, sample_weight=weight) == pytest.approx(metrics.roc_auc_score(y, s, sample_weight=weight))


#: An unpacking of the ROC curve in the course text: ``a, b, c = roc(...)`` or ``= itj.metrics.roc(...)``.
_ROC_UNPACK = re.compile(r"(\w+)\s*,\s*(\w+)\s*,\s*\w+\s*=\s*(?:itj\.metrics\.|metrics\.)?roc\(")


def test_roc_is_unpacked_in_its_order_in_the_course() -> None:
    """Every ``x, y, t = roc(...)`` of the docs, tasks, notebooks and examples names the signal efficiency first."""
    root = Path(__file__).resolve().parents[1]
    paths = [root / "TASKS.md", root / "README.md"]
    for folder, pattern in (("docs", "*.md"), ("tasks", "*.md"), ("notebooks", "*.ipynb"), ("examples", "**/*.py")):
        paths += sorted((root / folder).glob(pattern))
    found = 0
    for path in paths:
        if not path.exists():
            continue
        for first, second in _ROC_UNPACK.findall(path.read_text(encoding="utf-8")):
            found += 1
            where = f"{path.relative_to(root)}: roc returns (eff_s, eff_b, thresholds), unpacked as {first}, {second}"
            assert first.endswith("_s"), where
            assert second.endswith("_b"), where
    if (root / "docs").exists():
        assert found >= 2  # the primer and examples/train_tagger.py


def test_roc_limits() -> None:
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(y, np.ones(4)) == 0.5
    eff_s, eff_b, _ = roc(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert (eff_s[0], eff_b[0], eff_s[-1], eff_b[-1]) == (0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="signal"):
        roc(np.ones(3), np.arange(3.0))
    with pytest.raises(ValueError, match="0 \\(background\\) and 1"):
        roc(np.array([0, 2]), np.arange(2.0))
    with pytest.raises(ValueError, match="same length"):
        roc(np.array([0, 1]), np.arange(3.0))


def test_rejection_interpolates_linearly() -> None:
    # 4 signal jets above 4 background jets except one background jet at the top
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.5, 0.4, 0.3])
    # ROC points: (0, 0), (0, 0.25), (0.25, 0.25), (0.5, 0.25), (0.75, 0.25), (1, 0.25), ...
    assert rejection(y, s, 0.5) == pytest.approx(4.0)
    # remove the leading background jet: no background passes up to eff_s = 1
    assert rejection(y[s < 0.95], s[s < 0.95], 0.5) == math.inf
    # a tie of one signal and one background jet gives a diagonal segment from (2/3, 0) to (1, 1/3)
    y3 = np.array([1, 1, 0, 1, 0, 0])
    s3 = np.array([0.9, 0.8, 0.5, 0.5, 0.2, 0.1])
    assert rejection(y3, s3, 5 / 6) == pytest.approx(6.0)
    assert rejection(y3, s3, 2 / 3) == math.inf
    with pytest.raises(ValueError, match="eff_s"):
        rejection(y, s, 0.0)


def test_threshold_at(scores: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    y, s, _ = scores
    rng = np.random.default_rng(1)
    s = s + rng.uniform(0, 1e-3, s.size)  # no ties
    t = threshold_at(y, s, 0.3)
    assert np.mean(s[y == 1] >= t) == pytest.approx(0.3, abs=2 / (y == 1).sum())


def test_one_vs_one() -> None:
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.0, 1.0, 0.0], [0.1, 0.8, 0.1]])
    labels = np.array([0, 2, 0, 1])
    y, s, idx = one_vs_one(probs, labels, signal=2, background=0)
    np.testing.assert_array_equal(idx, [0, 1, 2])
    np.testing.assert_array_equal(y, [0, 1, 0])
    np.testing.assert_allclose(s, [0.1 / 0.7, 0.6 / 0.8, 0.5])  # 0.5 where p_S + p_B = 0
    assert y.dtype == np.int64
    with pytest.raises(ValueError, match="different"):
        one_vs_one(probs, labels, 1, 1)
    with pytest.raises(ValueError, match="outside"):
        one_vs_one(probs, labels, 3, 0)


def test_accuracy_and_confusion() -> None:
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.1, 0.0, 0.9], [0.1, 0.8, 0.1]])
    labels = np.array([0, 2, 0, 1])
    assert accuracy(probs, labels) == 0.75
    c = confusion(probs, labels, 3)
    np.testing.assert_array_equal(c, [[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    assert c.sum() == 4


def test_jsd_limits() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 20000)
    assert jsd(a, a) == 0.0
    assert jsd(a, a[::-1].copy(), bins=20) == pytest.approx(0.0, abs=1e-12)
    assert jsd(np.full(10, 1.0), np.full(10, 5.0), bins=4, range=(0.0, 8.0)) == pytest.approx(1.0)
    small = jsd(a[:10000], a[10000:], bins=20, range=(-4.0, 4.0))
    large = jsd(a, a + 1.0, bins=20, range=(-4.0, 5.0))
    assert 0.0 < small < 0.01 < large < 1.0
    # symmetric and weights act like repeated entries
    b = rng.normal(0.5, 1, 5000)
    assert jsd(a, b, range=(-5.0, 5.0)) == pytest.approx(jsd(b, a, range=(-5.0, 5.0)))
    x = np.array([0.5, 1.5, 2.5])
    assert jsd(x, np.array([0.5, 1.5, 1.5, 2.5]), bins=3, range=(0.0, 3.0)) == pytest.approx(
        jsd(x, x, bins=3, range=(0.0, 3.0), weights_b=np.array([1.0, 2.0, 1.0]))
    )


def test_sculpting_jsd() -> None:
    rng = np.random.default_rng(3)
    mass = rng.uniform(0, 300, 40000)
    assert sculpting_jsd(mass, rng.random(mass.size) < 0.3) < 0.005  # a mass-independent cut
    assert sculpting_jsd(mass, (mass > 60) & (mass < 105)) == pytest.approx(1.0)  # a mass window
    with pytest.raises(ValueError, match="pass and jets that fail"):
        sculpting_jsd(mass, np.ones(mass.size, bool))


def test_bootstrap(scores: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    y, s, _ = scores
    value, err = bootstrap(auc, y, s, n=50, seed=1)
    assert value == auc(y, s)
    # the standard error of the AUC of 2000 jets is about 0.01
    assert 0.003 < err < 0.03
    assert bootstrap(auc, y, s, n=50, seed=1) == (value, err)
    r50, r50_err = bootstrap(rejection, y, s, n=20)
    assert r50 == rejection(y, s)
    assert r50_err > 0


def test_append_result(tmp_path: Path) -> None:
    path = tmp_path / "results" / "results_table.csv"
    append_result(path, {"model": "pfn", "auc_w": np.float32(0.912345678), "n_train": np.int64(70000), "seed": 0})
    append_result(path, {"model": "partsmall", "note": "no U, background run"})
    with path.open() as f:
        rows = list(csv.DictReader(f))
    with path.open() as f:
        header = f.readline().strip().split(",")
    assert header == list(RESULT_COLUMNS)
    assert len(rows) == 2
    assert rows[0]["auc_w"] == "0.912346"
    assert rows[0]["n_train"] == "70000"
    assert rows[0]["r50_w"] == ""
    assert rows[1]["note"] == "no U, background run"
    with pytest.raises(ValueError, match="did you mean 'auc_top'"):
        append_result(path, {"auc_tp": 0.9})
