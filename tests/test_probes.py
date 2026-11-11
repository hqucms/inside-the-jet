"""Probes, control tasks, linear CKA and the extraction of representations (``itj.probes``)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from helpers import rss_growth
from itj.data import JetData
from itj.probes import ProbeResult, cka_matrix, control_task, extract, fit_probe, linear_cka, selectivity

# ---------------------------------------------------------------------------------------------------------------- CKA


def test_cka_identical_and_invariances() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 10))
    q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-12)
    assert linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-12)  # orthogonal map
    assert linear_cka(x, 7.5 * x @ q + 3.0) == pytest.approx(1.0, abs=1e-12)  # isotropic scaling and shift
    # padding with zero columns is an orthogonal embedding
    assert linear_cka(x, np.hstack([x @ q, np.zeros((400, 5))])) == pytest.approx(1.0, abs=1e-12)
    # an anisotropic linear map is not an invariance
    assert linear_cka(x, x * np.linspace(0.1, 10.0, 10)) < 0.9
    # symmetric
    y = rng.normal(size=(400, 4))
    assert linear_cka(x, y) == pytest.approx(linear_cka(y, x))


def test_cka_random_and_bounds() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4000, 8))
    y = rng.normal(size=(4000, 8))
    assert 0.0 <= linear_cka(x, y) < 0.01  # bias of order D / N
    # a shared low-dimensional part gives an intermediate value
    z = np.hstack([x[:, :4], y[:, :4]])
    assert 0.3 < linear_cka(x, z) < 0.7
    assert np.isnan(linear_cka(np.ones((10, 3)), x[:10]))
    with pytest.raises(ValueError, match="same N"):
        linear_cka(x, y[:10])


def test_cka_matrix() -> None:
    rng = np.random.default_rng(2)
    a = {"l0": rng.normal(size=(300, 5)), "l1": rng.normal(size=(300, 3))}
    b = {"x": a["l0"] @ np.linalg.qr(rng.normal(size=(5, 5)))[0], "y": rng.normal(size=(300, 2)), "z": a["l1"]}
    m, rows, cols = cka_matrix(a, b)
    assert m.shape == (2, 3)
    assert rows == ["l0", "l1"]
    assert cols == ["x", "y", "z"]
    assert m[0, 0] == pytest.approx(1.0)
    assert m[1, 2] == pytest.approx(1.0)
    assert m[0, 1] < 0.1


# -------------------------------------------------------------------------------------------------------------- probes


def test_linear_probe_regression() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(600, 6))
    y = x @ rng.normal(size=6) + 0.01 * rng.normal(size=600)
    res = fit_probe(x, y)
    assert isinstance(res, ProbeResult)
    assert (res.metric, res.task, res.kind) == ("r2", "regression", "linear")
    assert res.score > 0.999
    assert res.score_std < 0.01
    # a nonlinear target: the MLP probe does better than the linear one
    yq = x[:, 0] ** 2
    lin = fit_probe(x, yq)
    mlp = fit_probe(x, yq, kind="mlp")
    assert lin.score < 0.1
    assert mlp.score > 0.6


def test_probe_classification_metrics() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(500, 4))
    binary = (x[:, 0] + 0.1 * rng.normal(size=500) > 0).astype(np.int64)
    res = fit_probe(x, binary)
    assert (res.metric, res.task) == ("auc", "classification")
    assert res.score > 0.98
    three = np.digitize(x[:, 1], [-0.5, 0.5])
    res3 = fit_probe(x, three)
    assert res3.metric == "accuracy"
    assert res3.score > 0.85
    # bool targets are classification, many distinct integers are regression unless told otherwise
    assert fit_probe(x, binary.astype(bool)).task == "classification"
    counts = rng.integers(0, 60, 500)
    assert fit_probe(x, counts).task == "regression"
    assert fit_probe(x, counts % 2, task="classification").metric == "auc"


def test_probe_errors() -> None:
    pytest.importorskip("sklearn")
    x = np.random.default_rng(5).normal(size=(50, 3))
    y = np.arange(50, dtype=np.float64)
    y[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_probe(x, y)
    with pytest.raises(ValueError, match="kind"):
        fit_probe(x, np.arange(50.0), kind="tree")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cv"):
        fit_probe(x, np.arange(50.0), cv=1)
    with pytest.raises(ValueError, match="two classes"):
        fit_probe(x, np.zeros(50, dtype=np.int64), task="classification")


# ------------------------------------------------------------------------------------------------------------ controls


def test_control_tasks() -> None:
    y = np.arange(100) % 7
    s = control_task(y, seed=1)
    assert sorted(s.tolist()) == sorted(y.tolist())
    assert not np.array_equal(s, y)
    assert np.array_equal(control_task(y, seed=1), s)  # deterministic
    groups = np.repeat(np.arange(10), 10)
    g = control_task(y, groups=groups, seed=2)
    for k in range(10):
        assert sorted(g[groups == k].tolist()) == sorted(y[groups == k].tolist())
    r = control_task(y.astype(np.float64), kind="random_map", groups=groups, seed=3)
    assert r.dtype == np.float64
    for k in range(10):
        assert len(np.unique(r[groups == k])) == 1
    assert set(np.unique(r)) <= set(y.tolist())
    with pytest.raises(ValueError, match="kind"):
        control_task(y, kind="noise")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="groups"):
        control_task(y, groups=groups[:5])


def test_selectivity_of_a_real_probe() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(6)
    x = rng.normal(size=(800, 5))
    y = x[:, 0] - 2 * x[:, 3]
    real = fit_probe(x, y)
    control = fit_probe(x, control_task(y, seed=0))
    assert control.score < 0.05
    assert selectivity(real, control) > 0.9
    with pytest.raises(ValueError, match="metrics"):
        selectivity(real, fit_probe(x, (y > 0).astype(np.int64)))


# ---------------------------------------------------------------------------------------------------------- extraction


@pytest.mark.parametrize("name", ["pfn", "partsmall", "pnlite"])
def test_extract_matches_embed(jetclass_sample: JetData, name: str) -> None:
    from itj.features import JetDataset
    from itj.models import build_model, make_inputs

    torch.manual_seed(0)
    model = build_model(name)
    model.train()
    jets = jetclass_sample.head(40)
    out = extract(model, jets, batch_size=16)
    assert model.training  # mode restored
    assert list(out) == ["embedding"]
    model.eval()
    with torch.no_grad():
        direct = model.embed(make_inputs(model, jets)).numpy()
    assert out["embedding"].dtype == np.float32
    np.testing.assert_allclose(out["embedding"], direct, rtol=1e-4, atol=1e-5)
    full = extract(model, JetDataset(jets, model.feature_set, None), layers="all", batch_size=7)
    np.testing.assert_allclose(full["embedding"], direct, rtol=1e-4, atol=1e-5)
    pooled = [k for k in full if k.startswith("pooled/")]
    assert pooled
    assert all(full[k].shape[0] == len(jets) for k in pooled)
    with pytest.raises(ValueError, match="layers"):
        extract(model, jets, layers="some")  # type: ignore[arg-type]


def test_extract_memory_is_flat(jetclass_sample: JetData) -> None:
    from itj.models import build_model

    torch.manual_seed(0)
    model = build_model("partsmall")
    jets = jetclass_sample.select(np.arange(2000) % len(jetclass_sample))
    with rss_growth() as growth:
        out = extract(model, jets, layers="all", batch_size=256)
    assert out["embedding"].shape == (2000, 64)
    assert growth.gb < 1.0, f"extract grew the RSS by {growth.gb:.2f} GB"
