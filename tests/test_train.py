"""Tests of itj.train: the training loop, its options and determinism, and predict."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
import torch

from itj.data import JetData
from itj.features import JetDataset, make_loader
from itj.models import Tagger, build_model, make_inputs
from itj.train import History, _cross_entropy, fit, predict, set_seed, set_threads


def tiny_pfn(seed: int = 0, **kwargs: Any) -> Tagger:
    torch.manual_seed(seed)
    return build_model("pfn", phi_sizes=(16, 16), f_sizes=(16,), **kwargs)


@pytest.fixture(scope="module")
def dataset(jetclass_sample: JetData) -> JetDataset:
    return JetDataset(jetclass_sample)


def test_loss_decreases(dataset: JetDataset) -> None:
    model = tiny_pfn()
    h = fit(model, dataset, dataset, epochs=12, batch_size=32, lr=3e-3, quiet=True)
    assert isinstance(h, History)
    assert len(h) == 12
    assert h.train_loss[-1] < 0.8 * h.train_loss[0]
    assert h.val_acc[h.best_epoch] > 0.5  # 150 jets of three classes, chance is 1/3
    assert all(np.isfinite(h.train_acc))
    assert all(r > 0 for r in h.jets_per_s)
    assert h.wall_time_s == sorted(h.wall_time_s)
    assert not model.training
    # the best state by validation loss is restored
    probs = predict(model, dataset)
    labels = dataset.data.labels()
    loss = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    assert loss == pytest.approx(min(h.val_loss), rel=1e-4)
    d = h.to_dict()
    assert set(d) == {
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "wall_time_s",
        "jets_per_s",
        "steps",
        "best_epoch",
    }
    json.dumps(d)


def test_deterministic_for_a_seed(dataset: JetDataset) -> None:
    runs = []
    for _ in range(2):
        model = tiny_pfn(seed=3)
        h = fit(model, dataset, epochs=2, batch_size=32, seed=7, quiet=True)
        runs.append((h.train_loss, model.state_dict()))
    assert runs[0][0] == runs[1][0]
    for k, v in runs[0][1].items():
        assert torch.equal(v, runs[1][1][k]), k
    model = tiny_pfn(seed=3)
    h = fit(model, dataset, epochs=2, batch_size=32, seed=8, quiet=True)
    assert h.train_loss != runs[0][0]


def test_max_minutes_min_steps_patience_and_callback(dataset: JetDataset) -> None:
    h = fit(tiny_pfn(), dataset, epochs=5, batch_size=16, max_minutes=1e-9, quiet=True)
    assert len(h) == 1
    h = fit(tiny_pfn(), dataset, epochs=1, batch_size=50, min_steps=10, quiet=True)
    assert len(h) == math.ceil(10 * 50 / 150)
    seen: list[int] = []
    h = fit(
        tiny_pfn(), dataset, dataset, epochs=6, lr=0.0, patience=1, quiet=True, callback=lambda e, hist: seen.append(e)
    )
    assert len(h) == 2
    assert seen == [0, 1]
    assert h.best_epoch == 0
    assert np.isnan(fit(tiny_pfn(), dataset, epochs=1, quiet=True).val_loss[0])


def test_val_every_makes_patience_a_number_of_steps(dataset: JetDataset, capsys: pytest.CaptureFixture[str]) -> None:
    # 150 jets in batches of 50: 3 steps per epoch; validate once at least 7 steps have passed
    h = fit(tiny_pfn(), dataset, dataset, epochs=8, batch_size=50, val_every=7, quiet=True)
    assert h.steps == [3 * (e + 1) for e in range(8)]
    validated = [e for e, v in enumerate(h.val_loss) if np.isfinite(v)]
    assert validated == [2, 5, 7]  # steps 9 and 18, and always the last epoch (24)
    assert np.isnan(h.val_acc[0])
    assert np.isfinite(h.val_loss[h.best_epoch])
    # patience counts validations: with lr 0 the first validation stays the best
    h = fit(tiny_pfn(), dataset, dataset, epochs=50, batch_size=50, lr=0.0, val_every=6, patience=2, quiet=True)
    assert [e for e, v in enumerate(h.val_loss) if np.isfinite(v)] == [1, 3, 5]
    assert len(h) == 6
    assert h.best_epoch == 1
    # without validation data the epoch lines follow the same schedule
    capsys.readouterr()
    fit(tiny_pfn(), dataset, epochs=8, batch_size=50, val_every=7, log_every=0)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("epoch ")]
    assert [line.split()[1] for line in lines] == ["3/8", "6/8", "8/8"]
    with pytest.raises(ValueError, match="val_every"):
        fit(tiny_pfn(), dataset, dataset, val_every=0, quiet=True)
    with pytest.raises(ValueError, match="patience"):
        fit(tiny_pfn(), dataset, dataset, patience=0, quiet=True)


def test_weights(dataset: JetDataset, jetclass_sample: JetData) -> None:
    plain = fit(tiny_pfn(), dataset, epochs=2, batch_size=32, quiet=True)
    weighted = JetDataset(jetclass_sample, weights=np.full(len(jetclass_sample), 2.5))
    h = fit(tiny_pfn(), weighted, epochs=2, batch_size=32, quiet=True)
    np.testing.assert_allclose(h.train_loss, plain.train_loss, rtol=1e-5)
    # a weight of 2 is a jet counted twice
    batch = weighted.batch(np.arange(10))
    logits = torch.randn(10, 3)
    w = torch.ones(10)
    w[0] = 2.0
    twice = torch.cat([logits, logits[:1]])
    assert batch.label is not None
    ce = torch.nn.functional.cross_entropy(twice, torch.cat([batch.label, batch.label[:1]]))
    torch.testing.assert_close(_cross_entropy(logits, type(batch)(**{**batch.__dict__, "weight": w})), ce)


def test_custom_loss_and_iterable(dataset: JetDataset) -> None:
    batches = list(make_loader(dataset, 50, shuffle=False))

    def loss(model: Tagger, batch: object) -> torch.Tensor:
        return model.embed(batch).square().mean()  # type: ignore[arg-type]

    h = fit(tiny_pfn(), batches, batches, epochs=3, loss=loss, quiet=True)
    assert len(h) == 3
    assert np.isnan(h.train_acc[0])
    assert np.isfinite(h.val_acc[0])
    gen = (b for b in batches)
    assert len(fit(tiny_pfn(), gen, epochs=2, quiet=True)) == 2  # a generator is kept for the second epoch


def test_feature_mismatch(jetclass_sample: JetData) -> None:
    with pytest.raises(ValueError, match="feature_set='kin'"):
        fit(tiny_pfn(), JetDataset(jetclass_sample, "kinpid"), quiet=True)
    mlp = build_model("mlp", hidden=(8,))
    with pytest.raises(ValueError, match="jet_features"):
        fit(mlp, JetDataset(jetclass_sample, None, jet_feature_names=("jet_pt", "jet_sdmass")), quiet=True)
    good = JetDataset(jetclass_sample, None, jet_feature_names=mlp.jet_features)
    assert len(fit(mlp, good, epochs=1, quiet=True)) == 1


def test_threads_and_eta(dataset: JetDataset, capsys: pytest.CaptureFixture[str]) -> None:
    before = torch.get_num_threads()
    fit(tiny_pfn(), dataset, epochs=1, batch_size=4, threads=1, log_every=10)
    assert torch.get_num_threads() == before
    out = capsys.readouterr().out
    assert "20 batches" in out
    assert "ETA" in out
    assert "batch 10/38" in out
    assert "epoch 1/1" in out
    try:
        assert set_threads(1) == 1
        assert torch.get_num_threads() == 1
        assert set_threads() == min(6, __import__("os").cpu_count())
    finally:
        torch.set_num_threads(before)
    with pytest.raises(ValueError, match="threads"):
        set_threads(0)


def test_set_seed() -> None:
    set_seed(5)
    a = (torch.rand(3), np.random.rand(3))  # noqa: NPY002
    set_seed(5)
    b = (torch.rand(3), np.random.rand(3))  # noqa: NPY002
    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_predict(jetclass_sample: JetData, dataset: JetDataset) -> None:
    model = tiny_pfn().train()
    probs = predict(model, jetclass_sample, batch_size=40)
    assert model.training  # the mode is restored
    assert probs.shape == (150, 3)
    assert probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    model.eval()
    with torch.no_grad():
        ref = torch.softmax(model(make_inputs(model, jetclass_sample)), dim=1).numpy()
    np.testing.assert_allclose(probs, ref, atol=1e-6)
    np.testing.assert_allclose(predict(model, dataset, batch_size=1000), ref, atol=1e-6)
    assert predict(model, jetclass_sample.select(np.arange(0))).shape == (0, 3)
    mlp = build_model("mlp", hidden=(8,)).eval()
    assert predict(mlp, jetclass_sample).shape == (150, 3)
    with pytest.raises(ValueError, match="feature_set"):
        predict(model, JetDataset(jetclass_sample, "kinpid"))


def test_own_model_gets_clear_errors(jetclass_sample: JetData, tmp_path: Any) -> None:
    """A student's own nn.Module (T2.4): predict asks for feature_set, save_model and extract say what to do."""
    from itj.models import save_model
    from itj.probes import extract

    class Own(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.phi = torch.nn.Linear(7, 4)
            self.out = torch.nn.Linear(4, 3)

        def forward(self, batch: Any) -> torch.Tensor:
            x = self.phi(batch.features.transpose(1, 2)) * batch.mask.reshape(len(batch), -1, 1)
            return self.out(x.sum(dim=1))

    own = Own()
    with pytest.raises(ValueError, match="pass feature_set"):
        predict(own, jetclass_sample)
    assert predict(own, jetclass_sample, feature_set="kin").shape == (150, 3)
    with pytest.raises(TypeError, match="build_model"):
        save_model(own, tmp_path / "own.pt")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="build_model"):
        extract(own, jetclass_sample)  # type: ignore[arg-type]
