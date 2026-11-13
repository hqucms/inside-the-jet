"""Gradients, integrated gradients, attention, the pair bias and IRC scans (``itj.interpret``)."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from itj import interpret
from itj.data import JetData, Ragged
from itj.features import JetBatch
from itj.models import Tagger, Trace, build_model, make_inputs
from synthetic import three_prong_jet

# ------------------------------------------------------------------------------------------------------------- helpers


class LinearTagger(Tagger):
    """Logits linear in the masked sum of the particle features: integrated gradients are exact."""

    default_name = "linear"

    def __init__(self, n_classes: int = 3, feature_set: str = "kin") -> None:
        super().__init__()
        self.out = nn.Linear(7, n_classes)
        self._set_config(
            {"n_classes": n_classes, "feature_set": feature_set}, feature_set=feature_set, jet_features=None
        )

    @property
    def head(self) -> nn.Linear:
        return self.out

    def trace(self, batch: JetBatch, *, attention: bool = False) -> Trace:
        emb = (batch.features * batch.mask).sum(dim=-1)
        return Trace(logits=self.out(emb), embedding=emb, mask=batch.particle_mask)


@pytest.fixture(scope="module")
def jets(jetclass_sample: JetData) -> JetData:
    """24 JetClass jets: 8 QCD, 8 W, 8 top."""
    return jetclass_sample.select(np.r_[0:8, 50:58, 100:108])


def _calibrated(name: str, data: JetData, seed: int = 0) -> Tagger:
    """A model with BatchNorm statistics of real jets (forward passes in training mode, no weight update)."""
    torch.manual_seed(seed)
    model = build_model(name)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.momentum = None  # cumulative average
    model.train()
    with torch.no_grad():
        for chunk in data.chunks(50):
            model(make_inputs(model, chunk))
    return model.eval()


@pytest.fixture(scope="module")
def trained_part(jetclass_sample: JetData) -> Tagger:
    """ParT-small after 30 optimiser steps on the sample jets (sharper than a random network, as trained ones are)."""
    torch.manual_seed(0)
    model = build_model("partsmall")
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    model.train()
    for _ in range(30):
        batch = make_inputs(model, jetclass_sample.select(rng.choice(150, 32, replace=False)), label_set="jets3")
        loss = nn.functional.cross_entropy(model(batch), batch.label)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model.eval()


def _relative_gap(model: Tagger, batch: JetBatch, gap: np.ndarray, baseline: str, wrt: tuple[str, ...]) -> float:
    with torch.no_grad():
        idx = torch.arange(len(batch))
        s = torch.softmax(model(batch), 1)[idx, batch.label].double()
        s0 = torch.softmax(model(interpret._baseline(batch, baseline, wrt)), 1)[idx, batch.label].double()
    return float(np.abs(gap).sum() / (s - s0).abs().sum())


# ----------------------------------------------------------------------------------------------- gradients and saliency


def test_input_gradients_and_saliency(jets: JetData) -> None:
    model = _calibrated("partsmall", jets)
    model.mod.pair_embed.pairwise_lv_fts.keywords["finite_grad"]  # the attribute the helper switches
    batch = make_inputs(model, jets, label_set="jets3")
    model.train()
    grads = interpret.input_gradients(model, batch, wrt=("features", "vectors", "points"))
    assert model.training  # mode restored
    real = batch.particle_mask
    for name in ("features", "vectors", "points"):
        g = grads[name]
        assert g.shape == getattr(batch, name).shape
        assert torch.isfinite(g).all(), name
        assert torch.all(g.permute(1, 0, 2)[:, ~real] == 0)  # zero at padded slots
    assert grads["features"].abs().sum() > 0
    assert grads["vectors"].abs().sum() > 0
    assert torch.all(grads["points"] == 0)  # ParT does not read the points
    sal = interpret.saliency(model, batch, wrt=("features", "vectors"))
    assert sal.shape == tuple(real.shape)
    expected = torch.sqrt(grads["features"].double().pow(2).sum(1) + grads["vectors"].double().pow(2).sum(1))
    np.testing.assert_allclose(sal, expected.numpy(), rtol=1e-5, atol=1e-12)
    # a class index, a tensor of indices and logits
    g1 = interpret.input_gradients(model, batch, 1, output="logit")["features"]
    g2 = interpret.input_gradients(model, batch, torch.ones(len(batch), dtype=torch.int64), output="logit")
    torch.testing.assert_close(g1, g2["features"])
    with pytest.raises(ValueError, match="wrt"):
        interpret.saliency(model, batch, wrt="momenta")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="label"):
        interpret.saliency(model, make_inputs(model, jets))


def test_finite_gradients_without_finite_grad(jets: JetData) -> None:
    torch.manual_seed(0)
    model = build_model("partsmall", finite_grad=False).eval()
    batch = make_inputs(model, jets.head(4), label_set="jets3")
    v = batch.vectors.clone().requires_grad_(True)
    out = torch.softmax(model(JetBatch(**{**batch.__dict__, "vectors": v})), 1)[:, 0].sum()
    (raw,) = torch.autograd.grad(out, v)
    assert not torch.isfinite(raw).all()  # sqrt(Delta^2 = 0) on the diagonal
    grads = interpret.input_gradients(model, batch, wrt="vectors")
    assert torch.isfinite(grads["vectors"]).all()
    assert model.mod.pair_embed.pairwise_lv_fts.keywords["finite_grad"] is False  # restored


# ------------------------------------------------------------------------------------------------ integrated gradients


@pytest.mark.parametrize("output", ["logit", "prob"])
def test_ig_completeness_on_a_linear_model(jets: JetData, output: str) -> None:
    torch.manual_seed(0)
    model = LinearTagger()
    batch = make_inputs(model, jets, label_set="jets3")
    attr, gap = interpret.integrated_gradients(model, batch, steps=32, output=output)  # type: ignore[arg-type]
    assert attr.shape == tuple(batch.particle_mask.shape)
    assert np.all(attr[~batch.particle_mask.numpy()] == 0)
    with torch.no_grad():
        s = model(batch)[torch.arange(len(batch)), batch.label] if output == "logit" else None
    if output == "logit":
        # linear: IG is exactly (x - 0) * gradient, and completeness holds to float precision
        assert s is not None
        np.testing.assert_allclose(attr.sum(1), (s - model.out.bias[batch.label]).detach().numpy(), rtol=1e-4)
    assert np.abs(gap).max() < 1e-5


def test_ig_quadratures_and_baselines(jets: JetData) -> None:
    torch.manual_seed(0)
    model = LinearTagger()
    batch = make_inputs(model, jets, label_set="jets3")
    for method in ("gausslegendre", "riemann_middle", "riemann_trapezoid"):
        _, gap = interpret.integrated_gradients(model, batch, output="logit", method=method)  # type: ignore[arg-type]
        assert np.abs(gap).max() < 1e-4
    # a batch as the baseline: attributions of the difference; the batch itself gives zero
    attr, gap = interpret.integrated_gradients(model, batch, baseline=batch, output="logit")
    assert np.abs(attr).max() == 0.0
    assert np.abs(gap).max() < 1e-6
    with pytest.raises(ValueError, match="baseline"):
        interpret.integrated_gradients(model, batch, baseline="mean")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="steps"):
        interpret.integrated_gradients(model, batch, steps=0)


def test_ig_completeness_on_part_small(jets: JetData, trained_part: Tagger) -> None:
    """The completeness gap of a trained ParT-small shrinks with the number of steps (the integrand is sharp)."""
    model = trained_part
    batch = make_inputs(model, jets.select(np.arange(0, 24, 6)), label_set="jets3")
    wrt = ("features",)
    rel = {}
    for steps in (32, 128):
        _, gap = interpret.integrated_gradients(model, batch, baseline="zero", wrt=wrt, steps=steps)
        rel[steps] = _relative_gap(model, batch, gap, "zero", wrt)
    print(f"ParT-small zero baseline, relative gap by steps: {rel}")
    assert rel[128] < 0.2 * rel[32]
    assert rel[128] < 0.05


def test_soft_baseline_needs_the_vectors(jets: JetData, trained_part: Tagger) -> None:
    model = trained_part
    batch = make_inputs(model, jets.select(np.arange(0, 24, 3)), label_set="jets3")
    soft = interpret._soft_baseline(batch)
    torch.testing.assert_close(soft.vectors, batch.vectors * interpret.SOFT_SCALE)
    torch.testing.assert_close(soft.points, batch.points)
    # pT and energy features move by ln(1e-3) * 0.7 (before clipping), the ratios to the jet do not
    shift = (soft.features - batch.features)[:, :, 0][batch.particle_mask[:, 0]]
    assert torch.all(shift[:, 0] <= 0)
    torch.testing.assert_close(soft.features[:, 2:], batch.features[:, 2:])
    both = ("features", "vectors")
    model64 = deepcopy(model).double()
    batch64 = interpret._cast(model64, batch, torch.float64)[1]
    rel = {}
    for wrt in (both, ("features",)):
        for dtype in (torch.float32, torch.float64):
            _, gap = interpret.integrated_gradients(model, batch, baseline="soft", wrt=wrt, dtype=dtype)
            m, b = (model64, batch64) if dtype == torch.float64 else (model, batch)
            rel[len(wrt), dtype] = _relative_gap(m, b, gap, "soft", both)
    print(f"ParT-small soft baseline, 32 steps, relative gaps: {rel}")
    # float64: complete with the four-vectors in wrt (up to the quadrature), not without them
    assert rel[2, torch.float64] < 0.01
    assert rel[1, torch.float64] > 10 * rel[2, torch.float64]
    # float32 rounding in ln m^2 of nearly collinear pairs leaves a gap of a few per cent
    assert rel[2, torch.float32] < 0.1
    assert next(model.parameters()).dtype == torch.float32  # the model itself is not converted


@pytest.mark.slow
def test_ig_converges_on_part_small(jets: JetData, trained_part: Tagger) -> None:
    """Completeness to 1e-3 and better with enough steps (zero baseline in float32, soft baseline in float64)."""
    model = trained_part
    batch = make_inputs(model, jets.select(np.arange(0, 24, 3)), label_set="jets3")
    for baseline, wrt, dtype in (
        ("zero", ("features",), torch.float32),
        ("soft", ("features", "vectors"), torch.float64),
    ):
        m, b = interpret._cast(model, batch, dtype)
        gaps = {}
        for steps in (32, 128, 512):
            _, gap = interpret.integrated_gradients(model, batch, baseline=baseline, wrt=wrt, steps=steps, dtype=dtype)
            gaps[steps] = _relative_gap(m, b, gap, baseline, wrt)
        print(f"ParT-small, {baseline} baseline, {dtype}: relative completeness gap by steps {gaps}")
        assert gaps[512] < 1e-3


# ------------------------------------------------------------------------------------------- attention and pair bias


def test_attention_rollout(jets: JetData) -> None:
    model = _calibrated("partsmall", jets)
    batch = make_inputs(model, jets)
    trace = model.trace(batch, attention=True)
    r = interpret.attention_rollout(trace)
    real = batch.particle_mask.numpy()
    assert r.shape == real.shape
    np.testing.assert_allclose(r.sum(1), 1.0, rtol=1e-10)
    assert np.all(r[~real] == 0)
    assert np.all(r >= 0)
    with pytest.raises(ValueError, match="attention"):
        interpret.attention_rollout(model.trace(batch))


def test_pair_features(jets: JetData) -> None:
    from itj.observables import pt_y_phi_m

    batch = make_inputs(build_model("partsmall"), jets.head(3))
    f = interpret.pair_features(batch)
    p = batch.vectors.shape[-1]
    assert f.shape == (3, 4, p, p)
    torch.testing.assert_close(f, f.transpose(-1, -2))
    v = batch.vectors.double().numpy()
    pt, y, phi, _ = pt_y_phi_m(np.moveaxis(v[0], 0, -1))
    dphi = (phi[0] - phi[1] + np.pi) % (2 * np.pi) - np.pi
    delta = math.hypot(y[0] - y[1], dphi)
    assert float(f[0, 2, 0, 1]) == pytest.approx(math.log(delta), abs=1e-3)
    assert float(f[0, 0, 0, 1]) == pytest.approx(math.log(min(pt[0], pt[1]) * delta), abs=1e-3)
    assert float(f[0, 1, 0, 1]) == pytest.approx(math.log(min(pt[0], pt[1]) / (pt[0] + pt[1])), abs=1e-3)
    assert float(f[0, 2, 0, 0]) == pytest.approx(math.log(1e-8), abs=1e-3)  # ln Delta clamped on the diagonal


def test_ablation_contexts_restore_the_model(jets: JetData) -> None:
    model = _calibrated("partsmall", jets)
    batch = make_inputs(model, jets)
    with torch.no_grad():
        before = model(batch).clone()
        with interpret.ablate_pair_channels(model, [0, 3], reference=batch):
            fill = model.pair_channel_fill
            assert fill is not None
            assert set(fill) == {0, 3}
            ablated = model(batch)
            with interpret.ablate_pair_channels(model, [1], fill="zero"):
                assert model.pair_channel_fill == {**fill, 1: 0.0}
            assert model.pair_channel_fill == fill
        assert model.pair_channel_fill is None
        torch.testing.assert_close(model(batch), before)
        assert not torch.allclose(ablated, before)
        with interpret.zero_pair_bias(model):
            zeroed = model(batch)
            assert torch.all(model.trace(batch).pair_bias == 0)  # type: ignore[arg-type]
        torch.testing.assert_close(model(batch), before)
        assert not torch.allclose(zeroed, before)
    # the mean fill is the mean over the distinct real pairs of the reference
    feats = interpret.pair_features(batch)
    real = batch.particle_mask
    pairs = real[:, :, None] & real[:, None, :] & ~torch.eye(real.shape[1], dtype=torch.bool)
    with interpret.ablate_pair_channels(model, [2], reference=batch):
        assert model.pair_channel_fill is not None
        assert model.pair_channel_fill[2] == pytest.approx(float(feats[:, 2][pairs].double().mean()), rel=1e-6)
    with pytest.raises(ValueError, match="reference"), interpret.ablate_pair_channels(model, [0]):
        pass
    nopair = build_model("partsmall_nopair")
    with pytest.raises(ValueError, match="pair embedding"), interpret.zero_pair_bias(nopair):  # type: ignore[arg-type]
        pass


# ------------------------------------------------------------------------------------------------------------ IRC scans


def test_irc_scan_shapes_and_efn_safety(jetclass_sample: JetData) -> None:
    torch.manual_seed(0)
    efn = build_model("efn")
    # larger random weights: the freshly initialised EFN is nearly flat, so its responses would sit at the float32
    # rounding level (about 1e-9) and their order would depend on the platform
    with torch.no_grad():
        for module in efn.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(0.0, 0.3)
                module.bias.zero_()
    eps = [1e-2, 1e-3, 1e-4]
    soft = interpret.irc_scan(efn, jetclass_sample, mode="soft", values=eps, n_jets=60)
    assert set(soft) == {"values", "mean_abs_dscore", "std_abs_dscore", "mean_abs_dlogit"}
    assert all(v.shape == (3,) for v in soft.values())
    # the EFN is linear in the momentum fractions: the response to a soft particle falls like eps
    slope = np.diff(np.log(soft["mean_abs_dlogit"])) / np.diff(np.log(eps))
    assert np.all((slope > 0.8) & (slope < 1.2)), slope
    collinear = interpret.irc_scan(efn, jetclass_sample, mode="collinear", values=[1e-1, 1e-3, 0.0], n_jets=60)
    assert collinear["mean_abs_dlogit"][2] < 1e-4  # an exact collinear split leaves the EFN unchanged
    assert collinear["mean_abs_dlogit"][0] > collinear["mean_abs_dlogit"][1] > collinear["mean_abs_dlogit"][2]
    emb = interpret.irc_scan(efn, jetclass_sample, mode="soft", values=eps, output="embedding", n_jets=30, n=3)
    assert set(emb) == {"values", "mean_embed_dist", "std_embed_dist"}
    assert emb["mean_embed_dist"][0] > emb["mean_embed_dist"][2]
    fixed = interpret.irc_scan(efn, jetclass_sample, mode="soft", values=[1e-2], target=2, n_jets=20)
    assert fixed["mean_abs_dscore"].shape == (1,)
    with pytest.raises(ValueError, match="mode"):
        interpret.irc_scan(efn, jetclass_sample, mode="hard", values=[0.1])  # type: ignore[arg-type]


def test_irc_scan_leaves_out_long_jets(jetclass_sample: JetData) -> None:
    from itj import augment

    torch.manual_seed(0)
    pfn = build_model("pfn")
    jets = jetclass_sample.head(5)
    # five soft particles in the first jet until it has 126 particles: adding n = 5 more would exceed 128
    first = jets.select([0])
    grow = 126 - int(first.counts[0])
    long_jet = augment.add_soft(first, n=grow, eps=1e-6, rng=0)
    data = JetData.concat([long_jet, jets.select([1, 2, 3])])
    out = interpret.irc_scan(pfn, data, mode="soft", values=[1e-3], n=5)
    assert out["mean_abs_dscore"].shape == (1,)
    with pytest.raises(ValueError, match="no jets"):
        interpret.irc_scan(pfn, long_jet, mode="soft", values=[1e-3], n=5)


# ------------------------------------------------------------------------------------------------- groups of particles


def test_subjet_membership() -> None:
    jet = three_prong_jet()
    three = interpret.subjet_membership(jet, 3)
    assert isinstance(three, Ragged)
    assert three.dtype == np.int8
    assert sorted(three[0].tolist()) == [0, 1, 2]
    one = interpret.subjet_membership(jet, 1)
    assert one[0].tolist() == [0, 0, 0]
    np.testing.assert_array_equal(three.offsets, jet["part_px"].offsets)  # type: ignore[union-attr]


def test_attention_stats_does_not_count_self_attention_as_within_group(
    jets: JetData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A head that attends to the particle itself and uniformly to the rest is not a prong head."""
    model = _calibrated("partsmall", jets)
    data = jets.head(12)
    groups = interpret.subjet_membership(data, 2)
    original = type(model).trace

    def self_heads(self: Any, batch: Any, *, attention: bool = False) -> Any:
        tr = original(self, batch, attention=attention)
        real = tr.mask.double()  # (B, P)
        n = real.sum(-1)[:, None, None, None]
        p = real.shape[1]
        eye = torch.eye(p, dtype=torch.float64)
        uniform_others = real[:, None, None, :] * (1 - eye) / (n - 1)
        a = 0.6 * eye + 0.4 * uniform_others  # A_ii = 0.6, the rest uniform over the other real particles
        a = (a * real[:, None, :, None]).expand(-1, 4, -1, -1).to(torch.float32)
        tr.attention = dict.fromkeys(tr.attention, a)
        return tr

    monkeypatch.setattr(type(model), "trace", self_heads)
    stats = interpret.attention_stats(model, data, groups, batch_size=5)
    np.testing.assert_allclose(stats["self"], 0.6, rtol=1e-6)
    np.testing.assert_allclose(stats["within_group"], stats["random_expectation"], rtol=1e-5)
    assert np.all(stats["random_expectation"] < 0.6)


def test_attention_stats_equals_the_full_trace(jets: JetData) -> None:
    model = _calibrated("partsmall", jets)
    data = jets.head(20)
    groups = interpret.subjet_membership(data, 3)
    groups = Ragged(np.where(np.arange(len(groups.values)) % 5 == 0, -1, groups.values).astype(np.int8), groups.offsets)
    stats = interpret.attention_stats(model, data, groups, batch_size=7)
    n_blocks, n_heads = 3, 4
    assert stats["within_group"].shape == (n_blocks, n_heads)
    assert stats["random_expectation"].shape == (n_blocks, n_heads)
    assert stats["class_within"].shape == (1, n_heads, 3)

    assert stats["self"].shape == (n_blocks, n_heads)

    # the same numbers from one trace of all 20 jets, written out with loops
    with torch.no_grad():
        batch = make_inputs(model, data)
        trace = model.trace(batch, attention=True)
    within = np.zeros((n_blocks, n_heads))
    random = np.zeros((n_blocks, n_heads))
    selfs = np.zeros((n_blocks, n_heads))
    n_valid = np.zeros((n_blocks, n_heads))
    n_queries = 0
    cls = np.zeros((1, n_heads, 3))
    cls_random = np.zeros(3)
    for j in range(len(data)):
        g = groups[j].astype(np.int64)
        n = len(g)
        for q in range(n):
            if g[q] < 0 or n < 2:
                continue
            others = np.arange(n) != q
            same = (g == g[q]) & others
            n_queries += 1
            for blk in range(n_blocks):
                a = trace.attention[f"block{blk}"][j, :, q, :n].double().numpy()
                selfs[blk] += a[:, q]
                rest = 1.0 - a[:, q]
                ok = rest >= 1e-6
                within[blk] += np.where(ok, a[:, same].sum(axis=1) / np.maximum(rest, 1e-6), 0.0)
                random[blk] += np.where(ok, same.sum() / (n - 1), 0.0)
                n_valid[blk] += ok
        c = trace.class_attention["cls0"][j, :, 0, 1 : n + 1].double().numpy()
        for k in range(3):
            cls[0, :, k] += c[:, g == k].sum(axis=1)
            cls_random[k] += np.sum(g == k) / n
    np.testing.assert_allclose(stats["within_group"], within / n_valid, rtol=1e-5)
    np.testing.assert_allclose(stats["random_expectation"], random / n_valid, rtol=1e-10)
    np.testing.assert_allclose(stats["self"], selfs / n_queries, rtol=1e-5)
    np.testing.assert_allclose(stats["class_within"], cls / len(data), rtol=1e-5)
    np.testing.assert_allclose(stats["class_random_expectation"], cls_random / len(data), rtol=1e-10)
    with pytest.raises(ValueError, match="aligned"):
        interpret.attention_stats(model, data, Ragged.from_counts(np.zeros(3, np.int8), [3] + [0] * 19))
