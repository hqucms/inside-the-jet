"""Tests of itj.models: shapes, invariances, traces, gradients, configurations and checkpoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from itj.data import JetData
from itj.features import FEATURE_SETS, JetBatch
from itj.models import (
    EFN,
    MLP,
    MODEL_PRESETS,
    ParT,
    build_model,
    load_model,
    make_inputs,
    save_model,
)

#: Narrow versions of every preset (same structure, a few hundred to a few thousand parameters).
_PART_TINY = {"embed_dims": (16, 32, 16), "pair_embed_dims": (8, 8), "num_heads": 2, "num_layers": 2}
TINY: dict[str, dict] = {
    "mlp": {"hidden": (16, 16)},
    "pfn": {"phi_sizes": (16, 16), "f_sizes": (16,)},
    "efn": {"phi_sizes": (16, 16), "f_sizes": (16,)},
    "pnlite": {"conv_params": ((4, (8, 8)), (4, (16, 16))), "fc_params": ((16, 0.1),)},
    "pn": {"conv_params": ((4, (8, 8)), (4, (16, 16))), "fc_params": ((16, 0.1),)},
    "partsmall": dict(_PART_TINY),
    "partsmall_nopair": {**_PART_TINY, "pair_embed_dims": None},
    "partlite": dict(_PART_TINY),
    "part": {**_PART_TINY, "num_cls_layers": 2},
}
PRESETS = sorted(MODEL_PRESETS)
PARTICLE_PRESETS = [p for p in PRESETS if p != "mlp"]
PART_PRESETS = [p for p in PRESETS if MODEL_PRESETS[p][0] is ParT]

#: Trainable parameters of the full presets (models.md quotes them rounded).
N_PARAMS = {
    "mlp": 35_867,
    "pfn": 83_971,
    "efn": 56_731,
    "pnlite": 47_281,
    "pn": 366_417,
    "partsmall": 240_035,
    "partsmall_nopair": 237_423,
    "partlite": 943_135,
    "part": 2_141_263,
}


def tiny(name: str, **kwargs: Any) -> Any:
    torch.manual_seed(0)
    return build_model(name, **{**TINY[name], **kwargs}).eval()


@pytest.fixture(scope="module")
def jets(jetclass_sample: JetData) -> JetData:
    """24 JetClass jets of the three classes."""
    return jetclass_sample.select(np.r_[0:8, 50:58, 100:108])


def permute_slots(batch: JetBatch, perm: torch.Tensor) -> JetBatch:
    """The same jets with the padded slots (real and padding) in another order."""
    return replace(
        batch,
        points=batch.points[..., perm],
        features=batch.features[..., perm],
        vectors=batch.vectors[..., perm],
        mask=batch.mask[..., perm],
    )


def test_tiny_overrides_cover_every_preset() -> None:
    assert set(TINY) == set(MODEL_PRESETS)


@pytest.mark.parametrize("name", PRESETS)
def test_n_parameters_of_the_presets(name: str) -> None:
    assert build_model(name).n_parameters() == N_PARAMS[name]


@pytest.mark.parametrize("name", PRESETS)
def test_shapes_and_head(name: str, jets: JetData) -> None:
    model = tiny(name)
    batch = make_inputs(model, jets)
    with torch.no_grad():
        logits = model(batch)
        emb = model.embed(batch)
        tr = model.trace(batch)
    assert logits.shape == (len(jets), 3)
    assert emb.ndim == 2
    assert emb.shape[0] == len(jets)
    torch.testing.assert_close(tr.logits, logits)
    torch.testing.assert_close(tr.embedding, emb)
    torch.testing.assert_close(model.head(emb), logits, atol=1e-5, rtol=1e-5)
    assert tr.attention == {}
    assert tr.class_attention == {}
    if name == "mlp":
        assert tr.mask is None
        assert tr.layers == {}
        assert set(tr.pooled) == {"hidden0", "hidden1"}
    else:
        assert tr.mask is not None
        assert tr.mask.shape == batch.particle_mask.shape
        for key, layer in tr.layers.items():
            assert layer.shape[:2] == batch.particle_mask.shape, key
            assert torch.all(layer[~tr.mask] == 0), key
            assert tr.pooled[key].shape == (len(jets), layer.shape[-1])


@pytest.mark.parametrize("name", PRESETS)
def test_n_classes_and_feature_sets(name: str, jets: JetData) -> None:
    for feature_set in ("kin", "kinpid"):
        model = tiny(name, feature_set=feature_set, n_classes=5)
        batch = make_inputs(model, jets)
        if name == "mlp":
            assert model.feature_set is None
            assert batch.jet is not None
            assert batch.jet.shape == (len(jets), len(model.jet_features or ()))
        else:
            assert model.feature_set == feature_set
            assert batch.features.shape[1] == len(FEATURE_SETS[feature_set])
        with torch.no_grad():
            assert model(batch).shape == (len(jets), 5)


@pytest.mark.parametrize("name", PARTICLE_PRESETS)
def test_permutation_invariance(name: str, jets: JetData) -> None:
    model = tiny(name)
    batch = make_inputs(model, jets)
    perm = torch.from_numpy(np.random.default_rng(1).permutation(batch.points.shape[-1]))
    with torch.no_grad():
        a = model(batch)
        b = model(permute_slots(batch, perm))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name", PARTICLE_PRESETS)
def test_padding_invariance(name: str, jets: JetData) -> None:
    model = tiny(name)
    short = make_inputs(model, jets)
    long = make_inputs(model, jets, length=short.points.shape[-1] + 9)
    with torch.no_grad():
        torch.testing.assert_close(model(short), model(long), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(model.embed(short), model.embed(long), atol=1e-5, rtol=1e-5)


def test_efn_is_collinear_and_infrared_safe(jets: JetData) -> None:
    model = tiny("efn")
    assert isinstance(model, EFN)
    base = make_inputs(model, jets)
    batch = make_inputs(model, jets, length=base.points.shape[-1] + 1)
    n = batch.particle_mask.sum(dim=1)
    rows = torch.arange(len(jets))
    split, soft = (
        replace(batch, points=batch.points.clone(), vectors=batch.vectors.clone(), mask=batch.mask.clone())
        for _ in range(2)
    )
    lam = 0.3
    # collinear: particle 0 -> lam and 1 - lam of it at the same angles, the second piece in the first padded slot
    split.points[rows, :, n] = batch.points[rows, :, 0]
    split.vectors[rows, :, n] = (1 - lam) * batch.vectors[rows, :, 0]
    split.vectors[rows, :, 0] = lam * batch.vectors[rows, :, 0]
    split.mask[rows, 0, n] = 1.0
    # infrared: a particle of 1e-9 of the leading pT at a random angle
    soft.points[rows, :, n] = torch.tensor([0.31, -0.42])[None, :]
    soft.vectors[rows, :, n] = 1e-9 * batch.vectors[rows, :, 0]
    soft.mask[rows, 0, n] = 1.0
    with torch.no_grad():
        ref = model(batch)
        torch.testing.assert_close(model(split), ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(model(soft), ref, atol=1e-5, rtol=1e-5)
        # the PFN is not collinear safe
        pfn = tiny("pfn")
        assert not torch.allclose(pfn(replace(split, features=batch.features)), pfn(batch), atol=1e-3)
    z = EFN.momentum_fractions(batch)
    torch.testing.assert_close(z.sum(dim=1), torch.ones(len(jets)))
    assert torch.all(z[~batch.particle_mask] == 0)


def test_efn_filters() -> None:
    model = tiny("efn")
    f = model.filters(extent=0.5, bins=12)
    assert f.shape == (16, 12, 12)
    assert f.dtype == np.float32
    grid = torch.tensor([[-0.5 + (0.5 + 3) * 1.0 / 12, -0.5 + (0.5 + 7) * 1.0 / 12]])
    with torch.no_grad():
        direct = model.phi(grid).numpy()[0]
    np.testing.assert_allclose(f[:, 3, 7], direct, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", PART_PRESETS)
def test_part_trace_attention(name: str, jets: JetData) -> None:
    model = tiny(name)
    batch = make_inputs(model, jets)
    with torch.no_grad():
        tr = model.trace(batch, attention=True)
        logits = model(batch)
    torch.testing.assert_close(tr.logits, logits, atol=1e-5, rtol=1e-5)
    mask = batch.particle_mask
    b, p = mask.shape
    n_blocks = model.config["num_layers"]
    assert sorted(tr.attention) == [f"block{i}" for i in range(n_blocks)]
    assert sorted(tr.class_attention) == [f"cls{i}" for i in range(model.config["num_cls_layers"])]
    assert set(tr.layers) == {"embed", *(f"block{i}" for i in range(n_blocks))}
    assert {f"cls{i}" for i in range(model.config["num_cls_layers"])} <= set(tr.pooled)
    for a in tr.attention.values():
        assert a.shape == (b, 2, p, p)
        sums = a.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=0)
        assert torch.all(a.masked_select(~mask[:, None, None, :]) == 0)  # no attention to padding
    for a in tr.class_attention.values():
        assert a.shape == (b, 2, 1, 1 + p)
        torch.testing.assert_close(a.sum(dim=-1), torch.ones(b, 2, 1), atol=1e-5, rtol=0)
        assert torch.all(a[..., 1:].masked_select(~mask[:, None, None, :]) == 0)
    if MODEL_PRESETS[name][1].get("pair_embed_dims", ()) is None:
        assert tr.pair_bias is None
    else:
        assert tr.pair_bias is not None
        assert tr.pair_bias.shape == (b, 2, p, p)
        pair = mask[:, :, None] & mask[:, None, :]
        assert torch.all(tr.pair_bias.masked_select(~pair[:, None]) == 0)
        torch.testing.assert_close(tr.pair_bias, tr.pair_bias.transpose(-1, -2))
    # the explicit attention path gives the outputs of the fused one
    with torch.no_grad():
        torch.testing.assert_close(model.trace(batch).logits, tr.logits, atol=1e-5, rtol=1e-5)
    assert all(not blk.attn.keep_weights and blk.attn.weights is None for blk in model.mod.blocks)


def test_part_gradients_are_finite(jets: JetData) -> None:
    model = tiny("partsmall")
    batch = make_inputs(model, jets)
    features = batch.features.clone().requires_grad_(True)
    vectors = batch.vectors.clone().requires_grad_(True)
    model(replace(batch, features=features, vectors=vectors))[:, 1].sum().backward()
    assert features.grad is not None
    assert vectors.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(vectors.grad).all()
    assert vectors.grad.abs().sum() > 0
    padded = ~batch.particle_mask[:, None, :]
    assert torch.all(features.grad.masked_select(padded) == 0)
    assert torch.all(vectors.grad.masked_select(padded) == 0)
    # only the vectors: the trimmer is off, so the graph reaches them
    vectors2 = batch.vectors.clone().requires_grad_(True)
    model(replace(batch, vectors=vectors2))[:, 1].sum().backward()
    assert vectors2.grad is not None
    assert torch.isfinite(vectors2.grad).all()
    # weaver's sqrt(Delta^2) has an infinite derivative on the diagonal (Delta = 0)
    plain = tiny("partsmall", finite_grad=False)
    plain.load_state_dict(model.state_dict())
    vectors3 = batch.vectors.clone().requires_grad_(True)
    out = plain(replace(batch, vectors=vectors3))
    torch.testing.assert_close(out, model(batch), atol=1e-6, rtol=1e-6)
    out[:, 1].sum().backward()
    assert vectors3.grad is not None
    assert not torch.isfinite(vectors3.grad).all()


def test_gradients_train_mode_every_preset(jets: JetData) -> None:
    for name in PRESETS:
        model = tiny(name).train()
        batch = make_inputs(model, jets, label_set="jets3")
        assert batch.label is not None
        loss = torch.nn.functional.cross_entropy(model(batch), batch.label)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name


@pytest.mark.parametrize("pair_input_dim", [1, 4, 5])
def test_pair_input_dim_variants(pair_input_dim: int, jets: JetData) -> None:
    model = tiny("partsmall", pair_input_dim=pair_input_dim)
    assert model.mod.pair_embed.pairwise_lv_dim == pair_input_dim
    batch = make_inputs(model, jets)
    with torch.no_grad():
        tr = model.trace(batch)
    assert tr.logits.shape == (len(jets), 3)
    assert torch.isfinite(tr.logits).all()
    assert tr.pair_bias is not None


def test_pair_channel_fill(jets: JetData) -> None:
    model = tiny("partsmall")
    assert isinstance(model, ParT)
    batch = make_inputs(model, jets)
    with torch.no_grad():
        ref = model(batch)
        model.pair_channel_fill = {0: 0.0, 3: 1.5}
        assert model.pair_channel_fill == {0: 0.0, 3: 1.5}
        filled = model(batch)
        model.pair_channel_fill = None
        again = model(batch)
    assert not torch.allclose(filled, ref)
    torch.testing.assert_close(again, ref)
    with pytest.raises(ValueError, match="channels must be in"):
        model.pair_channel_fill = {4: 0.0}
    nopair = tiny("partsmall_nopair")
    assert isinstance(nopair, ParT)
    assert nopair.pair_channel_fill is None
    with pytest.raises(ValueError, match="no pair features"):
        nopair.pair_channel_fill = {0: 0.0}


@pytest.mark.parametrize("name", PRESETS)
def test_save_load_and_config_round_trip(name: str, jets: JetData, tmp_path: Path) -> None:
    model = tiny(name)
    batch = make_inputs(model, jets)
    path = save_model(model, tmp_path / "sub" / f"{name}.pt", extra={"seed": 3})
    stored = torch.load(path, weights_only=True)
    assert set(stored) == {"itj_version", "name", "config", "state_dict", "extra"}
    assert stored["extra"] == {"seed": 3}
    assert stored["name"] == name
    loaded = load_model(path)
    assert loaded.name == name
    assert not loaded.training
    assert loaded.feature_set == model.feature_set
    assert loaded.jet_features == model.jet_features
    rebuilt = build_model(**model.config).eval()
    rebuilt.load_state_dict(model.state_dict())
    with torch.no_grad():
        ref = model(batch)
        torch.testing.assert_close(loaded(batch), ref)
        torch.testing.assert_close(rebuilt(batch), ref)
    assert model.config["name"] == name


def test_load_bare_state_dict(jets: JetData, tmp_path: Path) -> None:
    model = tiny("partsmall", n_classes=4)
    path = tmp_path / "net_best_epoch_state.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match="bare state dict"):
        load_model(path)
    loaded = load_model(path, name="partsmall", **TINY["partsmall"])
    assert loaded.n_classes == 4
    assert loaded.feature_set == "kin"
    batch = make_inputs(model, jets)
    with torch.no_grad():
        torch.testing.assert_close(loaded(batch), model(batch))


def test_mlp_input_batchnorm(jets: JetData) -> None:
    model = tiny("mlp").train()
    assert isinstance(model, MLP)
    assert model.jet_features is not None
    assert model.jet_features[-2:] == ("tau21", "tau32")
    assert isinstance(model.input_bn, torch.nn.BatchNorm1d)
    assert model.input_bn.num_features == len(model.jet_features) == 12
    batch = make_inputs(model, jets)
    assert batch.jet is not None
    z = model.input_bn(batch.jet)
    torch.testing.assert_close(z.mean(dim=0), torch.zeros(12), atol=1e-4, rtol=0)
    with pytest.raises(ValueError, match=r"JetBatch\.jet"):
        model(replace(batch, jet=None))


def test_reset_parameters(jets: JetData) -> None:
    model = tiny("partsmall")
    batch = make_inputs(model, jets)
    with torch.no_grad():
        ref = model(batch)
    state = torch.random.get_rng_state()
    model.reset_parameters(seed=5)
    assert torch.equal(torch.random.get_rng_state(), state)
    with torch.no_grad():
        first = model(batch)
    model.reset_parameters(seed=5)
    with torch.no_grad():
        torch.testing.assert_close(model(batch), first)
    assert not torch.allclose(first, ref)


def test_errors(jets: JetData) -> None:
    with pytest.raises(ValueError, match="did you mean 'partsmall'"):
        build_model("partsmal")
    with pytest.raises(TypeError, match="phi_sizes"):
        build_model("pfn", phi_size=(8,))
    with pytest.raises(ValueError, match="feature set"):
        build_model("pfn", feature_set="kinn")
    model = tiny("pfn", feature_set="kinpid")
    batch = make_inputs(tiny("pfn"), jets)
    with pytest.raises(ValueError, match="expects 13 particle features"):
        model(batch)
    with pytest.raises(ValueError, match="at least one class-attention block"):
        build_model("partsmall", num_cls_layers=0)


def test_particlenet_short_jets(jets: JetData) -> None:
    """Batches shorter than k + 1 slots are padded for the kNN; the outputs do not depend on that padding."""
    model = tiny("pnlite", conv_params=((16, (8, 8)),))
    short = jets.select(np.arange(4))
    batch = make_inputs(model, short)
    assert batch.points.shape[-1] > 0
    cut = replace(
        batch,
        points=batch.points[..., :6],
        features=batch.features[..., :6],
        vectors=batch.vectors[..., :6],
        mask=batch.mask[..., :6],
    )
    with torch.no_grad():
        tr = model.trace(cut)
    assert tr.logits.shape == (4, 3)
    assert torch.isfinite(tr.logits).all()
    assert tr.layers["edgeconv0"].shape == (4, 6, 8)
