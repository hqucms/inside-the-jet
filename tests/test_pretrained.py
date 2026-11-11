"""Tests of the pretrained JetClass models: legacy-key conversion and the scores of weaver 0.5.4 on the sample.

``tests/data/jetclass_sample_scores.npz`` holds the softmax scores that ``weaver --predict`` (weaver-core 0.5.4,
``JetClass_kin.yaml``, the ParT repository's example networks, batch size 64) gave for the 150 jets of
``jetclass_sample.parquet`` with the checkpoints ``ParT_kin.pt`` and ``ParticleNet_kin.pt`` of particle_transformer
commit fa63c86 (see tests/data/README.md). The score tests look for the checkpoints in ``$ITJ_PRETRAINED_DIR`` or
``models/pretrained`` (``jetgen fetch models ParT_kin ParticleNet_kin``) and skip without them; the ``network`` test
downloads them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from itj.data import JetData
from itj.jetclass import MODELS
from itj.models import PRETRAINED, ParT, build_model, load_pretrained, make_inputs

SCORES = Path(__file__).parent / "data" / "jetclass_sample_scores.npz"
ROOT = Path(os.environ.get("ITJ_PRETRAINED_DIR", Path(__file__).parents[1] / "models" / "pretrained"))
COMPARED: tuple[Any, ...] = ("ParT_kin", "ParticleNet_kin")


def to_legacy(
    state: dict[str, torch.Tensor], c_attn: dict[str, torch.Tensor], head_dim: int
) -> dict[str, torch.Tensor]:
    """Write a state dict as weaver-core < 0.5 did (nn.MultiheadAttention keys, c_attn after a (d, h) reshuffle).

    For every block prefix of ``c_attn`` the output projection is stored so that weaver's conversion (fold the
    (head_dim, num_heads) channel order and c_attn into out_proj) gives back the weights of ``state``.
    """
    out = {}
    for k, v in state.items():
        out[
            k.replace("attn.in_proj.weight", "attn.in_proj_weight").replace("attn.in_proj.bias", "attn.in_proj_bias")
        ] = v
    for prefix, c in c_attn.items():
        h = c.numel()
        for name in ("weight", "bias"):
            key = f"{prefix}attn.out_proj.{name}"
            w = state[key]
            old = w.view(head_dim, h, *w.shape[1:]).transpose(0, 1) / c.view(-1, *([1] * w.dim()))
            out[key] = old.reshape(w.shape).clone()
        out[f"{prefix}c_attn"] = c
    return out


def test_registry_matches_the_manifest() -> None:
    assert set(PRETRAINED) == set(MODELS)
    for name, (preset, feature_set) in PRETRAINED.items():
        assert name.split("_")[1] == feature_set
        assert preset == ("part" if name.startswith("ParT") else "pn")


def test_legacy_key_conversion() -> None:
    torch.manual_seed(0)
    kwargs: dict[str, Any] = {
        "embed_dims": (16, 32, 16),
        "pair_embed_dims": (8, 8),
        "num_heads": 4,
        "num_layers": 2,
        "num_cls_layers": 2,
        "n_classes": 10,
    }
    model = build_model("part", **kwargs).eval()
    assert isinstance(model, ParT)
    state = model.state_dict()
    prefixes = [f"mod.blocks.{i}." for i in range(2)] + [f"mod.cls_blocks.{i}." for i in range(2)]
    for p in prefixes:
        assert torch.equal(state[f"{p}c_attn"], torch.ones(4))
    gen = torch.Generator().manual_seed(1)
    c_attn = {p: torch.rand(4, generator=gen) + 0.5 for p in prefixes}
    legacy = to_legacy(state, c_attn, head_dim=4)
    assert "mod.blocks.0.attn.in_proj_weight" in legacy
    assert "mod.blocks.0.attn.in_proj.weight" not in legacy
    converted = build_model("part", **kwargs).eval()
    converted.load_state_dict(legacy)
    for k, v in state.items():
        torch.testing.assert_close(converted.state_dict()[k], v, atol=1e-6, rtol=1e-6, msg=k)
    # a current state dict is loaded unchanged (c_attn is kept)
    state2 = {k: v.clone() for k, v in state.items()}
    state2["mod.blocks.0.c_attn"] = torch.full((4,), 2.0)
    converted.load_state_dict(state2)
    assert torch.equal(converted.state_dict()["mod.blocks.0.c_attn"], torch.full((4,), 2.0))


def test_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="jetgen fetch models ParT_kin"):
        load_pretrained("ParT_kin", root=tmp_path)
    with pytest.raises(ValueError, match="did you mean 'ParT_kin'"):
        load_pretrained("ParT_kn", root=tmp_path)  # type: ignore[arg-type]


def compare_scores(root: Path, data: JetData) -> None:
    ref = np.load(SCORES)
    np.testing.assert_array_equal(ref["jet_pt"], np.asarray(data["jet_pt"]))  # the same jets in the same order
    for name in COMPARED:
        model = load_pretrained(name, root=root)
        assert model.n_classes == 10
        assert model.name == name
        assert not model.training
        with torch.no_grad():
            # weaver pads every jet to 128 particles (ParticleNet's kNN can flip near ties at another length)
            probs = torch.softmax(model(make_inputs(model, data, length=128)), dim=1).numpy()
        np.testing.assert_allclose(probs, ref[name], atol=1e-5, rtol=0, err_msg=name)
        labels = data.labels(("label_QCD", "label_Wqq", "label_Tbqq"))
        assert np.mean(np.argmax(probs, axis=1) == np.array([0, 7, 8])[labels]) > 0.6


def test_scores_equal_weaver(jetclass_sample: JetData) -> None:
    missing = [n for n in COMPARED if not (ROOT / f"{n}.pt").is_file()]
    if missing:
        pytest.skip(f"no checkpoints {missing} in {ROOT} (jetgen fetch models {' '.join(missing)})")
    compare_scores(ROOT, jetclass_sample)


@pytest.mark.network
def test_scores_equal_weaver_downloaded(jetclass_sample: JetData, tmp_path: Path) -> None:
    from itj.jetclass import fetch_models

    fetch_models(list(COMPARED), dest=tmp_path, progress=False)
    compare_scores(tmp_path, jetclass_sample)
