"""Compatibility with weaver-core 0.5.4 (``weaver`` marker; needs the weaver extra).

* the vendored ParticleTransformer and ParticleNet give weaver's outputs with the same weights (1e-6), also for
  checkpoints in the key format of weaver-core < 0.5;
* weaver reads the split files of :func:`itj.io.write_split` with ``jets3_kin.yaml``, trains two CPU steps, and its
  predict run on the test files returns the observers of ``load(..., part="test")`` in the same order, with the
  scores of the same weights in :mod:`itj.models`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from itj.data import JetData
from itj.features import JET_OBSERVERS, config_path
from itj.io import load, save, write_split
from itj.models import build_model, load_model, make_inputs

pytestmark = pytest.mark.weaver

weaver_pt = pytest.importorskip("weaver.nn.model.ParticleTransformer")
weaver_pn = pytest.importorskip("weaver.nn.model.ParticleNet")

#: A small ParT of the course API and the same network as a weaver network config.
SMALL_PART: dict[str, Any] = {
    "embed_dims": (16, 32, 16),
    "pair_embed_dims": (8, 8),
    "num_heads": 2,
    "num_layers": 2,
    "num_cls_layers": 1,
}
NETWORK = """
import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"mod.cls_token"}

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):
    cfg = dict(input_dim=len(data_config.input_dicts["pf_features"]), num_classes=len(data_config.label_value),
               pair_input_dim=4, use_pre_activation_pair=False, embed_dims=[16, 32, 16], pair_embed_dims=[8, 8],
               num_heads=2, num_layers=2, num_cls_layers=1, fc_params=[], activation="gelu", trim=False,
               cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0})
    model = ParticleTransformerWrapper(**cfg)
    info = {"input_names": list(data_config.input_names),
            "input_shapes": {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
            "output_names": ["softmax"],
            "dynamic_axes": {**{k: {0: "N", 2: "n_" + k.split("_")[0]} for k in data_config.input_names},
                             "softmax": {0: "N"}}}
    return model, info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
"""


@pytest.fixture(autouse=True)
def _quiet_weaver() -> None:
    logging.getLogger("weaver").setLevel(logging.WARNING)


def weaver_part(**kwargs: object) -> torch.nn.Module:
    return weaver_pt.ParticleTransformer(trim=False, **kwargs)


@pytest.mark.parametrize(
    "preset",
    [
        ("partsmall", {}),
        ("part", {}),
        ("partsmall_nopair", {}),
        ("partsmall", {"pair_input_dim": 1}),
        ("partlite", {"fc_params": ((32, 0.1),)}),
    ],
    ids=["partsmall", "part", "nopair", "pair1", "partlite_fc"],
)
def test_part_equals_weaver(preset: tuple[str, dict], jetclass_sample: JetData) -> None:
    name, overrides = preset
    torch.manual_seed(0)
    model = build_model(name, n_classes=10, **overrides).eval()
    cfg = model.config
    ref = weaver_part(
        input_dim=7,
        num_classes=10,
        pair_input_dim=cfg["pair_input_dim"],
        use_pre_activation_pair=False,
        embed_dims=list(cfg["embed_dims"]),
        pair_embed_dims=None if cfg["pair_embed_dims"] is None else list(cfg["pair_embed_dims"]),
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        num_cls_layers=cfg["num_cls_layers"],
        fc_params=[tuple(p) for p in cfg["fc_params"]],
        cls_block_params=dict(cfg["cls_block_params"]),
    ).eval()
    ref.load_state_dict({k[4:]: v for k, v in model.state_dict().items()})
    batch = make_inputs(model, jetclass_sample, length=128)
    with torch.no_grad():
        expected = ref(batch.features, v=batch.vectors, mask=batch.mask)
        torch.testing.assert_close(model(batch), expected, atol=1e-6, rtol=1e-6)
        # the explicit softmax of the traced path agrees with the fused attention to float precision
        torch.testing.assert_close(model.trace(batch, attention=True).logits, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name", ["pnlite", "pn"])
def test_particlenet_equals_weaver(name: str, jetclass_sample: JetData) -> None:
    torch.manual_seed(0)
    model = build_model(name, n_classes=10).eval()
    cfg = model.config
    ref = weaver_pn.ParticleNet(
        input_dims=7,
        num_classes=10,
        conv_params=[(k, list(w)) for k, w in cfg["conv_params"]],
        fc_params=[tuple(p) for p in cfg["fc_params"]],
        use_fusion=cfg["use_fusion"],
        trim=False,
    ).eval()
    ref.load_state_dict({k[4:]: v for k, v in model.state_dict().items()})
    batch = make_inputs(model, jetclass_sample, length=128)
    with torch.no_grad():
        expected = ref(batch.points, batch.features, batch.mask)
        torch.testing.assert_close(model(batch), expected, atol=1e-6, rtol=1e-6)
    # the training forward too (BatchNorm with batch statistics; dropout off)
    model.train()
    ref.train()
    for m in (*model.modules(), *ref.modules()):
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    torch.testing.assert_close(model(batch), ref(batch.points, batch.features, batch.mask), atol=1e-5, rtol=1e-5)


def test_legacy_checkpoint_loads_as_in_weaver(jetclass_sample: JetData) -> None:
    from test_pretrained import to_legacy

    torch.manual_seed(0)
    model = build_model("partsmall", n_classes=10).eval()
    state = model.state_dict()
    prefixes = [k[: -len("c_attn")] for k in state if k.endswith("c_attn")]
    gen = torch.Generator().manual_seed(2)
    legacy = to_legacy(state, {p: torch.rand(4, generator=gen) + 0.5 for p in prefixes}, head_dim=16)
    cfg = model.config
    ref = weaver_part(
        input_dim=7,
        num_classes=10,
        use_pre_activation_pair=False,
        embed_dims=list(cfg["embed_dims"]),
        pair_embed_dims=list(cfg["pair_embed_dims"]),
        num_heads=4,
        num_layers=3,
        num_cls_layers=1,
        fc_params=[],
        cls_block_params=dict(cfg["cls_block_params"]),
    ).eval()
    ref.load_state_dict({k[4:]: v.clone() for k, v in legacy.items()})
    ours = build_model("partsmall", n_classes=10).eval()
    ours.load_state_dict({k: v.clone() for k, v in legacy.items()})
    batch = make_inputs(model, jetclass_sample.select(np.arange(0, 150, 5)), length=128)
    with torch.no_grad():
        torch.testing.assert_close(
            ours(batch), ref(batch.features, v=batch.vectors, mask=batch.mask), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(ours(batch), model(batch), atol=1e-5, rtol=1e-5)


def run_weaver(args: list[str], cwd: Path) -> None:
    code = "import sys; from weaver.train import main; sys.argv[0] = 'weaver'; main()"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, "-c", code, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=600, check=False
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.fixture(scope="module")
def class_files(jetclass_sample: JetData, tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Three per-class files (50 jets each): the toy's three processes if the generator is there, else JetClass."""
    root = tmp_path_factory.mktemp("weaver_inputs")
    try:
        from itj.toy import generate

        data = generate(150, seed=5)
        groups = [np.flatnonzero(data.labels() == c) for c in range(3)]
    except (ImportError, AttributeError, TypeError, NotImplementedError):  # the generator is not there (yet)
        data = jetclass_sample
        groups = [np.arange(0, 50), np.arange(50, 100), np.arange(100, 150)]
    return [
        save(data.select(rows), root / f"{name}.parquet")
        for name, rows in zip(("qcd", "wqq", "tbqq"), groups, strict=True)
    ]


def test_weaver_trains_and_predicts_on_split_files(class_files: list[Path], tmp_path: Path) -> None:
    split = write_split(class_files, tmp_path / "split")
    groups = [p.stem for p in class_files]

    def named(part: str) -> list[str]:
        """One named weaver group per file (weaver sorts the files of a group by name)."""
        return [f"{g}:{p}" for g, p in zip(groups, split[part], strict=True)]

    shutil.copy(config_path("jets3_kin"), tmp_path / "jets3_kin.yaml")  # weaver writes its .auto.yaml next to it
    (tmp_path / "network.py").write_text(textwrap.dedent(NETWORK))
    common = [
        "--data-config",
        "jets3_kin.yaml",
        "--network-config",
        "network.py",
        "--gpus",
        "",
        "--num-workers",
        "0",
        "--batch-size",
        "8",
    ]
    run_weaver(
        [
            "--data-train",
            *named("train"),
            "--data-val",
            *named("val"),
            *common,
            "--num-epochs",
            "1",
            "--steps-per-epoch",
            "2",
            "--steps-per-epoch-val",
            "1",
            "--fetch-step",
            "1",
            "--optimizer",
            "adam",
            "--start-lr",
            "1e-3",
            "--model-prefix",
            str(tmp_path / "train" / "net"),
        ],
        tmp_path,
    )
    state = tmp_path / "train" / "net_epoch-0_state.pt"
    assert state.is_file()
    run_weaver(
        [
            "--predict",
            "--data-test",
            *named("test"),
            *common,
            "--model-prefix",
            str(state),
            "--predict-output",
            str(tmp_path / "pred.parquet"),
        ],
        tmp_path,
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    # weaver writes one prediction file per named group: pred_<group>.parquet
    pred = pa.concat_tables([pq.read_table(tmp_path / f"pred_{g}.parquet") for g in groups])
    test = load(class_files, part="test")
    assert pred.num_rows == len(test)
    for name in JET_OBSERVERS:
        np.testing.assert_array_equal(pred.column(name).to_numpy(), np.asarray(test[name]), err_msg=name)
    np.testing.assert_array_equal(np.asarray(pred.column("_label_").to_numpy()), test.labels())
    # the trained weights in itj give weaver's scores
    model = load_model(state, name="partsmall", **SMALL_PART)
    assert model.n_classes == 3
    assert model.feature_set == "kin"
    with torch.no_grad():
        probs = torch.softmax(model(make_inputs(model, test, length=128)), dim=1).numpy()
    scores = np.stack(pred.column("scores").to_numpy(zero_copy_only=False))
    np.testing.assert_allclose(probs, scores, atol=1e-5, rtol=0)
