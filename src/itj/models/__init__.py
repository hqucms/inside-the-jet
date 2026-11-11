"""Jet taggers with one model API: MLP, PFN, EFN, ParticleNet and the Particle Transformer (ParT).

Every model is a :class:`Tagger` that takes a :class:`~itj.features.JetBatch`::

    logits = model(batch)                         # (B, C)
    emb = model.embed(batch)                      # (B, D), the input of the final Linear layer
    tr = model.trace(batch, attention=True)       # Trace: layers, pooled layers, attention, U (ParT)

Models are built by name (:data:`MODEL_PRESETS`, :func:`build_model`), saved and loaded with their configuration
(:func:`save_model`, :func:`load_model`), and the pretrained JetClass models of the ParT paper load with
:func:`load_pretrained`. ParticleNet and ParT are vendored from weaver-core v0.5.4 (NOTICE.md), so weaver checkpoints
load unchanged and the outputs equal weaver's.

Examples
--------
>>> import numpy as np
>>> from itj.data import JetData, Ragged
>>> from itj.models import build_model, make_inputs
>>> rng = np.random.default_rng(0)
>>> counts = [5, 3]
>>> pt = Ragged.from_counts(rng.uniform(10, 100, 8).astype(np.float32), counts)
>>> ang = Ragged.from_counts(rng.normal(0, 0.2, 8).astype(np.float32), counts)
>>> data = JetData({"part_px": pt, "part_py": ang.map(lambda a: a * 10), "part_pz": ang, "part_energy": pt.map(
...     lambda p: p * 1.5), "part_deta": ang, "part_dphi": ang, "jet_pt": np.array([300.0, 150.0], np.float32),
...     "jet_energy": np.array([450.0, 225.0], np.float32)})
>>> model = build_model("partsmall").eval()
>>> batch = make_inputs(model, data)
>>> model(batch).shape, model.embed(batch).shape, model.trace(batch, attention=True).attention["block0"].shape
(torch.Size([2, 3]), torch.Size([2, 64]), torch.Size([2, 4, 5, 5]))
"""

from __future__ import annotations

from ..data import JetData
from ..features import JetBatch, make_batch
from .base import Tagger, Trace
from .deepsets import EFN, PFN
from .mlp import MLP
from .part import ParT
from .particlenet import ParticleNet
from .registry import MODEL_PRESETS, PRETRAINED, build_model, load_model, load_pretrained, save_model

__all__ = [
    "EFN",
    "MLP",
    "MODEL_PRESETS",
    "PFN",
    "PRETRAINED",
    "ParT",
    "ParticleNet",
    "Tagger",
    "Trace",
    "build_model",
    "load_model",
    "load_pretrained",
    "make_inputs",
    "save_model",
]


def make_inputs(model: Tagger, data: JetData, *, label_set: str | None = None, length: int | None = None) -> JetBatch:
    """Return the batch ``model`` needs for ``data``: its feature set and jet features.

    Parameters
    ----------
    model : Tagger
        The model (any module with ``feature_set`` and ``jet_features`` attributes).
    data : JetData
        The jets (all of them go into one batch: chunk large data sets, e.g. with ``data.chunks(1024)``).
    label_set : str, optional
        Labels for ``batch.label`` (``"jets3"``, ``"jetclass10"``, ``"qg"``); None: no labels.
    length : int, optional
        Padded length; None pads to the longest jet (at most 128).

    Returns
    -------
    JetBatch
        The batch.
    """
    jet_features = getattr(model, "jet_features", None)
    return make_batch(
        data,
        getattr(model, "feature_set", None),
        label_set=label_set,
        length=length,
        jet_feature_names=None if jet_features is None else tuple(jet_features),
    )
