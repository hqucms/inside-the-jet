"""Model presets, building models by name, the pretrained JetClass models and checkpoints."""

from __future__ import annotations

import difflib
import inspect
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch

from .. import __version__
from ..features import FEATURE_SETS
from .base import Tagger
from .deepsets import EFN, PFN
from .mlp import MLP
from .part import ParT
from .particlenet import PN_CONV, PN_LITE_CONV, ParticleNet

__all__ = ["MODEL_PRESETS", "PRETRAINED", "build_model", "load_model", "load_pretrained", "save_model"]

_PART_SMALL: dict[str, Any] = {
    "embed_dims": (64, 256, 64),
    "pair_embed_dims": (32, 32, 32),
    "num_heads": 4,
    "num_layers": 3,
    "num_cls_layers": 1,
}

#: name -> (class, keyword arguments) of every model of the course (models.md has the table with the CPU costs).
#: ``partsmall`` is the course's "ParT-lite" (0.24 M parameters); ``partlite`` is the ParT-lite of dl4hep; ``pn`` and
#: ``part`` are the JetClass architectures of the ParT paper.
MODEL_PRESETS: dict[str, tuple[type[Tagger], dict[str, Any]]] = {
    "mlp": (MLP, {}),
    "pfn": (PFN, {}),
    "efn": (EFN, {}),
    "pnlite": (ParticleNet, {"conv_params": PN_LITE_CONV, "fc_params": ((128, 0.1),), "use_fusion": True}),
    "pn": (ParticleNet, {"conv_params": PN_CONV, "fc_params": ((256, 0.1),), "use_fusion": False}),
    "partsmall": (ParT, dict(_PART_SMALL)),
    "partsmall_nopair": (ParT, {**_PART_SMALL, "pair_embed_dims": None}),
    "partlite": (
        ParT,
        {
            "embed_dims": (128, 512, 128),
            "pair_embed_dims": (64, 64, 64),
            "num_heads": 8,
            "num_layers": 3,
            "num_cls_layers": 1,
        },
    ),
    "part": (ParT, {}),
}

#: The pretrained JetClass models of the ParT repository (10 classes): name -> (preset, feature set).
PRETRAINED: dict[str, tuple[str, str]] = {
    "ParT_kin": ("part", "kin"),
    "ParT_kinpid": ("part", "kinpid"),
    "ParT_full": ("part", "full"),
    "ParticleNet_kin": ("pn", "kin"),
    "ParticleNet_kinpid": ("pn", "kinpid"),
    "ParticleNet_full": ("pn", "full"),
}

PretrainedName = Literal[
    "ParT_kin", "ParT_kinpid", "ParT_full", "ParticleNet_kin", "ParticleNet_kinpid", "ParticleNet_full"
]


def _suggest(kind: str, name: str, choices: Mapping[str, Any] | list[str]) -> str:
    close = difflib.get_close_matches(name, list(choices), n=1)
    hint = f" (did you mean {close[0]!r}?)" if close else ""
    return f"unknown {kind} {name!r}{hint}; choose from {', '.join(choices)}"


def build_model(name: str, *, feature_set: str | None = "kin", n_classes: int = 3, **overrides: Any) -> Tagger:
    """Build a model of :data:`MODEL_PRESETS` (untrained).

    Parameters
    ----------
    name : str
        Preset name (``"mlp"``, ``"pfn"``, ``"efn"``, ``"pnlite"``, ``"pn"``, ``"partsmall"``,
        ``"partsmall_nopair"``, ``"partlite"``, ``"part"``).
    feature_set : str or None
        Particle features (``"kin"``, ``"kinpid"``, ``"full"``); sets the input dimension. Ignored by the MLP.
    n_classes : int
        Number of outputs.
    **overrides
        Constructor arguments that replace the preset values, e.g. ``phi_sizes=(100, 100, 8)``.

    Returns
    -------
    Tagger
        The model; ``build_model(**model.config)`` builds the same architecture again.

    Examples
    --------
    >>> model = build_model("pfn")
    >>> model.name, model.feature_set, model.n_classes, model.n_parameters()
    ('pfn', 'kin', 3, 83971)
    """
    if name not in MODEL_PRESETS:
        raise ValueError(_suggest("model", name, MODEL_PRESETS))
    cls, preset = MODEL_PRESETS[name]
    params = inspect.signature(cls).parameters
    unknown = [k for k in overrides if k not in params or k == "self"]
    if unknown:
        options = [k for k in params if k != "self"]
        hints = [_suggest("argument", k, options) for k in unknown]
        raise TypeError(f"{cls.__name__}: " + "; ".join(hints))
    kwargs = {**preset, **overrides, "n_classes": n_classes}
    if "feature_set" in params:
        kwargs["feature_set"] = feature_set
        for key in ("input_dim", "input_dims"):
            if key in params and key not in overrides:
                if feature_set is None:
                    raise ValueError(f"{name}: give feature_set or {key}")
                if feature_set not in FEATURE_SETS:
                    raise ValueError(_suggest("feature set", feature_set, FEATURE_SETS))
                kwargs[key] = len(FEATURE_SETS[feature_set])
    model = cls(**kwargs)
    model.name = name
    model.config["name"] = name
    return model


def save_model(model: Tagger, path: str | os.PathLike[str], *, extra: dict[str, Any] | None = None) -> Path:
    """Save a model with its configuration (``torch.save``), so that :func:`load_model` rebuilds it.

    Parameters
    ----------
    model : Tagger
        The model.
    path : str or os.PathLike
        Output file (``.pt``); missing parent directories are created.
    extra : dict, optional
        Anything else to keep with the weights (history, training data, seed, ...).

    Returns
    -------
    pathlib.Path
        The path written. The file holds ``{"itj_version", "name", "config", "state_dict", "extra"}``.
    """
    if not isinstance(model, Tagger):
        raise TypeError(
            f"save_model saves an itj Tagger, not a {type(model).__name__}: copy the weights of your own model into "
            "build_model(...) with load_state_dict (the key map of T2.4) and save that, or use torch.save"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "itj_version": __version__,
            "name": model.name,
            "config": model.config,
            "state_dict": state,
            "extra": dict(extra or {}),
        },
        path,
    )
    return path


def _infer_from_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Guess ``n_classes`` and ``feature_set`` of a bare weaver state dict (keys ``mod.*``)."""
    out: dict[str, Any] = {}
    fc = [(int(m.group(1)), k) for k in state if (m := re.fullmatch(r"mod\.fc\.(\d+)\.weight", k))]
    if fc:
        out["n_classes"] = int(state[max(fc)[1]].shape[0])
    for key in ("mod.embed.input_bn.weight", "mod.bn_fts.weight"):
        if key in state:
            n = int(state[key].shape[0])
            for fs, variables in FEATURE_SETS.items():
                if len(variables) == n:
                    out["feature_set"] = fs
    return out


def load_model(
    path: str | os.PathLike[str], *, name: str | None = None, device: str | torch.device = "cpu", **build_kwargs: Any
) -> Tagger:
    """Load a model saved by :func:`save_model`, or a bare weaver state dict (keys ``mod.*``) with ``name``.

    Parameters
    ----------
    path : str or os.PathLike
        A file of :func:`save_model`, or a weaver checkpoint (``--model-prefix`` ``*_state.pt``) of a ParT or
        ParticleNet trained with the networks of ``examples/weaver/networks``.
    name : str, optional
        The :data:`MODEL_PRESETS` architecture of a bare state dict (e.g. ``"partsmall"`` for the T4.6 weaver run);
        ``n_classes`` and ``feature_set`` are read from the weights unless given.
    device : str or torch.device
        Where to put the model.
    **build_kwargs
        Arguments for :func:`build_model` that replace the stored ones.

    Returns
    -------
    Tagger
        The model in eval mode.
    """
    obj = torch.load(Path(path), map_location=device, weights_only=True)
    if isinstance(obj, Mapping) and "state_dict" in obj and "config" in obj:
        config = {**obj["config"], **build_kwargs}
        if name is not None:
            config["name"] = name
        model = build_model(**config)
        model.load_state_dict(obj["state_dict"])
        model.name = str(obj.get("name", model.name))
    elif isinstance(obj, Mapping) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        if name is None:
            raise ValueError(
                f"{path} holds a bare state dict: give the architecture, e.g. load_model(path, name='partsmall')"
            )
        kwargs = {**_infer_from_state_dict(obj), **build_kwargs}
        model = build_model(name, **kwargs)
        model.load_state_dict(dict(obj))
    else:
        raise ValueError(f"{path} is neither an itj checkpoint nor a state dict")
    return model.to(device).eval()


def load_pretrained(
    name: PretrainedName, *, root: str | os.PathLike[str] = "models/pretrained", device: str | torch.device = "cpu"
) -> Tagger:
    """Load a pretrained JetClass model of the ParT paper (10 classes, trained on 100 M JetClass jets).

    Parameters
    ----------
    name : {"ParT_kin", "ParT_kinpid", "ParT_full", "ParticleNet_kin", "ParticleNet_kinpid", "ParticleNet_full"}
        The checkpoint (``<root>/<name>.pt``, downloaded with ``jetgen fetch models <name>``).
    root : str or os.PathLike
        Directory of the checkpoints.
    device : str or torch.device
        Where to put the model.

    Returns
    -------
    Tagger
        The model in eval mode; outputs in the JetClass class order (QCD 0, ..., Wqq 7, Tbqq 8, Tbl 9). Checkpoints of
        weaver-core < 0.5 are converted as weaver 0.5.4 converts them.
    """
    if name not in PRETRAINED:
        raise ValueError(_suggest("pretrained model", name, PRETRAINED))
    path = Path(root) / f"{name}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: download it with `jetgen fetch models {name}`")
    preset, feature_set = PRETRAINED[name]
    model = build_model(preset, feature_set=feature_set, n_classes=10)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.name = name
    return model.to(device).eval()
