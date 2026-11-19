"""The model API shared by every tagger: :class:`Tagger` and the :class:`Trace` of one forward pass.

Every tagger takes a :class:`~itj.features.JetBatch` and has three views of the same computation::

    model(batch)          logits (B, C)                       a plain Tensor: autograd and captum work on it
    model.embed(batch)    embedding (B, D)                    the input of the final Linear layer
    model.trace(batch)    Trace: logits, embedding, per-particle layers, pooled layers, attention, U

so that ``model.head(model.embed(batch)) == model(batch)`` for every model.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from torch import nn

from ..features import FEATURE_SETS, JetBatch

__all__ = ["Tagger", "Trace", "masked_mean"]


@dataclass
class Trace:
    """Everything one forward pass of a :class:`Tagger` computes, for probes and interpretability.

    Attributes
    ----------
    logits : torch.Tensor
        (B, C) the output of the model.
    embedding : torch.Tensor
        (B, D) the input of the final Linear layer: ``model.head(embedding) == logits``.
    mask : torch.Tensor or None
        (B, P) bool, True for a particle (None for the MLP).
    layers : dict of str to torch.Tensor
        Per-particle outputs (B, P, D_l), zero at padded slots: ``"phi"`` (PFN, EFN), ``"edgeconv{i}"`` and
        ``"fusion"`` (ParticleNet), ``"embed"`` and ``"block{i}"`` (ParT).
    pooled : dict of str to torch.Tensor
        (B, D_l) the masked mean over the particles of every entry of ``layers``; PFN and EFN also ``"latent"`` (the
        sum that F reads), ParT also ``"cls{i}"`` (the class token after class-attention block i), the MLP
        ``"hidden{i}"`` (its hidden layers).
    attention : dict of str to torch.Tensor
        ParT only, with ``attention=True``: ``"block{i}"`` (B, H, P, P); each row is a softmax over the real
        particles (padded columns are 0).
    class_attention : dict of str to torch.Tensor
        ParT only, with ``attention=True``: ``"cls{i}"`` (B, H, 1, 1 + P); column 0 is the class token itself.
    pair_bias : torch.Tensor or None
        ParT only: the interaction bias U = output of the pair embedding (B, H, P, P), zero for pairs with a padded
        particle (None without pair features).
    """

    logits: torch.Tensor
    embedding: torch.Tensor
    mask: torch.Tensor | None = None
    layers: dict[str, torch.Tensor] = field(default_factory=dict)
    pooled: dict[str, torch.Tensor] = field(default_factory=dict)
    attention: dict[str, torch.Tensor] = field(default_factory=dict)
    class_attention: dict[str, torch.Tensor] = field(default_factory=dict)
    pair_bias: torch.Tensor | None = None

    def detach(self) -> Trace:
        """Return a copy with every tensor detached from the autograd graph."""

        def d(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.detach()

        return Trace(
            logits=self.logits.detach(),
            embedding=self.embedding.detach(),
            mask=d(self.mask),
            layers={k: v.detach() for k, v in self.layers.items()},
            pooled={k: v.detach() for k, v in self.pooled.items()},
            attention={k: v.detach() for k, v in self.attention.items()},
            class_attention={k: v.detach() for k, v in self.class_attention.items()},
            pair_bias=d(self.pair_bias),
        )


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` (B, P, D) over the particles where ``mask`` (B, P) is True; (B, D), 0 for an empty jet."""
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class Tagger(nn.Module):
    """Base class of the itj taggers: one API for the MLP, PFN, EFN, ParticleNet and ParT.

    A subclass implements :meth:`embed` (or :meth:`trace`) and :attr:`head`; :meth:`forward` is then
    ``head(embed(batch))``. Subclasses call :meth:`_set_config` at the end of ``__init__`` with every constructor
    argument, so that ``build_model(**model.config)`` rebuilds the same architecture.

    Attributes
    ----------
    name : str
        The :data:`~itj.models.MODEL_PRESETS` name the model was built from (``"pfn"``, ``"partsmall"``, ...).
    feature_set : str or None
        The particle features the model reads (``"kin"``, ``"kinpid"``, ``"full"``); None for jet-level models.
    jet_features : tuple of str or None
        The jet-level features the model reads (``JetBatch.jet``); None for particle models.
    n_classes : int
        Number of outputs.
    config : dict
        ``{"name", "feature_set", "n_classes", ...}`` with every constructor argument: ``build_model(**config)``
        rebuilds the architecture.
    """

    #: The preset name used when the class is instantiated directly.
    default_name: ClassVar[str] = "tagger"

    name: str
    feature_set: str | None
    jet_features: tuple[str, ...] | None
    n_classes: int
    config: dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        self.name = self.default_name
        self.feature_set = None
        self.jet_features = None
        self.n_classes = 0
        self.config = {}

    # ------------------------------------------------------------------------------------------------- the API

    @property
    def head(self) -> nn.Linear:
        """The final Linear layer (``head(embed(batch)) == forward(batch)``)."""
        raise NotImplementedError

    def forward(self, batch: JetBatch) -> torch.Tensor:
        """Return the logits (B, C) of the jets of ``batch``."""
        return self.head(self.embed(batch))

    def embed(self, batch: JetBatch) -> torch.Tensor:
        """Return the embedding (B, D): the input of the final Linear layer."""
        return self.trace(batch).embedding

    def trace(self, batch: JetBatch, *, attention: bool = False) -> Trace:
        """Return the :class:`Trace` of one forward pass.

        Parameters
        ----------
        batch : JetBatch
            The jets.
        attention : bool
            Also return the attention maps of ParT (B, H, P, P) per block: memory grows with P squared, so they are
            computed only on request.

        Returns
        -------
        Trace
            Logits, embedding, per-particle and pooled layers (and attention, U for ParT).
        """
        raise NotImplementedError

    def n_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_parameters(self, seed: int = 0) -> None:
        """Re-initialise every weight as a new model of the same configuration built with ``torch.manual_seed(seed)``.

        Running statistics of the BatchNorm layers are reset too. Used for the random-weight control of probes
        (W3.1.2, W3.4.7); the global torch random state is not changed.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            fresh = type(self)(**self._init_kwargs)
        device = next((p.device for p in self.parameters()), torch.device("cpu"))
        self.load_state_dict(fresh.to(device).state_dict())

    def extra_repr(self) -> str:
        """Return the name, feature set and number of classes for ``repr(model)``."""
        return f"name={self.name!r}, feature_set={self.feature_set!r}, n_classes={self.n_classes}"

    # -------------------------------------------------------------------------------------------------- helpers

    def _set_config(
        self, init: Mapping[str, Any], *, feature_set: str | None, jet_features: tuple[str, ...] | None
    ) -> None:
        """Record the constructor arguments ``init`` (subclasses call this at the end of ``__init__``)."""
        self.feature_set = feature_set
        self.jet_features = jet_features
        self.n_classes = int(init["n_classes"])
        self._init_kwargs = copy.deepcopy(dict(init))
        self.config = {"name": self.name, **copy.deepcopy(dict(init))}

    def _check_features(self, batch: JetBatch, input_dim: int) -> None:
        """Raise a ValueError when the batch has the wrong number of particle features."""
        n = int(batch.features.shape[1])
        if n != input_dim:
            have = f" ({batch.feature_set})" if batch.feature_set else ""
            want = f" ({self.feature_set})" if self.feature_set else ""
            raise ValueError(
                f"{type(self).__name__} expects {input_dim} particle features{want}, the batch has {n}{have}; "
                "build the batch with itj.models.make_inputs(model, data)"
            )


def check_feature_set(feature_set: str | None, input_dim: int) -> None:
    """Raise a ValueError when ``input_dim`` does not match the size of ``feature_set``."""
    if feature_set is None:
        return
    if feature_set not in FEATURE_SETS:
        import difflib

        close = difflib.get_close_matches(feature_set, FEATURE_SETS, n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        raise ValueError(f"unknown feature set {feature_set!r}{hint}; choose from {sorted(FEATURE_SETS)}")
    n = len(FEATURE_SETS[feature_set])
    if n != input_dim:
        raise ValueError(f"feature set {feature_set!r} has {n} features, but input_dim is {input_dim}")


def pooled_layers(layers: Mapping[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the masked mean of every per-particle layer."""
    return {k: masked_mean(v, mask) for k, v in layers.items()}
