"""A multilayer perceptron on jet-level features (``JetBatch.jet``): the baseline of T2.3."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ..features import JET_OBSERVERS, JetBatch
from .base import Tagger, Trace

__all__ = ["MLP"]


class MLP(Tagger):
    """An MLP on jet-level features: input BatchNorm, then (Linear, BatchNorm, ReLU, Dropout) per hidden layer.

    The input BatchNorm standardises the features with running statistics (as the dl4hep ``MLPModule``), so raw
    GeV-valued columns can be fed directly.

    Parameters
    ----------
    jet_features : sequence of str
        The columns of ``JetBatch.jet`` in order (per-jet columns or the derived ``"tau21"``, ``"tau32"``,
        ``"tau43"``): by default the ten JetClass observers and ``tau21``, ``tau32``.
    n_classes : int
        Number of outputs.
    hidden : sequence of int
        Widths of the hidden layers.
    dropout : float
        Dropout after every hidden layer.

    Attributes
    ----------
    input_bn : torch.nn.BatchNorm1d
        The input standardisation.
    hidden : torch.nn.ModuleList
        The hidden layers.
    out : torch.nn.Linear
        The output layer.
    """

    default_name = "mlp"

    def __init__(
        self,
        jet_features: Sequence[str] = (*JET_OBSERVERS, "tau21", "tau32"),
        n_classes: int = 3,
        hidden: Sequence[int] = (128, 128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        names = tuple(jet_features)
        if not names:
            raise ValueError("the MLP needs at least one jet feature")
        self.input_bn = nn.BatchNorm1d(len(names))
        self.hidden = nn.ModuleList()
        n_in = len(names)
        for n_out in hidden:
            layer = nn.Sequential(nn.Linear(n_in, n_out), nn.BatchNorm1d(n_out), nn.ReLU(), nn.Dropout(dropout))
            self.hidden.append(layer)
            n_in = n_out
        self.out = nn.Linear(n_in, n_classes)
        init = {
            "jet_features": names,
            "n_classes": int(n_classes),
            "hidden": tuple(int(h) for h in hidden),
            "dropout": float(dropout),
        }
        self._set_config(init, feature_set=None, jet_features=names)

    @property
    def head(self) -> nn.Linear:
        """The output layer."""
        return self.out

    def _jet(self, batch: JetBatch) -> torch.Tensor:
        assert self.jet_features is not None
        if batch.jet is None:
            raise ValueError(
                "the MLP reads JetBatch.jet, which is None: build the batch with itj.models.make_inputs(model, data) "
                "or JetDataset(..., jet_feature_names=model.jet_features)"
            )
        if batch.jet.shape[1] != len(self.jet_features):
            n = len(self.jet_features)
            raise ValueError(
                f"the MLP expects {n} jet features {self.jet_features}, the batch has {batch.jet.shape[1]}"
            )
        return batch.jet

    def embed(self, batch: JetBatch) -> torch.Tensor:
        """Return the output of the last hidden layer (B, D)."""
        x = self.input_bn(self._jet(batch))
        for layer in self.hidden:
            x = layer(x)
        return x

    def trace(self, batch: JetBatch, *, attention: bool = False) -> Trace:
        """Return the :class:`Trace`: no particles (``mask`` None); pooled ``"hidden{i}"`` per hidden layer."""
        x = self.input_bn(self._jet(batch))
        pooled = {}
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            pooled[f"hidden{i}"] = x
        return Trace(logits=self.out(x), embedding=x, mask=None, pooled=pooled)
