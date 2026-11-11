# mypy: disable-error-code="arg-type,assignment,attr-defined,list-item,operator,union-attr"
"""ParticleNet, vendored from weaver-core v0.5.4, and the itj wrapper :class:`ParticleNet`.

Paper: "ParticleNet: Jet Tagging via Particle Clouds" - https://arxiv.org/abs/1902.08570

Adapted from the DGCNN implementation in https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.

The functions and classes down to the weaver ``ParticleNet`` (here :class:`ParticleNetModule`) are copied from
``weaver/nn/model/ParticleNet.py`` of weaver-core v0.5.4 (https://github.com/hqucms/weaver-core, MIT licence,
Copyright (c) 2020 Huilin Qu; see NOTICE.md) with the line structure kept. Removed: segmentation, ``compile_model``
and the ``ParticleNetTagger``/``FeatureConv`` classes. Added (marked ``# itj:``): the class is renamed
``ParticleNetModule`` (the itj wrapper is ``ParticleNet``), an optional ``record`` dict of the intermediate outputs,
and ``trim`` defaults to False (weaver: True). The state-dict keys are those of weaver.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..features import JetBatch
from .base import Tagger, Trace, check_feature_set, pooled_layers
from .part import SequenceTrimmer, delta_phi

_logger = logging.getLogger("itj")

__all__ = ["ParticleNet", "ParticleNetModule"]


def pairwise_distance(x):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    return (xx.transpose(2, 1) + inner + xx).clamp(min=0)


def pairwise_distance_etaphi(x):
    eta, phi = x.split((1, 1), dim=1)
    dphi = delta_phi(phi.unsqueeze(-1), phi.unsqueeze(-2))
    dist = dphi.square().sum(dim=1) + pairwise_distance(eta)
    return dist


def knn(x, k, distance_fn=pairwise_distance, exclude_self=True):
    # distance: (b, i, j)
    if exclude_self:
        idx = distance_fn(x).topk(k=k + 1, dim=-1, largest=False, sorted=True)[1][:, :, 1:]  # (b, i, k)
    else:
        idx = distance_fn(x).topk(k=k, dim=-1, largest=False, sorted=True)[1]  # (b, i, k)
    return idx


def gather(x, k, idx, cpu_mode=False):
    batch_size, num_dims, num_points = x.size()
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)  # (b*i*k,)
    if cpu_mode:
        # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
        fts = x.transpose(0, 1).reshape(num_dims, -1)
        # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
        fts = fts[:, idx].view(num_dims, batch_size, num_points, k)
        fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)
    else:
        # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
        fts = x.transpose(2, 1).reshape(-1, num_dims)
        # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
        fts = fts[idx, :].view(batch_size, num_points, k, num_dims)
        fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    return fts


def get_graph_feature(x, k, idx, cpu_mode=False):
    batch_size, num_dims, num_points = x.size()
    fts = gather(x, k, idx, cpu_mode=cpu_mode)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()  # noqa: UP008
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = partial(get_graph_feature, cpu_mode=cpu_mode)

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(
                nn.Conv2d(
                    2 * in_feat if i == 0 else out_feats[i - 1],
                    out_feats[i],
                    kernel_size=1,
                    bias=False if self.batch_norm else True,
                )
            )

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNetModule(nn.Module):  # itj: weaver's `ParticleNet`
    def __init__(
        self,
        input_dims,
        num_classes,
        conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
        fc_params=[(128, 0.1)],
        use_fusion=True,
        use_fts_bn=True,
        use_counts=True,
        trim=False,  # itj: weaver default True
        for_inference=False,
        use_amp=False,
        **kwargs,
    ):
        super(ParticleNetModule, self).__init__(**kwargs)  # noqa: UP008

        _logger.debug("ParticleNet init-ed: %s", locals())

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(
                nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU()
            )

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference
        self.use_amp = use_amp
        self.coord_shift = 99.0 if use_amp else 1e9

    def forward(self, points, features, mask=None, record=None):
        # print('points:\n', points)
        # print('features:\n', features)
        # itj: record (dict, optional) receives mask, edgeconv{i}, fusion and pooled
        with torch.set_grad_enabled(features.requires_grad):
            points, features, mask, _ = self.trimmer(points, features, mask=mask)
            # padding_mask: False = real particle, True = padded particles
            padding_mask = ~mask

            if self.use_counts:
                counts = mask.float().sum(dim=-1)
                counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features).masked_fill(padding_mask, 0)
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts).masked_fill(padding_mask, self.coord_shift)
            fts = conv(pts, fts).masked_fill(padding_mask, 0)
            if self.use_fusion:
                outputs.append(fts)
            if record is not None:  # itj
                record[f"edgeconv{idx}"] = fts
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)).masked_fill(padding_mask, 0)
            if record is not None:  # itj
                record["fusion"] = fts

        # assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)

        if self.use_counts:
            x = fts.sum(dim=-1) / counts  # divide by the real counts
        else:
            x = fts.mean(dim=-1)
        if record is not None:  # itj
            record.update(mask=mask.squeeze(1), pooled=x)

        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output


# ----------------------------------------------------------------------------------------------------------------------
# itj wrapper (not part of weaver)

#: kNN graphs and widths of the two ParticleNet presets (weaver's ``ParticleNet`` defaults are ParticleNet-lite).
PN_LITE_CONV = ((7, (32, 32, 32)), (7, (64, 64, 64)))
PN_CONV = ((16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256)))


class ParticleNet(Tagger):
    """ParticleNet: the vendored weaver EdgeConv network behind the itj model API.

    Each EdgeConv block builds the k-nearest-neighbour graph of the particles (in (Delta eta, Delta phi) for the
    first block, in the feature space of the previous block afterwards), applies an MLP to every edge
    ``(x_i, x_j - x_i)`` and averages over the k neighbours; the particles are then averaged over the jet and an MLP
    gives the logits. The state-dict keys are ``mod.*`` as in the ParT repository's ``example_ParticleNet.py``.

    Parameters
    ----------
    input_dims : int
        Number of particle features (7 kin, 13 kinpid, 17 full).
    n_classes : int
        Number of outputs.
    conv_params : sequence of (int, sequence of int)
        (k, widths) of every EdgeConv block (default: ParticleNet-lite; JetClass ParticleNet: ``PN_CONV``).
    fc_params : sequence of (int, float)
        (width, dropout) of the hidden layers after the pooling.
    use_fusion : bool
        Concatenate the outputs of every block and mix them with a 1x1 convolution before the pooling.
    use_fts_bn : bool
        BatchNorm of the input features.
    use_counts : bool
        Pool with the mean over the real particles (otherwise over all padded slots).
    feature_set : str or None
        The particle features the model reads.

    Attributes
    ----------
    mod : ParticleNetModule
        The vendored weaver module.
    """

    default_name = "pnlite"

    def __init__(
        self,
        input_dims: int = 7,
        n_classes: int = 3,
        conv_params: Sequence[tuple[int, Sequence[int]]] = PN_LITE_CONV,
        fc_params: Sequence[tuple[int, float]] = ((128, 0.1),),
        use_fusion: bool = True,
        use_fts_bn: bool = True,
        use_counts: bool = True,
        feature_set: str | None = "kin",
    ) -> None:
        super().__init__()
        check_feature_set(feature_set, input_dims)
        conv = tuple((int(k), tuple(int(c) for c in widths)) for k, widths in conv_params)
        fc = tuple((int(w), float(d)) for w, d in fc_params)
        if not conv or not fc:
            raise ValueError("ParticleNet needs at least one EdgeConv block and one fc layer")
        self.input_dims = int(input_dims)
        self.k_max = max(k for k, _ in conv)
        self.mod = ParticleNetModule(
            input_dims=input_dims,
            num_classes=n_classes,
            conv_params=[(k, list(w)) for k, w in conv],
            fc_params=[list(p) for p in fc],
            use_fusion=use_fusion,
            use_fts_bn=use_fts_bn,
            use_counts=use_counts,
            trim=False,
            for_inference=False,
        )
        init = {
            "input_dims": int(input_dims),
            "n_classes": int(n_classes),
            "conv_params": conv,
            "fc_params": fc,
            "use_fusion": use_fusion,
            "use_fts_bn": use_fts_bn,
            "use_counts": use_counts,
            "feature_set": feature_set,
        }
        self._set_config(init, feature_set=feature_set, jet_features=None)

    @property
    def head(self) -> nn.Linear:
        """The output Linear layer of ``mod.fc``."""
        layer = self.mod.fc[-1]
        assert isinstance(layer, nn.Linear)
        return layer

    def _inputs(self, batch: JetBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (points, features, mask), zero-padded to at least k + 1 slots (the kNN needs k + 1 points).

        Extra padded slots do not change the outputs: padded particles have zero features and all sit at the same
        far-away coordinate, so whichever of them the kNN picks contributes the same.
        """
        self._check_features(batch, self.input_dims)
        points, features, mask = batch.points, batch.features, batch.mask
        missing = self.k_max + 1 - points.shape[-1]
        if missing > 0:
            pad = torch.nn.functional.pad
            points, features, mask = pad(points, (0, missing)), pad(features, (0, missing)), pad(mask, (0, missing))
        return points, features, mask

    def forward(self, batch: JetBatch) -> torch.Tensor:
        """Return the logits (B, C): the vendored weaver forward on (points, features, mask)."""
        points, features, mask = self._inputs(batch)
        return self.mod(points, features, mask)

    def embed(self, batch: JetBatch) -> torch.Tensor:
        """Return the embedding (B, D): the output of the hidden fc layers (input of the output layer)."""
        points, features, mask = self._inputs(batch)
        record: dict[str, Any] = {}
        self.mod(points, features, mask, record=record)
        return self.mod.fc[:-1](record["pooled"])

    def trace(self, batch: JetBatch, *, attention: bool = False) -> Trace:
        """Return the :class:`Trace`: layers ``"edgeconv{i}"`` (and ``"fusion"``) as (B, P, C)."""
        points, features, mask = self._inputs(batch)
        record: dict[str, Any] = {}
        logits = self.mod(points, features, mask, record=record)
        p = batch.points.shape[-1]  # drop the slots added for the kNN
        m = record["mask"][:, :p]
        layers = {k: v[:, :, :p].transpose(1, 2) for k, v in record.items() if k.startswith(("edgeconv", "fusion"))}
        return Trace(
            logits=logits,
            embedding=self.mod.fc[:-1](record["pooled"]),
            mask=m,
            layers=layers,
            pooled=pooled_layers(layers, m),
        )
