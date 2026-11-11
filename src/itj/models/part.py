# mypy: disable-error-code="arg-type,assignment,attr-defined,list-item,operator,union-attr"
"""Particle Transformer (ParT), vendored from weaver-core v0.5.4, and the itj wrapper :class:`ParT`.

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772

The classes down to :class:`ParticleTransformer` are copied from ``weaver/nn/model/ParticleTransformer.py`` of
weaver-core v0.5.4 (https://github.com/hqucms/weaver-core, MIT licence, Copyright (c) 2020 Huilin Qu; see NOTICE.md)
with the line structure kept, so that ``diff`` against the original stays readable. Removed: the versions 2 to 3.6
(SwiGLU, RMSNorm, QK norm, attention gates, layer scale, drop path, global token), the ``ee`` and ``xyzt`` pair
features, segmentation and the ``ParticleTransformerTagger`` classes. Added (each marked ``# itj:``):

- ``finite_grad``: ``delta_r2 + 1e-30`` before the square root, so that gradients with respect to the four-vectors
  are finite on the diagonal (Delta = 0); the outputs do not change (Delta is clamped at 1e-8 anyway);
- ``PairEmbed.channel_fill``: replaces pair-feature channels by constants before the pair embedding (pair-feature
  ablation, :func:`itj.interpret.ablate_pair_channels`);
- ``Attention.keep_weights``: computes the attention weights explicitly and keeps them (otherwise the fused
  ``scaled_dot_product_attention``; the outputs agree to float precision);
- ``record``: an optional dict filled with the intermediate outputs (:meth:`ParT.trace`);
- ``trim`` defaults to False (weaver: True), so that the padded length never changes inside the model.

The state-dict keys are those of weaver, including the conversion of checkpoints written by weaver-core < 0.5
(``_load_from_state_dict``), so the pretrained JetClass models of the ParT repository load unchanged.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features import JetBatch
from .base import Tagger, Trace, check_feature_set, pooled_layers

_logger = logging.getLogger("itj")

__all__ = ["ParT", "ParticleTransformer", "pairwise_lv_fts_pp"]


def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2


def to_pt2(x, eps=1e-8):
    assert x.size(1) >= 2
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_p2(x, eps=1e-8):
    assert x.size(1) >= 3
    p2 = x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        p2 = p2.clamp(min=eps)
    return p2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - to_p2(x, eps=None)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def to_ptrapphim(x, return_mass=True, eps=1e-8):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps) ** (-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts_pp(xi, xj, num_outputs=4, eps=1e-8, finite_grad=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None).split((1, 1, 1), dim=1)

    # itj: finite_grad keeps the gradient of sqrt finite at Delta = 0 (the diagonal); never 1e-16 (it changes Delta)
    delta = (delta_r2(rapi, phii, rapj, phij) + 1e-30).sqrt() if finite_grad else delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)


def _fill_channels(x, fill):
    # itj: replace channels of the pair features (dim 1) by constants (pair-feature ablation)
    if not fill:
        return x
    x = x.clone()
    for channel, value in fill.items():
        x[:, channel] = value
    return x


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat(
        (
            torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
            torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
            idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
            idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
        ),
        dim=0,
    )
    return torch.sparse_coo_tensor(
        i, uu.flatten(), size=(batch_size, num_fts, seq_len + 1, seq_len + 1), device=uu.device
    ).to_dense()[:, :, :seq_len, :seq_len]


def tril_indices(row, col, offset=0, *, dtype=torch.long, device="cpu"):
    return torch.ones(row, col, dtype=dtype, device=device).tril(offset).nonzero().T


class SequenceTrimmer(nn.Module):
    def __init__(
        self, enabled=False, target=(0.9, 1.02), warmup_steps=5, round_to_32=False, num_extra_tokens=0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self.warmup_steps = warmup_steps
        self.round_to_32 = round_to_32
        self.num_extra_tokens = num_extra_tokens
        self.register_buffer("_counter", torch.LongTensor([0]), persistent=False)

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < self.warmup_steps:
                self._counter.add_(1)
            else:
                seq_len = mask.size(-1)
                if v is not None:
                    if not isinstance(v, (list, tuple)):
                        v = [v]
                if self.training:
                    # Use torch RNG instead of Python random to avoid graph breaks
                    q = torch.empty(1, device=mask.device).uniform_(*self.target).clamp_(max=1)
                    maxlen = torch.quantile(mask.float().sum(dim=-1), q).long()
                    if self.round_to_32:
                        # Round up to next multiple of 32
                        # effectively: ceil(x / 32) * 32
                        target_len = maxlen + self.num_extra_tokens
                        target_len = ((target_len + 31) // 32) * 32
                        target_len = torch.clamp(target_len, min=32)
                        maxlen = torch.clamp(target_len, max=seq_len) - self.num_extra_tokens
                    rand = torch.rand_like(mask.float())
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = [torch.gather(_v, -1, perm.expand_as(_v)) for _v in v]
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = maxlen.clamp(min=1)
                if maxlen < seq_len:
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = [_v[:, :, :maxlen] for _v in v]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]
                if v is not None:
                    if len(v) == 1:
                        v = v[0]

        return x, v, mask, uu


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation="gelu", use_conv_embed=False):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        self.use_conv_embed = bool(use_conv_embed)
        if self.use_conv_embed:
            assert normalize_input == True  # noqa: E712
            module_list = []
            for dim in dims:
                module_list.extend(
                    [
                        nn.Conv1d(input_dim, dim, 1, bias=False),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            self.conv_embed = nn.Sequential(*module_list[:-1])
            self.embed = nn.Identity()
        else:
            module_list = []
            for dim in dims:
                module_list.extend(
                    [
                        nn.LayerNorm(input_dim),
                        nn.Linear(input_dim, dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            self.conv_embed = nn.Identity()
            self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = self.conv_embed(x)
            x = x.transpose(1, 2).contiguous()
        # x: (batch, seq_len, embed_dim)
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
        self,
        pairwise_lv_dim,
        pairwise_input_dim,
        dims,
        pairwise_lv_type="pp",
        remove_self_pair=False,
        use_pre_activation_pair=True,
        normalize_input=True,
        activation="gelu",
        use_bias=True,
        eps=1e-8,
        for_onnx=False,
        sparse_eval=(True, True),
        finite_grad=False,  # itj
    ):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.remove_self_pair = remove_self_pair
        self.for_onnx = for_onnx
        self.sparse_eval = (False, False) if for_onnx else sparse_eval  # flags are for (train, inference)
        self.tril_indices_fn = tril_indices if self.for_onnx else torch.tril_indices
        self.out_dim = dims[-1]
        self.channel_fill: dict[int, float] | None = None  # itj: pair-feature ablation, applied before the embedding

        if pairwise_lv_type == "pp":
            self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
            self.pairwise_lv_fts = partial(
                pairwise_lv_fts_pp, num_outputs=pairwise_lv_dim, eps=eps, finite_grad=finite_grad
            )
        else:  # itj: the "ee" and "xyzt" pair features of weaver are not vendored
            raise RuntimeError("Invalid value for `pairwise_lv_type`: " + pairwise_lv_type)

        if pairwise_lv_dim > 0:
            input_dim = pairwise_lv_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend(
                    [
                        nn.Conv1d(input_dim, dim, 1, bias=use_bias),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)

        if pairwise_input_dim > 0:
            input_dim = pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend(
                    [
                        nn.Conv1d(input_dim, dim, 1, bias=use_bias),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.fts_embed = nn.Sequential(*module_list)

    def _embed_pairs(self, x, uu):
        """Run embedding networks on pair features. Returns (batch_or_1, out_dim, num_pairs)."""
        elements = None
        if x is not None:
            elements = self.embed(x)
        if uu is not None:
            fts = self.fts_embed(uu)
            elements = fts if elements is None else elements + fts
        return elements

    def _forward_dense(self, x, uu=None, mask=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        if x is not None:
            batch_size, _, seq_len = x.size()
        else:
            batch_size, _, seq_len, _ = uu.size()

        if self.is_symmetric:
            i, j = self.tril_indices_fn(
                seq_len,
                seq_len,
                offset=-1 if self.remove_self_pair else 0,
                device=(x if x is not None else uu).device,
            )
            if x is not None:
                xi = x[:, :, i]  # (batch, dim, num_tril_pairs)
                xj = x[:, :, j]
                x = _fill_channels(self.pairwise_lv_fts(xi, xj), self.channel_fill)  # itj: channel_fill
            if uu is not None:
                # (batch, dim, num_tril_pairs)
                uu = uu[:, :, i, j]
        else:
            if x is not None:
                x = _fill_channels(self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2)), self.channel_fill)
                if self.remove_self_pair:
                    diag_idx = torch.arange(0, seq_len, device=x.device)
                    x[:, :, diag_idx, diag_idx] = 0
                x = x.reshape(batch_size, self.pairwise_lv_dim, seq_len * seq_len)
            if uu is not None:
                uu = uu.reshape(batch_size, self.pairwise_input_dim, seq_len * seq_len)

        elements = self._embed_pairs(x, uu)

        if self.is_symmetric:
            y = elements.new_zeros(batch_size, self.out_dim, seq_len, seq_len)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.reshape(batch_size, self.out_dim, seq_len, seq_len)
        return y

    def _forward_sparse(self, x, uu=None, mask=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        if x is not None:
            batch_size, _, seq_len = x.size()
        else:
            batch_size, _, seq_len, _ = uu.size()

        i0, i2, i3 = (Ellipsis,) * 3
        if mask is not None:
            pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (batch_size, 1, seq_len, seq_len)
            if self.is_symmetric:
                offset = -1 if self.remove_self_pair else 0
                i0, _, i2, i3 = pair_mask.float().tril(offset).nonzero(as_tuple=True)
            else:
                i0, _, i2, i3 = pair_mask.nonzero(as_tuple=True)

        if x is not None:
            x = _fill_channels(self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2)), self.channel_fill)  # itj
            x = x.permute(0, 2, 3, 1)[i0, i2, i3, :]  # (num_elements, pairwise_lv_dim)
            x = x.T.unsqueeze(0)  # (1, pairwise_lv_dim, num_elements)
        if uu is not None:
            uu = uu.permute(0, 2, 3, 1)[i0, i2, i3, :]  # (num_elements, pairwise_input_dim)
            uu = uu.T.unsqueeze(0)  # (1, pairwise_input_dim, num_elements)

        elements = self._embed_pairs(x, uu)
        elements = elements.squeeze(0).T  # (num_elements, out_dim)

        y = torch.zeros(batch_size, seq_len, seq_len, self.out_dim, dtype=elements.dtype, device=elements.device)
        y[i0, i2, i3, :] = elements
        if self.is_symmetric:
            y[i0, i3, i2, :] = elements
        y = y.permute(0, 3, 1, 2).contiguous()

        return y

    def forward(self, x, uu=None, mask=None):
        sparse_eval = self.sparse_eval[0 if self.training else 1]
        if sparse_eval and mask is not None:
            return self._forward_sparse(x, uu=uu, mask=mask)
        else:
            return self._forward_dense(x, uu=uu, mask=mask)


def _canonical_mask(
    mask: torch.Tensor | None,
    mask_name: str,
    other_type: Any | None,
    other_name: str,
    target_type: Any,
    check_other: bool = True,
) -> torch.Tensor | None:

    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(f"only bool and floating types of {mask_name} are supported")
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(mask, float("-inf"))
    return mask


def _none_or_dtype(input: torch.Tensor | None):
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")


class Attention(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.use_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:
            self.use_sdpa = False
        self.keep_weights = False  # itj: compute the attention weights explicitly and keep them in `weights`
        self.weights: torch.Tensor | None = None

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):

        for k in list(state_dict.keys()):
            if k.endswith("in_proj_weight"):
                state_dict[k.replace("_weight", ".weight")] = state_dict.pop(k)
            elif k.endswith("in_proj_bias"):
                state_dict[k.replace("_bias", ".bias")] = state_dict.pop(k)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        bsz, tgt_len, _ = query.shape
        _, src_len, _ = key.shape

        # (bsz, src_len)
        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=_none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        # (bsz, num_heads, tgt_len, src_len)
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), (
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            )
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                assert attn_mask.shape == (bsz, self.num_heads, tgt_len, src_len), (
                    f"expecting attn_mask shape of {(bsz, self.num_heads, tgt_len, src_len)}, but got {attn_mask.shape}"
                )
                attn_mask = attn_mask + key_padding_mask

        # (bsz, seq_len, num_heads*head_dim)
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj.weight, self.in_proj.bias)

        # -> (bsz, num_heads, src/tgt_len, head_dim)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0

        if self.use_sdpa and not self.keep_weights:  # itj: keep_weights
            # attn_output: (bsz, num_heads, tgt_len, head_dim)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        else:
            q_scaled = q * math.sqrt(1.0 / float(self.head_dim))  # (bsz, num_heads, tgt_len, head_dim)
            attn_weight = q_scaled @ k.transpose(-2, -1)  # (bsz, num_heads, tgt_len, src_len)
            if attn_mask is not None:
                attn_weight = attn_weight + attn_mask
            attn_weight = F.softmax(attn_weight, dim=-1)
            if self.keep_weights:  # itj: the weights before dropout
                self.weights = attn_weight
            if dropout_p > 0:
                attn_weight = F.dropout(attn_weight, p=dropout_p)
            attn_output = attn_weight @ v  # (bsz, num_heads, tgt_len, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous()  # (bsz, tgt_len, num_heads, head_dim)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None


class Block(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        ffn_ratio=4,
        dropout=0.1,
        attn_dropout=0.1,
        activation_dropout=0.1,
        activation="gelu",
        use_bias=True,
        scale_attn_mask=False,
        scale_attn=True,
        scale_fc=True,
        scale_heads=True,
        scale_resids=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        norm_layer = nn.LayerNorm
        self.pre_attn_norm = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            bias=use_bias,
        )
        self.post_attn_norm = norm_layer(embed_dim) if scale_attn else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = norm_layer(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim, bias=use_bias)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = norm_layer(self.ffn_dim) if scale_fc else nn.Identity()
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim, bias=use_bias)

        self.c_mask = nn.Parameter(torch.ones(1), requires_grad=True) if scale_attn_mask else None
        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Checkpoints from weaver-core < 0.5 (nn.MultiheadAttention, keys `attn.in_proj_weight`) applied
        # `c_attn` via einsum("tbhd,h->tbdh"), which left the attention output in (head_dim, num_heads)
        # channel order. The current layout is (num_heads, head_dim), so fold that permutation and
        # `c_attn` into `out_proj` and reset `c_attn` to 1, which reproduces the old outputs exactly.
        # Runs before the children's hooks, so `Attention` has not renamed the legacy keys yet.
        c_key = prefix + "c_attn"
        if prefix + "attn.in_proj_weight" in state_dict and c_key in state_dict:
            c_attn = state_dict[c_key]
            num_heads, head_dim = c_attn.numel(), self.head_dim

            def _fold(t):
                t = t.view(num_heads, head_dim, *t.shape[1:]) * c_attn.view(-1, *([1] * t.dim()))
                return t.transpose(0, 1).reshape(num_heads * head_dim, *t.shape[2:])

            for k in ("weight", "bias"):
                key = prefix + "attn.out_proj." + k
                if key in state_dict:
                    state_dict[key] = _fold(state_dict[key])
            state_dict[c_key] = torch.ones_like(c_attn)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, seq_len, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(batch, 1, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``True``.

        Returns:
            encoded output of shape `(batch, seq_len, embed_dim)`
        """

        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=1)  # (batch, 1+seq_len, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (batch, 1, embed_dim)
        else:
            if self.c_mask is not None and attn_mask is not None:
                attn_mask = torch.mul(self.c_mask, attn_mask)
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)[0]  # (batch, seq_len, embed_dim)

        if self.c_attn is not None:
            bsz, tgt_len, _ = x.size()
            x = x.view(bsz, tgt_len, self.num_heads, self.head_dim)
            x = x * self.c_attn.view(1, 1, self.num_heads, 1)
            x = x.reshape(bsz, tgt_len, self.embed_dim)
        x = self.post_attn_norm(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = x + residual

        return x


class ParticleTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=None,
        # network configurations
        pair_input_type="pp",
        pair_input_dim=None,
        pair_extra_dim=0,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        use_conv_embed=False,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        pair_embed_sparse_eval=None,
        num_heads=8,
        num_layers=8,
        block_params=None,
        block_ids_with_attn_mask=None,
        num_cls_layers=2,
        cls_block_params=None,
        fc_params=(),
        activation="gelu",
        # misc
        weight_init="moco",
        fix_init=True,
        trim=False,  # itj: weaver default True
        for_inference=False,
        finite_grad=False,  # itj
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        _logger.debug("ParticleTransformer init-ed: %s", locals())

        self.for_inference = for_inference

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            activation=activation,
            use_bias=True,
            scale_attn_mask=False,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )
        norm_layer = nn.LayerNorm

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.debug("cfg_block: %s" % str(cfg_block))  # noqa: UP031

        if num_cls_layers > 0:
            cfg_cls_block = copy.deepcopy(default_cfg)
            cfg_cls_block.update({"dropout": 0, "attn_dropout": 0, "activation_dropout": 0})
            if cls_block_params is not None:
                cfg_cls_block.update(cls_block_params)
            _logger.debug("cfg_cls_block: %s" % str(cfg_cls_block))  # noqa: UP031

        if block_ids_with_attn_mask is None:
            self.block_ids_with_attn_mask = [True] * num_layers
        else:
            self.block_ids_with_attn_mask = [False] * num_layers
            for idx in block_ids_with_attn_mask:
                self.block_ids_with_attn_mask[idx] = True
        _logger.debug("block w/ attn_mask: %s" % str(self.block_ids_with_attn_mask))  # noqa: UP031

        self.embed = (
            Embed(input_dim, embed_dims, activation=activation, use_conv_embed=use_conv_embed)
            if len(embed_dims) > 0
            else nn.Identity()
        )

        if pair_input_dim is None:
            if pair_input_type == "pp":
                pair_input_dim = 4
        self.pair_extra_dim = pair_extra_dim
        self.pair_embed = (
            PairEmbed(
                pair_input_dim,
                pair_extra_dim,
                (*pair_embed_dims, cfg_block["num_heads"]),
                pairwise_lv_type=pair_input_type,
                remove_self_pair=remove_self_pair,
                use_pre_activation_pair=use_pre_activation_pair,
                use_bias=default_cfg["use_bias"],
                for_onnx=for_inference,
                sparse_eval=(True, True) if pair_embed_sparse_eval is None else pair_embed_sparse_eval,
                finite_grad=finite_grad,  # itj
            )
            if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0
            else None
        )
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = (
            nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)]) if num_cls_layers > 0 else None
        )
        self.norm = norm_layer(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for param in fc_params:
                try:
                    out_dim, drop_rate, act = param
                except ValueError:
                    (out_dim, drop_rate), act = param, "relu"
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim, bias=default_cfg["use_bias"]),
                    nn.GELU() if act == "gelu" else nn.ReLU(),
                    nn.Dropout(drop_rate),
                )
                fcs.append(layer)
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # cls tokens
        if num_cls_layers > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # sequence trimmer
        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        # weight initialization
        if weight_init is not None:
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = "") -> None:
        assert mode in ("timm", "moco")
        if mode == "timm":
            named_apply(init_weights_vit_timm, self)
        elif mode == "moco":
            named_apply(init_weights_vit_moco, self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
        }

    def _forward_encoder(self, x, v=None, mask=None, uu=None, uu_idx=None, record=None):
        with torch.set_grad_enabled(x.requires_grad):
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (batch_size, seq_len) -- `True`` means masked/ignored

        # input embedding
        x = self.embed(x).masked_fill(~mask.transpose(1, 2), 0)  # (batch_size, seq_len, embed_dim)
        attn_mask = None
        if (v is not None or uu is not None) and self.pair_embed is not None:
            attn_mask = self.pair_embed(v, uu=uu, mask=mask)  # (batch_size, num_heads, seq_len, seq_len)
        if record is not None:  # itj
            record.update(mask=mask.squeeze(1), embed=x, pair_bias=attn_mask)

        # transform
        for idx, block in enumerate(self.blocks):
            x = block(
                x,
                x_cls=None,
                padding_mask=padding_mask,
                attn_mask=attn_mask if self.block_ids_with_attn_mask[idx] else None,
            )
            if record is not None:  # itj
                record[f"block{idx}"] = x

        # x: (batch, seq_len, embed_dim)
        # padding_mask: (batch, seq_len)
        return x, padding_mask

    def _forward_aggregator(self, x, padding_mask, record=None):
        if self.cls_blocks is not None:
            # for classification: extract using class token
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (batch, 1, embed_dim)
            for idx, block in enumerate(self.cls_blocks):
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)  # (batch, 1, embed_dim)
                if record is not None:  # itj
                    record[f"cls{idx}"] = cls_tokens.squeeze(1)
            cls_tokens = cls_tokens.squeeze(1)  # (batch, embed_dim)
        else:
            # for classification: simple average pooling
            mask = ~padding_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            counts = mask.float().sum(dim=1).clamp(min=1)  # (batch, 1)
            cls_tokens = (x * mask).sum(dim=1) / counts  # (batch, embed_dim)

        x_cls = self.norm(cls_tokens)  # (batch, embed_dim)
        return x_cls

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None, record=None):
        # x: (batch_size, num_fts, seq_len)
        # v: (batch_size, 4, seq_len) [px,py,pz,energy]
        # mask: (batch_size, 1, seq_len) -- real particle = 1, padded = 0
        # for pytorch: uu (batch_size, C', num_pairs), uu_idx (batch_size, 2, num_pairs)
        # for onnx: uu (batch_size, C', seq_len, seq_len), uu_idx=None
        # itj: record (dict, optional) receives mask, embed, pair_bias, block{i}, cls{i} and x_cls

        x, padding_mask = self._forward_encoder(x, v=v, mask=mask, uu=uu, uu_idx=uu_idx, record=record)

        if self.cls_blocks is None and self.fc is None:
            # x: (batch, seq_len, embed_dim)
            # padding_mask: (batch, seq_len)
            return x, padding_mask

        x_cls = self._forward_aggregator(x, padding_mask, record=record)
        if record is not None:  # itj
            record["x_cls"] = x_cls
        if self.fc is None:
            return x_cls

        # fc
        output = self.fc(x_cls)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output


### weight initialization methods ###
def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed"""
    if isinstance(module, nn.Linear):
        if "in_proj" in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def named_apply(
    fn,
    module: nn.Module,
    name="",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


# ----------------------------------------------------------------------------------------------------------------------
# itj wrapper (not part of weaver)

#: The class-attention block settings of the JetClass ParT (no dropout in the class-attention blocks).
_CLS_BLOCK_PARAMS: dict[str, float] = {"dropout": 0, "attn_dropout": 0, "activation_dropout": 0}


class ParT(Tagger):
    """Particle Transformer: the vendored weaver :class:`ParticleTransformer` (v1) behind the itj model API.

    The particle features x_i are embedded per particle, then ``num_layers`` particle-attention blocks update them
    with ``softmax(Q K^T / sqrt(d) + U) V``, where the interaction bias U (one value per head and pair) is an
    embedding of the pair features ``[ln kT, ln z, ln Delta, ln m^2]`` of the four-vectors; ``num_cls_layers``
    class-attention blocks (CaiT) read the particles into a class token, whose LayerNorm is the embedding.
    The state-dict keys are ``mod.*`` as in the ParT repository's ``example_ParticleTransformer.py``, so weaver
    checkpoints load directly.

    Parameters
    ----------
    input_dim : int
        Number of particle features (7 kin, 13 kinpid, 17 full).
    n_classes : int
        Number of outputs.
    pair_input_dim : int or None
        Number of pair features of the four-vectors: 4 is ``[ln kT, ln z, ln Delta, ln m^2]``, 1 only ``ln Delta``,
        5 adds ``ln(-(p_i - p_j)^2)``, 6 and 7 add asymmetric features (weaver); None means 4 (weaver).
    embed_dims : sequence of int
        Widths of the per-particle embedding MLP (the last is the model width D).
    pair_embed_dims : sequence of int or None
        Widths of the pair embedding (a final layer of ``num_heads`` channels is added); None: no pair features and
        no U (plain attention).
    num_heads, num_layers, num_cls_layers : int
        Attention heads, particle-attention blocks and class-attention blocks.
    fc_params : sequence of (int, float)
        Hidden (width, dropout) layers between the class token and the output layer (JetClass ParT: none).
    activation : {"gelu", "relu"}
        Activation of the embeddings and blocks.
    use_pre_activation_pair : bool
        Drop the last activation of the pair embedding (weaver default True; the JetClass ParT uses False).
    cls_block_params : dict, optional
        Overrides of the class-attention blocks (default: no dropout).
    finite_grad : bool
        Add 1e-30 to Delta^2 before the square root, so that gradients with respect to the four-vectors are finite;
        the outputs are unchanged.
    trim : bool
        Enable weaver's ``SequenceTrimmer`` (randomly shortens the padded sequence during training); off by default,
        so that the padded length never changes inside the model.
    feature_set : str or None
        The particle features the model reads (``make_inputs`` builds batches with it).

    Attributes
    ----------
    mod : ParticleTransformer
        The vendored weaver module.
    pair_channel_fill : dict of int to float or None
        Pair-feature channels replaced by constants before the pair embedding (set by
        :func:`itj.interpret.ablate_pair_channels`); None: no replacement.
    """

    default_name = "part"

    def __init__(
        self,
        input_dim: int = 7,
        n_classes: int = 3,
        *,
        pair_input_dim: int | None = 4,
        embed_dims: Sequence[int] = (128, 512, 128),
        pair_embed_dims: Sequence[int] | None = (64, 64, 64),
        num_heads: int = 8,
        num_layers: int = 8,
        num_cls_layers: int = 2,
        fc_params: Sequence[tuple[int, float]] = (),
        activation: str = "gelu",
        use_pre_activation_pair: bool = False,
        cls_block_params: Mapping[str, float] | None = None,
        finite_grad: bool = True,
        trim: bool = False,
        feature_set: str | None = "kin",
    ) -> None:
        super().__init__()
        check_feature_set(feature_set, input_dim)
        if num_cls_layers < 1:
            raise ValueError("ParT needs at least one class-attention block (num_cls_layers >= 1)")
        cls_params = dict(_CLS_BLOCK_PARAMS if cls_block_params is None else cls_block_params)
        self.input_dim = int(input_dim)
        self.mod = ParticleTransformer(
            input_dim=input_dim,
            num_classes=n_classes,
            pair_input_dim=pair_input_dim,
            use_pre_activation_pair=use_pre_activation_pair,
            embed_dims=list(embed_dims),
            pair_embed_dims=None if pair_embed_dims is None else list(pair_embed_dims),
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            block_params=None,
            cls_block_params=cls_params,
            fc_params=[tuple(p) for p in fc_params],
            activation=activation,
            trim=trim,
            for_inference=False,
            finite_grad=finite_grad,
        )
        init = {
            "input_dim": int(input_dim),
            "n_classes": int(n_classes),
            "pair_input_dim": pair_input_dim,
            "embed_dims": tuple(embed_dims),
            "pair_embed_dims": None if pair_embed_dims is None else tuple(pair_embed_dims),
            "num_heads": num_heads,
            "num_layers": num_layers,
            "num_cls_layers": num_cls_layers,
            "fc_params": tuple(tuple(p) for p in fc_params),
            "activation": activation,
            "use_pre_activation_pair": use_pre_activation_pair,
            "cls_block_params": cls_params,
            "finite_grad": finite_grad,
            "trim": trim,
            "feature_set": feature_set,
        }
        self._set_config(init, feature_set=feature_set, jet_features=None)

    @property
    def pair_channel_fill(self) -> dict[int, float] | None:
        """Pair-feature channels replaced by constants before the pair embedding (None: none)."""
        return None if self.mod.pair_embed is None else self.mod.pair_embed.channel_fill

    @pair_channel_fill.setter
    def pair_channel_fill(self, fill: Mapping[int, float] | None) -> None:
        if self.mod.pair_embed is None:
            if fill:
                raise ValueError("this ParT has no pair features (pair_embed_dims=None)")
            return
        if fill is not None:
            n = self.mod.pair_embed.pairwise_lv_dim
            bad = [c for c in fill if not 0 <= int(c) < n]
            if bad:
                raise ValueError(f"pair-feature channels must be in [0, {n}), got {bad}")
            fill = {int(c): float(v) for c, v in fill.items()}
        self.mod.pair_embed.channel_fill = None if fill is None else dict(fill)

    @property
    def head(self) -> nn.Linear:
        """The output Linear layer of ``mod.fc``."""
        layer = self.mod.fc[-1]
        assert isinstance(layer, nn.Linear)
        return layer

    def _pre_head(self, x_cls: torch.Tensor) -> torch.Tensor:
        return self.mod.fc[:-1](x_cls) if len(self.mod.fc) > 1 else x_cls

    def forward(self, batch: JetBatch) -> torch.Tensor:
        """Return the logits (B, C): the vendored weaver forward on (features, vectors, mask)."""
        self._check_features(batch, self.input_dim)
        return self.mod(batch.features, v=batch.vectors, mask=batch.mask)

    def embed(self, batch: JetBatch) -> torch.Tensor:
        """Return the embedding (B, D): the LayerNorm of the class token (after the hidden ``fc_params`` layers)."""
        self._check_features(batch, self.input_dim)
        record: dict[str, Any] = {}
        self.mod(batch.features, v=batch.vectors, mask=batch.mask, record=record)
        return self._pre_head(record["x_cls"])

    def trace(self, batch: JetBatch, *, attention: bool = False) -> Trace:
        """Return the :class:`Trace` of one forward pass (attention maps only with ``attention=True``).

        Memory: the per-particle layers take B P D (num_layers + 1) floats; with ``attention=True`` add
        B H P^2 (num_layers + num_cls_layers / P) floats (for B = 256, P = 128, H = 8 and 8 blocks about 0.5 GB).
        """
        self._check_features(batch, self.input_dim)
        attns = [cast(Block, block).attn for block in self.mod.blocks]
        cls_blocks = list(self.mod.cls_blocks) if self.mod.cls_blocks is not None else []
        cls_attns = [cast(Block, block).attn for block in cls_blocks]
        record: dict[str, Any] = {}
        weights: list[torch.Tensor | None] = []
        try:
            for a in attns + cls_attns:
                a.keep_weights = attention
            logits = self.mod(batch.features, v=batch.vectors, mask=batch.mask, record=record)
            weights = [a.weights for a in attns + cls_attns]
        finally:
            for a in attns + cls_attns:
                a.keep_weights = False
                a.weights = None
        mask = record["mask"]
        keep = mask.unsqueeze(-1)
        layers = {"embed": record["embed"]}
        for i in range(len(attns)):
            layers[f"block{i}"] = record[f"block{i}"].masked_fill(~keep, 0.0)
        pooled = pooled_layers(layers, mask)
        for i in range(len(cls_attns)):
            pooled[f"cls{i}"] = record[f"cls{i}"]
        attention_maps: dict[str, torch.Tensor] = {}
        class_attention: dict[str, torch.Tensor] = {}
        if attention:
            for i, w in enumerate(weights[: len(attns)]):
                assert w is not None
                attention_maps[f"block{i}"] = w
            for i, w in enumerate(weights[len(attns) :]):
                assert w is not None
                class_attention[f"cls{i}"] = w
        return Trace(
            logits=logits,
            embedding=self._pre_head(record["x_cls"]),
            mask=mask,
            layers=layers,
            pooled=pooled,
            attention=attention_maps,
            class_attention=class_attention,
            pair_bias=record["pair_bias"],
        )
