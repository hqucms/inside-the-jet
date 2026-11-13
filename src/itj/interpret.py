"""What does the tagger look at: gradients, integrated gradients, attention, the pair bias U and IRC scans.

* :func:`input_gradients` and :func:`saliency`: gradients of a class score with respect to the network inputs
  (particle features, four-vectors, points), zero at padded slots;
* :func:`integrated_gradients`: the path integral of the gradient from a baseline to the jet (Sundararajan et al.,
  arXiv:1703.01365) with its completeness gap ``sum IG - (s(x) - s(x0))``;
* :func:`attention_rollout`, :func:`attention_stats`: the class-token rollout of ParT attention and per-head
  statistics against particle groups (prongs, or the reco proxy :func:`subjet_membership`);
* :func:`pair_features`, :func:`ablate_pair_channels`, :func:`zero_pair_bias`: the pair features of ParT and two
  interventions on its interaction bias U;
* :func:`irc_scan`: how much the score (or the embedding) moves when a soft particle is added or a particle is split
  collinearly, as a function of the size of the perturbation.

Gradients are taken in evaluation mode (the mode of the model is restored afterwards). With respect to the
four-vectors, ParT needs ``finite_grad`` (the default of :class:`itj.models.ParT`): Delta = 0 on the diagonal of the
pair features has an infinite derivative of the square root otherwise; :func:`input_gradients` switches it on for the
duration of the call.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import re
from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import torch

if TYPE_CHECKING:
    from .data import JetData, Ragged
    from .features import JetBatch
    from .models import ParT, Tagger, Trace

__all__ = [
    "Target",
    "Wrt",
    "ablate_pair_channels",
    "attention_rollout",
    "attention_stats",
    "input_gradients",
    "integrated_gradients",
    "irc_scan",
    "pair_features",
    "saliency",
    "subjet_membership",
    "zero_pair_bias",
]

_log = logging.getLogger("itj")

#: Which class score to explain: a class index, ``"label"`` (the true class of every jet, ``batch.label``) or a
#: (B,) tensor of class indices.
Target: TypeAlias = "int | Literal['label'] | torch.Tensor"
#: An input of a :class:`~itj.features.JetBatch` that gradients can be taken with respect to.
Wrt = Literal["features", "vectors", "points"]

_INPUTS: tuple[str, ...] = ("points", "features", "vectors")
#: Scale of the four-momenta of the "soft" IG baseline.
SOFT_SCALE = 1e-3
#: Maximum number of particles a perturbed jet may have (the padded length of weaver and of the models).
MAX_PARTICLES = 128


# ----------------------------------------------------------------------------------------------------------------------
# Helpers


@contextlib.contextmanager
def _eval_mode(model: torch.nn.Module) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


@contextlib.contextmanager
def _finite_gradients(model: torch.nn.Module) -> Iterator[None]:
    """Switch on ``finite_grad`` of every ParT pair embedding of ``model`` for the duration of the block."""
    saved: list[tuple[Any, Any]] = []
    modules: list[Any] = list(model.modules())
    for module in modules:
        fn = getattr(module, "pairwise_lv_fts", None)
        if isinstance(fn, functools.partial) and fn.keywords.get("finite_grad") is False:
            saved.append((module, fn))
            module.pairwise_lv_fts = functools.partial(fn.func, *fn.args, **{**fn.keywords, "finite_grad": True})
    try:
        yield
    finally:
        for module, fn in saved:
            module.pairwise_lv_fts = fn


def _wrt(wrt: str | Sequence[str]) -> tuple[str, ...]:
    names = (wrt,) if isinstance(wrt, str) else tuple(wrt)
    bad = [n for n in names if n not in _INPUTS]
    if bad or not names:
        raise ValueError(f"wrt must name inputs among {_INPUTS}, got {wrt!r}")
    return tuple(dict.fromkeys(names))


def _targets(batch: JetBatch, target: Any) -> torch.Tensor:
    b = len(batch)
    device = batch.features.device
    if isinstance(target, torch.Tensor):
        t = target.to(device=device, dtype=torch.int64).reshape(-1)
        if t.numel() == 1:
            t = t.expand(b)
        if t.shape != (b,):
            raise ValueError(f"target tensor must have shape ({b},), got {tuple(target.shape)}")
        return t
    if isinstance(target, str):
        if target != "label":
            raise ValueError(f"target must be a class index, 'label' or a tensor, got {target!r}")
        if batch.label is None:
            raise ValueError("target='label' needs a batch with labels (label_set) or pass a class index")
        return batch.label.to(device=device, dtype=torch.int64)
    if isinstance(target, (int, np.integer)):
        return torch.full((b,), int(target), dtype=torch.int64, device=device)
    raise TypeError(f"target must be an int, 'label' or a torch.Tensor, got {type(target).__name__}")


def _score(logits: torch.Tensor, target: torch.Tensor, output: str) -> torch.Tensor:
    if output == "prob":
        values = torch.softmax(logits, dim=1)
    elif output == "logit":
        values = logits
    else:
        raise ValueError(f"output must be 'prob' or 'logit', got {output!r}")
    return values.gather(1, target[:, None])[:, 0]


def _replace(batch: JetBatch, **inputs: Any) -> JetBatch:
    return dataclasses.replace(batch, **inputs)


def _grads(model: Tagger, batch: JetBatch, target: torch.Tensor, output: str, names: Sequence[str]) -> dict:
    inputs = {n: getattr(batch, n).detach().clone().requires_grad_(True) for n in names}
    with torch.enable_grad():
        s = _score(model(_replace(batch, **inputs)), target, output)
        grads = torch.autograd.grad(s.sum(), list(inputs.values()), allow_unused=True)
    mask = batch.mask.to(batch.features.dtype)
    return {
        n: (torch.zeros_like(inputs[n]) if g is None else g.detach()) * mask.to(inputs[n].dtype)
        for n, g in zip(names, grads, strict=True)
    }


def _label_set(model: Tagger) -> str:
    sets = {3: "jets3", 10: "jetclass10", 2: "qg"}
    if model.n_classes not in sets:
        raise ValueError(f"cannot tell the label set of a model with {model.n_classes} classes: pass a class index")
    return sets[model.n_classes]


# ----------------------------------------------------------------------------------------------------------------------
# Gradients and integrated gradients


def input_gradients(
    model: Tagger,
    batch: JetBatch,
    target: Target = "label",
    *,
    output: Literal["prob", "logit"] = "prob",
    wrt: Wrt | Sequence[Wrt] = ("features",),
) -> dict[str, torch.Tensor]:
    """Return the gradients of a class score with respect to inputs of the batch.

    The model runs in evaluation mode (restored afterwards), with ``finite_grad`` switched on for ParT, on copies of
    the inputs that require gradients; gradients of padded slots are set to zero.

    Parameters
    ----------
    model : itj.models.Tagger
        The tagger.
    batch : itj.features.JetBatch
        Jets (on the device of the model).
    target : int, "label" or torch.Tensor
        Class whose score is differentiated: one index for all jets, the true class, or one index per jet.
    output : {"prob", "logit"}
        Differentiate the softmax probability or the logit of the target class.
    wrt : str or sequence of str
        Inputs among "features", "vectors" (four-vectors) and "points" (Delta eta, Delta phi). A model that does not
        read an input (for example ParT and the points) gets zero gradients.

    Returns
    -------
    dict of str to torch.Tensor
        One gradient per input, with the shape of the input ((B, C, P)).
    """
    names = _wrt(wrt)
    t = _targets(batch, target)
    with _eval_mode(model), _finite_gradients(model):
        return _grads(model, batch, t, output, names)


def saliency(
    model: Tagger,
    batch: JetBatch,
    target: Target = "label",
    *,
    output: Literal["prob", "logit"] = "prob",
    wrt: Wrt | Sequence[Wrt] = "features",
) -> np.ndarray:
    """Return the saliency of every particle: the L2 norm of its gradient over all channels of the inputs ``wrt``.

    Saliency is the one-step special case of the attributions (a local slope, not a contribution).

    Returns
    -------
    numpy.ndarray
        (B, P) float32, zero at padded slots.
    """
    grads = input_gradients(model, batch, target, output=output, wrt=wrt)
    total = sum(g.double().pow(2).sum(dim=1) for g in grads.values())
    return torch.as_tensor(total).sqrt().float().cpu().numpy()


def _soft_baseline(batch: JetBatch) -> JetBatch:
    """Return the "soft" baseline: every four-momentum scaled by 1e-3, the pT and energy features recomputed."""
    from .features import FEATURE_SETS

    vectors = batch.vectors * SOFT_SCALE
    features = batch.features.clone()
    if batch.feature_set is not None and features.shape[1] > 0:
        variables = FEATURE_SETS[batch.feature_set]
        pt = torch.hypot(vectors[:, 0], vectors[:, 1])
        energy = vectors[:, 3]
        for c, var in enumerate(variables):
            if var.name in ("part_pt_log", "part_e_log"):
                raw = torch.log(pt if var.name == "part_pt_log" else energy)
                if var.center is not None:
                    raw = ((raw - var.center) * var.scale).clamp(var.clip_min, var.clip_max)
                features[:, c] = torch.nan_to_num(raw, nan=0.0)
            # the other features (ratios to the jet, angles, PID, impact parameters) do not change
    return _replace(batch, vectors=vectors, features=features)


def _baseline(batch: JetBatch, baseline: Any, names: Sequence[str]) -> JetBatch:
    if isinstance(baseline, str):
        if baseline == "zero":
            return _replace(batch, **{n: torch.zeros_like(getattr(batch, n)) for n in names})
        if baseline == "soft":
            return _soft_baseline(batch)
        raise ValueError(f"baseline must be 'zero', 'soft' or a JetBatch, got {baseline!r}")
    for n in _INPUTS:
        if getattr(baseline, n).shape != getattr(batch, n).shape:
            raise ValueError(f"the baseline batch has {n} of shape {tuple(getattr(baseline, n).shape)}")
    return _replace(batch, **{n: getattr(baseline, n).to(getattr(batch, n)) for n in _INPUTS})


def _quadrature(steps: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Nodes in [0, 1] and weights of the path integral."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if method == "gausslegendre":
        x, w = np.polynomial.legendre.leggauss(steps)
        return (x + 1.0) / 2.0, w / 2.0
    if method == "riemann_middle":
        return (np.arange(steps) + 0.5) / steps, np.full(steps, 1.0 / steps)
    if method == "riemann_trapezoid":
        if steps < 2:
            raise ValueError("the trapezoid rule needs steps >= 2")
        w = np.full(steps, 1.0 / (steps - 1))
        w[[0, -1]] /= 2
        return np.linspace(0.0, 1.0, steps), w
    raise ValueError(f"method must be 'gausslegendre', 'riemann_middle' or 'riemann_trapezoid', got {method!r}")


def _cast(model: Tagger, batch: JetBatch, dtype: torch.dtype) -> tuple[Tagger, JetBatch]:
    """Return a copy of the model and the batch in ``dtype`` (the originals if they have it already)."""
    copy = model if all(p.dtype == dtype for p in model.parameters()) else deepcopy(model).to(dtype)
    inputs = {n: getattr(batch, n).to(dtype) for n in (*_INPUTS, "mask")}
    return copy, _replace(batch, **inputs)


def integrated_gradients(
    model: Tagger,
    batch: JetBatch,
    target: Target = "label",
    *,
    baseline: Literal["zero", "soft"] | JetBatch = "zero",
    steps: int = 32,
    output: Literal["prob", "logit"] = "prob",
    wrt: Wrt | Sequence[Wrt] = "features",
    method: Literal["gausslegendre", "riemann_middle", "riemann_trapezoid"] = "gausslegendre",
    dtype: torch.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integrated-gradient attributions per particle and the completeness gap per jet.

    ``IG_i = (x_i - x0_i) * integral_0^1 ds/dx_i(x0 + a (x - x0)) da`` for every input channel in ``wrt``, summed
    over the channels (and inputs) of each particle. All inputs move along the straight path from the baseline to
    the jet; only those in ``wrt`` are attributed. Completeness ``sum_i IG_i = s(x) - s(x0)`` therefore holds (up to
    the quadrature error) only when every input that differs from the baseline is in ``wrt``: the zero baseline
    changes only the inputs in ``wrt``, the "soft" baseline changes the four-vectors and the pT and energy features,
    so on ParT it needs ``wrt=("features", "vectors")``.

    The gap is the check of the result: when it is not small compared with ``s(x) - s(x0)``, the integrand has
    structure the quadrature does not resolve and ``steps`` must grow. ParT switches sharply along the path from the
    zero baseline (the score wiggles on a scale of 0.01 in the path parameter): on ParT-small and the pretrained ParT
    the summed gap is 2-30 % of the summed ``|s(x) - s(x0)|`` with 32 steps, 1e-4 to 1e-2 with 128 and below 1e-6
    with 512 steps. With respect to the four-vectors ParT is limited by float32 rounding: ``ln m^2`` of nearly
    collinear pairs is a difference of large numbers, so the gradients are noisy and the gap of the soft baseline
    stops falling at about 1 % in float32; ``dtype=torch.float64`` evaluates a float64 copy of the model and removes
    that floor (soft baseline in float64: 0.02-20 % with 32 steps, below 2e-4 with 512).

    Parameters
    ----------
    model : itj.models.Tagger
        The tagger.
    batch : itj.features.JetBatch
        Jets (on the device of the model).
    target : int, "label" or torch.Tensor
        Class whose score is explained.
    baseline : {"zero", "soft"} or JetBatch
        "zero": the inputs in ``wrt`` set to 0; "soft": every four-momentum scaled by 1e-3 with ``part_pt_log`` and
        ``part_e_log`` recomputed (the ratios to the jet and the angles do not change); or any batch of the same
        shape.
    steps : int
        Number of gradient evaluations along the path.
    output : {"prob", "logit"}
        Explain the softmax probability or the logit of the target class.
    wrt : str or sequence of str
        Inputs to attribute ("features", "vectors", "points").
    method : {"gausslegendre", "riemann_middle", "riemann_trapezoid"}
        Quadrature of the path integral: Gauss-Legendre nodes (the default of captum; exact for polynomials of
        degree 2 steps - 1) or a Riemann sum.
    dtype : torch.dtype, optional
        Evaluate a copy of the model and the inputs in this dtype (``torch.float64`` for gradients with respect to
        the four-vectors of ParT); None: as given. The scores ``s`` of the gap are those of the copy.

    Returns
    -------
    attributions : numpy.ndarray
        (B, P) float64, zero at padded slots.
    gap : numpy.ndarray
        (B,) ``sum_i IG_i - (s(x) - s(x0))``.
    """
    names = _wrt(wrt)
    t = _targets(batch, target)
    if dtype is not None:
        model, batch = _cast(model, batch, dtype)
    base = _baseline(batch, baseline, names)
    alphas, weights = _quadrature(steps, method)
    diff = {n: getattr(batch, n) - getattr(base, n) for n in _INPUTS}
    total = {n: torch.zeros_like(getattr(batch, n), dtype=torch.float64) for n in names}
    with _eval_mode(model), _finite_gradients(model):
        for a, w in zip(alphas, weights, strict=True):
            path = _replace(batch, **{n: getattr(base, n) + float(a) * diff[n] for n in _INPUTS})
            for n, g in _grads(model, path, t, output, names).items():
                total[n] += float(w) * g.double()
        with torch.no_grad():
            s_x = _score(model(batch), t, output).double()
            s_0 = _score(model(base), t, output).double()
    mask = batch.mask.double()
    attr = sum((diff[n].double() * total[n] * mask).sum(dim=1) for n in names)
    attr_t = torch.as_tensor(attr)
    gap = attr_t.sum(dim=1) - (s_x - s_0)
    return attr_t.cpu().numpy(), gap.cpu().numpy()


# ----------------------------------------------------------------------------------------------------------------------
# Attention and the pair bias


def _ordered(entries: dict[str, torch.Tensor], prefix: str) -> list[torch.Tensor]:
    def index(name: str) -> int:
        m = re.fullmatch(rf"{prefix}(\d+)", name)
        return int(m.group(1)) if m else -1

    return [entries[k] for k in sorted((k for k in entries if index(k) >= 0), key=index)]


def attention_rollout(trace: Trace) -> np.ndarray:
    """Return the class-token attention rollout of ParT (Abnar and Zuidema, arXiv:2005.00928) per particle.

    The heads are averaged; every particle-attention block mixes the tokens with ``0.5 A + 0.5 I`` (the residual
    connection) and the products give how much each input particle flows into each output token. The class token
    starts empty and takes ``0.5 a_cls`` of the particles in every class-attention block (keeping
    ``0.5 + 0.5 a_self`` of itself). The result is normalised to sum to 1 over the real particles.

    Parameters
    ----------
    trace : itj.models.Trace
        A ParT trace computed with ``attention=True``.

    Returns
    -------
    numpy.ndarray
        (B, P) float64, zero at padded slots.
    """
    blocks = _ordered(trace.attention, "block")
    cls_blocks = _ordered(trace.class_attention, "cls")
    if not blocks or not cls_blocks:
        raise ValueError("the trace has no attention maps: use a ParT and model.trace(batch, attention=True)")
    real = (
        trace.mask if trace.mask is not None else torch.ones(blocks[0].shape[0], blocks[0].shape[-1], dtype=torch.bool)
    )
    real_f = real.double()
    b, p = real.shape
    eye = torch.eye(p, dtype=torch.float64, device=real.device).expand(b, p, p)
    rollout = eye.clone()
    for a in blocks:
        mixed = a.detach().double().mean(dim=1) * real_f[:, None, :]  # (B, P, P) keys restricted to real particles
        mixed = 0.5 * mixed + 0.5 * eye
        mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp(min=1e-300)
        rollout = mixed @ rollout
    r = torch.zeros(b, p, dtype=torch.float64, device=real.device)
    for c in cls_blocks:
        w = c.detach().double().mean(dim=1)[:, 0, :]  # (B, 1 + P)
        r = (0.5 + 0.5 * w[:, 0:1]) * r + 0.5 * w[:, 1:] * real_f
    out = (r[:, None, :] @ rollout)[:, 0, :] * real_f
    out = out / out.sum(dim=1, keepdim=True).clamp(min=1e-300)
    return out.cpu().numpy()


def pair_features(batch: JetBatch) -> torch.Tensor:
    """Return the ParT pair features ``[ln kT, ln z, ln Delta, ln m^2]`` of every pair of the batch.

    Computed by the vendored weaver function on the four-vectors (clamped at 1e-8 inside the logarithms, so a pair
    with Delta = 0, the diagonal for example, has ln Delta = ln kT = -18.4). Pairs with a padded slot hold the values
    of the wrap-padded copies; select real pairs with ``batch.particle_mask``.

    Returns
    -------
    torch.Tensor
        (B, 4, P, P).
    """
    from .models.part import pairwise_lv_fts_pp

    v = batch.vectors.detach()
    with torch.no_grad():
        out: torch.Tensor = pairwise_lv_fts_pp(v.unsqueeze(-1), v.unsqueeze(-2), num_outputs=4, eps=1e-8)
    return out


@contextlib.contextmanager
def ablate_pair_channels(
    model: ParT,
    channels: Sequence[int],
    *,
    fill: Literal["mean", "zero"] = "mean",
    reference: JetBatch | None = None,
) -> Iterator[None]:
    """Replace pair-feature channels of a ParT by constants inside the block (mean imputation by default).

    Channels: 0 ln kT, 1 ln z, 2 ln Delta, 3 ln m^2. "mean" uses the mean of the channel over the distinct real
    pairs (i != j) of ``reference``; "zero" uses 0 (the raw value, before the BatchNorm of the pair embedding). The
    previous ``model.pair_channel_fill`` is restored on exit.

    Examples
    --------
    ``with ablate_pair_channels(model, [0], reference=batch): probs = predict(model, data)``
    """
    if fill == "mean":
        if reference is None:
            raise ValueError("fill='mean' needs a reference batch")
        feats = pair_features(reference)
        real = reference.particle_mask
        pairs = real[:, :, None] & real[:, None, :]
        pairs &= ~torch.eye(pairs.shape[-1], dtype=torch.bool, device=pairs.device)
        values = {int(c): float(feats[:, int(c)][pairs].double().mean()) for c in channels}
    elif fill == "zero":
        values = {int(c): 0.0 for c in channels}
    else:
        raise ValueError(f"fill must be 'mean' or 'zero', got {fill!r}")
    previous = model.pair_channel_fill
    model.pair_channel_fill = {**(previous or {}), **values}
    try:
        yield
    finally:
        model.pair_channel_fill = None if previous is None else dict(previous)


@contextlib.contextmanager
def zero_pair_bias(model: ParT) -> Iterator[None]:
    """Set the interaction bias U of a ParT to zero inside the block (a forward hook on its pair embedding).

    A network trained with U and evaluated without it is far off its training distribution: the drop in accuracy
    says little about how useful U is (compare with a model trained without U, ``partsmall_nopair``).
    """
    pair_embed = getattr(getattr(model, "mod", None), "pair_embed", None)
    if pair_embed is None:
        raise ValueError("this model has no pair embedding (pair_embed_dims=None or not a ParT)")
    handle = pair_embed.register_forward_hook(lambda module, args, out: torch.zeros_like(out))
    try:
        yield
    finally:
        handle.remove()


# ----------------------------------------------------------------------------------------------------------------------
# IRC scans


def _forward_chunks(
    model: Tagger, data: JetData, *, label_set: str | None, batch_size: int, embedding: bool
) -> dict[str, np.ndarray]:
    """Logits (or embeddings) and labels of every jet, batch by batch, in evaluation mode without gradients."""
    from .models import make_inputs

    out: dict[str, list[np.ndarray]] = {"values": [], "label": []}
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    with _eval_mode(model), torch.no_grad():
        for chunk in data.chunks(batch_size):
            batch = make_inputs(model, chunk, label_set=label_set).to(device)
            values = model.embed(batch) if embedding else model(batch)
            out["values"].append(values.double().cpu().numpy())
            if batch.label is not None:
                out["label"].append(batch.label.cpu().numpy())
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def irc_scan(
    model: Tagger,
    data: JetData,
    *,
    mode: Literal["soft", "collinear"],
    values: Sequence[float],
    n: int = 1,
    lam: float = 0.5,
    n_jets: int = 1000,
    target: Target = "label",
    output: Literal["score", "embedding"] = "score",
    batch_size: int = 256,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Scan how much a tagger responds to soft or collinear perturbations of the jets (T5.5, T6.6).

    Every perturbed data set comes from :mod:`itj.augment` with ``keep_sign=True`` (the ``part_deta`` sign of the
    unperturbed jet) and is compared with the unperturbed jets passed through ``itj.augment.recompute``, so that
    only the perturbation differs. "soft": ``add_soft(n=n, eps=value)`` (n photons of pT ``value * jet_pt`` each,
    uniform in the jet disc); "collinear": ``split_collinear(which="hardest", lam=lam, delta=value, float64=False)``
    (float32 pieces, as the networks read them). The same random numbers (``seed``) are used for every value, so the
    curves are smooth in the value. Jets that would exceed 128 particles after the perturbation are left out (with a
    log message): the models would truncate them.

    Parameters
    ----------
    model : itj.models.Tagger
        The tagger (in evaluation mode during the scan).
    data : JetData
        Jets; the first ``n_jets`` are used.
    mode : {"soft", "collinear"}
        Perturbation.
    values : sequence of float
        eps (soft) or Delta R of the split (collinear).
    n : int
        Soft particles added per jet (soft mode only).
    lam : float
        Momentum fraction of the first piece of the split (collinear mode only).
    n_jets : int
        Number of jets.
    target : int, "label" or torch.Tensor
        Class whose score is followed ("label": the true class of every jet; ignored for embeddings).
    output : {"score", "embedding"}
        Follow the class probability and logit, or the embedding ``model.embed``.
    batch_size : int
        Jets per forward pass.
    seed : int
        Seed of the perturbations.

    Returns
    -------
    dict of str to numpy.ndarray
        "values" and, per value, for "score": "mean_abs_dscore", "std_abs_dscore" (probability of the target class)
        and "mean_abs_dlogit"; for "embedding": "mean_embed_dist", "std_embed_dist" (L2 distance of the embeddings
        divided by the mean embedding norm of the unperturbed jets).
    """
    from . import augment

    if mode not in ("soft", "collinear"):
        raise ValueError(f"mode must be 'soft' or 'collinear', got {mode!r}")
    if output not in ("score", "embedding"):
        raise ValueError(f"output must be 'score' or 'embedding', got {output!r}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    jets = data.head(n_jets)
    extra = n if mode == "soft" else 1
    fits = jets.counts + extra <= MAX_PARTICLES
    if not fits.all():
        _log.info(
            "irc_scan: %d of %d jets left out (more than %d particles)", int((~fits).sum()), len(jets), MAX_PARTICLES
        )
        jets = jets.select(np.flatnonzero(fits))
    if len(jets) == 0:
        raise ValueError("no jets to scan")
    embedding = output == "embedding"
    label_set = None
    if not embedding and isinstance(target, str):
        label_set = _label_set(model)
    reference = augment.recompute(jets, keep_sign=True)
    ref = _forward_chunks(model, reference, label_set=label_set, batch_size=batch_size, embedding=embedding)

    if not embedding:
        if isinstance(target, str):
            if target != "label":
                raise ValueError(f"target must be a class index, 'label' or a tensor, got {target!r}")
            t = ref["label"].astype(np.int64)
        elif isinstance(target, torch.Tensor):
            t = target.cpu().numpy().astype(np.int64).reshape(-1)
            t = t[: len(data)][:n_jets][fits] if t.size > 1 else np.full(len(jets), int(t.item()))
        else:
            t = np.full(len(jets), int(target))
        rows = np.arange(len(jets))

    def summarise(values: np.ndarray) -> dict[str, float]:
        if embedding:
            norm = float(np.mean(np.linalg.norm(ref["values"], axis=1))) or 1.0
            dist = np.linalg.norm(values - ref["values"], axis=1) / norm
            return {"mean_embed_dist": float(dist.mean()), "std_embed_dist": float(dist.std())}
        logits_ref, logits = ref["values"], values

        def probs(z: np.ndarray) -> np.ndarray:
            e = np.exp(z - z.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        ds = np.abs(probs(logits)[rows, t] - probs(logits_ref)[rows, t])
        dl = np.abs(logits[rows, t] - logits_ref[rows, t])
        return {
            "mean_abs_dscore": float(ds.mean()),
            "std_abs_dscore": float(ds.std()),
            "mean_abs_dlogit": float(dl.mean()),
        }

    results: dict[str, list[float]] = {}
    for value in values:
        rng = np.random.default_rng(seed)
        if mode == "soft":
            pert = augment.add_soft(jets, n=n, eps=float(value), rng=rng, keep_sign=True)
        else:
            pert = augment.split_collinear(
                jets, which="hardest", lam=lam, delta=float(value), rng=rng, keep_sign=True, float64=False
            )
        augment.max_particles_guard(pert, MAX_PARTICLES)
        out = _forward_chunks(model, pert, label_set=None, batch_size=batch_size, embedding=embedding)
        for key, v in summarise(out["values"]).items():
            results.setdefault(key, []).append(v)
    return {"values": np.asarray(values, dtype=np.float64), **{k: np.asarray(v) for k, v in results.items()}}


# ----------------------------------------------------------------------------------------------------------------------
# Groups of particles


def subjet_membership(data: JetData, n: int) -> Ragged:
    """Return the exclusive-kt subjet (0 .. n-1, by subjet pT) of every particle: a reco proxy for the prongs.

    The jets are reclustered with the exclusive kt algorithm (E scheme) until ``n`` subjets remain; on toy jets the
    proxy can be validated against ``truth_part_prong`` before it is used on JetClass (T5.3).

    Parameters
    ----------
    data : JetData
        Jets.
    n : int
        Number of subjets (2 for W jets, 3 for top jets).

    Returns
    -------
    itj.data.Ragged
        int8, aligned with the ``part_*`` columns.
    """
    from .cluster import cluster
    from .data import Ragged

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    parts = []
    for chunk in data.chunks(2048):
        p4, mask = chunk.padded_p4()
        labels = cluster(p4, mask, algorithm="kt", recombination="E").exclusive_labels(n)
        parts.append(labels[mask])
    values = np.concatenate(parts).astype(np.int8) if parts else np.zeros(0, dtype=np.int8)
    offsets = np.asarray(data["part_px"].offsets)  # type: ignore[union-attr]
    return Ragged(values, offsets)


def attention_stats(model: ParT, data: JetData, groups: Ragged, *, batch_size: int = 128) -> dict[str, np.ndarray]:
    """Return how much each attention head stays within particle groups (prongs or subjets), batch by batch.

    Self-attention is counted on its own: for a query particle i with a group (>= 0) in a jet of at least two real
    particles, the attention left after the self term, ``1 - A_ii``, is split into the share on the other real
    particles of the same group and the share on the rest, and the same-group share is averaged over all such queries
    of all jets. Uniform attention over the other real particles gives ``(n_same - 1) / (n_real - 1)`` (the
    "random_expectation", n_same including i). The mean of ``A_ii`` itself is returned as "self": without this split
    a head that mostly attends to the particle itself would look like a head that stays within the prong. Only one
    batch of attention maps is in memory at a time (about ``batch_size * H * P^2 * 4`` bytes per block).

    Parameters
    ----------
    model : itj.models.ParT
        A ParT (``trace(attention=True)``).
    data : JetData
        Jets.
    groups : itj.data.Ragged
        Group of every particle, aligned with ``part_*`` (``truth_part_prong`` or :func:`subjet_membership`); -1 for
        none (the underlying event): such particles are neither queries nor same-group keys.
    batch_size : int
        Jets per forward pass.

    Returns
    -------
    dict of str to numpy.ndarray
        "within_group" and "random_expectation" (n_blocks, H): the mean share of the attention on the other particles
        that stays in the group of the query, and its value for uniform attention (queries with ``1 - A_ii < 1e-6``
        are left out of both); "self" (n_blocks, H): the mean self-attention ``A_ii`` over all queries, those left
        out of "within_group" included;
        "class_within" (n_cls_blocks, H, G): the share of the class token's attention on the particles of group
        g = 0 .. G-1 (G = largest group + 1), averaged over jets; "class_random_expectation" (G,): the share of real
        particles in group g, averaged over jets.
    """
    from .models import make_inputs

    counts = np.asarray(groups.counts())
    if len(counts) != len(data) or not np.array_equal(counts, data.counts):
        raise ValueError("groups must be aligned with the particles of data (same counts per jet)")
    n_groups = int(groups.values.max()) + 1 if len(groups.values) else 0
    n_groups = max(n_groups, 0)
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    within = random = self_sum = n_valid = class_sum = class_random = None
    n_queries = 0.0
    n_jets = 0
    with _eval_mode(model), torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(start + batch_size, len(data)))
            batch = make_inputs(model, data.select(idx), label_set=None).to(device)
            trace = model.trace(batch, attention=True)
            blocks = _ordered(trace.attention, "block")
            cls_blocks = _ordered(trace.class_attention, "cls")
            if not blocks:
                raise ValueError("the model returned no attention maps: attention_stats needs a ParT")
            p = int(batch.mask.shape[-1])
            g = torch.as_tensor(groups.select(idx).padded(p, fill=-1).astype(np.int64), device=device)
            real = trace.mask if trace.mask is not None else batch.particle_mask
            g = torch.where(real, g, torch.full_like(g, -1))
            n_real = real.double().sum(-1, keepdim=True)  # (B, 1)
            query = real & (g >= 0) & (n_real >= 2)
            eye = torch.eye(p, dtype=torch.bool, device=device)
            same = (g[:, :, None] == g[:, None, :]) & query[:, :, None] & real[:, None, :] & ~eye
            same_f = same.double()
            query_f = query.double()
            expect = same_f.sum(-1) / (n_real - 1).clamp(min=1)  # (B, P): uniform over the other particles
            diag = torch.stack([torch.diagonal(a, dim1=-2, dim2=-1).double() for a in blocks])  # (L, B, H, P)
            off = torch.stack([(a.double() * same_f[:, None]).sum(-1) for a in blocks])  # (L, B, H, P)
            rest = 1.0 - diag
            valid = (rest >= 1e-6) & query[None, :, None, :]  # (L, B, H, P)
            valid_f = valid.double()
            share = torch.where(valid, off / rest.clamp(min=1e-6), torch.zeros_like(off))
            w = share.sum(dim=(1, 3)).cpu().numpy()  # (L, H)
            r = (expect[None, :, None, :] * valid_f).sum(dim=(1, 3)).cpu().numpy()
            sd = (diag * query_f[None, :, None, :]).sum(dim=(1, 3)).cpu().numpy()
            nv = valid_f.sum(dim=(1, 3)).cpu().numpy()
            within = w if within is None else within + w
            random = r if random is None else random + r
            self_sum = sd if self_sum is None else self_sum + sd
            n_valid = nv if n_valid is None else n_valid + nv
            n_queries += float(query_f.sum())
            onehot = torch.stack([(g == k) & real for k in range(n_groups)], dim=-1).double()  # (B, P, G)
            c = torch.stack([a[:, :, 0, 1:].double() for a in cls_blocks]) if cls_blocks else None  # (K, B, H, P)
            if c is not None:
                cs = torch.einsum("kbhp,bpg->khg", c, onehot).cpu().numpy()
                class_sum = cs if class_sum is None else class_sum + cs
            cr = (onehot.sum(1) / n_real.clamp(min=1)).sum(0).cpu().numpy()  # (G,)
            class_random = cr if class_random is None else class_random + cr
            n_jets += len(idx)
            del trace, off, diag, share, same
    n_heads = blocks[0].shape[1]
    assert within is not None
    assert random is not None
    assert self_sum is not None
    assert n_valid is not None
    denom = np.maximum(n_valid, 1.0)
    if class_sum is None:
        class_sum = np.zeros((0, n_heads, n_groups))
    return {
        "within_group": within / denom,
        "random_expectation": random / denom,
        "self": self_sum / max(n_queries, 1.0),
        "class_within": np.asarray(class_sum) / max(n_jets, 1),
        "class_random_expectation": np.asarray(class_random) / max(n_jets, 1)
        if class_random is not None
        else np.zeros(n_groups),
    }
