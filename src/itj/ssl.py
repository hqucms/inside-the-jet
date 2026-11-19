"""Learning without labels: JetCLR, masked particles, the JEPA, alignment and uniformity, linear evaluation.

The contrastive set-up of JetCLR (Dillon et al., arXiv:2108.04253): every jet is augmented twice
(:class:`JetCLRAugment`, :func:`two_views`), an encoder (any tagger's ``embed``) and a :class:`ProjectionHead` map
both views to vectors ``z``, and :func:`nt_xent` pulls the two views of a jet together and pushes the other 2B - 2
vectors of the batch away. :func:`alignment` and :func:`uniformity` (Wang and Isola, arXiv:2005.10242) monitor the
two halves of that objective; :func:`linear_eval` measures what the frozen encoder learned with a logistic regression
on few or all labels. :func:`mask_particles` and :class:`MaskedParticleDecoder` prepare masked-particle modelling.

The joint-embedding predictive architecture (JEPA; I-JEPA arXiv:2301.08243, J-JEPA arXiv:2412.05333, HEP-JEPA
arXiv:2502.03933) predicts *representations* of hidden particles instead of their features (B.2):
:func:`subjet_target_mask` hides one exclusive-kT prong of every jet, :func:`context_view` turns the other particles
into a jet of their own for the context encoder, :class:`EMATarget` (an exponential moving average of the context
encoder) encodes the full jet, :class:`JEPAPredictor` predicts the target encoder's output at the hidden particles
from the context and their positions, and :func:`jepa_loss` compares the two. :func:`effective_rank` watches for
collapse.

The pretraining loops themselves are what the students write (W3.4.3, B.2).

Examples
--------
>>> import math, torch
>>> z = torch.ones(8, 4)
>>> round(float(nt_xent(z, z)), 5) == round(math.log(2 * 8 - 1), 5)
True

One JEPA step (B.2) on a few toy jets, with ParT-small without pair features:

>>> from itj.features import make_batch
>>> from itj.models import build_model
>>> from itj.toy import generate
>>> _ = torch.manual_seed(0)
>>> batch = make_batch(generate(8, seed=1), "kin", label_set=None)
>>> model = build_model("partsmall_nopair")
>>> ema = EMATarget(model)
>>> predictor = JEPAPredictor(64, dim=32, num_layers=1)
>>> opt = torch.optim.AdamW([*model.parameters(), *predictor.parameters()], lr=1e-3)
>>> context, target, _ = subjet_target_mask(batch, 2, generator=torch.Generator().manual_seed(0))
>>> view = context_view(batch, context)
>>> trace = model.trace(view.batch)
>>> pred = predictor(trace.layers["block2"], view, target, cls=trace.embedding)
>>> loss = jepa_loss(pred, ema.trace(batch).layers["block2"], target)
>>> opt.zero_grad(); loss.backward(); opt.step()
>>> ema.update(model, ema_momentum(step=0, total=936))
>>> bool(loss > 0), all(p.grad is None for p in ema.model.parameters())
(True, True)
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from .data import JetData
    from .features import JetBatch
    from .models import Tagger
    from .models.base import Trace

__all__ = [
    "ContextView",
    "EMATarget",
    "JEPAPredictor",
    "JetCLRAugment",
    "MaskedParticleDecoder",
    "ProjectionHead",
    "alignment",
    "context_view",
    "effective_rank",
    "ema_momentum",
    "jepa_loss",
    "linear_eval",
    "mask_particles",
    "nt_xent",
    "subjet_labels",
    "subjet_target_mask",
    "two_views",
    "uniformity",
]

#: Class pairs (signal, background) of the one-vs-one AUCs of :func:`linear_eval` per label set.
_PAIRS: dict[str, dict[str, tuple[int, int]]] = {
    "jets3": {"auc_w": (1, 0), "auc_top": (2, 0)},
    "jetclass10": {"auc_w": (7, 0), "auc_top": (8, 0)},
    "qg": {"auc": (1, 0)},
}


def _rng(rng: Any) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


class JetCLRAugment:
    """The JetCLR augmentations as one callable: ``augment(data, rng) -> JetData``.

    Applied in this order (each step from :mod:`itj.augment`, which re-sorts the particles and recomputes
    ``part_deta``, ``part_dphi`` and the jet kinematics): soft-particle dropping, collinear splitting, soft smearing,
    rotation, translation. ``jet_sdmass`` and ``jet_tau*`` are not recomputed (the encoders do not read them). The
    ``truth_*`` columns are dropped first, for speed.

    Parameters
    ----------
    rotate : bool
        Rotate every jet by a random angle about its axis (``itj.augment.rotate``).
    translate : float
        Width of the normal shifts of (Delta eta, Delta phi) per jet (``itj.augment.translate``); 0: off.
    smear_soft : float
        Scale of the soft smearing ``N(0, (scale / pT)^2)`` of the angles (``itj.augment.smear_soft``); 0: off.
    collinear_frac : float
        Fraction of the particles split collinearly into two halves (``itj.augment.split_collinear``); 0: off.
    drop_soft : float
        Probability with which each particle below 1 GeV is removed (``itj.augment.drop_soft``); 0: off.

    Examples
    --------
    ``view = JetCLRAugment()(data, rng)``; ``JetCLRAugment(collinear_frac=0.0)`` for the W4.3.4 ablation.
    """

    def __init__(
        self,
        *,
        rotate: bool = True,
        translate: float = 0.0,
        smear_soft: float = 0.1,
        collinear_frac: float = 0.5,
        drop_soft: float = 0.0,
    ) -> None:
        for name, value in (("translate", translate), ("smear_soft", smear_soft), ("drop_soft", drop_soft)):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if not 0.0 <= collinear_frac <= 1.0:
            raise ValueError(f"collinear_frac must be in [0, 1], got {collinear_frac}")
        if drop_soft > 1.0:
            raise ValueError(f"drop_soft is a probability, got {drop_soft}")
        self.rotate = bool(rotate)
        self.translate = float(translate)
        self.smear_soft = float(smear_soft)
        self.collinear_frac = float(collinear_frac)
        self.drop_soft = float(drop_soft)

    def __repr__(self) -> str:
        return (
            f"JetCLRAugment(rotate={self.rotate}, translate={self.translate}, smear_soft={self.smear_soft}, "
            f"collinear_frac={self.collinear_frac}, drop_soft={self.drop_soft})"
        )

    def __call__(self, data: JetData, rng: Any) -> JetData:
        """Return one augmented view of ``data`` (a new ``JetData`` without truth columns)."""
        from . import augment

        g = _rng(rng)
        out = data.without_truth()
        if self.drop_soft > 0:
            out = augment.drop_soft(out, pt_max=1.0, p=self.drop_soft, rng=g)
        if self.collinear_frac > 0:
            out = augment.split_collinear(out, which="random", frac=self.collinear_frac, lam=0.5, rng=g, float64=False)
        if self.smear_soft > 0 or self.rotate:
            out = _smear_and_rotate(out, scale=self.smear_soft, rotate=self.rotate, rng=g)
        if self.translate > 0:
            n = len(out)
            out = augment.translate(out, deta=g.normal(0.0, self.translate, n), dphi=g.normal(0.0, self.translate, n))
        return out


def _smear_and_rotate(data: JetData, *, scale: float, rotate: bool, rng: np.random.Generator) -> JetData:
    """``augment.smear_soft`` then ``augment.rotate`` in one pass (one re-sort and recomputation instead of two).

    The same random numbers are drawn in the same order as the two separate calls, so the result equals them up to
    the float32 rounding of the intermediate jets.
    """
    from . import augment

    p4, counts, row = augment._particles(data)
    pt, y, phi, mt = augment._decompose(p4)
    if scale > 0:
        with np.errstate(divide="ignore"):
            sigma = np.where(pt > 0, scale / np.where(pt > 0, pt, 1.0), 0.0)
        noise = rng.standard_normal((2, len(pt)))
        y = y + sigma * noise[0]
        phi = phi + sigma * noise[1]
    if rotate:
        n_jets = len(data)
        alpha = rng.uniform(0.0, 2.0 * np.pi, n_jets)
        ay, aphi = augment._jet_axis(augment._compose(pt, y, phi, mt), row, n_jets)
        dy = y - ay[row]
        dphi = augment._wrap(phi - aphi[row])
        c, s = np.cos(alpha)[row], np.sin(alpha)[row]
        y, phi = ay[row] + c * dy - s * dphi, aphi[row] + s * dy + c * dphi
    new = augment._compose(pt, y, phi, mt)
    return augment._rebuild(data, new, counts, np.arange(len(p4)), sign=None)


def two_views(data: JetData, augment: Callable[[JetData, Any], JetData], rng: Any) -> tuple[JetData, JetData]:
    """Return two independently augmented views of the same jets (``augment(data, rng)`` twice, one generator)."""
    g = _rng(rng)
    return augment(data, g), augment(data, g)


# ----------------------------------------------------------------------------------------------------------------------
# Losses and monitors


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    """Return the NT-Xent loss of SimCLR for two batches of projections (reference implementation).

    The 2B vectors are L2-normalised; for every vector the positive is the other view of the same jet and the
    negatives are the other 2B - 2 vectors:
    ``loss = -1/(2B) sum_i log(exp(s_ip / T) / sum_{k != i} exp(s_ik / T))`` with cosine similarities ``s``. Identical
    embeddings for all jets give ``ln(2B - 1)``.

    Parameters
    ----------
    z1, z2 : torch.Tensor
        (B, D) projections of the two views (row i of both belongs to jet i).
    temperature : float
        Temperature T (JetCLR: 0.1).

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError(f"z1 and z2 must both be (B, D), got {tuple(z1.shape)} and {tuple(z2.shape)}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    b = z1.shape[0]
    z = nn.functional.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = (z @ z.T) / temperature
    eye = torch.eye(2 * b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    target = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return nn.functional.cross_entropy(sim, target)


def alignment(z1: torch.Tensor, z2: torch.Tensor, *, alpha: float = 2.0) -> torch.Tensor:
    """Return the alignment ``E ||f(x) - f(x+)||^alpha`` of the L2-normalised views (lower: views closer)."""
    a = nn.functional.normalize(z1, dim=1)
    b = nn.functional.normalize(z2, dim=1)
    return (a - b).norm(dim=1).pow(alpha).mean()


def uniformity(z: torch.Tensor, *, t: float = 2.0) -> torch.Tensor:
    """Return the uniformity ``log E exp(-t ||f(x) - f(y)||^2)`` over distinct pairs (lower: more uniform)."""
    u = nn.functional.normalize(z, dim=1)
    return torch.pdist(u, p=2).pow(2).mul(-t).exp().mean().log()


class ProjectionHead(nn.Module):
    """The projection head of SimCLR and JetCLR: Linear, ReLU, Linear (the loss sees its output).

    Parameters
    ----------
    d_in : int
        Width of the encoder output (``encoder.embed``).
    d_hidden : int
        Width of the hidden layer.
    d_out : int
        Width of the projection.
    """

    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the projections (B, d_out)."""
        out: torch.Tensor = self.net(x)
        return out


# ----------------------------------------------------------------------------------------------------------------------
# Linear evaluation


def _balanced(labels: np.ndarray, n_labels: int, seed: int) -> np.ndarray:
    """Return ``n_labels`` row indices balanced over the classes (sorted)."""
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    per, extra = divmod(int(n_labels), len(classes))
    picks = []
    for k, c in enumerate(classes):
        idx = np.flatnonzero(labels == c)
        want = per + (1 if k < extra else 0)
        if want > len(idx):
            raise ValueError(f"n_labels={n_labels} needs {want} jets of class {c}, the training set has {len(idx)}")
        picks.append(rng.choice(idx, want, replace=False))
    return np.sort(np.concatenate(picks))


def _one_vs_one_auc(probs: np.ndarray, labels: np.ndarray, signal: int, background: int) -> float:
    from sklearn.metrics import roc_auc_score

    sel = (labels == signal) | (labels == background)
    if probs.shape[1] <= max(signal, background) or len(np.unique(labels[sel])) < 2:
        return float("nan")
    ps, pb = probs[sel, signal], probs[sel, background]
    score = ps / np.maximum(ps + pb, 1e-300)
    return float(roc_auc_score(labels[sel] == signal, score))


def linear_eval(
    encoder: Tagger,
    train: JetData,
    test: JetData,
    *,
    n_labels: int | None = None,
    label_set: str = "jets3",
    seed: int = 0,
) -> dict[str, float]:
    """Linear evaluation of a frozen encoder: a logistic regression on its embeddings, scored on the test jets.

    The embeddings come from :func:`itj.probes.extract` (evaluation mode, no gradients) and are standardised with the
    statistics of the training jets.

    Parameters
    ----------
    encoder : itj.models.Tagger
        The encoder (its ``embed``, the input of its final Linear layer, is used).
    train, test : JetData
        Training and test jets with labels.
    n_labels : int, optional
        Number of labelled training jets, balanced over the classes and drawn with ``seed`` (default: all).
    label_set : str
        Label set of ``itj.features.labels`` ("jets3", "jetclass10", "qg").
    seed : int
        Seed of the label draw.

    Returns
    -------
    dict of str to float
        "accuracy", "n_labels" and the one-vs-one AUCs: "auc_w" (W vs QCD) and "auc_top" (top vs QCD) for "jets3"
        and "jetclass10", "auc" for "qg".
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .features import labels as get_labels
    from .probes import extract

    y_train = get_labels(train, label_set)
    y_test = get_labels(test, label_set)
    if n_labels is not None:
        idx = _balanced(y_train, n_labels, seed)
        train, y_train = train.select(idx), y_train[idx]
    x_train = extract(encoder, train)["embedding"]
    x_test = extract(encoder, test)["embedding"]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(x_train, y_train)
    probs = np.zeros((len(x_test), int(max(y_train.max(), y_test.max())) + 1))
    probs[:, clf.classes_] = clf.predict_proba(x_test)
    out = {"accuracy": float(np.mean(probs.argmax(axis=1) == y_test)), "n_labels": float(len(y_train))}
    for key, (s, b) in _PAIRS.get(label_set, {}).items():
        out[key] = _one_vs_one_auc(probs, y_test, s, b)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Masked particles


def mask_particles(
    batch: JetBatch, *, frac: float = 0.3, generator: torch.Generator | None = None, drop: bool = False
) -> tuple[JetBatch, torch.Tensor, torch.Tensor]:
    """Hide a random fraction of the particles of every jet for masked-particle modelling (B.2).

    In every jet ``max(1, round(frac * n))`` real particles are chosen; their features, points and vectors are set to
    0, also in the wrap-padded copies of them. With ``drop=False`` the particle mask is unchanged: the masked
    particles stay in the set as empty tokens at their slots. With ``drop=True`` they are also removed from the
    particle mask, so that the encoder sees only the visible particles (as in a masked autoencoder), and at least one
    particle of every jet stays visible (a jet of one particle keeps it).

    The features of the visible particles that depend on the jet (``part_logptrel``, ``part_logerel``,
    ``part_deltaR``, ``part_deta``, ``part_dphi`` and the points) are left as they were, relative to the *full* jet:
    the sum of ``pT / pT_jet`` over the visible particles is ``1 - `` the masked pT fraction, and their pT-weighted
    centroid is shifted away from the masked particles. The encoder can read both. For a view without these hints,
    see :func:`context_view` (which removes the hidden particles and recomputes the features).

    An empty token carries no position and a ParT without pair features has no positional information: every masked
    slot of a jet then gets the same encoder output, and a regression head on that output alone can only predict the
    mean of the hidden particles. Give the decoder the slot of each masked particle, for example with
    :class:`MaskedParticleDecoder` (the slots are in pT order).

    Parameters
    ----------
    batch : itj.features.JetBatch
        Input batch.
    frac : float
        Fraction of the real particles to mask, in (0, 1].
    generator : torch.Generator, optional
        CPU random generator (for reproducible masks; the draws are moved to the device of the batch).
    drop : bool
        Also remove the masked particles from the particle mask of the returned batch.

    Returns
    -------
    tuple of (JetBatch, torch.Tensor, torch.Tensor)
        The masked batch, the targets (B, F, P) (the original features) and the masked slots (B, P) bool (real
        particles only; the real particles of the input are ``batch.particle_mask``).
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    real = batch.particle_mask
    b, p = real.shape
    n = real.sum(dim=1)
    k = torch.clamp(torch.round(frac * n.to(torch.float64)).to(torch.int64), min=1)
    k = torch.minimum(k, n - 1 if drop else n)
    r = torch.rand(b, p, generator=generator, dtype=torch.float64).to(real.device)
    r = r.masked_fill(~real, float("inf"))
    rank = torch.argsort(torch.argsort(r, dim=1), dim=1)
    masked = (rank < k[:, None]) & real
    # wrap padding repeats particle j % n at padded slot j: hide those copies too
    src = torch.arange(p, device=real.device)[None, :] % n.clamp(min=1)[:, None]
    hidden = torch.gather(masked, 1, src) | masked
    keep = (~hidden).unsqueeze(1).to(batch.features.dtype)
    targets = batch.features.clone()
    out = dataclasses.replace(
        batch,
        features=batch.features * keep,
        points=batch.points * keep.to(batch.points.dtype),
        vectors=batch.vectors * keep.to(batch.vectors.dtype),
        mask=batch.mask * (~masked).unsqueeze(1).to(batch.mask.dtype) if drop else batch.mask,
    )
    return out, targets, masked


class MaskedParticleDecoder(nn.Module):
    """A decoder for masked-particle modelling that knows the slot of every particle (B.2).

    A learned embedding of the slot index (the pT rank, since the particles are stored in pT order) is added to the
    projected encoder output at every slot; without it all masked slots of a jet would get the same prediction. The
    default, ``num_layers=0``, is a shallow head (a GELU and a linear layer after the sum) for
    ``mask_particles(..., drop=False)``: the masked particles stay in the set as empty tokens, and the encoder itself
    has to write what it infers about the hidden particles into its output at their slots. With ``num_layers > 0`` it
    is the decoder of a masked autoencoder (He et al., arXiv:2111.06377), for ``drop=True``: a masked slot gets a
    learned mask token instead of the encoder output, and ``num_layers`` transformer layers over the real slots let it
    read the encoded visible particles. Such a decoder does part of the inference itself, so the encoder has less to
    learn; compare both with the linear evaluation.

    Parameters
    ----------
    d_in : int
        Width of the per-particle encoder output (64 for ParT-small).
    n_targets : int
        Number of predicted features per particle.
    dim : int
        Width of the decoder.
    num_layers : int
        Transformer layers of the decoder; 0 (the default) gives the shallow head.
    num_heads : int
        Attention heads of the transformer layers (only with ``num_layers > 0``).
    max_len : int
        Largest number of slots (:data:`itj.features.MAX_LENGTH`).

    Examples
    --------
    Two masked slots with the same encoder output still get different predictions:

    >>> import torch
    >>> dec = MaskedParticleDecoder(8, 3, dim=16)
    >>> h, real = torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.bool)
    >>> h[0, 2] = h[0, 0]
    >>> masked = torch.tensor([[True, False, True, False, False], [False] * 4 + [True]])
    >>> pred = dec(h, real, masked)
    >>> pred.shape, bool(torch.allclose(pred[0, 0], pred[0, 2]))
    (torch.Size([2, 5, 3]), False)
    """

    def __init__(
        self,
        d_in: int,
        n_targets: int = 3,
        *,
        dim: int = 128,
        num_layers: int = 0,
        num_heads: int = 4,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError(f"num_layers must be at least 0, got {num_layers}")
        self.max_len = max_len
        self.proj = nn.Linear(d_in, dim)
        self.slot = nn.Embedding(max_len, dim)
        nn.init.normal_(self.slot.weight, std=0.02)
        self.mask_token: nn.Parameter | None = None
        self.layers: nn.TransformerEncoder | None = None
        self.norm: nn.Module = nn.GELU()
        if num_layers > 0:
            self.mask_token = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.mask_token, std=0.02)
            layer = nn.TransformerEncoderLayer(
                dim,
                num_heads,
                dim_feedforward=4 * dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.layers = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, n_targets)

    def forward(self, h: torch.Tensor, real: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        """Return the predictions (B, P, n_targets) at every slot (only the masked ones are meaningful).

        Parameters
        ----------
        h : torch.Tensor
            (B, P, d_in) per-particle encoder output of the masked batch, for example
            ``model.trace(masked_batch).layers["block2"]`` for ParT-small (with ``num_layers > 0`` its values at the
            masked slots are not read).
        real : torch.Tensor
            (B, P) bool, the real particles of the batch before masking (``batch.particle_mask``).
        masked : torch.Tensor
            (B, P) bool, the masked slots (from :func:`mask_particles`).
        """
        p = h.shape[1]
        if p > self.max_len:
            raise ValueError(f"the batch has {p} slots, more than max_len={self.max_len}")
        x = self.proj(h)
        if self.mask_token is not None:
            x = torch.where(masked[..., None], self.mask_token.to(x.dtype), x)
        x = x + self.slot(torch.arange(p, device=h.device))[None].to(x.dtype)
        if self.layers is not None:
            x = self.layers(x, src_key_padding_mask=~real)
        return self.out(self.norm(x))


# ----------------------------------------------------------------------------------------------------------------------
# Joint-embedding predictive architecture (JEPA): mask one prong


def subjet_labels(batch: JetBatch, n_subjets: int = 2, *, algorithm: str = "kt", R: float = 0.8) -> torch.Tensor:
    """Return the exclusive subjet of every particle of a batch (the prongs that :func:`subjet_target_mask` hides).

    The particles of every jet are clustered with :func:`itj.cluster.cluster` (exclusive mode) on the four-vectors
    of the batch and the last ``n_subjets - 1`` merges are undone (``ClusterTree.exclusive_labels``). With the kT
    algorithm the soft particles merge first, so the exclusive subjets are the hard prongs: the two quarks of a W
    jet for ``n_subjets=2``.

    Parameters
    ----------
    batch : itj.features.JetBatch
        Input batch (its ``vectors`` and ``particle_mask`` are read).
    n_subjets : int
        Number of exclusive subjets (>= 1).
    algorithm : {"kt", "ca", "antikt"}
        Clustering algorithm.
    R : float
        Jet radius of the distance.

    Returns
    -------
    torch.Tensor
        (B, P) int64, the subjet 0 .. n_subjets - 1 of every particle (ordered by pT, 0 the hardest); -1 for padding.
        A jet with k < n_subjets particles has k subjets.

    Notes
    -----
    The clustering costs O(N^3) per jet: about 0.4 to 0.9 ms per jet of 40 particles on one core, 0.1 to 0.2 s per
    batch of 256 jets of ``t6_pretrain``, about 3 min over 936 batches. That is affordable on the fly
    (``labels=None`` in :func:`subjet_target_mask`). To pay it once instead (about 13 s for 30 000 jets), cluster the
    whole dataset padded to its full length and take the rows and slots of every batch (``batch.index`` are the
    dataset rows)::

        all_labels = subjet_labels(ds.batch(np.arange(len(ds)), length=ds.length), 2)  # (N, ds.length)
        ...
        labels = all_labels[batch.index, : batch.mask.shape[-1]]
        context, target, _ = subjet_target_mask(batch, 2, labels=labels)
    """
    from .cluster import cluster

    if n_subjets < 1:
        raise ValueError(f"n_subjets must be >= 1, got {n_subjets}")
    p4 = batch.vectors.detach().permute(0, 2, 1).to("cpu", torch.float64).numpy()
    real = batch.particle_mask.cpu().numpy()
    tree = cluster(p4, real, algorithm=algorithm, R=R)  # type: ignore[arg-type]
    return torch.from_numpy(tree.exclusive_labels(n_subjets)).to(batch.mask.device)


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


def _axis(vectors: torch.Tensor, keep: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return pT, E, eta and phi (B,) of the four-vector sum of the particles ``keep`` (float64)."""
    s = (vectors.to(torch.float64) * keep[:, None, :]).sum(dim=-1)  # (B, 4)
    pt = torch.hypot(s[:, 0], s[:, 1])
    return {
        "pt": pt,
        "energy": s[:, 3],
        "eta": torch.asinh(s[:, 2] / pt.clamp(min=1e-12)),
        "phi": torch.atan2(s[:, 1], s[:, 0]),
    }


def _relative(vectors: torch.Tensor, axis: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Delta eta, Delta phi) (B, P) of every slot relative to ``axis`` (the convention of ``part_deta``).

    ``part_deta = s (eta - eta_jet)`` with s = +1 if ``eta_jet >= 0`` else -1, and the wrapped Delta phi.
    """
    v = vectors.to(torch.float64)
    pt = torch.hypot(v[:, 0], v[:, 1])
    eta = torch.asinh(v[:, 2] / pt.clamp(min=1e-12))
    phi = torch.atan2(v[:, 1], v[:, 0])
    sign = torch.where(axis["eta"] >= 0, 1.0, -1.0).to(torch.float64)
    return sign[:, None] * (eta - axis["eta"][:, None]), _wrap_angle(phi - axis["phi"][:, None])


def subjet_target_mask(
    batch: JetBatch,
    n_subjets: int = 2,
    *,
    generator: torch.Generator | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose one exclusive-kT subjet of every jet as the JEPA target; the other particles are the context.

    The subjet is drawn uniformly among the subjets of the jet (:func:`subjet_labels`). A jet with a single particle
    has no target (all of it is context), so the context of every jet with particles is never empty.

    Parameters
    ----------
    batch : itj.features.JetBatch
        Input batch.
    n_subjets : int
        Number of exclusive kT subjets per jet (2: one of the two prongs of a W jet is hidden).
    generator : torch.Generator, optional
        CPU random generator of the subjet draw (for reproducible masks).
    labels : torch.Tensor, optional
        (B, P) precomputed :func:`subjet_labels` of the batch with the same ``n_subjets`` (see its Notes for how to
        slice them from labels of a whole dataset); default: cluster the batch now.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
        ``context`` (B, P) bool and ``target`` (B, P) bool (disjoint; together the real particles of jets with a
        target) and the position of the target subjet (B, 3): (Delta eta, Delta phi, ln(pT_target / pT_context)) of
        its four-vector sum relative to the four-vector sum of the context (the frame of :func:`context_view`); 0 for
        jets without target. The position is for the predictor only (a query placed at the hidden prong, as in a
        prong-level target); it describes the target, so it must never reach the context encoder. The per-particle
        recipe does not need it (``context, target, _ = ...``).
    """
    if n_subjets < 2:
        raise ValueError(f"n_subjets must be >= 2 (one target and at least one context subjet), got {n_subjets}")
    real = batch.particle_mask
    lab = subjet_labels(batch, n_subjets) if labels is None else labels.to(real.device)
    if lab.shape != real.shape:
        raise ValueError(f"labels must have the shape {tuple(real.shape)} of the batch, got {tuple(lab.shape)}")
    lab = torch.where(real, lab, -1)
    present = torch.stack([(lab == a).any(dim=1) for a in range(n_subjets)], dim=1)  # (B, n)
    n_present = present.sum(dim=1)
    r = torch.rand(len(real), generator=generator, dtype=torch.float64).to(real.device)
    k = torch.clamp((r * n_present).long(), max=(n_present - 1).clamp(min=0))  # k-th present subjet
    rank = torch.cumsum(present.long(), dim=1) - 1
    chosen = torch.argmax((present & (rank == k[:, None])).long(), dim=1)
    target = real & (lab == chosen[:, None]) & (n_present >= 2)[:, None]
    context = real & ~target
    ctx_axis = _axis(batch.vectors, context)
    tgt_axis = _axis(batch.vectors, target)
    sign = torch.where(ctx_axis["eta"] >= 0, 1.0, -1.0).to(torch.float64)
    has = target.any(dim=1)
    position = torch.stack(
        [
            sign * (tgt_axis["eta"] - ctx_axis["eta"]),
            _wrap_angle(tgt_axis["phi"] - ctx_axis["phi"]),
            torch.log(tgt_axis["pt"].clamp(min=1e-12) / ctx_axis["pt"].clamp(min=1e-12)),
        ],
        dim=1,
    )
    position = torch.where(has[:, None], position, 0.0).to(torch.float32)
    return context, target, position


@dataclasses.dataclass
class ContextView:
    """What the context encoder of a JEPA sees (:func:`context_view`).

    Attributes
    ----------
    batch : itj.features.JetBatch
        The context particles as jets of their own: moved to the front of every jet (pT order kept), wrap-padded
        with themselves, padded to the longest context (P_c slots), and with the jet-relative features and points
        recomputed relative to the four-vector sum of the context.
    slots : torch.Tensor
        (B, P_c) int64, the slot of the input batch of every context slot (meaningful where
        ``batch.particle_mask``).
    context : torch.Tensor
        (B, P) bool, the context particles in the slots of the input batch.
    positions : torch.Tensor
        (B, P, 2) (Delta eta, Delta phi) of every real particle of the input batch (context and hidden) relative to
        the context axis: the positional information of the predictor's queries; 0 at padded slots.
    """

    batch: JetBatch
    slots: torch.Tensor
    context: torch.Tensor
    positions: torch.Tensor


def context_view(batch: JetBatch, context: torch.Tensor) -> ContextView:
    """Return the context particles of every jet as a jet of their own, so that nothing of the target can leak.

    A jet made of the context particles only, as :class:`itj.features.JetDataset` would build it: the hidden
    particles are removed (not only masked), the context particles move to the front in their pT order, the padded
    slots repeat them (wrap padding), and the features that depend on the jet (``part_logptrel``,
    ``part_logerel``, ``part_deltaR``, ``part_deta``, ``part_dphi`` and the points) are recomputed relative to the
    four-vector sum of the context. Nothing of the hidden particles is left in the features, points, vectors, mask,
    padding, pair features (built from the vectors) or the batch statistics of the input BatchNorm. Keeping the
    original jet-relative features would leak: ``sum exp(ln pT_rel)`` over the context is 1 - pT_target / pT_jet,
    and the pT-weighted centroid of the context points sits opposite the hidden prong.

    The price: the context encoder never sees a whole jet. Its input BatchNorm learns the statistics of context
    jets (with ``n_subjets=2`` about half a jet, so larger ``part_logptrel`` and smaller ``part_deltaR`` than in
    full jets), and :class:`EMATarget` copies them, so both encoders meet full jets at evaluation with input
    statistics of another distribution.

    Parameters
    ----------
    batch : itj.features.JetBatch
        Input batch (feature set "kin", "kinpid", "full" or None).
    context : torch.Tensor
        (B, P) bool, the particles to keep (only real particles count).

    Returns
    -------
    ContextView
        The context batch, the map back to the input slots and the positions of every particle in the context
        frame. With ``context = batch.particle_mask`` the batch equals the input up to float rounding (for jets whose
        ``jet_pt``, ``jet_energy`` and axis are the four-vector sum of their particles, as in the toy).
    """
    from .features import FEATURE_SETS

    real = batch.particle_mask
    if context.shape != real.shape:
        raise ValueError(f"context must have the shape {tuple(real.shape)} of the batch, got {tuple(context.shape)}")
    keep = context.to(torch.bool) & real
    b, p = keep.shape
    k = keep.sum(dim=1)
    pc = max(int(k.max()) if b else 1, 1)
    # context slots first, in their order (a stable sort of the flags)
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    j = torch.arange(pc, device=keep.device)[None, :]
    src = torch.gather(order, 1, (j % k.clamp(min=1)[:, None]).clamp(max=p - 1))  # (B, P_c) wrap-padded
    valid_jet = (k > 0)[:, None]
    new_mask = (j < k[:, None]) & valid_jet

    def take(x: torch.Tensor) -> torch.Tensor:
        out = torch.gather(x, 2, src[:, None, :].expand(-1, x.shape[1], -1))
        return out * valid_jet[:, None, :].to(out.dtype)

    vectors = take(batch.vectors)
    features = take(batch.features)
    axis = _axis(batch.vectors, keep)
    deta_all, dphi_all = _relative(batch.vectors, axis)  # (B, P) in the input slots
    deta = torch.gather(deta_all, 1, src)
    dphi = torch.gather(dphi_all, 1, src)
    points = torch.stack([deta, dphi], dim=1).to(batch.points.dtype) * valid_jet[:, None, :].to(batch.points.dtype)
    if batch.feature_set is not None and features.shape[1] > 0:
        names = [v.name for v in FEATURE_SETS[batch.feature_set]]
        v64 = vectors.to(torch.float64)
        pt = torch.hypot(v64[:, 0], v64[:, 1])
        raw = {
            "part_logptrel": torch.log(pt / axis["pt"][:, None]),
            "part_logerel": torch.log(v64[:, 3] / axis["energy"][:, None]),
            "part_deltaR": torch.hypot(deta, dphi),
            "part_deta": deta,
            "part_dphi": dphi,
        }
        features = features.clone()
        for var in FEATURE_SETS[batch.feature_set]:
            if var.name not in raw:
                continue
            x = raw[var.name]
            if var.center is not None:
                x = ((x - var.center) * var.scale).clamp(var.clip_min, var.clip_max)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) * valid_jet
            features[:, names.index(var.name)] = x.to(features.dtype)
    out = dataclasses.replace(
        batch,
        points=points,
        features=features,
        vectors=vectors,
        mask=new_mask[:, None, :].to(batch.mask.dtype),
    )
    positions = torch.stack([deta_all, dphi_all], dim=-1) * real[..., None]
    positions = torch.nan_to_num(positions, nan=0.0).to(torch.float32)
    return ContextView(batch=out, slots=src, context=keep, positions=positions)


class EMATarget:
    """The target encoder of a JEPA: an exponential-moving-average copy of the context encoder, without gradients.

    ``update(model, m)`` sets every parameter to ``m * target + (1 - m) * online`` (BYOL, I-JEPA) and copies the
    buffers (the running statistics of the input BatchNorm, which the online encoder gathers on context views, not on
    full jets: see :func:`context_view`). The copy always runs in evaluation mode (no dropout) and under
    ``torch.no_grad``.

    Parameters
    ----------
    model : itj.models.Tagger
        The context (online) encoder; it is deep-copied.

    Attributes
    ----------
    model : itj.models.Tagger
        The target encoder (``requires_grad`` False everywhere).
    """

    def __init__(self, model: Tagger) -> None:
        import copy

        self.model: Tagger = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, model: Tagger, momentum: float) -> None:
        """Move the target towards ``model``: ``theta_t <- m theta_t + (1 - m) theta`` (m = 1: frozen)."""
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        online = dict(model.named_parameters())
        for name, p in self.model.named_parameters():
            p.mul_(momentum).add_(online[name].detach().to(p.dtype), alpha=1.0 - momentum)
        buffers = dict(model.named_buffers())
        for name, buf in self.model.named_buffers():
            buf.copy_(buffers[name])

    @torch.no_grad()
    def trace(self, batch: JetBatch) -> Trace:
        """Return ``model.trace(batch)`` of the target encoder (evaluation mode, no gradients, detached)."""
        self.model.eval()
        return self.model.trace(batch).detach()


def ema_momentum(step: int, total: int, *, start: float = 0.996, end: float = 1.0) -> float:
    """Return the EMA momentum of step ``step`` of ``total``: linear from ``start`` to ``end`` (I-JEPA).

    The I-JEPA default 0.996 -> 1 was made for runs of about 1e5 steps. Over a short run the target keeps a share
    ``prod_t m_t`` (about ``exp(-(1 - start) * total / 2)``) of its starting weights: 0.15 for 0.996 -> 1 over 936
    steps (the B.2 budget), 0.01 for 0.99 -> 1. For a target that has forgotten its random start, choose ``start``
    with ``(1 - start) * total / 2`` at least about 4.

    Examples
    --------
    >>> import math
    >>> round(math.prod(ema_momentum(t, 936) for t in range(1, 937)), 2)
    0.15
    """
    if total <= 0:
        raise ValueError(f"total must be > 0, got {total}")
    return float(start + (end - start) * min(max(step, 0), total) / total)


class JEPAPredictor(nn.Module):
    """The predictor of a JEPA: a small transformer from the context encoding to the hidden particles.

    Every context particle becomes a token (a projection of its encoder output); every hidden particle of the target
    becomes a query, a learned mask token (as in I-JEPA). The queries get a positional embedding of their (Delta eta,
    Delta phi) in the frame of the context (``ContextView.positions``): the predictor is told *where* the target
    particles are, not their momenta. With ``context_positions`` (the default, as I-JEPA) the context tokens get the
    same embedding of their own positions. With ``cls`` the class token of the context encoder is a further token,
    so that the loss also trains the class-attention blocks. ``num_layers`` transformer layers mix the tokens; the
    output at every target slot is the predicted representation of that particle.

    The positions are data, not a fixed grid as in I-JEPA: they carry the angular layout of the whole jet and the
    number of hidden particles. A low loss alone therefore does not show that the encoder learned anything. Compare
    the loss with that of a positions-only predictor (``h`` set to 0 and no ``cls``), and with that of a predictor
    trained on a frozen untrained encoder.

    Parameters
    ----------
    d_in : int
        Width of the per-particle encoder output (64 for ParT-small).
    d_out : int, optional
        Width of the predicted representation (default ``d_in``).
    dim : int
        Width of the predictor.
    num_layers : int
        Transformer layers (>= 1).
    num_heads : int
        Attention heads.
    context_positions : bool
        Add the positional embedding to the context tokens too (False: to the queries only; the context tokens then
        carry only what the encoder wrote into them).

    Examples
    --------
    One jet of four particles, the third hidden:

    >>> import torch
    >>> from itj.features import JetBatch
    >>> vec = torch.tensor([[[3.0, 2.0, 1.0, 1.0], [0.0, 0.5, -0.5, 0.2], [0.0, 0.0, 0.0, 0.0], [3.0, 2.1, 1.2, 1.1]]])
    >>> batch = JetBatch(points=torch.zeros(1, 2, 4), features=torch.zeros(1, 0, 4), vectors=vec,
    ...                  mask=torch.ones(1, 1, 4), label=None, weight=None, index=torch.zeros(1, dtype=torch.long),
    ...                  jet=None, feature_set=None)
    >>> target = torch.tensor([[False, False, True, False]])
    >>> view = context_view(batch, ~target)
    >>> view.batch.mask.shape, view.slots.tolist()
    (torch.Size([1, 1, 3]), [[0, 1, 3]])
    >>> pred = JEPAPredictor(8, dim=16, num_heads=2)
    >>> h = torch.randn(1, 3, 8)  # the context encoder's output on view.batch
    >>> pred(h, view, target, cls=torch.randn(1, 8)).shape
    torch.Size([1, 4, 8])
    """

    def __init__(
        self,
        d_in: int,
        d_out: int | None = None,
        *,
        dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        context_positions: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        self.context_positions = bool(context_positions)
        self.proj = nn.Linear(d_in, dim)
        self.cls_proj = nn.Linear(d_in, dim)
        self.pos = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dim, num_heads, dim_feedforward=4 * dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.layers = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, d_in if d_out is None else d_out)

    def forward(
        self, h: torch.Tensor, view: ContextView, target: torch.Tensor, cls: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the predictions (B, P, d_out) in the slots of the input batch (only the target slots are trained).

        Parameters
        ----------
        h : torch.Tensor
            (B, P_c, d_in) per-particle output of the context encoder on ``view.batch``, for example
            ``model.trace(view.batch).layers["block2"]``.
        view : ContextView
            The context view (:func:`context_view`): its slots, real context particles and positions are read.
        target : torch.Tensor
            (B, P) bool, the hidden particles to predict (from :func:`subjet_target_mask`).
        cls : torch.Tensor, optional
            (B, d_in) class token of the context encoder (``model.trace(view.batch).embedding``).
        """
        slots, valid, positions = view.slots, view.batch.particle_mask, view.positions
        b, p = target.shape
        # the input slots of the real context particles (padded context slots repeat them and add nothing)
        ctx = torch.zeros(b, p, dtype=torch.int64, device=target.device).scatter_add(1, slots, valid.long()) > 0
        if (ctx & target).any():
            raise ValueError("a target slot is also a context slot: the context must not contain the target")
        src = self.proj(h) * valid[..., None].to(h.dtype)
        x = h.new_zeros(b, p, src.shape[-1]).scatter_add(1, slots[..., None].expand(-1, -1, src.shape[-1]), src)
        x = torch.where(target[..., None], self.mask_token.to(x.dtype), x)
        pos = self.pos(positions.to(x.dtype))
        x = x + (pos if self.context_positions else pos * target[..., None].to(x.dtype))
        keep = target | ctx
        if cls is not None:
            x = torch.cat([self.cls_proj(cls)[:, None, :], x], dim=1)
            keep = torch.cat([torch.ones_like(keep[:, :1]), keep], dim=1)
        y = self.out(self.norm(self.layers(x, src_key_padding_mask=~keep)))
        return y[:, 1:] if cls is not None else y


def jepa_loss(
    pred: torch.Tensor, target_repr: torch.Tensor, target: torch.Tensor, *, normalise: bool = True, kind: str = "l2"
) -> torch.Tensor:
    """Return the JEPA loss: the distance in embedding space between predicted and target representations.

    Parameters
    ----------
    pred : torch.Tensor
        (B, P, D) predictions (:class:`JEPAPredictor`).
    target_repr : torch.Tensor
        (B, P, D) representations of the target encoder on the full jets, for example
        ``ema.trace(batch).layers["block2"]``; never differentiated (detached here).
    target : torch.Tensor
        (B, P) bool, the hidden particles; the loss is the mean over them.
    normalise : bool
        Layer-normalise every target vector first (no affine parameters, as I-JEPA and data2vec): the loss is then
        independent of the scale of the target encoder, which can neither shrink nor blow up to change it.
        Normalisation fixes the scale, not collapse: a target encoder that gives every particle of every jet the
        same vector still gives a loss of 0 to a predictor that outputs its layer norm. Only the asymmetry of the
        EMA target and the predictor stands against such a constant solution; watch :func:`effective_rank` of the
        embeddings during pretraining.
    kind : {"l2", "smooth_l1"}
        Mean squared error per component (I-JEPA) or the smooth L1 loss with beta = 1 (data2vec).

    Returns
    -------
    torch.Tensor
        Scalar loss; 0 for a perfect prediction (and for a batch without target particles).

    Examples
    --------
    >>> import torch
    >>> t = torch.randn(2, 4, 8)
    >>> target = torch.tensor([[True, False, True, False], [False] * 3 + [True]])
    >>> float(jepa_loss(torch.nn.functional.layer_norm(t, (8,)), t, target))
    0.0
    """
    if pred.shape != target_repr.shape:
        raise ValueError(
            f"pred and target_repr must have the same shape, got {tuple(pred.shape)} and {tuple(target_repr.shape)}"
        )
    t = target_repr.detach()
    if normalise:
        t = nn.functional.layer_norm(t, t.shape[-1:])
    if not target.any():
        return pred.sum() * 0.0
    a, b = pred[target], t[target]
    if kind == "l2":
        return nn.functional.mse_loss(a, b)
    if kind == "smooth_l1":
        return nn.functional.smooth_l1_loss(a, b, beta=1.0)
    raise ValueError(f"kind must be 'l2' or 'smooth_l1', got {kind!r}")


def effective_rank(z: torch.Tensor | np.ndarray) -> float:
    """Return the effective rank of a set of embeddings (RankMe; Roy and Vetterli 2007, Garrido et al. 2023).

    ``exp(-sum_k p_k ln p_k)`` with ``p_k = s_k / sum s`` of the singular values s of the centred (N, D) matrix:
    D for embeddings spread evenly over D directions, 1 for embeddings on a line; a collapsed encoder (every jet the
    same vector) has no spread at all and gives 1 by convention.

    Examples
    --------
    >>> import torch
    >>> round(effective_rank(torch.eye(4) - 0.25), 3)
    3.0
    """
    x = torch.as_tensor(np.asarray(z) if not isinstance(z, torch.Tensor) else z).detach().to(torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    total = s.sum()
    if not total > 1e-12 * max(1.0, float(x.abs().max())):
        return 1.0
    q = s / total
    q = q[q > 0]
    return float(torch.exp(-(q * q.log()).sum()))
