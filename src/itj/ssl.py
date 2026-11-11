"""Learning without labels: JetCLR augmentations, the NT-Xent loss, alignment and uniformity, linear evaluation.

The contrastive set-up of JetCLR (Dillon et al., arXiv:2108.04253): every jet is augmented twice
(:class:`JetCLRAugment`, :func:`two_views`), an encoder (any tagger's ``embed``) and a :class:`ProjectionHead` map
both views to vectors ``z``, and :func:`nt_xent` pulls the two views of a jet together and pushes the other 2B - 2
vectors of the batch away. :func:`alignment` and :func:`uniformity` (Wang and Isola, arXiv:2005.10242) monitor the
two halves of that objective; :func:`linear_eval` measures what the frozen encoder learned with a logistic regression
on few or all labels. :func:`mask_particles` prepares masked-particle modelling (T7.1).

The pretraining loop itself is what the students write (T6.3).

Examples
--------
>>> import math, torch
>>> z = torch.ones(8, 4)
>>> round(float(nt_xent(z, z)), 5) == round(math.log(2 * 8 - 1), 5)
True
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from .data import JetData
    from .features import JetBatch
    from .models import Tagger

__all__ = [
    "JetCLRAugment",
    "ProjectionHead",
    "alignment",
    "linear_eval",
    "mask_particles",
    "nt_xent",
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
    ``view = JetCLRAugment()(data, rng)``; ``JetCLRAugment(collinear_frac=0.0)`` for the T6.6 ablation.
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
            out = augment.split_collinear(out, which="random", frac=self.collinear_frac, lam=0.5, rng=g)
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
    batch: JetBatch, *, frac: float = 0.3, generator: torch.Generator | None = None
) -> tuple[JetBatch, torch.Tensor, torch.Tensor]:
    """Hide a random fraction of the particles of every jet for masked-particle modelling (T7.1).

    In every jet ``max(1, round(frac * n))`` real particles are chosen; their features, points and vectors are set to
    0 (also in the wrap-padded copies of them, so that nothing leaks). The particle mask is unchanged: the masked
    particles stay in the set as empty tokens at their slots, and the model predicts their features there.

    Parameters
    ----------
    batch : itj.features.JetBatch
        Input batch.
    frac : float
        Fraction of the real particles to mask, in (0, 1].
    generator : torch.Generator, optional
        CPU random generator (for reproducible masks; the draws are moved to the device of the batch).

    Returns
    -------
    tuple of (JetBatch, torch.Tensor, torch.Tensor)
        The masked batch, the targets (B, F, P) (the original features) and the masked slots (B, P) bool (real
        particles only).
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    real = batch.particle_mask
    b, p = real.shape
    n = real.sum(dim=1)
    k = torch.clamp(torch.round(frac * n.to(torch.float64)).to(torch.int64), min=1)
    k = torch.minimum(k, n)
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
    )
    return out, targets, masked
