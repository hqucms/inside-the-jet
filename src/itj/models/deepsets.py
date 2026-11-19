"""Deep Sets taggers: the Particle Flow Network (PFN) and the Energy Flow Network (EFN).

Both sum a per-particle network Phi over the particles of a jet and feed the sum to a network F
(Komiske, Metodiev and Thaler, `arXiv:1810.05165 <https://arxiv.org/abs/1810.05165>`_)::

    PFN:  F( sum_i Phi(x_i) )                       x_i = the particle features of the feature set
    EFN:  F( sum_i z_i Phi(Delta eta_i, Delta phi_i) )   z_i = pT_i / sum_j pT_j

The sum over the particles makes both invariant under permutations and padding; the EFN, linear in the momentum
fractions and blind to everything but the angles, is also infrared and collinear safe: a particle split into two
collinear ones (same angles, z = z_a + z_b) or a particle with z -> 0 leaves its output unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from ..features import FEATURE_SETS, JetBatch
from .base import Tagger, Trace, check_feature_set, masked_mean

__all__ = ["EFN", "PFN"]


def _mlp(n_in: int, sizes: Sequence[int]) -> nn.Sequential:
    """Linear + ReLU layers of the given widths (an empty Sequential for no layers)."""
    layers: list[nn.Module] = []
    for n_out in sizes:
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        n_in = n_out
    return nn.Sequential(*layers)


class PFN(Tagger):
    """Particle Flow Network: ``F(sum_i Phi(x_i))`` with the sum over the real particles only.

    Parameters
    ----------
    input_dim : int
        Number of particle features (7 kin, 13 kinpid, 17 full).
    n_classes : int
        Number of outputs.
    phi_sizes : sequence of int
        Widths of Phi (Linear + ReLU per layer); the last is the latent dimension.
    f_sizes : sequence of int
        Widths of F (Linear + ReLU per layer); the last is the embedding dimension.
    feature_set : str or None
        The particle features the model reads.

    Attributes
    ----------
    phi, f : torch.nn.Sequential
        The per-particle network and the jet network.
    out : torch.nn.Linear
        The output layer.
    """

    default_name = "pfn"

    def __init__(
        self,
        input_dim: int = 7,
        n_classes: int = 3,
        phi_sizes: Sequence[int] = (128, 128, 128),
        f_sizes: Sequence[int] = (128, 128, 128),
        feature_set: str | None = "kin",
    ) -> None:
        super().__init__()
        check_feature_set(feature_set, input_dim)
        if not phi_sizes:
            raise ValueError("phi_sizes must have at least one layer")
        self.input_dim = int(input_dim)
        self.phi = _mlp(self.input_dim, phi_sizes)
        self.f = _mlp(phi_sizes[-1], f_sizes)
        self.out = nn.Linear(f_sizes[-1] if f_sizes else phi_sizes[-1], n_classes)
        init = {
            "input_dim": self.input_dim,
            "n_classes": int(n_classes),
            "phi_sizes": tuple(int(s) for s in phi_sizes),
            "f_sizes": tuple(int(s) for s in f_sizes),
            "feature_set": feature_set,
        }
        self._set_config(init, feature_set=feature_set, jet_features=None)

    @property
    def head(self) -> nn.Linear:
        """The output layer."""
        return self.out

    def _phi(self, batch: JetBatch) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_features(batch, self.input_dim)
        mask = batch.particle_mask
        h = self.phi(batch.features.transpose(1, 2)) * mask.unsqueeze(-1).to(batch.features.dtype)
        return h, mask

    def embed(self, batch: JetBatch) -> torch.Tensor:
        """Return F of the latent sum (B, D)."""
        h, _ = self._phi(batch)
        return self.f(h.sum(dim=1))

    def trace(self, batch: JetBatch, *, attention: bool = False) -> Trace:
        """Return the :class:`Trace`: layer ``"phi"`` (B, P, latent), pooled ``"phi"`` and ``"latent"`` (the sum)."""
        h, mask = self._phi(batch)
        latent = h.sum(dim=1)
        embedding = self.f(latent)
        return Trace(
            logits=self.out(embedding),
            embedding=embedding,
            mask=mask,
            layers={"phi": h},
            pooled={"phi": masked_mean(h, mask), "latent": latent},
        )


class EFN(Tagger):
    """Energy Flow Network: ``F(sum_i z_i Phi(Delta eta_i, Delta phi_i))``, infrared and collinear safe.

    It reads only ``batch.points`` (Delta eta, Delta phi) and ``batch.vectors`` (for z_i = pT_i / sum_j pT_j), never
    ``batch.features``.

    Parameters
    ----------
    n_classes : int
        Number of outputs.
    phi_sizes : sequence of int
        Widths of Phi (Linear + ReLU per layer, input the two angles); the last is the latent dimension.
    f_sizes : sequence of int
        Widths of F (Linear + ReLU per layer).
    feature_set : str or None
        Recorded for :func:`~itj.models.make_inputs` (the features themselves are not used).

    Attributes
    ----------
    phi, f : torch.nn.Sequential
        The angular filter network and the jet network.
    out : torch.nn.Linear
        The output layer.
    """

    default_name = "efn"

    def __init__(
        self,
        n_classes: int = 3,
        phi_sizes: Sequence[int] = (100, 100, 128),
        f_sizes: Sequence[int] = (100, 100, 100),
        feature_set: str | None = "kin",
    ) -> None:
        super().__init__()
        if feature_set is not None:
            check_feature_set(feature_set, len(FEATURE_SETS.get(feature_set, ())))
        if not phi_sizes:
            raise ValueError("phi_sizes must have at least one layer")
        self.phi = _mlp(2, phi_sizes)
        self.f = _mlp(phi_sizes[-1], f_sizes)
        self.out = nn.Linear(f_sizes[-1] if f_sizes else phi_sizes[-1], n_classes)
        init = {
            "n_classes": int(n_classes),
            "phi_sizes": tuple(int(s) for s in phi_sizes),
            "f_sizes": tuple(int(s) for s in f_sizes),
            "feature_set": feature_set,
        }
        self._set_config(init, feature_set=feature_set, jet_features=None)

    @property
    def head(self) -> nn.Linear:
        """The output layer."""
        return self.out

    @staticmethod
    def momentum_fractions(batch: JetBatch) -> torch.Tensor:
        """Return z_i = pT_i / sum_j pT_j (B, P) from ``batch.vectors``, 0 at padded slots."""
        mask = batch.particle_mask.to(batch.vectors.dtype)
        pt = torch.sqrt(batch.vectors[:, 0].square() + batch.vectors[:, 1].square()) * mask
        return pt / pt.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(pt.dtype).tiny)

    def _phi(self, batch: JetBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = batch.particle_mask
        z = self.momentum_fractions(batch)
        h = self.phi(batch.points.transpose(1, 2)) * mask.unsqueeze(-1).to(batch.points.dtype)
        return h, z, mask

    def embed(self, batch: JetBatch) -> torch.Tensor:
        """Return F of the z-weighted latent sum (B, D)."""
        h, z, _ = self._phi(batch)
        return self.f((z.unsqueeze(-1) * h).sum(dim=1))

    def trace(self, batch: JetBatch, *, attention: bool = False) -> Trace:
        """Return the :class:`Trace`: layer ``"phi"`` (B, P, latent), pooled ``"phi"`` and ``"latent"`` (the sum)."""
        h, z, mask = self._phi(batch)
        latent = (z.unsqueeze(-1) * h).sum(dim=1)
        embedding = self.f(latent)
        return Trace(
            logits=self.out(embedding),
            embedding=embedding,
            mask=mask,
            layers={"phi": h},
            pooled={"phi": masked_mean(h, mask), "latent": latent},
        )

    def filters(self, *, extent: float = 0.8, bins: int = 64) -> np.ndarray:
        """Evaluate the filters Phi_a(Delta eta, Delta phi) on a grid (W3.2.2, ``plot.efn_filters``).

        Parameters
        ----------
        extent : float
            The grid covers [-extent, extent] in Delta eta and Delta phi.
        bins : int
            Grid points per axis (bin centres).

        Returns
        -------
        numpy.ndarray
            (latent, bins, bins) float32; ``[a, i, j]`` is filter a at Delta eta of bin i and Delta phi of bin j.
        """
        edges = np.linspace(-extent, extent, bins + 1)
        centres = 0.5 * (edges[1:] + edges[:-1])
        eta, phi = np.meshgrid(centres, centres, indexing="ij")
        grid = torch.as_tensor(np.stack([eta.ravel(), phi.ravel()], axis=1), dtype=torch.float32)
        param = next(self.phi.parameters())
        with torch.no_grad():
            values = self.phi(grid.to(param.device)).cpu().numpy()
        return np.ascontiguousarray(values.T.reshape(-1, bins, bins).astype(np.float32))
