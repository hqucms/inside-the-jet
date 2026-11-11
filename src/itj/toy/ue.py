"""The underlying event: soft hadrons spread uniformly in (y, phi) over a disc about the seed axis.

Per event ``N ~ Poisson(ue_density pi ue_radius^2)`` hadrons, uniform in the disc of radius ``ue_radius`` about the
seed axis (generation frame: y = 0, phi = 0), with ``pT = ue_pt_min + Exp(ue_pt_mean)`` and the species of the
hadronisation (pi0 decay to two photons). ``ue_density`` is a calibration knob of the toy, not a literature value
(the order of magnitude follows the ATLAS measurement 1701.05390; ``docs/physics.md#underlying-event``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .hadronize import MASSES, PHOTON, PI0, HadronArrays, decay_pi0, draw_species, species_probabilities

if TYPE_CHECKING:
    from .config import SimConfig

__all__ = ["underlying_event"]


def underlying_event(n_jets: int, cfg: SimConfig, rng: np.random.Generator) -> HadronArrays:
    """Draw the underlying-event hadrons of ``n_jets`` events (generation frame).

    Parameters
    ----------
    n_jets : int
        Number of events of the group.
    cfg : SimConfig
        The ``ue`` part sets the density, disc and spectrum; the ``hadron`` part the species.
    rng : numpy.random.Generator
        The ue stream.

    Returns
    -------
    HadronArrays
        ``is_ue`` True, prong and parton -1, production vertex at the origin; ``index`` counts the hadrons of
        each event in production order (pi0 photons in place of their pi0).
    """
    u_cfg = cfg.ue
    counts = rng.poisson(u_cfg.ue_density * math.pi * u_cfg.ue_radius**2, n_jets)
    total = int(counts.sum())
    event = np.repeat(np.arange(n_jets, dtype=np.int64), counts)
    u = rng.random((5, total))
    r = u_cfg.ue_radius * np.sqrt(u[0])
    angle = 2.0 * math.pi * u[1]
    dy, dphi = r * np.cos(angle), r * np.sin(angle)
    pt = u_cfg.ue_pt_min - u_cfg.ue_pt_mean * np.log1p(-u[2])
    species = draw_species(u[3], species_probabilities(cfg.hadron))
    m = MASSES[species]
    mt = np.sqrt(pt * pt + m * m)
    p4 = np.stack([pt * np.cos(dphi), pt * np.sin(dphi), mt * np.sinh(dy), mt * np.cosh(dy)], axis=1)
    charge = np.where(species <= 1, np.where(u[4] < 0.5, 1, -1), 0).astype(np.int8)
    order_key = 2 * np.arange(total, dtype=np.int64)
    pi0 = np.flatnonzero(species == PI0)
    if len(pi0):
        g1, g2 = decay_pi0(p4[pi0], rng.random((2, len(pi0))))
        keep = species != PI0
        p4 = np.concatenate([p4[keep], g1, g2])
        event = np.concatenate([event[keep], event[pi0], event[pi0]])
        charge = np.concatenate([charge[keep], np.zeros(2 * len(pi0), np.int8)])
        species = np.concatenate([species[keep], np.full(2 * len(pi0), PHOTON)])
        order_key = np.concatenate([order_key[keep], order_key[pi0], order_key[pi0] + 1])
    order = np.argsort(order_key, kind="stable")
    event = event[order]
    h = len(order)
    return HadronArrays(
        p4=p4[order],
        vertex=np.zeros((h, 4)),
        species=species[order].astype(np.int8),
        charge=charge[order],
        event=event,
        index=np.arange(h, dtype=np.int64) - np.searchsorted(event, event, side="left"),
        prong=np.full(h, -1, dtype=np.int64),
        parton=np.full(h, -1, dtype=np.int64),
        from_b=np.zeros(h, dtype=bool),
        is_ue=np.ones(h, dtype=bool),
    )
