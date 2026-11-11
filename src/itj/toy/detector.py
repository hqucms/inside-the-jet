"""The detector: tracks with impact parameters, ECAL, HCAL and forward cells, and particle flow (lab frame).

A cartoon of the Delphes CMS card behind JetClass (``docs/physics.md#detector``):

* **Tracks.** Charged hadrons, electrons and muons with pT above ``track_pt_min`` and |eta| <= 2.5 are tracked
  with the efficiency of the (|eta|, pT) table (``lepton_eff`` for e and mu); the pT is smeared with
  ``sigma/pT = a (+) b pT`` (direction kept) and the impact parameters are the projections of the production vertex
  plus ``N(0, sigma^2)`` with ``sigma = sqrt(a^2 + (b / pT)^2)`` (stored as the errors).
* **Calorimeters.** Photons and untracked electrons go to ECAL cells, neutral hadrons and untracked charged hadrons
  to HCAL cells; untracked muons are lost. Beyond |eta| = 2.5 a forward calorimeter with the HCAL cell size and
  the forward resolution takes both. Cell energies are smeared log-normally (mean E, width sigma(E)); a cell is
  kept if its measured energy passes the threshold and the significance. Each kept cell becomes one massless photon
  (ECAL) or neutral hadron (HCAL) at a uniform point inside the cell (or at its centre) and inherits the truth of
  its largest contributor.
* **Particle flow** is perfect for tracked particles (no calorimeter deposit); neutrals have d0 = dz = 0.

All randomness is counter based (:func:`itj.toy.kinematics.keyed_hash`): a hadron's variates are keyed by
(seed, event id, origin, index), a cell's by (seed, event id, calorimeter, i_eta, i_phi), so removing the
underlying event or switching to cell centres leaves every jet-origin track bit-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from .hadronize import ELECTRON, K_CH, K_L, MUON, NEUTRON, PHOTON, PI_CH, HadronArrays
from .kinematics import hash_normal, hash_uniform, keyed_hash

if TYPE_CHECKING:
    from .config import DetectorPart

__all__ = ["CALORIMETERS", "ParticleArrays", "cell_phi_bins", "detector", "impact_parameters", "track_efficiency"]

#: Calorimeter codes of the cell keys.
CALORIMETERS: tuple[str, ...] = ("ECAL", "HCAL", "forward EM", "forward hadronic")
_ECAL, _HCAL, _FWD_EM, _FWD_HAD = range(4)

#: Origins of the keys of the detector randomness.
_ORIGIN_JET, _ORIGIN_UE, _ORIGIN_CELL = 0, 1, 2

#: Purposes of the variates of a hadron and of a cell.
_EFFICIENCY, _TRACK_SMEAR, _D0, _DZ = 0, 1, 2, 3
_CELL_SMEAR, _CELL_ETA, _CELL_PHI = 0, 2, 3

#: Tracker and barrel-calorimeter acceptance.
_TRACKER_ETA, _BARREL_ETA = 2.5, 1.5


@dataclass
class ParticleArrays:
    """The final particles of a group of events (lab frame), as stored in ``part_*`` and ``truth_part_*``.

    Attributes
    ----------
    p4 : numpy.ndarray
        ``(k, 4)`` four-vectors.
    event : numpy.ndarray
        ``(k,)`` event row.
    charge : numpy.ndarray
        ``(k,)`` -1, 0, 1.
    flags : numpy.ndarray
        ``(k, 5)`` int32 one-hot PID: charged hadron, neutral hadron, photon, electron, muon (all 0 for partons).
    d0, d0err, dz, dzerr : numpy.ndarray
        ``(k,)`` impact parameters in mm (0 for neutrals).
    prong, parton : numpy.ndarray
        ``(k,)`` truth of the particle (or of the largest contributor of a cell); -1 for the underlying event.
    is_ue, from_b : numpy.ndarray
        ``(k,)`` bool.
    purity : numpy.ndarray
        ``(k,)`` energy share of the largest contributor (1 for everything but calorimeter cells).
    """

    p4: np.ndarray
    event: np.ndarray
    charge: np.ndarray
    flags: np.ndarray
    d0: np.ndarray
    d0err: np.ndarray
    dz: np.ndarray
    dzerr: np.ndarray
    prong: np.ndarray
    parton: np.ndarray
    is_ue: np.ndarray
    from_b: np.ndarray
    purity: np.ndarray

    def __len__(self) -> int:
        return len(self.event)

    def select(self, rows: np.ndarray) -> ParticleArrays:
        """Return the rows ``rows``."""
        return ParticleArrays(**{f.name: getattr(self, f.name)[rows] for f in fields(self)})

    @staticmethod
    def concat(parts: list[ParticleArrays]) -> ParticleArrays:
        """Concatenate row-wise."""
        return ParticleArrays(
            **{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(ParticleArrays)}
        )

    @staticmethod
    def from_hadrons(hadrons: HadronArrays) -> ParticleArrays:
        """Every hadron (or parton) is a particle: no detector; true impact parameters, errors 0."""
        species = hadrons.species
        d0, dz = impact_parameters(hadrons.vertex, hadrons.p4)
        neutral = hadrons.charge == 0
        k = len(hadrons)
        return ParticleArrays(
            p4=hadrons.p4,
            event=hadrons.event,
            charge=hadrons.charge.astype(np.float64),
            flags=_flags(species),
            d0=np.where(neutral, 0.0, d0),
            d0err=np.zeros(k),
            dz=np.where(neutral, 0.0, dz),
            dzerr=np.zeros(k),
            prong=hadrons.prong,
            parton=hadrons.parton,
            is_ue=hadrons.is_ue,
            from_b=hadrons.from_b,
            purity=np.ones(k),
        )


def _flags(species: np.ndarray) -> np.ndarray:
    flags = np.zeros((len(species), 5), dtype=np.int32)
    flags[:, 0] = (species == PI_CH) | (species == K_CH)
    flags[:, 1] = (species == K_L) | (species == NEUTRON)
    flags[:, 2] = species == PHOTON
    flags[:, 3] = species == ELECTRON
    flags[:, 4] = species == MUON
    return flags


def impact_parameters(vertex: np.ndarray, p4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transverse and longitudinal impact parameters of straight tracks from ``vertex`` (``(k, 4)``, mm).

    ``d0 = (x py - y px) / pT``, ``dz = z - (x px + y py) pz / pT^2``.
    """
    x, y, z = vertex[:, 0], vertex[:, 1], vertex[:, 2]
    px, py, pz = p4[:, 0], p4[:, 1], p4[:, 2]
    pt2 = np.maximum(px * px + py * py, 1e-300)
    d0 = (x * py - y * px) / np.sqrt(pt2)
    dz = z - (x * px + y * py) * pz / pt2
    return d0, dz


def track_efficiency(pt: np.ndarray, eta: np.ndarray, lepton: np.ndarray, cfg: DetectorPart) -> np.ndarray:
    """Tracking efficiency of the (|eta|, pT) table of the Delphes CMS card (``lepton_eff`` for e and mu)."""
    abs_eta = np.abs(eta)
    barrel = np.where(pt > 1.0, cfg.track_eff_high, cfg.track_eff_low)
    endcap = np.where(pt > 1.0, cfg.track_eff_fwd_high, cfg.track_eff_fwd_low)
    eff = np.where(abs_eta <= _BARREL_ETA, barrel, endcap)
    eff = np.where(lepton, cfg.lepton_eff, eff)
    return np.where((pt <= cfg.track_pt_min) | (abs_eta > _TRACKER_ETA), 0.0, eff)


def cell_phi_bins(cell: float) -> int:
    """Return the number of azimuthal cells of size about ``cell``, a multiple of 4 (ECAL 360, HCAL 72)."""
    return 4 * max(1, round(2.0 * math.pi / (4.0 * cell)))


def _resolution(cal: np.ndarray, energy: np.ndarray, eta: np.ndarray, cfg: DetectorPart) -> np.ndarray:
    """sigma(E) of the cells: ECAL with the barrel/endcap factor f(eta), HCAL, forward."""
    abs_eta = np.abs(eta)
    f = np.where(abs_eta <= _BARREL_ETA, 1.0 + 0.64 * eta * eta, 2.16 + 5.6 * (abs_eta - 2.0) ** 2)
    ecal = f * np.sqrt((cfg.ecal_res_const * energy) ** 2 + cfg.ecal_res_stoch**2 * energy + cfg.ecal_res_noise**2)
    hcal = np.sqrt((cfg.hcal_res_const * energy) ** 2 + cfg.hcal_res_stoch**2 * energy)
    fwd = np.sqrt((cfg.fwd_res_const * energy) ** 2 + cfg.fwd_res_stoch**2 * energy)
    return np.where(cal == _ECAL, ecal, np.where(cal == _HCAL, hcal, fwd))


def detector(hadrons: HadronArrays, cfg: DetectorPart, key: tuple[int, np.ndarray]) -> ParticleArrays:
    """Simulate the detector response to the hadrons of a group of events (lab frame).

    Parameters
    ----------
    hadrons : HadronArrays
        Stable particles in the lab (jet-origin and underlying-event hadrons).
    cfg : DetectorPart
        Detector parameters.
    key : tuple of int and numpy.ndarray
        ``(seed, event_ids)``: the run seed and the ``truth_event_id`` of every event row (the keys of the
        counter-based randomness).

    Returns
    -------
    ParticleArrays
        Tracks, then calorimeter particles, sorted by event (stable).
    """
    seed, event_ids = key
    event_ids = np.asarray(event_ids, dtype=np.int64)
    p4, species = hadrons.p4, hadrons.species
    pt = np.hypot(p4[:, 0], p4[:, 1])
    eta = np.arcsinh(p4[:, 2] / np.maximum(pt, 1e-300))
    phi = np.arctan2(p4[:, 1], p4[:, 0])
    lepton = (species == ELECTRON) | (species == MUON)
    charged_hadron = (species == PI_CH) | (species == K_CH)
    charged = charged_hadron | lepton
    origin = np.where(hadrons.is_ue, _ORIGIN_UE, _ORIGIN_JET)
    h = keyed_hash(seed, event_ids[hadrons.event], origin, hadrons.index)
    # every variate is drawn for every hadron (keyed: the draws do not depend on the other hadrons)
    u_eff = hash_uniform(h, _EFFICIENCY)
    n_pt, n_d0, n_dz = hash_normal(h, _TRACK_SMEAR), hash_normal(h, _D0), hash_normal(h, _DZ)
    tracked = charged & (u_eff < track_efficiency(pt, eta, lepton, cfg))
    # tracks
    t = np.flatnonzero(tracked)
    sigma_rel = np.hypot(cfg.track_res_const, cfg.track_res_slope * pt[t])
    scale = np.maximum(1.0 + sigma_rel * n_pt[t], 1e-3)
    m2 = np.maximum(p4[t, 3] ** 2 - np.sum(p4[t, :3] ** 2, axis=1), 0.0)
    p3 = p4[t, :3] * scale[:, None]
    track_p4 = np.concatenate([p3, np.sqrt(np.sum(p3 * p3, axis=1) + m2)[:, None]], axis=1)
    d0_true, dz_true = impact_parameters(hadrons.vertex[t], track_p4)
    sigma_ip = np.hypot(cfg.ip_res_const, cfg.ip_res_pt / (pt[t] * scale))
    k = len(t)
    tracks = ParticleArrays(
        p4=track_p4,
        event=hadrons.event[t],
        charge=hadrons.charge[t].astype(np.float64),
        flags=_flags(species[t]),
        d0=d0_true + sigma_ip * n_d0[t],
        d0err=sigma_ip,
        dz=dz_true + sigma_ip * n_dz[t],
        dzerr=sigma_ip,
        prong=hadrons.prong[t],
        parton=hadrons.parton[t],
        is_ue=hadrons.is_ue[t],
        from_b=hadrons.from_b[t],
        purity=np.ones(k),
    )
    # calorimeters
    em = (species == PHOTON) | ((species == ELECTRON) & ~tracked)
    had = (species == K_L) | (species == NEUTRON) | (charged_hadron & ~tracked)
    forward = np.abs(eta) > _TRACKER_ETA
    cal = np.full(len(species), -1, dtype=np.int64)
    cal[em] = np.where(forward[em], _FWD_EM, _ECAL)
    cal[had] = np.where(forward[had], _FWD_HAD, _HCAL)
    cells = _calorimeter(hadrons, cal, eta, phi, cfg, seed, event_ids)
    out = ParticleArrays.concat([tracks, cells])
    return out.select(np.argsort(out.event, kind="stable"))


def _calorimeter(
    hadrons: HadronArrays,
    cal: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    cfg: DetectorPart,
    seed: int,
    event_ids: np.ndarray,
) -> ParticleArrays:
    """Bin the deposits ``cal >= 0`` into cells, smear, apply thresholds and turn the cells into particles."""
    d = np.flatnonzero(cal >= 0)
    c = cal[d]
    size = np.where(c == _ECAL, cfg.ecal_cell, cfg.hcal_cell)
    n_phi = np.where(c == _ECAL, cell_phi_bins(cfg.ecal_cell), cell_phi_bins(cfg.hcal_cell))
    i_eta = np.floor(eta[d] / size).astype(np.int64)
    i_phi = np.clip(np.floor((phi[d] + math.pi) / (2.0 * math.pi) * n_phi).astype(np.int64), 0, n_phi - 1)
    event = hadrons.event[d]
    packed = (event << 32) | (c << 28) | ((i_eta + 32768) << 12) | i_phi
    cells, inverse = np.unique(packed, return_inverse=True)
    inverse = inverse.reshape(-1)
    energy = hadrons.p4[d, 3]
    e_true = np.bincount(inverse, weights=energy, minlength=len(cells))
    # the largest contributor of every cell
    order = np.lexsort((-energy, inverse))
    first = order[np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1]] if len(order) else order
    lead = d[first]
    purity = energy[first] / np.maximum(e_true, 1e-300)
    c_event = cells >> 32
    c_cal = (cells >> 28) & 0xF
    c_ieta = ((cells >> 12) & 0xFFFF) - 32768
    c_iphi = cells & 0xFFF
    c_size = np.where(c_cal == _ECAL, cfg.ecal_cell, cfg.hcal_cell)
    c_nphi = np.where(c_cal == _ECAL, cell_phi_bins(cfg.ecal_cell), cell_phi_bins(cfg.hcal_cell))
    hc = keyed_hash(seed, event_ids[c_event], _ORIGIN_CELL, c_cal, c_ieta, c_iphi)
    n_smear = hash_normal(hc, _CELL_SMEAR)
    u_eta, u_phi = hash_uniform(hc, _CELL_ETA), hash_uniform(hc, _CELL_PHI)
    centre_eta = (c_ieta + 0.5) * c_size
    sigma = _resolution(c_cal, e_true, centre_eta, cfg)
    s2 = np.log1p((sigma / np.maximum(e_true, 1e-300)) ** 2)
    e_meas = e_true * np.exp(np.sqrt(s2) * n_smear - 0.5 * s2)  # log-normal with mean E and width sigma(E)
    sigma_meas = _resolution(c_cal, e_meas, centre_eta, cfg)
    ecal = c_cal == _ECAL
    threshold = np.where(ecal, cfg.ecal_threshold, cfg.hcal_threshold)
    signif = np.where(ecal, cfg.ecal_signif, cfg.hcal_signif)
    keep = (e_meas > threshold) & (e_meas > signif * sigma_meas)
    if cfg.cell_position == "centre":
        u_eta, u_phi = np.full_like(u_eta, 0.5), np.full_like(u_phi, 0.5)
    cell_eta = (c_ieta + u_eta) * c_size
    cell_phi = -math.pi + (c_iphi + u_phi) * (2.0 * math.pi / c_nphi)
    k = np.flatnonzero(keep)
    e, ce, cp = e_meas[k], cell_eta[k], cell_phi[k]
    pt = e / np.cosh(ce)
    p4 = np.stack([pt * np.cos(cp), pt * np.sin(cp), pt * np.sinh(ce), e], axis=1)
    flags = np.zeros((len(k), 5), dtype=np.int32)
    em = (c_cal[k] == _ECAL) | (c_cal[k] == _FWD_EM)
    flags[:, 2] = em
    flags[:, 1] = ~em
    zeros = np.zeros(len(k))
    return ParticleArrays(
        p4=p4,
        event=c_event[k],
        charge=zeros,
        flags=flags,
        d0=zeros,
        d0err=zeros,
        dz=zeros,
        dzerr=zeros,
        prong=hadrons.prong[lead[k]],
        parton=hadrons.parton[lead[k]],
        is_ue=hadrons.is_ue[lead[k]],
        from_b=hadrons.from_b[lead[k]],
        purity=purity[k],
    )
