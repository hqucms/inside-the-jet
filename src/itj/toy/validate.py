"""Validation of the toy: exact identities, leading-log formulas and the validation table of ``jetgen validate``.

* :func:`check_identities` checks the exact identities of generated data (``docs/data_format.md#exact-identities``);
  the tests assert that every entry is True on every preset and the notebook prints it. Augmented jets
  (:mod:`itj.augment`) are exempt.
* :func:`sudakov_cdf`, :func:`lund_ll` and :func:`mass_peak_ll` are the analytic references of W1.2.10, W1.2.4 and
  W1.3.5.
* :func:`validate` generates the validation samples and returns the rows of the table of ``docs/physics.md``
  (:func:`format_validation` prints it). Every two-sided row has a calibrated value (the mean of five seeds with
  ``n = 10 000``, measured when the toy was calibrated against the JetClass validation files) and passes if its
  value lies within ``max(5 sigma, 3 %)`` of it, ``sigma`` being the statistical error of the run, and inside the
  acceptance range of the row (``docs/physics.md#calibration-and-validation``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data import COLUMNS, JETCLASS_LABELS, JetData, Ragged
from .shower import CA, CF, TR

__all__ = [
    "N_LUND_MIN",
    "N_SIGMA_WINDOW",
    "REL_WINDOW",
    "ROWS",
    "ValidationRow",
    "check_identities",
    "format_validation",
    "lund_ll",
    "mass_peak_ll",
    "sudakov_cdf",
    "validate",
]

#: Relative tolerance of the identities on float32 file values.
_REL = 1e-5

#: Absolute tolerance of the ``part_deta`` sign identity.
_ABS_DETA = 1e-5


# ----------------------------------------------------------------------------------------------------------------------
# Analytic references


def lund_ll(cr: float, alphas: float) -> float:
    """Leading-log primary Lund-plane density ``2 alpha_s C_R / pi`` per unit ln(1/Delta) and ln(kt).

    Examples
    --------
    >>> round(lund_ll(4 / 3, 0.12), 3), round(lund_ll(3.0, 0.12), 3)
    (0.102, 0.229)
    """
    return 2.0 * alphas * cr / math.pi


def mass_peak_ll(cr: float, alphas: float, *, hard_collinear: bool = True, pt: float = 600.0, R: float = 0.8) -> float:
    """Jet mass at the peak of ``rho dsigma/d ln rho`` at fixed coupling (leading log, ``rho = m^2 / (pT R)^2``).

    The cumulative ``ln Sigma = -(alpha_s C_R / pi)(L^2 / 2 + B L)``, ``L = ln(1 / rho)``, peaks in ``ln rho`` at
    ``L = sqrt(pi / (alpha_s C_R)) - B`` with the hard-collinear term ``B_q = -3/4`` or
    ``B_g = -(11 C_A - 4 T_R n_f) / (12 C_A)`` (``hard_collinear=False``: B = 0, the double-log value).

    Parameters
    ----------
    cr : float
        Colour factor (4/3 quark, 3 gluon; B is chosen by the colour factor).
    alphas : float
        Fixed coupling.
    hard_collinear : bool
        Include the B term.
    pt, R : float
        Jet pT in GeV and radius.

    Examples
    --------
    >>> [round(mass_peak_ll(c, 0.12, hard_collinear=h)) for c in (4 / 3, 3.0) for h in (False, True)]
    [52, 36, 110, 80]
    """
    b = 0.0
    if hard_collinear:
        b = -(11.0 * CA - 4.0 * TR * 5) / (12.0 * CA) if abs(cr - CA) < abs(cr - CF) else -0.75
    big_l = math.sqrt(math.pi / (alphas * cr)) - b
    return pt * R * math.exp(-0.5 * big_l)


def sudakov_cdf(theta: np.ndarray, *, energy: float, alphas: float, kt_cut: float, theta0: float) -> np.ndarray:
    """Cumulative distribution of the first-emission angle of a quark at fixed coupling.

    ``P(theta_1 <= theta) = exp(-R(theta))`` with ``R(theta) = alpha_s / (2 pi) int_{ln theta^2}^{ln theta0^2} dt
    int P_qq(z) dz`` over the exact phase space ``z (1 - z) E theta' > kt_cut`` (``z`` between the roots
    ``(1 -+ sqrt(1 - 4 kt_cut / (E theta'))) / 2``; the z integral is analytic, the angle integral numerical). A quark
    without any emission (``theta_1 < theta_min = 4 kt_cut / E``) counts as ``theta_1 = 0``: the CDF is
    ``exp(-R(theta_min))`` below ``theta_min`` and 1 above ``theta0``.

    Parameters
    ----------
    theta : numpy.ndarray
        Angles in rad.
    energy : float
        Quark energy in GeV.
    alphas : float
        Fixed coupling.
    kt_cut : float
        Shower cutoff in GeV.
    theta0 : float
        Start angle.

    Returns
    -------
    numpy.ndarray
        The CDF at ``theta``.
    """
    theta = np.asarray(theta, dtype=np.float64)
    theta_min = 4.0 * kt_cut / energy
    if theta0 <= theta_min:
        return np.ones_like(theta)
    t = np.linspace(2.0 * math.log(theta_min), 2.0 * math.log(theta0), 20001)
    x = np.clip(4.0 * kt_cut / (energy * np.exp(0.5 * t)), 0.0, 1.0)
    root = np.sqrt(1.0 - x)
    z_lo, z_hi = 0.5 * (1.0 - root), 0.5 * (1.0 + root)

    def primitive(z: np.ndarray) -> np.ndarray:  # int (1 + z^2) / (1 - z) dz
        return -z - 0.5 * z * z - 2.0 * np.log1p(-z)

    density = alphas / (2.0 * math.pi) * CF * (primitive(z_hi) - primitive(z_lo))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(t))])
    radiator = cum[-1] - cum  # R from t up to ln theta0^2
    t_of = 2.0 * np.log(np.clip(theta, theta_min, theta0))
    out = np.exp(-np.interp(t_of, t, radiator))
    return np.where(theta >= theta0, 1.0, out)


# ----------------------------------------------------------------------------------------------------------------------
# Exact identities


def _p4(data: JetData, prefix: str, energy: str = "energy") -> tuple[np.ndarray, Ragged]:
    cols = [data[f"{prefix}{c}"] for c in ("px", "py", "pz", energy)]
    first = cols[0]
    assert isinstance(first, Ragged)
    return np.stack([np.asarray(c.values, dtype=np.float64) for c in cols], axis=1), first  # type: ignore[union-attr]


def _jet_sum(values: np.ndarray, ragged: Ragged) -> np.ndarray:
    out = np.zeros((len(ragged), values.shape[1]))
    np.add.at(out, ragged.row_index(), values)
    return out


def _resonance_p4(data: JetData) -> np.ndarray:
    pt = np.asarray(data["aux_genpart_pt"], dtype=np.float64)
    eta = np.asarray(data["aux_genpart_eta"], dtype=np.float64)
    phi = np.asarray(data["aux_genpart_phi"], dtype=np.float64)
    m = np.asarray(data["truth_mass"], dtype=np.float64)
    pz = pt * np.sinh(eta)
    e = np.sqrt(pt * pt + pz * pz + m * m)
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), pz, e], axis=1)


def _same_four_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(b[:, 3]), 1e-30)
    return np.all(np.abs(a - b) <= 10 * _REL * scale[:, None], axis=1)


def _switch(config: dict | None, part: str, name: str) -> bool | None:
    if not config:
        return None
    return bool(config.get(part, {}).get(name, False))


def check_identities(data: JetData) -> dict[str, bool]:
    """Check the exact identities of generated toy data (float32 tolerances, relative 1e-5).

    The checks (an entry is missing when its columns are missing, e.g. without truth):

    * ``jet_nparticles``: ``jet_nparticles`` equals the number of constituents;
    * ``pt_sorted``: constituents pT-descending;
    * ``jet_pt``: the length of the summed pT vector equals ``jet_pt``;
    * ``deta_sign``: ``part_deta = s (eta_i - eta_jet)``, s = +1 if ``jet_eta >= 0`` else -1 (abs. 1e-5);
    * ``dphi_range``: ``|part_dphi| <= pi``;
    * ``labels``: exactly one of the 10 labels set per jet, with the JetClass dtypes;
    * ``pid_flags``: the PID flags sum to 1 per hadron (``hadrons``, ``detector``), 0 per parton;
    * ``neutral_ip``: neutral particles have d0 = dz = 0 and errors 0;
    * ``prong_range``: ``truth_part_prong`` in [-1, nprongs - 1], -1 exactly for the underlying event;
    * ``prong_sum``: the prongs sum to the resonance (W, top) or equal the seed (QCD);
    * ``parton_sum``: the final partons sum to the resonance after the reshuffle (W, top); for QCD jets the
      parton energies sum to the seed energy in the generation frame (the seed at y = 0) and
      ``|sum p_perp| / E < 0.05`` in that frame (the tolerance of the spec; the generator rotates every QCD jet
      onto the seed axis, so it is 0 up to rounding);
    * ``hadron_sum``: (whenever the hadron reshuffle ran, also through a per-process override) the hadrons of every
      restored W and top, with all decay products and the neutrinos, sum to the parton-level resonance, relative
      1e-8 in the generation frame. The file holds neither the hadrons outside the jet nor the neutrinos, so this
      reads the generator's own measurement, ``provenance["hadron_reshuffle"]["max_residual"]`` (a merged file of
      different runs has no provenance and skips it);
    * ``containment``: (``partons`` preset) the jet holds every decay parton;
    * ``emission_order``: emission angles decrease along every branch;
    * ``coherence_start``: (with coherence) the first emission of every W quark is below the W opening angle.

    Parameters
    ----------
    data : JetData
        Data generated by :func:`itj.toy.generate` (not augmented).

    Returns
    -------
    dict of str to bool
        Check name to result; raises nothing.
    """
    out: dict[str, bool] = {}
    checks: list[tuple[str, Callable[[JetData], bool]]] = [
        ("jet_nparticles", _check_nparticles),
        ("pt_sorted", _check_sorted),
        ("jet_pt", _check_jet_pt),
        ("deta_sign", _check_deta),
        ("dphi_range", _check_dphi),
        ("labels", _check_labels),
        ("pid_flags", _check_flags),
        ("neutral_ip", _check_neutral_ip),
        ("prong_range", _check_prong_range),
        ("prong_sum", _check_prong_sum),
        ("parton_sum", _check_parton_sum),
        ("hadron_sum", _check_hadron_sum),
        ("containment", _check_containment),
        ("emission_order", _check_emission_order),
        ("coherence_start", _check_coherence),
    ]
    for name, check in checks:
        try:
            out[name] = bool(check(data))
        except _NotApplicable:
            continue
        except (KeyError, ValueError, TypeError, AssertionError, IndexError):
            out[name] = False
    return out


class _NotApplicable(Exception):
    """The check does not apply to these data."""


def _need(data: JetData, *names: str) -> None:
    if any(name not in data for name in names):
        raise _NotApplicable


def _check_nparticles(data: JetData) -> bool:
    _need(data, "jet_nparticles", "part_px")
    return bool(np.array_equal(np.asarray(data["jet_nparticles"]).astype(np.int64), data.counts))


def _pt(data: JetData) -> tuple[np.ndarray, Ragged]:
    _need(data, "part_px", "part_py")
    px, py = data["part_px"], data["part_py"]
    assert isinstance(px, Ragged)
    assert isinstance(py, Ragged)
    return np.hypot(px.values.astype(np.float64), py.values.astype(np.float64)), px


def _check_sorted(data: JetData) -> bool:
    pt, ragged = _pt(data)
    if len(pt) < 2:
        return True
    same = ragged.row_index()[1:] == ragged.row_index()[:-1]
    return bool(np.all(pt[1:][same] <= pt[:-1][same]))


def _check_jet_pt(data: JetData) -> bool:
    _need(data, "part_px", "part_py", "part_pz", "part_energy", "jet_pt")
    values, ragged = _p4(data, "part_")
    total = _jet_sum(values, ragged)
    pt = np.hypot(total[:, 0], total[:, 1])
    ref = np.asarray(data["jet_pt"], dtype=np.float64)
    return bool(np.all(np.abs(pt - ref) <= _REL * np.maximum(ref, 1e-30) + 1e-6))


def _check_deta(data: JetData) -> bool:
    _need(data, "part_px", "part_py", "part_pz", "part_deta", "jet_eta")
    pt, ragged = _pt(data)
    pz = np.asarray(data["part_pz"].values, dtype=np.float64)  # type: ignore[union-attr]
    eta = np.arcsinh(pz / np.maximum(pt, 1e-300))
    jet_eta = np.asarray(data["jet_eta"], dtype=np.float64)[ragged.row_index()]
    sign = np.where(jet_eta >= 0, 1.0, -1.0)
    deta = np.asarray(data["part_deta"].values, dtype=np.float64)  # type: ignore[union-attr]
    return bool(np.all(np.abs(deta - sign * (eta - jet_eta)) <= _ABS_DETA))


def _check_dphi(data: JetData) -> bool:
    _need(data, "part_dphi")
    dphi = data["part_dphi"].values  # type: ignore[union-attr]
    return bool(np.all(np.abs(dphi) <= np.float32(math.pi)))


def _check_labels(data: JetData) -> bool:
    _need(data, *JETCLASS_LABELS)
    total = np.zeros(len(data), dtype=np.int64)
    for name in JETCLASS_LABELS:
        col = np.asarray(data[name])
        if col.dtype != np.dtype(COLUMNS[name].dtype):
            return False
        total += col.astype(np.int64) != 0
    return bool(np.all(total == 1))


_FLAGS = ("part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton", "part_isElectron", "part_isMuon")


def _check_flags(data: JetData) -> bool:
    _need(data, *_FLAGS)
    total = np.stack([np.asarray(data[f].values, dtype=np.int64) for f in _FLAGS]).sum(axis=0)  # type: ignore[union-attr]
    hadronize = _switch(data.config, "hadron", "hadronize")
    if hadronize is None:
        return bool(np.all((total == 0) | (total == 1)))
    if hadronize:
        return bool(np.all(total == 1))
    if "truth_part_is_ue" in data:
        expected = np.asarray(data["truth_part_is_ue"].values, dtype=np.int64)  # type: ignore[union-attr]
        return bool(np.array_equal(total, expected))
    return bool(np.all((total == 0) | (total == 1)))


def _check_neutral_ip(data: JetData) -> bool:
    names = ("part_charge", "part_d0val", "part_d0err", "part_dzval", "part_dzerr")
    _need(data, *names)
    neutral = data["part_charge"].values == 0  # type: ignore[union-attr]
    return all(bool(np.all(data[n].values[neutral] == 0)) for n in names[1:])  # type: ignore[union-attr]


def _check_prong_range(data: JetData) -> bool:
    _need(data, "truth_part_prong", "truth_part_is_ue", "truth_nprongs")
    prong = data["truth_part_prong"]
    assert isinstance(prong, Ragged)
    nprongs = np.asarray(data["truth_nprongs"], dtype=np.int64)[prong.row_index()]
    p = prong.values.astype(np.int64)
    ue = data["truth_part_is_ue"].values.astype(bool)  # type: ignore[union-attr]
    return bool(np.all((p >= -1) & (p < nprongs)) and np.array_equal(p == -1, ue))


def _codes(data: JetData) -> np.ndarray:
    return np.asarray(data["truth_process"], dtype=np.int64)


def _check_prong_sum(data: JetData) -> bool:
    _need(data, "truth_prong_px", "truth_process", "truth_mass", "aux_genpart_pt", "truth_seed_pt")
    values, ragged = _p4(data, "truth_prong_")
    total = _jet_sum(values, ragged)
    res = (_codes(data) == 1) | (_codes(data) == 2)
    ok = _same_four_vector(total[res], _resonance_p4(data)[res])
    seed_pt = np.asarray(data["truth_seed_pt"], dtype=np.float64)[~res]
    pt = np.hypot(total[~res, 0], total[~res, 1])
    return bool(np.all(ok) and np.all(np.abs(pt - seed_pt) <= _REL * seed_pt + 1e-6))


def _check_parton_sum(data: JetData) -> bool:
    _need(data, "truth_parton_px", "truth_process", "truth_mass", "aux_genpart_pt", "truth_seed_pt")
    values, ragged = _p4(data, "truth_parton_")
    total = _jet_sum(values, ragged)
    res = (_codes(data) == 1) | (_codes(data) == 2)
    ok = bool(np.all(_same_four_vector(total[res], _resonance_p4(data)[res])))
    if (~res).any():
        pt = np.asarray(data["truth_seed_pt"], dtype=np.float64)[~res]
        eta = np.asarray(data["truth_seed_eta"], dtype=np.float64)[~res]
        phi = np.asarray(data["truth_seed_phi"], dtype=np.float64)[~res]
        e_seed = pt * np.cosh(eta)
        q = total[~res]
        # the shower conserves the energy in the generation frame (the seed at y = 0): undo the boost along z
        e_gen = np.cosh(eta) * q[:, 3] - np.sinh(eta) * q[:, 2]
        ok &= bool(np.all(np.abs(e_gen - pt) <= 10 * _REL * e_seed))
        # the 3-momentum transverse to the seed in that frame (the seed along (cos phi, sin phi, 0))
        p_gen = np.stack([q[:, 0], q[:, 1], np.cosh(eta) * q[:, 2] - np.sinh(eta) * q[:, 3]], axis=1)
        axis = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
        perp = p_gen - np.sum(p_gen * axis, axis=1, keepdims=True) * axis
        ok &= bool(np.all(np.linalg.norm(perp, axis=1) < 0.05 * pt))
    return ok


def _check_containment(data: JetData) -> bool:
    _need(data, "truth_nprongs", "jet_nparticles")
    config = data.config
    if not config or any(_switch(config, part, name) for part, name in _PARTON_LEVEL):
        raise _NotApplicable
    return bool(np.array_equal(data.counts, np.asarray(data["truth_nprongs"], dtype=np.int64)))


_PARTON_LEVEL = (("shower", "shower"), ("hadron", "hadronize"), ("ue", "ue"))

#: Tolerance of the hadron-level reshuffle identity (relative, float64, generation frame).
_HADRON_SUM_TOL = 1e-8


def _check_hadron_sum(data: JetData) -> bool:
    # the provenance counts the steps of every process, so a per-process override (--override
    # wqq:hadron_reshuffle=true) is covered although the global switch is off
    record = (data.provenance or {}).get("hadron_reshuffle")
    if not isinstance(record, dict) or not record.get("steps"):
        raise _NotApplicable
    return float(record["max_residual"]) <= _HADRON_SUM_TOL


def _check_emission_order(data: JetData) -> bool:
    _need(data, "truth_emis_parent", "truth_emis_daughter1", "truth_emis_daughter2", "truth_emis_theta")
    parent = data["truth_emis_parent"]
    assert isinstance(parent, Ragged)
    if len(parent.values) == 0:
        return True
    jet = parent.row_index().astype(np.int64)
    theta = np.asarray(data["truth_emis_theta"].values, dtype=np.float64)  # type: ignore[union-attr]
    d1 = data["truth_emis_daughter1"].values.astype(np.int64)  # type: ignore[union-attr]
    d2 = data["truth_emis_daughter2"].values.astype(np.int64)  # type: ignore[union-attr]
    keys = np.concatenate([jet * 65536 + d1, jet * 65536 + d2])
    created_at = np.concatenate([theta, theta])
    order = np.argsort(keys)
    keys, created_at = keys[order], created_at[order]
    want = jet * 65536 + parent.values.astype(np.int64)
    pos = np.clip(np.searchsorted(keys, want), 0, len(keys) - 1)
    found = keys[pos] == want
    return bool(np.all(theta[found] <= created_at[pos[found]]))


def _check_coherence(data: JetData) -> bool:
    _need(data, "truth_emis_parent", "truth_emis_depth", "truth_emis_theta", "truth_prong_px", "truth_process")
    if not _switch(data.config, "shower", "coherence"):
        raise _NotApplicable
    codes = _codes(data)
    values, ragged = _p4(data, "truth_prong_")
    offsets = ragged.offsets
    res = np.flatnonzero((codes == 1) | (codes == 2))
    if len(res) == 0:
        return True
    first_w = np.where(codes[res] == 1, 0, 1)  # index of the first W quark among the prongs
    qa = values[offsets[res] + first_w]
    qb = values[offsets[res] + first_w + 1]
    total = _jet_sum(values, ragged)[res]
    y = 0.5 * np.log((total[:, 3] + total[:, 2]) / (total[:, 3] - total[:, 2]))
    theta_qq = _angle_after_z_boost(qa, qb, y)
    theta_qq_jet = np.full(len(data), np.inf)
    theta_qq_jet[res] = theta_qq
    w_first = np.full(len(data), -1)
    w_first[res] = first_w
    parent = data["truth_emis_parent"]
    assert isinstance(parent, Ragged)
    jet = parent.row_index()
    p = parent.values.astype(np.int64)
    is_w = (w_first[jet] >= 0) & (p >= w_first[jet]) & (p <= w_first[jet] + 1)
    theta = np.asarray(data["truth_emis_theta"].values, dtype=np.float64)  # type: ignore[union-attr]
    return bool(np.all(theta[is_w] <= theta_qq_jet[jet[is_w]] * (1.0 + 1e-4)))


def _angle_after_z_boost(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the opening angle of ``a`` and ``b`` after the boost along z that brings rapidity ``y`` to 0."""
    ch, sh = np.cosh(y), np.sinh(y)

    def unboost(p: np.ndarray) -> np.ndarray:
        return np.stack([p[:, 0], p[:, 1], ch * p[:, 2] - sh * p[:, 3]], axis=1)

    ua, ub = unboost(a), unboost(b)
    c = np.sum(ua * ub, axis=1) / (np.linalg.norm(ua, axis=1) * np.linalg.norm(ub, axis=1))
    return np.arccos(np.clip(c, -1.0, 1.0))


# ----------------------------------------------------------------------------------------------------------------------
# The validation table


@dataclass(frozen=True)
class ValidationRow:
    """One row of the validation table.

    Attributes
    ----------
    name : str
        Row name (``jetgen validate --only`` matches its start).
    sample : str
        The sample it is measured on.
    value, error : float
        Measured value and its statistical error (nan if not estimated).
    target : str
        The calibrated value of the toy and the reference (JetClass or leading-log value).
    low, high : float
        The test window: calibrated value +- max(5 error, 3 %) within the acceptance range of the row (the
        acceptance range alone for the one-sided rows).
    ok : bool
        ``low <= value <= high`` (and the extra condition of the row).
    """

    name: str
    sample: str
    value: float
    error: float
    target: str
    low: float
    high: float
    ok: bool


#: The rows: name -> (sample, target, calibrated value, acceptance low, acceptance high). The calibrated values are
#: the means over the seeds 0-4 of ``validate(n=10_000)`` for the calibrated defaults; JetClass values are measured on
#: the first 100 000 jets of ``ZJetsToNuNu_120``, ``WToQQ_120`` and ``TTBar_120``. ``nan`` marks a one-sided row
#: (only the acceptance range applies).
ROWS: dict[str, tuple[str, str, float, float, float]] = {
    "lund_q": ("lund q 600 GeV, fixed 0.12, shower", "0.107 (LL 0.102)", 0.1065, 0.095, 0.125),
    "lund_g": ("lund g 600 GeV, fixed 0.12, shower", "0.218 (LL 0.229)", 0.2183, 0.195, 0.245),
    "lund_ratio": ("lund g / q", "2.05 (LL 2.25)", 2.049, 1.70, 2.40),
    "n_primary_q": ("lund q", "1.58 (LL formula 1.60)", 1.581, 1.45, 1.75),
    "mass_peak_q": ("lund q, partons in R", "44 GeV (LL 36-52)", 43.6, 36.0, 52.0),
    "mass_peak_g": ("lund g, partons in R", "85 GeV (LL 80-110)", 85.2, 80.0, 110.0),
    "ng_nq_shower": ("q, g running, shower", "1.68", 1.681, 1.55, 1.90),
    "ng_nq_hadrons": ("q, g running, hadrons", "1.46", 1.460, 1.35, 1.65),
    "ng_nq_detector": ("q, g running, detector", "1.46", 1.460, 1.30, 1.65),
    "containment_w": ("hard process", "0.934", 0.9340, 0.90, 0.97),
    "containment_top": ("hard process", "0.554", 0.5538, 0.50, 0.72),
    "msd_median_w": ("detector (sel.)", "81.6 GeV (JetClass 77.3)", 81.56, 75.0, 85.0),
    "msd_median_top": ("detector (sel.)", "169.8 GeV (JetClass 164.9)", 169.81, 156.0, 176.0),
    "msd_window_w": ("detector (sel.), m_SD in [65, 95]", "0.78 (JetClass 0.68)", 0.7799, 0.62, 0.84),
    "msd_window_top": ("detector (sel.), m_SD in [140, 200]", "0.82 (JetClass 0.69)", 0.8178, 0.62, 0.88),
    "tau21_qcd": ("detector (sel.), median", "0.610 (JetClass 0.608)", 0.6102, 0.56, 0.68),
    "tau21_w": ("detector (sel.), median", "0.364 (JetClass 0.366)", 0.3641, 0.33, 0.44),
    "tau32_qcd": ("detector (sel.), median", "0.765 (JetClass 0.763)", 0.7647, 0.72, 0.84),
    "tau32_top": ("detector (sel.), median", "0.471 (JetClass 0.501)", 0.4713, 0.43, 0.53),
    "d2_ratio": ("detector (sel.), median D2 (beta 2) QCD / W", "4.7 (JetClass 4.0)", math.nan, 3.0, math.inf),
    "n_qcd": ("detector (sel.), mean multiplicity", "35.8 (JetClass 35.3)", 35.85, 31.0, 40.0),
    "n_w": ("detector (sel.), mean multiplicity", "38.0 (JetClass 32.2)", 38.00, 27.0, 40.0),
    "n_top": ("detector (sel.), mean multiplicity", "53.2 (JetClass 50.9)", 53.21, 44.0, 57.0),
    "frac_charged": ("QCD detector (sel.), by count", "0.487 (JetClass 0.485)", 0.4866, 0.44, 0.52),
    "frac_neutral": ("QCD detector (sel.), by count", "0.107 (JetClass 0.107)", 0.1068, 0.07, 0.15),
    "frac_photon": ("QCD detector (sel.), by count", "0.406 (JetClass 0.402)", 0.4064, 0.36, 0.44),
    "lepton_top": (
        "top detector (sel.), (e + mu) share",
        "0.67 % and > 3 x QCD (JetClass 1.2 %)",
        0.00666,
        0.003,
        0.012,
    ),
    "hadron_cap_fraction": ("hadrons preset", "0.055", math.nan, 0.0, 0.20),
    "speed_detector": ("detector, jets/s on one core", ">= 500 (assert >= 100)", math.nan, 100.0, math.inf),
}

#: A two-sided row passes within this many statistical errors of its calibrated value ...
N_SIGMA_WINDOW = 5.0

#: ... or within this relative distance, whichever is larger.
REL_WINDOW = 0.03

#: Smallest number of jets of the two Lund samples (rows ``lund_*``, ``n_primary_q`` and ``mass_peak_*``, about a second
#: each): the parabola fit of the broad mass peaks is unstable on fewer jets (at 3 000 jets ``mass_peak_q`` can land
#: 4 errors from its calibrated value).
N_LUND_MIN = 10_000


def _window(name: str, error: float) -> tuple[float, float]:
    """Return the test window of a row for a run with this statistical error."""
    _, _, calibrated, low, high = ROWS[name]
    if math.isnan(calibrated):
        return low, high
    width = max(N_SIGMA_WINDOW * error if math.isfinite(error) else 0.0, REL_WINDOW * abs(calibrated))
    return max(low, calibrated - width), min(high, calibrated + width)


def _row(name: str, value: float, error: float, extra_ok: bool = True) -> ValidationRow:
    sample, target, _, _, _ = ROWS[name]
    low, high = _window(name, error)
    ok = bool(np.isfinite(value) and low <= value <= high and extra_ok)
    return ValidationRow(name, sample, float(value), float(error), target, low, high, ok)


def _selected(names: Sequence[str] | None) -> list[str]:
    if not names:
        return list(ROWS)
    out = [row for row in ROWS if any(row.startswith(prefix) for prefix in names)]
    if not out:
        import difflib

        close = difflib.get_close_matches(names[0], list(ROWS), n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(f"no validation row matches {list(names)}.{hint}")
    return out


def _ratio(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ma, mb = float(np.mean(a)), float(np.mean(b))
    ea, eb = float(np.std(a)) / math.sqrt(len(a)), float(np.std(b)) / math.sqrt(len(b))
    r = ma / mb
    return r, r * math.hypot(ea / ma, eb / mb)


def _lund_box(data: JetData) -> tuple[float, float]:
    x = data["truth_lund_ln_inv_delta"].values  # type: ignore[union-attr]
    y = data["truth_lund_ln_kt"].values  # type: ignore[union-attr]
    count = int(np.sum((x >= 0.5) & (x <= 2.5) & (y >= 0.5) & (y <= 2.0)))
    area = 2.0 * 1.5
    return count / (len(data) * area), math.sqrt(max(count, 1)) / (len(data) * area)


def _mass_in_cone(data: JetData, R: float = 0.8) -> np.ndarray:
    values, ragged = _p4(data, "part_")
    row = ragged.row_index()
    pt = np.hypot(values[:, 0], values[:, 1])
    eta = np.arcsinh(values[:, 2] / np.maximum(pt, 1e-300))
    phi = np.arctan2(values[:, 1], values[:, 0])
    seed_eta = np.asarray(data["truth_seed_eta"], dtype=np.float64)[row]
    seed_phi = np.asarray(data["truth_seed_phi"], dtype=np.float64)[row]
    dphi = np.mod(phi - seed_phi + math.pi, 2 * math.pi) - math.pi
    inside = np.hypot(eta - seed_eta, dphi) < R
    total = _jet_sum(values * inside[:, None], ragged)
    return np.sqrt(np.maximum(total[:, 3] ** 2 - np.sum(total[:, :3] ** 2, axis=1), 0.0))


def _mass_peak(data: JetData, R: float = 0.8, *, half: float = 1.2, width: float = 0.1) -> tuple[float, float]:
    """Return the jet mass at the peak of ``rho dsigma/d ln rho`` and its statistical error.

    A parabola is fitted to the histogram of ``ln rho`` (bins of ``width``) within ``half`` of the peak (three
    iterations from the maximum of the smoothed histogram); the error comes from the fit covariance.
    """
    m = _mass_in_cone(data, R)
    pt = np.asarray(data["truth_seed_pt"], dtype=np.float64)
    m, pt = m[m > 0], pt[m > 0]
    log_rho = np.log(m * m / (pt * pt * R * R))
    edges = np.arange(-16.0, width / 2, width)
    hist, _ = np.histogram(log_rho, edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    x0 = float(centres[np.argmax(np.convolve(hist, np.ones(9) / 9.0, mode="same"))])
    error = math.nan
    for _ in range(3):
        sel = np.abs(centres - x0) <= half
        weights = 1.0 / np.sqrt(np.maximum(hist[sel], 1))
        (a, b, _), cov = np.polyfit(centres[sel] - x0, hist[sel], 2, w=weights, cov="unscaled")
        if a >= 0:  # no maximum inside the window
            return math.nan, math.nan
        shift = float(np.clip(-b / (2 * a), -half, half))
        gradient = np.array([b / (2 * a * a), -1 / (2 * a), 0.0])
        error = float(np.sqrt(max(gradient @ cov @ gradient, 0.0)))
        x0 += shift
    mass = float(np.median(pt)) * R * math.exp(0.5 * x0)
    return mass, 0.5 * mass * error


def validate(
    *, n: int = 10_000, seed: int = 0, workers: int | str = 1, only: Sequence[str] | None = None
) -> list[ValidationRow]:
    """Generate the validation samples and measure the rows of the validation table.

    Parameters
    ----------
    n : int
        Jets per sample (the detector sample has ``n`` jets of qcd, wqq and tbqq together); the two Lund samples
        (shower only) have at least :data:`N_LUND_MIN`.
    seed : int
        Seed of every sample.
    workers : int or "auto"
        Worker processes of the generation.
    only : sequence of str, optional
        Row names or name prefixes (e.g. ``["lund", "msd"]``); default all rows.

    Returns
    -------
    list of ValidationRow
        In the order of :data:`ROWS`.
    """
    from ..features import jetclass_selection
    from ..observables import d2
    from .config import preset
    from .generate import generate
    from .hardprocess import sample_hard

    wanted = _selected(only)
    values: dict[str, ValidationRow] = {}

    def need(*prefixes: str) -> bool:
        return any(name.startswith(prefixes) for name in wanted)

    gen: dict[str, Any] = {"seed": seed, "workers": workers}
    if need("lund", "n_primary", "mass_peak"):
        lund_cfg = preset("shower", alphas_mode="fixed", alphas_value=0.12, pt_spectrum="fixed", pt_fixed=600.0)
        n_lund = max(n, N_LUND_MIN)
        q = generate(n_lund, config=lund_cfg, processes=("q",), **gen)
        g = generate(n_lund, config=lund_cfg, processes=("g",), **gen)
        (lq, eq), (lg, eg) = _lund_box(q), _lund_box(g)
        values["lund_q"] = _row("lund_q", lq, eq)
        values["lund_g"] = _row("lund_g", lg, eg)
        values["lund_ratio"] = _row("lund_ratio", lg / lq, lg / lq * math.hypot(eq / lq, eg / lg))
        n_prim = np.asarray(q["truth_n_primary"], dtype=np.float64)
        values["n_primary_q"] = _row("n_primary_q", float(n_prim.mean()), float(n_prim.std()) / math.sqrt(len(q)))
        values["mass_peak_q"] = _row("mass_peak_q", *_mass_peak(q))
        values["mass_peak_g"] = _row("mass_peak_g", *_mass_peak(g))
    cap = math.nan
    for level in ("shower", "hadrons", "detector"):
        if need(f"ng_nq_{level}") or (level == "hadrons" and need("hadron_cap")):
            qg = generate(2 * n, config=preset(level), processes=("q", "g"), **gen)
            names = qg.process_names
            mult = qg.counts.astype(np.float64)
            r, e = _ratio(mult[names == "g"], mult[names == "q"])
            values[f"ng_nq_{level}"] = _row(f"ng_nq_{level}", r, e)
            if level == "hadrons":
                cap = float(qg.provenance["hadron_kt_capped"]) if qg.provenance else math.nan
    if need("hadron_cap"):
        values["hadron_cap_fraction"] = _row("hadron_cap_fraction", cap, math.nan)
    if need("containment"):
        rng = np.random.default_rng(seed)
        for name, code in (("containment_w", 1), ("containment_top", 2)):
            hard = sample_hard(n, code, preset("partons"), rng)
            eff = hard.n_contained / hard.n_tested
            values[name] = _row(name, eff, math.sqrt(eff * (1 - eff) / hard.n_tested))
    detector_rows = ("msd", "tau", "d2", "n_qcd", "n_w", "n_top", "frac", "lepton", "speed")
    if need(*detector_rows):
        start = time.perf_counter()
        data = generate(n, **gen)
        wall = time.perf_counter() - start
        if need("speed"):
            n_speed = n
            if (int(data.provenance["workers"]) if data.provenance else 1) > 1:
                # jets per second on one core: time one worker (several share the memory bandwidth and caches)
                n_speed = min(n, 2000)
                start = time.perf_counter()
                generate(n_speed, seed=seed, workers=1)
                wall = time.perf_counter() - start
            values["speed_detector"] = _row("speed_detector", n_speed / wall, math.nan)
        data = data.select(jetclass_selection(data))
        names = data.process_names
        msd = np.asarray(data["jet_sdmass"], dtype=np.float64)
        tau = {k: np.asarray(data[f"jet_tau{k}"], dtype=np.float64) for k in (1, 2, 3)}
        tau21 = tau[2] / np.maximum(tau[1], 1e-30)
        tau32 = tau[3] / np.maximum(tau[2], 1e-30)
        w, top, qcd = names == "wqq", names == "tbqq", names == "qcd"

        def median(x: np.ndarray) -> tuple[float, float]:
            return float(np.median(x)), 1.2533 * float(np.std(x)) / math.sqrt(max(len(x), 1))

        def share(x: np.ndarray) -> tuple[float, float]:
            p = float(np.mean(x))
            return p, math.sqrt(max(p * (1 - p), 1e-12) / max(len(x), 1))

        values["msd_median_w"] = _row("msd_median_w", *median(msd[w]))
        values["msd_median_top"] = _row("msd_median_top", *median(msd[top]))
        values["msd_window_w"] = _row("msd_window_w", *share((msd[w] > 65) & (msd[w] < 95)))
        values["msd_window_top"] = _row("msd_window_top", *share((msd[top] > 140) & (msd[top] < 200)))
        values["tau21_qcd"] = _row("tau21_qcd", *median(tau21[qcd]))
        values["tau21_w"] = _row("tau21_w", *median(tau21[w]))
        values["tau32_qcd"] = _row("tau32_qcd", *median(tau32[qcd]))
        values["tau32_top"] = _row("tau32_top", *median(tau32[top]))
        if need("d2"):
            parts = {}
            for label, rows in (("qcd", qcd), ("w", w)):
                sub = data.select(rows)
                p4, mask = sub.padded_p4()
                parts[label] = float(np.median(d2(p4, mask, beta=2.0)))
            values["d2_ratio"] = _row("d2_ratio", parts["qcd"] / parts["w"], math.nan)
        mult = data.counts.astype(np.float64)
        for name, rows in (("n_qcd", qcd), ("n_w", w), ("n_top", top)):
            values[name] = _row(name, float(mult[rows].mean()), float(mult[rows].std()) / math.sqrt(rows.sum()))
        particle_rows = np.repeat(np.arange(len(data)), data.counts)

        def flag_share(flag: str, rows: np.ndarray) -> tuple[float, float]:
            sel = rows[particle_rows]
            return share(np.asarray(data[flag].values[sel], dtype=np.float64))  # type: ignore[union-attr]

        values["frac_charged"] = _row("frac_charged", *flag_share("part_isChargedHadron", qcd))
        values["frac_neutral"] = _row("frac_neutral", *flag_share("part_isNeutralHadron", qcd))
        values["frac_photon"] = _row("frac_photon", *flag_share("part_isPhoton", qcd))

        def lepton_share(rows: np.ndarray) -> tuple[float, float]:
            sel = rows[particle_rows]
            e = data["part_isElectron"].values[sel]  # type: ignore[union-attr]
            mu = data["part_isMuon"].values[sel]  # type: ignore[union-attr]
            return share((e + mu).astype(np.float64))

        lt, lt_err = lepton_share(top)
        lq, _ = lepton_share(qcd)
        values["lepton_top"] = _row("lepton_top", lt, lt_err, extra_ok=lt > 3 * lq)
    return [values[name] for name in wanted if name in values]


def format_validation(rows: Sequence[ValidationRow]) -> str:
    """Format the validation table as text (``jetgen validate``)."""
    lines = [f"{'name':<20} {'value':>10} {'error':>9}  {'window':<19} {'ok':<4} target", ""]
    for r in rows:
        high = "inf" if math.isinf(r.high) else f"{r.high:.4g}"
        err = "" if math.isnan(r.error) else f"{r.error:.3g}"
        lines.append(
            f"{r.name:<20} {r.value:>10.4g} {err:>9}  {f'[{r.low:.4g}, {high}]':<19} {'yes' if r.ok else 'NO':<4} "
            f"{r.target}"
        )
    passed = sum(r.ok for r in rows)
    lines += ["", f"{passed} of {len(rows)} rows inside their test windows"]
    return "\n".join(lines)
