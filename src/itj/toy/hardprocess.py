"""The hard process: seeds, Breit-Wigner masses, W and top decays, W helicity and containment.

Everything is built in the generation frame (seed along +x at y = 0) for a group of jets of one process. The
random numbers come from the hard stream only, so that every preset and every switch acting after the hard process
sees the same hard events. Containment (every decay quark within the jet radius of the resonance axis and one
anti-kt jet on the decay partons) is decided here by rejection: candidates are drawn in batches of 1.5 times the
missing number until enough are accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import HardPart, SimConfig
from .kinematics import boost, perp_basis, rapidity, to_lab, two_body, wrap_phi

__all__ = ["PROCESSES", "HardEvents", "lab_vectors", "pt_spectrum_cdf", "sample_hard", "truncated_bw_cdf"]

#: The processes of the toy; ``truth_process`` is the index into this tuple.
PROCESSES: tuple[str, ...] = ("qcd", "wqq", "tbqq", "q", "g")

#: Process codes of the resonances.
_QCD, _W, _TOP, _Q, _G = range(5)


@dataclass
class HardEvents:
    """Structure of arrays of the hard events of one process group (generation frame, float64).

    Attributes
    ----------
    code : numpy.ndarray
        ``(n,)`` process code (index into :data:`PROCESSES`).
    pt, y, phi : numpy.ndarray
        ``(n,)`` seed pT, rapidity and azimuth in the lab.
    mass : numpy.ndarray
        ``(n,)`` resonance mass (0 for QCD).
    res_pid : numpy.ndarray
        ``(n,)`` resonance PDG id (+-24, +-6; 0 for QCD).
    init_pid : numpy.ndarray
        ``(n,)`` 1 (quark), 21 (gluon), 24 (W) or 6 (top).
    nprongs : numpy.ndarray
        ``(n,)`` 1, 2 or 3.
    prong_p4 : numpy.ndarray
        ``(n, 3, 4)`` decay partons (the initiator for QCD), zero padded; top: 0 = b (massive), 1-2 = W quarks by
        lab energy; W: by lab energy.
    prong_pid : numpy.ndarray
        ``(n, 3)`` PDG ids of the prongs.
    theta_start : numpy.ndarray
        ``(n, 3)`` shower start angles (colour coherence or ``qcd_start_angle``).
    rest_q : numpy.ndarray
        ``(n, 3, 3)`` momenta of the decay partons in the resonance rest frame (zeros for QCD).
    res_p4 : numpy.ndarray
        ``(n, 4)`` seed or resonance four-vector.
    cos_theta_star : numpy.ndarray
        ``(n,)`` W decay angle (NaN for QCD).
    dr_min, dr_max : numpy.ndarray
        ``(n,)`` smallest and largest pairwise Delta R of the decay quarks (NaN for QCD).
    n_tested, n_contained : int
        Candidates drawn and candidates that passed the containment (for the efficiency).
    """

    code: np.ndarray
    pt: np.ndarray
    y: np.ndarray
    phi: np.ndarray
    mass: np.ndarray
    res_pid: np.ndarray
    init_pid: np.ndarray
    nprongs: np.ndarray
    prong_p4: np.ndarray
    prong_pid: np.ndarray
    theta_start: np.ndarray
    rest_q: np.ndarray
    res_p4: np.ndarray
    cos_theta_star: np.ndarray
    dr_min: np.ndarray
    dr_max: np.ndarray
    n_tested: int = 0
    n_contained: int = 0

    def __len__(self) -> int:
        return len(self.code)

    def select(self, rows: np.ndarray) -> HardEvents:
        """Return the rows ``rows`` (the counters are kept)."""
        values = {name: getattr(self, name)[rows] for name in _ARRAY_FIELDS}
        return HardEvents(**values, n_tested=self.n_tested, n_contained=self.n_contained)

    @staticmethod
    def concat(parts: list[HardEvents]) -> HardEvents:
        """Concatenate row-wise (counters add)."""
        values = {name: np.concatenate([getattr(p, name) for p in parts]) for name in _ARRAY_FIELDS}
        return HardEvents(
            **values,
            n_tested=sum(p.n_tested for p in parts),
            n_contained=sum(p.n_contained for p in parts),
        )


_ARRAY_FIELDS = (
    "code",
    "pt",
    "y",
    "phi",
    "mass",
    "res_pid",
    "init_pid",
    "nprongs",
    "prong_p4",
    "prong_pid",
    "theta_start",
    "rest_q",
    "res_p4",
    "cos_theta_star",
    "dr_min",
    "dr_max",
)


# ----------------------------------------------------------------------------------------------------------------------
# Distributions


def _sample_pt(u: np.ndarray, cfg: HardPart) -> np.ndarray:
    if cfg.pt_spectrum == "fixed":
        return np.full(u.shape, cfg.pt_fixed)
    lo, hi = cfg.pt_min, cfg.pt_max
    if cfg.pt_spectrum == "flat":
        return lo + u * (hi - lo)
    a = 1.0 - cfg.pt_power
    return (lo**a + u * (hi**a - lo**a)) ** (1.0 / a)


def pt_spectrum_cdf(pt: np.ndarray, cfg: HardPart) -> np.ndarray:
    """Cumulative distribution of the seed pT (``pt_spectrum`` power or flat; for the statistical tests)."""
    lo, hi = cfg.pt_min, cfg.pt_max
    x = np.clip(np.asarray(pt, dtype=np.float64), lo, hi)
    if cfg.pt_spectrum == "flat":
        return (x - lo) / (hi - lo)
    a = 1.0 - cfg.pt_power
    return (x**a - lo**a) / (hi**a - lo**a)


def _bw_f(x: np.ndarray | float, m: float, w: float) -> np.ndarray:
    return 0.5 + np.arctan(2.0 * (np.asarray(x, dtype=np.float64) - m) / w) / np.pi


def _sample_bw(u: np.ndarray, m: float, w: float, lo: np.ndarray | float, hi: np.ndarray | float) -> np.ndarray:
    """Breit-Wigner (Cauchy) truncated to [lo, hi] by the inverse CDF (exact, no rejection)."""
    f_lo, f_hi = _bw_f(lo, m, w), _bw_f(hi, m, w)
    f = f_lo + u * (f_hi - f_lo)
    return np.clip(m + 0.5 * w * np.tan(np.pi * (f - 0.5)), lo, hi)


def truncated_bw_cdf(x: np.ndarray, m: float, w: float, lo: float, hi: float) -> np.ndarray:
    """Cumulative distribution of the Breit-Wigner truncated to [lo, hi] (for the statistical tests)."""
    f_lo, f_hi = _bw_f(lo, m, w), _bw_f(hi, m, w)
    return np.clip((_bw_f(x, m, w) - f_lo) / (f_hi - f_lo), 0.0, 1.0)


def _about(axis: np.ndarray, cos_theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Return unit vectors at polar angle ``arccos(cos_theta)`` and azimuth ``phi`` about the unit vectors ``axis``."""
    e1, e2 = perp_basis(axis)
    s = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))
    return cos_theta[:, None] * axis + (s * np.cos(phi))[:, None] * e1 + (s * np.sin(phi))[:, None] * e2


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the opening angle of the 3-momenta of four-vectors ``a`` and ``b`` (``(..., 4)``)."""
    na = np.linalg.norm(a[..., :3], axis=-1)
    nb = np.linalg.norm(b[..., :3], axis=-1)
    c = np.sum(a[..., :3] * b[..., :3], axis=-1) / np.maximum(na * nb, 1e-300)
    return np.arccos(np.clip(c, -1.0, 1.0))


# ----------------------------------------------------------------------------------------------------------------------
# Containment


def _y_phi(p4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return rapidity(p4), np.arctan2(p4[..., 1], p4[..., 0])


def _one_antikt_jet(p4: np.ndarray, radius: float) -> np.ndarray:
    """Return True where inclusive anti-kt (E-scheme) on the partons ``p4`` (``(m, k, 4)``) gives one jet."""
    k = p4.shape[1]
    y, phi = _y_phi(p4)
    if k == 2:
        return np.hypot(y[:, 0] - y[:, 1], wrap_phi(phi[:, 0] - phi[:, 1])) < radius
    pt = np.hypot(p4[..., 0], p4[..., 1])
    inv = 1.0 / np.maximum(pt * pt, 1e-300)
    pairs = ((0, 1), (0, 2), (1, 2))
    d = np.stack(
        [
            np.minimum(inv[:, i], inv[:, j])
            * (np.hypot(y[:, i] - y[:, j], wrap_phi(phi[:, i] - phi[:, j])) / radius) ** 2
            for i, j in pairs
        ],
        axis=1,
    )
    best = np.argmin(d, axis=1)
    first = d[np.arange(len(d)), best] < inv.min(axis=1)
    rows = np.arange(len(p4))
    i_idx = np.array([p[0] for p in pairs])[best]
    j_idx = np.array([p[1] for p in pairs])[best]
    third = 3 - i_idx - j_idx
    merged = p4[rows, i_idx] + p4[rows, j_idx]
    other = p4[rows, third]
    ym, phim = _y_phi(merged)
    yo, phio = _y_phi(other)
    second = np.hypot(ym - yo, wrap_phi(phim - phio)) < radius
    return first & second


def _containment(prongs: np.ndarray, res: np.ndarray, radius: float) -> np.ndarray:
    """Every decay parton within ``radius`` of the resonance axis and one anti-kt jet holding all of them."""
    y, phi = _y_phi(prongs)
    yr, phir = _y_phi(res)
    inside = np.all(np.hypot(y - yr[:, None], wrap_phi(phi - phir[:, None])) < radius, axis=1)
    return inside & _one_antikt_jet(prongs, radius)


def _massless_partons(rest_q: np.ndarray, mass: np.ndarray, res_p4: np.ndarray) -> np.ndarray:
    """Return the decay partons as the ``partons`` preset gives them: massless after the reshuffle.

    Without a shower every system is one massless parton, so the reshuffle (``shower.reshuffle``) re-centres the
    rest-frame momenta ``rest_q`` (``(m, k, 3)``), scales them to ``k q_i`` with ``k sum |q_i| = M`` and boosts them
    with the resonance ``res_p4`` (``(m, 4)``). Only the top changes (its b quark is massive in the decay).
    """
    q = rest_q - rest_q.mean(axis=1, keepdims=True)
    qn = np.linalg.norm(q, axis=2)
    k = mass / np.maximum(qn.sum(axis=1), 1e-300)
    rest = np.concatenate([k[:, None, None] * q, (k[:, None] * qn)[..., None]], axis=2)
    return boost(rest, (res_p4[:, :3] / res_p4[:, 3:4])[:, None, :])


def _pair_dr(prongs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, phi = _y_phi(prongs)
    k = prongs.shape[1]
    drs = [np.hypot(y[:, i] - y[:, j], wrap_phi(phi[:, i] - phi[:, j])) for i in range(k) for j in range(i + 1, k)]
    stacked = np.stack(drs, axis=1)
    return stacked.min(axis=1), stacked.max(axis=1)


# ----------------------------------------------------------------------------------------------------------------------
# Candidates


def _lab_energy(p4: np.ndarray, y: np.ndarray) -> np.ndarray:
    return p4[..., 3] * np.cosh(y)[:, None] + p4[..., 2] * np.sinh(y)[:, None]


def _empty(m: int, code: int) -> dict[str, np.ndarray]:
    return {
        "code": np.full(m, code, dtype=np.int8),
        "mass": np.zeros(m),
        "res_pid": np.zeros(m, dtype=np.int32),
        "init_pid": np.zeros(m, dtype=np.int32),
        "nprongs": np.zeros(m, dtype=np.int8),
        "prong_p4": np.zeros((m, 3, 4)),
        "prong_pid": np.zeros((m, 3), dtype=np.int32),
        "theta_start": np.zeros((m, 3)),
        "rest_q": np.zeros((m, 3, 3)),
        "cos_theta_star": np.full(m, np.nan),
        "dr_min": np.full(m, np.nan),
        "dr_max": np.full(m, np.nan),
    }


def _qcd_candidates(m: int, code: int, cfg: SimConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    h = cfg.hard
    u = rng.random((6, m))
    out = _empty(m, code)
    out["pt"] = _sample_pt(u[0], h)
    out["y"] = (2.0 * u[1] - 1.0) * h.y_max
    out["phi"] = (2.0 * u[2] - 1.0) * np.pi
    weights = np.asarray(h.qcd_flavour_weights, dtype=np.float64)
    cum = np.cumsum(weights / weights.sum())
    flavour = np.minimum(np.searchsorted(cum, u[4], side="right"), 4) + 1
    sign = np.where(u[5] < 0.5, 1, -1)
    if code == _Q:
        quark = np.ones(m, dtype=bool)
    elif code == _G:
        quark = np.zeros(m, dtype=bool)
    else:
        quark = u[3] < h.qcd_quark_fraction
    out["prong_pid"][:, 0] = np.where(quark, sign * flavour, 21)
    out["init_pid"] = np.where(quark, 1, 21).astype(np.int32)
    out["nprongs"][:] = 1
    pt = out["pt"]
    out["prong_p4"][:, 0] = np.stack([pt, np.zeros(m), np.zeros(m), pt], axis=1)
    out["res_p4"] = out["prong_p4"][:, 0].copy()
    out["theta_start"][:, 0] = cfg.shower.qcd_start_angle
    return out


def _w_flavours(u: np.ndarray, sign: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PDG ids (fermion, antifermion) of W+ -> u dbar / c sbar, W- -> d ubar / s cbar."""
    cs = u < 0.5
    up = np.where(cs, 4, 2)
    down = np.where(cs, 3, 1)
    fermion = np.where(sign > 0, up, down)
    antifermion = np.where(sign > 0, -down, -up)
    return fermion, antifermion


def _w_candidates(m: int, cfg: SimConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    h = cfg.hard
    u = rng.random((8, m))
    out = _empty(m, _W)
    pt = _sample_pt(u[0], h)
    out["pt"], out["y"], out["phi"] = pt, (2.0 * u[1] - 1.0) * h.y_max, (2.0 * u[2] - 1.0) * np.pi
    mw = _sample_bw(u[3], h.w_mass, h.w_width, h.w_mass - h.w_window, h.w_mass + h.w_window)
    out["mass"] = mw
    sign = np.where(u[6] < 0.5, 1, -1)
    out["res_pid"] = (24 * sign).astype(np.int32)
    out["init_pid"][:] = 24
    out["nprongs"][:] = 2
    cos = 2.0 * u[4] - 1.0
    out["cos_theta_star"] = cos
    axis = np.tile(np.array([1.0, 0.0, 0.0]), (m, 1))
    d = _about(axis, cos, 2.0 * np.pi * u[5]) * (0.5 * mw)[:, None]
    q_f = np.concatenate([d, (0.5 * mw)[:, None]], axis=1)
    q_a = np.concatenate([-d, (0.5 * mw)[:, None]], axis=1)
    e = np.sqrt(pt * pt + mw * mw)
    beta = np.stack([pt / e, np.zeros(m), np.zeros(m)], axis=1)
    lab_f, lab_a = boost(q_f, beta), boost(q_a, beta)
    fermion, antifermion = _w_flavours(u[7], sign)
    out["res_p4"] = np.stack([pt, np.zeros(m), np.zeros(m), e], axis=1)
    prongs = np.stack([lab_f, lab_a], axis=1)
    rest = np.stack([q_f[:, :3], q_a[:, :3]], axis=1)
    pids = np.stack([fermion, antifermion], axis=1)
    swap = _lab_energy(prongs, out["y"])
    order = np.argsort(-swap, axis=1, kind="stable")
    rows = np.arange(m)[:, None]
    out["prong_p4"][:, :2] = prongs[rows, order]
    out["rest_q"][:, :2] = rest[rows, order]
    out["prong_pid"][:, :2] = pids[rows, order]
    theta_qq = _angle(lab_f, lab_a)
    start = theta_qq if cfg.shower.coherence else np.full(m, cfg.shower.qcd_start_angle)
    out["theta_start"][:, :2] = start[:, None]
    return out


def _helicity_cos(m: int, w_helicity: bool, f0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw the W helicity angle: f(c) = 3/8 (1 + c^2)(1 - F0) + 3/4 (1 - c^2) F0 (rejection, bound 0.75)."""
    if not w_helicity:
        return 2.0 * rng.random(m) - 1.0
    c = np.empty(m)
    todo = np.arange(m)
    while len(todo):
        u = rng.random((2, len(todo)))
        x = 2.0 * u[0] - 1.0
        f = f0[todo]
        density = 0.375 * (1.0 + x * x) * (1.0 - f) + 0.75 * (1.0 - x * x) * f
        ok = u[1] * 0.75 < density
        c[todo[ok]] = x[ok]
        todo = todo[~ok]
    return c


def _top_candidates(m: int, cfg: SimConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    h = cfg.hard
    u = rng.random((10, m))
    out = _empty(m, _TOP)
    pt = _sample_pt(u[0], h)
    out["pt"], out["y"], out["phi"] = pt, (2.0 * u[1] - 1.0) * h.y_max, (2.0 * u[2] - 1.0) * np.pi
    mt = _sample_bw(u[3], h.top_mass, h.top_width, h.top_mass - h.top_window, h.top_mass + h.top_window)
    cap = mt - max(5.0, h.b_mass + 1.0)
    mw = _sample_bw(u[4], h.w_mass, h.w_width, h.w_mass - h.w_window, np.minimum(h.w_mass + h.w_window, cap))
    out["mass"] = mt
    sign = np.where(u[8] < 0.5, 1, -1)
    out["res_pid"] = (6 * sign).astype(np.int32)
    out["init_pid"][:] = 6
    out["nprongs"][:] = 3
    # t -> b W in the top rest frame (polar axis +x, the top flight direction)
    axis_x = np.tile(np.array([1.0, 0.0, 0.0]), (m, 1))
    cos_b = 2.0 * u[5] - 1.0
    pb_dir = _about(axis_x, cos_b, 2.0 * np.pi * u[6])
    zero = np.zeros(m)
    one = np.ones(m)
    p_b, p_w = two_body(mt, h.b_mass, mw, one, zero)  # b along +z and W along -z: rotated to pb_dir below
    pb = np.concatenate([pb_dir * p_b[:, 2:3], p_b[:, 3:]], axis=1)
    pw = np.concatenate([-pb[:, :3], p_w[:, 3:]], axis=1)
    # W -> q q' with the helicity angle about the W direction (in the W rest frame)
    f0 = mt * mt / (mt * mt + 2.0 * mw * mw)
    c = _helicity_cos(m, h.w_helicity, f0, rng)
    w_axis = -pb_dir
    d = _about(w_axis, c, 2.0 * np.pi * u[7]) * (0.5 * mw)[:, None]
    q_f = np.concatenate([d, (0.5 * mw)[:, None]], axis=1)
    q_a = np.concatenate([-d, (0.5 * mw)[:, None]], axis=1)
    beta_w = pw[:, :3] / pw[:, 3:4]
    q_f, q_a = boost(q_f, beta_w), boost(q_a, beta_w)
    out["cos_theta_star"] = c
    e = np.sqrt(pt * pt + mt * mt)
    beta = np.stack([pt / e, zero, zero], axis=1)
    lab_b, lab_f, lab_a = boost(pb, beta), boost(q_f, beta), boost(q_a, beta)
    fermion, antifermion = _w_flavours(u[9], sign)
    out["res_p4"] = np.stack([pt, zero, zero, e], axis=1)
    wq = np.stack([lab_f, lab_a], axis=1)
    wrest = np.stack([q_f[:, :3], q_a[:, :3]], axis=1)
    wpid = np.stack([fermion, antifermion], axis=1)
    order = np.argsort(-_lab_energy(wq, out["y"]), axis=1, kind="stable")
    rows = np.arange(m)[:, None]
    out["prong_p4"][:, 0] = lab_b
    out["prong_p4"][:, 1:] = wq[rows, order]
    out["rest_q"][:, 0] = pb[:, :3]
    out["rest_q"][:, 1:] = wrest[rows, order]
    out["prong_pid"][:, 0] = 5 * sign
    out["prong_pid"][:, 1:] = wpid[rows, order]
    if cfg.shower.coherence:
        out["theta_start"][:, 0] = _angle(lab_b, out["res_p4"])
        out["theta_start"][:, 1:] = _angle(lab_f, lab_a)[:, None]
    else:
        out["theta_start"][:, :] = cfg.shower.qcd_start_angle
    return out


def sample_hard(n: int, code: int, cfg: SimConfig, rng: np.random.Generator) -> HardEvents:
    """Draw ``n`` hard events of the process ``code`` (rejection sampling for containment).

    Parameters
    ----------
    n : int
        Number of events.
    code : int
        Process code (index into :data:`PROCESSES`).
    cfg : SimConfig
        The configuration (hard part; the start angles also use the shower part).
    rng : numpy.random.Generator
        The hard stream.

    Returns
    -------
    HardEvents
    """
    parts: list[HardEvents] = []
    need = n
    tested = contained = 0
    resonance = code in (_W, _TOP)
    first = True
    while need > 0 or first:
        first = False
        m = math.ceil(1.5 * need) if (resonance and cfg.hard.containment) else need
        if code == _W:
            cand = _w_candidates(m, cfg, rng)
        elif code == _TOP:
            cand = _top_candidates(m, cfg, rng)
        else:
            cand = _qcd_candidates(m, code, cfg, rng)
        if resonance:
            k = 2 if code == _W else 3
            prongs = cand["prong_p4"][:, :k]
            cand["dr_min"], cand["dr_max"] = _pair_dr(prongs)
            if cfg.hard.containment:
                ok = _containment(prongs, cand["res_p4"], cfg.jet.jet_radius)
                if code == _TOP:  # the massive b moves in the reshuffle: the partons preset must be contained too
                    partons = _massless_partons(cand["rest_q"][:, :k], cand["mass"], cand["res_p4"])
                    ok &= _containment(partons, cand["res_p4"], cfg.jet.jet_radius)
            else:
                ok = np.ones(m, bool)
        else:
            ok = np.ones(m, dtype=bool)
        tested += m
        contained += int(ok.sum())
        rows = np.flatnonzero(ok)[:need]
        events = HardEvents(**{name: cand[name] for name in _ARRAY_FIELDS})  # type: ignore[arg-type]
        parts.append(events.select(rows))
        need -= len(rows)
        if m == 0:
            break
    out = HardEvents.concat(parts)
    out.n_tested, out.n_contained = tested, contained
    return out


def lab_vectors(hard: HardEvents, p4: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Move generation-frame four-vectors of the events ``event`` of ``hard`` to the lab."""
    return to_lab(p4, hard.y[event], hard.phi[event])
