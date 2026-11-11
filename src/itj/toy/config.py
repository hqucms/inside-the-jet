"""Configuration of the jet toy: hard process, shower, hadronisation, underlying event, detector and jet definition.

A configuration is a frozen :class:`SimConfig` made of six parts, each a frozen dataclass whose fields are the
options of the simulation (every physical constant and detector parameter of the toy is a field here, with its
default, its unit and a help text that the command line uses):

* :class:`HardPart`: the seed pT spectrum, the QCD flavours, the W and top masses and decays, containment;
* :class:`ShowerPart`: the angular-ordered parton shower (coupling, cutoff, start angles, splittings);
* :class:`HadronPart`: independent fragmentation, the hadron species and the B hadrons;
* :class:`UEPart`: the underlying event;
* :class:`DetectorPart`: tracker, calorimeters and particle flow;
* :class:`JetPart`: the jet algorithm and the parameters of the stored substructure branches.

Parameter names are unique across the parts, so a modified copy is made with flat keywords::

    from itj.toy import preset

    config = preset("detector", ue=False, alphas_mode="fixed", alphas_value=0.12)
    config = config.replace(cell_position="centre")

Every default equals the ``"detector"`` preset (``SimConfig() == PRESETS["detector"]``).
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "PRESETS",
    "DetectorPart",
    "HadronPart",
    "HardPart",
    "JetPart",
    "ShowerPart",
    "SimConfig",
    "UEPart",
    "config_fields",
    "describe_presets",
    "matching_preset",
    "preset",
]


def _param(
    default: Any,
    help: str,
    unit: str = "",
    *,
    choices: tuple[str, ...] | None = None,
    integer: bool = False,
    optional: bool = False,
    allow_inf: bool = False,
    nargs: int | None = None,
    metavar: str | tuple[str, ...] | None = None,
) -> Any:
    """Dataclass field with a help text and a unit (used by ``describe()`` and the command line).

    Extra metadata: ``choices`` (allowed strings), ``integer`` (the value is an integer), ``optional`` (``None`` is a
    valid value), ``allow_inf`` (``inf`` is a valid value), ``nargs`` (length of a tuple value) and ``metavar``
    (names of the values on the command line).
    """
    meta: dict[str, Any] = {"help": help, "unit": unit}
    if choices is not None:
        meta["choices"] = choices
    if integer:
        meta["integer"] = True
    if optional:
        meta["optional"] = True
    if allow_inf:
        meta["allow_inf"] = True
    if nargs is not None:
        meta["nargs"] = nargs
    if metavar is not None:
        meta["metavar"] = metavar
    return field(default=default, metadata=meta)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str, *, integer: bool, allow_inf: bool) -> float | int:
    """Convert one value to a Python float or int (NumPy scalars included) and check that it is finite."""
    if isinstance(value, (bool, str, np.bool_)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, got {value!r}") from None
    if math.isnan(number) or (math.isinf(number) and not allow_inf):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if integer:
        if math.isinf(number) or number != int(number):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        return int(number)
    return number


def _coerce(f: dataclasses.Field[Any], value: Any, owner: str) -> Any:
    """Convert ``value`` to the type of the field (also turns NumPy scalars and lists into Python values)."""
    name = f"{owner}.{f.name}"
    meta = f.metadata
    if value is None:
        if meta.get("optional"):
            return None
        raise TypeError(f"{name} must not be None")
    if "choices" in meta:
        if value not in meta["choices"]:
            close = difflib.get_close_matches(str(value), meta["choices"], n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"{name} must be one of {', '.join(map(repr, meta['choices']))}, got {value!r}.{hint}")
        return str(value)
    if isinstance(f.default, bool):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be True or False, got {value!r}")
        return bool(value)
    integer = bool(meta.get("integer")) or (isinstance(f.default, int) and not isinstance(f.default, bool))
    allow_inf = bool(meta.get("allow_inf"))
    nargs = meta.get("nargs")
    if nargs:
        if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != nargs:
            raise ValueError(f"{name} must be a sequence of {nargs} numbers, got {value!r}")
        return tuple(_number(v, name, integer=integer, allow_inf=allow_inf) for v in value)
    return _number(value, name, integer=integer, allow_inf=allow_inf)


class _Part:
    """Behaviour shared by the configuration parts: type coercion and validation on construction."""

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            object.__setattr__(self, f.name, _coerce(f, getattr(self, f.name), type(self).__name__))
        self._validate()

    def _validate(self) -> None:
        """Check the parameter values (each part overrides this)."""

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary of plain Python values (tuples become lists; JSON-serialisable)."""
        return {
            f.name: list(getattr(self, f.name)) if isinstance(getattr(self, f.name), tuple) else getattr(self, f.name)
            for f in dataclasses.fields(self)  # type: ignore[arg-type]
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Any:
        """Inverse of :meth:`to_dict`; missing parameters take their default values."""
        names = [f.name for f in dataclasses.fields(cls)]  # type: ignore[arg-type]
        extra = sorted(set(data) - set(names))
        if extra:
            close = difflib.get_close_matches(extra[0], names, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"unknown parameters of {cls.__name__}: {extra}.{hint}")
        return cls(**data)


# ----------------------------------------------------------------------------------------------------------------------
# The parts


@dataclass(frozen=True)
class HardPart(_Part):
    """The hard process: the seed parton or resonance and its decay (``docs/physics.md#hard-process``).

    The seed pT follows ``dN/dpT ~ pT^-pt_power`` on [``pt_min``, ``pt_max``] (``pt_spectrum = "power"``), is
    uniform there (``"flat"``) or equals ``pt_fixed`` (``"fixed"``); the rapidity is uniform in
    [-``y_max``, ``y_max``]. QCD jets start from a quark with probability ``qcd_quark_fraction`` (flavours u, d, s,
    c, b with the weights ``qcd_flavour_weights``), else from a gluon. W and top masses follow Breit-Wigner
    distributions truncated to +-``w_window`` and +-``top_window``; the W from a top decay has the helicity fractions
    of the Standard Model (``w_helicity``). With ``containment`` every decay quark lies within the jet radius of the
    resonance and anti-kt on the decay partons gives one jet (rejection sampling).
    """

    pt_spectrum: str = _param(
        "power",
        "seed pT spectrum: 'power' (dN/dpT ~ pT^-pt_power), 'flat' or 'fixed' (pt_fixed)",
        choices=("power", "flat", "fixed"),
    )
    pt_power: float = _param(4.5, "power n of the seed pT spectrum dN/dpT ~ pT^-n")
    pt_min: float = _param(450.0, "lower edge of the seed pT range (truth level)", "GeV")
    pt_max: float = _param(1100.0, "upper edge of the seed pT range (truth level)", "GeV")
    pt_fixed: float = _param(600.0, "seed pT when pt_spectrum is 'fixed'", "GeV")
    y_max: float = _param(2.0, "seeds are uniform in the rapidity range |y| < y_max")
    qcd_quark_fraction: float = _param(0.75, "fraction f_q of quark-initiated jets in the process qcd")
    qcd_flavour_weights: tuple[float, float, float, float, float] = _param(
        (0.50, 0.30, 0.08, 0.08, 0.04),
        "relative weights of the quark flavours u, d, s, c, b of QCD quark jets",
        nargs=5,
        metavar=("U", "D", "S", "C", "B"),
    )
    w_mass: float = _param(80.4, "W mass m_W", "GeV")
    w_width: float = _param(2.085, "W width Gamma_W", "GeV")
    w_window: float = _param(15.0, "the W mass is truncated to m_W +- w_window", "GeV")
    top_mass: float = _param(172.5, "top mass m_t", "GeV")
    top_width: float = _param(1.42, "top width Gamma_t", "GeV")
    top_window: float = _param(25.0, "the top mass is truncated to m_t +- top_window", "GeV")
    b_mass: float = _param(4.8, "b-quark mass in the top-decay kinematics (massless in the shower)", "GeV")
    w_helicity: bool = _param(True, "W helicity fractions in top decays (off: isotropic W decay)")
    containment: bool = _param(True, "require every decay quark inside the jet (rejection sampling)")

    def _validate(self) -> None:
        _require(self.pt_power > 1, f"pt_power must be > 1, got {self.pt_power}")
        _require(0 < self.pt_min < self.pt_max, f"need 0 < pt_min < pt_max, got {self.pt_min}, {self.pt_max}")
        _require(self.pt_fixed > 0, f"pt_fixed must be positive, got {self.pt_fixed}")
        _require(0 <= self.y_max <= 5, f"y_max must be in [0, 5], got {self.y_max}")
        _require(
            0 <= self.qcd_quark_fraction <= 1, f"qcd_quark_fraction must be in [0, 1], got {self.qcd_quark_fraction}"
        )
        weights = self.qcd_flavour_weights
        _require(all(w >= 0 for w in weights) and sum(weights) > 0, f"bad qcd_flavour_weights {weights}")
        for name in ("w_mass", "w_width", "w_window", "top_mass", "top_width", "top_window"):
            _require(getattr(self, name) > 0, f"{name} must be positive, got {getattr(self, name)}")
        _require(self.w_window < self.w_mass, f"w_window must be below w_mass, got {self.w_window}")
        _require(0 <= self.b_mass < 20, f"b_mass must be in [0, 20), got {self.b_mass}")
        _require(
            self.top_mass - self.top_window > self.w_mass - self.w_window + self.b_mass + 5,
            "the top mass window must lie above m_W - w_window + b_mass + 5 GeV (the W of the top decay needs room)",
        )


@dataclass(frozen=True)
class ShowerPart(_Part):
    """The angular-ordered parton shower (``docs/physics.md#parton-shower``).

    Branchings ``dP = alpha_s(kt) / (2 pi) dtheta^2 / theta^2 P(z) dz`` with ``kt = z (1 - z) E theta > kt_cut``,
    sampled with the veto algorithm in ``ln theta^2``. ``alphas_mode = "running"``: one-loop coupling with n_f = 5
    and ``alphas_value`` = alpha_s(M_Z); ``"fixed"``: ``alphas_value`` everywhere. Start angles come from colour
    coherence (``coherence``) or are ``qcd_start_angle`` for every parton.
    """

    shower: bool = _param(True, "parton shower (off: the decay partons are the final partons)")
    alphas_mode: str = _param(
        "running",
        "strong coupling: 'running' (one loop, alpha_s(M_Z) = alphas_value) or 'fixed'",
        choices=("running", "fixed"),
    )
    alphas_value: float = _param(0.118, "alpha_s(M_Z) (running) or the fixed coupling (fixed)")
    kt_cut: float = _param(1.0, "shower cutoff on kt = z (1 - z) E theta", "GeV")
    qcd_start_angle: float = _param(1.0, "start angle of QCD jets (and of every parton without coherence)", "rad")
    coherence: bool = _param(True, "colour coherence: decay partons start at the angle to their colour partner")
    g_to_qq: bool = _param(True, "g -> q qbar splittings")
    n_flavours: int = _param(5, "number of quark flavours n_f of g -> q qbar")
    dead_cone: bool = _param(False, "dead cone of the b quarks from the hard process (theta < m_b / E vetoed)")

    def _validate(self) -> None:
        _require(0 < self.alphas_value < 1, f"alphas_value must be in (0, 1), got {self.alphas_value}")
        _require(self.kt_cut >= 0.2, f"kt_cut must be >= 0.2 GeV (the running coupling diverges), got {self.kt_cut}")
        if self.alphas_mode == "running":
            # one loop, n_f = 5 (as itj.toy.shower.alphas): alpha_s(kt_cut) <= 1 needs kt_cut >= Lambda exp(1 / (2 b0))
            b0 = (33.0 - 2.0 * 5) / (12.0 * math.pi)
            mz = 91.1876
            lam = mz * math.exp(-1.0 / (2.0 * b0 * self.alphas_value))
            largest = 1.0 / (2.0 * b0 * math.log(mz / (self.kt_cut * math.exp(-1.0 / (2.0 * b0)))))
            _require(
                self.kt_cut >= lam * math.exp(1.0 / (2.0 * b0)),
                f"with the running coupling, alphas_value = {self.alphas_value:g} gives Lambda = {lam:.3g} GeV and "
                f"alpha_s(kt_cut = {self.kt_cut:g} GeV) > 1 (or no value at all): the shower would not end; use "
                f"alphas_value <= {largest:.3f} for this kt_cut, or alphas_mode='fixed'",
            )
        _require(0 < self.qcd_start_angle <= 3, f"qcd_start_angle must be in (0, 3], got {self.qcd_start_angle}")
        _require(1 <= self.n_flavours <= 5, f"n_flavours must be in [1, 5], got {self.n_flavours}")


@dataclass(frozen=True)
class HadronPart(_Part):
    """Independent (Field-Feynman) fragmentation and the B hadrons (``docs/physics.md#hadronisation``).

    Every final parton starts a chain: per rank the species is drawn first (pi0 with ``pi0_fraction``, a neutral
    hadron with ``neutral_fraction``, else a charged hadron), then ``z`` from ``f(z) = (1 - a) + 3 a (1 - z)^2``;
    the hadron takes ``max(z E_rem, e_min_hadron, m + 0.05 GeV)``; the chain stops when ``E_rem < 2 e_stop``.
    Transverse momenta ``k ~ N(0, kt_sigma^2)`` per break and component are compensated within the chain. With
    ``singlet_frame`` the quarks of every W fragment in the W rest frame; with ``hadron_reshuffle`` the hadrons of
    every W and top are then moved back to the parton-level four-momentum of the resonance (both off by default,
    ``docs/physics.md#hadron-level-reshuffle``).
    """

    hadronize: bool = _param(True, "hadronisation (off: the partons are the particles)")
    singlet_frame: bool = _param(
        False,
        "fragment the quarks of every W (also in a top) in the W rest frame (experimental, not calibrated: a third "
        "fewer particles in W jets; see physics.md#hadron-level-reshuffle)",
    )
    hadron_reshuffle: bool = _param(
        False,
        "restore the four-momentum of W and top after hadronisation (experimental, not calibrated: W m_SD 66 GeV, "
        "about one W in five cannot be restored; see physics.md#hadron-level-reshuffle)",
    )
    ff_a: float = _param(0.77, "Field-Feynman parameter a of f(z) = (1 - a) + 3 a (1 - z)^2")
    kt_sigma: float = _param(0.24, "Gaussian width of the transverse momentum per break and component", "GeV")
    e_stop: float = _param(1.5, "a chain stops when the remaining energy is below 2 e_stop", "GeV")
    e_min_hadron: float = _param(0.6, "minimum energy of a hadron in a chain", "GeV")
    pi0_fraction: float = _param(0.34, "probability that a hadron is a pi0 (decays to two photons)")
    neutral_fraction: float = _param(0.13, "probability that a hadron is a neutral hadron (K_L or n)")
    kaon_fraction: float = _param(0.15, "share of K+- among the charged hadrons (else pi+-)")
    klong_fraction: float = _param(0.60, "share of K_L among the neutral hadrons (else n)")
    b_hadrons: bool = _param(True, "B hadrons from the b quarks of the hard process, with displaced decays")
    b_ctau: float = _param(0.46, "c tau of the B hadrons", "mm")
    b_frag_mean: float = _param(0.7, "mean energy fraction <z_B> of the B hadron, f(z) ~ z^alpha (1 - z)")
    b_decay_multiplicity: int = _param(7, "number of B-decay products (massless, RAMBO)")
    b_charged_share: float = _param(0.7, "probability that a B-decay product is a charged hadron (else a photon)")
    semileptonic: float = _param(0.11, "probability of a semileptonic B decay, per lepton flavour (e, mu)")
    semileptonic_cascade: float = _param(0.10, "probability of a cascade lepton (b -> c -> l), per lepton flavour")

    def _validate(self) -> None:
        _require(0 <= self.ff_a <= 1, f"ff_a must be in [0, 1], got {self.ff_a}")
        _require(self.kt_sigma >= 0, f"kt_sigma must be >= 0, got {self.kt_sigma}")
        _require(self.e_min_hadron >= 0.2, f"e_min_hadron must be >= 0.2 GeV, got {self.e_min_hadron}")
        _require(
            2 * self.e_stop >= self.e_min_hadron + 1.0,
            f"need 2 e_stop >= e_min_hadron + 1 GeV (room for two hadrons), got e_stop = {self.e_stop}",
        )
        for name in (
            "pi0_fraction",
            "neutral_fraction",
            "kaon_fraction",
            "klong_fraction",
            "b_charged_share",
            "semileptonic",
            "semileptonic_cascade",
        ):
            value = getattr(self, name)
            _require(0 <= value <= 1, f"{name} must be in [0, 1], got {value}")
        _require(
            self.pi0_fraction + self.neutral_fraction <= 1,
            f"pi0_fraction + neutral_fraction must be <= 1, got {self.pi0_fraction + self.neutral_fraction}",
        )
        _require(self.b_ctau >= 0, f"b_ctau must be >= 0, got {self.b_ctau}")
        _require(0.1 <= self.b_frag_mean < 0.95, f"b_frag_mean must be in [0.1, 0.95), got {self.b_frag_mean}")
        _require(
            2 <= self.b_decay_multiplicity <= 20,
            f"b_decay_multiplicity must be in [2, 20], got {self.b_decay_multiplicity}",
        )


@dataclass(frozen=True)
class UEPart(_Part):
    """The underlying event (``docs/physics.md#underlying-event``).

    ``Poisson(ue_density pi ue_radius^2)`` hadrons uniform in (y, phi) in a disc of radius ``ue_radius`` about the
    seed axis, with ``pT = ue_pt_min + Exp(ue_pt_mean)`` and the species of the hadronisation. ``ue_density`` is a
    calibration knob, not a literature value.
    """

    ue: bool = _param(True, "underlying event")
    ue_density: float = _param(0.7, "mean number of underlying-event hadrons per unit (y, phi) area")
    ue_radius: float = _param(1.0, "radius of the disc about the seed axis that holds the underlying event")
    ue_pt_min: float = _param(0.3, "minimum pT of an underlying-event hadron", "GeV")
    ue_pt_mean: float = _param(0.7, "mean of the exponential pT above ue_pt_min", "GeV")

    def _validate(self) -> None:
        _require(0 <= self.ue_density <= 100, f"ue_density must be in [0, 100], got {self.ue_density}")
        _require(0 < self.ue_radius <= 3, f"ue_radius must be in (0, 3], got {self.ue_radius}")
        _require(self.ue_pt_min > 0, f"ue_pt_min must be positive, got {self.ue_pt_min}")
        _require(self.ue_pt_mean >= 0, f"ue_pt_mean must be >= 0, got {self.ue_pt_mean}")


@dataclass(frozen=True)
class DetectorPart(_Part):
    """Tracker, calorimeters and particle flow (``docs/physics.md#detector``; values of the Delphes CMS card).

    Charged particles become tracks with the efficiency of the (|eta|, pT) table and a smeared pT; untracked charged
    hadrons and neutral hadrons go to the HCAL, photons and untracked electrons to the ECAL, everything beyond
    |eta| = 2.5 to a forward calorimeter. Cell energies are smeared log-normally; every kept cell becomes one massless
    photon (ECAL) or neutral hadron (HCAL) at a random point inside the cell (``cell_position = "random"``) or at its
    centre (``"centre"``).
    """

    detector: bool = _param(True, "detector simulation (off: the hadrons are the particles)")
    track_pt_min: float = _param(0.1, "tracks need pT above this threshold", "GeV")
    track_eff_low: float = _param(0.70, "track efficiency for |eta| <= 1.5 and pT <= 1 GeV")
    track_eff_high: float = _param(0.95, "track efficiency for |eta| <= 1.5 and pT > 1 GeV")
    track_eff_fwd_low: float = _param(0.60, "track efficiency for 1.5 < |eta| <= 2.5 and pT <= 1 GeV")
    track_eff_fwd_high: float = _param(0.85, "track efficiency for 1.5 < |eta| <= 2.5 and pT > 1 GeV")
    lepton_eff: float = _param(0.95, "track efficiency of electrons and muons")
    track_res_const: float = _param(0.01, "track resolution sigma/pT = a (+) b pT: constant a")
    track_res_slope: float = _param(2e-4, "track resolution sigma/pT = a (+) b pT: slope b", "1/GeV")
    ip_res_const: float = _param(0.015, "impact-parameter resolution sqrt(a^2 + (b / pT)^2): a", "mm")
    ip_res_pt: float = _param(0.1, "impact-parameter resolution sqrt(a^2 + (b / pT)^2): b", "mm GeV")
    ecal_cell: float = _param(0.0174, "ECAL cell size in eta (and about in phi)")
    ecal_threshold: float = _param(0.5, "minimum measured energy of an ECAL cell", "GeV")
    ecal_signif: float = _param(2.0, "minimum significance E / sigma(E) of an ECAL cell")
    ecal_res_const: float = _param(0.008, "ECAL resolution f(eta) sqrt((a E)^2 + b^2 E + c^2): a")
    ecal_res_stoch: float = _param(0.11, "ECAL resolution: stochastic term b", "sqrt(GeV)")
    ecal_res_noise: float = _param(0.40, "ECAL resolution: noise term c", "GeV")
    hcal_cell: float = _param(0.087, "HCAL (and forward calorimeter) cell size in eta (and about in phi)")
    hcal_threshold: float = _param(1.0, "minimum measured energy of an HCAL or forward cell", "GeV")
    hcal_signif: float = _param(1.0, "minimum significance E / sigma(E) of an HCAL or forward cell")
    hcal_res_const: float = _param(0.05, "HCAL resolution sqrt((a E)^2 + b^2 E): a")
    hcal_res_stoch: float = _param(1.5, "HCAL resolution: stochastic term b", "sqrt(GeV)")
    fwd_res_const: float = _param(0.107, "forward-calorimeter (|eta| > 2.5) resolution sqrt((a E)^2 + b^2 E): a")
    fwd_res_stoch: float = _param(2.08, "forward-calorimeter resolution: stochastic term b", "sqrt(GeV)")
    cell_position: str = _param(
        "random",
        "position of a calorimeter particle: 'random' (uniform in the cell) or 'centre' (a learnable grid)",
        choices=("random", "centre"),
    )

    def _validate(self) -> None:
        for name in ("track_eff_low", "track_eff_high", "track_eff_fwd_low", "track_eff_fwd_high", "lepton_eff"):
            value = getattr(self, name)
            _require(0 <= value <= 1, f"{name} must be in [0, 1], got {value}")
        for name in (
            "track_pt_min",
            "track_res_const",
            "track_res_slope",
            "ip_res_const",
            "ip_res_pt",
            "ecal_threshold",
            "ecal_signif",
            "ecal_res_const",
            "ecal_res_stoch",
            "ecal_res_noise",
            "hcal_threshold",
            "hcal_signif",
            "hcal_res_const",
            "hcal_res_stoch",
            "fwd_res_const",
            "fwd_res_stoch",
        ):
            _require(getattr(self, name) >= 0, f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("ecal_cell", "hcal_cell"):
            _require(0.001 <= getattr(self, name) <= 0.5, f"{name} must be in [0.001, 0.5], got {getattr(self, name)}")


@dataclass(frozen=True)
class JetPart(_Part):
    """The jet and its stored substructure branches (``docs/physics.md#building-the-jet``).

    The leading anti-kt jet of radius ``jet_radius`` of all particles (``jet_def = "antikt"``) or the particles
    within ``jet_radius`` of the seed axis (``"cone"``). ``jet_pt_min``/``jet_pt_max`` drop jets outside a reco pT
    window (off by default: a cut breaks the pairing of samples; use ``itj.features.jetclass_selection``).
    """

    jet_def: str = _param(
        "antikt", "jet definition: 'antikt' (leading jet) or 'cone' (about the seed axis)", choices=("antikt", "cone")
    )
    jet_radius: float = _param(0.8, "jet radius R")
    jet_pt_min: float | None = _param(
        None, "drop jets with jet_pt below this value (breaks the pairing)", "GeV", optional=True, metavar="PT"
    )
    jet_pt_max: float | None = _param(
        None, "drop jets with jet_pt above this value (breaks the pairing)", "GeV", optional=True, metavar="PT"
    )
    sd_zcut: float = _param(0.1, "soft drop z_cut of jet_sdmass")
    sd_beta: float = _param(0.0, "soft drop beta of jet_sdmass")
    tau_beta: float = _param(1.0, "N-subjettiness beta of jet_tau1..4")
    tau_r0: float = _param(0.8, "N-subjettiness R0 of jet_tau1..4 (also the soft drop R0)")
    tau_axes: str = _param(
        "wta_kt", "N-subjettiness axes: 'wta_kt', 'kt' or 'onepass_kt'", choices=("wta_kt", "kt", "onepass_kt")
    )

    def _validate(self) -> None:
        _require(0.1 <= self.jet_radius <= 1.5, f"jet_radius must be in [0.1, 1.5], got {self.jet_radius}")
        if self.jet_pt_min is not None and self.jet_pt_max is not None:
            _require(self.jet_pt_min < self.jet_pt_max, "jet_pt_min must be below jet_pt_max")
        _require(0 <= self.sd_zcut < 0.5, f"sd_zcut must be in [0, 0.5), got {self.sd_zcut}")
        _require(self.sd_beta >= 0, f"sd_beta must be >= 0, got {self.sd_beta}")
        _require(self.tau_beta > 0, f"tau_beta must be positive, got {self.tau_beta}")
        _require(self.tau_r0 > 0, f"tau_r0 must be positive, got {self.tau_r0}")


# ----------------------------------------------------------------------------------------------------------------------
# The full configuration


@dataclass(frozen=True)
class SimConfig:
    """Complete configuration of a simulation (everything except the seed, the processes and the run options).

    Parameters
    ----------
    hard, shower, hadron, ue, detector, jet
        The six parts; each defaults to the ``"detector"`` preset.

    Examples
    --------
    >>> config = SimConfig().replace(ue=False, kt_sigma=0.3)
    >>> config.ue.ue, config.hadron.kt_sigma
    (False, 0.3)
    """

    hard: HardPart = field(default_factory=HardPart)
    shower: ShowerPart = field(default_factory=ShowerPart)
    hadron: HadronPart = field(default_factory=HadronPart)
    ue: UEPart = field(default_factory=UEPart)
    detector: DetectorPart = field(default_factory=DetectorPart)
    jet: JetPart = field(default_factory=JetPart)

    SECTIONS: ClassVar[tuple[str, ...]] = ("hard", "shower", "hadron", "ue", "detector", "jet")

    def __post_init__(self) -> None:
        for name, cls in zip(self.SECTIONS, _PART_CLASSES, strict=True):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"SimConfig.{name} must be a {cls.__name__}, got {getattr(self, name)!r}")

    def replace(self, **overrides: Any) -> SimConfig:
        """Return a copy with some parameters changed.

        Parameters
        ----------
        **overrides
            Flat parameter names of any part (e.g. ``ue=False``, ``kt_sigma=0.3``, ``w_mass=60``) or a whole part
            (e.g. ``detector=DetectorPart(cell_position="centre")``).

        Returns
        -------
        SimConfig
            The modified copy (the original is unchanged).
        """
        parts: dict[str, Any] = {name: getattr(self, name) for name in self.SECTIONS}
        changes: dict[str, dict[str, Any]] = {name: {} for name in self.SECTIONS}
        for key, value in overrides.items():
            part_cls = _PART_CLASSES[self.SECTIONS.index(key)] if key in self.SECTIONS else None
            if part_cls is not None and isinstance(value, part_cls):
                parts[key] = value
            elif key in _PARAMETER_PART:
                changes[_PARAMETER_PART[key]][key] = value
            else:
                close = difflib.get_close_matches(key, list(_PARAMETER_PART), n=3)
                hint = f" Did you mean {' or '.join(map(repr, close))}?" if close else ""
                raise TypeError(f"unknown configuration parameter {key!r}.{hint}")
        for name, kwargs in changes.items():
            if kwargs:
                parts[name] = dataclasses.replace(parts[name], **kwargs)
        return SimConfig(**parts)

    def to_dict(self) -> dict[str, Any]:
        """Nested dictionary of plain Python values (JSON-serialisable)."""
        return {name: getattr(self, name).to_dict() for name in self.SECTIONS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimConfig:
        """Inverse of :meth:`to_dict`; missing parameters take their default values."""
        unknown = sorted(set(data) - set(cls.SECTIONS))
        if unknown:
            raise ValueError(f"unknown configuration sections: {unknown}")
        parts = {
            name: part_cls.from_dict(data.get(name) or {})
            for name, part_cls in zip(cls.SECTIONS, _PART_CLASSES, strict=True)
        }
        return cls(**parts)

    def to_json(self, indent: int | None = None) -> str:
        """JSON string of :meth:`to_dict` (stored in every data file)."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> SimConfig:
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        """Human-readable description of the configuration (used by ``jetgen presets`` and the docs)."""
        h, s, a, u, d, j = self.hard, self.shower, self.hadron, self.ue, self.detector, self.jet
        if h.pt_spectrum == "power":
            spectrum = f"dN/dpT ~ pT^-{h.pt_power:g} on [{h.pt_min:g}, {h.pt_max:g}] GeV"
        elif h.pt_spectrum == "flat":
            spectrum = f"flat on [{h.pt_min:g}, {h.pt_max:g}] GeV"
        else:
            spectrum = f"fixed {h.pt_fixed:g} GeV"
        flavours = " ".join(f"{q} {w:g}" for q, w in zip("udscb", h.qcd_flavour_weights, strict=True))
        coupling = (
            f"running (alpha_s(M_Z) = {s.alphas_value:g})"
            if s.alphas_mode == "running"
            else f"fixed {s.alphas_value:g}"
        )
        shower = (
            f"{coupling}, kt_cut {s.kt_cut:g} GeV, QCD start angle {s.qcd_start_angle:g}, coherence "
            f"{_on(s.coherence)}, g -> qq {_on(s.g_to_qq)} (n_f = {s.n_flavours}), dead cone {_on(s.dead_cone)}"
            if s.shower
            else "off"
        )
        hadron = (
            f"Field-Feynman a = {a.ff_a:g}, kt_sigma {a.kt_sigma:g} GeV, e_stop {a.e_stop:g} GeV, e_min "
            f"{a.e_min_hadron:g} GeV; pi0 {a.pi0_fraction:g}, neutral {a.neutral_fraction:g}; W singlet frame "
            f"{_on(a.singlet_frame)}, hadron reshuffle {_on(a.hadron_reshuffle)}"
            if a.hadronize
            else "off"
        )
        b_hadrons = (
            f"c tau {a.b_ctau:g} mm, <z_B> {a.b_frag_mean:g}, {a.b_decay_multiplicity} products "
            f"(charged {a.b_charged_share:g}), semileptonic {a.semileptonic:g} + cascade {a.semileptonic_cascade:g}"
            if a.hadronize and a.b_hadrons
            else "off"
        )
        ue = (
            f"{u.ue_density:g} per unit area in a disc of radius {u.ue_radius:g}, pT = {u.ue_pt_min:g} + "
            f"Exp({u.ue_pt_mean:g}) GeV"
            if u.ue
            else "off"
        )
        detector = (
            f"tracks pT > {d.track_pt_min:g} GeV (eff {d.track_eff_low:g}/{d.track_eff_high:g}), ECAL cells "
            f"{d.ecal_cell:g}, HCAL cells {d.hcal_cell:g}, cell position {d.cell_position}"
            if d.detector
            else "off"
        )
        cuts = ""
        if j.jet_pt_min is not None or j.jet_pt_max is not None:
            cuts = f", jet pT in [{j.jet_pt_min}, {j.jet_pt_max}] GeV"
        return "\n".join(
            [
                f"hard:      seed pT {spectrum}, |y| < {h.y_max:g}; QCD quark fraction {h.qcd_quark_fraction:g} "
                f"({flavours})",
                f"           W {h.w_mass:g} +- {h.w_window:g} GeV (width {h.w_width:g}), top {h.top_mass:g} +- "
                f"{h.top_window:g} GeV (width {h.top_width:g}), W helicity {_on(h.w_helicity)}, containment "
                f"{_on(h.containment)}",
                f"shower:    {shower}",
                f"hadrons:   {hadron}",
                f"B hadrons: {b_hadrons}",
                f"UE:        {ue}",
                f"detector:  {detector}",
                f"jet:       {j.jet_def} R = {j.jet_radius:g}{cuts}; soft drop z_cut {j.sd_zcut:g} beta {j.sd_beta:g}; "
                f"tau_N beta {j.tau_beta:g} R0 {j.tau_r0:g} axes {j.tau_axes}",
            ]
        )


def _on(flag: bool) -> str:
    return "on" if flag else "off"


_PART_CLASSES: tuple[type[Any], ...] = (HardPart, ShowerPart, HadronPart, UEPart, DetectorPart, JetPart)
_PARAMETER_PART: dict[str, str] = {}  # parameter name -> name of the part that owns it
for _part, _cls in zip(SimConfig.SECTIONS, _PART_CLASSES, strict=True):
    for _f in dataclasses.fields(_cls):
        if _f.name in _PARAMETER_PART:  # pragma: no cover - guarded at import
            raise RuntimeError(f"duplicate parameter name {_f.name!r}")
        _PARAMETER_PART[_f.name] = _part


def config_fields() -> Iterator[tuple[str, str, dataclasses.Field[Any]]]:
    """Iterate over ``(part name, field name, dataclass field)`` for every parameter, in definition order.

    The field metadata hold ``help``, ``unit`` and, where relevant, ``choices``, ``integer``, ``optional``,
    ``allow_inf``, ``nargs`` and ``metavar`` (used by the command line to build one flag per parameter).
    """
    for name, cls in zip(SimConfig.SECTIONS, _PART_CLASSES, strict=True):
        for f in dataclasses.fields(cls):
            yield name, f.name, f


#: The switches of the physics ladder (the only fields in which the presets differ).
_LADDER_SWITCHES: tuple[str, ...] = ("shower", "hadronize", "b_hadrons", "ue", "detector")


def _ladder(*, shower: bool, hadrons: bool, detector: bool) -> SimConfig:
    return SimConfig().replace(shower=shower, hadronize=hadrons, b_hadrons=hadrons, ue=hadrons, detector=detector)


#: The cumulative physics ladder: each preset switches on more physics; ``SimConfig() == PRESETS["detector"]``.
PRESETS: dict[str, SimConfig] = {
    "partons": _ladder(shower=False, hadrons=False, detector=False),
    "shower": _ladder(shower=True, hadrons=False, detector=False),
    "hadrons": _ladder(shower=True, hadrons=True, detector=False),
    "detector": _ladder(shower=True, hadrons=True, detector=True),
}

_PRESET_TEXT = {
    "partons": "the decay partons: 1 (QCD), 2 (W) or 3 (top) partons",
    "shower": "+ parton shower: massless showered partons",
    "hadrons": "+ hadronisation, B hadrons and the underlying event",
    "detector": "+ tracker, calorimeters and particle flow (the JetClass-like level)",
}


def preset(name: str = "detector", **overrides: Any) -> SimConfig:
    """Return the configuration of a ladder preset, optionally with parameters changed on top.

    Parameters
    ----------
    name
        One of ``"partons"``, ``"shower"``, ``"hadrons"``, ``"detector"`` (see :data:`PRESETS`).
    **overrides
        Parameters to change, as for :meth:`SimConfig.replace`.

    Returns
    -------
    SimConfig

    Examples
    --------
    >>> preset("shower", alphas_mode="fixed", alphas_value=0.12).shower.alphas_value
    0.12
    """
    if name not in PRESETS:
        close = difflib.get_close_matches(name, list(PRESETS), n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}.{hint}")
    return PRESETS[name].replace(**overrides) if overrides else PRESETS[name]


def matching_preset(config: SimConfig) -> str | None:
    """Name of the ladder preset equal to ``config``, or ``None`` if it was modified."""
    for name, candidate in PRESETS.items():
        if candidate == config:
            return name
    return None


def describe_presets() -> str:
    """Table of the ladder presets and the default configuration (the text of ``jetgen presets``)."""
    lines = ["presets (cumulative physics ladder):", ""]
    lines.append(f"  {'preset':<9} {'shower':<7} {'hadronize':<10} {'b_hadrons':<10} {'ue':<4} {'detector':<9}")
    for name, c in PRESETS.items():
        switches = [c.shower.shower, c.hadron.hadronize, c.hadron.b_hadrons, c.ue.ue, c.detector.detector]
        widths = (7, 10, 10, 4, 9)
        row = " ".join(f"{_on(s):<{w}}" for s, w in zip(switches, widths, strict=True))
        lines.append(f"  {name:<9} {row}  {_PRESET_TEXT[name]}")
    lines += ["", "default configuration (detector):", SimConfig().describe()]
    return "\n".join(line.rstrip() for line in lines)
