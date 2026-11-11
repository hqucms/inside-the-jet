"""Perturbations of jets: symmetries, smearing, collinear splits, soft particles and particle removal.

Every function takes a :class:`~itj.data.JetData` and returns a new one in which everything derived from the
four-momenta is recomputed, so a perturbed jet looks exactly like a jet read from a file:

* the particles are re-sorted by pT (descending) and ``part_px``, ``part_py``, ``part_pz``, ``part_energy`` hold
  the new four-momenta;
* ``part_deta = s (eta_i - eta_jet)`` and ``part_dphi = wrap(phi_i - phi_jet)`` are recomputed (pseudorapidity),
  with ``s = +1`` if ``jet_eta >= 0`` else -1 (the JetClass convention);
* ``jet_pt``, ``jet_eta``, ``jet_phi``, ``jet_energy`` and ``jet_nparticles`` are recomputed from the constituents
  (``translate`` keeps the stored jet axis, on purpose);
* ``jet_sdmass`` and ``jet_tau1`` .. ``jet_tau4`` are recomputed only with ``substructure=True``
  (:func:`itj.observables.jet_branches`, the slow part); otherwise they keep the values of the unperturbed jet.

The other particle columns (``part_charge``, the PID flags, the impact parameters and the toy's ``truth_part_*``)
travel with their particle: both pieces of a split particle copy them. An added soft particle is massless with the
flags of a photon or a charged hadron, charge 0 or +-1, impact parameters 0, ``truth_part_prong = -1``,
``truth_part_parton = -1``, ``truth_part_is_ue = 0``, ``truth_part_from_b = 0`` and ``truth_part_purity = 1`` (every
other particle column 0). Per-jet and other truth columns are kept. Perturbed jets are not generated jets: the
identities of ``itj.toy.validate.check_identities`` do not apply to them.

``keep_sign=True`` takes s from the unperturbed jet (its stored ``jet_eta``) instead of the perturbed one: a soft
particle that moves ``jet_eta`` across 0 would otherwise reflect every ``part_deta`` of the jet. The IRC scans of
``itj.interpret.irc_scan`` use it.

Examples
--------
>>> import numpy as np
>>> from itj.data import JetData, Ragged
>>> one = lambda values: Ragged.from_counts(np.asarray(values, dtype=np.float32), [len(values)])
>>> data = JetData({"part_px": one([300.0, 100.0]), "part_py": one([0.0, 10.0]), "part_pz": one([0.0, 0.0]),
...                 "part_energy": one([300.0, np.hypot(100.0, 10.0)])})
>>> split = split_collinear(data, lam=0.25)  # the hardest particle into 1/4 and 3/4, then re-sorted by pT
>>> split["part_px"][0].tolist(), split.counts.tolist()
([225.0, 100.0, 75.0], [3])
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from typing import Literal

import numpy as np

from .cluster import _pt_y_phi_m, _wrap
from .data import JetData, Ragged, column_group

__all__ = [
    "add_soft",
    "drop_soft",
    "keep_particles",
    "max_particles_guard",
    "recompute",
    "remove_particles",
    "rotate",
    "smear_pt",
    "smear_soft",
    "split_collinear",
    "translate",
]

_P4_COLUMNS = ("part_px", "part_py", "part_pz", "part_energy")
_JET_KINEMATICS = ("jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_nparticles")
_SUBSTRUCTURE = ("jet_sdmass", "jet_tau1", "jet_tau2", "jet_tau3", "jet_tau4")

#: Values of the truth particle columns for an added particle (every other particle column is 0).
_ADDED_TRUTH: dict[str, float] = {
    "truth_part_prong": -1,
    "truth_part_parton": -1,
    "truth_part_is_ue": 0,
    "truth_part_from_b": 0,
    "truth_part_purity": 1,
}


def _hint(value: str, choices: tuple[str, ...]) -> str:
    """Return " (did you mean ...?)" for the closest choice, or an empty string."""
    close = difflib.get_close_matches(str(value), choices, n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _per_jet(value: float | np.ndarray, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar or a (n,) array to float64 (n,)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} must be a number or an array of one value per jet ({n}), got shape {arr.shape}")
    return arr


def _particles(data: JetData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the flat four-momenta (M, 4) float64, the particles per jet and the jet of every particle."""
    cols = []
    for name in _P4_COLUMNS:
        if name not in data:
            raise ValueError(f"the jets need the particle four-momenta {_P4_COLUMNS}; {name!r} is missing")
        col = data[name]
        if not isinstance(col, Ragged):
            raise TypeError(f"column {name!r} must be a Ragged")
        cols.append(col)
    counts = cols[0].counts()
    p4 = np.stack([c.values.astype(np.float64) for c in cols], axis=1)
    return p4, counts, cols[0].row_index()


def _decompose(p4: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pT, rapidity, azimuth and transverse mass with ``E = mt cosh(y)``, ``pz = mt sinh(y)`` exactly."""
    pt, y, phi, m = _pt_y_phi_m(p4)
    e, pz = p4[:, 3], p4[:, 2]
    ok = (e + pz > 0) & (e - pz > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_exact = 0.5 * np.log((e + pz) / (e - pz))
        mt = np.sqrt((e + pz) * (e - pz))
    y = np.where(ok, y_exact, y)
    mt = np.where(ok, mt, np.sqrt(pt * pt + m * m))
    return pt, y, phi, mt


def _compose(pt: np.ndarray, y: np.ndarray, phi: np.ndarray, mt: np.ndarray) -> np.ndarray:
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y), mt * np.cosh(y)], axis=1)


def _jet_sums(p4: np.ndarray, row: np.ndarray, n_jets: int) -> np.ndarray:
    return np.stack([np.bincount(row, weights=p4[:, k], minlength=n_jets) for k in range(4)], axis=1)


def _eta(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
    pt = np.hypot(px, py)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pt > 0, np.arcsinh(pz / np.where(pt > 0, pt, 1.0)), 0.0)


def _old_sign(data: JetData, p4: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Sign s of the unperturbed jets: from the stored ``jet_eta``, else from the constituents."""
    if "jet_eta" in data:
        eta = np.asarray(data["jet_eta"], dtype=np.float64)
    else:
        total = _jet_sums(p4, row, len(data))
        eta = _eta(total[:, 0], total[:, 1], total[:, 2])
    return np.where(eta >= 0, 1.0, -1.0)


def _jet_axis(p4: np.ndarray, row: np.ndarray, n_jets: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (y, phi) of the summed four-momentum of every jet."""
    _, y, phi, _ = _pt_y_phi_m(_jet_sums(p4, row, n_jets))
    return y, phi


def _padded(p4: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    length = max(int(counts.max()) if counts.size else 0, 1)
    out = np.zeros((len(counts), length, 4))
    row = np.repeat(np.arange(len(counts)), counts)
    pos = np.arange(len(p4)) - offsets[row]
    out[row, pos] = p4
    return out, np.arange(length)[None, :] < counts[:, None]


def _rebuild(
    data: JetData,
    p4: np.ndarray,
    counts: np.ndarray,
    source: np.ndarray,
    *,
    sign: np.ndarray | None,
    added: Mapping[str, np.ndarray] | None = None,
    axis: tuple[np.ndarray, np.ndarray] | None = None,
    substructure: bool = False,
) -> JetData:
    """Return ``data`` with new particles and every derived column recomputed.

    ``p4`` (M, 4) are the new particles jet by jet (``counts`` per jet), ``source`` the index of the old particle
    whose other columns each one copies (-1: a new particle, its columns from ``added`` or the defaults). ``sign``
    fixes s per jet (None: from the new jets); ``axis`` fixes (eta_jet, phi_jet) (None: from the new jets).
    """
    n_jets = len(data)
    row = np.repeat(np.arange(n_jets, dtype=np.int64), counts)
    # the four-momenta as stored (float32 for files); the sorting and everything else are computed from them
    stored: dict[str, np.ndarray] = {}
    for k, name in enumerate(_P4_COLUMNS):
        col = data[name]
        assert isinstance(col, Ragged)
        stored[name] = p4[:, k].astype(col.values.dtype)
    p4 = np.stack([stored[name].astype(np.float64) for name in _P4_COLUMNS], axis=1)
    pt = np.hypot(p4[:, 0], p4[:, 1])
    order = np.lexsort((-pt, row))
    p4, source = p4[order], source[order]
    stored = {name: values[order] for name, values in stored.items()}
    added = {k: np.asarray(v)[order] for k, v in (added or {}).items()}
    offsets = np.zeros(n_jets + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    total = _jet_sums(p4, row, n_jets)
    jet_pt = np.hypot(total[:, 0], total[:, 1])
    jet_eta = _eta(total[:, 0], total[:, 1], total[:, 2])
    jet_phi = np.arctan2(total[:, 1], total[:, 0])
    jet: dict[str, np.ndarray] = {
        "jet_pt": jet_pt,
        "jet_energy": total[:, 3],
        "jet_nparticles": counts.astype(np.float64),
    }
    if axis is None:
        jet["jet_eta"], jet["jet_phi"] = jet_eta, jet_phi
        axis = (jet_eta, jet_phi)
    if sign is None:
        sign = np.where(axis[0] >= 0, 1.0, -1.0)
    eta = _eta(p4[:, 0], p4[:, 1], p4[:, 2])
    stored["part_deta"] = sign[row] * (eta - axis[0][row])
    stored["part_dphi"] = _wrap(np.arctan2(p4[:, 1], p4[:, 0]) - axis[1][row])
    if substructure:
        from .observables import jet_branches

        padded, mask = _padded(p4, counts)
        branches = jet_branches(padded, mask)
        jet.update({name: branches[name] for name in _SUBSTRUCTURE})

    old = source >= 0
    new_index = np.flatnonzero(~old)
    columns: dict[str, np.ndarray | Ragged] = {}
    for name, col in data.columns.items():
        if isinstance(col, Ragged) and column_group(name, jagged=True) == "particle":
            dtype = col.values.dtype
            if name in stored:
                values = stored[name].astype(dtype)
            else:
                values = np.zeros((len(source), *col.values.shape[1:]), dtype=dtype)
                values[old] = col.values[source[old]]
                if new_index.size:
                    default = added.get(name, _ADDED_TRUTH.get(name, 0))
                    values[new_index] = np.asarray(default)[new_index] if np.ndim(default) else default
            columns[name] = Ragged(values, offsets)
        elif name in jet:
            columns[name] = np.asarray(jet[name]).astype(np.asarray(col).dtype)
        else:
            columns[name] = col
    if substructure:
        for name in _SUBSTRUCTURE:
            if name not in columns:
                columns[name] = jet[name]
    return JetData(columns, config=data.config, provenance=data.provenance, source=data.source)


def _sign(data: JetData, p4: np.ndarray, row: np.ndarray, keep_sign: bool) -> np.ndarray | None:
    return _old_sign(data, p4, row) if keep_sign else None


# ------------------------------------------------------------------------------------------------------- functions


def recompute(data: JetData, *, substructure: bool = False, keep_sign: bool = False) -> JetData:
    """Re-sort the particles and recompute every column derived from ``part_px/py/pz/energy``.

    Parameters
    ----------
    data : JetData
        Jets with the four particle momentum columns.
    substructure : bool
        Also recompute ``jet_sdmass`` and ``jet_tau1`` .. ``jet_tau4``.
    keep_sign : bool
        Take the sign s of ``part_deta`` from the stored ``jet_eta``.

    Returns
    -------
    JetData
    """
    p4, counts, row = _particles(data)
    source = np.arange(len(p4), dtype=np.int64)
    return _rebuild(data, p4, counts, source, sign=_sign(data, p4, row, keep_sign), substructure=substructure)


def rotate(
    data: JetData,
    *,
    angle: float | np.ndarray | None = None,
    rng: np.random.Generator | int | None = None,
    substructure: bool = False,
    keep_sign: bool = False,
) -> JetData:
    """Rotate every jet about its axis in the (y, phi) plane.

    The offsets ``(dy, dphi)`` of every particle from the jet axis (the direction of the summed four-momentum) are
    rotated by the angle; pT and transverse mass of every particle are kept. Pairwise Delta R is exactly invariant;
    the jet mass only for small angles.

    Parameters
    ----------
    data : JetData
        The jets.
    angle : float or numpy.ndarray, optional
        Rotation angle in rad, one for all jets or one per jet; None: uniform in [0, 2 pi) per jet.
    rng : numpy.random.Generator or int, optional
        Random numbers for ``angle=None``.
    substructure, keep_sign : bool
        See :func:`recompute`.

    Returns
    -------
    JetData
    """
    p4, counts, row = _particles(data)
    sign = _sign(data, p4, row, keep_sign)
    n_jets = len(data)
    alpha = _rng(rng).uniform(0.0, 2.0 * np.pi, n_jets) if angle is None else _per_jet(angle, n_jets, "angle")
    pt, y, phi, mt = _decompose(p4)
    ay, aphi = _jet_axis(p4, row, n_jets)
    dy = y - ay[row]
    dphi = _wrap(phi - aphi[row])
    c, s = np.cos(alpha)[row], np.sin(alpha)[row]
    new = _compose(pt, ay[row] + c * dy - s * dphi, aphi[row] + s * dy + c * dphi, mt)
    return _rebuild(data, new, counts, np.arange(len(p4)), sign=sign, substructure=substructure)


def translate(
    data: JetData,
    *,
    deta: float | np.ndarray,
    dphi: float | np.ndarray,
    substructure: bool = False,
    keep_sign: bool = False,
) -> JetData:
    """Shift the relative coordinates of every particle: ``part_deta += deta`` and ``part_dphi += dphi``.

    The particles move by ``s * deta`` in pseudorapidity and ``dphi`` in azimuth (pT and mass kept) while the stored
    jet axis ``jet_eta``, ``jet_phi`` stays where it was, so the jet image moves off centre (the JetCLR
    translation). ``jet_pt``, ``jet_energy`` and ``jet_nparticles`` are recomputed.

    Parameters
    ----------
    data : JetData
        The jets.
    deta, dphi : float or numpy.ndarray
        Shifts, one for all jets or one per jet.
    substructure, keep_sign : bool
        See :func:`recompute` (the sign always comes from the stored jet axis, which does not move).

    Returns
    -------
    JetData
    """
    del keep_sign  # the axis, hence s, is that of the unperturbed jet anyway
    p4, counts, row = _particles(data)
    n_jets = len(data)
    sign = _old_sign(data, p4, row)
    if "jet_eta" in data and "jet_phi" in data:
        axis = (np.asarray(data["jet_eta"], np.float64), np.asarray(data["jet_phi"], np.float64))
    else:
        total = _jet_sums(p4, row, n_jets)
        axis = (_eta(total[:, 0], total[:, 1], total[:, 2]), np.arctan2(total[:, 1], total[:, 0]))
    shift_eta = (sign * _per_jet(deta, n_jets, "deta"))[row]
    shift_phi = _per_jet(dphi, n_jets, "dphi")[row]
    pt = np.hypot(p4[:, 0], p4[:, 1])
    m2 = np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(axis=1), 0.0)
    eta = _eta(p4[:, 0], p4[:, 1], p4[:, 2]) + shift_eta
    phi = np.arctan2(p4[:, 1], p4[:, 0]) + shift_phi
    pz = pt * np.sinh(eta)
    energy = np.sqrt(pt * pt + pz * pz + m2)
    new = np.stack([pt * np.cos(phi), pt * np.sin(phi), pz, energy], axis=1)
    return _rebuild(data, new, counts, np.arange(len(p4)), sign=sign, axis=axis, substructure=substructure)


def smear_soft(
    data: JetData,
    *,
    scale: float = 0.1,
    rng: np.random.Generator | int | None = None,
    substructure: bool = False,
    keep_sign: bool = False,
) -> JetData:
    """Smear the direction of every particle by ``dy, dphi += N(0, (scale / pt_i[GeV])^2)`` (JetCLR soft smearing).

    Soft particles move most; pT and transverse mass are kept.

    Parameters
    ----------
    data : JetData
        The jets.
    scale : float
        Width times pT, in GeV (rad GeV).
    rng : numpy.random.Generator or int, optional
        Random numbers.
    substructure, keep_sign : bool
        See :func:`recompute`.

    Returns
    -------
    JetData
    """
    p4, counts, row = _particles(data)
    sign = _sign(data, p4, row, keep_sign)
    gen = _rng(rng)
    pt, y, phi, mt = _decompose(p4)
    with np.errstate(divide="ignore"):
        sigma = np.where(pt > 0, scale / np.where(pt > 0, pt, 1.0), 0.0)
    noise = gen.standard_normal((2, len(pt)))
    new = _compose(pt, y + sigma * noise[0], phi + sigma * noise[1], mt)
    return _rebuild(data, new, counts, np.arange(len(p4)), sign=sign, substructure=substructure)


def smear_pt(
    data: JetData,
    *,
    sigma: float,
    rng: np.random.Generator | int | None = None,
    substructure: bool = False,
    keep_sign: bool = False,
) -> JetData:
    """Smear the pT of every particle by a log-normal factor ``exp(sigma g - sigma^2 / 2)``, g ~ N(0, 1).

    The factor has mean 1 and is always positive; rapidity, azimuth and mass are kept.

    Parameters
    ----------
    data : JetData
        The jets.
    sigma : float
        Relative width (for small sigma).
    rng : numpy.random.Generator or int, optional
        Random numbers.
    substructure, keep_sign : bool
        See :func:`recompute`.

    Returns
    -------
    JetData
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    p4, counts, row = _particles(data)
    sign = _sign(data, p4, row, keep_sign)
    pt, y, phi, mt = _decompose(p4)
    factor = np.exp(sigma * _rng(rng).standard_normal(len(pt)) - 0.5 * sigma * sigma)
    m2 = mt * mt - pt * pt
    new_pt = pt * factor
    new = _compose(new_pt, y, phi, np.sqrt(np.maximum(new_pt * new_pt + m2, 0.0)))
    return _rebuild(data, new, counts, np.arange(len(p4)), sign=sign, substructure=substructure)


def _selection(
    which: str | np.ndarray | Ragged, pt: np.ndarray, counts: np.ndarray, row: np.ndarray, gen: np.random.Generator
) -> np.ndarray:
    """Return a flat bool mask of the particles chosen by ``which``."""
    n_values = len(pt)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    nonempty = np.flatnonzero(counts > 0)
    chosen = np.zeros(n_values, dtype=bool)
    if isinstance(which, Ragged):
        which = which.values
    if isinstance(which, str):
        if which == "all":
            chosen[:] = True
        elif which == "hardest":
            if n_values:
                best = np.lexsort((-pt, row))[offsets[nonempty]]
                chosen[best] = True
        elif which == "random":
            pick = offsets[nonempty] + (gen.random(len(nonempty)) * counts[nonempty]).astype(np.int64)
            chosen[pick] = True
        else:
            raise ValueError(
                f"unknown which {which!r}{_hint(which, ('hardest', 'random', 'all'))}: use 'hardest', 'random', 'all' "
                "or an array"
            )
        return chosen
    arr = np.asarray(which)
    if arr.dtype == bool and arr.shape == (n_values,):
        return arr.copy()
    if np.issubdtype(arr.dtype, np.integer) and arr.shape == (len(counts),):
        ok = (arr >= 0) & (arr < counts)
        chosen[offsets[:-1][ok] + arr[ok]] = True
        return chosen
    raise ValueError(
        f"which must be 'hardest', 'random', 'all', a bool array of one value per particle ({n_values}) or an "
        f"integer array of one particle index per jet ({len(counts)}, -1 for none); got shape {arr.shape}"
    )


def split_collinear(
    data: JetData,
    *,
    which: Literal["hardest", "random", "all"] | np.ndarray = "hardest",
    lam: float = 0.5,
    delta: float = 0.0,
    frac: float | None = None,
    rng: np.random.Generator | int | None = None,
    substructure: bool = False,
    keep_sign: bool = False,
) -> JetData:
    """Split particles into two collinear (or nearly collinear) pieces with momentum fractions lam and 1 - lam.

    With ``delta = 0`` a particle p becomes ``lam p`` and ``(1 - lam) p`` (exact four-momentum conservation). With
    ``delta > 0`` the pieces get pT ``lam pt`` and ``(1 - lam) pt`` and transverse masses ``lam mt`` and
    ``(1 - lam) mt`` (so masses ``lam m`` and ``(1 - lam) m``) and sit ``delta`` apart in (y, phi), on a line at a
    random azimuth through the particle, with their pT-weighted centre on the original direction; their sum tends to
    the particle as ``delta -> 0``, also for float32 particles with a slightly negative m^2. Both pieces copy the other
    columns of the particle.

    Parameters
    ----------
    data : JetData
        The jets.
    which : {"hardest", "random", "all"} or numpy.ndarray
        The particles to split: the hardest or a random one per jet, all of them, a bool array of one value per
        particle (the flat ``Ragged.values`` order) or an integer array of one particle index per jet (-1: none).
    lam : float
        Momentum fraction of the first piece, in (0, 1).
    delta : float
        Separation of the pieces in (y, phi).
    frac : float, optional
        Split every particle independently with this probability (replaces ``which``).
    rng : numpy.random.Generator or int, optional
        Random numbers (azimuths, ``"random"``, ``frac``).
    substructure, keep_sign : bool
        See :func:`recompute`.

    Returns
    -------
    JetData
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    p4, counts, row = _particles(data)
    sign = _sign(data, p4, row, keep_sign)
    gen = _rng(rng)
    pt = np.hypot(p4[:, 0], p4[:, 1])
    if frac is not None:
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"frac must be in [0, 1], got {frac}")
        chosen = gen.random(len(pt)) < frac
    else:
        chosen = _selection(which, pt, counts, row, gen)
    reps = 1 + chosen.astype(np.int64)
    source = np.repeat(np.arange(len(p4), dtype=np.int64), reps)
    second = np.zeros(len(source), dtype=bool)
    starts = np.cumsum(reps) - reps
    second[starts[chosen] + 1] = True
    first = np.zeros(len(source), dtype=bool)
    first[starts[chosen]] = True
    new = p4[source].copy()
    if delta == 0:
        new[first] *= lam
        new[second] *= 1.0 - lam
    else:
        sel = np.flatnonzero(chosen)
        # scale pT and the exact transverse mass mt = sqrt((E + pz)(E - pz)) of the particle: the pieces then sum to
        # it as delta -> 0 even for float32 inputs with m^2 < 0 (a mass recomputed and clipped at 0 would not)
        s_pt, s_y, s_phi, s_mt = _decompose(p4[sel])
        azimuth = gen.uniform(0.0, 2.0 * np.pi, len(sel))
        ux, uy = np.cos(azimuth), np.sin(azimuth)
        for mask_, f, shift in ((first, lam, (1.0 - lam) * delta), (second, 1.0 - lam, -lam * delta)):
            new[mask_] = _compose(f * s_pt, s_y + shift * ux, s_phi + shift * uy, f * s_mt)
    new_counts = counts + np.bincount(row[chosen], minlength=len(counts))
    return _rebuild(data, new, new_counts, source, sign=sign, substructure=substructure)


def add_soft(
    data: JetData,
    *,
    n: int = 1,
    eps: float = 1e-3,
    kind: Literal["photon", "charged"] = "photon",
    R: float = 0.8,
    rng: np.random.Generator | int | None = None,
    substructure: bool = False,
    keep_sign: bool = False,
) -> JetData:
    """Add n soft massless particles of pT ``eps * pt_J`` each to every jet, uniform in the disc ``dR < R``.

    ``pt_J`` is the pT of the unperturbed jet (sum of its constituents) and the disc is centred on its axis in
    (y, phi). The particles are photons (``part_isPhoton = 1``, charge 0) or charged hadrons
    (``part_isChargedHadron = 1``, charge +-1 at random); impact parameters are 0. Jets without particles get none.

    Parameters
    ----------
    data : JetData
        The jets.
    n : int
        Particles per jet.
    eps : float
        pT of every added particle as a fraction of the jet pT.
    kind : {"photon", "charged"}
        Type of the added particles.
    R : float
        Radius of the disc.
    rng : numpy.random.Generator or int, optional
        Random numbers.
    substructure, keep_sign : bool
        See :func:`recompute`.

    Returns
    -------
    JetData
    """
    if kind not in ("photon", "charged"):
        raise ValueError(f"unknown kind {kind!r}{_hint(kind, ('photon', 'charged'))}: use 'photon' or 'charged'")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    p4, counts, row = _particles(data)
    sign = _sign(data, p4, row, keep_sign)
    gen = _rng(rng)
    n_jets = len(data)
    total = _jet_sums(p4, row, n_jets)
    ay, aphi = _jet_axis(p4, row, n_jets)
    extra = np.where(counts > 0, int(n), 0)
    jet_of = np.repeat(np.arange(n_jets), extra)
    k = len(jet_of)
    radius = R * np.sqrt(gen.random(k))
    theta = gen.uniform(0.0, 2.0 * np.pi, k)
    pt_soft = eps * np.hypot(total[jet_of, 0], total[jet_of, 1])
    soft = _compose(pt_soft, ay[jet_of] + radius * np.cos(theta), aphi[jet_of] + radius * np.sin(theta), pt_soft)
    # old particles then the new ones of every jet; _rebuild sorts by pT
    all_rows = np.concatenate([row, jet_of])
    order = np.argsort(all_rows, kind="stable")
    new = np.concatenate([p4, soft])[order]
    source = np.concatenate([np.arange(len(p4)), np.full(k, -1)])[order]
    is_new = source < 0
    added: dict[str, np.ndarray] = {}
    if kind == "photon":
        added["part_isPhoton"] = is_new.astype(np.int64)
    else:
        added["part_isChargedHadron"] = is_new.astype(np.int64)
        charge = np.zeros(len(source))
        charge[is_new] = gen.choice([-1.0, 1.0], size=k)
        added["part_charge"] = charge
    return _rebuild(data, new, counts + extra, source, sign=sign, added=added, substructure=substructure)


def drop_soft(
    data: JetData,
    *,
    pt_max: float = 1.0,
    p: float = 1.0,
    rng: np.random.Generator | int | None = None,
    substructure: bool = False,
    keep_sign: bool = False,
) -> JetData:
    """Remove every particle with pT below ``pt_max`` with probability p (the hardest particle of a jet stays).

    Parameters
    ----------
    data : JetData
        The jets.
    pt_max : float
        pT threshold in GeV.
    p : float
        Probability to remove a particle below the threshold.
    rng : numpy.random.Generator or int, optional
        Random numbers.
    substructure, keep_sign : bool
        See :func:`recompute`.

    Returns
    -------
    JetData
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    p4, counts, row = _particles(data)
    pt = np.hypot(p4[:, 0], p4[:, 1])
    drop = (pt < pt_max) & (_rng(rng).random(len(pt)) < p)
    drop &= ~_selection("hardest", pt, counts, row, np.random.default_rng(0))
    return _keep(data, ~drop, substructure=substructure, keep_sign=keep_sign)


def _keep(data: JetData, keep: np.ndarray, *, substructure: bool, keep_sign: bool) -> JetData:
    p4, counts, row = _particles(data)
    sign = _sign(data, p4, row, keep_sign)
    index = np.flatnonzero(keep)
    new_counts = np.bincount(row[index], minlength=len(counts)).astype(np.int64)
    return _rebuild(data, p4[index], new_counts, index, sign=sign, substructure=substructure)


def _flat_mask(data: JetData, mask: Ragged | np.ndarray, name: str) -> np.ndarray:
    counts = data.counts
    if isinstance(mask, Ragged):
        if not np.array_equal(mask.counts(), counts):
            raise ValueError(f"{name} must have one value per particle (the row lengths of part_px)")
        values = mask.values
    else:
        values = np.asarray(mask)
        if values.shape != (int(counts.sum()),):
            raise ValueError(f"{name} must be a Ragged aligned with part_* or a flat array of one value per particle")
    return np.asarray(values, dtype=bool)


def remove_particles(data: JetData, drop: Ragged, *, substructure: bool = False, keep_sign: bool = False) -> JetData:
    """Remove the particles where ``drop`` is True (a bool Ragged aligned with ``part_*``); jets may become empty."""
    return _keep(data, ~_flat_mask(data, drop, "drop"), substructure=substructure, keep_sign=keep_sign)


def keep_particles(data: JetData, keep: Ragged, *, substructure: bool = False, keep_sign: bool = False) -> JetData:
    """Keep only the particles where ``keep`` is True (a bool Ragged aligned with ``part_*``)."""
    return _keep(data, _flat_mask(data, keep, "keep"), substructure=substructure, keep_sign=keep_sign)


def max_particles_guard(data: JetData, length: int = 128) -> None:
    """Raise ValueError if a jet has more than ``length`` particles.

    The taggers see at most ``length`` particles (the weaver configs pad and truncate to 128): a perturbation that
    pushes a jet above it would change the input by truncation, not by the perturbation (an IRC-test artefact).
    """
    counts = data.counts
    over = counts > length
    if over.any():
        raise ValueError(
            f"{int(over.sum())} jets have more than {length} particles (up to {int(counts.max())}): the taggers "
            f"truncate them, so the perturbation would not be seen as such; add fewer particles or select the jets "
            f"with data.select(data.counts <= {length})"
        )
