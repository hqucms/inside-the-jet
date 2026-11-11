"""Figures of the course: jet displays, the preset ladder, jet images, Lund planes, ROC curves and analysis plots.

matplotlib is imported only when a function is called (``pip install -e ".[analysis]"``); :func:`animate_ladder`
also needs pillow. Every function takes ``ax=None`` (a new figure) or an existing Axes and returns the Axes, or the
Figure for the multi-panel figures (:func:`ladder`, :func:`efn_filters`).

* :func:`jet_display`: one jet in the (Delta eta, Delta phi) plane, marker area proportional to pT, coloured by
  particle type, prong, parton, underlying event or any per-particle value (attributions);
* :func:`ladder` and :func:`animate_ladder`: the same jet at every preset (``itj.toy.show_ladder``);
* :func:`jet_image`, :func:`lund_plane`: averaged jet images and Lund-plane densities;
* :func:`roc_curves`, :func:`curves`, :func:`history`: ROC curves (rejection on a log axis), scans and learning
  curves;
* :func:`attention_matrix`, :func:`attribution_display`, :func:`efn_filters`, :func:`cka_heatmap`: the
  interpretability figures of T3 and T5.

Colours come from :data:`palette` (the Okabe-Ito colours, distinguishable with colour-vision deficiencies) with named
roles; continuous values use ``viridis`` (unsigned) or ``RdBu_r`` (signed, centred on 0). Plain matplotlib
otherwise: the figures follow the active style.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    import os

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .data import JetData, JetView

__all__ = [
    "animate_ladder",
    "attention_matrix",
    "attribution_display",
    "cka_heatmap",
    "curves",
    "efn_filters",
    "history",
    "jet_display",
    "jet_image",
    "ladder",
    "lund_plane",
    "palette",
    "roc_curves",
]

#: Colour-vision-deficiency-safe colours (Okabe & Ito) and the roles they play in the figures. The process keys use
#: the lower-case vocabulary of ``JetData.process_names``.
palette: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
    # processes
    "qcd": "#0072B2",
    "quark": "#56B4E9",
    "gluon": "#CC79A7",
    "wqq": "#E69F00",
    "tbqq": "#009E73",
    # prongs (top: 0 = b, 1-2 = the W quarks) and the underlying event
    "prong0": "#D55E00",
    "prong1": "#0072B2",
    "prong2": "#E69F00",
    "ue": "#999999",
    # particle types
    "charged": "#0072B2",
    "neutral": "#009E73",
    "photon": "#E69F00",
    "lepton": "#CC79A7",
}

#: Colour of particles without a type (the partons and shower presets) and of the hard jet in the "ue" colouring.
_PLAIN = "#000000"
#: Colours cycled over the partons of the "parton" colouring (grey is reserved for the underlying event).
_CYCLE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")
#: Marker area (points^2) of a particle that carries the whole scalar pT sum of the jet.
_AREA = 1500.0
#: Smallest marker area, so that the softest particles stay visible.
_MIN_AREA = 2.0
#: Labels of the display axes (``part_deta`` carries the JetClass sign convention).
_XLABEL = r"$\Delta\eta$"
_YLABEL = r"$\Delta\phi$"

ColorBy = Literal["type", "prong", "parton", "ue"]


# ----------------------------------------------------------------------------------------------------------------------
# Helpers


def _pyplot() -> Any:
    import matplotlib.pyplot as plt

    return plt


def _new_axes(ax: Axes | None, figsize: tuple[float, float] = (5.0, 4.4)) -> Axes:
    if ax is not None:
        return ax
    _, ax = _pyplot().subplots(figsize=figsize, layout="constrained")
    return ax


def _numpy(a: Any) -> np.ndarray:
    """Return a NumPy array from an array, a list or a torch tensor (detached, on the CPU)."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def _wrap(dphi: np.ndarray) -> np.ndarray:
    out = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
    return np.where(out == -np.pi, np.pi, out)


def _process(data: JetData, i: int) -> str | None:
    try:
        return str(data.process_names[i])
    except (ValueError, KeyError, IndexError):
        return None


def _jet_kinematics(view: JetView) -> tuple[float, float, float]:
    """Return (pT, eta, mass) of the jet from its constituents (falling back to the stored branches)."""
    p = view.particles
    if all(k in p for k in ("part_px", "part_py", "part_pz", "part_energy")):
        px, py, pz, e = (
            float(np.sum(p[k], dtype=np.float64)) for k in ("part_px", "part_py", "part_pz", "part_energy")
        )
        pt = math.hypot(px, py)
        eta = math.asinh(pz / pt) if pt > 0 else 0.0
        mass = math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))
        return pt, float(view.jet.get("jet_eta", eta)), mass
    return float(view.jet.get("jet_pt", 0.0)), float(view.jet.get("jet_eta", 0.0)), float("nan")


def _particle_pt(view: JetView) -> np.ndarray:
    p = view.particles
    return np.hypot(np.asarray(p["part_px"], dtype=np.float64), np.asarray(p["part_py"], dtype=np.float64))


def _sizes(view: JetView, size: str | ArrayLike) -> np.ndarray:
    """Marker areas proportional to pT (or energy, or the given values) as a fraction of the jet's sum."""
    if isinstance(size, str):
        if size == "pt":
            values = _particle_pt(view)
        elif size in ("energy", "e"):
            values = np.asarray(view.particles["part_energy"], dtype=np.float64)
        else:
            raise ValueError(f"size must be 'pt', 'energy' or an array, got {size!r}")
    else:
        values = np.abs(np.asarray(size, dtype=np.float64)).reshape(-1)
    total = float(values.sum())
    frac = values / total if total > 0 else np.zeros_like(values)
    return np.maximum(_AREA * frac, _MIN_AREA)


def _type_colours(view: JetView) -> tuple[list[str], dict[str, str]]:
    """Colour by particle type: charged hadron, neutral hadron, photon, lepton (no type: black)."""
    p = view.particles
    n = len(p["part_deta"])

    def flag(name: str) -> np.ndarray:
        return np.asarray(p[name]) > 0 if name in p else np.zeros(n, dtype=bool)

    charged, neutral, photon = flag("part_isChargedHadron"), flag("part_isNeutralHadron"), flag("part_isPhoton")
    lepton = flag("part_isElectron") | flag("part_isMuon")
    if not (charged | neutral | photon | lepton).any() and "part_charge" in p:
        # no PID flags (partons and shower presets): charged or neutral only, if the charge is known
        charge = np.asarray(p["part_charge"])
        if np.any(charge != 0):
            charged, neutral = charge != 0, charge == 0
    colours = [_PLAIN] * n
    used: dict[str, str] = {}
    for mask, key, label in (
        (charged, "charged", "charged hadron"),
        (neutral, "neutral", "neutral hadron"),
        (photon, "photon", "photon"),
        (lepton, "lepton", "electron or muon"),
    ):
        for k in np.flatnonzero(mask):
            colours[k] = palette[key]
        if mask.any():
            used[label] = palette[key]
    if not used:
        used["particle"] = _PLAIN
    return colours, used


def _require(view: JetView, name: str, color: str) -> np.ndarray:
    if name not in view.particles:
        raise ValueError(f"color={color!r} needs the truth column {name!r} (toy data with truth)")
    return np.asarray(view.particles[name])


def _categorical_colours(view: JetView, color: str) -> tuple[list[str], dict[str, str]]:
    if color == "type":
        return _type_colours(view)
    if color == "prong":
        prong = _require(view, "truth_part_prong", color).astype(np.int64)
        colours = [palette[f"prong{k}"] if 0 <= k <= 2 else (palette["ue"] if k < 0 else _PLAIN) for k in prong]
        used = {f"prong {k}": palette[f"prong{k}"] for k in np.unique(prong).tolist() if 0 <= k <= 2}
        if (prong < 0).any():
            used["underlying event"] = palette["ue"]
        return colours, used
    if color == "parton":
        parton = _require(view, "truth_part_parton", color).astype(np.int64)
        colours = [palette["ue"] if k < 0 else _CYCLE[k % len(_CYCLE)] for k in parton]
        used = {"one colour per parton": _CYCLE[0]}
        if (parton < 0).any():
            used["underlying event"] = palette["ue"]
        return colours, used
    if color == "ue":
        is_ue = _require(view, "truth_part_is_ue", color) > 0
        colours = [palette["ue"] if u else _PLAIN for u in is_ue]
        return colours, {"hard jet": _PLAIN, "underlying event": palette["ue"]}
    raise ValueError(f"color must be 'type', 'prong', 'parton', 'ue' or an array, got {color!r}")


def _value_norm(values: np.ndarray, cmap: str | None) -> tuple[str, Any]:
    """Colour map and norm: signed values centred on 0 (RdBu_r), unsigned ones on viridis."""
    from matplotlib.colors import Normalize

    finite = values[np.isfinite(values)]
    if finite.size and finite.min() < 0:
        vmax = float(np.max(np.abs(finite))) or 1.0
        return cmap or "RdBu_r", Normalize(-vmax, vmax)
    vmax = float(finite.max()) if finite.size else 1.0
    return cmap or "viridis", Normalize(0.0 if finite.size == 0 else min(0.0, float(finite.min())), vmax or 1.0)


def _draw_jet(
    ax: Axes,
    data: JetData,
    i: int,
    *,
    color: ColorBy | ArrayLike,
    size: str | ArrayLike,
    show_prongs: bool,
    show_axes: int | None,
    alpha: float = 0.85,
    cmap: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Draw jet ``i`` on ``ax``; return (the scatter of the particles, legend handles by label)."""
    from matplotlib.lines import Line2D

    view = data.jet(i)
    deta = np.asarray(view.particles["part_deta"], dtype=np.float64)
    dphi = np.asarray(view.particles["part_dphi"], dtype=np.float64)
    area = _sizes(view, size)
    if len(area) != len(deta):
        raise ValueError(f"size has {len(area)} values for {len(deta)} particles")
    order = np.argsort(-area, kind="stable")  # large markers first, small ones on top
    handles: dict[str, Any] = {}
    if isinstance(color, str):
        colours, used = _categorical_colours(view, color)
        sc = ax.scatter(
            deta[order],
            dphi[order],
            s=area[order],
            c=[colours[k] for k in order],
            alpha=alpha,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        for label, colour in used.items():
            handles[label] = Line2D([], [], ls="none", marker="o", ms=7, color=colour, alpha=alpha, label=label)
    else:
        given = _numpy(color).astype(np.float64).reshape(-1)
        if len(given) < len(deta):
            raise ValueError(f"color has {len(given)} values for {len(deta)} particles")
        values = given[: len(deta)]  # padded per-particle arrays are accepted
        cmap_name, norm = _value_norm(values, cmap)
        sc = ax.scatter(
            deta[order],
            dphi[order],
            s=area[order],
            c=values[order],
            cmap=cmap_name,
            norm=norm,
            alpha=alpha,
            edgecolors="0.3",
            linewidths=0.4,
            zorder=3,
        )
    if show_prongs and "truth_prong_deta" in view.prongs:
        pdeta = np.asarray(view.prongs["truth_prong_deta"], dtype=np.float64)
        pdphi = np.asarray(view.prongs["truth_prong_dphi"], dtype=np.float64)
        for k, (x, y) in enumerate(zip(pdeta, pdphi, strict=True)):
            colour = palette[f"prong{k}"] if k <= 2 else _PLAIN
            ax.plot(x, y, ls="none", marker="X", ms=10, mfc=colour, mec="black", mew=0.8, zorder=5)
        if len(pdeta):
            label = "initiator" if len(pdeta) == 1 else "prong (decay quark)"
            handles[label] = Line2D([], [], ls="none", marker="X", ms=9, mfc="white", mec="black", label=label)
    if show_axes:
        axes = _subjet_axes(data, i, int(show_axes))
        ax.plot(axes[:, 0], axes[:, 1], ls="none", marker="P", ms=9, mfc="white", mec="black", mew=1.0, zorder=6)
        label = f"{int(show_axes)} WTA-kt axes"
        handles[label] = Line2D([], [], ls="none", marker="P", ms=9, mfc="white", mec="black", label=label)
    return sc, handles


def _subjet_axes(data: JetData, i: int, n: int) -> np.ndarray:
    """Return the ``n`` exclusive-kt WTA axes of jet ``i`` as (Delta eta, Delta phi) with the JetClass sign."""
    from .observables import subjet_axes

    one = data.select(np.array([i]))
    p4, mask = one.padded_p4()
    axes = np.asarray(subjet_axes(p4, mask, n, axes="wta_kt"))[0]  # (n, 2) (y, phi)
    view = data.jet(i)
    _, jet_eta, _ = _jet_kinematics(view)
    jet_phi = float(view.jet.get("jet_phi", math.atan2(float(np.sum(p4[0, :, 1])), float(np.sum(p4[0, :, 0])))))
    sign = 1.0 if jet_eta >= 0 else -1.0
    # the axes are in rapidity; the display uses pseudorapidity (equal for massless particles)
    return np.stack([sign * (axes[:, 0] - jet_eta), _wrap(axes[:, 1] - jet_phi)], axis=1)


def _decorate_display(ax: Axes, lim: float, R: float | None = 0.8) -> None:
    from matplotlib.patches import Circle

    if R:
        ax.add_patch(Circle((0.0, 0.0), R, fill=False, ls="--", lw=0.8, color="0.6", zorder=1))
    ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(_XLABEL)
    ax.set_ylabel(_YLABEL)


def _title(data: JetData, i: int) -> str:
    view = data.jet(i)
    pt, _, mass = _jet_kinematics(view)
    process = _process(data, i)
    parts = [f"jet {view.index}"] + ([process] if process else [])
    parts.append(f"$p_T$ = {pt:.0f} GeV")
    if math.isfinite(mass):
        parts.append(f"m = {mass:.0f} GeV")
    parts.append(f"N = {len(view.particles['part_deta'])}")
    return ", ".join(parts)


# ----------------------------------------------------------------------------------------------------------------------
# Jet displays and the ladder


def jet_display(
    data: JetData,
    i: int,
    *,
    color: ColorBy | ArrayLike = "type",
    size: Literal["pt", "energy"] | ArrayLike = "pt",
    show_prongs: bool = True,
    show_axes: int | None = None,
    lim: float = 0.8,
    ax: Axes | None = None,
    colorbar: bool = True,
    cmap: str | None = None,
    label: str | None = None,
    legend: bool = True,
    title: str | None = None,
) -> Axes:
    """Draw one jet in the (Delta eta, Delta phi) plane: marker area proportional to pT, equal axes.

    The coordinates are ``part_deta`` and ``part_dphi`` (``part_deta`` has the JetClass sign: positive towards larger
    ``|eta|``). The dashed circle is the jet radius R = 0.8. Unlike the ``visualize()`` helper of the dl4hep
    notebooks, the areas are proportional to pT (not its square root), the axes are equal and the limits are a
    parameter.

    Parameters
    ----------
    data : JetData
        Jets (toy or JetClass).
    i : int
        Index of the jet.
    color : {"type", "prong", "parton", "ue"} or array_like
        "type": charged hadron, neutral hadron, photon, lepton (black without PID flags); "prong":
        ``truth_part_prong`` (grey: underlying event); "parton": one colour per ``truth_part_parton``; "ue":
        ``truth_part_is_ue``; an array: one value per particle (for example attributions; a padded row is
        accepted), drawn with a colour bar (``RdBu_r`` centred on 0 if any value is negative, else ``viridis``).
    size : {"pt", "energy"} or array_like
        Marker areas proportional to pT, energy or the absolute given values (as a fraction of their sum).
    show_prongs : bool
        Mark the decay quarks (``truth_prong_deta``, ``truth_prong_dphi``) with crosses, if present.
    show_axes : int, optional
        Mark the ``show_axes`` exclusive-kt WTA axes of :func:`itj.observables.subjet_axes`.
    lim : float
        Half-width of the plotted window.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (default: a new figure).
    colorbar : bool
        Draw a colour bar for array colours.
    cmap : str, optional
        Colour map for array colours.
    label : str, optional
        Colour-bar label for array colours (default "value").
    legend : bool
        Draw the legend of the categorical colours and markers.
    title : str, optional
        Title (default: index, process, pT, mass and multiplicity).

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    ax = _new_axes(ax)
    sc, handles = _draw_jet(
        ax, data, i, color=color, size=size, show_prongs=show_prongs, show_axes=show_axes, cmap=cmap
    )
    _decorate_display(ax, lim)
    if not isinstance(color, str) and colorbar:
        ax.figure.colorbar(sc, ax=ax, label=label or "value", shrink=0.85)
    if legend and handles:
        ax.legend(handles=list(handles.values()), loc="upper right", fontsize="small", framealpha=0.8)
    ax.set_title(title if title is not None else _title(data, i), fontsize="medium")
    return ax


def _ladder_color(data: JetData, color: str) -> str:
    """Fall back to the particle type when a level has no truth column for ``color``."""
    needs = {"prong": "truth_part_prong", "parton": "truth_part_parton", "ue": "truth_part_is_ue"}
    return color if color not in needs or needs[color] in data else "type"


def ladder(
    jets: Mapping[str, JetData],
    *,
    color: ColorBy = "prong",
    i: int = 0,
    lim: float = 0.8,
    show_prongs: bool = True,
) -> Figure:
    """Draw the same jet at every level of the ladder side by side (``itj.toy.show_ladder`` gives the input).

    Parameters
    ----------
    jets : Mapping of str to JetData
        Level name (for example ``"partons"``, ``"shower"``, ``"hadrons"``, ``"detector"``) to the data set of that
        level, in drawing order.
    color : {"prong", "type", "parton", "ue"}
        Colouring of :func:`jet_display`; levels without the needed truth column are coloured by type.
    i : int
        Index of the jet in every data set.
    lim : float
        Half-width of every window.
    show_prongs : bool
        Mark the decay quarks.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per level and a shared legend.
    """
    if not jets:
        raise ValueError("jets is empty: pass {level name: JetData}")
    plt = _pyplot()
    n = len(jets)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.9), layout="constrained", squeeze=False)
    handles: dict[str, Any] = {}
    for ax, (name, data) in zip(axes[0], jets.items(), strict=True):
        _, h = _draw_jet(
            ax, data, i, color=_ladder_color(data, color), size="pt", show_prongs=show_prongs, show_axes=None
        )
        handles.update(h)
        _decorate_display(ax, lim)
        ax.set_title(f"{name}: N = {int(data.counts[i])}", fontsize="medium")
        if ax is not axes[0, 0]:
            ax.set_ylabel("")
    first = next(iter(jets.values()))
    fig.suptitle(_title(first, i).split(", N =")[0], fontsize="medium")
    if handles:
        fig.legend(
            handles=list(handles.values()), loc="outside lower center", ncol=min(len(handles), 6), fontsize="small"
        )
    return fig


def animate_ladder(
    jets: Mapping[str, JetData],
    path: str | os.PathLike,
    *,
    color: ColorBy = "prong",
    i: int = 0,
    lim: float = 0.8,
    hold: float = 1.4,
    fade: float = 0.6,
    fps: int = 12,
    width: int = 560,
) -> Path:
    """Write a looping GIF of one jet climbing the ladder: each level is held, then cross-faded into the next.

    Needs pillow (``analysis`` extra). The frames share one colour palette (no dithering), so the file stays small
    (well below 1 MB at the default size).

    Parameters
    ----------
    jets : Mapping of str to JetData
        Level name to data set, in order (as for :func:`ladder`).
    path : str or os.PathLike
        Output ``.gif`` file (parent directories are created).
    color : {"prong", "type", "parton", "ue"}
        Colouring of every level (with the fallback of :func:`ladder`).
    i : int
        Index of the jet.
    lim : float
        Half-width of the window.
    hold, fade : float
        Seconds each level is shown and seconds of each cross-fade.
    fps : int
        Frames per second of the fades.
    width : int
        Width of the GIF in pixels.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    from PIL import Image

    if not jets:
        raise ValueError("jets is empty: pass {level name: JetData}")
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, width / dpi * 0.92), dpi=dpi, layout="constrained")
    names = list(jets)
    first = jets[names[0]]

    def draw(levels: Sequence[tuple[str, float]]) -> Image.Image:
        ax.clear()
        handles: dict[str, Any] = {}
        for name, alpha in levels:
            data = jets[name]
            _, h = _draw_jet(
                ax, data, i, color=_ladder_color(data, color), size="pt", show_prongs=True, show_axes=None,
                alpha=0.85 * alpha,
            )  # fmt: skip
            if alpha >= 0.5:
                handles = h
        _decorate_display(ax, lim)
        label = " -> ".join(f"[{n}]" if n == max(levels, key=lambda t: t[1])[0] else n for n in names)
        ax.set_title(f"{_title(first, i).split(', N =')[0]}\n{label}", fontsize="small")
        if handles:
            ax.legend(handles=list(handles.values()), loc="upper right", fontsize="x-small", framealpha=0.8)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        return Image.fromarray(np.ascontiguousarray(rgba[..., :3]))

    frames: list[Image.Image] = []
    durations: list[int] = []
    n_fade = max(round(fade * fps), 0)
    for k, name in enumerate(names):
        frames.append(draw([(name, 1.0)]))
        durations.append(round(1000 * hold))
        nxt = names[(k + 1) % len(names)]
        if len(names) > 1:
            for f in range(1, n_fade + 1):
                t = f / (n_fade + 1)
                frames.append(draw([(name, 1.0 - t), (nxt, t)]))
                durations.append(round(1000 / fps))
    plt.close(fig)
    montage = Image.new("RGB", (frames[0].width, frames[0].height * min(len(frames), 6)))
    picks = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), 6)).round().astype(int))
    for row, k in enumerate(picks):
        montage.paste(frames[k], (0, row * frames[0].height))
    shared = montage.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantised = [im.quantize(palette=shared, dither=Image.Dither.NONE) for im in frames]
    quantised[0].save(path, save_all=True, append_images=quantised[1:], duration=durations, loop=0, disposal=1)
    return path


# ----------------------------------------------------------------------------------------------------------------------
# Images and planes


def jet_image(
    data: JetData,
    *,
    bins: int = 33,
    extent: float = 0.8,
    average: bool = True,
    ax: Axes | None = None,
    log: bool = True,
) -> Axes:
    """Draw the jet image: the pT fraction ``z_i = pT_i / sum pT`` per (Delta eta, Delta phi) pixel.

    Parameters
    ----------
    data : JetData
        Jets; every jet is drawn in its own frame (``part_deta``, ``part_dphi``).
    bins : int
        Pixels per axis.
    extent : float
        Half-width of the image.
    average : bool
        Average over the jets (True) or sum them (False).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    log : bool
        Logarithmic colour scale.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    from matplotlib.colors import LogNorm

    deta = np.asarray(data["part_deta"].values, dtype=np.float64)  # type: ignore[union-attr]
    dphi = np.asarray(data["part_dphi"].values, dtype=np.float64)  # type: ignore[union-attr]
    px = data["part_px"]
    py = data["part_py"]
    pt = np.hypot(np.asarray(px.values, dtype=np.float64), np.asarray(py.values, dtype=np.float64))  # type: ignore[union-attr]
    row = px.row_index()  # type: ignore[union-attr]
    total = np.bincount(row, weights=pt, minlength=len(data))
    z = pt / np.where(total[row] > 0, total[row], 1.0)
    edges = np.linspace(-extent, extent, bins + 1)
    image, _, _ = np.histogram2d(deta, dphi, bins=(edges, edges), weights=z)
    if average and len(data):
        image /= len(data)
    ax = _new_axes(ax)
    positive = image[image > 0]
    norm = (
        LogNorm(vmin=max(float(np.min(positive)), float(np.max(positive)) * 1e-5), vmax=float(np.max(positive)))
        if (log and positive.size)
        else None
    )
    im = ax.imshow(
        np.where(image > 0, image, np.nan).T if log else image.T,
        origin="lower",
        extent=(-extent, extent, -extent, extent),
        cmap="viridis",
        norm=norm,
        interpolation="nearest",
    )
    ax.figure.colorbar(im, ax=ax, label=r"$\langle z \rangle$ per pixel" if average else r"$\sum z$ per pixel")
    ax.set_aspect("equal")
    ax.set_xlabel(_XLABEL)
    ax.set_ylabel(_YLABEL)
    ax.set_title(f"jet image, {len(data)} jets", fontsize="medium")
    return ax


def lund_plane(
    density: ArrayLike,
    x_edges: ArrayLike,
    y_edges: ArrayLike,
    *,
    ax: Axes | None = None,
    vmax: float | None = None,
    kt_cut: float | None = None,
    n_jets: int | None = None,
    label: str = r"$\rho$ (emissions per jet per unit area)",
) -> Axes:
    """Draw a primary Lund-plane density (``itj.observables.lund_density``): ln(1/Delta) across, ln(kt/GeV) up.

    Parameters
    ----------
    density : array_like
        (len(x_edges) - 1, len(y_edges) - 1) density per jet per unit area.
    x_edges, y_edges : array_like
        Bin edges in ln(1/Delta) and ln(kt/GeV).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    vmax : float, optional
        Top of the colour scale (default: the maximum).
    kt_cut : float, optional
        Draw the shower cutoff as a horizontal line at ln(kt_cut / GeV).
    n_jets : int, optional
        Number of jets, written in the title.
    label : str
        Colour-bar label.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    rho = np.asarray(density, dtype=np.float64)
    x = np.asarray(x_edges, dtype=np.float64)
    y = np.asarray(y_edges, dtype=np.float64)
    if rho.shape != (len(x) - 1, len(y) - 1):
        raise ValueError(f"density has shape {rho.shape}, expected {(len(x) - 1, len(y) - 1)} from the edges")
    ax = _new_axes(ax)
    mesh = ax.pcolormesh(x, y, rho.T, cmap="viridis", vmin=0.0, vmax=vmax, shading="flat")
    ax.figure.colorbar(mesh, ax=ax, label=label)
    if kt_cut is not None and kt_cut > 0:
        ax.axhline(math.log(kt_cut), color="white", ls="--", lw=1.0, label=f"cutoff $k_t$ = {kt_cut:g} GeV")
        ax.legend(loc="upper right", fontsize="small")
    ax.set_xlabel(r"$\ln(1/\Delta)$")
    ax.set_ylabel(r"$\ln(k_t/\mathrm{GeV})$")
    ax.set_title("primary Lund plane" + (f", {n_jets} jets" if n_jets is not None else ""), fontsize="medium")
    return ax


# ----------------------------------------------------------------------------------------------------------------------
# Performance curves


def _rejection_at(eff_s: np.ndarray, eff_b: np.ndarray, wp: float) -> float:
    order = np.argsort(eff_s, kind="stable")
    eb = float(np.interp(wp, eff_s[order], eff_b[order]))
    return 1.0 / eb if eb > 0 else float("inf")


def roc_curves(
    curves: Mapping[str, tuple[ArrayLike, ArrayLike]],
    *,
    ax: Axes | None = None,
    working_point: float | None = 0.5,
    title: str | None = None,
) -> Axes:
    """Draw ROC curves as background rejection 1/eps_B (log axis) against signal efficiency eps_S.

    Parameters
    ----------
    curves : Mapping of str to (eff_s, eff_b)
        Name to the efficiencies of :func:`itj.metrics.roc`; the legend gives the rejection at the working point.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    working_point : float, optional
        Signal efficiency marked with a dotted line (None: no mark).
    title : str, optional
        Title, for example the pair of classes and the test-set size.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    ax = _new_axes(ax)
    colours = list(_CYCLE)
    for k, (name, (es, eb)) in enumerate(curves.items()):
        eff_s = np.asarray(es, dtype=np.float64)
        eff_b = np.asarray(eb, dtype=np.float64)
        keep = (eff_b > 0) & (eff_s > 0)
        label = name
        if working_point is not None:
            rej = _rejection_at(eff_s, eff_b, working_point)
            label = f"{name} ($R_{{{100 * working_point:.0f}}}$ = {rej:.3g})"
        ax.plot(eff_s[keep], 1.0 / eff_b[keep], color=colours[k % len(colours)], lw=1.5, label=label)
    if working_point is not None:
        ax.axvline(working_point, color="0.5", ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"signal efficiency $\varepsilon_S$")
    ax.set_ylabel(r"background rejection $1/\varepsilon_B$")
    if curves:
        ax.legend(fontsize="small")
    if title:
        ax.set_title(title, fontsize="medium")
    return ax


def curves(
    x: ArrayLike,
    ys: Mapping[str, ArrayLike | tuple[ArrayLike, ArrayLike]],
    *,
    xlabel: str,
    ylabel: str,
    logx: bool = False,
    logy: bool = False,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Draw one or more curves against a common x (IRC scans, learning curves, deletion curves).

    Parameters
    ----------
    x : array_like
        Common x values.
    ys : Mapping of str to array_like or (mean, std)
        Name to the y values, or to a pair (mean, standard deviation) drawn with a shaded band.
    xlabel, ylabel : str
        Axis labels.
    logx, logy : bool
        Logarithmic axes.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    title : str, optional
        Title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    ax = _new_axes(ax)
    xv = np.asarray(x, dtype=np.float64)
    for k, (name, y) in enumerate(ys.items()):
        colour = _CYCLE[k % len(_CYCLE)]
        if isinstance(y, tuple):
            mean = np.asarray(y[0], dtype=np.float64)
            std = np.asarray(y[1], dtype=np.float64)
            ax.fill_between(xv, mean - std, mean + std, color=colour, alpha=0.2, lw=0)
        else:
            mean = np.asarray(y, dtype=np.float64)
        if mean.shape != xv.shape:
            raise ValueError(f"{name!r} has shape {mean.shape}, x has shape {xv.shape}")
        ax.plot(xv, mean, marker="o", ms=3.5, color=colour, lw=1.4, label=name)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ys:
        ax.legend(fontsize="small")
    if title:
        ax.set_title(title, fontsize="medium")
    return ax


def history(h: Any, *, ax: Axes | None = None) -> Axes:
    """Draw the losses (solid, left axis) and accuracies (dashed, right axis) of an ``itj.train.History``.

    Parameters
    ----------
    h : itj.train.History
        Training history (``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``, ``best_epoch``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes of the losses (the accuracies are on its twin).
    """
    ax = _new_axes(ax)
    blue, orange = palette["blue"], palette["orange"]
    for values, colour, label in ((h.train_loss, blue, "train loss"), (h.val_loss, orange, "validation loss")):
        if len(values):
            ax.plot(np.arange(1, len(values) + 1), values, color=colour, marker="o", ms=3, label=label)
    twin = ax.twinx()
    for values, colour, label in ((h.train_acc, blue, "train accuracy"), (h.val_acc, orange, "validation accuracy")):
        if len(values):
            twin.plot(np.arange(1, len(values) + 1), values, color=colour, ls="--", marker="s", ms=3, label=label)
    best = getattr(h, "best_epoch", None)
    if best is not None and best >= 0 and len(h.val_loss):
        ax.axvline(best + 1, color="0.5", ls=":", lw=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    twin.set_ylabel("accuracy")
    lines = ax.get_lines() + twin.get_lines()
    labelled = [line for line in lines if not str(line.get_label()).startswith("_")]
    if labelled:
        ax.legend(handles=labelled, fontsize="small", loc="center right")
    return ax


# ----------------------------------------------------------------------------------------------------------------------
# Interpretability


def attention_matrix(
    A: ArrayLike,
    *,
    order: ArrayLike | None = None,
    groups: ArrayLike | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Draw an attention matrix (rows: queries, columns: keys) with optional reordering and group boundaries.

    Parameters
    ----------
    A : array_like
        (P, P) attention weights of one jet and head (a torch tensor is accepted); padded rows and columns should
        be cut off by the caller.
    order : array_like, optional
        Permutation of the particles; default: sorted by ``groups`` (stable) when given, else as stored.
    groups : array_like, optional
        Group of every particle (prong or subjet index, -1 for none); boundaries between groups are drawn.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    title : str, optional
        Title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    a = _numpy(A).astype(np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"A must be a square (P, P) matrix, got shape {a.shape}")
    p = a.shape[0]
    g = None if groups is None else _numpy(groups).astype(np.int64).reshape(-1)[:p]
    if order is None:
        perm = np.argsort(g, kind="stable") if g is not None else np.arange(p)
    else:
        perm = _numpy(order).astype(np.int64).reshape(-1)
    a = a[np.ix_(perm, perm)]
    ax = _new_axes(ax)
    im = ax.imshow(a, cmap="viridis", vmin=0.0, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, label="attention weight")
    if g is not None:
        gs = g[perm]
        for b in np.flatnonzero(np.diff(gs)) + 0.5:
            ax.axhline(b, color="white", lw=0.8)
            ax.axvline(b, color="white", lw=0.8)
    ax.set_xlabel("key particle")
    ax.set_ylabel("query particle")
    if title:
        ax.set_title(title, fontsize="medium")
    return ax


def attribution_display(
    data: JetData,
    i: int,
    attributions: ArrayLike,
    *,
    ax: Axes | None = None,
    absolute: bool = False,
    lim: float = 0.8,
) -> Axes:
    """Draw jet ``i`` coloured by per-particle attributions (signed: ``RdBu_r`` centred on 0).

    Parameters
    ----------
    data : JetData
        Jets.
    i : int
        Index of the jet.
    attributions : array_like
        The attributions of this jet (P,) or of all jets (N, P) (row ``i`` is taken); padded slots are ignored.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    absolute : bool
        Colour by the absolute attribution (and say so on the colour bar).
    lim : float
        Half-width of the window.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    a = _numpy(attributions).astype(np.float64)
    if a.ndim == 2:
        a = a[i]
    if absolute:
        a = np.abs(a)
    signed = bool(np.any(a < 0))
    label = "attribution (signed)" if signed else ("attribution (absolute)" if absolute else "attribution")
    return jet_display(data, i, color=a, ax=ax, label=label, lim=lim)


def efn_filters(model: Any, *, n: int = 8, extent: float = 0.8, bins: int = 64) -> Figure:
    """Draw the ``n`` most active EFN filters Phi_a(Delta eta, Delta phi) (from ``EFN.filters``).

    The filters are ranked by their integral of ``|Phi_a|`` over the window. Every panel has its own colour bar
    (signed values on a diverging map, non-negative ones on viridis); a filter that is constant over the window, for
    example a ReLU unit that died in training, is marked "dead" in its title.

    Parameters
    ----------
    model : itj.models.EFN
        A trained EFN.
    n : int
        Number of filters to draw.
    extent : float
        Half-width of the window.
    bins : int
        Grid points per axis.

    Returns
    -------
    matplotlib.figure.Figure
        A grid of panels.
    """
    filters = _numpy(model.filters(extent=extent, bins=bins)).astype(np.float64)  # (latent, bins, bins)
    if filters.ndim != 3:
        raise ValueError(f"model.filters() returned shape {filters.shape}, expected (latent, bins, bins)")
    rank = np.argsort(-np.abs(filters).sum(axis=(1, 2)), kind="stable")[: max(int(n), 1)]
    ncols = min(len(rank), 4)
    nrows = math.ceil(len(rank) / ncols)
    plt = _pyplot()
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows), layout="constrained", squeeze=False)
    for ax in axes.flat[len(rank) :]:
        ax.set_visible(False)
    for ax, k in zip(axes.flat, rank, strict=False):
        f = filters[k]
        signed = bool(np.any(f < 0))
        vmax = float(np.max(np.abs(f))) or 1.0
        image = ax.imshow(
            f.T,
            origin="lower",
            extent=(-extent, extent, -extent, extent),
            cmap="RdBu_r" if signed else "viridis",
            vmin=-vmax if signed else 0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        fig.colorbar(image, ax=ax, shrink=0.8, label=r"$\Phi_a$" + (" (signed)" if signed else ""))
        dead = float(np.ptp(f)) <= 1e-6 * max(vmax, 1e-12)
        ax.set_title(f"filter {k}" + (" (dead: constant)" if dead else ""), fontsize="small")
        ax.set_aspect("equal")
        ax.set_xlabel(_XLABEL)
        ax.set_ylabel(_YLABEL)
    return fig


def cka_heatmap(
    M: ArrayLike,
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    *,
    ax: Axes | None = None,
    annotate: bool = True,
    title: str | None = None,
) -> Axes:
    """Draw a CKA matrix (``itj.probes.cka_matrix``) on a fixed 0-1 colour scale, with the values written in.

    Parameters
    ----------
    M : array_like
        (len(labels_a), len(labels_b)) CKA values.
    labels_a, labels_b : sequence of str
        Row and column names.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    annotate : bool
        Write the values into the cells.
    title : str, optional
        Title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    m = _numpy(M).astype(np.float64)
    if m.shape != (len(labels_a), len(labels_b)):
        raise ValueError(f"M has shape {m.shape}, expected {(len(labels_a), len(labels_b))} from the labels")
    ax = _new_axes(ax, figsize=(1.0 + 0.6 * len(labels_b) + 1.5, 1.0 + 0.5 * len(labels_a)))
    im = ax.imshow(m, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax, label="linear CKA")
    ax.set_xticks(range(len(labels_b)), labels=list(labels_b), rotation=45, ha="right")
    ax.set_yticks(range(len(labels_a)), labels=list(labels_a))
    if annotate:
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                v = m[r, c]
                ax.text(
                    c,
                    r,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize="x-small",
                    color="black" if v > 0.6 else "white",
                )
    if title:
        ax.set_title(title, fontsize="medium")
    return ax
