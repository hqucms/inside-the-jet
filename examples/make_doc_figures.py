"""Make the figures of the documentation (``docs/img/``) from the toy and, if present, the JetClass files.

Run from the repository root (needs the ``analysis`` extra: matplotlib and pillow)::

    python examples/make_doc_figures.py                     # every figure
    python examples/make_doc_figures.py --only ladder       # ladder.gif and ladder.png only

It writes

* ``ladder.gif``: the hero of the README, one top jet at the four presets (partons, shower, hadrons, detector), held
  and cross-faded (at most 2.9 MB);
* ``ladder.png``: the same four levels side by side (docs/physics.md);
* ``classes.png``: one QCD, one W and one top jet at the detector level, coloured by prong;
* ``lund_plane.png``: the primary Lund density of quark and gluon jets of 600 GeV at the shower level with a fixed
  coupling (the shower's own emissions, ``truth_lund_*``);
* ``toy_vs_jetclass.png``: the soft-drop mass, tau21, tau32 and the multiplicity of the toy at the detector level
  (with the JetClass selection 500 < pT < 1000 GeV) against the JetClass validation files; skipped with a message
  when the files of ``jetgen fetch jetclass --split val`` are missing.
* ``primer_coordinates.png``: the detector cylinder with the beam axis z, the polar angle theta, the azimuth phi and
  lines of constant pseudorapidity eta (docs/primer.md, section 3);
* ``primer_boosted.png``: a W decay into two quarks in the W rest frame and after a boost along x, with the opening
  angle of the quarks (docs/primer.md, section 6);
* ``primer_lund.png``: one gluon jet at the shower level and its primary Lund plane, found by declustering its
  partons, with the shower's own primary emissions on top (docs/primer.md, section 7).

Everything is generated with fixed seeds, so the figures are reproducible; all of them take about a minute with 4
workers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    "ladder",
    "classes",
    "lund_plane",
    "toy_vs_jetclass",
    "primer_coordinates",
    "primer_boosted",
    "primer_lund",
)
MAX_GIF_BYTES = int(2.9e6)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Make the figures of docs/img.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "img", help="output folder (default: docs/img)")
    parser.add_argument("--only", action="append", choices=FIGURES, help="make only this figure (repeatable)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the ladder jet (default: 1)")
    parser.add_argument("--jet", type=int, default=0, help="index of the ladder jet (default: 0)")
    parser.add_argument("--jets", type=int, default=30_000, help="toy jets for the comparison with JetClass")
    parser.add_argument("--lund-jets", type=int, default=20_000, help="quark and gluon jets for the Lund plane")
    parser.add_argument(
        "--jetclass", type=Path, default=ROOT / "data" / "jetclass", help="JetClass folder (default: data/jetclass)"
    )
    parser.add_argument("--jetclass-jets", type=int, default=20_000, help="JetClass jets per class (default: 20000)")
    parser.add_argument("--workers", default="auto", help="worker processes of the generation (default: auto)")
    return parser.parse_args(argv)


def ladder_figures(out_dir: Path, seed: int, jet: int) -> list[Path]:
    """Write ladder.gif and ladder.png: one top jet at the four presets."""
    import matplotlib.pyplot as plt

    from itj import plot
    from itj.toy import show_ladder

    jets = show_ladder("tbqq", seed=seed, jet=jet)
    gif = plot.animate_ladder(jets, out_dir / "ladder.gif", width=560)
    size = gif.stat().st_size
    if size > MAX_GIF_BYTES:
        raise RuntimeError(f"{gif} has {size / 1e6:.1f} MB, more than the 2.9 MB of the pre-commit size check")
    fig = plot.ladder(jets)
    png = out_dir / "ladder.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return [gif, png]


def classes_figure(out_dir: Path, seed: int) -> list[Path]:
    """Write classes.png: one QCD, one W and one top jet at the detector level, coloured by prong."""
    import matplotlib.pyplot as plt

    from itj import plot
    from itj.toy import generate

    data = generate(90, seed=seed)
    names = data.process_names
    msd, n = np.asarray(data["jet_sdmass"]), np.asarray(data["jet_nparticles"])
    dr_min = np.asarray(data["truth_dr_min"])
    energy = data["truth_prong_energy"]
    balance = np.array([e.min() / e.max() if len(e) else 0.0 for e in (energy[i] for i in range(len(data)))])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained")
    for ax, process in zip(axes, ("qcd", "wqq", "tbqq"), strict=True):
        rows = np.flatnonzero(names == process)
        if process == "qcd":  # a typical QCD jet: mass and multiplicity near the median
            score = np.abs(msd[rows] / np.median(msd[rows]) - 1) + np.abs(n[rows] / np.median(n[rows]) - 1)
            i = int(rows[np.argmin(score)])
        else:  # the first jet whose prongs are resolved and of similar energy
            good = rows[(dr_min[rows] > 0.3) & (balance[rows] > 0.3)]
            i = int(good[0] if len(good) else rows[0])
        plot.jet_display(data, i, color="prong", ax=ax)
    path = out_dir / "classes.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def lund_figure(out_dir: Path, n_jets: int, workers: int | str) -> list[Path]:
    """Write lund_plane.png: the primary Lund density of quark and gluon jets (fixed coupling, 600 GeV)."""
    import matplotlib.pyplot as plt

    from itj import plot
    from itj.data import Ragged
    from itj.observables import lund_density
    from itj.toy import generate, lund_ll, preset

    config = preset("shower", alphas_mode="fixed", alphas_value=0.12, pt_spectrum="fixed", pt_fixed=600.0)
    x_edges, y_edges = np.linspace(0.0, 7.0, 36), np.linspace(-1.0, 6.5, 31)
    densities = {}
    for process in ("q", "g"):
        data = generate(n_jets, config=config, processes=(process,), seed=12, workers=workers)  # type: ignore[arg-type]
        lund: dict[str, Ragged] = {}
        for key in ("ln_inv_delta", "ln_kt", "z"):
            column = data[f"truth_lund_{key}"]
            assert isinstance(column, Ragged)
            lund[key] = column
        densities[process] = lund_density(lund, n_jets=len(data), x_edges=x_edges, y_edges=y_edges)[0]
    vmax = float(max(d.max() for d in densities.values()))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), layout="constrained")
    for ax, (process, density) in zip(axes, densities.items(), strict=True):
        cr, name = (4.0 / 3.0, "quark") if process == "q" else (3.0, "gluon")
        plot.lund_plane(density, x_edges, y_edges, ax=ax, vmax=vmax, kt_cut=1.0, n_jets=n_jets)
        ax.set_title(
            f"{name} jets, 600 GeV, fixed $\\alpha_s$ = 0.12: plateau $2\\alpha_s C_R/\\pi$ = {lund_ll(cr, 0.12):.3f}",
            fontsize="medium",
        )
    path = out_dir / "lund_plane.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def _ratios(data: object) -> dict[str, np.ndarray]:
    """m_SD, tau21, tau32 and the multiplicity of every jet (from the stored branches)."""
    from itj.features import jet_features

    values, _ = jet_features(data, ("jet_sdmass", "jet_nparticles"), derived=("tau21", "tau32"))  # type: ignore[arg-type]
    return dict(zip(("msd", "n", "tau21", "tau32"), values.T.astype(np.float64), strict=True))


def toy_vs_jetclass_figure(out_dir: Path, args: argparse.Namespace) -> list[Path]:
    """Write toy_vs_jetclass.png; skipped (empty list) without the JetClass validation files."""
    import matplotlib.pyplot as plt

    from itj import plot
    from itj.features import jetclass_selection, labels
    from itj.io import load
    from itj.jetclass import files
    from itj.toy import generate

    try:
        paths = files("val", root=args.jetclass)
    except FileNotFoundError as error:
        print(f"toy_vs_jetclass.png skipped: {error}")
        return []
    jetclass = load(paths, max_jets=args.jetclass_jets)
    toy = generate(args.jets, seed=41, workers=args.workers)
    toy = toy.select(np.flatnonzero(jetclass_selection(toy)))
    samples = {"toy": (toy, labels(toy)), "JetClass": (jetclass, labels(jetclass))}
    panels = (
        ("msd", r"$m_\mathrm{SD}$ [GeV]", (0.0, 250.0)),
        ("tau21", r"$\tau_{21}$", (0.0, 1.0)),
        ("tau32", r"$\tau_{32}$", (0.0, 1.0)),
        ("n", "number of particles", (0.0, 140.0)),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.9), layout="constrained")
    for name, (data, y) in samples.items():
        values = _ratios(data)
        style = {"ls": "--", "lw": 1.6} if name == "toy" else {"ls": "-", "lw": 1.3}
        for ax, (key, xlabel, span) in zip(axes, panels, strict=True):
            for k, process in enumerate(("qcd", "wqq", "tbqq")):
                bins = np.arange(span[0], span[1] + 1, 2) if key == "n" else np.linspace(*span, 51)
                ax.hist(
                    values[key][y == k],
                    bins=bins,
                    density=True,
                    histtype="step",
                    color=plot.palette[process],
                    label=f"{process}, {name}",
                    **style,
                )
            ax.set_xlabel(xlabel)
            ax.set_yticks([])
    axes[0].set_ylabel("normalised")
    axes[0].legend(fontsize="x-small", ncol=2, title="solid: JetClass val, dashed: toy", title_fontsize="x-small")
    fig.suptitle(
        f"toy detector level ({len(toy)} jets with 500 < $p_T$ < 1000 GeV) against JetClass "
        f"({len(jetclass)} validation jets)",
        fontsize="medium",
    )
    path = out_dir / "toy_vs_jetclass.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def primer_coordinates_figure(out_dir: Path) -> list[Path]:
    """Write primer_coordinates.png: the detector cylinder with z, theta, phi, pT, and lines of constant eta."""
    import matplotlib.pyplot as plt

    from itj import plot

    grey, red, blue = plot.palette["grey"], plot.palette["vermillion"], plot.palette["blue"]
    green, orange = plot.palette["bluish_green"], plot.palette["orange"]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.4), layout="constrained", height_ratios=(2.0, 1.1))
    ax = axes[0]
    kx, ky = -0.55, -0.4  # oblique projection: z to the right, y up, x towards the viewer (down and left)

    def proj(x: object, y: object, z: object) -> tuple[np.ndarray, np.ndarray]:
        x, y, z = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
        return z + kx * x, y + ky * x

    half = 2.2
    t = np.linspace(0.0, 2.0 * np.pi, 145)
    for zc, style in ((-half, "-"), (0.0, ":"), (half, "-")):  # the two ends and the ring through the collision
        ax.plot(*proj(np.cos(t), np.sin(t), np.full_like(t, zc)), color=grey, lw=0.9, ls=style)
    top = np.arctan2(1.0, ky)  # the outline: the highest and lowest points of a projected ring
    for a in (top, top + np.pi):
        ax.plot(*proj([np.cos(a)] * 2, [np.sin(a)] * 2, [-half, half]), color=grey, lw=0.9)
    ax.annotate(
        "",
        xy=proj(0, 0, half + 1.0),
        xytext=proj(0, 0, -half - 1.0),
        arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.2},
    )
    ax.text(*proj(0, 0, half + 0.6), "z (beam)", va="bottom")
    ax.text(*proj(0, 0, -half - 0.6), "beam", ha="right", va="bottom")
    for vec, name in (((1.6, 0, 0), "x"), ((0, 1.6, 0), "y")):
        ax.annotate("", xy=proj(*vec), xytext=proj(0, 0, 0), arrowprops={"arrowstyle": "->", "color": "k", "lw": 0.8})
        ax.text(*proj(*(1.12 * np.array(vec))), name, ha="center", va="center")
    theta, phi = np.radians(50.0), np.radians(55.0)
    length = 1.0 / np.sin(theta)  # the particle reaches the wall of the cylinder
    tip = length * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    ax.annotate("", xy=proj(*tip), xytext=proj(0, 0, 0), arrowprops={"arrowstyle": "-|>", "color": red, "lw": 2})
    ax.text(*proj(*(1.07 * tip)), r"$\vec p$", color=red, fontsize="large", va="bottom")
    ax.plot(*proj([0, tip[0]], [0, tip[1]], [0, 0]), color=blue, lw=1.8, ls="--")
    ax.plot(*proj([tip[0]] * 2, [tip[1]] * 2, [0, tip[2]]), color=grey, lw=0.8, ls=":")
    ax.text(*proj(tip[0], tip[1], 0), r"$p_T$ ", color=blue, ha="right", va="bottom")
    arc = np.linspace(0.0, phi, 30)
    ax.plot(*proj(0.7 * np.cos(arc), 0.7 * np.sin(arc), 0 * arc), color=green, lw=1.6)
    ax.text(*proj(0.85 * np.cos(phi / 3), 0.85 * np.sin(phi / 3), 0), r"$\phi$", color=green, ha="right", va="center")
    arc = np.linspace(0.0, theta, 30)
    ax.plot(
        *proj(0.6 * np.sin(arc) * np.cos(phi), 0.6 * np.sin(arc) * np.sin(phi), 0.6 * np.cos(arc)), color=orange, lw=1.6
    )
    mid = theta / 2
    ax.text(
        *proj(0.8 * np.sin(mid) * np.cos(phi), 0.8 * np.sin(mid) * np.sin(phi), 0.8 * np.cos(mid)),
        r"$\theta$",
        color=orange,
        ha="left",
        va="center",
    )
    ax.set_xlim(-half - 1.6, half + 1.9)
    ax.set_ylim(-1.5, 1.95)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        r"the detector, a cylinder around the beam: $p_T$ is the part of $\vec p$ across the beam", fontsize="medium"
    )

    ax = axes[1]
    zmax, rmax = 3.0, 1.2
    ax.add_patch(plt.Rectangle((-zmax, 0.0), 2 * zmax, rmax, fill=False, ec=grey, lw=0.8))
    ax.annotate("", xy=(zmax + 0.5, 0), xytext=(-zmax - 0.5, 0), arrowprops={"arrowstyle": "->", "color": "k"})
    ax.text(zmax + 0.5, -0.06, "z (beam)", va="top", ha="right")
    for eta in (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        th = 2.0 * np.arctan(np.exp(-eta))
        dz, dr = np.cos(th), np.sin(th)
        scale = min(rmax / dr, zmax / abs(dz)) if abs(dz) > 1e-9 else rmax / dr
        ax.plot([0, scale * dz], [0, scale * dr], color=red, lw=1.0)
        ha = "center" if abs(eta) < 1.75 else ("left" if eta > 0 else "right")
        ax.text(
            scale * dz + (0.06 * np.sign(eta) if ha != "center" else 0.0),
            scale * dr + 0.03,
            rf"$\eta = {eta:g}$",
            fontsize="small",
            ha=ha,
            va="bottom" if ha == "center" else "center",
        )
    th = 2.0 * np.arctan(np.exp(-1.0))
    arc = np.linspace(0.0, th, 30)
    ax.plot(0.55 * np.cos(arc), 0.55 * np.sin(arc), color=orange, lw=1.5)
    ax.text(0.65 * np.cos(th / 2), 0.65 * np.sin(th / 2), r"$\theta$", color=orange, va="center")
    ax.text(
        0.0,
        -0.22,
        r"$\eta = -\ln\tan(\theta/2)$: equal steps in $\eta$ crowd towards the beam",
        ha="center",
        va="top",
        fontsize="small",
    )
    ax.set_xlim(-zmax - 0.8, zmax + 0.8)
    ax.set_ylim(-0.45, rmax + 0.3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("side view of one half: lines of constant pseudorapidity", fontsize="medium")
    path = out_dir / "primer_coordinates.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def primer_boosted_figure(out_dir: Path) -> list[Path]:
    """Write primer_boosted.png: a W decay at rest and boosted along x, drawn in the plane of the decay."""
    import matplotlib.pyplot as plt

    from itj import plot
    from itj.toy.kinematics import boost, two_body

    m, pt, cos_star = 80.4, 400.0, 0.5  # decay at 60 degrees to the direction of flight
    q1, q2 = two_body(np.array([m]), 0.0, 0.0, np.array([cos_star]), np.array([0.0]))  # direction (cos, phi) of q1
    # cos is measured from z: rotate (x, y, z) -> (z, x, y) so that the flight direction x plays the role of z
    rest = tuple(np.array([p[2], p[0], p[1], p[3]]) for p in (q1[0], q2[0]))
    lab = tuple(boost(p[None, :], np.array([pt / np.hypot(pt, m), 0.0, 0.0]))[0] for p in rest)
    colours = (plot.palette["prong1"], plot.palette["prong0"])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), layout="constrained", width_ratios=(1.0, 1.9))
    ax = axes[0]
    for p, colour, name in zip(rest, colours, ("quark 1", "quark 2"), strict=True):
        ax.annotate("", xy=(p[0], p[1]), xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 2})
        ax.text(1.12 * p[0], 1.12 * p[1], name, color=colour, ha="center", va="center")
    ax.plot(0, 0, "o", color=plot.palette["wqq"], ms=12)
    ax.text(3, -6, "W at rest", color=plot.palette["wqq"])
    ax.axhline(0, color=plot.palette["grey"], lw=0.8, ls="--")
    ax.text(
        48,
        1.5,
        "direction in which\nthe W will move",
        ha="right",
        va="bottom",
        fontsize="small",
        color=plot.palette["grey"],
    )
    arc = np.linspace(0.0, np.arccos(cos_star), 30)
    ax.plot(14 * np.cos(arc), 14 * np.sin(arc), color="k", lw=1)
    ax.text(17 * np.cos(arc[15]), 17 * np.sin(arc[15]), r"$\theta^*$")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$p_x$ [GeV]")
    ax.set_ylabel(r"$p_y$ [GeV]")
    ax.set_title("rest frame: back to back, any direction", fontsize="medium")
    ax = axes[1]
    for p, colour, name in zip(lab, colours, ("quark 1", "quark 2"), strict=True):
        ax.annotate("", xy=(p[0], p[1]), xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 2})
        ax.text(p[0] + 8, p[1], name, color=colour, va="center")
    w = lab[0] + lab[1]
    ax.annotate(
        "",
        xy=(w[0], w[1]),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": plot.palette["wqq"], "lw": 1.2, "ls": "--"},
    )
    ax.text(w[0] + 8, w[1], r"W ($\vec p_1 + \vec p_2$)", color=plot.palette["wqq"], va="center")
    angles = [np.arctan2(p[1], p[0]) for p in lab]
    arc = np.linspace(min(angles), max(angles), 30)
    ax.plot(120 * np.cos(arc), 120 * np.sin(arc), color="k", lw=1)
    ax.annotate(
        "opening angle",
        xy=(120 * np.cos(arc[20]), 120 * np.sin(arc[20])),
        xytext=(60, 75),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize="small",
    )
    ax.set_xlim(-10, 520)
    ax.set_ylim(-110, 110)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$p_x$ [GeV]")
    ax.set_ylabel(r"$p_y$ [GeV]")
    ax.set_title(f"the same decay, W boosted to $p_T$ = {pt:.0f} GeV along x: a narrow pair", fontsize="medium")
    path = out_dir / "primer_boosted.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def primer_lund_figure(out_dir: Path) -> list[Path]:
    """Write primer_lund.png: one gluon jet at the shower level and the emissions of its primary Lund plane."""
    import matplotlib.pyplot as plt

    from itj import plot
    from itj.observables import lund_primary
    from itj.toy import generate, preset

    config = preset("shower", pt_spectrum="fixed", pt_fixed=800.0)
    data = generate(60, config=config, processes=("g",), seed=22)
    i = 45  # emissions spread over the plane, each found again by the declustering
    jet = data.select(np.array([i]))
    p4, mask = jet.padded_p4()
    lund = lund_primary(p4, mask)
    x, y, z = lund["ln_inv_delta"].values, lund["ln_kt"].values, lund["z"].values
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    plot.jet_display(
        jet,
        0,
        color="prong",
        ax=axes[0],
        legend=False,
        title=f"a gluon jet of {float(jet['jet_pt'][0]):.0f} GeV at the shower level",
    )
    ax = axes[1]
    sc = ax.scatter(
        x,
        y,
        c=z,
        cmap="viridis",
        vmin=0.0,
        vmax=0.5,
        s=60,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
        label="found by declustering the partons",
    )
    fig.colorbar(sc, ax=ax, label=r"$z$ (momentum fraction of the softer branch)", shrink=0.9)
    ax.scatter(
        jet["truth_lund_ln_inv_delta"][0],
        jet["truth_lund_ln_kt"][0],
        s=150,
        facecolors="none",
        edgecolors=plot.palette["vermillion"],
        linewidths=1.2,
        zorder=2,
        label="the shower's own emissions",
    )
    for k, (xk, yk) in enumerate(zip(x, y, strict=True), 1):
        ax.annotate(str(k), (xk, yk), xytext=(6, 5), textcoords="offset points", fontsize="small")
    ax.axhline(0.0, color=plot.palette["grey"], ls="--", lw=1)
    ax.text(
        0.3, -0.08, r"shower cutoff, $k_t$ = 1 GeV", ha="left", va="top", fontsize="small", color=plot.palette["grey"]
    )
    ax.axvline(np.log(1 / 0.8), color=plot.palette["grey"], ls=":", lw=1)
    ax.text(0.27, 4.9, r"jet edge, $\Delta = R$", fontsize="small", color=plot.palette["grey"], va="top")
    ax.annotate("", xy=(4.3, -0.65), xytext=(2.9, -0.65), arrowprops={"arrowstyle": "->"})
    ax.text(2.85, -0.65, "smaller angle (more collinear)", ha="right", va="center", fontsize="small")
    ax.annotate("", xy=(4.15, 4.8), xytext=(4.15, 3.6), arrowprops={"arrowstyle": "->"})
    ax.text(4.05, 4.2, "harder", ha="right", va="center", fontsize="small")
    ax.set_xlim(0.0, 4.5)
    ax.set_ylim(-0.9, 5.0)
    ax.set_xlabel(r"$\ln(1/\Delta)$, $\Delta$ the angle of the emission")
    ax.set_ylabel(r"$\ln(k_t/\mathrm{GeV})$")
    ax.legend(loc="upper center", fontsize="x-small")
    ax.set_title(f"its primary Lund plane: {len(x)} emissions, numbered from the widest", fontsize="medium")
    path = out_dir / "primer_lund.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def main(argv: list[str] | None = None) -> int:
    """Make the figures; return 0."""
    args = parse_args(argv)
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib is not installed: pip install -e '.[analysis]'", file=sys.stderr)
        return 1
    args.workers = args.workers if args.workers == "auto" else int(args.workers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = args.only or list(FIGURES)
    start = time.perf_counter()
    written: list[Path] = []
    for name in selected:
        t0 = time.perf_counter()
        if name == "ladder":
            out = ladder_figures(args.out_dir, args.seed, args.jet)
        elif name == "classes":
            out = classes_figure(args.out_dir, args.seed)
        elif name == "lund_plane":
            out = lund_figure(args.out_dir, args.lund_jets, args.workers)
        elif name == "toy_vs_jetclass":
            out = toy_vs_jetclass_figure(args.out_dir, args)
        elif name == "primer_coordinates":
            out = primer_coordinates_figure(args.out_dir)
        elif name == "primer_boosted":
            out = primer_boosted_figure(args.out_dir)
        else:
            out = primer_lund_figure(args.out_dir)
        for path in out:
            print(f"{path} ({path.stat().st_size / 1e3:.0f} kB, {time.perf_counter() - t0:.0f} s)")
        written += out
    print(f"{len(written)} figures in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
