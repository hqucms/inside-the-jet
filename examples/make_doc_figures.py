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
FIGURES = ("ladder", "classes", "lund_plane", "toy_vs_jetclass")
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
        else:
            out = toy_vs_jetclass_figure(args.out_dir, args)
        for path in out:
            print(f"{path} ({path.stat().st_size / 1e3:.0f} kB, {time.perf_counter() - t0:.0f} s)")
        written += out
    print(f"{len(written)} figures in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
