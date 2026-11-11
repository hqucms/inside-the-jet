"""Quick start: generate jets with the toy "A Spray of Hadrons", save them, read them back and summarise them.

Run from the repository root; it takes seconds on a CPU::

    python examples/quickstart.py
    python examples/quickstart.py --jets 6000 --preset hadrons --seed 2 --workers 2

It writes ``data/quickstart_<preset>_n<jets>_seed<seed>.parquet`` and, if matplotlib is installed, a display of the
first jet of each process next to it (``.png``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from itj.toy import PRESETS, PROCESSES, JetData, check_identities, generate, load, preset, save


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Generate, save and read back a small data set of the jet toy.")
    parser.add_argument("--jets", type=int, default=3000, help="number of jets (default: 3000)")
    parser.add_argument(
        "--preset", choices=list(PRESETS), default="detector", help="physics preset (default: detector)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--workers", default="1", help="worker processes, or 'auto' (default: 1)")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory (default: data)")
    return parser.parse_args()


def summarise(data: JetData) -> None:
    """Print the jets per process, the mean multiplicity, the soft-drop mass and the identities of the toy."""
    names = data.process_names
    n = np.asarray(data["jet_nparticles"])
    msd = np.asarray(data["jet_sdmass"])
    tau1, tau2 = np.asarray(data["jet_tau1"]), np.asarray(data["jet_tau2"])
    tau21 = np.divide(tau2, tau1, out=np.zeros_like(tau2), where=tau1 > 0)
    print(f"{'process':<8s} {'jets':>6s} {'<N>':>6s} {'median m_SD':>12s} {'median tau21':>13s}")
    for process in [p for p in PROCESSES if p in set(names.tolist())]:
        sel = names == process
        print(
            f"{process:<8s} {sel.sum():6d} {n[sel].mean():6.1f} {np.median(msd[sel]):8.1f} GeV "
            f"{np.median(tau21[sel]):13.3f}"
        )
    if data.has_truth:
        checks = check_identities(data)
        print(f"exact identities of the toy: {sum(checks.values())} of {len(checks)} hold")


def plot_jets(data: JetData, path: Path) -> None:
    """Save a display of the first jet of each process, coloured by the prong of every particle."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from itj import plot

    names = data.process_names
    processes = [p for p in PROCESSES if p in set(names.tolist())]
    fig, axes = plt.subplots(1, len(processes), figsize=(4.6 * len(processes), 4.4), constrained_layout=True)
    for ax, process in zip(np.atleast_1d(axes), processes, strict=True):
        i = int(np.flatnonzero(names == process)[0])
        plot.jet_display(data, i, color="prong" if data.has_truth else "type", ax=ax)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Generate a small data set, save it, read it back, print a summary and draw one jet per process."""
    args = parse_args()
    workers: int | str = args.workers if args.workers == "auto" else int(args.workers)
    config = preset(args.preset)
    print(f"preset {args.preset}:\n{config.describe()}\n")

    # Generate: QCD, W and top jets in equal numbers, the same events for the same seed at every preset.
    start = time.perf_counter()
    data = generate(args.jets, config=config, seed=args.seed, workers=workers)  # type: ignore[arg-type]
    print(f"{data}\ngenerated in {time.perf_counter() - start:.1f} s")
    print("containment efficiency:", data.provenance["containment_efficiency"] if data.provenance else None)

    # Save and read back: the file holds the JetClass branches, the truth record, the configuration and provenance.
    path = save(data, args.out / f"quickstart_{args.preset}_n{args.jets}_seed{args.seed}.parquet", overwrite=True)
    again = load(path)
    same = all(np.array_equal(np.asarray(again[k]), np.asarray(data[k])) for k in ("jet_pt", "jet_sdmass"))
    print(f"saved to {path} ({path.stat().st_size / 1e6:.2f} MB) and read back: {again} (same jets: {same})\n")
    summarise(again)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed: no figure (pip install -e '.[analysis]')")
        return
    figure = path.with_suffix(".png")
    plot_jets(again, figure)
    print(f"figure: {figure}")
    print(f"done in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
