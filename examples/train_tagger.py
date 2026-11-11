"""Train a jet tagger of :data:`itj.models.MODEL_PRESETS` on the CPU and evaluate it on the test part.

The jets are read with the split of the course (``itj.io.load(..., part=...)``: test = the first 20 % of each file,
val = the next 10 %, train = the rest): the model trains on ``--part`` (the first ``--train-size`` jets of it),
is validated after every epoch on the val part and evaluated once on the test part. Run from the repository root::

    python examples/train_tagger.py --model pfn --data data/t2_detector.parquet --train-size 105000 --epochs 10
    python examples/train_tagger.py --model partsmall --train-size 30000 --epochs 5 --out models/toy/partsmall
    python examples/train_tagger.py --course-models      # the T2 models of the course into models/toy/ (about 35 min)

With ``--out PREFIX`` (default ``runs/<model>``) it writes ``PREFIX.pt`` (the model with its configuration, read
back with ``itj.models.load_model``), ``PREFIX_history.json`` (loss, accuracy and speed per epoch),
``PREFIX_scores.npz`` (the test probabilities with the labels, ``jet_sdmass`` and ``jet_pt``), ``PREFIX_roc.png``
(if matplotlib is installed) and appends a row to ``results_table.csv`` (``--results``).

By default the script uses half of the cores (``--threads``), so that two trainings run side by side, as T2.4
suggests for the two ParT-small models. ``--course-models`` trains, one after the other with seed 0, the models that
the later tasks use when a student's own models of T2 failed: ``pfn`` and ``efn`` (105 000 jets, 10 epochs),
``pnlite`` (60 000 jets, 5 epochs), ``partsmall`` and ``partsmall_nopair`` (30 000 jets, 5 epochs each).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

#: The T2 models of the course: (preset, training jets, epochs), trained by ``--course-models`` into models/toy/.
COURSE_MODELS: tuple[tuple[str, int, int], ...] = (
    ("pfn", 105_000, 10),
    ("efn", 105_000, 10),
    ("pnlite", 60_000, 5),
    ("partsmall", 30_000, 5),
    ("partsmall_nopair", 30_000, 5),
)

#: Number of classes and the (W, top, QCD) class indices of each label set (None: no W/top columns in the results).
LABEL_SETS: dict[str, tuple[int, tuple[int, int, int] | None]] = {
    "jets3": (3, (1, 2, 0)),
    "jetclass10": (10, (7, 8, 0)),
    "qg": (2, None),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    from itj.models import MODEL_PRESETS

    parser = argparse.ArgumentParser(description="Train a jet tagger on the CPU and evaluate it on the test part.")
    parser.add_argument("--model", choices=list(MODEL_PRESETS), default="pfn", help="model preset (default: pfn)")
    parser.add_argument(
        "--data",
        nargs="+",
        default=["data/t2_detector.parquet"],
        metavar="FILE",
        help="input files (default: data/t2_detector.parquet)",
    )
    parser.add_argument(
        "--part", choices=("train", "val", "test"), default="train", help="part of the split to train on"
    )
    parser.add_argument("--feature-set", choices=("kin", "kinpid", "full"), default="kin", help="particle features")
    parser.add_argument("--label-set", choices=list(LABEL_SETS), default="jets3", help="classes (default: jets3)")
    parser.add_argument("--train-size", type=int, default=None, help="training jets per file (default: all)")
    parser.add_argument("--val-size", type=int, default=None, help="validation jets per file (default: all)")
    parser.add_argument("--test-size", type=int, default=None, help="test jets per file (default: all)")
    parser.add_argument("--epochs", type=int, default=10, help="epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the initial weights and batches (default: 0)")
    parser.add_argument(
        "--threads", type=int, default=None, help="torch threads (default: half of the cores; all for --course-models)"
    )
    parser.add_argument("--max-minutes", type=float, default=None, help="stop the training after this many minutes")
    parser.add_argument(
        "--out", type=Path, default=None, help="output prefix (default: runs/<model>; a folder for --course-models)"
    )
    parser.add_argument("--results", type=Path, default=Path("results_table.csv"), help="results table to append to")
    parser.add_argument("--note", default="", help="note for the results row")
    parser.add_argument(
        "--course-models",
        action="store_true",
        help="train the T2 models of the course into models/toy/ (--train-size and --epochs then cap their sizes)",
    )
    return parser.parse_args(argv)


def _metrics(probs: np.ndarray, y: np.ndarray, label_set: str) -> dict[str, float]:
    """Return the accuracy and the one-vs-one AUC and rejections (W and top against QCD) of the test scores.

    Every AUC and rejection comes with its bootstrap error (100 resamplings) as ``<name>_err``.
    """
    from itj.metrics import accuracy, auc, bootstrap, one_vs_one, rejection

    out = {"accuracy": accuracy(probs, y)}
    classes = LABEL_SETS[label_set][1]
    if classes is None:  # quark (0) against gluon (1)
        out["auc"] = auc(y == 0, probs[:, 0])
        return out
    w, top, qcd = classes
    yw, sw, _ = one_vs_one(probs, y, w, qcd)
    yt, st, _ = one_vs_one(probs, y, top, qcd)

    def r50(yy: np.ndarray, ss: np.ndarray) -> float:
        return rejection(yy, ss, 0.5)

    def r80(yy: np.ndarray, ss: np.ndarray) -> float:
        return rejection(yy, ss, 0.8)

    for name, metric, (yy, ss) in (
        ("auc_w", auc, (yw, sw)),
        ("r50_w", r50, (yw, sw)),
        ("auc_top", auc, (yt, st)),
        ("r50_top", r50, (yt, st)),
        ("r80_top", r80, (yt, st)),
    ):
        out[name], out[name + "_err"] = bootstrap(metric, yy, ss)
    return out


def _roc_figure(probs: np.ndarray, y: np.ndarray, label_set: str, title: str, path: Path) -> Path | None:
    """Save the ROC curves (W and top against QCD) of the test scores; None without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    from itj import plot
    from itj.metrics import one_vs_one, roc

    classes = LABEL_SETS[label_set][1]
    curves = {}
    if classes is None:
        eff_s, eff_b, _ = roc(y == 0, probs[:, 0])
        curves["quark vs gluon"] = (eff_s, eff_b)
    else:
        w, top, qcd = classes
        for name, signal in (("W vs QCD", w), ("top vs QCD", top)):
            ys, ss, _ = one_vs_one(probs, y, signal, qcd)
            eff_s, eff_b, _ = roc(ys, ss)
            curves[name] = (eff_s, eff_b)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    plot.roc_curves(curves, ax=ax, title=title)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def train_one(
    model_name: str,
    *,
    data: list[str],
    part: str,
    feature_set: str,
    label_set: str,
    train_size: int | None,
    val_size: int | None,
    test_size: int | None,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    threads: int,
    max_minutes: float | None,
    out: Path,
    results: Path,
    note: str = "",
) -> dict[str, float]:
    """Train one model, save it and its outputs, append a results row and return the test metrics."""
    from itj.features import JetDataset, labels
    from itj.io import load
    from itj.metrics import append_result
    from itj.models import build_model, save_model
    from itj.train import fit, predict, set_seed, set_threads

    set_threads(threads)
    set_seed(seed)
    n_classes = LABEL_SETS[label_set][0]
    model = build_model(model_name, feature_set=feature_set, n_classes=n_classes)
    jet_features = getattr(model, "jet_features", None)
    columns_note = f"{model_name} ({model.n_parameters():,} parameters), features {model.feature_set or jet_features}"
    print(f"{columns_note}; {threads} threads", flush=True)

    train = load(data, part=part, max_jets=train_size)  # type: ignore[arg-type]
    val = load(data, part="val", max_jets=val_size) if part != "val" else None
    test = load(data, part="test", max_jets=test_size) if part != "test" else load(data, part="val", max_jets=test_size)
    print(f"train {len(train)} jets ({part}), val {0 if val is None else len(val)}, test {len(test)}", flush=True)

    def dataset(d: object) -> JetDataset:
        return JetDataset(
            d,  # type: ignore[arg-type]
            model.feature_set,
            label_set,
            jet_feature_names=None if jet_features is None else tuple(jet_features),
        )

    start = time.perf_counter()
    history = fit(
        model,
        dataset(train),
        None if val is None else dataset(val),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        threads=threads,
        max_minutes=max_minutes,
    )
    train_time = time.perf_counter() - start
    t0 = time.perf_counter()
    probs = predict(model, test)
    eval_rate = len(test) / max(time.perf_counter() - t0, 1e-9)
    y = labels(test, label_set)
    metrics = _metrics(probs, y, label_set)

    out.parent.mkdir(parents=True, exist_ok=True)
    data_name = ",".join(Path(p).stem for p in data)
    extra = {"history": history.to_dict(), "data": data, "part": part, "n_train": len(train), "seed": seed}
    checkpoint = save_model(model, out.with_name(out.name + ".pt"), extra=extra)
    record = {
        "model": model_name,
        "n_parameters": model.n_parameters(),
        "train_time_s": train_time,
        "eval_jets_per_s": eval_rate,
        "metrics": metrics,
        **extra,
    }
    out.with_name(out.name + "_history.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    np.savez_compressed(
        out.with_name(out.name + "_scores.npz"),
        probs=probs,
        labels=y,
        jet_sdmass=np.asarray(test["jet_sdmass"]),
        jet_pt=np.asarray(test["jet_pt"]),
    )
    figure = _roc_figure(probs, y, label_set, f"{model_name}, {len(train)} jets", out.with_name(out.name + "_roc.png"))

    row: dict[str, object] = {
        "model": model_name,
        "train_data": f"{data_name}:{part}",
        "test_data": f"{data_name}:{'test' if part != 'test' else 'val'}",
        "n_train": len(train),
        "feature_set": model.feature_set or "jet",
        "accuracy": round(metrics["accuracy"], 5),
        "n_params": model.n_parameters(),
        "train_time_s": round(train_time, 1),
        "seed": seed,
        "note": (note + (f" auc_qg={metrics['auc']:.4f}" if "auc" in metrics else "")).strip(),
    }
    for key in ("auc_w", "r50_w", "auc_top", "r50_top", "r80_top"):
        for name in (key, key + "_err"):
            if name in metrics:
                row[name] = round(metrics[name], 5)
    append_result(results, row)

    summary = ", ".join(f"{k} {v:.4g}" for k, v in metrics.items())
    print(f"test: {summary}")
    print(
        f"training {train_time / 60:.1f} min ({np.mean(history.jets_per_s):.0f} jets/s), "
        f"inference {eval_rate:.0f} jets/s"
    )
    print(
        f"wrote {checkpoint}, {out.name}_history.json, {out.name}_scores.npz" + (f", {figure.name}" if figure else "")
    )
    print(f"appended a row to {results}", flush=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Train one model, or the course models with ``--course-models``; return the exit status."""
    args = parse_args(argv)
    for path in args.data:
        if not Path(path).exists():
            print(f"{path} is missing: make it with python examples/make_task_data.py --week 1", file=sys.stderr)
            return 1
    common = {
        "data": args.data,
        "part": args.part,
        "label_set": args.label_set,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_minutes": args.max_minutes,
        "results": args.results,
    }
    import torch

    previous_threads = torch.get_num_threads()
    start = time.perf_counter()
    try:
        if args.course_models:
            threads = args.threads or os.cpu_count() or 1
            for name, size, epochs in COURSE_MODELS:
                print(f"\n=== {name}: {size} jets, {epochs} epochs ===", flush=True)
                train_one(
                    name,
                    feature_set="kin",
                    train_size=size if args.train_size is None else min(size, args.train_size),
                    epochs=min(epochs, args.epochs),
                    seed=0,
                    threads=threads,
                    out=(args.out or Path("models/toy")) / name,
                    note=("course model " + args.note).strip(),
                    **common,
                )
        else:
            train_one(
                args.model,
                feature_set=args.feature_set,
                train_size=args.train_size,
                epochs=args.epochs,
                seed=args.seed,
                threads=args.threads or max(1, (os.cpu_count() or 2) // 2),
                out=args.out or Path("runs") / args.model,
                note=args.note,
                **common,
            )
    finally:
        torch.set_num_threads(previous_threads)
    print(f"done in {math.ceil(time.perf_counter() - start)} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
