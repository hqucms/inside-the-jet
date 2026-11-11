"""Command line interface ``jetgen`` (also available as ``python -m itj.toy``).

::

    jetgen presets
    jetgen generate [-n N] [--seed S] [--preset P] [--process LIST] [--override PROC:KEY=VALUE]... [--mystery]
                    [config flags] [-o FILE] [--overwrite] [--no-truth] [--workers N|auto] ...
    jetgen info FILE [--columns] [--provenance] [--reveal]
    jetgen display FILE [--jet I] [--color type|prong|parton|ue] [--lund] [-o FIG]
    jetgen display --ladder [--process wqq] [--seed S] [--jet I] [-o FIG]
    jetgen merge FILE... -o OUT [--shuffle] [--seed S] [--renumber] [--overwrite]
    jetgen split FILE... --out-dir DIR [--test 0.2] [--val 0.1] [--overwrite]
    jetgen configs
    jetgen fetch jetclass [--split val|train|example] [--classes QCD,Wqq,Tbqq|all] [--dest DIR] [--dry-run]
    jetgen fetch models [NAME...] [--dest DIR] [--dry-run]
    jetgen validate [-n N] [--seed S] [--workers N|auto] [--only NAME]

Every configuration parameter has an option of the same name (``pi0_fraction`` -> ``--pi0-fraction``, the switches
as ``--ue/--no-ue``); options given on the command line change the chosen preset, e.g.
``jetgen generate --preset shower --process q --alphas-mode fixed --alphas-value 0.12``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import TOY_NAME
from .config import PRESETS, SimConfig, config_fields, preset
from .hardprocess import PROCESSES

__all__ = ["build_parser", "default_output", "main"]

log = logging.getLogger("itj")

_PART_TITLES = {
    "hard": "hard process",
    "shower": "shower",
    "hadron": "hadronisation",
    "ue": "underlying event",
    "detector": "detector",
    "jet": "jet",
}

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show the default of an option, unless it is empty (``None`` or ``False``)."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None or action.default is False or action.default is argparse.SUPPRESS:
            return action.help or ""
        return super()._get_help_string(action) or ""


# ----------------------------------------------------------------------------------------------------------------------
# Parser


def _workers(text: str) -> int | str:
    if text == "auto":
        return "auto"
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer or 'auto', got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer or 'auto', got {text!r}")
    return value


def _processes(text: str) -> tuple[str, ...]:
    names = tuple(p.strip() for p in text.split(",") if p.strip())
    bad = [p for p in names if p not in PROCESSES]
    if bad or not names:
        raise argparse.ArgumentTypeError(
            f"unknown process {bad[0] if bad else text!r}; choose from {','.join(PROCESSES)}"
        )
    return names


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of ``jetgen``."""
    parser = argparse.ArgumentParser(
        prog="jetgen",
        description=f'The jet toy "{TOY_NAME}": generate QCD, W and top jets in the JetClass format, inspect, '
        "draw, merge and split the files, and fetch JetClass and the pretrained models.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    pre = commands.add_parser(
        "presets",
        help="list the physics ladder and the default configuration",
        description="List the presets of the physics ladder and the default configuration.",
    )
    _add_logging_options(pre)

    _add_generate(commands)

    info = commands.add_parser(
        "info", help="describe a data file", description="Describe a data file (toy or JetClass)."
    )
    info.add_argument("file", type=Path, help="Parquet file")
    info.add_argument("--columns", action="store_true", help="list the columns with type, units and description")
    info.add_argument("--provenance", action="store_true", help="print the full provenance (incl. stage times)")
    info.add_argument(
        "--reveal", action="store_true", help="decode the planted shortcut of a --mystery file (spoils T7.2)"
    )
    _add_logging_options(info)

    disp = commands.add_parser(
        "display",
        help="draw one jet, or one jet at every level of the ladder",
        description="Draw one jet of a file (marker area = pT, equal axes in (Delta eta, Delta phi)), or with "
        "--ladder the same jet at the four presets partons, shower, hadrons and detector. A figure name ending in "
        ".gif animates the ladder.",
        formatter_class=_HelpFormatter,
    )
    disp.add_argument("file", type=Path, nargs="?", default=None, help="data file (not with --ladder)")
    disp.add_argument("--jet", type=int, default=0, metavar="I", help="index of the jet")
    disp.add_argument(
        "--color",
        choices=("type", "prong", "parton", "ue"),
        default=None,
        help="colouring (default: type; ladder: prong)",
    )
    disp.add_argument("--lund", action="store_true", help="add a panel with the primary Lund plane of the jet")
    disp.add_argument("--ladder", action="store_true", help="generate the jet at every preset of the ladder")
    disp.add_argument("--process", choices=PROCESSES, default="wqq", help="process of the ladder jet")
    disp.add_argument("--seed", type=int, default=0, help="seed of the ladder jet")
    disp.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="FIG",
        help="figure file (default: jet<I>.png, ladder: ladder_<process>_seed<S>_jet<I>.png)",
    )
    disp.add_argument("--dpi", type=int, default=150, help="resolution of a raster figure")
    _add_logging_options(disp)

    mrg = commands.add_parser(
        "merge",
        help="concatenate data files (cluster shards)",
        description="Concatenate toy files, e.g. shards generated with different seeds and disjoint "
        "--first-event-id ranges.",
        formatter_class=_HelpFormatter,
    )
    mrg.add_argument("files", type=Path, nargs="+", metavar="FILE", help="input files")
    mrg.add_argument("-o", "--output", type=Path, required=True, metavar="OUT", help="output file")
    mrg.add_argument("--shuffle", action="store_true", help="shuffle the jets after concatenating")
    mrg.add_argument("--seed", type=int, default=0, help="seed of the shuffle")
    mrg.add_argument("--renumber", action="store_true", help="renumber truth_event_id 0 ... N-1 instead of checking")
    mrg.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    _add_logging_options(mrg)

    spl = commands.add_parser(
        "split",
        help="write the train/val/test split of files (for weaver)",
        description="Write <out-dir>/{train,val,test}/<stem>.parquet with the itj split of every file: test = first "
        "20 %% of its jets, val = next 10 %%, train = the rest (row for row what itj.io.load(part=...) returns).",
        formatter_class=_HelpFormatter,
    )
    spl.add_argument("files", type=Path, nargs="+", metavar="FILE", help="input files (toy or JetClass)")
    spl.add_argument("--out-dir", type=Path, required=True, metavar="DIR", help="output directory")
    spl.add_argument("--test", type=float, default=0.2, help="test fraction (the first jets of each file)")
    spl.add_argument("--val", type=float, default=0.1, help="validation fraction (the next jets)")
    spl.add_argument("--overwrite", action="store_true", help="replace existing output files")
    _add_logging_options(spl)

    cfg = commands.add_parser(
        "configs",
        help="print the paths of the packaged weaver data configs",
        description="Print the paths of the "
        "packaged weaver data configs (jets3_* for the course, JetClass_* of the JetClass paper).",
    )
    _add_logging_options(cfg)

    fetch = commands.add_parser(
        "fetch",
        help="download JetClass files or the pretrained models",
        description="Download JetClass files from the CERN JetClassMini mirror or the pretrained ParT and "
        "ParticleNet models (sizes and MD5 checked; existing good files are kept).",
    )
    targets = fetch.add_subparsers(dest="target", required=True, metavar="TARGET")
    fjc = targets.add_parser(
        "jetclass", help="JetClass files", description="Download JetClass files.", formatter_class=_HelpFormatter
    )
    fjc.add_argument("--split", choices=("val", "train", "example"), default="val", help="which files")
    fjc.add_argument("--classes", default="QCD,Wqq,Tbqq", help="comma-separated classes, or 'all'")
    fjc.add_argument("--dest", type=Path, default=Path("data/jetclass"), help="destination directory")
    fjc.add_argument(
        "--base-url", default=None, metavar="URL", help="replace the mirror (also $ITJ_JETCLASS_URL), e.g. file://..."
    )
    fjc.add_argument("--overwrite", action="store_true", help="download again even if the file is good")
    fjc.add_argument("--dry-run", action="store_true", help="only list the URLs and destinations")
    _add_logging_options(fjc)
    fmo = targets.add_parser(
        "models", help="pretrained models", description="Download pretrained models.", formatter_class=_HelpFormatter
    )
    fmo.add_argument("names", nargs="*", metavar="NAME", help="model names (default: all six)")
    fmo.add_argument("--dest", type=Path, default=Path("models/pretrained"), help="destination directory")
    fmo.add_argument("--overwrite", action="store_true", help="download again even if the file is good")
    fmo.add_argument("--dry-run", action="store_true", help="only list the URLs and destinations")
    _add_logging_options(fmo)

    val = commands.add_parser(
        "validate",
        help="generate the validation samples and print the validation table",
        description="Generate the validation samples and compare the toy with the leading-log formulas and the "
        "JetClass numbers (each row passes within max(5 sigma, 3 %) of its calibrated value).",
        formatter_class=_HelpFormatter,
    )
    val.add_argument("-n", "--n-jets", type=int, default=10_000, metavar="N", help="jets per sample")
    val.add_argument("--seed", type=int, default=0, help="random seed")
    val.add_argument("--workers", type=_workers, default=1, metavar="N|auto", help="worker processes")
    val.add_argument(
        "--only", action="append", default=None, metavar="NAME", help="rows starting with NAME (repeatable, or a,b)"
    )
    _add_logging_options(val)
    return parser


def _add_generate(commands: Any) -> None:
    gen = commands.add_parser(
        "generate",
        help="generate jets and write them to a Parquet file",
        description="Generate jets and write them in the JetClass format (plus the truth record). Every "
        "configuration parameter has an option; the options change the chosen preset. Same seed, same events: "
        "runs that differ only in switches acting after the hard process hold the same jets.",
        formatter_class=_HelpFormatter,
    )
    gen.add_argument("-n", "--n-jets", type=int, default=10_000, metavar="N", help="number of jets")
    gen.add_argument("--seed", type=int, default=0, help="random seed (non-negative integer)")
    gen.add_argument("--preset", choices=list(PRESETS), default="detector", help="physics preset (cumulative ladder)")
    gen.add_argument(
        "--process",
        type=_processes,
        default=("qcd", "wqq", "tbqq"),
        metavar="LIST",
        help=f"comma-separated processes from {','.join(PROCESSES)} (default: qcd,wqq,tbqq)",
    )
    gen.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PROC:KEY=VALUE",
        help="change a parameter for one process only, e.g. wqq:pi0_fraction=0.40 (repeatable)",
    )
    gen.add_argument("--mystery", action="store_true", help="plant a hidden shortcut drawn from the seed (bonus T7.2)")
    out = gen.add_argument_group("output")
    out.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="output file (default: data/toy_<preset>_<processes>_n<N>_seed<S>.parquet)",
    )
    out.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    out.add_argument("--no-truth", action="store_true", help="do not write the truth_* columns")
    out.add_argument("--row-group-size", type=int, default=1000, metavar="ROWS", help="jets per Parquet row group")
    groups = {name: gen.add_argument_group(f"{title} options") for name, title in _PART_TITLES.items()}
    _add_config_options(groups)
    run = gen.add_argument_group("run options")
    run.add_argument("--workers", type=_workers, default=1, metavar="N|auto", help="worker processes")
    run.add_argument("--first-event-id", type=int, default=0, metavar="I", help="truth_event_id of the first jet")
    run.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True, help="show a progress indicator"
    )
    _add_logging_options(gen)


def _add_logging_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("logging").add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="more log messages")
    group.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")


def _number_type(f: Any) -> type:
    meta = f.metadata
    integer = meta.get("integer") or (isinstance(f.default, int) and not isinstance(f.default, bool))
    return int if integer else float


def _add_config_options(groups: Mapping[str, Any]) -> None:
    """Add one option per configuration parameter from the field metadata; ``None`` keeps the preset's value."""
    defaults = SimConfig()
    for part, name, f in config_fields():
        flag = "--" + name.replace("_", "-")
        meta = f.metadata
        unit = f" [{meta['unit']}]" if meta.get("unit") else ""
        value = getattr(getattr(defaults, part), name)
        kwargs: dict[str, Any] = {"default": None, "dest": name}
        if isinstance(f.default, bool):
            on = [p for p, cfg in PRESETS.items() if getattr(getattr(cfg, part), name)]
            kwargs["action"] = argparse.BooleanOptionalAction
            kwargs["help"] = f"{meta['help']} (on in: {', '.join(on) or 'no preset'})"
        else:
            shown = list(value) if isinstance(value, tuple) else value
            kwargs["help"] = f"{meta['help']}{unit}" + (
                " (detector: off)" if value is None else f" (detector: {_fmt(shown)})"
            )
            number = _number_type(f)
            if "choices" in meta:
                kwargs["choices"] = list(meta["choices"])
            elif meta.get("nargs"):
                kwargs |= {"nargs": int(meta["nargs"]), "type": number}
                if "metavar" in meta:
                    kwargs["metavar"] = tuple(meta["metavar"])
            else:
                kwargs |= {"type": number, "metavar": meta.get("metavar", "N" if number is int else "VALUE")}
        groups[part].add_argument(flag, **kwargs)


def _fmt(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_fmt(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# ----------------------------------------------------------------------------------------------------------------------
# Helpers


def default_output(
    config: SimConfig,
    processes: Sequence[str],
    n: int,
    seed: int,
    *,
    process_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> Path:
    """Return the default file name ``data/toy_<preset>_<processes>_n<N>_seed<S>.parquet``.

    ``<processes>`` joins the process names with ``-``. A configuration that is not a preset (or one with
    per-process overrides) gets the tag ``<nearest preset>-custom-<first 6 hex digits of the SHA-1 of its JSON>``.

    Examples
    --------
    >>> default_output(preset("shower"), ["q", "g"], 2000, 3).as_posix()
    'data/toy_shower_q-g_n2000_seed3.parquet'
    """
    from .config import matching_preset
    from .generate import nearest_preset

    name = matching_preset(config)
    tag = name or nearest_preset(config)[0]
    if name is None or process_overrides:
        text = config.to_json()
        if process_overrides:
            text += json.dumps({k: dict(v) for k, v in sorted(process_overrides.items())}, sort_keys=True)
        tag += "-custom-" + hashlib.sha1(text.encode()).hexdigest()[:6]
    return Path("data") / f"toy_{tag}_{'-'.join(processes)}_n{n}_seed{seed}.parquet"


def _field_table() -> dict[str, tuple[str, Any]]:
    return {name: (part, f) for part, name, f in config_fields()}


def _parse_value(name: str, f: Any, text: str) -> Any:
    """Convert the text of ``--override`` to the type of the configuration field ``f``."""
    meta = f.metadata
    raw = text.strip()
    if meta.get("optional") and raw.lower() in ("none", "off", ""):
        return None
    if isinstance(f.default, bool):
        if raw.lower() in _TRUE:
            return True
        if raw.lower() in _FALSE:
            return False
        raise ValueError(f"{name} is a switch: give true or false, got {text!r}")
    if "choices" in meta:
        return raw
    number = _number_type(f)
    try:
        if meta.get("nargs"):
            return tuple(number(v) for v in raw.replace(" ", ",").split(",") if v)
        return number(raw)
    except ValueError:
        raise ValueError(f"{name} needs {'an integer' if number is int else 'a number'}, got {text!r}") from None


def parse_override(text: str) -> tuple[str, str, Any]:
    """Parse one ``--override PROC:KEY=VALUE`` into ``(process, field name, value)``."""
    import difflib

    if ":" not in text or "=" not in text.split(":", 1)[1]:
        raise ValueError(f"--override needs PROC:KEY=VALUE (e.g. wqq:pi0_fraction=0.40), got {text!r}")
    process, rest = text.split(":", 1)
    key, value = rest.split("=", 1)
    process, key = process.strip(), key.strip().replace("-", "_")
    if process not in PROCESSES:
        raise ValueError(f"--override {text!r}: unknown process {process!r}; choose from {', '.join(PROCESSES)}")
    fields = _field_table()
    if key not in fields:
        close = difflib.get_close_matches(key, list(fields), n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(f"--override {text!r}: unknown configuration parameter {key!r}.{hint}")
    return process, key, _parse_value(key, fields[key][1], value)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for _, name, f in config_fields():
        value = getattr(args, name)
        if value is not None:
            out[name] = tuple(value) if f.metadata.get("nargs") else value
    return out


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", " ")


# ----------------------------------------------------------------------------------------------------------------------
# Commands


def _configure_logging(level: int) -> None:
    """Send the messages of the ``itj`` logger to stderr (other libraries' loggers are left alone)."""
    if not any(getattr(h, "_jetgen", False) for h in log.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._jetgen = True  # type: ignore[attr-defined]
        log.addHandler(handler)
    for h in log.handlers:
        if getattr(h, "_jetgen", False):
            h.stream = sys.stderr  # type: ignore[attr-defined]
    log.setLevel(level)
    log.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``jetgen``; returns the exit status."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    _configure_logging(level)
    command = args.command if args.command != "fetch" else f"fetch {args.target}"
    try:
        if args.command == "presets":
            from .config import describe_presets

            print(describe_presets())
            print("\nexample: jetgen generate --preset hadrons --no-ue --process q,g -n 1000")
        elif args.command == "generate":
            _generate(args, argv)
        elif args.command == "info":
            _info(args)
        elif args.command == "display":
            _display(args)
        elif args.command == "merge":
            from ..io import merge

            out = merge(
                args.files,
                args.output,
                shuffle=args.shuffle,
                seed=args.seed,
                renumber=args.renumber,
                overwrite=args.overwrite,
            )
            log.info("merged %d files into %s", len(args.files), out)
        elif args.command == "split":
            _split(args)
        elif args.command == "configs":
            from ..features import CONFIG_NAMES, config_path

            for name in CONFIG_NAMES:
                print(config_path(name))
        elif args.command == "fetch":
            _fetch(args)
        elif args.command == "validate":
            _validate(args)
    except (ValueError, TypeError, OSError, RuntimeError, IndexError, KeyError, ImportError) as error:
        print(f"jetgen {command}: error: {error}", file=sys.stderr)
        return 1
    return 0


def _generate(args: argparse.Namespace, argv: list[str]) -> None:
    from ..io import save
    from .generate import generate
    from .mystery import encode, mystery_overrides

    if args.n_jets < 1:
        raise ValueError(f"-n/--n-jets must be a positive integer, got {args.n_jets}")
    if args.row_group_size < 1:
        raise ValueError(f"--row-group-size must be a positive integer, got {args.row_group_size}")
    config = preset(args.preset, **_overrides(args))
    processes = tuple(args.process)
    process_overrides: dict[str, dict[str, Any]] = {}
    for text in args.override:
        process, key, value = parse_override(text)
        if process not in processes:
            raise ValueError(f"--override {text!r}: {process} is not among the processes {','.join(processes)}")
        process_overrides.setdefault(process, {})[key] = value
    if args.mystery:
        if process_overrides:
            raise ValueError("--mystery cannot be combined with --override")
        planted = mystery_overrides(args.seed)
        missing = [p for p in planted if p not in processes]
        if missing:
            raise ValueError("--mystery needs the processes qcd, wqq and tbqq")
        run_overrides: dict[str, dict[str, Any]] = {k: dict(v) for k, v in planted.items()}
    else:
        run_overrides = process_overrides
    output = args.output or default_output(
        config, processes, args.n_jets, args.seed, process_overrides=process_overrides
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists (use --overwrite to replace it)")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(output.parent, os.W_OK):
        raise PermissionError(f"cannot write {output}: no write permission in {output.parent.resolve()}")
    changed = _overrides(args)
    log.info(
        "generating %s jets (%s), preset %s%s",
        _thousands(args.n_jets),
        ",".join(processes),
        args.preset,
        f" with {changed}" if changed else "",
    )
    data = generate(
        args.n_jets,
        config=config,
        processes=processes,
        seed=args.seed,
        workers=args.workers,
        first_event_id=args.first_event_id,
        process_overrides=run_overrides or None,
        truth=not args.no_truth,
        progress=args.progress and not args.quiet,
    )
    provenance = dict(data.provenance or {})
    provenance["command_line"] = "jetgen " + shlex.join(argv)
    if args.mystery:
        provenance.pop("process_overrides", None)
        provenance["mystery"] = encode(args.seed)
    # the provenance is written with the file: the write time is measured on the first chunk and scaled
    stage_times = dict(provenance.get("stage_times_s") or {})
    stage_times["write"] = round(_estimate_write(data, args.row_group_size, truth=not args.no_truth), 4)
    provenance["stage_times_s"] = stage_times
    data = type(data)(data.columns, config=data.config, provenance=provenance, source=data.source)
    start = time.perf_counter()
    save(data, output, overwrite=args.overwrite, row_group_size=args.row_group_size, truth=not args.no_truth)
    write = time.perf_counter() - start
    seconds = float(provenance.get("wall_time_s", math.nan)) + write
    workers = int(provenance.get("workers", 1))
    log.info(
        "wrote %s: %s jets in %.1f s (%.2f ms/jet, %d worker%s)",
        output.as_posix(),
        _thousands(len(data)),
        seconds,
        1e3 * seconds / max(len(data), 1),
        workers,
        "" if workers == 1 else "s",
    )


def _estimate_write(data: Any, row_group_size: int, *, truth: bool) -> float:
    """Return the time to write ``data``, estimated from writing its first chunk to a temporary file."""
    import tempfile

    from ..io import save
    from .generate import CHUNK_SIZE

    head = data.head(min(len(data), CHUNK_SIZE))
    with tempfile.TemporaryDirectory(prefix="jetgen-") as tmp:
        start = time.perf_counter()
        save(head, Path(tmp) / "head.parquet", row_group_size=row_group_size, truth=truth)
        seconds = time.perf_counter() - start
    return seconds * len(data) / max(len(head), 1)


def _info(args: argparse.Namespace) -> None:
    from ..io import describe, read_metadata

    print(describe(args.file))
    if args.columns:
        from ..data import COLUMNS

        info = read_metadata(args.file)
        print()
        print(f"{'column':<26s} {'dtype':<15s} {'units':<5s} description")
        for column, dtype in info["columns"].items():
            meta = COLUMNS.get(column)
            units = meta.units if meta else ""
            text = meta.description if meta else ""
            print(f"{column:<26s} {dtype!s:<15s} {units:<5s} {text}".rstrip())
    if args.provenance:
        info = read_metadata(args.file)
        provenance = info["provenance"]
        print()
        if provenance is None:
            print("no provenance (not a toy file)")
        else:
            from .mystery import hide

            print(json.dumps(hide(provenance), indent=2, sort_keys=False, default=str))
    if args.reveal:
        from .mystery import reveal

        print()
        print(reveal(args.file))


def _display(args: argparse.Namespace) -> None:
    if args.ladder == (args.file is not None):
        raise ValueError("give either a FILE or --ladder")
    if args.jet < 0:
        raise ValueError(f"--jet must be a non-negative integer, got {args.jet}")
    plt = _pyplot()
    from .. import plot

    if args.ladder:
        from .generate import show_ladder

        jets = show_ladder(args.process, seed=args.seed, jet=args.jet)
        output = args.output or Path(f"ladder_{args.process}_seed{args.seed}_jet{args.jet}.png")
        color: Any = args.color or "prong"
        if output.suffix.lower() == ".gif":
            plot.animate_ladder(jets, output, color=color)
        else:
            fig = plot.ladder(jets, color=color)
            _savefig(fig, output, args.dpi)
            plt.close(fig)
        log.info("wrote %s", output.as_posix())
        return
    from ..io import load

    data = load(args.file)
    if args.jet >= len(data):
        raise IndexError(f"--jet {args.jet} is out of range: {args.file} holds {len(data)} jets")
    output = args.output or Path(f"jet{args.jet}.png")
    ncols = 2 if args.lund else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 4.8), squeeze=False, layout="constrained")
    plot.jet_display(data, args.jet, color=args.color or "type", ax=axes[0, 0])
    if args.lund:
        _lund_panel(axes[0, 1], data, args.jet)
    _savefig(fig, output, args.dpi)
    plt.close(fig)
    log.info("wrote %s", output.as_posix())


def _lund_panel(ax: Any, data: Any, i: int) -> None:
    """Draw the primary Lund-plane emissions of jet ``i`` (C/A declustering of its constituents)."""
    import numpy as np

    from ..observables import lund_primary

    jet = data.select(np.array([i]))
    p4, mask = jet.padded_p4()
    lund = lund_primary(p4, mask)
    x, y, z = lund["ln_inv_delta"].values, lund["ln_kt"].values, lund["z"].values
    sc = ax.scatter(x, y, c=z, cmap="viridis", vmin=0.0, vmax=0.5, s=30, edgecolors="k", linewidths=0.4)
    ax.figure.colorbar(sc, ax=ax, label="z", shrink=0.85)
    ax.set_xlabel(r"$\ln(1/\Delta)$")
    ax.set_ylabel(r"$\ln(k_t/\mathrm{GeV})$")
    ax.set_title(f"primary Lund plane, jet {i} ({len(x)} declusterings)", fontsize="medium")


def _split(args: argparse.Namespace) -> None:
    from ..io import write_split

    written = write_split(args.files, args.out_dir, test=args.test, val=args.val, overwrite=args.overwrite)
    for part, paths in written.items():
        for path in paths:
            log.info("%s: %s", part, Path(path).as_posix())


def _fetch(args: argparse.Namespace) -> None:
    from .. import jetclass

    if args.target == "jetclass":
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        paths = jetclass.fetch(
            args.split,
            classes,
            dest=args.dest,
            overwrite=args.overwrite,
            progress=not args.quiet,
            dry_run=args.dry_run,
            base_url=args.base_url,
        )
    else:
        paths = jetclass.fetch_models(
            args.names or None, dest=args.dest, overwrite=args.overwrite, progress=not args.quiet, dry_run=args.dry_run
        )
    if not args.dry_run:
        for path in paths:
            log.info("ready: %s", Path(path).as_posix())


def _validate(args: argparse.Namespace) -> None:
    from .validate import format_validation, validate

    if args.n_jets < 1:
        raise ValueError(f"-n/--n-jets must be a positive integer, got {args.n_jets}")
    only = [name.strip() for text in args.only or [] for name in text.split(",") if name.strip()] or None
    rows = validate(n=args.n_jets, seed=args.seed, workers=args.workers, only=only)
    print(format_validation(rows))


def _pyplot() -> Any:
    """Return matplotlib's pyplot with a file backend (the commands write figures, they never open windows)."""
    try:
        import matplotlib
    except ImportError as error:
        raise ImportError("drawing needs matplotlib: pip install 'itj[analysis]'") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _savefig(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
