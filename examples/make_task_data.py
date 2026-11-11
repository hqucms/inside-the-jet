"""Make the data sets of TASKS.md ("Data for T1 to T3" and "Data for T4 to T7") on any operating system.

The commands are read from the bash blocks under the two headings "Data for ..." of TASKS.md, so the script and the
task sheet cannot disagree, and run through the entry point of the command line interface (``itj.toy.cli.main``), so
they write the same files as the commands typed by hand. Run it from anywhere inside the repository::

    python examples/make_task_data.py --week 1            # the toy files of week 1; files that exist are kept
    python examples/make_task_data.py --week 2 --only fetch   # start the downloads of week 2 early
    python examples/make_task_data.py --week 2            # JetClass, pretrained models, split files, more toy files
    python examples/make_task_data.py --overwrite         # make every file of both weeks again
    python examples/make_task_data.py --dry-run           # only print the commands
    python examples/make_task_data.py --only t3_w80       # only the commands whose command line contains "t3_w80"

Week 1 has 18 commands: about 510 000 toy jets and 1.0 GB in ``data/``, about 9 minutes on one core of a 2016 desktop
CPU, 3 to 4 minutes with 4 workers and under 3 minutes with 8. Week 2 has 8 commands: it downloads 0.4 GB for the
core tasks (three JetClass validation files and three pretrained models) and 136 MB more for the bonus T7.3, and writes
about 1.4 GB (the split files for weaver and 210 000 more toy jets, under 2 minutes with 8 workers); it needs
``data/t2_detector.parquet`` from week 1. ``--skip-fetch`` leaves out the downloads; ``jetgen fetch`` itself keeps
files with the right size and MD5, so the fetch lines always run.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repository root: every path of TASKS.md is relative to it
TASKS = ROOT / "TASKS.md"
HEADINGS = {"1": "## Data for T1 to T3", "2": "## Data for T4 to T7"}
SPLIT_PARTS = ("train", "val", "test")


def _block(lines: list[str], heading: str, tasks: Path) -> list[str]:
    """Return the lines of the first bash block after ``heading``."""
    try:
        start = lines.index(heading)
    except ValueError:
        raise RuntimeError(f"{tasks} has no heading {heading!r}") from None
    fence = next((k for k in range(start, len(lines)) if lines[k].strip() == "```bash"), None)
    if fence is None:
        raise RuntimeError(f"{tasks} has no bash block after {heading!r}")
    end = next(k for k in range(fence + 1, len(lines)) if lines[k].strip() == "```")
    return lines[fence + 1 : end]


def task_commands(tasks: Path = TASKS, week: str = "all") -> list[list[str]]:
    """Return the ``jetgen`` commands of the data blocks of TASKS.md, in order, without the leading ``jetgen``.

    Parameters
    ----------
    tasks : pathlib.Path
        The task sheet.
    week : {"1", "2", "all"}
        The block of week 1 ("Data for T1 to T3"), of week 2 ("Data for T4 to T7") or both.

    Returns
    -------
    list of list of str
        One list of words per command.
    """
    if week not in (*HEADINGS, "all"):
        raise ValueError(f"week must be 1, 2 or all, got {week!r}")
    lines = tasks.read_text(encoding="utf-8").splitlines()
    commands = []
    for key in HEADINGS if week == "all" else (week,):
        for line in _block(lines, HEADINGS[key], tasks):
            words = shlex.split(line, comments=True)
            if not words:
                continue
            if words[0] != "jetgen":
                raise RuntimeError(f"unexpected line in the data block of {tasks}: {line!r}")
            commands.append(words[1:])
    return commands


def output_of(command: Sequence[str]) -> list[Path]:
    """Return the files a ``generate``, ``merge`` or ``split`` command writes (relative to the repository root).

    ``generate`` and ``merge``: the value after ``-o``; ``split``: ``<out-dir>/{train,val,test}/<stem>.parquet`` for
    every input file.
    """
    if command[0] in ("generate", "merge"):
        flag = "-o" if "-o" in command else "--output"
        return [Path(command[command.index(flag) + 1])]
    if command[0] == "split":
        out_dir = Path(command[command.index("--out-dir") + 1])
        inputs = [Path(w) for w in command[1:] if w.endswith((".parquet", ".root"))]
        return [out_dir / part / f"{p.stem}.parquet" for part in SPLIT_PARTS for p in inputs]
    raise ValueError(f"output_of is only defined for generate, merge and split, not {command[0]!r}")


def _inputs_of(command: Sequence[str]) -> list[Path]:
    """Return the input files of a ``split`` or ``merge`` command."""
    if command[0] not in ("split", "merge"):
        return []
    words, skip = [], False
    for word in command[1:]:
        if skip:
            skip = False
        elif word in ("-o", "--output", "--out-dir", "--seed", "--test", "--val"):
            skip = True
        elif not word.startswith("-"):
            words.append(Path(word))
    return words


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Make the data sets of TASKS.md (T1 to T7) on any operating system.")
    parser.add_argument("--week", choices=("1", "2", "all"), default="all", help="which data block (default: all)")
    parser.add_argument("--skip-fetch", action="store_true", help="leave out the downloads (the fetch lines)")
    parser.add_argument("--only", metavar="TEXT", help="only the commands whose command line contains TEXT")
    parser.add_argument("--dry-run", action="store_true", help="only print the commands")
    parser.add_argument("--overwrite", action="store_true", help="make every file again, also those that exist")
    parser.add_argument(
        "--workers", default="auto", metavar="N|auto", help="worker processes of the generate lines (default: auto)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the commands of the data blocks; return the exit status of the first command that fails, else 0."""
    args = parse_args(argv)
    os.chdir(ROOT)
    commands = task_commands(TASKS, args.week)
    if args.skip_fetch:
        commands = [c for c in commands if c[0] != "fetch"]
    if args.only:
        commands = [c for c in commands if args.only in "jetgen " + shlex.join(c)]
    print(f"{len(commands)} data sets for TASKS.md, written to {ROOT / 'data'}")
    if args.dry_run:
        for command in commands:
            print("jetgen " + shlex.join(command))
        return 0

    from itj.toy.cli import main as jetgen

    start = time.perf_counter()
    for command in commands:
        line = "jetgen " + shlex.join(command)
        extra: list[str] = []
        if command[0] in ("generate", "merge", "split"):
            targets = output_of(command)
            present = [t.exists() for t in targets]
            if all(present) and not args.overwrite:
                print(f"exists, kept: {', '.join(str(t) for t in targets)}")
                continue
            missing = [p for p in _inputs_of(command) if not p.exists()]
            if missing:
                week = "1" if "t2_detector" in missing[0].name else "2 --only fetch"
                print(
                    f"{missing[0]} is missing for: {line}\n(make it first: python examples/make_task_data.py --week "
                    f"{week})",
                    file=sys.stderr,
                )
                return 1
            if args.overwrite or any(present):  # a split interrupted half way is made again
                extra.append("--overwrite")
        if command[0] == "generate":
            extra += ["--workers", str(args.workers), "--no-progress"]
        print(line, flush=True)
        t0 = time.perf_counter()
        status = jetgen([*command, *extra])
        if status:
            print(f"failed with exit status {status}: {line}", file=sys.stderr)
            return int(status)
        print(f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
    print(f"done in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
