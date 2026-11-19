"""Make the data sets of the course (the blocks "Data for week 1" to "Data for week 4") on any operating system.

The commands are read from the bash blocks under the headings "Data for week k" of the week files tasks/week1.md to
tasks/week4.md, so the script and the questions cannot disagree, and run through the entry point of the command line
interface (``itj.toy.cli.main``), so they write the same files as the commands typed by hand. Run it from anywhere
inside the repository::

    python examples/make_task_data.py --week 1            # the toy files of week 1; files that exist are kept
    python examples/make_task_data.py --week 4 --only fetch   # start the downloads of week 4 (at the start of week 3)
    python examples/make_task_data.py --week 4            # JetClass, pretrained models, split files, bonus toy files
    python examples/make_task_data.py --overwrite         # make every file of the four weeks again
    python examples/make_task_data.py --dry-run           # only print the commands
    python examples/make_task_data.py --only t3_w80       # only the commands whose command line contains "t3_w80"

Week 1 has 10 commands: about 290 000 toy jets and 0.48 GB in ``data/``, about 5 minutes on one core of a 2016 desktop
CPU and 1.5 to 2 minutes with 8 workers. Week 2 has 1 command: the training set of 150 000 jets (0.38 GB, about a
minute with 8 workers). Week 3 has 8 commands: 160 000 jets (0.39 GB, about a minute with 8 workers). Week 4 has 7
commands: it downloads 0.4 GB for the core questions (three JetClass validation files and three pretrained models) and
136 MB more for the bonus B.4, and writes about 1.2 GB (the split files for weaver and 120 000 bonus toy jets); it
needs ``data/t2_detector.parquet`` from week 2. Start its downloads at the start of week 3 with ``--week 4 --only
fetch``. ``--skip-fetch`` leaves out the downloads; ``jetgen fetch`` itself keeps files with the right size and MD5, so
the fetch lines always run.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repository root: every path of the week files is relative to it
TASKS_DIR = ROOT / "tasks"  # the week files week1.md to week4.md (and bonus.md, whose data the week-4 block makes)
WEEKS = ("1", "2", "3", "4")
HEADINGS = {week: f"## Data for week {week}" for week in WEEKS}
SPLIT_PARTS = ("train", "val", "test")


def week_file(week: str, tasks: Path = TASKS_DIR) -> Path:
    """Return the file of week ``week`` (``tasks/week<k>.md``) that holds the block "Data for week k"."""
    return tasks / f"week{week}.md"


def _block(path: Path, heading: str) -> list[str]:
    """Return the lines of the first bash block after ``heading`` in ``path``."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise RuntimeError(f"{path} is missing: it holds the block {heading!r}") from None
    try:
        start = lines.index(heading)
    except ValueError:
        raise RuntimeError(f"{path} has no heading {heading!r}") from None
    fence = next((k for k in range(start, len(lines)) if lines[k].strip() == "```bash"), None)
    if fence is None:
        raise RuntimeError(f"{path} has no bash block after {heading!r}")
    end = next(k for k in range(fence + 1, len(lines)) if lines[k].strip() == "```")
    return lines[fence + 1 : end]


def task_commands(tasks: Path = TASKS_DIR, week: str = "all") -> list[list[str]]:
    """Return the ``jetgen`` commands of the data blocks of the week files, in order, without the leading ``jetgen``.

    Parameters
    ----------
    tasks : pathlib.Path
        The folder of the week files (``tasks/`` of the repository).
    week : {"1", "2", "3", "4", "all"}
        The block of one week ("Data for week 1" in tasks/week1.md to "Data for week 4" in tasks/week4.md) or all
        four, in order.

    Returns
    -------
    list of list of str
        One list of words per command.
    """
    if week not in (*HEADINGS, "all"):
        raise ValueError(f"week must be 1, 2, 3, 4 or all, got {week!r}")
    commands = []
    for key in HEADINGS if week == "all" else (week,):
        path = week_file(key, tasks)
        for line in _block(path, HEADINGS[key]):
            words = shlex.split(line, comments=True)
            if not words:
                continue
            if words[0] != "jetgen":
                raise RuntimeError(f"unexpected line in the data block of {path}: {line!r}")
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


def block_for(path: str | Path, tasks: Path = TASKS_DIR) -> str | None:
    """Return the options of this script that make ``path`` (relative to the repository root), or None.

    ``"--week k"`` for a file that a ``generate``, ``merge`` or ``split`` line of week k writes; ``"--week k --only
    fetch"`` for a JetClass file under ``data/jetclass/<split>/`` that a ``fetch jetclass`` line of week k downloads;
    None for a file that no data block makes.
    """
    name = Path(path).as_posix()
    blocks = {week: task_commands(tasks, week) for week in WEEKS}
    for week, commands in blocks.items():
        for command in commands:
            if command[0] != "fetch" and name in {p.as_posix() for p in output_of(command)}:
                return f"--week {week}"
    for week, commands in blocks.items():
        for command in commands:
            if command[:2] == ["fetch", "jetclass"]:
                split = command[command.index("--split") + 1] if "--split" in command else "val"
                if name.startswith(f"data/jetclass/{split}/"):
                    return f"--week {week} --only fetch"
    return None


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
    parser = argparse.ArgumentParser(
        description="Make the data sets of the course (tasks/week1.md to tasks/week4.md) on any operating system."
    )
    parser.add_argument(
        "--week", choices=(*WEEKS, "all"), default="all", help="the data block of one week (default: all four)"
    )
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
    commands = task_commands(TASKS_DIR, args.week)
    if args.skip_fetch:
        commands = [c for c in commands if c[0] != "fetch"]
    if args.only:
        commands = [c for c in commands if args.only in "jetgen " + shlex.join(c)]
    which = "tasks/week1.md to tasks/week4.md" if args.week == "all" else f"tasks/week{args.week}.md"
    print(f"{len(commands)} data sets for {which}, written to {ROOT / 'data'}")
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
                block = block_for(missing[0], TASKS_DIR)
                how = f"python examples/make_task_data.py {block}" if block else "see the data blocks in tasks/"
                print(f"{missing[0]} is missing for: {line}\n(make it first: {how})", file=sys.stderr)
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
