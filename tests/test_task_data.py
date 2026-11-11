"""The data commands of TASKS.md (read by examples/make_task_data.py) parse and run, with fewer jets."""

import importlib.util
import shlex
from pathlib import Path

import numpy as np
import pytest

from itj.io import load, save
from itj.jetclass import CLASSES, MANIFEST
from itj.toy.cli import main as jetgen

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "make_task_data.py"
SAMPLE = Path(__file__).parent / "data" / "jetclass_sample.parquet"

pytestmark = pytest.mark.skipif(
    not (SCRIPT.exists() and (ROOT / "TASKS.md").exists()), reason="needs the repository checkout"
)

#: The number of commands in the blocks "Data for T1 to T3" and "Data for T4 to T7".
N_WEEK1, N_WEEK2 = 18, 8


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("make_task_data", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small(command: list[str], data_dir: Path) -> list[str]:
    """The command with at most 8 jets and every data path moved to ``data_dir``."""
    out = [str(data_dir / word[len("data/") :]) if word.startswith("data/") else word for word in command]
    if "-n" in out:
        k = out.index("-n") + 1
        out[k] = str(min(int(out[k]), 8))
    if out[0] == "generate":
        out += ["--no-progress", "--quiet"]
    return out


def test_blocks_parse(script):
    week1, week2 = script.task_commands(week="1"), script.task_commands(week="2")
    assert len(week1) == N_WEEK1
    assert len(week2) == N_WEEK2
    assert script.task_commands() == week1 + week2
    assert all(c[0] == "generate" for c in week1)
    assert [c[0] for c in week2].count("fetch") == 3
    assert [c[0] for c in week2].count("split") == 2
    assert all(c[0] in ("generate", "split", "fetch") for c in week2)
    with pytest.raises(ValueError, match="week"):
        script.task_commands(week="3")


def test_outputs(script):
    commands = script.task_commands()
    outputs = [c[c.index("-o") + 1] for c in commands if "-o" in c]
    assert len(set(outputs)) == len(outputs)
    assert all(o.startswith("data/") and o.endswith(".parquet") for o in outputs)
    for command in commands:
        if command[0] in ("generate", "merge", "split"):
            assert all(str(p).startswith("data") for p in script.output_of(command))
        else:
            with pytest.raises(ValueError, match="only defined"):
                script.output_of(command)
    split = next(c for c in commands if c[0] == "split" and "data/t2_detector.parquet" in c)
    assert [p.as_posix() for p in script.output_of(split)] == [
        f"data/t2_split/{part}/t2_detector.parquet" for part in ("train", "val", "test")
    ]
    produced = {p.as_posix() for c in commands if c[0] != "fetch" for p in script.output_of(c)}
    assert len(produced) == len(outputs) + 3 * 4


def _jetclass_val(directory: Path) -> None:
    """Write the committed sample as three JetClass validation files (QCD, W, top) under ``directory``."""
    sample = load(SAMPLE)
    for short in ("QCD", "Wqq", "Tbqq"):
        cls = next(c for c in CLASSES if c.short == short)
        rows = np.flatnonzero(np.asarray(sample[cls.label]))
        save(sample.select(rows), directory / MANIFEST["val", short].filename)


def test_task_commands_run(script, tmp_path, capsys):
    commands = script.task_commands()
    data_dir = tmp_path / "data"
    _jetclass_val(data_dir / "jetclass" / "val")
    generates = [c for c in commands if c[0] == "generate"]
    for command in generates:
        assert jetgen(_small(command, data_dir)) == 0, command
    for command in (c for c in commands if c[0] == "split"):
        assert jetgen(_small(command, data_dir)) == 0, command
    for command in (c for c in commands if c[0] == "fetch"):
        assert jetgen([*_small(command, data_dir), "--dest", str(tmp_path / "fetch"), "--dry-run"]) == 0, command
    text = capsys.readouterr().out
    assert "hqu.web.cern.ch" in text
    assert "particle_transformer" in text
    for command in commands:
        if command[0] in ("generate", "split"):
            for path in script.output_of(command):
                assert (tmp_path / path).exists(), path
    assert len(load(data_dir / "t2_detector.parquet")) == 8
    assert len(load(data_dir / "jetclass" / "split" / "test" / "TTBar_120.parquet")) == 10


def _fake_repo(root: Path, week1: list[str], week2: list[str]) -> Path:
    text = ["# Tasks", "", "## Data for T1 to T3", "", "```bash", "# week 1", *week1, "```", ""]
    text += ["## Data for T4 to T7", "", "```bash", *week2, "```", ""]
    (root / "TASKS.md").write_text("\n".join(text), encoding="utf-8")
    return root / "TASKS.md"


def test_main_runs_keeps_and_overwrites(script, tmp_path, monkeypatch, capsys):
    tasks = _fake_repo(
        tmp_path,
        ["jetgen generate --process wqq -n 20 --seed 3 -o data/a.parquet"],
        [
            "jetgen fetch models ParT_kin --dry-run",
            "jetgen split data/a.parquet --out-dir data/a_split",
            "jetgen generate --preset partons -n 10 --seed 4 -o data/b.parquet",
        ],
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "TASKS", tasks)
    monkeypatch.chdir(tmp_path)

    assert script.main(["--dry-run", "--week", "2", "--skip-fetch"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("2 data sets for TASKS.md, written to")
    assert "fetch" not in out

    assert script.main(["--week", "2", "--only", "split"]) == 1  # data/a.parquet is not there yet
    assert "make it first" in capsys.readouterr().err

    assert script.main(["--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "done in" in out
    assert len(load(tmp_path / "data" / "a.parquet")) == 20
    assert len(load(tmp_path / "data" / "a_split" / "test" / "a.parquet")) == 4
    assert len(load(tmp_path / "data" / "b.parquet")) == 10

    before = (tmp_path / "data" / "a.parquet").stat().st_mtime_ns
    assert script.main(["--only", "a.parquet"]) == 0
    out = capsys.readouterr().out
    assert out.count("exists, kept") == 2
    assert (tmp_path / "data" / "a.parquet").stat().st_mtime_ns == before

    (tmp_path / "data" / "a_split" / "val" / "a.parquet").unlink()  # an interrupted split is made again
    assert script.main(["--overwrite", "--only", "a"]) == 0
    out = capsys.readouterr().out
    assert "exists, kept" not in out
    assert (tmp_path / "data" / "a_split" / "val" / "a.parquet").exists()
    assert shlex.split(out.splitlines()[1])[:2] == ["jetgen", "generate"]
