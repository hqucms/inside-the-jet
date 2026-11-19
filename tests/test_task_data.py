"""The data commands of the week files (read by examples/make_task_data.py) parse and run, with fewer jets."""

import importlib.util
import re
import shlex
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from itj.io import load, save
from itj.jetclass import CLASSES, MANIFEST
from itj.toy.cli import main as jetgen

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "make_task_data.py"
TRAIN_TAGGER = ROOT / "examples" / "train_tagger.py"
SAMPLE = Path(__file__).parent / "data" / "jetclass_sample.parquet"

pytestmark = pytest.mark.skipif(
    not (SCRIPT.exists() and (ROOT / "tasks" / "week1.md").exists()), reason="needs the repository checkout"
)

#: The number of commands in the blocks "Data for week 1" to "Data for week 4" (tasks/week1.md to tasks/week4.md).
N_WEEK = {"1": 10, "2": 1, "3": 8, "4": 7}

#: The 26 commands of the two data blocks of release 0.1.1 (without the leading ``jetgen``). Release 0.2.0 only
#: regroups them into four weeks: the same options and seeds make the same files, bit for bit, so the reference
#: solutions keep their data. A change here changes the data of the course.
GOLDEN_0_1_1 = (
    "generate --preset partons -n 30000 --seed 11 -o data/t1_partons.parquet",
    "generate --preset shower -n 30000 --seed 11 -o data/t1_shower.parquet",
    "generate --preset hadrons -n 30000 --seed 11 -o data/t1_hadrons.parquet",
    "generate --preset detector -n 30000 --seed 11 -o data/t1_detector.parquet",
    "generate --preset detector --no-ue -n 30000 --seed 11 -o data/t1_detector_noue.parquet",
    "generate --preset shower --process q --alphas-mode fixed --alphas-value 0.12 --pt-spectrum fixed --pt-fixed 600 "
    "-n 20000 --seed 12 -o data/t1_lund_q.parquet",
    "generate --preset shower --process g --alphas-mode fixed --alphas-value 0.12 --pt-spectrum fixed --pt-fixed 600 "
    "-n 20000 --seed 12 -o data/t1_lund_g.parquet",
    "generate --preset shower --process q,g -n 20000 --seed 13 -o data/t1_qg_shower.parquet",
    "generate --preset hadrons --process q,g -n 20000 --seed 13 -o data/t1_qg_hadrons.parquet",
    "generate --preset detector --process q,g -n 60000 --seed 13 -o data/t1_qg_detector.parquet",
    "generate -n 150000 --seed 21 -o data/t2_detector.parquet",
    "generate --process wqq --w-mass 60 -n 10000 --seed 35 -o data/t3_w60.parquet",
    "generate --process wqq -n 10000 --seed 35 -o data/t3_w80.parquet",
    "generate --process wqq --w-mass 100 -n 10000 --seed 35 -o data/t3_w100.parquet",
    "generate --process wqq --w-mass 120 -n 10000 --seed 35 -o data/t3_w120.parquet",
    "generate --process wqq --no-coherence -n 10000 --seed 35 -o data/t3_w80_nocoh.parquet",
    "generate --process wqq --no-ue -n 10000 --seed 35 -o data/t3_w80_noue.parquet",
    "generate --process wqq --cell-position centre -n 10000 --seed 35 -o data/t3_w80_centre.parquet",
    "fetch jetclass --split val --classes QCD,Wqq,Tbqq",
    "fetch models ParT_kin ParT_full ParticleNet_kin",
    "split data/jetclass/val/ZJetsToNuNu_120.parquet data/jetclass/val/WToQQ_120.parquet "
    "data/jetclass/val/TTBar_120.parquet --out-dir data/jetclass/split",
    "split data/t2_detector.parquet --out-dir data/t2_split",
    "generate -n 90000 --seed 61 -o data/t6_pretrain.parquet",
    "generate --override wqq:pi0_fraction=0.50 -n 60000 --seed 72 -o data/t7_shortcut.parquet",
    "generate --mystery -n 60000 --seed 73 -o data/t7_mystery.parquet",
    "fetch jetclass --split example",
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load(SCRIPT, "make_task_data")


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
    weeks = {week: script.task_commands(week=week) for week in N_WEEK}
    assert {week: len(commands) for week, commands in weeks.items()} == N_WEEK
    assert script.task_commands() == [c for week in N_WEEK for c in weeks[week]]
    for week in ("1", "2", "3"):
        assert all(c[0] == "generate" for c in weeks[week]), week
    assert [c[0] for c in weeks["4"]] == ["fetch", "fetch", "split", "split", "generate", "generate", "fetch"]
    for bad in ("5", "0", "T4", ""):
        with pytest.raises(ValueError, match="week"):
            script.task_commands(week=bad)


def test_each_block_is_in_the_file_of_its_week(script):
    """The heading "Data for week k" is in tasks/week<k>.md only: no other task file holds a data block."""
    files = [ROOT / "TASKS.md", *sorted((ROOT / "tasks").glob("*.md"))]
    for path in files:
        found = re.findall(r"^#+ Data for week (\d)$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
        mine = [path.name.removeprefix("week").removesuffix(".md")] if path.name.startswith("week") else []
        assert found == mine, f"{path.relative_to(ROOT)}: data blocks of weeks {found}"
        assert all(script.week_file(week) == path for week in found)


def test_commands_are_those_of_0_1_1(script):
    """The guard for bit-identical data: the 26 commands of 0.1.1, character for character, only regrouped."""
    commands = [shlex.join(c) for c in script.task_commands()]
    assert Counter(commands) == Counter(GOLDEN_0_1_1)


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


def test_inputs_come_from_the_same_week_or_earlier(script):
    """No block needs the data of a later week: every input is made, or downloaded, by a block at or before it."""
    for week in N_WEEK:
        for command in script.task_commands(week=week):
            for path in script._inputs_of(command):
                block = script.block_for(path)
                assert block is not None, (week, path)
                assert int(block.split()[1]) <= int(week), (week, path, block)


def test_block_for(script):
    assert script.block_for("data/t1_partons.parquet") == "--week 1"
    assert script.block_for(Path("data/t1_qg_detector.parquet")) == "--week 1"
    assert script.block_for("data/t2_detector.parquet") == "--week 2"
    assert script.block_for("data/t3_w80_noue.parquet") == "--week 3"
    assert script.block_for("data/t6_pretrain.parquet") == "--week 3"
    assert script.block_for("data/t7_mystery.parquet") == "--week 4"
    assert script.block_for("data/t2_split/train/t2_detector.parquet") == "--week 4"
    assert script.block_for("data/jetclass/split/test/TTBar_120.parquet") == "--week 4"
    assert script.block_for("data/jetclass/val/WToQQ_120.parquet") == "--week 4 --only fetch"
    assert script.block_for("data/jetclass/example/JetClass_example_100k.root") == "--week 4 --only fetch"
    assert script.block_for("data/jetclass/train/HToBB_000.parquet") is None
    assert script.block_for("data/elsewhere.parquet") is None


def test_train_tagger_names_the_same_block(script):
    """The hint of examples/train_tagger.py for a missing input agrees with the data blocks of the week files."""
    train_tagger = _load(TRAIN_TAGGER, "train_tagger")
    paths = [p for c in script.task_commands() if c[0] != "fetch" for p in script.output_of(c)]
    paths += [Path("data/jetclass/val") / MANIFEST["val", short].filename for short in ("QCD", "Wqq", "Tbqq")]
    for path in paths:
        assert train_tagger.data_block(path) == script.block_for(path), path
    assert train_tagger.data_block("elsewhere/mine.parquet") is None
    assert train_tagger.data_block("data/jetclass/train/TTBar_000.parquet") is None  # no data block fetches it


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


def _fake_repo(root: Path, blocks: dict[str, list[str]]) -> Path:
    """Write tasks/week1.md to tasks/week4.md under ``root`` with these data blocks; return the folder."""
    (root / "tasks").mkdir()
    for week in N_WEEK:
        text = [f"# Week {week}: a title", "", f"## Data for week {week}", "", "```bash", f"# week {week}"]
        text += [*blocks.get(week, []), "```", ""]
        (root / "tasks" / f"week{week}.md").write_text("\n".join(text), encoding="utf-8")
    return root / "tasks"


def test_main_runs_keeps_and_overwrites(script, tmp_path, monkeypatch, capsys):
    tasks = _fake_repo(
        tmp_path,
        {
            "2": ["jetgen generate --process wqq -n 20 --seed 3 -o data/a.parquet"],
            "4": [
                "jetgen fetch models ParT_kin --dry-run",
                "jetgen split data/a.parquet --out-dir data/a_split",
                "jetgen generate --preset partons -n 10 --seed 4 -o data/b.parquet",
            ],
        },
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "TASKS_DIR", tasks)
    monkeypatch.chdir(tmp_path)

    assert script.main(["--dry-run", "--week", "4", "--skip-fetch"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("2 data sets for tasks/week4.md, written to")
    assert "fetch" not in out

    assert script.main(["--dry-run", "--week", "1"]) == 0  # an empty block
    assert capsys.readouterr().out.startswith("0 data sets")

    assert script.main(["--week", "4", "--only", "split"]) == 1  # data/a.parquet is not there yet
    assert "make it first: python examples/make_task_data.py --week 2" in capsys.readouterr().err

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


def test_missing_jetclass_names_the_fetch(script, tmp_path, monkeypatch, capsys):
    tasks = _fake_repo(
        tmp_path,
        {
            "4": [
                "jetgen fetch jetclass --split val --classes QCD",
                "jetgen split data/jetclass/val/ZJetsToNuNu_120.parquet --out-dir data/jetclass/split",
            ]
        },
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "TASKS_DIR", tasks)
    monkeypatch.chdir(tmp_path)
    assert script.main(["--week", "4", "--skip-fetch"]) == 1
    assert "make it first: python examples/make_task_data.py --week 4 --only fetch" in capsys.readouterr().err
