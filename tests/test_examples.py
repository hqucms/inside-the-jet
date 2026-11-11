"""The example scripts run end to end on small inputs (in a temporary folder)."""

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

pytestmark = pytest.mark.skipif(not EXAMPLES.exists(), reason="needs the repository checkout")


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quickstart(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "quickstart.py"), "--jets", "120", "--seed", "3", "--out", str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "JetData(120 jets" in out
    assert "same jets: True" in out
    assert "exact identities of the toy" in out
    for process in ("qcd", "wqq", "tbqq"):
        assert f"\n{process} " in out
    assert (tmp_path / "quickstart_detector_n120_seed3.parquet").exists()
    if "figure:" in out:
        assert (tmp_path / "quickstart_detector_n120_seed3.png").exists()


def test_train_tagger(tmp_toy_file, tmp_path, capsys):
    from itj.metrics import RESULT_COLUMNS
    from itj.models import load_model

    script = _module("train_tagger")
    out = tmp_path / "runs" / "pfn"
    results = tmp_path / "results_table.csv"
    argv = ["--model", "pfn", "--data", str(tmp_toy_file), "--epochs", "2", "--threads", "1"]
    assert script.main([*argv, "--out", str(out), "--results", str(results), "--note", "tiny"]) == 0
    text = capsys.readouterr().out
    assert "test: accuracy" in text
    model = load_model(out.with_name("pfn.pt"))
    assert model.name == "pfn"
    record = json.loads(out.with_name("pfn_history.json").read_text())
    assert record["n_train"] == 210
    assert len(record["history"]["train_loss"]) == 2
    scores = np.load(out.with_name("pfn_scores.npz"))
    assert scores["probs"].shape == (60, 3)
    np.testing.assert_allclose(scores["probs"].sum(axis=1), 1.0, rtol=1e-5)
    with results.open() as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == RESULT_COLUMNS
    assert rows[0]["model"] == "pfn"
    assert rows[0]["n_train"] == "210"
    assert rows[0]["note"] == "tiny"
    assert 0 <= float(rows[0]["auc_w"]) <= 1
    assert 0 <= float(rows[0]["auc_w_err"]) < 0.5  # the bootstrap error


def test_train_tagger_mlp_and_qg(tmp_path):
    from itj.io import save
    from itj.toy import generate, preset

    script = _module("train_tagger")
    qg = save(generate(100, config=preset("shower"), processes=("q", "g"), seed=5), tmp_path / "qg.parquet")
    results = tmp_path / "results.csv"
    common = ["--data", str(qg), "--epochs", "1", "--threads", "1", "--results", str(results)]
    assert script.main(["--model", "mlp", "--label-set", "qg", "--out", str(tmp_path / "mlp"), *common]) == 0
    with results.open() as f:
        row = next(csv.DictReader(f))
    assert row["feature_set"] == "jet"
    assert row["auc_w"] == ""
    assert "auc_qg=" in row["note"]


def test_train_tagger_course_models(tmp_toy_file, tmp_path):
    script = _module("train_tagger")
    folder = tmp_path / "models" / "toy"
    argv = ["--course-models", "--data", str(tmp_toy_file), "--train-size", "64", "--epochs", "1", "--threads", "1"]
    assert script.main([*argv, "--out", str(folder), "--results", str(tmp_path / "results.csv")]) == 0
    names = [name for name, _, _ in script.COURSE_MODELS]
    assert names == ["pfn", "efn", "pnlite", "partsmall", "partsmall_nopair"]
    assert sorted(p.stem for p in folder.glob("*.pt")) == sorted(names)
    with (tmp_path / "results.csv").open() as f:
        assert [row["model"] for row in csv.DictReader(f)] == names


def test_train_tagger_missing_data(tmp_path, capsys):
    script = _module("train_tagger")
    assert script.main(["--data", str(tmp_path / "nothing.parquet")]) == 1
    assert "make_task_data.py" in capsys.readouterr().err


def test_make_doc_figures(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    script = _module("make_doc_figures")
    argv = ["--out-dir", str(tmp_path), "--only", "classes", "--only", "toy_vs_jetclass", "--workers", "1"]
    assert script.main([*argv, "--jetclass", str(tmp_path / "no_jetclass")]) == 0
    out = capsys.readouterr().out
    assert "toy_vs_jetclass.png skipped" in out
    assert (tmp_path / "classes.png").stat().st_size > 10_000
    assert not (tmp_path / "toy_vs_jetclass.png").exists()
