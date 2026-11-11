"""The notebooks: stripped outputs and the course kernel (fast), and an end-to-end execution of each (slow)."""

import json
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
SAMPLE = Path(__file__).parent / "data" / "jetclass_sample.parquet"
KERNEL = {"display_name": "Python (inside-the-jet)", "name": "inside-the-jet"}


def test_both_notebooks_exist():
    assert [p.name for p in NOTEBOOKS] == ["00_quickstart.ipynb", "01_jetclass.ipynb"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_format_and_stripped(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)
    kernelspec = nb["metadata"]["kernelspec"]
    assert {k: kernelspec[k] for k in KERNEL} == KERNEL
    ids = [cell["id"] for cell in nb["cells"]]
    assert len(set(ids)) == len(ids)
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert "".join(nb["cells"][0]["source"]).startswith("# Quick start: ")
    for cell in nb["cells"]:
        assert cell["metadata"] == {}
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
    text = path.read_text(encoding="utf-8")
    assert chr(0x2014) not in text  # no em dashes
    assert text.endswith("\n")


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path, tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    # a copy of the repository layout without data/: 01_jetclass falls back to the committed sample
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "tests" / "data").mkdir(parents=True)
    shutil.copy(SAMPLE, tmp_path / "tests" / "data" / SAMPLE.name)
    copy = tmp_path / "notebooks" / path.name
    shutil.copy(path, copy)
    pretrained = Path(os.environ.get("ITJ_PRETRAINED_DIR", ROOT / "models" / "pretrained"))
    monkeypatch.setenv("ITJ_PRETRAINED_DIR", str(pretrained.resolve()))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    nb = nbformat.read(copy, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(copy.parent)}}
    )
    client.execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors
    text = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    if path.stem == "00_quickstart":
        assert "check identities: True" in text
        assert "same columns: True" in text
        assert (tmp_path / "data" / "quickstart_notebook.parquet").exists()
    else:
        assert "Using the 150 JetClass jets of tests/data instead." in text
