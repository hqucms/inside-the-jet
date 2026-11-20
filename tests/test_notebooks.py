"""The notebooks: stripped outputs and the course kernel (fast), and an end-to-end execution of each (slow)."""

import json
import os
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
SAMPLE = Path(__file__).parent / "data" / "jetclass_sample.parquet"
KERNEL = {"display_name": "Python (inside-the-jet)", "name": "inside-the-jet"}


#: The notebooks of the course, by the week that first needs them: 1 (00_primer, 00_quickstart), 4 (01_jetclass)
#: and 2 (02_first_network).
NAMES = ["00_primer.ipynb", "00_quickstart.ipynb", "01_jetclass.ipynb", "02_first_network.ipynb"]


def test_the_four_notebooks_exist():
    assert [p.name for p in NOTEBOOKS] == NAMES


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_format_and_stripped(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)
    kernelspec = nb["metadata"]["kernelspec"]
    assert {k: kernelspec[k] for k in KERNEL} == KERNEL
    ids = [cell["id"] for cell in nb["cells"]]
    assert len(set(ids)) == len(ids)
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert re.match(r"# \S", "".join(nb["cells"][0]["source"])), "the first cell is a level-1 title"
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
    elif path.stem == "01_jetclass":
        assert "Using the 150 JetClass jets of tests/data instead." in text


def test_hand_written_loss_of_notebook_02_stays_finite():
    """The loss of section 3 of 02_first_network is the cross-entropy, and finite at the step sizes of its *Try*."""
    torch = pytest.importorskip("torch")
    nb = json.loads((ROOT / "notebooks" / "02_first_network.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    cell = next(source for source in sources if "range(n_steps)" in source)
    loop = cell.split("\nprint(")[0]  # the loop, without the printout and the figure
    generator = torch.Generator().manual_seed(0)
    x_train = torch.rand(500, 2, generator=generator) * torch.tensor([2.0, 1.0])
    y_train = (x_train[:, 0] + 0.3 * torch.randn(500, generator=generator) > 1.0).float()
    for n_steps, eta in ((300, 1.0), (300, 50.0)):
        ns = {"torch": torch, "x_train": x_train, "y_train": y_train}
        exec(loop.replace("n_steps, eta = 300, 1.0", f"n_steps, eta = {n_steps}, {eta}"), ns)
        losses = torch.tensor(ns["losses"])
        assert len(losses) == n_steps
        assert torch.isfinite(losses).all(), f"eta = {eta}: {int((~torch.isfinite(losses)).sum())} non-finite losses"
        # the loss of the last step, at the weights before that step's update (the logit z of the loop)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(ns["z"], y_train)
        assert float(ns["loss"]) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)
