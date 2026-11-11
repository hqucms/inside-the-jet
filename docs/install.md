# Installation details

The short version is in the [README](../README.md#installation): a virtual environment, one `pip install`, the tests
and the Jupyter kernel. This page covers what the short version leaves out: Windows, the PyTorch build for your
machine (CPU, GPU, Intel Macs, Apple Silicon), `uv`, clusters, the optional weaver extra, GPUs and Colab, followed by
the common problems. The repository is called `inside-the-jet`; the package it installs is `itj`, with the command
`jetgen`.

No fastjet or ROOT installation is needed: `itj` has its own clustering (pure NumPy), and the ROOT file of JetClass is
read with `uproot`, which comes with the `analysis` extra. `fastjet` (`pip install fastjet`) is used only as an
optional cross-check by the tests.

## Windows

Create and activate the environment in the Command Prompt or in PowerShell; the other commands are the same as in the
README:

```bat
REM Command Prompt
python -m venv .venv
.venv\Scripts\activate.bat
```

```powershell
# PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If `python` is not found, use `py` instead. Every `jetgen` command of [TASKS.md](../TASKS.md) is a single line and
works in any shell; `python examples/make_task_data.py --week 1` (and `--week 2`) makes the data sets of the tasks.
The weaver commands of [examples/weaver](../examples/weaver/README.md) are given for bash and for the Command Prompt
(which needs double quotes: `--gpus ""`). [WSL](https://learn.microsoft.com/windows/wsl/) gives a Linux shell on
Windows if you prefer bash.

## Which PyTorch

A plain `pip install -e ".[analysis,dev]"` installs the PyTorch build that PyPI serves for your machine. It works
everywhere: every model of the course trains on any CPU, and `itj` uses a GPU only when you ask for it
(`device="cuda"`). The size of the download is the only reason to choose a build by hand:

| machine                              | what PyPI installs                              | smaller alternative                       |
|--------------------------------------|-------------------------------------------------|-------------------------------------------|
| Linux laptop or CPU node             | CUDA build with NVIDIA libraries, several GB    | CPU build, about 200 MB (below)           |
| Windows                              | CPU build                                       | none needed                               |
| Mac with Apple silicon               | CPU build (with MPS)                            | none needed                               |
| Mac with an Intel processor          | nothing recent (see below)                      |                                           |
| machine with an NVIDIA GPU           | CUDA build                                      | the build matching your driver (below)    |

Install PyTorch **before** the package to choose a build; `pip` then keeps it:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU build, about 200 MB
python -m pip install -e ".[analysis,dev]"
```

**NVIDIA GPU.** `nvidia-smi` prints the newest CUDA version the driver supports in its header. Install the build for
that CUDA major version, e.g. `--index-url https://download.pytorch.org/whl/cu126` for CUDA 12 or `cu130` for
CUDA 13. The [PyTorch selector](https://pytorch.org/get-started/locally/) lists the current builds. A CUDA build takes
several GB: put the environment on a file system with enough space. A GPU is never needed for the core tasks; the
parts marked *GPU (optional)* in TASKS.md run faster with one.

**Macs with an Intel processor.** PyTorch 2.2.2 is the last release for them (Python 3.10 to 3.12) and it needs
NumPy 1:

```bash
python -m pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[analysis,dev]" "numpy<2"
```

weaver 0.5.4 needs NumPy 2.1.1 or newer, so the `weaver` extra does not install on Intel Macs: do task T4.6 with
`itj` only, as TASKS.md says.

**Apple Silicon.** The default PyPI build works and includes MPS (the GPU of the Mac). `itj` runs on the CPU unless you
pass `device="mps"`; weaver selects MPS by itself when it is asked for the CPU (`--gpus ''`), so set
`PYTORCH_ENABLE_MPS_FALLBACK=1` before running weaver (some operations of ParT have no MPS kernel).

**Check what you got:** `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

## With uv

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[analysis,dev]"
python -m ipykernel install --user --name inside-the-jet --display-name "Python (inside-the-jet)"
```

An environment made by `uv venv` has no pip of its own: use `uv pip install ...` for every later install (or create it
with `uv venv --seed`), since `python -m pip` fails there and a bare `pip` would install into another environment.

## On a cluster

A laptop is enough for the core tasks; a cluster helps for large toy samples and for the optional GPU trainings of ParT
(see [examples/cluster](../examples/cluster/README.md)). Load the cluster's Python module (if there is one) before you
create the environment. Put the environment (about 2 GB with the CPU build of PyTorch and the `analysis` extra) and the
data (about 1.0 GB for week 1 and 1.9 GB for week 2, downloads included) on a work or scratch file system, not in a
small home directory. Run the simulations and trainings in a job or an interactive session, not on the login node.

## Notebooks and kernels

The README registers the environment as the Jupyter kernel `Python (inside-the-jet)`; the notebooks of this repository
ask for that kernel by name, so JupyterLab, VS Code and JupyterHub pick it up. `python -m jupyter lab`, run in the
activated environment, starts JupyterLab from this environment (a bare `jupyter` may belong to another Python
installation). If a notebook reports `ModuleNotFoundError: No module named 'itj'` or `'torch'`, it runs on another
kernel: choose `Python (inside-the-jet)` in the kernel menu (JupyterLab: Kernel > Change Kernel; VS Code: the kernel
picker in the top right). In VS Code, open the repository folder itself, so that it finds `.venv`.

## Optional: weaver

[weaver](https://github.com/hqucms/weaver-core) is the training framework of ParT and of the JetClass paper. Only task
T4.6 uses it; every other task runs with `itj` alone, which loads the pretrained models without weaver.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu    # torch first: weaver does not install it
python -m pip install -e ".[weaver]"                                            # weaver-core 0.5.4, about 300 MB
python -c "import weaver"
```

- **Python 3.10 to 3.13.** weaver's dependencies (numba among them) may not install on Python 3.14 yet; the core of
  the course does not need them.
- **Not on Intel Macs** (see above: weaver needs NumPy 2.1.1 or newer).
- **On the CPU** pass `--gpus ''` (the default `'0'` asks for a GPU), and `--num-workers 0` on Windows (it is not
  slower on a CPU). On Apple Silicon set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **From the repository root.** The commands of [examples/weaver](../examples/weaver/README.md) assume an editable
  checkout and use the YAML files in `src/itj/configs/` (`jetgen configs` prints their paths for any installation).
  weaver writes a `*.auto.yaml` next to the data configuration (ignored by git).

## GPU and Colab

On a machine with an NVIDIA GPU, install the CUDA build of PyTorch as above and pass `device="cuda"` to `itj.train.fit`,
`itj.train.predict` and `itj.probes.extract` (or `--gpus 0` to weaver). On [Colab](https://colab.research.google.com),
choose a GPU runtime and run one cell:

```bash
!git clone https://github.com/hqucms/inside-the-jet.git && cd inside-the-jet && pip install -q -e ".[analysis,weaver]" && python examples/make_task_data.py --week 2 --only fetch
```

Colab already has PyTorch with CUDA; then `%cd inside-the-jet` and work as on a laptop. A Colab session forgets its
files when it ends: copy your results to your own storage.

## Common problems

- `No module named 'itj'` or `'torch'` in a notebook: the kernel is not the environment, see above.
- `jetgen: command not found`: the environment is not activated; `python -m itj.toy` is the same command.
- `No space left on device` during `pip install`: on Linux, PyPI serves a CUDA build of PyTorch of several GB; the
  CPU build (`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`, before the package)
  is about 200 MB. Free space (`python -m pip cache purge` empties the pip cache) or create the environment on another
  disk and link it: `python3 -m venv /path/with/space/inside-the-jet && ln -s /path/with/space/inside-the-jet .venv`.
- `pytest` not found: it comes with the `dev` extra, `pip install -e ".[analysis,dev]"`.
- `jetgen fetch` is slow or blocked: the downloads resume where they stopped when you run the command again; a teacher
  can provide the files on a local share, read with `jetgen fetch jetclass --base-url URL`, where `URL` holds the
  folders `val/`, `train/` and `example/` as on the mirror ([jetclass.md](jetclass.md#download)).
- A training is killed or the machine swaps in T2.4: the two ParT-small trainings need about 13 GB together; run them
  one after the other on a laptop with 8 GB.
- `uproot` or `awkward` not found when reading the ROOT example of JetClass: install the `analysis` extra.
- weaver fails with "CUDA" errors on a laptop: pass `--gpus ''`.
- In every new terminal, activate the environment again: `cd inside-the-jet && source .venv/bin/activate`.
