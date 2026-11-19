# Installation details

The short version is in the [README](../README.md#installation): a virtual environment, one `pip install`, the tests and
the Jupyter kernel. This page covers what the short version leaves out, with the common problems at the end. It covers
Windows and the build of PyTorch (the neural-network library of the course) for your machine: CPU, GPU (graphics
processor), Intel Macs or Apple Silicon. It also covers `uv`, clusters, the optional weaver extra, GPUs and Colab. The
repository is called `inside-the-jet`; the package it installs is `itj`, with the command `jetgen`.

No fastjet or ROOT installation is needed. fastjet is the standard library that groups particles into jets
(clustering), and ROOT is the analysis framework and file format of particle physics. `itj` has its own clustering (pure
NumPy), and the ROOT file of JetClass (the data set of [jetclass.md](jetclass.md)) is read with `uproot`, which comes
with the `analysis` extra. `fastjet` (`pip install fastjet`) is used only as an optional cross-check by the tests.

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
works in any shell; `python examples/make_task_data.py --week 1` (to `--week 4`) makes the data sets of each week.
The weaver commands of [examples/weaver](../examples/weaver/README.md) are given for bash and for the Command Prompt
(which needs double quotes: `--gpus ""`). [WSL](https://learn.microsoft.com/windows/wsl/) gives a Linux shell on
Windows if you prefer bash.

## Which PyTorch

A plain `pip install -e ".[analysis,dev]"` installs the PyTorch build that PyPI (the Python package index) serves for
your machine. It works everywhere: every model of the course trains on any CPU, and `itj` uses a GPU only when you ask
for it (`device="cuda"`). The size of the download is the only reason to choose a build by hand:

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

**NVIDIA GPU.** CUDA is NVIDIA's software layer for computing on its GPUs. `nvidia-smi` prints the newest CUDA version
the driver supports in its header. Install the build for
that CUDA major version, e.g. `--index-url https://download.pytorch.org/whl/cu126` for CUDA 12 or `cu130` for
CUDA 13. The [PyTorch selector](https://pytorch.org/get-started/locally/) lists the current builds. A CUDA build takes
several GB: put the environment on a file system with enough space. A GPU is never needed for the core questions; the
parts marked *Optional (GPU)* in TASKS.md run faster with one.

**Macs with an Intel processor.** PyTorch 2.2.2 is the last release for them (Python 3.10 to 3.12) and it needs
NumPy 1:

```bash
python -m pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[analysis,dev]" "numpy<2"
```

weaver 0.5.4 needs NumPy 2.1.1 or newer, so the `weaver` extra does not install on Intel Macs: do question `W4.2.7`
with `itj` only, as TASKS.md says.

**Apple Silicon.** The default PyPI build works and includes MPS (the GPU of the Mac). `itj` runs on the CPU unless you
pass `device="mps"`. weaver selects MPS by itself when it is asked for the CPU (`--gpus ''`). Some operations of ParT
(the Particle Transformer network) have no MPS kernel. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before running weaver.

**Check what you got:** `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

## With uv

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[analysis,dev]"
python -m ipykernel install --user --name inside-the-jet --display-name "Python (inside-the-jet)"
```

An environment made by `uv venv` has no pip of its own. There, `python -m pip` fails and a bare `pip` would install
into another environment. Use `uv pip install ...` for every later install (or create the environment with
`uv venv --seed`).

## On a cluster

A laptop is enough for the core questions. A cluster helps for large toy samples and for the optional GPU trainings of
ParT (see [examples/cluster](../examples/cluster/README.md)). Load the cluster's Python module
(if there is one) before you create the environment. Put the environment and the data on a work or scratch file system,
not in a small home directory. The environment takes about 2 GB with the CPU build of PyTorch and the `analysis` extra.
The data take about 0.5 GB in week 1, 0.4 GB in week 2, 0.4 GB in week 3 and 1.6 GB in week 4 with the JetClass
downloads, about 2.9 GB in all. Run the simulations and trainings in a job or an interactive session, not on the login
node.

## Notebooks and kernels

The README registers the environment as the Jupyter kernel `Python (inside-the-jet)`; the notebooks of this repository
ask for that kernel by name, so JupyterLab, VS Code and JupyterHub pick it up. `python -m jupyter lab`, run in the
activated environment, starts JupyterLab from this environment (a bare `jupyter` may belong to another Python
installation). Start it from the repository root and keep your own notebooks and scripts there. Then the paths
`data/...` and `models/...` of TASKS.md work as written. A notebook inside `notebooks/` runs in that folder and reads
`../data/...`. If a notebook reports `ModuleNotFoundError: No module named 'itj'` or `'torch'`, it runs on another
kernel: choose `Python (inside-the-jet)` in the kernel menu (JupyterLab: Kernel > Change Kernel; VS Code: the kernel
picker in the top right). In VS Code, open the repository folder itself, so that it finds `.venv`.

## Optional: weaver

[weaver](https://github.com/hqucms/weaver-core) is the training framework of ParT and of the JetClass paper. Only the
week-4 stretch questions `W4.2.7` and `W4.2.8` use it; every other question runs with `itj` alone, which loads the
pretrained models without weaver. Install it when you reach `W4.2.7`.

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
  checkout. They use the YAML files in `src/itj/configs/` (`jetgen configs` prints their paths for any installation).
  weaver writes a `*.auto.yaml` next to the data configuration (ignored by git).

## GPU and Colab

On a machine with an NVIDIA GPU, install the CUDA build of PyTorch as above and pass `device="cuda"` to `itj.train.fit`,
`itj.train.predict` and `itj.probes.extract` (or `--gpus 0` to weaver). On [Colab](https://colab.research.google.com)
(Google's online Jupyter notebooks), choose a GPU runtime and run one cell:

```bash
!git clone https://github.com/hqucms/inside-the-jet.git && cd inside-the-jet && pip install -q -e ".[analysis,weaver]" && python examples/make_task_data.py --week 4 --only fetch
```

Colab already has PyTorch with CUDA; then `%cd inside-the-jet` and work as on a laptop. A Colab session forgets its
files when it ends: copy your results to your own storage.

## Common problems

The first five are those of the set-up (question `W1.0.1`).

- `jetgen: command not found`: the environment is not activated; activate it, or use `python -m itj.toy`, which is the
  same command.
- `pytest` not found: it comes with the `dev` extra, `pip install -e ".[analysis,dev]"`.
- About 30 `import file mismatch` errors when `pytest` collects the tests: the package is installed without `-e` (for
  example straight from git), so the tests find two copies of `itj`. Reinstall it editable from the repository root,
  `python -m pip install -e ".[analysis,dev]"`.
- `module 'itj' has no attribute 'toy'` (or `'observables'`, ...): `import itj` alone does not import the parts of the
  package. Import the submodule you use: `import itj.toy`, `import itj.observables`, `from itj.io import load`.
- `No module named 'itj'` or `'torch'` in a notebook: the kernel is not the environment, see above.
- `No space left on device` during `pip install`: on Linux, PyPI serves a CUDA build of PyTorch of several GB; the
  CPU build (`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`, before the package)
  is about 200 MB. Free space (`python -m pip cache purge` empties the pip cache) or create the environment on another
  disk and link it: `python3 -m venv /path/with/space/inside-the-jet && ln -s /path/with/space/inside-the-jet .venv`.
- `jetgen fetch` is slow or blocked: run the command again, and the downloads resume where they stopped. A teacher
  can also provide the files on a local share, read with `jetgen fetch jetclass --base-url URL`. There `URL` holds the
  folders `val/`, `train/` and `example/` as on the mirror ([jetclass.md](jetclass.md#download)).
- A training is killed or the machine swaps (runs out of memory and becomes very slow) during the week-2 start-of-week
  box. The course models train one after the other and peak at about 9 GB (ParT-small). On a laptop with 8 GB run the
  box overnight with
  nothing else open, and close other sessions that hold `data/t2_detector.parquet`.
- `uproot` or `awkward` not found when reading the ROOT example of JetClass: install the `analysis` extra.
- weaver fails with "CUDA" errors on a laptop: pass `--gpus ''`.
- In every new terminal, activate the environment again: `cd inside-the-jet && source .venv/bin/activate`.
