# inside-the-jet

[![CI](https://github.com/hqucms/inside-the-jet/actions/workflows/ci.yml/badge.svg)](https://github.com/hqucms/inside-the-jet/actions/workflows/ci.yml)
[![Python 3.10 to 3.14](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Representation learning and interpretability for jet tagging**, built up from a toy whose truth we know. At the
LHC, quarks, gluons and the decays of fast W bosons and top quarks show up as jets: collimated sprays of tens of
particles; a jet tagger is a classifier that tells from these particles what started the jet, and the best taggers
today are graph networks and transformers trained on millions of jets. This project asks what such taggers learn
and how to check it. It starts with **A Spray of Hadrons**, a jet generator written from first principles (a hard
process, a parton shower, hadronisation, an underlying event and a simple detector) whose files have the branches of
the JetClass data set plus the full truth of every jet. In the first week the students build and dissect taggers on
the toy, where every claim can be checked against the truth; in the second week they take the same code to JetClass
and to the Particle Transformer (ParT) trained on it. The repository `inside-the-jet` holds the Python package `itj`
(the toy is `itj.toy`, its command line `jetgen`), its documentation and the tasks for the students.

<p align="center">
  <img src="docs/img/ladder.gif" width="560" alt="Animation of one top jet through the four presets of the toy: the three decay partons, the parton shower, the hadrons and the detector particles in the plane of Delta eta and Delta phi">
  <br>
  <em>One top jet through the ladder of the toy, in the plane (Δη, Δφ) about the jet axis: the three decay partons
  (crosses: the b and the two quarks of the W), the parton shower, the hadrons with the underlying event and the
  tracks and calorimeter cells of the detector. The area of a marker is its pT, its colour the prong it comes from
  (grey: the underlying event); the circle is the jet radius R = 0.8.</em>
</p>

## Roadmap

| stage | content                                                                                                   | status               |
|-------|-----------------------------------------------------------------------------------------------------------|----------------------|
| A     | **A Spray of Hadrons** and JetClass: jets, taggers, representations and interpretability, tasks T0-T7 ([TASKS.md](TASKS.md)) | this release (0.1.1) |
| B     | Symmetry-aware taggers: Lorentz-equivariant networks (LorentzNet, PELICAN, L-GATr) on the toy and on JetClass | next                 |
| C     | Pretraining at scale and foundation models for jets                                                       | later                |
| D     | Research projects                                                                                         | later                |

## The toy jets

<p align="center">
  <img src="docs/img/classes.png" width="720" alt="One QCD jet, one W jet and one top jet of the toy at the detector level, the particles coloured by the prong they come from">
</p>

A jet starts from a seed: a quark or gluon (QCD), a W boson decaying to two quarks or a top quark decaying to a b
quark and a W. An angular-ordered parton shower with a Sudakov veto algorithm turns the partons into a spray of
partons, independent fragmentation turns those into hadrons (with $\pi^0 \to \gamma\gamma$ and B-hadron decays), an
underlying event adds soft particles, and a cartoon of the Delphes CMS card behind JetClass makes tracks and
calorimeter cells. The leading anti-kT jet ($R = 0.8$) is stored with the 41 branches of JetClass and the truth: the
prongs, the final partons, every shower emission, the primary Lund plane and the origin of every particle. The stages
are switched on step by step along a ladder of presets (`partons`, `shower`, `hadrons`, `detector`); files made with
the same seed hold the same events at every preset. The model, with all equations and derivations, is in
[docs/physics.md](docs/physics.md).

| parameter                              | default (`detector`)                                                      |
|----------------------------------------|---------------------------------------------------------------------------|
| processes                              | QCD (75 % quark, 25 % gluon), W → qq̄′, t → bW → bqq̄′, in equal numbers; quark-only and gluon-only QCD on request |
| seed $p_T$, rapidity                   | $p_T^{-4.5}$ on [450, 1100] GeV (JetClass window 500-1000 GeV applied at analysis time), $\lvert y\rvert < 2$ |
| W and top                              | $m_W$ = 80.4 GeV, $\Gamma_W$ = 2.085 GeV; $m_t$ = 172.5 GeV, $\Gamma_t$ = 1.42 GeV; W helicity in top decays; every decay quark inside the jet |
| shower                                 | angular ordering, running $\alpha_s$ ($\alpha_s(M_Z)$ = 0.118, one loop), cutoff $k_t$ = 1 GeV, colour coherence |
| hadronisation                          | Field-Feynman chains, $a$ = 0.77, $k_T$ width 0.24 GeV per component |
| species                                | $\pi^0$ share 0.34, neutral hadrons 0.13, charged the rest (15 % kaons), calibrated to JetClass |
| B hadrons                              | $c\tau$ = 0.46 mm, $\langle z_B\rangle$ = 0.7, 7 decay products, semileptonic 11 % per lepton flavour |
| underlying event                       | 0.7 particles per unit area in $(y, \phi)$, $p_T$ = 0.3 GeV + Exp(0.7 GeV) |
| tracks                                 | efficiency 0.70 / 0.95 below / above 1 GeV ($\lvert\eta\rvert \le 1.5$), resolution 1 % ⊕ $2 \times 10^{-4} p_T$, impact parameters |
| calorimeters                           | ECAL cells 0.0174, HCAL cells 0.087 in $(\eta, \phi)$, Delphes CMS resolutions and thresholds, perfect particle flow |
| jet                                    | anti-kT $R$ = 0.8; soft drop $z_\mathrm{cut}$ = 0.1, $\beta$ = 0; N-subjettiness $\beta$ = 1 with WTA-kT axes |
| speed                                  | about 850 jets per second and core at `detector` (2016 CPU)            |

## Installation

Python 3.10 or newer on Linux, macOS or Windows; a laptop is enough (CPU only, no GPU needed).

```bash
git clone https://github.com/hqucms/inside-the-jet.git
cd inside-the-jet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux without a GPU: the small CPU build
python -m pip install -e ".[analysis,dev]"
python -m pytest                                 # fast tests (the "deselected" ones are the slow tests)
python -m ipykernel install --user --name inside-the-jet --display-name "Python (inside-the-jet)"
```

The last line registers the environment as the Jupyter kernel `Python (inside-the-jet)` that the notebooks use.
`python -m pip install -e ".[weaver]"` adds weaver, the training framework of ParT (Python 3.10 to 3.13; only task T4.6
needs it). The environment takes about 2 GB; the task data about 1.0 GB (week 1) and 1.9 GB (week 2, downloads
included): keep about 6 GB free. Plan on 16 GB of memory: the two ParT-small trainings of T2.4 peak at about 9 and 4 GB
(13 GB together), a weaver run at 6-7 GB; with 8 GB run them one at a time. Windows, a smaller CPU-only PyTorch, GPUs
and Colab, Apple Silicon, `uv`, clusters and common problems: [docs/install.md](docs/install.md).

## Quick start

On the command line (`python -m itj.toy` works the same way):

```bash
jetgen presets                                     # the physics ladder and the defaults
jetgen generate -n 10000 --seed 1                  # -> data/toy_detector_qcd-wqq-tbqq_n10000_seed1.parquet
jetgen info data/toy_detector_qcd-wqq-tbqq_n10000_seed1.parquet     # counts, configuration, provenance
jetgen display data/toy_detector_qcd-wqq-tbqq_n10000_seed1.parquet --jet 0 -o jet0.png
jetgen display --ladder --process tbqq -o ladder.png                 # one top jet at the four presets
jetgen generate --help                             # every option
```

In Python:

```python
import itj.features
from itj.io import load, save
from itj.models import build_model
from itj.toy import generate, preset
from itj.train import fit

data = generate(10_000, config=preset("detector"), seed=1)
save(data, "data/toy.parquet")
data = load("data/toy.parquet")

data["jet_sdmass"]  # soft-drop mass of every jet, [10000]
data["part_px"]  # Ragged: the px of all particles; jet i: data["part_px"][i]
data["truth_part_prong"]  # Ragged: the prong each particle comes from (-1: underlying event)
train, val, test = itj.features.split(data)  # test = first 20 %, val = next 10 %, train = the rest
model = build_model("pfn")  # a Particle Flow Network on the "kin" features of ParT
history = fit(model, itj.features.JetDataset(train), itj.features.JetDataset(val), epochs=2)
```

[examples/quickstart.py](examples/quickstart.py) is a complete short script, and
[notebooks/00_quickstart.ipynb](notebooks/00_quickstart.ipynb) a guided first look at the data.

## The data

One Parquet file per run holds one row per jet. The first 41 columns are those of JetClass, with the same names,
dtypes and meaning (`part_*` for the particles, `label_*`, `jet_*`, `aux_*`); the toy adds its truth in columns that
all start with `truth_` (per jet, per particle, per prong, per parton, per shower emission and per Lund emission), so
that a tagger that reads `part_*` can never see the answer. The configuration and the provenance (seed, software
versions, time per stage) are stored in the file. Types, units, the exact identities every file satisfies and reading
examples are in [docs/data_format.md](docs/data_format.md).

## JetClass and ParT

The same reader, the same network inputs and the same weaver data configuration work on the toy files and on the
JetClass files: `itj.io.load` opens both, `itj.features.inputs` computes the inputs of ParT exactly as weaver does,
and `src/itj/configs/jets3_kin.yaml` runs weaver unchanged on either. `jetgen fetch jetclass` downloads single
JetClass files (100 000 jets per class, 86-166 MB each) from the CERN mirror with size and MD5 checks, and
`jetgen fetch models` the trained ParT and ParticleNet of the JetClass paper, which `itj.models.load_pretrained` runs
without weaver, with access to the attention, the pair bias and every intermediate representation. JetClass, ParT and
weaver in one page: [docs/jetclass.md](docs/jetclass.md).

<p align="center">
  <img src="docs/img/toy_vs_jetclass.png" width="720" alt="Distributions of the soft-drop mass, tau21, tau32 and the multiplicity for QCD, W and top jets of the toy at the detector level and of JetClass">
  <br>
  <em>The toy at the detector level (lines) against the JetClass validation files (filled), for QCD, W and top jets
  with 500 &lt; pT &lt; 1000 GeV.</em>
</p>

## Tasks

[TASKS.md](TASKS.md) holds the tasks of stage A, two weeks of about three hours of hands-on work a day. Week 1 on the
toy: the set-up and the JetClass dictionary (T0), jets and their observables against leading-log QCD (T1), a ladder
of taggers from cuts to transformers (T2) and what their representations contain (T3). Week 2: JetClass, the
pretrained ParT and weaver (T4), interpretability with attributions, attention and IRC tests (T5), self-supervised
learning (T6) and a bonus on the way to a research project (T7). `python examples/make_task_data.py --week 1` and
`--week 2` make the data of each week on any operating system.

## Documentation

- [docs/physics.md](docs/physics.md): the toy, its parameters and the derivations the tasks use
- [docs/observables.md](docs/observables.md): jet observables, clustering, IRC safety and how to test it
- [docs/data_format.md](docs/data_format.md): the Parquet files, the JetClass branches and the truth
- [docs/models.md](docs/models.md): network inputs, the models, the pretrained ParT and ParticleNet
- [docs/methods.md](docs/methods.md): metrics, probes, CKA, attributions, attention, decorrelation, self-supervision
- [docs/jetclass.md](docs/jetclass.md): JetClass, ParT and weaver in one page
- [docs/references.md](docs/references.md): the literature behind the defaults and the reading list
- [docs/glossary.md](docs/glossary.md): the terms of jet physics and machine learning used in the tasks
- [docs/install.md](docs/install.md): installation details
- [examples/weaver](examples/weaver/README.md): weaver on the toy and on JetClass
- [examples/cluster](examples/cluster/README.md): large data sets and GPU trainings on a cluster
- [NOTICE.md](NOTICE.md): third-party code and data
- [CHANGELOG.md](CHANGELOG.md): changes between versions

## Development

`python -m pytest` runs the fast tests, `python -m pytest -m slow` the slow ones (physics validation, timing,
notebooks), `python -m pytest -m weaver` the comparisons with weaver (needs the `weaver` extra) and
`python -m pytest -m network` the tests that download (JetClass URLs and the pretrained models). `pre-commit run
--all-files` runs the lint and format checks (install the hooks once with `pre-commit install`). Continuous integration
runs the fast and slow tests on Linux (Python 3.10 to 3.14), macOS and Windows, the weaver and fastjet comparisons in
a job of their own, and the network tests once a week. Reproducibility of the simulations:
[docs/physics.md](docs/physics.md#reproducibility); clusters: [examples/cluster](examples/cluster/README.md); changes
between versions: [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use this material, please cite it as described in [CITATION.cff](CITATION.cff) (GitHub shows it under "Cite
this repository"). If you use JetClass or ParT, please also cite H. Qu, C. Li and S. Qian, *Particle Transformer for
Jet Tagging*, [arXiv:2202.03772](https://arxiv.org/abs/2202.03772).

## License

[MIT](LICENSE). Third-party code and data (weaver-core, particle_transformer, a sample of JetClass) and their licences
are listed in [NOTICE.md](NOTICE.md).
