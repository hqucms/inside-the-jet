# Tasks: A Spray of Hadrons

Two weeks of questions: the first on a toy whose truth we know, the second on JetClass. They prepare a research
project on what jet taggers learn and how to check it. The toy, **A Spray of Hadrons**, writes jets with the branches
of JetClass plus the truth that only a simulation has (the prongs, every shower emission, the origin of every
particle), so that every claim about a tagger can be checked on the toy before it is made about a real one.

- **Core tasks:** T0 to T6. **Bonus:** T7. **Week 1** (the toy): T0 to T3. **Week 2** (JetClass and the toy): T4 to T7.
- The question IDs (T0.1, T2.3, ...) are fixed: we go through the answers by ID together at the end of each week.
- **Assumed:** Python, NumPy, matplotlib and one PyTorch tutorial (the dl4hep notebooks 1-3 are enough). The times
  assume this; if you have never written a PyTorch training loop, add about 2 hours for T2.3.
- Background: the toy and the derivations the tasks use are in [docs/physics.md](docs/physics.md), the observables
  and IRC safety in [docs/observables.md](docs/observables.md), the network inputs and the models in
  [docs/models.md](docs/models.md), the analysis methods in [docs/methods.md](docs/methods.md), the files in
  [docs/data_format.md](docs/data_format.md), JetClass, ParT and weaver in [docs/jetclass.md](docs/jetclass.md), the
  reading list (with the papers to read first in each week) in [docs/references.md](docs/references.md#reading-list)
  and the terms in [docs/glossary.md](docs/glossary.md). [notebooks/00_quickstart.ipynb](notebooks/00_quickstart.ipynb)
  and [notebooks/01_jetclass.ipynb](notebooks/01_jetclass.ipynb) are guided first looks at the toy and at JetClass.
- **Where to run.** A laptop is enough for every core question (Linux, macOS or Windows; CPU only; at most about 15
  minutes per training run; the times below are for a 2016 six-core CPU, and laptops from 2022 on are 1.5-3 times
  faster). Every `jetgen` command below is a single line and works in any shell, and
  `python examples/make_task_data.py --week 1` (or `--week 2`) makes the data sets of each week. Parts marked
  *GPU (optional)* run on a GPU, on Colab or on a cluster ([examples/cluster](examples/cluster/README.md)).
- **Write it, then check it.** A question marked "(write your own)" asks you to implement a tool first and then to
  compare it with the reference in `itj` in one `allclose` line. Later tasks use the reference, so a bug in your
  version cannot spoil them.
- **Predict first.** Where a question says *Predict first*, write a one-sentence prediction into your answers before
  you run anything, and come back to it afterwards.
- **Minimum path.** If time is short, these can wait: T1.2(b), T2.6 beyond the PFN curve, T3.6 except
  `t3_w80_noue`, T4.6(b), T6.6, T6.7, the proposals beyond one per week and every part marked *Optional*. T3.3 (the
  EFN filters) stays.

## What to hand in

1. **Figures** (PDF or PNG) with the question ID in the file name, e.g. `T2.5_roc_all.pdf`.
2. **Short answers**: a few sentences per question ID, with the numbers you measured (and their errors), in one
   Markdown file or notebook.
3. **Your code** (scripts or notebooks) and the seeds of your data sets and trainings, so that every figure can be
   made again.
4. **`results_table.csv`**: one row per (model, training data, test data), written with `itj.metrics.append_result`
   (the columns are `itj.metrics.RESULT_COLUMNS`), kept from T2 to T6 and committed on your branch.

Trained models go to `models/toy/` and `models/jetclass/` (both ignored by git). **Proposals** (T1.7, T2.7, T3.7,
T5.7; about 30 minutes each, with a figure): hand in at least one per week. Make your own plots too: whatever you find
telling, surprising or useful.

**Figure checklist.** A figure should be readable on its own: axis labels with units, a legend that does not hide the
data, and a title or caption that says what is shown (preset, process, number of jets, $p_T$ range, model, training
size). Show statistical uncertainties where they matter and use a log axis when values span several orders of
magnitude. In addition, for this course:
- ROC curves as $1/\epsilon_B$ on a log axis against $\epsilon_S$, with the working point and the size of the test
  set. A rejection above $N_B/10$ ($N_B$ background jets in the test set) has a statistical error of more than 30 %:
  say so and show the bootstrap error.
- Jet displays and jet images with equal axes in $(\Delta\eta, \Delta\phi)$ (`ax.set_aspect("equal")`).
- Lund planes as densities per jet and per unit area, with the number of jets and the cutoff line.
- Attribution maps with a colour bar that says whether the values are signed or absolute.
- Every learning result with its training size, number of epochs and seed(s).

**Where to work.** Keep your scripts on your own branch of the repository (`git switch -c <your-name>`) or in a
folder outside it. Data and models are not committed: `data/`, `models/`, `runs/`, `*.parquet` and `*.pt` are ignored
by git.

## Notation

| symbol | meaning |
|--------|---------|
| $p_T$, $y$, $\eta$, $\phi$ | transverse momentum, rapidity, pseudorapidity, azimuth |
| $\Delta\eta_i$, $\Delta\phi_i$ | coordinates of particle $i$ relative to the jet axis (`part_deta`, `part_dphi`; $\Delta\eta$ with the JetClass sign, [physics.md](docs/physics.md#units-and-coordinates)) |
| $\Delta R$ | distance in $(y, \phi)$, $\Delta R^2 = \Delta y^2 + \Delta\phi^2$ |
| $R$ | jet radius, 0.8 |
| $z_i = p_{T,i}/p_{T,J}$ | momentum fraction of particle $i$ in the jet |
| $z$, $\theta$, $k_t$ | energy fraction, opening angle and transverse momentum of a shower branching (`truth_emis_z`, `truth_emis_theta`, `truth_emis_kt`) |
| $\Delta$, $k_t$, $z$ (Lund) | opening angle, transverse momentum and momentum fraction of a primary declustering (`truth_lund_*`, `itj.observables.lund_primary`) |
| $C_F = 4/3$, $C_A = 3$ | colour factors of quarks and gluons; $C_R$ the one of the emitter |
| $\alpha_s$ | strong coupling (`alphas_value`) |
| $\Delta(t_0, t)$ | Sudakov factor: probability of no emission between $t_0$ and $t$ |
| $m_J$, $m_\mathrm{SD}$ | jet mass, soft-drop mass (`jet_sdmass`) |
| $\rho = m^2/(p_T^2R^2)$ | normalised jet mass |
| $\tau_N$, $\tau_{21}$, $\tau_{32}$ | N-subjettiness (`jet_tau1` ... `jet_tau4`) and its ratios $\tau_2/\tau_1$, $\tau_3/\tau_2$ |
| $e_2$, $e_3$, $D_2$, $C_2$ | energy correlation functions and their ratios $D_2 = e_3/e_2^3$, $C_2 = e_3/e_2^2$ |
| $N$ | multiplicity, the number of particles (`jet_nparticles`) |
| $\epsilon_S$, $\epsilon_B$ | signal and background efficiency of a cut on a score |
| $R_{50}$ | background rejection $1/\epsilon_B$ at $\epsilon_S = 0.5$ |
| $s(x)$ | score of a tagger for the jet $x$ |
| $h$ | embedding of a jet (`model.embed`, the input of the last linear layer) |
| $A$, $U$ | attention weights of a transformer head; the pair bias of ParT |
| IG | integrated gradients |
| $\epsilon$, $\delta$, $\lambda$ | IRC tests: soft $p_T$ fraction, collinear separation, momentum share of a split |
| $\tau_T$ | temperature of the NT-Xent loss (not an N-subjettiness) |

---

## T0 Set-up (about 1 hour)

**Goal:** a working installation, one jet at every level of the ladder, the cost of a simulation and the map between the
toy and JetClass, with the downloads of week 2 started.

**T0.1** Clone the repository, create a virtual environment, install the package, run the tests and generate a first
small data set.

```bash
git clone https://github.com/hqucms/inside-the-jet.git
cd inside-the-jet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[analysis,dev]"
python -m pytest                 # ends with "... passed, ... deselected": the deselected ones are the slow tests
python -m ipykernel install --user --name inside-the-jet --display-name "Python (inside-the-jet)"
jetgen presets                   # the physics ladder and the default configuration
jetgen generate -n 3000 --seed 1   # -> data/toy_detector_qcd-wqq-tbqq_n3000_seed1.parquet
```

The repository is `inside-the-jet`; the package it installs is `itj` (the toy is `itj.toy`, its command line
`jetgen`, the same as `python -m itj.toy`). Windows, a smaller CPU-only PyTorch, Apple Silicon and common problems:
[docs/install.md](docs/install.md).

**T0.2** One jet up the ladder. Draw one W jet at the four presets, inspect the file of T0.1 and look at one jet in
Python.

```bash
jetgen display --ladder --process wqq --seed 1 --jet 0 -o T0.2_ladder.png
jetgen info data/toy_detector_qcd-wqq-tbqq_n3000_seed1.parquet
```

```python
from itj.io import load

data = load("data/toy_detector_qcd-wqq-tbqq_n3000_seed1.parquet")
print(data)  # JetData(3 000 jets: qcd 1 002, wqq 999, tbqq 999; preset detector; ...; with truth)
jet = data.jet(0)
print(jet.jet)  # the per-jet branches: jet_*, label_*, aux_*, truth_*
print(jet.particles_frame())  # one row per particle: part_* and truth_part_*
```

*Predict first:* how many particles does the W jet have at the detector level? Then: how many particles are there at
each level; where are the prongs; which detector particles come from which prong (`truth_part_prong`) and which from
the underlying event; what fraction of the jet $p_T$ do the 5 hardest particles carry? Why does the number of
particles grow by a large factor from `shower` to `hadrons` and shrink again at `detector` (in this toy)?

**T0.3** Timing. Time the generation of 1000, 3000 and 10 000 jets for every preset (`data.provenance["wall_time_s"]`)
and one run with `--workers 4`. Plot the time against $N$ per preset. Which stage dominates
(`data.provenance["stage_times_s"]`, `jetgen info --provenance`)? Extrapolate to the data of T1 to T3 below. About 2
minutes of CPU. *Optional:* run `jetgen validate -n 3000` and compare with
[physics.md](docs/physics.md#calibration-and-validation).

```python
from itj.toy import generate, preset

for name in ("partons", "shower", "hadrons", "detector"):
    data = generate(3000, config=preset(name), seed=3)
    print(name, f"{data.provenance['wall_time_s']:.2f} s", data.provenance["stage_times_s"])
```

**T0.4** The JetClass dictionary and the downloads of week 2. First start the downloads, which run while you work
(about 0.4 GB, plus 136 MB for the bonus), and, if you want to run weaver in T4.6, install its extra (Python 3.10 to
3.13, not on Intel Macs):

```bash
python examples/make_task_data.py --week 2 --only fetch
python -m pip install -e ".[weaver]"
python -c "import weaver"
```

Then, with [docs/jetclass.md](docs/jetclass.md), [docs/data_format.md](docs/data_format.md) and `jetgen info --columns`,
make a table: every column of a toy file and its JetClass meaning; the JetClass columns that the toy fills trivially
(the H labels, the aux match code); the columns that only the toy has (`truth_*`). Explain the sign of `part_deta`
with a jet at $\eta = -1.2$. Name the two truth columns you expect to be most useful for interpretability, and say why.

**Deliverables:** the summary line of the tests (T0.1); `T0.2_ladder.png` and the table of multiplicities per level
(T0.2); `T0.3_timing.pdf` (T0.3); the dictionary table (T0.4).

**Hints**
- `python -m itj.toy` works when `jetgen` is not found (the environment is not activated).
- `--workers auto` uses up to 8 processes; the output does not depend on the number of workers.
- `--only fetch` selects the commands of the week-2 block whose command line contains "fetch".
- `jetgen generate --help` lists every option with its default, grouped by stage.

## Data for T1 to T3

With the installation working, make the data of the first week. All files go to `data/` in the repository root, and
every path in this document is relative to the repository root (a notebook in `notebooks/` reads the same file as
`../data/t1_detector.parquet`). `python examples/make_task_data.py --week 1` runs exactly these commands on any
operating system with `--workers auto`; it keeps files that already exist (`--overwrite` makes them again,
`--dry-run` only prints them, `--only TEXT` runs the commands whose command line contains TEXT). Together they make
about 510 000 jets and 1.0 GB, in about 9 minutes on one 2016 core or 3-4 minutes with 4 workers. On a
cluster, run them in a job or an interactive session, not on the login node
([examples/cluster](examples/cluster/README.md)).

Files made with the same seed share their hard processes jet by jet: the `t1_*` ladder holds the same events at every
preset, and `t3_w80_noue`, `t3_w80_centre` and `t3_w80_nocoh` are `t3_w80` with one thing changed
([physics.md](docs/physics.md#reproducibility)). Such paired samples are made for differences, not for pooling. A
change of the hard process (for example `--w-mass`) gives an independent sample, even with the same seed.

```bash
# T1.1, T1.3-T1.6: the physics ladder, 10 000 jets per process, the same events at every preset
jetgen generate --preset partons -n 30000 --seed 11 -o data/t1_partons.parquet
jetgen generate --preset shower -n 30000 --seed 11 -o data/t1_shower.parquet
jetgen generate --preset hadrons -n 30000 --seed 11 -o data/t1_hadrons.parquet
jetgen generate --preset detector -n 30000 --seed 11 -o data/t1_detector.parquet
# T1.4: the same jets without the underlying event
jetgen generate --preset detector --no-ue -n 30000 --seed 11 -o data/t1_detector_noue.parquet
# T1.2-T1.4: 600 GeV quark and gluon jets with a fixed coupling, for the leading-log formulas
jetgen generate --preset shower --process q --alphas-mode fixed --alphas-value 0.12 --pt-spectrum fixed --pt-fixed 600 -n 20000 --seed 12 -o data/t1_lund_q.parquet
jetgen generate --preset shower --process g --alphas-mode fixed --alphas-value 0.12 --pt-spectrum fixed --pt-fixed 600 -n 20000 --seed 12 -o data/t1_lund_g.parquet
# T1.1, T1.3, T1.4, T2.7: quark and gluon jets with the running coupling through the ladder
jetgen generate --preset shower --process q,g -n 20000 --seed 13 -o data/t1_qg_shower.parquet
jetgen generate --preset hadrons --process q,g -n 20000 --seed 13 -o data/t1_qg_hadrons.parquet
jetgen generate --preset detector --process q,g -n 60000 --seed 13 -o data/t1_qg_detector.parquet
# T2-T6: the training set: QCD, W and top at detector level, 50 000 jets each (split with itj.features.split)
jetgen generate -n 150000 --seed 21 -o data/t2_detector.parquet
# T3.5: W jets at four masses (independent samples: the mass changes the hard process)
jetgen generate --process wqq --w-mass 60 -n 10000 --seed 35 -o data/t3_w60.parquet
jetgen generate --process wqq -n 10000 --seed 35 -o data/t3_w80.parquet
jetgen generate --process wqq --w-mass 100 -n 10000 --seed 35 -o data/t3_w100.parquet
jetgen generate --process wqq --w-mass 120 -n 10000 --seed 35 -o data/t3_w120.parquet
# T3.6: the jets of data/t3_w80.parquet with one thing changed (paired samples)
jetgen generate --process wqq --no-coherence -n 10000 --seed 35 -o data/t3_w80_nocoh.parquet
jetgen generate --process wqq --no-ue -n 10000 --seed 35 -o data/t3_w80_noue.parquet
jetgen generate --process wqq --cell-position centre -n 10000 --seed 35 -o data/t3_w80_centre.parquet
```

Every file is read with `load`:

```python
import itj.features
from itj.io import load

data = load("data/t2_detector.parquet")
train, val, test = itj.features.split(data)  # test = first 20 %, val = next 10 %, train = the rest
data["jet_sdmass"], data["part_px"]  # a per-jet array [N]; a Ragged column (flat values and offsets)
p4, mask = test.head(2000).padded_p4()  # (2000, P, 4) float64 (px, py, pz, E) and (2000, P) bool
for chunk in test.chunks(2000):  # anything that builds (B, N, N) arrays: work on chunks
    ...
```

## T1 Jets and their observables (about 4 hours)

**Goal:** what a jet is made of at each level of the ladder, what the classic observables measure and why, and which
of them are IRC safe, checked against leading-log formulas and against the stored branches.

**T1.1** The ladder and colour charge. Data: `data/t1_partons.parquet`, `data/t1_shower.parquet`,
`data/t1_hadrons.parquet`, `data/t1_detector.parquet`; `data/t1_qg_shower.parquet`, `data/t1_qg_hadrons.parquet`,
`data/t1_qg_detector.parquet`. Using the paired events, plot the multiplicity, the jet mass and
$p_{T,J}/p_{T,\mathrm{seed}}$ (`truth_seed_pt`) per preset and process. (a) Why is the QCD jet mass 0 at `partons`, and
the W and top mass the Breit-Wigner mass (`truth_mass`)? The containment rule should keep every prong in the jet: check
it. (b) Which step changes $N$ most, in this toy? (c) Where does the jet lose $p_T$, and why mostly QCD jets at `shower`
(hint: compare the angle at which the QCD shower starts with $R$)? (d) The ratio $\langle N_g\rangle/\langle N_q\rangle$
at three levels against $C_A/C_F = 9/4$: explain the difference ([physics.md](docs/physics.md#multiplicity)).

**T1.2** The Sudakov factor. Data: `data/t1_lund_q.parquet`, `data/t1_lund_g.parquet`. (a) Plot the cumulative
distribution of the angle of the first emission of the quark jets (`truth_emis_theta` where `truth_emis_depth == 0`;
jets without any emission count as "no emission above the cutoff") against the double-log Sudakov factor

```math
-\ln\Delta(\theta_0, \theta) = \frac{\alpha_s C_F}{\pi}\left[\ln^2\frac{E\theta_0}{k_{t,\mathrm{cut}}} - \ln^2\frac{E\theta}{k_{t,\mathrm{cut}}}\right]
```

([physics.md](docs/physics.md#sudakov-factor)). Where does it fail, and why? (b) *Optional:* derive it[^sudakov], and
repeat for gluons.

**T1.3** The Lund plane (write your own density). Data: `data/t1_lund_q.parquet` and `data/t1_lund_g.parquet` (truth),
then `data/t1_detector.parquet`. (a) Histogram the primary emissions `truth_lund_*` as a density per jet and per unit
area in $(\ln 1/\Delta, \ln k_t)$, with the cutoff line. Measure the plateau in the box $\ln(1/\Delta) \in [0.5, 2.5]$,
$\ln(k_t/\mathrm{GeV}) \in [0.5, 2.0]$ for quarks and gluons, with errors, and compare it with the leading-log
density $2\alpha_s C_R/\pi$[^lund] (evaluate it). (b) The ratio g/q against $C_A/C_F$. (c) The reconstructed plane of
the detector-level jets (`itj.observables.lund_primary`; optionally walk the C/A tree of
`itj.cluster.cluster(p4, mask, algorithm="ca")` yourself) against the truth: where do hadronisation and the detector
move the density (low $k_t$, small $\Delta$: the cell size)? About a minute of CPU for the C/A clustering of 60 000
jets. *Optional:* the same at `shower` and `hadrons`; the plane of W jets (where is the W splitting?). Keep this plane
in mind for T5.4: the pair features of ParT ($\ln k_T$, $\ln z$, $\ln\Delta$) are the Lund coordinates of every pair
of particles.

**T1.4** Mass and soft drop (write your own mass). Data: `data/t1_lund_q.parquet`, `data/t1_lund_g.parquet`,
`data/t1_detector.parquet`, `data/t1_detector_noue.parquet`, `data/t1_qg_detector.parquet`. Compute $m_J$ yourself and
compare it with `itj.observables.mass`. (a) On the fixed-coupling files: plot $\rho\,d\sigma/d\ln\rho$ and compare the
position of its peak in $\ln(1/\rho)$ with the double-log value and with the value that includes the hard-collinear
term (both in [physics.md](docs/physics.md#jet-mass); a two-line check, no derivation needed[^peak]); the peak is
broad, so quote how much its position moves with the bin width and the fit window. Then on
`t1_qg_detector`: how far do hadronisation, the underlying event and the detector move the peak? (b) $m$ and
$m_\mathrm{SD}$ per process; with the paired file without underlying event: the shift of $m$ and of $m_\mathrm{SD}$.
(c) Why does the top $m_\mathrm{SD}$ peak sit below the top mass (think of the neutrinos of semileptonic B decays,
radiation out of the cone and the detector), and why are the W and top peaks wider than the Breit-Wigner width?

**T1.5** Counting prongs: $\tau_N$ and $D_2$ (write your own $\tau_N$ and $e_2$, $e_3$). Data:
`data/t1_detector.parquet`, `data/t1_partons.parquet`. With the axes of `itj.observables.subjet_axes`, compute
$\tau_1$ to $\tau_3$ and reproduce `jet_tau1` to `jet_tau3` to $10^{-5}$. Compute $e_2$, $e_3$ and $D_2$ ($\beta = 1$)
with $e_3 = \mathrm{tr}\bigl((ZA)^3\bigr)/6$ and check them in one `allclose` against `itj.observables.ecf` on 10
jets. Then: $\tau_{21}$, $\tau_{32}$ and $D_2$ per process at `detector`; why are they degenerate at `partons` (what is
$\tau_2$ of a two-particle jet, why is $\tau_1 = 0$ for one particle, and why are $m_\mathrm{SD}$ and $D_2$ undefined
there)? Use power counting[^power] to explain why $D_2 \gg 1$ for one-prong jets and $D_2 = O(1)$ for two-prong jets
([physics.md](docs/physics.md#power-counting)). Plot $\tau_{21}$ against `truth_dr_min` for the W jets.
*Optional:* $D_2$ at $\beta = 2$; the other levels.

**T1.6** IRC safety by hand. Data: 1000 jets of `data/t1_detector.parquet`. Perturb them with `itj.augment.add_soft`
($\epsilon = 10^{-1}, \ldots, 10^{-6}$) and `itj.augment.split_collinear` ($\lambda = 0.5, 0.1$;
$\delta = 10^{-1}, \ldots, 10^{-5}$), both with `substructure=True`, and plot the mean relative change of $m$,
$m_\mathrm{SD}$, $\tau_2$, $\tau_{21}$, $e_2$, $D_2$, $N$, $p_T^D$ and the leading fraction against $\epsilon$ and
$\delta$. Classify each observable as safe, Sudakov safe or unsafe ([observables.md](docs/observables.md)). Why can a
scan on single jets with many particles not tell an IRC-safe observable from a Sudakov-safe one (hint: which limit
does IRC safety concern, and which one does the scan take)? *Optional:* a two-particle
jet where $\tau_{21} = 0/0$. These curves are the template of T5.5.

**T1.7** Your own proposal (about 30 minutes, with a figure). For example: the dead cone of b quarks
(`--dead-cone`), the W decay angle (`truth_cos_theta_star`), pull and colour coherence (`--no-coherence`), the
displacement of B-hadron decay products in top jets (`truth_part_from_b`, `part_d0val`), angularities for quark and
gluon jets.

**Deliverables:** the figures named by question ID (`T1.1_ladder.pdf`, `T1.1_qg_ratio.pdf`, `T1.2_sudakov.pdf`,
`T1.3_lund_truth.pdf`, `T1.3_lund_reco.pdf`, `T1.4_mass.pdf`, `T1.4_ue_shift.pdf`, `T1.5_prongs.pdf`,
`T1.5_tau21_vs_dr.pdf`, `T1.6_irc.pdf`); the two Lund plateaus with their errors (T1.3) and the positions of the mass
peaks with their errors (T1.4).

**Hints**
- The error of a histogram density is $\sqrt{n}/(N_\mathrm{jet}\,\mathrm{area})$ per bin
  (`itj.observables.lund_density` returns both).
- Guard $\tau_1 = 0$ (and $\tau_2 = 0$) before taking ratios; use logarithmic bins in $\rho$ and leave out the jets
  with $m = 0$ (many fixed-coupling quark jets have a single parton inside $R$).
- Work on chunks of 2000 jets for anything that builds $(B, N, N)$ arrays (`for chunk in data.chunks(2000)`), and use
  `@` (matmul) for $e_3$, not a three-operand `np.einsum` (much more memory and about 7 times slower).
- `itj.augment` returns new `part_*` columns: call the observables again on the result.
- `data.select(data.process_names == "wqq")` keeps the W jets.

[^sudakov]: Take the soft limit of $P_{qq}(z) = C_F(1 + z^2)/(1 - z)$ and write $k_t \approx (1 - z)E\theta$. The
    probability of no emission is the exponential of minus the integral of the emission density over the region
    between $\theta$ and $\theta_0$ with $k_t > k_{t,\mathrm{cut}}$.
[^lund]: Write the soft-collinear emission density $\frac{\alpha_s}{2\pi}\frac{d\theta^2}{\theta^2}P(z)\,dz$ in the
    variables $\ln(1/\theta)$ and $\ln k_t$ at fixed $\theta$, with $k_t = (1 - z)E\theta$.
[^peak]: With $\Sigma(\rho) = e^{-\mathcal{R}(\rho)}$, the distribution in $L = \ln(1/\rho)$ is
    $\mathcal{R}'e^{-\mathcal{R}}$; set its derivative to zero.
[^power]: For a one-prong jet, $e_2$ comes from one soft-collinear emission. What does $e_3$ need beyond that
    emission, and how does it scale with $e_2$? For a two-prong jet, $e_2$ is fixed by the hard splitting; what does
    $e_3$ need?

## T2 Taggers, from cuts to transformers (about 5 hours)

**Goal:** a ladder of taggers on the same jets and one metric, to see what each step buys and what it costs.

All questions of T2 use `data/t2_detector.parquet`, split with `itj.features.split` (train 105 000, val 15 000, test
30 000 jets). The taggers have three outputs (QCD, W, top); compare W against QCD and top against QCD with the
one-vs-one scores of `itj.metrics.one_vs_one`, and report the AUC and $R_{50}$ with bootstrap errors. Save every
model to `models/toy/` and add a row to `results_table.csv` for each.

**T2.1** Cuts. A window in $m_\mathrm{SD}$ plus a cut on $\tau_{21}$ (W) and a window in $m_\mathrm{SD}$ plus a cut on
$\tau_{32}$ (top), with the windows optimised on the val part; the ROC curve by scanning the cut on $\tau_N$; AUC and
$R_{50}$ on the test part. *Predict first:* is $D_2$ or $\tau_{21}$ the better second variable for W jets? Seconds of
CPU.

**T2.2** Boosted trees. Train scikit-learn's `HistGradientBoostingClassifier` on `itj.features.jet_features` (the 10
JetClass `jet_*` branches plus $\tau_{21}$ and $\tau_{32}$), then add your $D_2$ of T1.5. Measure the permutation
importance. Does `jet_phi` help (it should not), and why is it among the inputs at all? Does $N$ help beyond the
IRC-safe observables, and why could that be dangerous on real data (you will see in T4.5)? Less than a minute of CPU.

**T2.3** An MLP and your first training loop (write your own). In plain PyTorch: standardise the 12 inputs of T2.2,
train an MLP with mini-batches and AdamW, validate after each epoch, stop early, and plot the train and val loss.
Compare it with the BDT on the same inputs: why can a BDT beat an MLP on 12 features? Less than a minute of CPU. From
here on you may use `itj.train.fit` instead of your own loop.

**T2.4** Deep Sets: PFN and EFN (write your own). **First** start the two transformer trainings of T2.5 in a second
terminal; each uses half of the cores and takes about 20-25 minutes when both run (12 and 9 minutes alone on six
threads):

```bash
python examples/train_tagger.py --model partsmall --data data/t2_detector.parquet --train-size 30000 --epochs 5 --out models/toy/partsmall
python examples/train_tagger.py --model partsmall_nopair --data data/t2_detector.parquet --train-size 30000 --epochs 5 --out models/toy/partsmall_nopair
```

Then write a PFN, $F\bigl(\sum_i \Phi(x_i)\bigr)$ with $\Phi$ on the `kin` features and a masked sum, and an EFN,
$F\bigl(\sum_i z_i\, \Phi(\Delta\eta_i, \Delta\phi_i)\bigr)$, both on `JetBatch` ([models.md](docs/models.md)). Check
that both are invariant under permutations and padding, and that the EFN is exactly collinear safe
(`split_collinear(delta=0)`). Copy the weights of `itj.models.build_model("pfn")` into your PFN (a key map of a few
lines) and check that the outputs agree on one batch. Train your two models on 105 000 jets for 10 epochs (about 2-3
minutes each). What does the PFN see that the EFN cannot?

**T2.5** Graphs and attention. Train `build_model("pnlite")` on 60 000 jets for 5 epochs (about 9 minutes) and collect
`partsmall` and `partsmall_nopair` (30 000 jets, 5 epochs each, started in T2.4). ParT-small is our ParT-lite (0.24 M
parameters; the preset `partlite` is the larger ParT-lite of the dl4hep notebooks). The models see different training
sizes (105 000, 60 000, 30 000 jets): the ranking compares taggers at roughly equal CPU time, and T2.6 separates the
data from the architecture. *Predict first:* the ranking of all your taggers, and what the pair bias $U$ buys. Draw
all ROC curves in one figure (two panels: W and top against QCD) and make a table with the number of parameters and
the training and inference speeds in jets per second. Read [models.md](docs/models.md) while the models train.
*Optional (GPU welcome):* `partlite`, or more training data.

**T2.6** How much data? The AUC (W against QCD) against the training size (1 000, 3 000, 10 000, 30 000, 105 000 jets)
for the BDT and the PFN (3 seeds below 10 000), with ParT-small at 30 000 (from T2.5); plot it against the training
time as well. Train every point for the same number of optimiser steps, at least 2000
(`fit(..., min_steps=2000, patience=5)`), and validate every size on at least 5000 jets: otherwise the small points
measure under-training, not the size of the data. Where do the curves cross? This curve returns in T6.5. About 15
minutes of CPU: run it in the background while you write up T2.5.

**T2.7** Proposals (about 30 minutes): a quark/gluon tagger on `data/t1_qg_detector.parquet` (label set `"qg"`)
against a cut on the multiplicity and against the EFN; the `kinpid` features; training at `hadrons` and testing at
`detector`; training on a flat $p_T$ spectrum (`--pt-spectrum flat`).

**Deliverables:** `T2.1_roc_cuts.pdf` and the first rows of the table (T2.1); `T2.2_importance.pdf` and the ROC curves
(T2.2); `T2.3_loss.pdf` and the ROC curves (T2.3); the ROC curves, rows and `models/toy/pfn.pt`, `models/toy/efn.pt`
(T2.4); `T2.5_roc_all.pdf`, the table of costs, `models/toy/pnlite.pt` and the two ParT-small checkpoints (T2.5);
`T2.6_learning_curve.pdf` (T2.6).

**Hints**
- `itj.metrics.rejection(y, s, 0.5)` gives $R_{50}$ and `itj.metrics.bootstrap` its error. The top $R_{50}$ is often
  limited by a handful of QCD jets: report it with its error, and the rejection at $\epsilon_S = 0.8$ as well.
- Train on the train part, choose on the val part, and report on the test part once.
- `fit(..., max_minutes=15)` stops a run that would take too long; `fit` prints an estimate of the time after 20
  batches.
- `itj.models.save_model(model, "models/toy/pfn.pt")` saves a model with its configuration; `load_model` reads it back.
- If your models of T2 failed, `python examples/train_tagger.py --course-models` trains the set used later (about 35
  minutes): say so in your answers.

## T3 What is inside the representation? (about 4 hours)

**Goal:** measure what the trained taggers represent, against truth that only a toy gives, with the controls that
make such claims honest, and see what a tagger does to the mass distribution.

Models: the PFN, EFN and ParT-small of T2, and an untrained ParT-small (`reset_parameters`) as the control (PN-lite
optional). Data: the test part of `data/t2_detector.parquet` (30 000 jets) unless stated otherwise.

**T3.1** A look at the embeddings. Extract them with `itj.probes.extract` and draw PCA, t-SNE and UMAP projections
(UMAP needs `pip install -e ".[ml]"`; on Python 3.14, where umap may not install, use t-SNE) coloured by the class,
$m_\mathrm{SD}$, `truth_init_pid` (QCD jets only), `truth_n_primary` and the multiplicity. What organises the space?
Seconds of CPU (UMAP about a minute).

**T3.2** Probes against the truth. Train linear probes (`itj.probes.fit_probe`) from each embedding (4 models) to
$m_\mathrm{SD}$, $\tau_{21}$, $D_2$, $N$, gluon against quark (`truth_init_pid == 21`, QCD jets only: never a training
label) and `truth_n_primary`. For each target compare the trained model, the random-weight model and a shuffled-target
control (`itj.probes.control_task`); the selectivity is the score of the trained model minus that of the control. For
$N$, add the baseline of a probe from the IRC-safe observables
$X_\mathrm{IRC} = [m, m_\mathrm{SD}, \tau_1, \tau_2, \tau_3, e_2, e_3, \text{width}]$: is $N$ more predictable from the
embedding than from IRC-safe observables alone? ($\tau_N$ and $e_n$ are IRC safe; their ratios are only Sudakov
safe.) *Predict first:* which target does ParT-small encode best, and does the EFN beat the $X_\mathrm{IRC}$ baseline
for $N$? *Optional:* an MLP probe, PN-lite, `truth_mass`, `truth_dr_min`, probes after every block of ParT
(`extract(..., layers="all")`).

**T3.3** EFN filters. Train an EFN with `build_model("efn", phi_sizes=(100, 100, 8))` (about 2 minutes) and draw its
filters $\Phi_a(\Delta\eta, \Delta\phi)$ with `itj.plot.efn_filters`. Relate the rings and lobes to the prong
separation $\Delta R \approx 2m/p_T$ (evaluate it for W and top jets at 600 GeV, and compare with `truth_dr_min` per
class), as in Fig. 6 of arXiv:1810.05165.

**T3.4** Comparing representations with CKA (write your own linear CKA). Check it against `itj.probes.linear_cka`;
check that it is invariant under orthogonal maps and isotropic scaling, but not under general linear maps. Compute
the CKA (a) between the blocks of ParT-small and (b) between each model and the observables
$X_\mathrm{phys} = [m_\mathrm{SD}, \tau_{21}, \tau_{32}, D_2, N, \ln p_T]$. Which model is most "observable-like"?
*Optional:* two EFN seeds (train a second EFN with `seed=1`, 3 minutes); the four models against each other; a second
ParT-small seed (12 minutes).

**T3.5** Mass sculpting and decorrelation. Cut the W-against-QCD score of the PFN at $\epsilon_S = 50\,\%$ and compare
the $m_\mathrm{SD}$ distribution of the QCD jets before and after the cut (`itj.metrics.sculpting_jsd`). Retrain the
PFN with planing weights that make $m_\mathrm{SD}$ flat in each class (write the weights yourself: 20 bins in
[40, 250] GeV, normalised per class to the number of jets of the class, capped at 10; pass them with
`JetDataset(weights=...)`), and plot $R_{50}$ against $1/\mathrm{JSD}$ for both PFNs (2 trainings of about 3 minutes).
Then evaluate both on `data/t3_w60.parquet`, `data/t3_w80.parquet`, `data/t3_w100.parquet` and `data/t3_w120.parquet`
(two-prong jets at other masses): does the decorrelated tagger keep tagging? *Optional:* DisCo (arXiv:2001.05310).

**T3.6** Counterfactual jets. Evaluate the T2 taggers on `data/t3_w80.parquet` and on the paired
`data/t3_w80_noue.parquet`, `data/t3_w80_centre.parquet` and `data/t3_w80_nocoh.parquet` (the same hard processes, one
switch changed: without the underlying event and with the cells at their centres every track from the jet is
identical, while without coherence the shower itself changes). Plot the change of the score jet by jet. Which change
matters most, and which of the T3.2 probe targets change with it? What does this say about what the taggers use (the
underlying event, the cell grid, the radiation pattern of a colour singlet)?

**T3.7** Your own proposal (about 30 minutes): probes on the outputs of the attention blocks, CKA across training
sizes, probes for `truth_part_from_b`.

**Deliverables:** `T3.1_embeddings.pdf`; `T3.2_probes.pdf` (a heat map of model against target) and the table of
probe scores; `T3.3_filters.pdf`; `T3.4_cka.pdf`; `T3.5_sculpting.pdf`, `T3.5_mass_scan.pdf` and the table;
`T3.6_counterfactual.pdf`; one paragraph "What our taggers know" for the discussion at the end of week 1.

**Hints**
- `extract` keeps the memory flat (it never computes attention maps). `fit_probe` standardises the inputs; report the
  probe scores with their spread over the cross-validation folds.
- `control_task(y, kind="shuffle")` keeps the marginal distribution of the target.
- A probe score means little on its own: the random-weight model and the control are part of the answer
  ([methods.md](docs/methods.md#probes)).
- The CKA invariances and the planing weights are explained in [methods.md](docs/methods.md); what the paired files
  share is in [physics.md](docs/physics.md#reproducibility).

## Data for T4 to T7

The data of the second week: three JetClass validation files, the pretrained models of the ParT paper, the split files
for weaver and three more toy files. `python examples/make_task_data.py --week 2` runs these commands; it needs
`data/t2_detector.parquet` from week 1. The core tasks download about 0.4 GB and the bonus T7.3 136 MB more, and about
1.4 GB are written; `--skip-fetch` leaves out the downloads. Start the downloads early (T0.4, `--only fetch`):
`jetgen fetch` resumes an interrupted download, checks the size and the MD5 of every file and keeps the files that
are already complete ([jetclass.md](docs/jetclass.md#download)).

```bash
# T4-T7: JetClass validation files of three classes (QCD = ZJetsToNuNu, W = WToQQ, top = TTBar), 100 000 jets each, 374 MB
jetgen fetch jetclass --split val --classes QCD,Wqq,Tbqq
# T4.3-T5.6: the pretrained ParT and ParticleNet of the JetClass paper (19 MB)
jetgen fetch models ParT_kin ParT_full ParticleNet_kin
# T4.6: the itj split (test 20 %, val 10 %, train 70 % of each file) written as files for weaver
jetgen split data/jetclass/val/ZJetsToNuNu_120.parquet data/jetclass/val/WToQQ_120.parquet data/jetclass/val/TTBar_120.parquet --out-dir data/jetclass/split
jetgen split data/t2_detector.parquet --out-dir data/t2_split
# T6: an unlabelled pretraining set (its labels are used only for the evaluation)
jetgen generate -n 90000 --seed 61 -o data/t6_pretrain.parquet
# T7.2 (bonus): a known planted shortcut: W jets with more photons from pi0 decays
jetgen generate --override wqq:pi0_fraction=0.50 -n 60000 --seed 72 -o data/t7_shortcut.parquet
# T7.2 (bonus): a hidden planted shortcut (do not run jetgen info --reveal on it before you have found it)
jetgen generate --mystery -n 60000 --seed 73 -o data/t7_mystery.parquet
# T7.3 (bonus): all ten JetClass classes, 10 000 jets each (ROOT, 136 MB; needs the analysis extra)
jetgen fetch jetclass --split example
```

The JetClass files are read like the toy files, with the same split per file (test = the first 20 000 jets of each
file, val the next 10 000, train the last 70 000):

```python
import itj.jetclass
from itj.io import load

test = itj.jetclass.load_jetclass("val", part="test")  # QCD, W and top, 20 000 jets each
train = load(itj.jetclass.files("val"), part="train", max_jets=20000)  # at most 20 000 jets per file
print(test)  # JetData(60 000 jets: ...; JetClass; ... particles)
```

## T4 Real jets: JetClass, ParT and weaver (about 4 hours)

**Goal:** the same code on JetClass, a fully trained ParT and which conclusions of week 1 survive.

Data: `data/jetclass/val/ZJetsToNuNu_120.parquet`, `data/jetclass/val/WToQQ_120.parquet` and
`data/jetclass/val/TTBar_120.parquet`, read with `load(..., part=...)`; their split copies in `data/jetclass/split/` for
weaver; the checkpoints in `models/pretrained/`.

**T4.1** Toy against JetClass. Take 10 000 jets per class from each: the first 10 000 jets of each JetClass test part,
and from `data/t2_detector.parquet` the first 10 000 jets per class after `itj.features.jetclass_selection`
($500 < p_T^\mathrm{jet} < 1000$ GeV, the cut of JetClass). Compare $m_\mathrm{SD}$, $\tau_{21}$, $\tau_{32}$, $D_2$,
$N$, the composition by particle type (counts and energy) and $p_T$ per class, and make a table of the medians next
to the validation numbers of [physics.md](docs/physics.md#calibration-and-validation). Explain the two largest
differences with [What the toy leaves out](docs/physics.md#what-the-toy-leaves-out).

**T4.2** Reproduce the stored branches. Recompute `jet_sdmass` and `jet_tau1` to `jet_tau3` from `part_*` with your T1
code and with `itj.observables`. Which axis choice (WTA-kT, plain exclusive kT, `onepass_kt` in
`itj.observables.subjet_axes`) reproduces the stored $\tau_N$, and why? Give the fraction of jets within 1 %. Check the
sign of `part_deta` from `part_px`, `part_py`, `part_pz` and `jet_eta`.

**T4.3** Pretrained taggers. Run `itj.models.load_pretrained("ParT_kin")`, `"ParT_full"` and `"ParticleNet_kin"` on
5 000 test jets per class (about 70 s per ParT and 2 minutes for ParticleNet). (a) The accuracy of the argmax over the
ten outputs on the three downloaded classes, then the three-class accuracy after `itj.jetclass.probs_to_jets3`: why do
they differ? (b) `one_vs_one` W against QCD and top against QCD (class indices 7, 8 and 0 of the ten outputs); add the
rows to the table, and the top $R_{50}$ with its bootstrap error (*optional:* on all 20 000 QCD test jets). What does
`ParT_full` gain from the particle types and the impact parameters, and on which pair?

**T4.4** Train on JetClass on a CPU. Train the PFN and the EFN (60 000 jets, 20 000 per class, from the train parts, 10
epochs, about 1.5 minutes each), PN-lite (60 000 jets, 5 epochs, about 6 minutes) and the BDT; *optional:* ParT-small
(30 000 jets, 5 epochs, 9 minutes). Compare with T4.3 (100 M jets and GPUs) and with the same models on the toy: is
JetClass harder, and for which pair? Save the models to `models/jetclass/`.

**T4.5** Transfer between the toy and JetClass. (a) Evaluate the T2 toy models on JetClass, and (c) the pretrained
`ParT_kin` on toy jets (after `jetclass_selection`): does a ParT trained on real jets recognise toy W and top jets?
Make the AUC matrix (trained on toy or JetClass, tested on toy or JetClass) per model. Diagnose the differences with
two tools: reweight the toy to the JetClass multiplicity per class, and drop the photons or the neutral particles at
evaluation (`itj.augment.remove_particles`). *Predict first:* will the IRC-safe EFN transfer best? About 6 minutes of
CPU. *Optional:* (b) the JetClass models of T4.4 on the toy; the score distributions per class; the T3.2 probes across
the two data sets; fine-tuning the toy EFN on 1 000 or 10 000 JetClass jets against training from scratch (a preview
of T6.5).

**T4.6** weaver. If `import weaver` failed in T0.4, do (a) with `itj` only and say so. (a) A short CPU run (about 35
s): train a ParT of the size of `partsmall` for 30 steps with `jets3_kin.yaml` (`jetgen configs` prints its path) on
the split toy files, then the identical command on the JetClass split files (one named group per class). Then run
`weaver --predict` with the pretrained `ParT_kin` (`JetClass_kin.yaml`, `--data-fraction 0.05`, `--gpus ''`) on the
split test files of JetClass. In test mode weaver reads the first `--data-fraction` of every file in order, so its
jets are `load(path, max_jets=1000)` of each split test file (e.g. `data/jetclass/split/test/WToQQ_120.parquet`):
compare its scores with those of `itj` and report the largest difference. Read the confusion matrix. Finally load the
checkpoint of the toy run into `itj` with `load_model(path, name="partsmall")`. The toy run, as one line (the others
are in [examples/weaver/README.md](examples/weaver/README.md)):

```bash
weaver --data-train 't2:data/t2_split/train/t2_detector.parquet' --data-val data/t2_split/val/t2_detector.parquet --data-test 't2:data/t2_split/test/t2_detector.parquet' --data-config src/itj/configs/jets3_kin.yaml --network-config examples/weaver/networks/example_ParticleTransformer.py -o num_layers 3 -o num_cls_layers 1 -o embed_dims "[64,256,64]" -o pair_embed_dims "[32,32,32]" -o num_heads 4 --model-prefix runs/weaver_partsmall/net --gpus '' --num-workers 0 --batch-size 256 --start-lr 1e-3 --num-epochs 1 --steps-per-epoch 30 --in-memory --fetch-step 1 --predict-output pred.parquet
```

(b) *GPU (optional):* a full ParT or ParticleNet training on the three-class subset
([examples/weaver/README.md](examples/weaver/README.md)).

**Deliverables:** `T4.1_toy_vs_jetclass.pdf` and the table of medians (T4.1); `T4.2_reproduce.pdf` (T4.2);
`T4.3_roc_pretrained.pdf` and the rows (T4.3); `T4.4_roc_jetclass.pdf` and the rows (T4.4); `T4.5_transfer.pdf` and
the AUC matrix (T4.5); the agreement between weaver and `itj` and the accuracy of the weaver run (T4.6).

**Hints**
- The order of the ten JetClass classes: QCD 0, ..., Wqq 7, Tbqq 8 (`itj.jetclass.CLASSES`).
- Guard $\tau = 0$, and remember that some JetClass particles lie at $\Delta R > 0.8$
  ([data_format.md](docs/data_format.md#jetclass-oddities)).
- weaver on a laptop: `--gpus ''` (the CPU), `--num-workers` at most the number of files in each named group (`0` is
  fine), `--in-memory` for small subsets, `--fetch-step 1` (weaver 0.5.4 fails on a partial read of a Parquet file)
  and always the split files: never an index selection in the YAML, because weaver applies selections after
  concatenating the files of a fetch. On Apple Silicon weaver picks MPS by itself: set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Check your numbers of T4.2 and T4.3 afterwards against [observables.md](docs/observables.md#jetclass-conventions)
  and [jetclass.md](docs/jetclass.md#models-and-numbers).

## T5 Interpretability: what does the tagger look at? (about 4 hours)

**Goal:** attributions, attention and IRC tests, each validated on toy jets, where the truth is known, before it is
used on the pretrained ParT.

Models: the toy PFN, EFN, PN-lite, ParT-small and ParT-small without $U$ of T2; on JetClass the pretrained `ParT_kin`.
Data: the test parts of `data/t2_detector.parquet` and of the JetClass files. Read this week: Jain and Wallace,
*Attention is not Explanation* (arXiv:1902.10186).

**T5.1** Integrated gradients (write your own; saliency is the one-step special case, take it from
`itj.interpret.saliency`). Implement IG with 32 steps and two baselines, the zero baseline and a "soft" one, and check
completeness: the attributions sum to $s(x) - s(x_0)$; if the gap is not small, plot it against the number of steps
(32, 128, 512). Compare with `itj.interpret.integrated_gradients`. Completeness holds for the zero baseline with
`wrt="features"`; for the soft baseline on ParT it needs
`wrt=("features", "vectors")`: why? On 500 toy W and top jets (ParT-small, PN-lite): the share of the attribution per
prong, on the underlying event and on the softest 50 % of the particles; five jet displays coloured by attribution.
1-2 minutes of CPU.

**T5.2** Do attributions tell the truth? (a) Deletion curves: remove particles in the order of their attribution, of
their $p_T$ and at random, and plot the score against the fraction removed. (b) Truth deletion: remove one W prong
(`truth_part_prong`) against the same number of random particles. Which attribution method ranks the particles best?

**T5.3** Attention heads. On 500 toy jets, `itj.interpret.attention_stats` gives per block and head the fraction of
attention that stays within the same prong, next to the expectation for uniform attention (the memory stays flat).
Are there "prong heads"? Validate the reco-level proxy `itj.interpret.subjet_membership(data, n)` against
`truth_part_prong` on the toy, then repeat with the proxy on `ParT_kin` and 500 JetClass W and top jets. Run IG (zero
baseline) on 200 JetClass W jets with `ParT_kin`: what can you say without truth? What can and what cannot be
concluded from attention (arXiv:1902.10186)? *Optional:* the rank correlation of class attention and IG. About 3
minutes of CPU.

**T5.4** The pair bias $U$. (a) Regress $U_{ij}$ of each head on $[\ln k_T, \ln z, \ln\Delta, \ln m^2]$
(`itj.interpret.pair_features`) for the toy ParT-small and for `ParT_kin`: which Lund coordinate does each head use
(compare with T1.3)? Replace one pair channel at a time by its mean (`itj.interpret.ablate_pair_channels`) and report
the change of the AUC. *Predict first:* what happens to the top accuracy of `ParT_kin` when $U$ is set to 0
(`itj.interpret.zero_pair_bias`)? Why does that say little about how useful $U$ is? (b) The honest ablation: compare
`partsmall` and `partsmall_nopair` of T2.5 (AUC and $R_{50}$ with errors). What does $U$ buy, and does the
mean-imputation of (a) predict it? *Optional:* retrain with `pair_input_dim=1` (about 6 minutes).

**T5.5** Are the taggers IRC safe? Run `itj.interpret.irc_scan` soft ($n = 1, 5$; $\epsilon = 10^{-1}, \ldots, 10^{-6}$)
and collinear ($\lambda = 0.5, 0.1$; $\delta = 10^{-1}, \ldots, 10^{-5}$) on the EFN, PFN, PN-lite and ParT-small (toy)
and on `ParT_kin` (JetClass), and plot the mean $|\Delta s|$ against $\epsilon$ and $\delta$. *Predict first:* which
models are safe by construction? About 2 minutes of CPU for `ParT_kin`, seconds for the toy models. Mind the pitfalls
([observables.md](docs/observables.md#numerical-irc-tests)): the truncation at 128 particles (`irc_scan` refuses jets
that would exceed it); ParT clamps $k_T$, $\Delta$ and $m^2$ at $10^{-8}$ inside its pair features ($\ln = -18.4$)
before a BatchNorm; soft particles below the detector thresholds are outside the training data; the sign of
`part_deta` flips when a soft particle moves $\eta_\mathrm{jet}$ across 0 (`irc_scan` keeps the sign of the unperturbed
jet, `keep_sign=True`).

**T5.6** Probes on the real ParT. Run the probes of T3.2 ($m_\mathrm{SD}$, $\tau_{21}$, $\tau_{32}$, $D_2$, $N$ and
$\ln k_t$ of the hardest primary declustering from `lund_primary`) on the class token of `ParT_kin` (10 000 JetClass
test jets) and on the pooled output of each block; the same for the toy ParT-small. Is it the same picture? Compare
with arXiv:2605.09881 (the ECF basis against the N-subjettiness basis). About a minute of inference.

**T5.7** Your own proposal (about 30 minutes): SmoothGrad, attention rollout (arXiv:2005.00928,
`itj.interpret.attention_rollout`), layer-wise relevance propagation.

**Deliverables:** `T5.1_attribution_gallery.pdf`, `T5.1_prong_fraction.pdf` (T5.1); `T5.2_deletion.pdf` (T5.2);
`T5.3_heads.pdf` (block against head: toy truth, toy proxy, JetClass proxy) and `T5.3_attention_example.pdf` (T5.3);
`T5.4_pair_bias.pdf` and the ablation table (T5.4); `T5.5_irc_soft.pdf`, `T5.5_irc_collinear.pdf` (T5.5);
`T5.6_part_probes.pdf` (T5.6).

**Hints**
- Report both IG baselines. The rows of an attention matrix sum to 1: compare with $1/N$.
- Never call `.backward()` on the raw vectors of ParT yourself: use `itj.interpret.input_gradients` (the pair
  features of a particle with itself give NaN gradients otherwise).
- `itj.augment.max_particles_guard` tells you whether a perturbation would push a jet beyond 128 particles.
- Never `trace` with attention on more than a few hundred jets at once (memory); `attention_stats` does the reduction
  batch by batch.
- The reasons why zeroing $U$ misleads are in [methods.md](docs/methods.md#pair-feature-ablation): read them after
  your prediction.

## T6 Learning without labels (about 4 hours)

**Goal:** a jet representation from contrastive self-supervision with physics augmentations (JetCLR,
arXiv:2108.04253): what it knows, how few labels it needs and which invariances it inherits.

Encoder: `build_model("pfn")` with `itj.ssl.ProjectionHead(128, 128, 64)` (the fast default); `partsmall_nopair` is
optional. Data: `data/t6_pretrain.parquet` (do not look at its labels before the evaluation), and
`data/t2_detector.parquet` for the evaluation.

**T6.1** Augmentations (write your own rotation). Compare your rotation about the jet axis with `itj.augment.rotate`.
For `rotate`, `smear_soft` and `split_collinear`: the change of $m_\mathrm{SD}$, $\tau_{21}$, $N$ and $p_T^D$ on 2000
jets (`substructure=True`). Which are symmetries of the physics, which only symmetries of IRC-safe observables, and
which are approximate? What should a representation trained with them forget? *Optional:* `translate`, `drop_soft`,
$D_2$.

**T6.2** The contrastive loss (write your own NT-Xent). Compare it with `itj.ssl.nt_xent` on random tensors. Work out
its limits and check them: two identical views; all embeddings equal (what value does the loss take for a batch of
$B$ pairs?); the effect of the temperature $\tau_T$.

**T6.3** Pretraining (write the loop). With `itj.ssl.JetCLRAugment`, two views per jet, batch 256 and
$\tau_T = 0.1$, pretrain on the 90 000 jets for 4 epochs (about 3.5 minutes). After each epoch monitor the loss, the
alignment, the uniformity and the linear evaluation with 1 000 labels. The NT-Xent loss falls close to 0 within two
epochs: that is expected with these augmentations; judge the run by the alignment, the uniformity and the linear
evaluation. Save the encoder as `models/toy/jetclr_pfn.pt`. *Optional:* 10 epochs (does the linear evaluation keep
improving?); `partsmall_nopair` on 30 000 jets for 4 epochs.

**T6.4** Linear evaluation and what it learned. Freeze the encoder and train a logistic regression on its embedding
(`itj.ssl.linear_eval`) with 1 000 labels and with all labels (train part of `data/t2_detector.parquet`, tested on its
test part), always next to the random encoder (`reset_parameters`), the supervised PFN of T2.4 and the observables
$X_\mathrm{phys}$ of T3.4. Where does the pretraining help? Then run the truth probes of T3.2 on the SSL embedding: does
it know quark from gluon jets, the number of emissions, the opening angle of the W?

**T6.5** Few labels. The AUC (W against QCD and top against QCD) against the number of labelled jets (100, 300, 1 000,
3 000, 10 000; 3 seeds) for SSL + linear evaluation, SSL + fine-tuning and supervised training from scratch (with the
curve of T2.6). Train every point with the rule of T2.6 (`min_steps=2000`, `patience=5`, at least 5000 validation
jets). Where does pretraining help? About 15 minutes of CPU: run it in the background.

**T6.6** Inherited invariances. Run `irc_scan(..., output="embedding")` on the SSL encoder, and the score scan on its
linear head. Is it more collinear safe than the supervised PFN? Retrain it without `split_collinear` (3.5 minutes) and
compare.

**T6.7** *Optional (GPU welcome):* pretrain on unlabelled JetClass jets and evaluate linearly on JetClass; or transfer
the toy-pretrained encoder to JetClass.

**Deliverables:** `T6.1_augmentations.pdf` (augmentation against observable) (T6.1); the checks of your NT-Xent (T6.2);
`T6.3_ssl_training.pdf` (T6.3); the rows and `T6.4_probes.pdf` (T6.4); `T6.5_few_labels.pdf` (T6.5);
`T6.6_irc_ssl.pdf` (T6.6).

**Hints**
- A batch of 256 pairs gives 510 negatives per positive. Normalise the projections before the loss.
- Evaluate the encoder output (the embedding before the projection head), not the projection.
- `JetCLRAugment` drops the truth columns first, for speed.
- `fit(..., loss=callable)` takes your own loss, or write your own loop.
- If the linear evaluation of your encoder is far below the reference in
  [methods.md](docs/methods.md#linear-evaluation), check the augmentations first.

## T7 Bonus: towards a research project (2 to 4 hours)

**Goal:** the first steps of a research project: a pretraining objective beyond contrastive learning, a hunt for
shortcuts with the tools of both weeks, a paper from the frontier and a one-page proposal.

**T7.1** Masked particle modelling (arXiv:2401.13537, arXiv:2409.12589). Mask 30 % of the particles with
`itj.ssl.mask_particles(batch, frac=0.3)` and train `partsmall_nopair` to predict the masked
($\ln p_T$, $\Delta\eta$, $\Delta\phi$) with a regression loss, in $p_T$ order (about 15 minutes). Evaluate the
encoder linearly as in T6.4.

**T7.2** Shortcut hunting. (a) A known shortcut: train a PFN with the particle types,
`build_model("pfn", feature_set="kinpid")` (3 minutes), on `data/t7_shortcut.parquet` (W jets with a $\pi^0$ share of
0.50 instead of the default) and evaluate it on clean W jets (`data/t2_detector.parquet`). Can
saliency, IG summed by particle type or a probe for the photon fraction find the shortcut without being told? (b) A
hidden shortcut: `data/t7_mystery.parquet` holds one of the five shortcuts of
[physics.md](docs/physics.md#planted-shortcuts). Find it with the tools of T3 and T5 and with counterfactual edits,
remove it, and only then run `jetgen info data/t7_mystery.parquet --reveal`.

**T7.3** A paper from the frontier: one of arXiv:2412.03673, arXiv:2512.00210, arXiv:2605.09881 and arXiv:2401.13537.
Reproduce one figure qualitatively on the toy, or probe every block of `ParT_full` on all ten classes (the example
file, `data/jetclass/example/JetClass_example_100k.root`, 20 000 jets, about 1.5 minutes of inference) and compare
with arXiv:2412.03673.

**T7.4** A research proposal (one page): the question, why it matters, the data, the method, the first experiment and
its figure, what would change your mind, the compute and the risks; motivated by one of your results of T1 to T6.

**Deliverables:** one row and one figure (T7.1); the evidence chain in five sentences and the figure that gave the
shortcut away (T7.2); one figure (T7.3); the proposal (T7.4).

**Hints**
- T7.2: compare the counts of each particle type between the classes before you train anything.
- The provenance of the mystery file hides the override (`jetgen info --reveal` prints it); the mapping from the seed
  to the shortcut is in the source code, so do not read `itj.toy.mystery` before you are done.
- T7.3: the example file is ROOT; `load` reads it with the `analysis` extra.
