# Tasks: A Spray of Hadrons

These are four weeks of questions on jets and on the taggers that classify them, for students who start with no particle
physics. A jet is the narrow spray of particles that one fast quark or gluon, or the decay products of one heavy
particle, leave in a detector ([what a jet is](docs/primer.md#what-a-jet-is)). A tagger is a program that reads the
particles of a jet and says what started it. A jet can start from the decay of a W boson (a heavy, unstable particle;
[see the primer](docs/primer.md#the-particles)) or a top quark, or from an ordinary quark or gluon. The last kind is a
QCD jet, named after quantum chromodynamics (QCD), the theory of quarks and gluons. The questions prepare a research
project on what jet taggers learn and how to check it.

The toy, **A Spray of Hadrons**, is a small simulation written for this course. Hadrons are the particles built from
quarks, such as pions and protons, that make up most of a jet. The toy writes jets with the branches (the columns of the
files) of JetClass, a large public set of simulated jets ([docs/jetclass.md](docs/jetclass.md)). It also writes the
truth that only a simulation has. This truth holds the prongs (the hard clusters of energy in a jet, one per decay quark
of a W or a top). It also holds every shower emission (each split of a quark or gluon as the jet forms) and the origin
of every particle. With this truth, every claim about a tagger can be checked on the toy before it is made about a real
one.

- **Four weeks of 2-3 working days.** Week 1 is *What is a jet?*. It covers the set-up, jets from zero, the mass,
  counting prongs and the ladder of the toy, which switches the physics on one step at a time. Week 2 is *Taggers, from
  cuts to transformers*. Week 3 is *What do taggers learn?* and week 4 *Real jets, and where does a tagger look?*. A
  bonus prepares the research project.
- **Assumed:** first-year physics (mechanics, energy and momentum, $E = mc^2$; the box in [primer section
  2](docs/primer.md#four-vectors-and-invariant-mass) gives the one Lorentz transformation the course uses), Python with
  NumPy and matplotlib. No particle physics and no PyTorch (the Python library for neural networks that the course
  uses): the primer (task **W1.1**) and the first network
  (task **W2.3**) start from zero.
- **Start here:** the questions begin in [tasks/week1.md](tasks/week1.md) with the set-up (**W1.0**); the table below
  links each week to its file. [docs/primer.md](docs/primer.md), *Jets from zero*, is read in pieces during week 1. Each
  piece comes just before the question that uses it. [docs/ml_primer.md](docs/ml_primer.md), *Machine learning from
  zero*, does the same for the machine learning of weeks 2-4.
- **Three tiers.** Every task has *warm-up* questions (short steps that build confidence), *core* questions (the course)
  and *stretch* questions (the hardest material, the extra studies and the proposals). A week done to the end of its
  core questions is complete.
- **Question IDs** are `W<week>.<task>.<question>`, for example **W2.3.3** (week 2, task 3, question 3); the bonus
  questions are **B.1** to **B.5**. Every question shows its tier and a realistic time next to its ID. We go through the
  answers by ID together at the end of each week. The IDs of release 0.1 (T0.1 to T7.4) are mapped to these in
  [docs/id_mapping_0.2.md](docs/id_mapping_0.2.md).
- **Hours.** The core path takes about 13-14 hours a week for a student with the background above. It covers the
  warm-up and core questions, the weekly summary, the hand-in and the discussion. The times include reading, thinking
  and writing up. They are estimates: a student who has met jets or PyTorch before needs less, and every week lists
  what to drop when you are behind.

| week | title | core path | stretch on offer |
|---|---|---|---|
| 1 | [What is a jet?](tasks/week1.md) | about 13.75 h | about 27.5 h |
| 2 | [Taggers, from cuts to transformers](tasks/week2.md) | about 12.75 h | about 11.5 h |
| 3 | [What do taggers learn?](tasks/week3.md) | about 13.5 h | about 26.75 h |
| 4 | [Real jets, and where does a tagger look?](tasks/week4.md) | about 12.75 h | about 32.75 h |
| bonus | [Towards a research project](tasks/bonus.md) | | about 24.5 h |

## How to work

**Tiers and the stopping rule.** Do the warm-up and core questions of every task, in order. They are the course: a week
done to the end of its core questions is complete. Then do stretch questions in the order listed, as time allows. The
first stretch question of a task is the most useful one, and each week names its *First stretch picks*. A question
marked *Optional (GPU)* needs a GPU (a graphics processing unit, which trains networks much faster than a CPU).

**Reading is part of the time.** Each question names what to read first (a primer section, an ml_primer section, a docs
section or a notebook cell). Its time includes that reading and the check-yourself questions at the end of each primer
section. About half of a warm-up is reading. The time also covers writing the answer and making its figure.

**Write it, then check it.** A question marked "(write your own)" asks you to implement a tool first. You then compare
it with the reference in `itj` (the Python package of the course) in one `allclose` line, a NumPy check that two arrays
agree up to rounding. Later questions use the reference or the course models, so a bug
in your version cannot spoil them.

**Predict first.** Where a question says *Predict first*, write a one-sentence prediction into your answers before you
run anything, and come back to it afterwards.

**Start of week.** Each week opens with a short box of commands. They make the data of the week and start the long runs
in the background, so that nothing blocks you later. Run them in the first hour. Every run of more than about 5 minutes
says "in the background" and starts before you read the question.

**Planning your week.** The core path of each week takes 12.75-13.75 hours: two working days of about 6.5-7 hours, or
three shorter days of 4-5 hours. In a week of only two days of about 6 hours, use the fallbacks marked *two-day* from
the start. Fallbacks are smaller versions of questions, listed under *If you are behind*; the *two-day* ones of each
week save 0.5-0.75 hours. Agree any others at the check-in. A third day is for stretch questions and catch-up. Each
week gives its day plan, with the split for three shorter days. A **check-in** at the end of your first day (15
minutes, or a short message to your instructor) makes sure that nothing blocks the rest of the week. Each week names
what should work by then.

**If you are behind.** Each week lists fallbacks that are agreed in advance. Each one gives a smaller version of a
question, the time it saves and what is lost. Write "fallback" next to the ID in your answers when you use one. Every
fallback keeps what the later weeks need.

## What to hand in

1. **Short answers** in `answers/week<k>.md`: a few sentences per question, headed by its ID, with the numbers you
   measured (and their errors), and the weekly summary.
2. **Figures** (PDF or PNG) named by the question ID, e.g. `W2.5.2_roc_all.pdf`.
3. **Your code** (scripts or notebooks) and the seeds of your data sets and trainings, so that every figure can be made
   again.
4. **`results_table.csv`** from week 2 on: one row per (model, training data, test data), written with
   `itj.metrics.append_result` (the columns are `itj.metrics.RESULT_COLUMNS`), kept until week 4 and committed on your
   branch.

Trained models go to `models/course/` (the course models of week 2), `models/toy/` and `models/jetclass/` (all ignored
by git). **Proposals:** the last task of every week ends with a stretch question *Your own question* (about 2 hours,
with a figure). If you continue to the research project, we recommend that you do one over the course. Make your
own plots too: whatever you find telling, surprising or useful.

**Figure checklist.** A figure should be readable on its own. It needs axis labels with units and a legend that does not
hide the data. It also needs a title or caption that says what is shown. The caption names the preset (the rung of the
toy's ladder), the process (W, top or QCD jets), the number of jets and the $p_T$ range. For a tagger, it also names the
model and the training size. Show statistical uncertainties where they matter and use a log axis when values span
several orders of magnitude. In addition, for this course:
- For taggers: ROC curves. A tagger gives each jet a score, the number it computes for that jet, and a jet passes when
  its score is above a cut. The signal jets are the class the tagger looks for, and the background jets are the rest.
  Here $\epsilon_S$ and $\epsilon_B$ are the fractions of signal and background jets that pass, and the rejection is
  $1/\epsilon_B$. A ROC (receiver operating characteristic) curve shows how many background jets a tagger rejects for
  each fraction of signal jets it keeps. Draw it as $1/\epsilon_B$ on a log axis against $\epsilon_S$, with the working
  point and the size of the test part. The working point is the cut on the score that you quote ([ml_primer: a tagger is
  a function](docs/ml_primer.md#a-tagger-is-a-function)). A rejection above $N_B/10$ ($N_B$ background jets in the test
  part) has a statistical error of more than 30 %. Say so and show the bootstrap error, the spread of the result over
  copies of the test part resampled with replacement.
- Jet displays and jet images (pictures of the particles of a jet around its axis) with equal axes in
  $(\Delta\eta, \Delta\phi)$ (`ax.set_aspect("equal")`).
- Lund planes as densities per jet and per unit area, with the number of jets and the cutoff line. A Lund plane is a map
  of the shower emissions of a jet by angle and transverse momentum (see [the parton
  shower](docs/primer.md#the-parton-shower)).
- Attribution maps with a colour bar that says whether the values are signed or absolute. An attribution map shows how
  much each particle or region moves the score of a tagger (see [gradients of the
  input](docs/ml_primer.md#gradients-of-the-input)).
- Every learning result with its training size, number of epochs (passes through the training set) and random seed(s)
  (the numbers that fix the random choices).

**Where to work.** Keep your scripts on your own branch of the repository (`git switch -c <your-name>`) or in a folder
outside it. Data and models are not committed: `data/`, `models/`, `runs/`, `*.parquet` and `*.pt` are ignored by git.
Start `jupyter lab` (or your editor) from the repository root and keep your notebooks and scripts there, so that the
paths `data/...` and `models/...` of the week files work. A notebook opened inside `notebooks/` runs there and needs
`Path("..") / "data"`. `import itj` alone does not import its parts: write `import itj.toy`, `import itj.observables`,
`from itj.io import load` and so on, as each question shows.

**Where to run.** A laptop is enough for every core question (Linux, macOS or Windows; CPU only). The times in the week
files are for an otherwise idle 2016 six-core CPU; laptops from 2022 on are 1.5-3 times faster. Plan on 16 GB of memory.
The course models of week 2 peak at about 9 GB (ParT-small, a small Particle Transformer). A run of weaver in **W4.2.7**
peaks at about 6-7 GB (weaver is the training framework of ParT and the JetClass paper; [weaver in one
page](docs/jetclass.md#weaver-in-one-page)). With 8 GB, run the week-2 start-of-week box overnight and close other
sessions that hold `t2_detector`, the data file that this box makes. Long trainings take `--threads`: leave two cores
free for your own work. Every `jetgen` command of the week files is a single line and works in any shell.
`python examples/make_task_data.py --week 1` (to `--week 4`) makes the data sets of each week. Parts marked *Optional
(GPU)* run on a GPU, on Colab or on a cluster ([examples/cluster](examples/cluster/README.md)).

**Docs and notebooks**, with the week each is first needed:

| document | what it holds | first needed |
|---|---|---|
| [docs/primer.md](docs/primer.md) | *Jets from zero*: units, four-vectors, coordinates, particles, jets, showers, hadrons, the detector, the toy's files, histograms, the observables in words | week 1 |
| [notebooks/00_primer.ipynb](notebooks/00_primer.ipynb) | worked examples of every tool of **W1.1**, on other jets than the questions | week 1 |
| [notebooks/00_quickstart.ipynb](notebooks/00_quickstart.ipynb) | a guided first look at the toy (optional reading; it previews weeks 1-2) | week 1 |
| [docs/physics.md](docs/physics.md) | the physics of the toy and the derivations the questions use | week 1 |
| [docs/observables.md](docs/observables.md) | the observables and IRC (infrared and collinear) safety | week 1 |
| [docs/data_format.md](docs/data_format.md) | the files and their columns | week 1 |
| [docs/install.md](docs/install.md) | Windows, the CPU build of PyTorch, Apple Silicon, notebooks and common problems | week 1 |
| [docs/ml_primer.md](docs/ml_primer.md) | *Machine learning from zero*: taggers, metrics, trees, networks, PyTorch, probes, contrastive learning, attributions | week 2 |
| [notebooks/02_first_network.ipynb](notebooks/02_first_network.ipynb) | your first neural network, in two parts (**W2.3.1**, **W2.3.2**) | week 2 |
| [docs/models.md](docs/models.md) | the network inputs and the models | week 2 |
| [docs/methods.md](docs/methods.md) | the methods for representations and interpretability | week 3 |
| [docs/jetclass.md](docs/jetclass.md) | JetClass, ParT and weaver in one page | week 4 |
| [notebooks/01_jetclass.ipynb](notebooks/01_jetclass.ipynb) | a guided first look at JetClass (**W4.1.1**) | week 4 |
| [docs/references.md](docs/references.md#reading-list) | the reading list, with the papers to read first in each week | any week |
| [docs/glossary.md](docs/glossary.md) | the terms, each with the question that first needs it | any week |

## Notation

| symbol | meaning | first met |
|--------|---------|-----------|
| $E$, $\vec p$, $m$ | energy, momentum and mass, all in GeV ($c = 1$) | [primer 1](docs/primer.md#units-and-energies) |
| $p = (p_x, p_y, p_z, E)$ | four-vector, in the order of the files (`part_px`, `part_py`, `part_pz`, `part_energy`); $m^2 = E^2 - \lvert\vec p\rvert^2$ | [primer 2](docs/primer.md#four-vectors-and-invariant-mass), **W1.1.1** |
| $p_T$, $y$, $\eta$, $\phi$ | transverse momentum, rapidity, pseudorapidity, azimuth | [primer 3](docs/primer.md#collider-coordinates), **W1.1.2** |
| $\Delta R$ | distance in $(y, \phi)$, $\Delta R^2 = \Delta y^2 + \Delta\phi^2$ | [primer 3](docs/primer.md#collider-coordinates), **W1.1.2** |
| $R$ | jet radius, 0.8 | [primer 5](docs/primer.md#what-a-jet-is) |
| $\Delta\eta_i$, $\Delta\phi_i$ | coordinates of particle $i$ relative to the jet axis (`part_deta`, `part_dphi`; $\Delta\eta$ with the JetClass sign, [physics.md](docs/physics.md#units-and-coordinates)) | [primer 3](docs/primer.md#collider-coordinates), **W1.1.7** |
| $z_i = p_{T,i}/p_{T,J}$ | momentum fraction of particle $i$ in the jet | **W2.4.4** |
| $z$, $\theta$, $k_t$ | momentum fraction, opening angle and transverse momentum of a shower branching (`truth_emis_z`, `truth_emis_theta`, `truth_emis_kt`) | [primer 7](docs/primer.md#the-parton-shower), **W1.1.8** |
| $\Delta$, $k_t$, $z$ (Lund) | opening angle (the angle $\theta$ as found by undoing the clustering), transverse momentum and momentum fraction of a primary declustering (`truth_lund_*`, `itj.observables.lund_primary`) | [primer 7](docs/primer.md#the-parton-shower), **W1.2.3** |
| $C_F = 4/3$, $C_A = 3$ | colour factors of quarks and gluons; $C_R$ the one of the emitter | [primer 7](docs/primer.md#the-parton-shower) |
| $\alpha_s$ | strong coupling (`alphas_value`) | [primer 4](docs/primer.md#the-particles) |
| $\Delta(t_0, t)$ | Sudakov factor: probability of no emission between $t_0$ and $t$ | **W1.2.10** |
| $m_J$, $m_\mathrm{SD}$ | jet mass, soft-drop mass (`jet_sdmass`) | [primer 12](docs/primer.md#a-first-look-at-the-observables), **W1.1.6** |
| $\rho = m^2/(p_T^2R^2)$ | normalised jet mass | **W1.3.5** |
| $\tau_N$, $\tau_{21}$, $\tau_{32}$ | N-subjettiness (`jet_tau1` ... `jet_tau4`) and its ratios $\tau_2/\tau_1$, $\tau_3/\tau_2$ | [primer 12](docs/primer.md#a-first-look-at-the-observables), **W1.1.6** |
| $e_2$, $e_3$, $D_2$, $C_2$ | energy correlation functions and their ratios $D_2 = e_3/e_2^3$, $C_2 = e_3/e_2^2$ | [primer 12](docs/primer.md#a-first-look-at-the-observables), **W1.4.1** |
| $N$ | multiplicity, the number of particles (`jet_nparticles`) | **W1.1.4** |
| $p_T^D$ | $\sqrt{\sum_i z_i^2}$, how unevenly the $p_T$ of a jet is shared between its particles | **W3.3.2** |
| $\epsilon_S$, $\epsilon_B$ | signal and background efficiency of a cut on a score | [ml_primer 1](docs/ml_primer.md#a-tagger-is-a-function), **W2.1.1** |
| $R_{50}$ | background rejection $1/\epsilon_B$ at $\epsilon_S = 0.5$ | [ml_primer 1](docs/ml_primer.md#a-tagger-is-a-function), **W2.1.2** |
| $s(x)$ | score of a tagger for the jet $x$ | [ml_primer 1](docs/ml_primer.md#a-tagger-is-a-function) |
| $h$ | embedding of a jet (`model.embed`, the input of the last linear layer) | [ml_primer 7](docs/ml_primer.md#embeddings-and-maps), **W3.1.1** |
| $A$, $U$ | attention weights of a transformer head; the pair bias of ParT | [ml_primer 6](docs/ml_primer.md#networks-on-particles), **W2.5.1** |
| IG | integrated gradients | [ml_primer 11](docs/ml_primer.md#gradients-of-the-input), **W4.4.2** |
| $\epsilon$, $\delta$, $\lambda$ | IRC tests: soft $p_T$ fraction, collinear separation, momentum share of a split | **W3.3.1**, **W3.3.2** |
| $\tau_T$ | temperature of the NT-Xent loss (not an N-subjettiness) | [ml_primer 9](docs/ml_primer.md#learning-without-labels), **W3.4.1** |

## The weeks

The questions are in one file per week. Open the week you are working on; the rules above apply to all of them.

| week | file | title | core path |
|---|---|---|---|
| 1 | [tasks/week1.md](tasks/week1.md) | What is a jet? | about 13.75 h |
| 2 | [tasks/week2.md](tasks/week2.md) | Taggers, from cuts to transformers | about 12.75 h |
| 3 | [tasks/week3.md](tasks/week3.md) | What do taggers learn? | about 13.5 h |
| 4 | [tasks/week4.md](tasks/week4.md) | Real jets, and where does a tagger look? | about 12.75 h |
| bonus | [tasks/bonus.md](tasks/bonus.md) | Towards a research project | none (about 24.5 h on offer) |
