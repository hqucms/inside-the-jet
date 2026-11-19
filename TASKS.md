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
- **Start here:** [docs/primer.md](docs/primer.md), *Jets from zero*, is read in pieces during week 1. Each piece comes
  just before the question that uses it. [docs/ml_primer.md](docs/ml_primer.md), *Machine learning from zero*, does the
  same for the machine learning of weeks 2-4.
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
| 1 | What is a jet? | about 13.75 h | about 27.5 h |
| 2 | Taggers, from cuts to transformers | about 12.75 h | about 11.5 h |
| 3 | What do taggers learn? | about 13.5 h | about 26.75 h |
| 4 | Real jets, and where does a tagger look? | about 12.75 h | about 32.75 h |
| bonus | Towards a research project | | about 24.5 h |

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
paths `data/...` and `models/...` of this sheet work. A notebook opened inside `notebooks/` runs there and needs
`Path("..") / "data"`. `import itj` alone does not import its parts: write `import itj.toy`, `import itj.observables`,
`from itj.io import load` and so on, as each question shows.

**Where to run.** A laptop is enough for every core question (Linux, macOS or Windows; CPU only). The times below are
for an otherwise idle 2016 six-core CPU; laptops from 2022 on are 1.5-3 times faster. Plan on 16 GB of memory. The
course models of week 2 peak at about 9 GB (ParT-small, a small Particle Transformer). A run of weaver in **W4.2.7**
peaks at about 6-7 GB (weaver is the training framework of ParT and the JetClass paper; [weaver in one
page](docs/jetclass.md#weaver-in-one-page)). With 8 GB, run the week-2 start-of-week box overnight and close other
sessions that hold `t2_detector`, the data file that this box makes. Long trainings take `--threads`: leave two cores
free for your own work. Every `jetgen` command below is a single line and works in any shell.
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

---

## Week 1: What is a jet? (core path about 13.75 h, stretch about 27.5 h)

**Why this week.** A tagger is a function of a jet. Before we ask what a tagger looks at, we need to know what is in a
jet and where each of its particles comes from. We also need to know what the classic observables measure. An
observable is a number computed from the particles of a jet, such as its mass. The toy switches the physics on one step
at a time and keeps the truth.

**Thread.** You start with nothing assumed but first-year physics. A W jet holds the decay products of a W boson, a
heavy unstable particle ([see the primer](docs/primer.md#the-particles)). By the end of the week you can say what a W
jet is made of at each level of the ladder. You can also say which observables tell it apart from a QCD jet. They are
the mass, the soft-drop mass (the jet mass after soft, wide-angle particles are removed), $\tau_{21}$ and $D_2$. The
last two count the hard clusters of energy in a jet. The first tagger of week 2 is built from these observables.

**Concepts this week**, in order of use. The set-up: a virtual environment and tests, and `JetData`, the object that
holds a file in Python. Units and kinematics: GeV and $c = 1$ ([primer 1](docs/primer.md#units-and-energies));
four-vectors, invariant mass and a boost along the beam ([primer 2](docs/primer.md#four-vectors-and-invariant-mass));
$p_T$, $\phi$, $y$, $\eta$ and $\Delta R$ ([primer 3](docs/primer.md#collider-coordinates)). Particles and jets: quarks,
gluons, W, top, hadrons, photons, neutrinos ([primer 4](docs/primer.md#the-particles)); jets, anti-kT and $R$ ([primer
5](docs/primer.md#what-a-jet-is)); boosted decays and prongs ([primer 6](docs/primer.md#boosted-w-and-top-jets)). From
the first quark to the detector: the parton shower ([primer 7](docs/primer.md#the-parton-shower)), hadronisation and the
underlying event ([primer 8](docs/primer.md#hadronisation-and-the-underlying-event)), the detector ([primer
9](docs/primer.md#what-a-detector-sees)). The toy: the presets, paired samples, padding and masks ([primer
10](docs/primer.md#the-toy-and-its-files)). Analysis: histograms, densities and Poisson errors ([primer
11](docs/primer.md#histograms-and-errors)); jet mass, soft drop, N-subjettiness, $D_2$ and multiplicity ([primer
12](docs/primer.md#a-first-look-at-the-observables)). The stretch questions add the Lund plane as a picture ([primer
7](docs/primer.md#the-parton-shower), skipped on a first reading). They also add relative coordinates and the JetClass
sign, colour factors, the Lund plane of one jet and its density, the Sudakov factor, leading logarithms and power
counting.

**This week at a glance** (times in minutes, or hours where marked):

| task | warm-up | core | stretch |
|---|---|---|---|
| **W1.0** Set-up | **W1.0.1** 60 | **W1.0.2** 60 | **W1.0.3** 75, **W1.0.4** 30 |
| **W1.1** Jets from zero | **W1.1.1** 60, **W1.1.2** 45 | **W1.1.3** 105, **W1.1.4** 90, **W1.1.5** 45, **W1.1.6** 75 | **W1.1.7** 45, **W1.1.8** 90, **W1.1.9** 60, **W1.1.10** 90 |
| **W1.2** The ladder on many jets | **W1.2.1** 30 | **W1.2.2** 45 | **W1.2.3** 30, **W1.2.4** 60, **W1.2.5** 75, **W1.2.6** 60, **W1.2.7** 75, **W1.2.8** 30, **W1.2.9** 45, **W1.2.10** 105, **W1.2.11** 60 |
| **W1.3** The jet mass | **W1.3.1** 30 | **W1.3.2** 60 | **W1.3.3** 45, **W1.3.4** 75, **W1.3.5** 2 h, **W1.3.6** 75 |
| **W1.4** Counting prongs | **W1.4.1** 30 | **W1.4.2** 30 | **W1.4.3** 30, **W1.4.4** 90, **W1.4.5** 90, **W1.4.6** 75, **W1.4.7** 2 h |
| summary, hand-in and discussion | | 1 h | |

**Start of week** (inside **W1.0.2**). Right after the installation, run `python examples/make_task_data.py --week 1` in
a second terminal. It makes the 10 files of [Data for week 1](#data-for-week-1) while you read the primer. They hold
about 290 000 jets and 0.48 GB, and take about 1.5-2 minutes with 8 workers (parallel processes) or about 5 minutes on
one core.

**Reading in this week's times.** The primer is read in pieces, each just before the question that uses it. The time of
that question includes the reading, the check-yourself questions and the matching cell of
[notebooks/00_primer.ipynb](notebooks/00_primer.ipynb) (about 3.5 hours of **W1.1** in all). This is why the warm-ups
of **W1.1** take 45-60 minutes: about half of each is reading. Week 1 is the longest week because everything in it is
new.

**Day plan.** Day 1 (about 7 h): **W1.0.1**, **W1.0.2** (start the week-1 data), **W1.1.1**-**W1.1.4** (primer
sections 1-8 and the *JetData* paragraph of section 10). **W1.1.3**, the hardest question of the week, comes in the
middle of the day, not at its end. Day 2 (about 6.75 h): **W1.1.5**-**W1.1.6** (primer sections 9-12),
then **W1.2.1**-**W1.2.2**, **W1.3.1**-**W1.3.2**, **W1.4.1**-**W1.4.2**, the weekly summary and the discussion. Day 3
(optional): the first stretch picks, catch-up. Three shorter days instead: the set-up (**W1.0**) with the
warm-ups **W1.1.1**-**W1.1.2** (3.75 h) / **W1.1.3**-**W1.1.6** (5.25 h) / **W1.2**-**W1.4**, summary and
discussion (4.75 h).

**Check-in at the end of day 1:** `pytest` passes; the week-1 start-of-week box has finished; the **W1.0.2** file opens
in Python.

**If you are behind** (agreed in advance; each saves the time given):
- **W1.1.3** without the top jets (0.25 h; the top comes back in **W1.1.10**).
- *Two-day:* one check-yourself question per primer section instead of all (0.25 h).
- *Two-day:* **W1.1.5** without the QCD ladder, the W jet of **W1.1.4** only (0.25 h).
- **W1.2.2** with the masses and the $N$ question only (0.25 h; you lose where the jet loses $p_T$).
- **W1.3.2** for W and QCD jets only (0.25 h; you lose the top and its tail near the W mass).
- Never skip **W1.1.4**, **W1.1.6** and **W1.4.2**: weeks 2-4 build on them. If **W1.0** and **W1.1** take you more than
  about 11 hours, agree the fallbacks of this week at the check-in or at the start of day 2. Plan those of week 2 before
  it starts.

**First stretch picks:** **W1.1.7** (the JetClass sign, used again in week 4), **W1.2.3** (the Lund plane of one
jet), **W1.2.4** (the Lund density), **W1.4.4** (your own $\tau_N$), **W1.3.5** (the Sudakov peak).

### Data for week 1

All files go to `data/` in the repository root. Every path in this document is relative to the repository root (a
notebook in `notebooks/` reads the same file as `../data/t1_detector.parquet`). `python examples/make_task_data.py
--week 1` runs exactly these commands on any operating system with `--workers auto`. It keeps files that already exist:
`--overwrite` makes them again, `--dry-run` only prints them and `--only TEXT` runs the commands whose command line
contains TEXT. Together they make about 290 000 jets and 0.48 GB, in about 5 minutes on one 2016 core or 1.5-2 minutes
with 8 workers. On a cluster, run them in a job or an interactive session, not on the login node
([examples/cluster](examples/cluster/README.md)). The file names keep the task numbers of release 0.1 (`t1_`, `t2_`,
`t3_`, `t6_`, `t7_`): they are names, not question IDs.

Files made with the same random seed share their hard processes jet by jet. The hard process is the first step of the
toy: it makes the quark, gluon, W or top that starts the jet. The `t1_*` ladder holds the same events at every preset.
In week 3, `t3_w80_noue`, `t3_w80_centre` and `t3_w80_nocoh` are `t3_w80` with one thing changed
([physics.md](docs/physics.md#reproducibility)). Colour coherence is the effect that a soft gluon at a wide
angle sees only the total colour of a pair ([the parton shower](docs/primer.md#the-parton-shower)). Without it, the
whole shower is drawn again, so `t3_w80_nocoh` shares only the hard processes. Such paired samples are made for
differences, not for pooling into one larger sample. A change of the hard process (for example `--w-mass`) gives an
independent sample, even with the same random seed.

```bash
# W1.2-W1.4, W3.3: the physics ladder, 10 000 jets per process, the same events at every preset
jetgen generate --preset partons -n 30000 --seed 11 -o data/t1_partons.parquet
jetgen generate --preset shower -n 30000 --seed 11 -o data/t1_shower.parquet
jetgen generate --preset hadrons -n 30000 --seed 11 -o data/t1_hadrons.parquet
jetgen generate --preset detector -n 30000 --seed 11 -o data/t1_detector.parquet
# W1.3.3: the same jets without the underlying event
jetgen generate --preset detector --no-ue -n 30000 --seed 11 -o data/t1_detector_noue.parquet
# W1.2.4, W1.2.6, W1.2.10, W1.2.11, W1.3.5: 600 GeV quark and gluon jets, fixed coupling (leading-log formulas)
jetgen generate --preset shower --process q --alphas-mode fixed --alphas-value 0.12 --pt-spectrum fixed --pt-fixed 600 -n 20000 --seed 12 -o data/t1_lund_q.parquet
jetgen generate --preset shower --process g --alphas-mode fixed --alphas-value 0.12 --pt-spectrum fixed --pt-fixed 600 -n 20000 --seed 12 -o data/t1_lund_g.parquet
# W1.2.5, W1.3.6, W2.5.6: quark and gluon jets with the running coupling through the ladder
jetgen generate --preset shower --process q,g -n 20000 --seed 13 -o data/t1_qg_shower.parquet
jetgen generate --preset hadrons --process q,g -n 20000 --seed 13 -o data/t1_qg_hadrons.parquet
jetgen generate --preset detector --process q,g -n 60000 --seed 13 -o data/t1_qg_detector.parquet
```

Every file is read with `load` ([primer section 10](docs/primer.md#the-toy-and-its-files) explains the columns, the
padding and the chunks):

```python
from itj.io import load

data = load("data/t1_detector.parquet")
data["jet_sdmass"], data["part_px"]  # a per-jet array [N]; a Ragged column (flat values and offsets)
p4, mask = data.head(2000).padded_p4()  # (2000, P, 4) float64 (px, py, pz, E) and (2000, P) bool
for chunk in data.chunks(2000):  # anything that builds (B, N, N) arrays: work on chunks
    ...
```

### W1.0 Set-up (core path about 2 h, stretch about 1.75 h)

**Goal:** a working installation, the data of week 1 in the making and a first file opened in Python.

**W1.0.1** *Warm-up, 60 min.* Install and test. Clone the repository, create a virtual environment, install the package
and run the tests.

```bash
git clone https://github.com/hqucms/inside-the-jet.git
cd inside-the-jet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux without a GPU: the small CPU build (docs/install.md)
python -m pip install -e ".[analysis,dev]"
python -m pytest                 # ends with "... passed, ... skipped, ... deselected" (see below)
python -m ipykernel install --user --name inside-the-jet --display-name "Python (inside-the-jet)"
jetgen presets                   # the physics ladder and the default configuration
```

What these lines do. A *virtual environment* (`.venv`) is a private copy of Python for this course, so that its
packages do not mix with others. *Activate* it in every new terminal. An *editable install* (`-e`) makes `import itj`
read the files of your checkout, so a change there needs no reinstall. `pytest` runs the tests of the package. In its
summary line, "passed" counts the tests that ran and passed. "Skipped" counts those that need an optional package
(fastjet, weaver) or a download of a later week. "Deselected" counts the slow ones, which are left out by default. Only
"failed" or "error" is a problem. The repository is `inside-the-jet`; the package it installs is `itj` (the toy is
`itj.toy`, its command line `jetgen`, the same as `python -m itj.toy`). Windows, a smaller CPU-only PyTorch, Apple
Silicon and common problems: [docs/install.md](docs/install.md).

**W1.0.2** *Core, 60 min.* A first data set and a first look. First start the data of the week in a second terminal (it
runs while you read the primer):

```bash
python examples/make_task_data.py --week 1
```

Then make your first file, describe it and load it in Python:

```bash
jetgen generate -n 3000 --seed 1   # -> data/toy_detector_qcd-wqq-tbqq_n3000_seed1.parquet
jetgen info data/toy_detector_qcd-wqq-tbqq_n3000_seed1.parquet
```

```python
import numpy as np

from itj.io import load

data = load("data/toy_detector_qcd-wqq-tbqq_n3000_seed1.parquet")
print(data)
i = int(np.flatnonzero(data["label_Wqq"])[0])  # the first W jet
jet = data.jet(i)
print(jet.jet)  # the per-jet branches: jet_*, label_*, aux_*, truth_*
print(jet.particles_frame())  # one row per particle: part_* and truth_part_* (pandas shortens wide tables)
print(data.names)  # every column of the file
```

*What is in the file.* Each row is one jet. "QCD jets" come from a single quark or gluon. "W jets" and "top jets" come
from the decay of a W or a top ([primer section 5](docs/primer.md#what-a-jet-is)). A branch is a column. The `part_*`
columns hold one value per particle and the `jet_*` columns one value per jet. The `truth_*` columns hold what only the
simulation knows, and `label_*` says which process made the jet
([primer section 10](docs/primer.md#the-toy-and-its-files) explains them). How many jets of each process does the file
hold? Which branches does it have per jet, per particle and as truth? How many particles does the first W jet have? From
here on this file is "the **W1.0.2** file".

**W1.0.3** *Stretch, 75 min.* What every step of the ladder costs. Time the generation of 1 000, 3 000 and 10 000 jets
for every preset (`data.provenance["wall_time_s"]`), and one run with `--workers 4`. Plot the time against $N$, here the
number of generated jets, for each preset. Which stage dominates (`data.provenance["stage_times_s"]`,
`jetgen info --provenance`)? Extrapolate to the data of week 1. The runs take about a minute in all.

```python
from itj.toy import generate, preset

for name in ("partons", "shower", "hadrons", "detector"):
    data = generate(3000, config=preset(name), seed=3)
    print(name, f"{data.provenance['wall_time_s']:.2f} s", data.provenance["stage_times_s"])
```

**W1.0.4** *Stretch, 30 min.* Validate the toy. Run `jetgen validate -n 3000` (about 30 s) and read its table next to
[physics.md](docs/physics.md#calibration-and-validation): what is validated, and against what? Read it again after
week 1, when the rows mean something.

**Deliverables:** the summary line of the tests (**W1.0.1**); the number of jets per process, the number of particles of
the first W jet, and the wall time of the week-1 start-of-week box once it has finished (**W1.0.2**);
`W1.0.3_timing.pdf` (**W1.0.3**); two sentences on the validation table (**W1.0.4**).

**Hints**
- **W1.0.1**, troubleshooting. If `jetgen` is not found, the environment is not activated (activate it, or use
  `python -m itj.toy`). If `pytest` is not found, the `dev` extra is missing. About 30 "import file mismatch" errors
  mean a non-editable install (reinstall with `-e`, [install.md](docs/install.md#common-problems)). On Linux without a
  GPU, install the CPU build of PyTorch first, or pip pulls the multi-GB CUDA build for NVIDIA GPUs. The Windows
  activation line is in [install.md](docs/install.md#windows). "module 'itj' has no attribute ..." means a submodule was
  not imported (`import itj.toy`, `import itj.observables`).
- **W1.0.2:** jet 0 of the file is a QCD jet, which is why the snippet looks for the first W jet with `label_Wqq`. The
  data script keeps files that exist, and `--dry-run` prints its commands. `particles_frame()` is a pandas table:
  `.to_numpy()` gives a column as an array.
- **W1.0.3:** `--workers auto` uses up to 8 processes, and the output does not depend on the number of workers. In a
  script, `generate(..., workers=4)` needs the `if __name__ == "__main__":` guard (otherwise BrokenProcessPool). Wall
  time, the time on the clock, is not CPU time, the time summed over all cores. `jetgen generate --help` lists every
  option with its default, grouped by stage.

### W1.1 Jets from zero (core path about 7 h, stretch about 4.75 h)

**Goal:** the language of the course, each word tried out on one jet of the toy. Each question names the sections of
[docs/primer.md](docs/primer.md) to read first. It also names the cell of
[notebooks/00_primer.ipynb](notebooks/00_primer.ipynb) that shows the tool on another object. The time of both is
inside the question's time. The displayed jets are the W jet of random seed 12, the top jet of random seed 3 and the QCD
jet of random seed 2.

**W1.1.1** *Warm-up, 60 min.* The mass of two particles. Read [primer sections 1-2](docs/primer.md#units-and-energies)
and cells 1-3 of 00_primer. By hand: take two massless particles of 300 GeV each at an opening angle of 0.27 rad.
Compute their invariant mass with the exact formula and with the small-angle form, and say how much the two differ. In
Python: take the W jet of seed 12 at the `partons` level, where it holds only its two decay quarks. Add their
four-vectors and compute the mass of the pair. Compare it with `truth_mass`, the mass the toy drew for this W (the W
mass is not one number; [the primer](docs/primer.md#the-particles) says why, and **W1.2.2** returns to it). Then do the
same for the top jet of seed 3 at `partons` (three quarks).

```python
import numpy as np

import itj.observables
import itj.toy

jets = itj.toy.show_ladder("wqq", seed=12)  # the W jet of seed 12 at the four presets: one JetData each
p4, mask = jets["partons"].padded_p4()  # four-vectors (px, py, pz, E) and the mask of the real particles
print(p4.shape, mask.shape, jets["partons"]["truth_mass"])
tops = itj.toy.show_ladder("tbqq", seed=3)  # the top jet of seed 3
```

**W1.1.2** *Warm-up, 45 min.* Where is a particle? Read [primer section 3](docs/primer.md#collider-coordinates). Take
the `detector`-level W jet of **W1.1.1**. For every particle, compute the transverse momentum $p_T$, the azimuth $\phi$,
the rapidity $y$ and the pseudorapidity $\eta$ from `part_px`, `part_py`, `part_pz` and `part_energy`. Check them in one
`allclose` against `itj.observables.pt_y_phi_m`. How much do $y$ and $\eta$ differ in this jet, and why so little?
Compute $\Delta R$ of the two hardest particles (the two with the most transverse momentum) in $(y, \phi)$, with
$\Delta\phi$ wrapped. Do they come from the same prong (`truth_part_prong`)? Then compute $\Delta R$ of the hardest
particle of each prong.

```python
import itj.observables
import itj.toy

jets = itj.toy.show_ladder("wqq", seed=12)
frame = jets["detector"].jet(0).particles_frame()  # one row per particle, sorted by pT
px = frame["part_px"].to_numpy()
```

**W1.1.3** *Core, 105 min.* Why W and top jets have prongs. Read [primer sections 4-6](docs/primer.md#the-particles) and
cell 4 of 00_primer. The goal is a formula for the opening angle $\Delta R$ of a boosted two-body decay (a fast particle
that decays into two) with equal momentum sharing. Start from the small-angle mass of **W1.1.1**. Derive the angle for
a W at $\eta = 0$, where $E = p_T$ and the angle between the quarks is $\Delta R$. A boost along the beam does not
change $\Delta R$ ([primer sections 2-3](docs/primer.md#four-vectors-and-invariant-mass)): use this to carry the result
to every $\eta$. Evaluate it for a W of 600 GeV. The W jets of the **W1.0.2** file have many different $p_T$, so
compare jet by jet. For each W jet, evaluate your formula at the $p_T$ of that W (`truth_seed_pt`) and at its drawn
mass (`truth_mass`). Then divide the separation of its two quarks (`truth_dr_min`) by it. Histogram this ratio. Where
does your formula sit in the distribution: is it the typical, the smallest or the largest separation, and why? Then add
the top jets. Draw `truth_dr_min` of the W and of the top jets in one figure, and say in two sentences why the top
distribution differs. (The stretch question **W1.1.10** measures which pair of its quarks is the closest.)

```python
import numpy as np

from itj.io import load

data = load("data/toy_detector_qcd-wqq-tbqq_n3000_seed1.parquet")
dr, process = data["truth_dr_min"], data.process_names  # process: "qcd", "wqq" or "tbqq"
pt_seed, m_drawn = data["truth_seed_pt"], data["truth_mass"]  # pT and drawn mass of the W (or top) of each jet
```

**W1.1.4** *Core, 90 min.* One jet up the ladder. Read [primer sections 7-8](docs/primer.md#the-parton-shower), the
truth table of [primer section 10](docs/primer.md#the-toy-and-its-files) and cell 5 of 00_primer. *Predict first:* how
many particles does the W jet have at the `detector` level? Then draw the W jet of seed 12 at the four presets and
answer these questions for this jet. How many particles are there at each level, and what did the shower add? Where are
the prongs? Which detector particles come from which prong (`truth_part_prong`: 0 and 1 for the two quarks)? Which come
from the underlying event, the soft particles from the rest of the same proton-proton collision (`truth_part_is_ue`)?
What fraction of the $p_T$ do the 5 hardest particles carry at `hadrons` and at `detector`?

```bash
jetgen display --ladder --process wqq --seed 12 --jet 0 -o W1.1.4_ladder.png
```

```python
import itj.toy

jets = itj.toy.show_ladder("wqq", seed=12)  # the same jet as the display
frame = jets["hadrons"].jet(0).particles_frame()  # the same for "partons", "shower" and "detector"
```

**W1.1.5** *Core, 45 min.* Why the number of particles changes. Read [primer sections
9-10](docs/primer.md#what-a-detector-sees). In this toy, why does the number of particles grow by a large factor from
`shower` to `hadrons`, and why does it shrink again at `detector`? Draw the ladder of the QCD jet of seed 2 with the
underlying event coloured. Compare it with the W jet of **W1.1.4**: what is different at `partons`, and what at
`detector`?

```bash
jetgen display --ladder --process qcd --seed 2 --color ue -o W1.1.5_ladder_qcd.png
```

**W1.1.6** *Core, 75 min.* Your first histograms (write your own). Read [primer sections
11-12](docs/primer.md#histograms-and-errors) and cell 7 of 00_primer. *Predict first:* which of the multiplicity $N$
(the number of particles), the soft-drop mass and $\tau_{21}$ separates W jets from QCD jets best? Which separates top
jets from W jets? Write a small function for a normalised histogram (scaled to unit area) with $\sqrt{n}$ error
bars; you will reuse it all course. With it, draw `jet_nparticles`, `jet_sdmass` and $\tau_{21}$ =
`jet_tau2`/`jet_tau1` for the three processes of the **W1.0.2** file, one panel each. Compare the result with your
prediction in two sentences.

**W1.1.7** *Stretch, 45 min.* Relative coordinates and the JetClass sign. Take the W jet of **W1.1.2**. Compute
$\Delta\eta$ and $\Delta\phi$ of every particle relative to the jet axis, the direction of the whole jet (`jet_eta`,
`jet_phi`, $\Delta\phi$ wrapped), and reproduce `part_deta` and `part_dphi`. Your first attempt will disagree for one of
them: why ([primer section 3](docs/primer.md#collider-coordinates))? Then take a jet with `jet_eta` above $+1$ from
the **W1.0.2** file and check that the same rule holds. The rule returns in week 4 (**W4.1.4**, **W4.3.2**).

**W1.1.8** *Stretch, 90 min.* A shower by hand. `itj.toy.shower_one` is the readable version of the toy's shower (cell 6
of 00_primer shows one quark shower). Shower a 300 GeV quark and a 300 GeV gluon 2 000 times each from a start angle of
1. The start angle is the largest angle (in radians) that the first emission may have. This takes about 0.3 s for 4 000
showers. Compare the ratio of the mean numbers of emissions of gluons and quarks, with its error, with $C_A/C_F$, the
ratio of the colour factors of a gluon and a quark. Say why the two need not agree: which emissions does $C_A/C_F$
describe? Histogram $\theta$ and the soft fraction $\min(z, 1 - z)$ of the emissions on log axes, and relate them to the
emission probability of [primer section 7](docs/primer.md#the-parton-shower). Follow the angles along one chain: do they
shrink?

```python
import numpy as np

import itj.toy

cfg = itj.toy.preset("shower").shower
rng = np.random.default_rng(0)
emissions = itj.toy.shower_one(300.0, "q", 1.0, cfg, rng)  # one dict per emission: theta, z, kt, depth, primary, ...
```

**W1.1.9** *Stretch, 60 min.* What the detector does. Generate the same 2 000 jets at `hadrons` and at `detector` (about
9 s), and check the pairing with `truth_event_id`. For each level, find the mean number of charged hadrons, photons and
neutral hadrons per jet (`part_isChargedHadron`, `part_isPhoton`, `part_isNeutralHadron`). Which kinds does the detector
lose or merge most, and at which $p_T$? Relate this to the thresholds (the smallest momenta or energies the detector
records) and the cell sizes of [physics.md](docs/physics.md#detector).

```python
import itj.toy

hadrons = itj.toy.generate(2000, config=itj.toy.preset("hadrons"), seed=5)
detector = itj.toy.generate(2000, config=itj.toy.preset("detector"), seed=5)
```

**W1.1.10** *Stretch, 90 min.* A W decay, boosted, and the top. Decay a W at rest into two massless quarks at random
angles and boost it to $p_T = 300$, 600 and 1000 GeV. Plot $\Delta R$ between the quarks against $\cos\theta^*$, the
decay angle in the frame where the W is at rest. What is the smallest opening angle, and how does it compare with
your **W1.1.3** formula? Below which $p_T$ would the two quarks no longer fit into a jet of $R = 0.8$? Then turn to the
top. Use the three decay quarks of the top jets of `data/t1_partons.parquet` (`truth_part_prong`: 0 the b quark, 1 and 2
the quarks of the W). Which $p_T$ does the W inside the top carry (the summed $p_x$, $p_y$ of quarks 1 and 2)? How often
is each of the three pairs the closest? Compare with what you said in **W1.1.3**. The start:

```python
import numpy as np
from itj.toy.kinematics import boost, delta_r, pt_eta_phi, rapidity, two_body

n, m, pt = 1000, 80.4, 600.0
rng = np.random.default_rng(0)
cos, phi = rng.uniform(-1, 1, n), rng.uniform(0, 2 * np.pi, n)
q1, q2 = two_body(np.full(n, m), 0.0, 0.0, cos, phi)  # rest frame, (n, 4) each
beta = np.array([pt / np.hypot(pt, m), 0.0, 0.0])  # a W of this pT moving along x
q1, q2 = boost(q1, beta), boost(q2, beta)
```

Then `pt_eta_phi`, `rapidity` and `delta_r` give the angles.

**Deliverables:** the two hand values and the W and top masses next to `truth_mass` (**W1.1.1**); the largest
$\lvert y - \eta\rvert$ in the jet, the `allclose` line and one sentence on why (**W1.1.2**); the derivation in three
lines and `W1.1.3_dr_min.pdf` with two panels, the ratio for the W jets and `truth_dr_min` for the W and top jets
(**W1.1.3**); `W1.1.4_ladder.png` and a table of particles per level and per origin: prong 0, prong 1, underlying event
(**W1.1.4**); `W1.1.5_ladder_qcd.png` and a short paragraph (**W1.1.5**); `W1.1.6_first_look.pdf` and your histogram
function (**W1.1.6**); two `allclose` lines and two sentences on the sign (**W1.1.7**); `W1.1.8_shower.pdf`
(**W1.1.8**); `W1.1.9_detector.pdf` (**W1.1.9**); `W1.1.10_opening.pdf` and the fractions of the closest pairs
(**W1.1.10**).

**Hints**
- **W1.1.1:** a four-vector is stored as $(p_x, p_y, p_z, E)$. `padded_p4()` returns arrays for many jets at once.
  `p4` has the shape (jets, particles, 4) and `mask` (jets, particles), here (1, 2, 4) and (1, 2), so take `[0]` for the
  one jet. Short jets are filled up with zeros, and the mask marks the real particles; cell 3 of 00_primer and the
  paragraph *JetData* of [primer section 10](docs/primer.md#the-toy-and-its-files) explain this padding. Check your
  result with `itj.observables.mass(p4, mask)`. A difference in the fourth digit comes from storing momenta in float32
  (single precision).
- **W1.1.2:** `np.arctan2` and the wrap to $(-\pi, \pi]$; [physics.md](docs/physics.md#units-and-coordinates) has the
  formulas.
- **W1.1.3:** `truth_dr_min` is NaN (not a number) for QCD jets. Use `truth_seed_pt`, the $p_T$ of the W itself, not
  `jet_pt`. The jet has lost some radiation and has been through the detector, so its $p_T$ is a little different.
  Equal sharing $z = 1/2$ is a special case of $m^2 \approx z(1 - z)\,p_T^2\,\Delta R^2$: what does unequal sharing do
  to $\Delta R$? Keep this figure: **W1.4.2** and **W3.2.2** come back to it.
- **W1.1.4:** `truth_part_prong` is $-1$ for the underlying event (or an unassigned particle), and `truth_part_is_ue`
  marks the underlying event. The shower level of the same jet answers "what did the shower add". The jets of the
  **W1.0.2** file are other events than the displayed one: use `show_ladder`. For the $p_T$ share, divide by the scalar
  sum of the particle $p_T$. `jet_pt` is the $p_T$ of the summed four-vector, which is smaller.
- **W1.1.5:** see [physics.md](docs/physics.md#presets) (the shower stops at about 1 GeV, hence "in this toy") and
  [physics.md](docs/physics.md#detector) (thresholds, cells).
- **W1.1.6:** a density divides by the number of jets and the bin width, and so does its error. Guard against
  $\tau_1 = 0$.
- **W1.1.7:** the JetClass convention is in [physics.md](docs/physics.md#units-and-coordinates).
- **W1.1.8:** `z` is the momentum fraction of the first daughter (close to 1 for a soft gluon off a quark), hence
  $\min(z, 1 - z)$. See [physics.md](docs/physics.md#parton-shower) and [its branching
  probability](docs/physics.md#branching-probability). The cutoff in $k_t$ limits the soft (low-energy) and collinear
  (small-angle) emissions.
- **W1.1.10:** at `partons` the particles of a top jet are its three decay quarks. The small-angle mass of **W1.1.1**
  holds for every pair, with the mass of that pair.

### W1.2 The ladder on many jets (core path about 1.25 h, stretch about 9 h)

**Goal:** what each step of the ladder does to many jets, measured on paired events. Stretch: the emissions inside a jet
as a picture, then the shower against the leading-log formulas (Lund density, Sudakov factor). Leading log is the
approximation behind the emission probability in [the primer](docs/primer.md#the-parton-shower): only soft,
small-angle emissions, each independent of the others.

**W1.2.1** *Warm-up, 30 min.* The ladder on many jets. Use the paired files `data/t1_partons.parquet`,
`data/t1_shower.parquet`, `data/t1_hadrons.parquet` and `data/t1_detector.parquet`. With your **W1.1.6** function,
plot per preset and process the multiplicity, the jet mass and $p_{T,J}/p_{T,\mathrm{seed}}$ (`truth_seed_pt`), the
transverse momentum of the jet over that of the seed parton.

```python
import itj.observables
from itj.io import load

data = load("data/t1_detector.parquet")
mass = [itj.observables.mass(*chunk.padded_p4()) for chunk in data.chunks(2000)]  # concatenate the pieces
```

**W1.2.2** *Core, 45 min.* Reading the ladder. Use your **W1.2.1** figure. Why is the QCD jet mass zero at `partons`
(up to the float32 rounding of a massless TeV four-vector: about 1 GeV at most)? Why is the W and top mass the drawn
mass `truth_mass`, and why does the drawn mass vary from jet to jet? Which step changes $N$ most, in this toy? Does your
one-jet answer of **W1.1.5** hold on average? Where does the jet lose $p_T$, and why does this happen mostly to QCD
jets at `shower`? Compare the angle at which the QCD shower starts with $R$
([physics.md](docs/physics.md#start-angles-and-colour-coherence)). Three short answers with numbers.

**W1.2.3** *Stretch, 30 min.* The emissions of one jet. Draw one QCD jet and one W jet of `data/t1_detector.parquet`
with their primary Lund plane. Every point of this plane is one emission, found by undoing the clustering along the
harder branch ([primer section 7](docs/primer.md#the-parton-shower)). It sits at its angle $\Delta$ and transverse
momentum $k_t$. Where is the W splitting, and how is it different from the other points? Find a jet with
`i = np.flatnonzero(data.process_names == "qcd")[0]` (and `"wqq"`). Then draw it with
`jetgen display data/t1_detector.parquet --jet <i> --lund -o W1.2.3_lund_qcd.png` (your number in place of `<i>`;
about 3 s each), and the W jet into `W1.2.3_lund_wqq.png`.

**W1.2.4** *Stretch, 60 min.* The Lund plane density. Histogram the truth primary emissions `truth_lund_*` of
`data/t1_lund_q.parquet` and `data/t1_lund_g.parquet` as a density per jet and per unit area in $(\ln 1/\Delta, \ln
k_t)$, with the cutoff line. Use `itj.observables.lund_density({"ln_inv_delta": d["truth_lund_ln_inv_delta"], "ln_kt":
d["truth_lund_ln_kt"]}, n_jets=len(d), x_edges=..., y_edges=...)`, which returns the density and its error. Measure the
plateau (the flat part of the density) in the box $\ln(1/\Delta) \in [0.5, 2.5]$, $\ln(k_t/\mathrm{GeV}) \in [0.5,
2.0]$ for quarks and gluons, with errors. Compare it with the leading-log density $2\alpha_s C_R/\pi$[^lund] (evaluate
it). Then compare the ratio g/q with $C_A/C_F$.

**W1.2.5** *Stretch, 75 min.* Quark and gluon jets. Use the paired quark/gluon ladder (`data/t1_qg_shower.parquet`,
`data/t1_qg_hadrons.parquet`, `data/t1_qg_detector.parquet`; their first 20 000 jets are the same events). Measure the
ratio $\langle N_g\rangle/\langle N_q\rangle$ of the mean multiplicities of gluon and quark jets at `shower`, `hadrons`
and `detector`, with the error of the ratio. Compare it with $C_A/C_F = 9/4$ and explain the difference
([physics.md](docs/physics.md#multiplicity)). If you did **W1.1.8**, you have half of it.

**W1.2.6** *Stretch, 60 min.* Your own Lund density (write your own). Histogram the emissions of **W1.2.4** yourself
(`np.histogram2d`, divided by $N_\mathrm{jet}$ and the bin area, errors $\sqrt{n}/(N_\mathrm{jet}\,\mathrm{area})$), and
check it against `lund_density` in one `allclose`.

**W1.2.7** *Stretch, 75 min.* The reconstructed Lund plane. Compare the plane of the detector-level jets of
`data/t1_detector.parquet` with the truth (`itj.observables.lund_primary`). Optionally, walk the Cambridge/Aachen
(C/A) tree of `itj.cluster.cluster(p4, mask, algorithm="ca")` yourself; this algorithm merges particles in order of
angle. Where do hadronisation and the detector move the density? Look at low $k_t$ and at small $\Delta$,
and think of the cell size. The C/A clustering of the 30 000 jets takes about 25 s of CPU.

**W1.2.8** *Stretch, 30 min.* Other levels and the W plane. Draw the plane of **W1.2.7** at `shower` and `hadrons`, and
the plane of W jets: where is the W splitting? Keep this plane in mind for week 4 (**W4.5.7**). The Particle Transformer
(ParT; [models.md](docs/models.md#particle-transformer)) computes for each pair of particles the pair features
($\ln k_T$, $\ln z$, $\ln\Delta$). Its $k_T$ is the transverse momentum of the Lund plane, so these features are the
Lund coordinates of every pair.

**W1.2.9** *Stretch, 45 min.* Containment. The containment rule of the toy should keep every prong in the jet: check
this from `truth_prong_deta` and `truth_prong_dphi` for each preset. The toy applies the rule when it generates the jet,
to the direction of the resonance (the decaying W or top), with distances in rapidity. Measured from the final jet axis,
a few prongs in $10^4$ lie just outside $R$: explain why.

**W1.2.10** *Stretch, 105 min.* The Sudakov factor, the probability of no emission between two angles. Take the quark
jets of `data/t1_lund_q.parquet` and the angle of their first emission (`truth_emis_theta` where
`truth_emis_depth == 0`). Jets without any emission count as "no emission above the cutoff". Plot the cumulative
distribution of this angle against the double-log Sudakov factor

```math
-\ln\Delta(\theta_0, \theta) = \frac{\alpha_s C_F}{\pi}\left[\ln^2\frac{E\theta_0}{k_{t,\mathrm{cut}}} - \ln^2\frac{E\theta}{k_{t,\mathrm{cut}}}\right]
```

([physics.md](docs/physics.md#sudakov-factor)). Here $E$ is the quark energy, $\theta_0$ the start angle and
$k_{t,\mathrm{cut}}$ the shower cutoff. Double-log means that only the squared-logarithm terms are kept. Where does it
fail, and why? Check your curve with `itj.toy.sudakov_cdf`.

**W1.2.11** *Stretch, 60 min.* Derive it, and the gluons. Derive the Sudakov factor of **W1.2.10**[^sudakov], and repeat
the comparison for the gluon jets of `data/t1_lund_g.parquet`.

**Deliverables:** `W1.2.1_ladder.pdf` (**W1.2.1**); three short answers with numbers (**W1.2.2**); `W1.2.3_lund_qcd.png`
and `W1.2.3_lund_wqq.png` (**W1.2.3**); `W1.2.4_lund_truth.pdf` and the two plateaus with their errors (**W1.2.4**);
`W1.2.5_qg_ratio.pdf` (**W1.2.5**); the `allclose` line (**W1.2.6**); `W1.2.7_lund_reco.pdf` (**W1.2.7**);
`W1.2.10_sudakov.pdf` (**W1.2.10**).

**Hints**
- **W1.2.1:** the four files hold the same events in the same order ([physics.md](docs/physics.md#reproducibility)). The
  seed parton is the quark or gluon (or the W or top) that the toy starts the jet from ([primer section
  10](docs/primer.md#the-toy-and-its-files)). Compute the mass by chunks of 2 000 jets, because the padded array of 30
  000 jets does not fit comfortably in memory. Your own mass comes in **W1.3.1**.
- **W1.2.2:** the toy draws the W and top masses from a Breit-Wigner distribution. [Primer section
  4](docs/primer.md#the-particles) says what that is and why an unstable particle has no single mass.
  [physics.md](docs/physics.md#hard-process) says how the toy draws it.
- **W1.2.3:** the two figures have different axis ranges, so compare the ranges before the pictures. At `detector` the
  points include splittings between hadrons, which is why some lie below the shower cutoff of about 1 GeV.
  [observables.md](docs/observables.md#the-lund-plane) explains the plane in words first.
- **W1.2.4**, **W1.2.6:** the error of a histogram density is $\sqrt{n}/(N_\mathrm{jet}\,\mathrm{area})$ per bin
  (`itj.observables.lund_density` returns both).
- **W1.2.7:** check your answer afterwards against [physics.md](docs/physics.md#lund-density).

### W1.3 The jet mass (core path about 1.5 h, stretch about 5.25 h)

**Goal:** the most important observable of a W or top tagger, and what grooming does to it. Grooming removes the soft,
wide-angle radiation from a jet and keeps its hard core.

**W1.3.1** *Warm-up, 30 min.* Your own jet mass (write your own). Turn the two-particle mass of **W1.1.1** into a
function of the padded four-vectors of many jets. Compare it with `itj.observables.mass` in one `allclose` on
`data/t1_detector.parquet`, by chunks (`for chunk in data.chunks(2000)`, `p4, mask = chunk.padded_p4()`).

**W1.3.2** *Core, 60 min.* Mass and soft-drop mass. Plot $m$ (your **W1.3.1** mass) and $m_\mathrm{SD}$ (`jet_sdmass`)
per process at `detector` (`data/t1_detector.parquet`) with your **W1.1.6** function. (a) Which peak is narrower for W
and top jets? Measure the width of a peak as its full width at half maximum, or as the fraction of jets within $\pm 10$
GeV of the peak. (b) Which W jets have $m_\mathrm{SD}$ near 0, and which top jets have $m_\mathrm{SD}$ near 80 GeV?
(c) What does grooming remove from QCD jets? Give one number that supports each answer.

**W1.3.3** *Stretch, 45 min.* The underlying event and the mass. Compare `data/t1_detector.parquet` with the same events
without the underlying event (`data/t1_detector_noue.parquet`), jet by jet. How far do the peaks of $m$ and of
$m_\mathrm{SD}$ move, with errors? Which of the two depends less on the underlying event, and why: the plain mass or
$m_\mathrm{SD}$?

**W1.3.4** *Stretch, 75 min.* The top peak and the widths. Where does the top $m_\mathrm{SD}$ peak sit compared with the
top mass, at `hadrons` and at `detector`? What could pull it down (the neutrinos of semileptonic B decays, radiation
that falls outside the jet radius $R$, the detector), and what could push it up (the underlying event)? Test one
explanation on the files (for example with `truth_part_from_b` or `data/t1_hadrons.parquet`). Then: why are the W and
top peaks wider than the Breit-Wigner width? Candidates to test are resolution, radiation outside $R$ and the detector.
Support your answer with one figure.

**W1.3.5** *Stretch, 2 h.* The Sudakov peak of QCD jets. Use the fixed-coupling files `data/t1_lund_q.parquet` and
`data/t1_lund_g.parquet`, made with a constant strength of the strong force. Plot $\rho\,d\sigma/d\ln\rho$, the
distribution of the normalised jet mass (see [Notation](#notation)). Physicists write $\sigma$ for the rate of a
process (the cross-section), so $d\sigma/d\ln\rho$ is the distribution of the jets in $\ln\rho$, not a standard
deviation. Compare the position of its peak in $\ln(1/\rho)$
with the double-log value and with the value that includes the hard-collinear term. Both are in
[physics.md](docs/physics.md#jet-mass); this is a two-line check, no derivation needed[^peak]. The peak is broad, so
quote how much its position moves with the bin width and the fit window.

**W1.3.6** *Stretch, 75 min.* How hadronisation and the detector move the peak. Use the paired quark/gluon ladder
(`data/t1_qg_shower.parquet`, `data/t1_qg_hadrons.parquet`, `data/t1_qg_detector.parquet`). It is made with the
running coupling, a strength of the strong force that changes with energy. The first 20 000 jets of the three files are
the same events. How far
does hadronisation with the underlying event move the peak of **W1.3.5**, and how far does the detector move it after
that?

**Deliverables:** the `allclose` line (**W1.3.1**); `W1.3.2_mass.pdf` and a paragraph with two supporting numbers
(**W1.3.2**); `W1.3.3_ue_shift.pdf` (**W1.3.3**); `W1.3.4_top_peak.pdf` (**W1.3.4**); `W1.3.5_mass_peak.pdf` and the
peak positions with their errors (**W1.3.5**); `W1.3.6_peak_shift.pdf` (**W1.3.6**).

**Hints**
- **W1.3.1:** sum only where `mask` is true, and use `np.clip(..., 0, None)` before the square root.
- **W1.3.2:** read [primer section 12](docs/primer.md#a-first-look-at-the-observables) and
  [observables.md](docs/observables.md#soft-drop) (the lead *In words*). A standard deviation, or a range between
  quantiles, of the whole distribution is dominated by its tails. Measure the peak itself, then look at the tails
  separately. For (b), think of what soft drop removes first and what a top jet holds.
- **W1.3.4:** use fine bins near the peak, or the median, because a shift of about a GeV hides in bins of 4 GeV. B
  hadrons, the hadrons that carry a b quark, are in [physics.md](docs/physics.md#hadronisation). A semileptonic decay is
  a decay with a lepton and a neutrino.
- **W1.3.5:** use logarithmic bins in $\rho$ and leave out the jets with $m = 0$ (many fixed-coupling quark jets have a
  single parton inside $R$).

### W1.4 Counting prongs (core path about 1 h, stretch about 6.75 h)

**Goal:** observables that count prongs, ready for the first tagger of week 2.

**W1.4.1** *Warm-up, 30 min.* $\tau_{21}$, $\tau_{32}$ and $D_2$ per process. The first two are N-subjettiness ratios,
the third a ratio of energy correlation functions ([see the primer](docs/primer.md#a-first-look-at-the-observables)).
At `detector` (`data/t1_detector.parquet`), histogram $\tau_{21}$, $\tau_{32}$ (from the stored `jet_tau*`) and $D_2$
per process with your **W1.1.6** function. Which counts two prongs, and which three?

```python
import itj.observables
from itj.io import load

data = load("data/t1_detector.parquet")
d2 = itj.observables.compute(data, ["d2"])["d2"]  # compute returns a dict of arrays, one per name; about 10 s
```

**W1.4.2** *Core, 30 min.* $\tau_{21}$ and the prong separation. How does $\tau_{21}$ of W jets change with the
separation of the two prongs (`truth_dr_min`)? Plot it, look at both ends of the range and explain each. **W1.1.3**
helps at the large end: what does a large separation at fixed mass say about the momentum sharing $z$? Keep only the W
jets with `data.select(data.process_names == "wqq")`.

**W1.4.3** *Stretch, 30 min.* Jets with too few particles. At `partons` (`data/t1_partons.parquet`), why are
$\tau_{21}$, $\tau_{32}$ and $D_2$ degenerate, that is, limited to a few special values? What is $\tau_2$ of a
two-particle jet? Why is $\tau_1 = 0$ for one particle, and why are $m_\mathrm{SD}$ and $D_2$ undefined there? Give the
counts of the special cases.

**W1.4.4** *Stretch, 90 min.* Your own $\tau_N$ (write your own). With the axes of `itj.observables.subjet_axes`,
compute $\tau_1$ to $\tau_3$ and reproduce `jet_tau1` to `jet_tau3` to within $10^{-5}$.

**W1.4.5** *Stretch, 90 min.* Your own $e_2$, $e_3$ and $D_2$ (write your own). Compute $e_2$, $e_3$ and $D_2$
($\beta = 1$) with $e_3 = \mathrm{tr}\bigl((ZA)^3\bigr)/6$. Here Z is the diagonal matrix of the momentum fractions of
the particles and A the matrix of their pairwise angles
([observables.md](docs/observables.md#energy-correlation-functions)). Check them in one `allclose` against
`itj.observables.ecf` on 10 jets.

**W1.4.6** *Stretch, 75 min.* Power counting. Power counting[^power] estimates how each term of an observable scales
with the small momentum fractions and angles in a jet. Use it to explain why $D_2 \gg 1$ for one-prong jets and
$D_2 = O(1)$ for two-prong jets ([physics.md](docs/physics.md#power-counting)). Then study $D_2$ at $\beta = 2$, and at
the other levels of the ladder.

**W1.4.7** *Stretch, 2 h.* Your own question. A free study with a figure. Some examples: the dead cone of b quarks
(`--dead-cone`), the reduced radiation of a heavy quark at small angles; the W decay angle (`truth_cos_theta_star`);
pull and colour coherence (`--no-coherence`). Two more: the displacement of B-hadron decay products in top jets
(`truth_part_from_b`, `part_d0val`), and angularities for quark and gluon jets
([observables.md](docs/observables.md#angularities-and-pt_d)).

**Deliverables:** `W1.4.1_prongs.pdf` (**W1.4.1**); `W1.4.2_tau21_vs_dr.pdf` (**W1.4.2**); the counts of the special
cases (**W1.4.3**); the `allclose` lines (**W1.4.4**, **W1.4.5**); a figure of your own (**W1.4.7**).

**Hints**
- **W1.4.1:** read [observables.md](docs/observables.md#n-subjettiness) and [the energy correlation
  functions](docs/observables.md#energy-correlation-functions), each with a lead *In words*. Guard against $\tau_1 = 0$
  (and $\tau_2 = 0$) before taking ratios. Before you read the histograms, decide from the definitions whether a jet
  with a hard two-prong core should give a small or a large $D_2$. Which of the three processes have such a core?
- **W1.4.2:** use a 2D histogram, or the median of $\tau_{21}$ in bins of `truth_dr_min`. At the small end, compare the
  separations with the size of the detector cells ([physics.md](docs/physics.md#detector)).
- **W1.4.3:** write out $\tau_N$ for two particles and two axes.
- **W1.4.4:** WTA-kT axes (winner-take-all axes from kT clustering) from another routine will not pass.
- **W1.4.5:** work on chunks of 2 000 jets for anything that builds $(B, N, N)$ arrays (jets by particles by particles).
  Use `@` (matmul) for $e_3$, not a three-operand `np.einsum`, which needs much more memory and is about 7 times slower.

#### Weekly summary

Write half a page that answers the thread question of the week: *What is a W jet made of, and what tells it apart from
a QCD jet?* Use three of your figures (0.5 h with the hand-in). Hand in `answers/week1.md` with the summary and the
answers to every warm-up and core ID. Add the figures `W1.1.3_dr_min.pdf`, `W1.1.4_ladder.png` with its table,
`W1.1.5_ladder_qcd.png`, `W1.1.6_first_look.pdf`, `W1.2.1_ladder.pdf`, `W1.3.2_mass.pdf`, `W1.4.1_prongs.pdf` and
`W1.4.2_tau21_vs_dr.pdf`, plus whatever stretch questions you did. In the discussion (0.5 h) we go through the core IDs.

[^sudakov]: Take the soft limit of $P_{qq}(z) = C_F(1 + z^2)/(1 - z)$ and write $k_t \approx (1 - z)E\theta$. The
    probability of no emission is the exponential of minus the integral of the emission density over the region between
    $\theta$ and $\theta_0$ with $k_t > k_{t,\mathrm{cut}}$.
[^lund]: Write the soft-collinear emission density $\frac{\alpha_s}{2\pi}\frac{d\theta^2}{\theta^2}P(z)\,dz$ in the
    variables $\ln(1/\theta)$ and $\ln k_t$ at fixed $\theta$, with $k_t = (1 - z)E\theta$.
[^peak]: With $\Sigma(\rho) = e^{-\mathcal{R}(\rho)}$, the distribution in $L = \ln(1/\rho)$ is
    $\mathcal{R}'e^{-\mathcal{R}}$; set its derivative to zero.
[^power]: For a one-prong jet, $e_2$ comes from one soft-collinear emission. What does $e_3$ need beyond that emission,
    and how does it scale with $e_2$? For a two-prong jet, $e_2$ is fixed by the hard splitting; what does $e_3$ need?

---

## Week 2: Taggers, from cuts to transformers (core path about 12.75 h, stretch about 11.5 h)

**Why this week.** You can tell W and top jets from QCD jets by eye in the mass and $\tau_{21}$. Now a computer does it,
one step at a time: one cut, trees on 12 numbers, a neural network on 12 numbers, a network on all particles, graphs and
attention. At each step, ask what it buys and what it costs.

**Thread.** Last week gave the observables that separate the classes. This week you build taggers that use them, then
taggers that see the particles. A tagger is a function that gives each jet a score, large for jets that look like the
signal ([the ml_primer](docs/ml_primer.md#a-tagger-is-a-function)). Next week looks inside the trained models of
`models/course/`.

**Concepts this week**, in order of use, with the ml_primer section that explains them. From [ml_primer
1](docs/ml_primer.md#a-tagger-is-a-function): score and threshold, signal and background efficiency, the ROC curve
in the $1/\epsilon_B$ convention, the AUC (area under the ROC curve), $R_{50}$ and
the working point. From [ml_primer 2](docs/ml_primer.md#train-validation-and-test): train, validation and test parts,
and bootstrap errors. From [ml_primer 3](docs/ml_primer.md#decision-trees-and-boosting): a decision tree as a set of
cuts, boosting and one-vs-one scores. From [ml_primer 4](docs/ml_primer.md#neural-networks) and [ml_primer
5](docs/ml_primer.md#pytorch-in-one-page): tensors and gradients, a linear model, logistic regression and a multi-layer
perceptron (MLP), loss, optimiser, learning rate, mini-batch and epoch, overfitting, standardisation and early stopping.
From [ml_primer 6](docs/ml_primer.md#networks-on-particles): sets, padding masks and permutation invariance, Deep Sets
and the Particle Flow Network (PFN), and the `state_dict`. The rest of that section is reading only: graphs and
EdgeConv, attention, transformers, the class token and the pair bias. Stretch adds permutation importance, learning
curves, seeds, inference speed and IRC safety by construction in the Energy Flow Network (EFN;
[observables.md](docs/observables.md#irc-safety)).

**This week at a glance** (times in minutes, or hours where marked):

| task | warm-up | core | stretch |
|---|---|---|---|
| start of week | box (15) | | |
| **W2.1** Your first tagger: cuts | **W2.1.1** 45 | **W2.1.2** 90 | **W2.1.3** 60 |
| **W2.2** Boosted decision trees | **W2.2.1** 30 | **W2.2.2** 45 | **W2.2.3** 45, **W2.2.4** 45 |
| **W2.3** A neural network from zero | **W2.3.1** 60, **W2.3.2** 75 | **W2.3.3** 90 | **W2.3.4** 45 |
| **W2.4** Networks that see particles: Deep Sets | **W2.4.1** 45 | **W2.4.2** 105 | **W2.4.3** 45, **W2.4.4** 2 h |
| **W2.5** Graphs and attention | **W2.5.1** 45 | **W2.5.2** 60 | **W2.5.3** 2 h, **W2.5.4** 60, **W2.5.5** 30 + GPU, **W2.5.6** 2 h |
| summary, hand-in and discussion | | 1 h | |

**Start of week** (15 minutes, then about 1-1.5 hours in the background while you work):

```bash
python examples/make_task_data.py --week 2
python examples/train_tagger.py --course-models --out models/course --results models/course/results_table.csv --threads 4
```

The first line makes `data/t2_detector.parquet` (150 000 jets, 0.38 GB, about a minute). The second line trains the five
course models into `models/course/`, one after the other, all with random seed 0. They are the PFN and the EFN (105 000
jets, 10 epochs) and PN-lite, a small ParticleNet (a graph network; see [models.md](docs/models.md#particlenet)), with
60 000 jets and 5 epochs. The last two are ParT-small models with and without the pair bias (extra inputs that ParT
computes for each pair of particles; see [models.md](docs/models.md#particle-transformer)). They are trained on 30 000
jets for 5 epochs. You meet these models in **W2.4** and **W2.5**. For now they are the networks that this week builds
up to, and weeks 3 and 4 use them as well.

Set `--threads` to the number of your cores minus two (4 on a six-core laptop), so that your own work stays responsive.
With all cores and nothing else running, the five take about 36 minutes (135 + 110 + 524 + 820 + 562 s); with the cores
shared, about 1-1.5 hours. Peak memory is about 9 GB during ParT-small. With 8 GB, close other sessions that hold
`t2_detector`, or run the start-of-week box overnight. `--results` writes the result rows of these models into
`models/course/`, so your own `results_table.csv` holds only your own taggers. While the start-of-week box runs,
notebook 02 and your own trainings use `torch.set_num_threads(2)` (the notebook's first cell sets it). If a run failed,
run the command of the start-of-week box again overnight (it trains one model after the other) and say in your answers
which models you used.

**Day plan.** Day 1 (about 6 h): the start-of-week box, **W2.1.1**-**W2.1.2**, **W2.2.1**-**W2.2.2** and
then **W2.3.1**-**W2.3.2**. Day 2 (about 6.75 h): **W2.3.3**, **W2.4.1**-**W2.4.2**, **W2.5.1**-**W2.5.2**, the weekly
summary and the discussion. Day 2 holds the two hardest PyTorch steps (**W2.3.3** and **W2.4.2**); if it is tight, read
the warm-up **W2.5.1** on the evening of day 1. Day 3 (optional): the first stretch picks and catch-up. Three shorter
days instead: box, **W2.1**, **W2.2** (3.75 h) / **W2.3** (3.75 h) / **W2.4**, **W2.5**, summary, discussion (5.25 h).

**Check-in at the end of day 1:** the rows of **W2.1.2** are written; `models/course/pfn.pt` exists and the
start-of-week box is still running; the first cell of notebook 02 runs.

**If you are behind** (agreed in advance; each saves the time given):
- **W2.1.2** with the W tagger only (0.5 h; the top tagger with its window is lost).
- **W2.2.1** with the tree drawn, no comparison with **W2.1.2** (0.15 h; the link from cuts to trees is lost).
- **W2.3.3** with early stopping from `itj.train.fit(..., patience=...)` instead of by hand (0.5 h; writing early
  stopping and the best-state restore is lost).
- *Two-day:* **W2.4.2** with the permutation check and the key map on one batch, no padding check (0.25 h).
- *Two-day:* **W2.5.2** with the ROC figure and without the cost table (0.25 h; the costs are lost).

**First stretch picks:** **W2.2.3** (permutation importance), **W2.4.4** (your own EFN), **W2.5.3** (the learning curve,
in the background), **W2.1.3** (better cuts).

### Data for week 2

One file: the training set of weeks 2-4. `python examples/make_task_data.py --week 2` makes it (about a minute with 8
workers).

```bash
# weeks 2-4: the training set: QCD, W and top at detector level, 50 000 jets each (split with itj.features.split)
jetgen generate -n 150000 --seed 21 -o data/t2_detector.parquet
```

All questions of this week use `data/t2_detector.parquet`, split with `itj.features.split` into a train, a validation
(val) and a test part (105 000, 15 000 and 30 000 jets). The trained taggers have three outputs (QCD, W, top). Compare W
against QCD and top against QCD with the one-vs-one scores of `itj.metrics.one_vs_one`. Report the AUC and $R_{50}$
with bootstrap errors. Add a row to your `results_table.csv` for every tagger of your own; the table starts this week.

```python
import itj.features
from itj.io import load

data = load("data/t2_detector.parquet")
train, val, test = itj.features.split(data)  # test = first 20 %, val = next 10 %, train = the rest
p4, mask = test.head(2000).padded_p4()  # (2000, P, 4) float64 (px, py, pz, E) and (2000, P) bool
```

### W2.1 Your first tagger: cuts (core path about 2.25 h, stretch about 1 h)

**Goal:** learn what a tagger is and how to measure one, and build the cut-based tagger that every later tagger is
compared with.

**W2.1.1** *Warm-up, 45 min.* One window, two efficiencies. Read [ml_primer sections
1-2](docs/ml_primer.md#a-tagger-is-a-function). Split `data/t2_detector.parquet`. On the val part, keep the W and QCD
jets. Count how many pass the window $65 < m_\mathrm{SD} < 105$ GeV of notebook 00_quickstart. This gives the signal
efficiency $\epsilon_S$ (the fraction of W jets that pass), the background efficiency $\epsilon_B$ (the fraction of QCD
jets that pass) and the rejection $1/\epsilon_B$. Repeat with a narrower and a wider window. Draw the three points as
$1/\epsilon_B$ (log axis) against $\epsilon_S$: they are three points of a ROC curve, drawn by counting. Why must the
window be chosen on val and not on test?

**W2.1.2** *Core, 90 min.* A cut-based W and top tagger. The goal is a ROC curve from a cut tagger. Keep the mass
window fixed and vary a cut on a second variable; the hint shows how to turn the two into one score. Inside the W
window, scan a cut on $\tau_{21}$. Inside a top
window that you choose from your histograms on the val part, scan a cut on $\tau_{32}$. Draw the ROC curves (W against
QCD, top against QCD). Give the AUC and $R_{50}$ with bootstrap errors on the test part (`itj.metrics.roc`, `auc`,
`rejection`, `bootstrap`; seconds of CPU). Write the first rows of `results_table.csv` with `itj.metrics.append_result`.

**W2.1.3** *Stretch, 60 min.* Better cuts. Optimise both windows on the val part for the best $R_{50}$ (say what you
maximise). *Predict first:* is $D_2$ or $\tau_{21}$ the better second variable for W jets? Then check, with
`itj.observables.compute(data, ["d2"])` (about 10 s) or your **W1.4.5**.

**Deliverables:** `W2.1.1_three_windows.pdf` and one sentence on val and test (**W2.1.1**); `W2.1.2_roc_cuts.pdf` and
the first rows of the table (**W2.1.2**); the optimised rows (**W2.1.3**).

**Hints**
- **W2.1.1:** choose on val, report on test once. That rule starts here and holds for the whole course.
- **W2.1.2:** to use the cuts as a score, give a jet outside the window the lowest score and a jet inside it the score
  $1 - \tau_{21}$ (notebook 00_quickstart). Before you compute the ROC curve, keep only the two classes of a comparison
  (W and QCD, or top and QCD). `bootstrap` takes the metric as its first argument: `auc_mean, auc_err =
  itj.metrics.bootstrap(itj.metrics.auc, y, s)`, and `itj.metrics.rejection(y, s, 0.5)` gives $R_{50}$. The top $R_{50}$
  rests on few QCD jets: give the rejection at $\epsilon_S = 0.8$ as well, and say how many QCD jets pass.

### W2.2 Boosted decision trees (core path about 1.25 h, stretch about 1.5 h)

**Goal:** a boosted decision tree (BDT), a standard machine-learning tagger, on the 12 jet-level numbers. A BDT adds up
many small trees, each correcting the errors of the sum before it. You build it up from one tree, which is nothing but a
set of cuts.

**W2.2.1** *Warm-up, 30 min.* One tree is a set of cuts. Read [ml_primer section
3](docs/ml_primer.md#decision-trees-and-boosting). You use scikit-learn (sklearn), a Python library of standard
machine-learning tools that is installed with the course package. Train a decision tree of depth 2
(`sklearn.tree.DecisionTreeClassifier(max_depth=2)`) on $m_\mathrm{SD}$ and $\tau_{21}$ of the W and QCD jets of the
train part, and draw it (`sklearn.tree.plot_tree`; a second). Read off its cuts: how do they compare with the window and
the $\tau_{21}$ cut of **W2.1.2**? What would a deeper tree, or many trees, add?

**W2.2.2** *Core, 45 min.* A BDT on twelve numbers. Train scikit-learn's `HistGradientBoostingClassifier` on
`itj.features.jet_features` of the train part (a fit takes 1-3 s). These are the 10 JetClass `jet_*` branches plus
$\tau_{21}$ and $\tau_{32}$. Give the one-vs-one AUC and $R_{50}$ with errors on the test part. Add the ROC curves to
your **W2.1.2** figure and the rows to the table.

```python
import itj.features
import itj.metrics
from itj.io import load

train, val, test = itj.features.split(load("data/t2_detector.parquet"))
X, names = itj.features.jet_features(train)  # the array and the column names
y = train.labels()  # 0 QCD, 1 W, 2 top
```

**W2.2.3** *Stretch, 45 min.* Which inputs matter? Measure the permutation importance of each input on the val part
(`sklearn.inspection.permutation_importance`): the loss of performance when that one input is shuffled across the jets.
Should `jet_phi` (the azimuth of the jet, its angle around the beam) help? Check whether it does, and say why it is
among the inputs at all.

**W2.2.4** *Stretch, 45 min.* $D_2$ and the multiplicity. Add $D_2$ as a 13th input: does the AUC change beyond its
error? Does $N$ help beyond the other observables? Why could that be dangerous on real data? (**W3.3** tests how $N$
reacts to a soft particle; **W4.2.4** shows what happens on JetClass.) Retrain without it to check (about 95 s for all
fits).

**Deliverables:** `W2.2.1_tree.pdf` and two sentences (**W2.2.1**); `W2.2.2_roc_bdt.pdf` and the rows (**W2.2.2**);
`W2.2.3_importance.pdf` (**W2.2.3**); the rows with and without $D_2$ and $N$ (**W2.2.4**).

**Hints**
- **W2.2.1:** the tree is fitted on train and your window was chosen on val, so compare the cut values, not the scores.
  A tree of depth 2 may spend both of its levels on $m_\mathrm{SD}$: why could that be the best use of two cuts? Try
  `max_depth=3` and see whether, and where, a $\tau_{21}$ cut appears.
- **W2.2.2:** `predict_proba` gives three columns (QCD, W, top). `itj.metrics.one_vs_one` turns them into the score of W
  against QCD (and of top against QCD) on the jets of those two classes.
- **W2.2.3:** importance on the train part measures memorisation. A large importance does not mean "needed": correlated
  inputs share it.

### W2.3 A neural network from zero (core path about 3.75 h, stretch about 0.75 h)

**Goal:** your first neural network and training loop in PyTorch, on the same
12 inputs as the BDT. You get there in three steps: the parts of a network by hand, then a network and its loop in the
notebook, then your own loop.

**W2.3.1** *Warm-up, 60 min.* Tensors and gradients. Read [ml_primer section 4](docs/ml_primer.md#neural-networks) up to
logistic regression, and [section 5](docs/ml_primer.md#pytorch-in-one-page). Work through part A of
[notebooks/02_first_network.ipynb](notebooks/02_first_network.ipynb) (its sections 1-3). It starts with the 12 jet
features of a small generated sample (6 000 jets, seconds of CPU). Next come tensors and autograd (PyTorch's automatic
gradients) on a two-line function. Last comes a logistic regression on $m_\mathrm{SD}$ and $\tau_{21}$, written by hand
with gradient descent (repeated small steps against the gradient of the loss), and its AUC. Then change the number of
gradient steps and the step size, and describe what happens to the loss.

**W2.3.2** *Warm-up, 75 min.* A network and its training loop. Read the rest of [ml_primer section
4](docs/ml_primer.md#neural-networks). Then work through part B of the notebook (its sections 4-8, under a minute of
CPU). It rewrites the logistic regression of part A as an `nn.Module` with `torch.optim`, and builds an MLP with
`nn.Sequential`. It trains with a loop over mini-batches with AdamW (the optimiser of the course) and computes a
validation loss after each epoch. It ends with an overfitting demonstration: a large network on 500 jets, train against
val loss. Then multiply and divide the learning rate by 10 and describe what changes.

**W2.3.3** *Core, 90 min.* Your own training loop (write your own). Start from the loop of the notebook. Standardise the
12 inputs of **W2.2.2** with the statistics of the train part. Train an MLP with mini-batches and AdamW and validate
after each epoch. Stop early and restore the best state (the weights at the lowest val loss), then plot the train and
val loss (about 34 s of CPU). Compare the MLP with the BDT on the same inputs: which wins here, and by how much (with
errors)? From here on you may use `itj.train.fit` instead of your own loop.

**W2.3.4** *Stretch, 45 min.* Trees and networks on tables. Why can a BDT often beat an MLP on a few tabular features,
and what does an MLP need to match it? Try one idea (more epochs, a quantile transform of the inputs, a wider network).

**Deliverables:** `W2.3.1_logistic.pdf` (the loss against the step) and one sentence per notebook section on what you
changed (**W2.3.1**); `W2.3.2_loss.pdf` (the train and val loss at three learning rates) and one sentence on overfitting
(**W2.3.2**); `W2.3.3_loss.pdf`, the rows and the comparison (**W2.3.3**); the result of your idea (**W2.3.4**).

**Hints**
- **W2.3.1:** run the cells in order, and restart the kernel if a cell ran twice. A tensor is a NumPy array that
  remembers how it was computed, so that `.backward()` can return the gradient.
- **W2.3.2:** the notebook's loop is the starting point of **W2.3.3**: copy it, do not import it.
- **W2.3.3:** standardise with train statistics only. You may add dropout ([ml_primer section
  4](docs/ml_primer.md#neural-networks): a layer that switches off random units during training). Then compare train
  and val loss in the same mode (`model.eval()` for both); if you do not, say so. Dropout is on in training mode and
  makes the train loss look worse. An MLP far behind the BDT usually
  lacks standardisation, the best-state restore or enough epochs. `fit(..., max_minutes=15)` stops a run that would
  take too long, and `fit` prints an estimate of the time after 20 batches.

### W2.4 Networks that see particles: Deep Sets (core path about 2.5 h, stretch about 2.75 h)

**Goal:** taggers that read all the particles of a jet as a set, with no order, and why the order of the particles must
not matter.

**W2.4.1** *Warm-up, 45 min.* A network that does not care about the order. Read [ml_primer section
6](docs/ml_primer.md#networks-on-particles) up to graphs, and [models.md](docs/models.md#pfn-and-efn). Load the course
PFN and make one `JetBatch` (the course's batch of padded jets) of 64 test jets. Check numerically that its outputs do
not change when you shuffle the particles of every jet. Check also that they do not change when you pad the batch with
empty particles.

```python
import dataclasses

import torch

import itj.features
import itj.models
import itj.train
from itj.io import load

train, val, test = itj.features.split(load("data/t2_detector.parquet"))
model = itj.models.load_model("models/course/pfn.pt")
batch = itj.train.make_batch(test.head(64))  # make_batch(..., length=...) pads to a given length
with torch.no_grad():
    out = model(batch)
perm = torch.randperm(batch.mask.shape[-1])  # one new order of the particle axis (the last one)
moved = {k: getattr(batch, k)[..., perm] for k in ("features", "points", "vectors", "mask")}
shuffled = dataclasses.replace(batch, **moved)  # the same jets with their particles in another order
```

**W2.4.2** *Core, 105 min.* Your own PFN (write your own). Write a PFN, $F\bigl(\sum_i \Phi(x_i)\bigr)$, on `JetBatch`
([models.md](docs/models.md#the-model-api)). The inner network $\Phi$ acts on each particle and the outer network $F$ on
the sum. Apply $\Phi$ to the `kin` features of each particle and add the results with a masked sum, a sum over the real
particles only. The seven `kin` features are listed in [models.md](docs/models.md#inputs). Check that your PFN is
invariant under permutations and under extra padding (as in **W2.4.1**). Next, check that your PFN is the same network
as the course PFN. A key map is a dictionary from the parameter names of the course PFN to yours. Copy the weights of
the course PFN into your PFN with a key map of a few lines (`model.state_dict()`), and check that the outputs agree on
one batch.

**W2.4.3** *Stretch, 45 min.* Train your own PFN. Train your PFN on the 105 000 training jets for 10 epochs (2-3 minutes
on an idle CPU). Copy the trained weights back into `itj.models.build_model("pfn")` (the same key map,
`load_state_dict`) and save it with `itj.models.save_model(model, "models/toy/pfn.pt")`. This step is needed because
`predict` without `feature_set=`, `save_model` and `itj.probes.extract` work only on the `itj` models. Compare its ROC
curve with that of the course PFN. Later weeks may use it in place of `models/course/pfn.pt`: say which.

**W2.4.4** *Stretch, 2 h.* Your own EFN and collinear safety. Write the EFN, $F\bigl(\sum_i z_i\, \Phi(\Delta\eta_i,
\Delta\phi_i)\bigr)$. A collinear-safe network gives the same output when one particle is split into two parallel
particles that share its momentum. Check that your EFN is exactly collinear safe with
`itj.augment.split_collinear(delta=0)` on **one** particle per jet, which is the default (`which="hardest"`). Splitting
every particle pushes jets beyond the truncation at 128 particles, the most that a network reads per jet. Train the
EFN, copy the weights into `build_model("efn")` and save `models/toy/efn.pt`. What does the PFN see that the EFN cannot?

**Deliverables:** the two checks, as the largest absolute difference of the outputs (**W2.4.1**); the three checks of
your PFN (**W2.4.2**); `models/toy/pfn.pt` and the ROC comparison (**W2.4.3**); the collinear check and
`models/toy/efn.pt` (**W2.4.4**).

**Hints**
- **W2.4.1:** the tensors of a `JetBatch` are channel-first: `features` has the shape (jets, channels, particles), where
  a channel is one feature of a particle. The particles are on the last axis. Shuffle that axis, with the same
  permutation for `features`, `points`, `vectors` and `mask`. Shuffling axis 1 mixes the features and changes the
  outputs: that is not a failure of the invariance. The course PFN is the first model of the week-2 start-of-week box
  and is ready after a few minutes.
- **W2.4.2:** `print(model)` shows the layers and their activations, which the `state_dict` does not. Copy that layout
  so that the keys map one to one. `nn.Linear` acts on the last axis, so transpose `features` to (jets, particles,
  channels) before $\Phi$. A mask that lets padded slots into the sum fails the padding check. `itj.models.save_model`
  saves a model with its configuration; `load_model` reads it back.
- **W2.4.3**, **W2.4.4:** while the week-2 start-of-week box still runs, give `fit` two threads (`threads=2`) and expect
  a longer run.

### W2.5 Graphs and attention (core path about 1.75 h, stretch about 5.5 h)

**Goal:** the network designs of the best published taggers, trained on a CPU, and the ranking of all your taggers.

**W2.5.1** *Warm-up, 45 min.* Read and predict. Read the rest of [ml_primer section
6](docs/ml_primer.md#networks-on-particles): graphs; attention as a weighted average; heads, the class token, the pair
bias $U$. Then read [models.md](docs/models.md#particlenet) on ParticleNet, a graph network, and on the [Particle
Transformer](docs/models.md#particle-transformer) (ParT). ParT-small (0.24 M parameters) is the course's small
transformer. The preset `partlite` is a larger one, the ParT-lite of the dl4hep notebooks (the notebooks this course
grew from; [ml_primer: where to read more](docs/ml_primer.md#where-to-read-more)). *Predict first:* the ranking of all
your taggers, and what the pair bias buys. Check that `models/course/` holds the five models (`ls models/course`).

**W2.5.2** *Core, 60 min.* All taggers in one figure, and what they cost. Draw the ROC curves of all your taggers in one
figure, with two panels: W against QCD and top against QCD. The taggers are the cuts, the BDT, the MLP and the five
course models (PFN, EFN, PN-lite, ParT-small, ParT-small without $U$), plus your own PFN if you trained it. Make a table
with the number of parameters (`model.n_parameters()`) and the training time of each (`models/course/*_history.json`).
The models saw different training sizes (105 000, 60 000, 30 000 jets) and different CPU times. Say what the ranking can
and cannot tell, and compare it with your prediction. Running the models with `itj.train.predict` takes about 200 s;
`itj.plot.roc_curves` draws the figure.

**W2.5.3** *Stretch, 2 h.* How much data? Draw the learning curve: the AUC (W against QCD) against the training size
(1 000, 3 000, 10 000, 30 000, 105 000 jets) for the BDT and the PFN. Each point should be trained until its validation
loss stops improving. Give every point at least 2000 gradient steps, and validate every size on at least 5000 jets.
Otherwise the small points measure under-training or a noisy choice of epoch, not the size of the data. The settings are
`fit(..., min_steps=2000, val_every=50, patience=5)`. `min_steps` raises the number of epochs of a small training size.
`patience` counts validations, and an epoch of 1 000 jets is only 4 steps of 256 jets. `val_every=50` therefore
validates every 50 steps instead of after every epoch, and the run stops after 250 steps without improvement, keeping
the best state. The curve takes about 10 minutes of CPU: run it in the background while you write up **W2.5.2**. This
curve returns in **W3.4.7**.

**W2.5.4** *Stretch, 60 min.* Seeds, the transformer and the time axis. For the curves of **W2.5.3**, add three seeds at
the sizes below 10 000 jets, and ParT-small at 30 000 (the course model). Draw the curves against the training time as
well. Do they cross, against the training size and against the training time, and if so where? Then measure the
inference speed of every tagger (how fast a trained model scores new jets) in jets per second, with `torch.no_grad()`,
on one thread and on all threads. Add it as a column of the **W2.5.2** cost table.

**W2.5.5** *Stretch, 30 min + GPU time.* Larger transformers. *Optional (GPU):* `partlite`, or more training data, on a
GPU ([examples/cluster](examples/cluster/README.md)).

**W2.5.6** *Stretch, 2 h.* Your own question. For example: a quark/gluon tagger on `data/t1_qg_detector.parquet` (label
set `"qg"`), compared with a cut on the multiplicity and with the EFN. Or the `kinpid` features: the kinematic features
plus the charge and type of each particle. Or training at `hadrons` and testing at `detector`. Or training on a flat
$p_T$ spectrum, with jets spread evenly over the momentum range (`--pt-spectrum flat`).

**Deliverables:** your prediction, in your answers before **W2.5.2** (**W2.5.1**); `W2.5.2_roc_all.pdf` and the cost
table (**W2.5.2**); `W2.5.3_learning_curve.pdf` (**W2.5.3**); the speeds and the crossing (**W2.5.4**); a figure of your
own (**W2.5.6**).

**Hints**
- **W2.5.1:** if a course model is missing, run the command of the start-of-week box again; `fit` prints a time estimate
  after 20 batches.
- **W2.5.2:** predict with gradients off (`torch.no_grad()`). The course models were trained for a few minutes of CPU
  each; the transformers got the most CPU and the least data.
- **W2.5.3:** train on the train part, choose on the val part and report on the test part once.

#### Weekly summary

The ranking of your taggers with errors and costs, in half a page. It answers the thread question *Which tagger wins, by
how much and at what price?* (0.5 h with the hand-in). Hand in `answers/week2.md` with the summary and the answers to
every warm-up and core ID. Hand in as well the figures `W2.1.1_three_windows.pdf`, `W2.1.2_roc_cuts.pdf`,
`W2.2.1_tree.pdf`, `W2.2.2_roc_bdt.pdf`, `W2.3.1_logistic.pdf`, `W2.3.2_loss.pdf`, `W2.3.3_loss.pdf` and
`W2.5.2_roc_all.pdf` with the cost table, and `results_table.csv` (cuts, BDT, MLP). In the discussion (0.5 h) we go
through the core IDs.

---
## Week 3: What do taggers learn? (core path about 13.5 h, stretch about 26.75 h)

**Why this week.** A ROC curve ([ml_primer](docs/ml_primer.md#a-tagger-is-a-function)) says how well a tagger
separates the classes, not what it has learned. A tagger's representation is the set of numbers the network computes
inside to describe a jet. On the toy we know the truth of every jet. This lets us test what the representation holds,
what the tagger uses, which changes of a jet it should ignore and whether a representation can be learned without
labels.

**Thread.** Last week gave trained taggers and their ranking. This week looks inside them, with controls that keep the
claims honest. Then it turns to the physics of what should *not* matter (soft and collinear particles, rotations), and
to a representation trained only to ignore those changes. Next week takes the same questions to real jets and asks
*where* in a jet a tagger looks.

**Concepts this week**, in order of use: embedding and PCA ([ml_primer 7](docs/ml_primer.md#embeddings-and-maps));
linear and logistic regression, $R^2$ and AUC as probe scores, cross-validation folds, probes, control tasks,
random-weight baselines and selectivity ([ml_primer 8](docs/ml_primer.md#regression-and-probes)); counterfactual samples
and paired errors; the filters of a network; soft and collinear perturbations and IRC safety
([primer 12](docs/primer.md#a-first-look-at-the-observables)); augmentations, exact and approximate symmetries; cosine
similarity, softmax with temperature and the NT-Xent loss; projection head, pretraining, alignment and uniformity;
linear evaluation ([ml_primer 9](docs/ml_primer.md#learning-without-labels)). Stretch adds t-SNE and UMAP, the
quark/gluon probe, CKA, mass sculpting and the Jensen-Shannon divergence, planing weights, Sudakov safety, fine-tuning
and label budgets (each in the [glossary](docs/glossary.md)).

**This week at a glance** (times in minutes, or hours where marked):

| task | warm-up | core | stretch |
|---|---|---|---|
| start of week | box (15) | | |
| **W3.1** Inside the representation | **W3.1.1** 60 | **W3.1.2** 105 | **W3.1.3** 90, **W3.1.4** 90, **W3.1.5** 105, **W3.1.6** 90 |
| **W3.2** What do the taggers use? | **W3.2.1** 45 | **W3.2.2** 60 | **W3.2.3** 45, **W3.2.4** 2.25 h, **W3.2.5** 45, **W3.2.6** 2 h, **W3.2.7** 30, **W3.2.8** 2 h |
| **W3.3** Safe observables and symmetries | **W3.3.1** 30 | **W3.3.2** 90, **W3.3.3** 60 | **W3.3.4** 75, **W3.3.5** 45, **W3.3.6** 60, **W3.3.7** 30, **W3.3.8** 2 h |
| **W3.4** Learning without labels | **W3.4.1** 45 | **W3.4.2** 90, **W3.4.3** 105, **W3.4.4** 45 | **W3.4.5** 75, **W3.4.6** 2 h, **W3.4.7** 3.5 h |
| summary, hand-in and discussion | | 1 h | |

**Start of week** (15 minutes; the downloads run in the background all week):

```bash
python examples/make_task_data.py --week 3
python examples/make_task_data.py --week 4 --only fetch
```

The first line makes the seven `t3_*` files and `t6_pretrain` of [Data for week 3](#data-for-week-3) (160 000 jets, 0.39
GB, about a minute with 8 workers). The second line starts the downloads of week 4: three JetClass validation files of
374 MB, and the pretrained models of 19 MB and 136 MB for the bonus **B.4**. `jetgen fetch` resumes an interrupted
download. It checks the size and the MD5 checksum (a fingerprint that changes if a file is damaged) of every file, and
it keeps the files that are already complete ([jetclass.md](docs/jetclass.md#download)).

**Day plan.** Day 1 (about 6.75 h): the start-of-week box (data and downloads), **W3.1.1**-**W3.1.2**,
then **W3.2.1**-**W3.2.2** and **W3.3.1**-**W3.3.2**. Day 2 (about 6.75 h): **W3.3.3**, **W3.4.1**-**W3.4.4**, the
weekly summary and the discussion. Day 3 (optional): the first stretch picks, catch-up. Three shorter days instead:
box, **W3.1**, **W3.2** (4.75 h) / **W3.3**, **W3.4.1** (3.75 h) / **W3.4.2**-**W3.4.4**, summary, discussion (5 h).

**Check-in at the end of day 1:** the downloads for week 4 are complete, and you can extract an embedding
(**W3.1.1**).

**If you are behind** (agreed in advance; each saves the time given):
- *Two-day:* **W3.1.2** with two models (PFN, ParT-small) and their controls (0.5 h; the comparison across four models
  is lost).
- **W3.3.2** with three observables ($m$, $m_\mathrm{SD}$, $N$) (0.25 h; $\tau_2$ and $p_T^D$ are lost).
- **W3.3.3** with `rotate` and `split_collinear` only (0.25 h; the soft smearing is lost).
- *Two-day:* **W3.4.3** with the loss and the final linear evaluation only (0.25 h; alignment and uniformity are lost).
- **W3.4.4** with the self-supervised (SSL) encoder against the random encoder only (0.25 h; the comparison with the
  supervised PFN is lost).

**First stretch picks:** **W3.2.3** (sculpting), **W3.1.4** (the quark/gluon probe), **W3.1.5** (your own
CKA), **W3.4.7** (few labels, in the background).

**Models and data of this week.** The models are the course models of `models/course/` (or your own PFN
of **W2.4.3**: say which). The control is an untrained ParT-small with random weights
(`itj.models.build_model("partsmall")` after `reset_parameters()`). The data are the test part of
`data/t2_detector.parquet` (30 000 jets) unless stated otherwise.

### Data for week 3

`python examples/make_task_data.py --week 3` runs these commands (160 000 jets, 0.39 GB, about a minute with 8 workers).
The `t3_w80_*` files are paired with `t3_w80` (the same hard processes, jet by jet) as described under
[Data for week 1](#data-for-week-1).

```bash
# W3.2.5: W jets at four masses (independent samples: the mass changes the hard process)
jetgen generate --process wqq --w-mass 60 -n 10000 --seed 35 -o data/t3_w60.parquet
jetgen generate --process wqq -n 10000 --seed 35 -o data/t3_w80.parquet
jetgen generate --process wqq --w-mass 100 -n 10000 --seed 35 -o data/t3_w100.parquet
jetgen generate --process wqq --w-mass 120 -n 10000 --seed 35 -o data/t3_w120.parquet
# W3.2.1, W3.2.6: the jets of data/t3_w80.parquet with one thing changed (paired samples)
jetgen generate --process wqq --no-coherence -n 10000 --seed 35 -o data/t3_w80_nocoh.parquet
jetgen generate --process wqq --no-ue -n 10000 --seed 35 -o data/t3_w80_noue.parquet
jetgen generate --process wqq --cell-position centre -n 10000 --seed 35 -o data/t3_w80_centre.parquet
# W3.4: an unlabelled pretraining set (its labels are used only for the evaluation)
jetgen generate -n 90000 --seed 61 -o data/t6_pretrain.parquet
```

### W3.1 Inside the representation (core path about 2.75 h, stretch about 6.25 h)

**Goal:** what the embedding of a tagger contains, measured with probes and their controls. The embedding is the vector
of numbers the network computes for a jet just before its last layer. A probe is a small model that tries to read one
quantity from the embedding.

**W3.1.1** *Warm-up, 60 min.* The embedding of one tagger. Read [ml_primer section
7](docs/ml_primer.md#embeddings-and-maps). Extract the embedding of the course PFN on the test part of
`data/t2_detector.parquet`. Project it on its first two principal components (the two directions in which the jets
differ most; `sklearn.decomposition.PCA`), coloured by the class and by $m_\mathrm{SD}$. Which property of the jets does
the largest direction of variation follow?

```python
import itj.features
import itj.models
import itj.probes
from itj.io import load

train, val, test = itj.features.split(load("data/t2_detector.parquet"))
model = itj.models.load_model("models/course/pfn.pt")
h = itj.probes.extract(model, test)["embedding"]  # one row per jet
```

**W3.1.2** *Core, 105 min.* Probes with controls. Read [ml_primer section 8](docs/ml_primer.md#regression-and-probes)
and [methods.md](docs/methods.md#probes). (a) One probe first. Fit a linear probe (`itj.probes.fit_probe`) from the PFN
embedding to $m_\mathrm{SD}$. Fit the same probe from the embedding of the untrained ParT-small, and once more with a
shuffled target (`itj.probes.control_task(y, kind="shuffle")`). Then compute the selectivity (`itj.probes.selectivity`:
the score of the trained model minus that of the control). Why are the two controls part of the answer? (b) Then the
map. Fit linear probes to $m_\mathrm{SD}$ and to the multiplicity $N$ from the embeddings of four models: the PFN, the
EFN, ParT-small and the untrained ParT-small. The EFN is a PFN that sees only the angles and
momentum fractions of the particles ([models.md](docs/models.md#pfn-and-efn)). Give each probe its random-weight and
shuffled controls. *Predict first:* which target does ParT-small encode best? The extraction takes about 2 minutes for
the four models, the linear probes under a minute.

**W3.1.3** *Stretch, 90 min.* Maps of the embedding. Draw PCA, t-SNE and UMAP maps of the four embeddings. Unlike PCA,
t-SNE and UMAP are non-linear: they keep neighbouring jets together. Colour the maps by the class, $m_\mathrm{SD}$,
`truth_init_pid` (QCD jets only), `truth_n_primary` and $N$: what organises the space? UMAP needs `pip install -e
".[ml]"` (on Python 3.14, where umap may not install, use t-SNE). About 3 minutes of CPU for the four models.

**W3.1.4** *Stretch, 90 min.* More probes. Repeat the probes of **W3.1.2** for gluon against quark
(`truth_init_pid == 21`, on QCD jets only; it is never a training label). [Primer section
7](docs/primer.md#the-parton-shower) and **W1.2.5** say why the two differ. Do the same for $\tau_{21}$, $D_2$ and
`truth_n_primary`. Then extend the map: an MLP probe (a small non-linear network; `kind="mlp"`, about 4 minutes), the
PN-lite model, the targets `truth_mass` and `truth_dr_min`, and probes after every block (`extract(..., layers="all")`).

**W3.1.5** *Stretch, 105 min.* Your own linear CKA (write your own). CKA (centred kernel alignment) is one number that
says how alike two representations of the same jets are ([methods.md](docs/methods.md#linear-cka)). Check yours against
`itj.probes.linear_cka`. Check that it is invariant under orthogonal maps (rotations and reflections) and isotropic
scaling (multiplying every entry by the same factor), but not under general linear maps. Test the last point with a
whitening map $X\Sigma^{-1/2}$: a random linear map hardly shows it.

**W3.1.6** *Stretch, 90 min.* Which model is most like the observables? Compute the CKA (a) between the blocks of
ParT-small and (b) between each model and the observables
$X_\mathrm{phys} = [m_\mathrm{SD}, \tau_{21}, \tau_{32}, D_2, N, \ln p_T]$. Standardise each column first: in GeV the
mass column would dominate the Gram matrices (the jet-by-jet tables of dot products that CKA compares). Which model is
most "observable-like"? *Optional:* a second EFN seed (about 2 minutes); a second ParT-small (about 12 minutes: start it
in the background first, then read); the four models against each other.

**Deliverables:** `W3.1.1_pca.pdf` (**W3.1.1**); `W3.1.2_probes.pdf` (model against target) and the table of probe
scores with their spread over the folds (**W3.1.2**); `W3.1.3_embeddings.pdf` (**W3.1.3**); the extended table
(**W3.1.4**); the `allclose` line and the invariance checks (**W3.1.5**); `W3.1.6_cka.pdf` (**W3.1.6**).

**Hints**
- **W3.1.1:** [methods.md](docs/methods.md#looking-at-embeddings); an embedding is the input of the last linear layer
  (`model.embed`). `extract` keeps the memory use low (it never computes attention maps).
- **W3.1.2:** a probe score means little on its own: the random model and the control are part of the answer.
  `fit_probe` standardises the inputs and returns $R^2$ (regression) or AUC (classification) per fold. `control_task(y,
  kind="shuffle")` keeps the marginal distribution of the target: the same values, given to the wrong jets.
- **W3.1.3:** distances between t-SNE clusters mean little.
- **W3.1.5**, **W3.1.6:** the CKA invariances are explained in [methods.md](docs/methods.md#linear-cka).

### W3.2 What do the taggers use? (core path about 1.75 h, stretch about 8.25 h)

**Goal:** what a tagger uses, found by changing one thing in the jets and by looking at its filters. Stretch: what a cut
on the tagger does to the shape of the background mass distribution (sculpting), and a tagger that avoids it.

**W3.2.1** *Warm-up, 45 min.* Does the tagger use the underlying event
([primer](docs/primer.md#hadronisation-and-the-underlying-event))? Evaluate the course PFN
(`itj.train.predict`) on `data/t3_w80.parquet` and on the same jets without the underlying event
(`data/t3_w80_noue.parquet`). Plot the change of the W score jet by jet. Is the mean change significant? Pair the jets
by `truth_event_id`: check the IDs, and do not assume that the rows are in the same order.

**W3.2.2** *Core, 60 min.* The filters of an EFN. Each filter is one output of the function that the EFN applies to a
particle, drawn as a picture over the jet ([models.md](docs/models.md#pfn-and-efn)). Train
`itj.models.build_model("efn", phi_sizes=(100, 100, 8))` with `itj.train.fit` as for the course models (10 epochs, about
1.5 minutes). Draw its 8 filters $\Phi_a(\Delta\eta, \Delta\phi)$ with `itj.plot.efn_filters`. Some filters can die in
training: their ReLU unit (which outputs zero for a negative input) is 0 everywhere, and the plot marks them "dead".
Report these filters as a result. Then relate the size of the structures you see to the prong separation of
your **W1.1.3** figure. Evaluate the separation at the median $p_T$ of the training jets as well as at 600 GeV (as in
Fig. 6 of arXiv:1810.05165).

**W3.2.3** *Stretch, 45 min.* Does a cut on the tagger sculpt the mass? Cut the W-against-QCD score of the PFN at
$\epsilon_S = 50\,\%$ (`itj.metrics.threshold_at`). Compare the $m_\mathrm{SD}$ distribution of the QCD jets that pass
with that of those that fail. Quantify the difference with `itj.metrics.sculpting_jsd`, the Jensen-Shannon divergence
(a measure of how far apart two distributions are) between pass and fail. Why would an analyst who fits a W peak on top
of the QCD background mind?

**W3.2.4** *Stretch, 2.25 h.* A decorrelated tagger (one whose cut leaves the background mass shape alone). Retrain the
PFN with planing weights: weights per jet that make $m_\mathrm{SD}$ flat in each class. Write the weights yourself. Use
20 bins in [40, 250] GeV. A jet in bin $b$ of its class gets $w = (n_\mathrm{in}/20)/n_b$, with $n_\mathrm{in}$ jets of
the class in the range and $n_b$ in the bin. Every filled bin then carries the same total weight. A jet outside the
range gets $w = 1$. Then normalise the weights per class to the number of jets of the class, cap them at 10 and pass
them with `JetDataset(weights=...)`. Plot $R_{50}$ against $1/\mathrm{JSD}$ for both PFNs (2 trainings of about 2-3
minutes). Expect only a partial decorrelation (20 bins with a cap of 10 flatten the W peak only in part), and say what
that does to your answer.

**W3.2.5** *Stretch, 45 min.* Two-prong jets at other masses. Evaluate both PFNs of **W3.2.4** on `data/t3_w60.parquet`,
`data/t3_w80.parquet`, `data/t3_w100.parquet` and `data/t3_w120.parquet`: does the decorrelated tagger still tag these
jets? *Optional:* DisCo, which adds a penalty on the correlation between score and mass to the loss (arXiv:2001.05310).

**W3.2.6** *Stretch, 2 h.* Counterfactual jets (the same jets with one thing changed). Evaluate all your week-2 taggers
on `data/t3_w80.parquet` and on the paired `data/t3_w80_noue.parquet`, `data/t3_w80_centre.parquet` and
`data/t3_w80_nocoh.parquet`. Each file has the same hard processes with one switch changed. Without the underlying
event, and with each calorimeter cell's particle placed at the cell's centre, every track from the jet is identical.
Without coherence, the shower itself changes. Plot the change of the score jet by jet for `noue` and `centre`. `nocoh`
shares only the hard process with `t3_w80` (its whole shower is drawn again), so compare its score distribution and mean
with those of `t3_w80` instead. Which change matters most, and which of the **W3.1.2** probe targets change with it?
What does this say about what the taggers use: the underlying event, the cell grid or the radiation pattern of a colour
singlet? A colour singlet is a colour-neutral source such as the W
([physics.md](docs/physics.md#start-angles-and-colour-coherence)).

**W3.2.7** *Stretch, 30 min.* A second seed and the paper. Train the EFN of **W3.2.2** with another seed (a different
random start can give a different set of filters). Compare both sets with Fig. 6 of arXiv:1810.05165.

**W3.2.8** *Stretch, 2 h.* Your own question. For example: probes on the outputs of the attention blocks, CKA across
training sizes, probes for `truth_part_from_b`.

**Deliverables:** `W3.2.1_noue.pdf` and the mean change with its paired error (**W3.2.1**); `W3.2.2_filters.pdf`
(**W3.2.2**); `W3.2.3_sculpting.pdf` (**W3.2.3**); `W3.2.4_decorrelation.pdf` (**W3.2.4**); `W3.2.5_mass_scan.pdf`
(**W3.2.5**); `W3.2.6_counterfactual.pdf` (**W3.2.6**); a figure of your own (**W3.2.8**).

**Hints**
- **W3.2.1:** [physics.md](docs/physics.md#reproducibility) says what the paired files share. The error of a mean of
  differences is not the error of a difference of means.
- **W3.2.2:** the toy jets are not rotated to a common orientation. Before you look at the filters, think about which
  shapes a filter needs to respond to a prong. Also ask how the answer would change if every jet were rotated so that
  its prongs lined up.
- **W3.2.3:** few QCD jets pass the cut, which sets a floor to the divergence. Compare with a random cut that keeps as
  many jets. [methods.md](docs/methods.md#mass-sculpting-and-decorrelation) explains the planing weights of **W3.2.4**
  as well.

### W3.3 Safe observables and symmetries (core path about 3 h, stretch about 5.5 h)

**Goal:** which observables a soft or collinear particle cannot change, and which changes of a jet a representation
should forget. A soft particle carries a tiny fraction of the jet's momentum. A collinear split replaces one particle by
two that move in almost the same direction.

**W3.3.1** *Warm-up, 30 min.* One soft particle. Read [primer section
12](docs/primer.md#a-first-look-at-the-observables) again. Add one soft particle to each of 1 000 jets of
`data/t1_detector.parquet`, with a fraction $\epsilon = 10^{-1}, \ldots, 10^{-6}$ of the jet $p_T$. Plot the mean
relative change of the mass and of the multiplicity against $\epsilon$ on log axes. Which one goes to zero?

```python
import itj.augment
import itj.observables
from itj.io import load

jets = load("data/t1_detector.parquet").head(1000)
soft = itj.augment.add_soft(jets, eps=1e-3, substructure=True)  # new part_* columns
values = itj.observables.compute(soft, ["mass", "multiplicity"])
```

**W3.3.2** *Core, 90 min.* Which observables are IRC safe? *Predict first:* which of $m$, $m_\mathrm{SD}$, $\tau_2$, $N$
and $p_T^D = \sqrt{\sum_i z_i^2}$ (the momentum fractions $z_i$ of the particles squared and summed;
[observables.md](docs/observables.md#angularities-and-pt_d)) are IRC safe? Then perturb the 1 000 jets in two ways,
both with `substructure=True`: with soft particles (`add_soft`), and with collinear splits
(`itj.augment.split_collinear`, momentum share $\lambda = 0.5$ and 0.2, separation $\delta = 10^{-1}, \ldots, 10^{-5}$).
Plot the mean relative change of the five observables (`compute(data, ["mass", "sdmass", "tau2", "multiplicity",
"ptd"])`, about 25 s) against $\epsilon$ and $\delta$, and classify each as safe or unsafe. Then check your
classification against [observables.md](docs/observables.md#irc-safety).

**W3.3.3** *Core, 60 min.* Augmentations are symmetries. An augmentation is a change of a jet that should not change
what the jet is ([ml_primer](docs/ml_primer.md#learning-without-labels)). Apply `itj.augment.rotate`, `smear_soft` and
`split_collinear` (with `substructure=True`; about 30 s) to 2 000 jets, and tabulate the change of $m_\mathrm{SD}$,
$\tau_{21}$, $N$ and $p_T^D$. Which are symmetries of the physics, which only of IRC-safe observables and which only
approximate? What should a representation trained to ignore them forget?

**W3.3.4** *Stretch, 75 min.* All nine observables and Sudakov safety. Run the scans of **W3.3.2** for $m$,
$m_\mathrm{SD}$, $\tau_2$, $\tau_{21}$, $e_2$, $D_2$, $N$, $p_T^D$ and the leading fraction (the momentum fraction of
the hardest particle). Then sort them into the classes safe, Sudakov safe (a weaker property than IRC safety) or unsafe
([observables.md](docs/observables.md#irc-safety)). Why can a scan on single jets with many particles not tell an
IRC-safe observable from a Sudakov-safe one? (Hint: which limit does IRC safety concern, and which one does the scan
take?) These curves are the template of **W4.3**.

**W3.3.5** *Stretch, 45 min.* The corner cases. Look at a two-particle jet, where $\tau_{21} = 0/0$. Then scan
$m_\mathrm{SD}$ at $\lambda = 0.05$, below the soft-drop threshold $z_\mathrm{cut} = 0.1$ (see the pitfalls of
[observables.md](docs/observables.md#numerical-irc-tests)), and find the jets that change.

**W3.3.6** *Stretch, 60 min.* Your own rotation (write your own). Rotate every jet about its $p_T$-weighted axis and
compare with `itj.augment.rotate`.

**W3.3.7** *Stretch, 30 min.* More augmentations. Add `itj.augment.translate`, `drop_soft` and $D_2$ to the table
of **W3.3.3**.

**W3.3.8** *Stretch, 2 h.* Is $N$ more than the safe observables? For $N$, add a baseline to the probes of **W3.1.2**: a
probe from the observables $X_\mathrm{IRC} = [m, m_\mathrm{SD}, \tau_1, \tau_2, \tau_3, e_2, e_3, \text{width}]$
(compare this list with your classes of **W3.3.4**). Is $N$ more predictable from the embeddings than from these
observables alone? *Predict first:* does the EFN beat that baseline? In your answer, say that the baseline is linear and
that its $m_\mathrm{SD}$ column makes an $m_\mathrm{SD}$ probe trivial; the MLP probe is the fairer comparison.

**Deliverables:** `W3.3.1_soft.pdf` (**W3.3.1**); `W3.3.2_irc.pdf` and the classification (**W3.3.2**);
`W3.3.3_augmentations.pdf` or the table (**W3.3.3**); the nine curves and the classes (**W3.3.4**); the jets that change
(**W3.3.5**); the `allclose` line (**W3.3.6**); the baseline next to the probes (**W3.3.8**).

**Hints**
- **W3.3.1:** [primer section 12](docs/primer.md#a-first-look-at-the-observables) says what IRC safety means; keep
  [observables.md](docs/observables.md#irc-safety) for after your classification of **W3.3.2**. `itj.augment` returns
  new `part_*` columns, so call the observables again on the result.
- **W3.3.2:** about 0.2 % of the jets have $m_\mathrm{SD} = 0$, which makes the mean of $\lvert\Delta O\rvert/O$
  infinite. Leave out the jets with $O = 0$ before the perturbation, or plot the median of $\lvert\Delta O\rvert/O$ as
  well. Without `substructure=True` the stored $m_\mathrm{SD}$ and $\tau_N$ are not recomputed and look perfectly safe.
  `split_collinear` returns its pieces in double precision (float64). `float64=False` stores them as float32, rounded
  piece by piece, which gives a spurious mass of about 0.1 GeV per piece of a hard particle.
- **W3.3.3:** [methods.md](docs/methods.md#contrastive-learning), the paragraph "Augmentations are symmetries";
  rotations are about the jet axis.
- **W3.3.6:** rotate about the axis of the summed four-momentum, not about (0, 0).

### W3.4 Learning without labels (core path about 4.75 h, stretch about 6.75 h)

**Goal:** a jet representation learned from unlabelled jets with the augmentations of **W3.3** (JetCLR,
arXiv:2108.04253), and what it knows. The method is contrastive learning: the network learns to place two augmented
copies of the same jet close together and different jets far apart.

The encoder (the network that turns a jet into an embedding) is `itj.models.build_model("pfn")`. On top of it sits a
projection head (a small network whose output enters the loss), `itj.ssl.ProjectionHead(128, 128, 64)` (the fast
default). The pretraining uses `data/t6_pretrain.parquet`. Do not look at its labels: drop the `label_`, `truth_` and
`aux_` columns first and do not print the file. The evaluation uses `data/t2_detector.parquet`.

**W3.4.1** *Warm-up, 45 min.* The contrastive loss on paper. Read [ml_primer section
9](docs/ml_primer.md#learning-without-labels). Work out on paper the NT-Xent loss (normalised temperature-scaled
cross-entropy), with cosine similarity and a softmax with temperature, for a batch of $B$ pairs in two cases. First, the
two views (augmented copies) of every jet are identical: what does the loss still depend on? Second, all embeddings are
equal: what value does it take? What does a small temperature do?

**W3.4.2** *Core, 90 min.* Your own NT-Xent (write your own). Read [methods.md](docs/methods.md#contrastive-learning),
which gives the loss as a formula. Write the NT-Xent loss and compare it with
`itj.ssl.nt_xent(z1, z2, temperature=...)` on random tensors (`torch.randn`). Check your two limits of **W3.4.1**
numerically, and plot the loss of random embeddings against the temperature.

**W3.4.3** *Core, 105 min.* Pretrain an encoder. Pretrain the PFN encoder with the projection head on the 90 000 jets of
`data/t6_pretrain.parquet` for 4 epochs (about 3 minutes). Use two views per jet with `itj.ssl.JetCLRAugment` and
`two_views`, batch 256 and $\tau_T = 0.1$. Fill the three blanks of the loop below (the two views, the loss, the step).
After each epoch, record the loss, the alignment, the uniformity and a linear evaluation with 1 000 labels. The
alignment says how close the two views of a jet land, the uniformity how evenly the jets spread. The four monitoring
calls are given, and you decide where they go. Start the run, then read the hints while it trains. Save the encoder as
`models/toy/jetclr_pfn.pt`.

```python
import numpy as np
import torch

import itj.features
import itj.models
import itj.ssl
from itj.io import load

jets = load("data/t6_pretrain.parquet").drop("label_", "truth_", "aux_")  # never look at the labels
train, val, test = itj.features.split(load("data/t2_detector.parquet"))  # labelled jets, for the evaluation only
encoder = itj.models.build_model("pfn")
head = itj.ssl.ProjectionHead(128, 128, 64)
opt = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], lr=1e-3)
augment, rng = itj.ssl.JetCLRAugment(), np.random.default_rng(0)
for epoch in range(4):
    view1, view2 = ...  # blank 1: two augmented views of all jets
    ds1, ds2 = itj.features.JetDataset(view1, "kin", None), itj.features.JetDataset(view2, "kin", None)
    encoder.train(), head.train()
    for idx in rng.permutation(len(jets))[: len(jets) // 256 * 256].reshape(-1, 256):
        z1, z2 = head(encoder.embed(ds1.batch(idx))), head(encoder.embed(ds2.batch(idx)))
        loss = ...  # blank 2: the NT-Xent loss at temperature 0.1
        ...  # blank 3: the optimiser step (three lines)
itj.models.save_model(encoder, "models/toy/jetclr_pfn.pt")

# the four monitoring calls (the last three in evaluation mode, under torch.no_grad(),
# on fixed views of a few thousand jets)
loss.item()
itj.ssl.alignment(z1, z2)
itj.ssl.uniformity(z1)
itj.ssl.linear_eval(encoder, train, val, n_labels=1000)
```

**W3.4.4** *Core, 45 min.* What did it learn? Freeze the encoder and train a logistic regression on its embedding with
1 000 labels from the train part of `data/t2_detector.parquet`. Test it on the test part
(`itj.ssl.linear_eval(encoder, train, test,
n_labels=1000)`, about a minute). Do the same for the random encoder
(`reset_parameters`) and for the course PFN, with the same logistic regression. Where does the pretraining help, and
where not?

**W3.4.5** *Stretch, 75 min.* Your own loop, longer and other encoders. Write the pretraining loop without the skeleton.
Then run it for 10 epochs (about 7 minutes, in the background): does the linear evaluation keep improving? Then
pretrain `partsmall_nopair` for 4 epochs on 15 000 jets, also in the background (on 30 000 jets it takes about 21
minutes).

**W3.4.6** *Stretch, 2 h.* Against the observables, and truth probes. Run the linear evaluation with all labels. Add the
baselines of the standardised observables $X_\mathrm{phys} = [m_\mathrm{SD}, \tau_{21}, \tau_{32}, D_2, N, \ln p_T]$,
with the same logistic regression and with a BDT as in **W2.2.2**. Then run the truth probes
of **W3.1.2** and **W3.1.4** on the SSL embedding. Does it know quark from gluon jets, the number of emissions, the
opening angle of the W (`truth_dr_min` of the W jets as the target)?

**W3.4.7** *Stretch, 3.5 h.* Few labels. This takes about 13 minutes of CPU: run it in the background. Plot the AUC (W
against QCD and top against QCD) against the number of labelled jets (100, 300, 1 000, 3 000, 10 000; 3 seeds). Do it
for three methods: SSL + linear evaluation, SSL + fine-tuning (training the whole pretrained encoder on the labels) and
supervised training from scratch. Train every point with the rule of **W2.5.3** (`min_steps=2000`, `val_every=50`,
`patience=5`), but on a validation set of as many labelled jets as the training set. Validation labels are labels too,
so count them in the budget (the linear evaluation needs none). The alternative is to train for a fixed number of steps
without early stopping. At 100 and 300 labels an epoch is one or two steps, and a validation costs almost nothing. Use
`val_every=10`, and if the best state is still the first validation, validate more often. Add the curve of **W2.5.3**
for reference if you made it. It used 5000 validation labels at every size, so its gap to your scratch curve is one
measure of what the validation labels are worth. Where does pretraining help, if anywhere?

**Deliverables:** your two expressions (**W3.4.1**); the `allclose` line, the two numerical limits and the plot against
the temperature (**W3.4.2**); `W3.4.3_ssl_training.pdf` and the encoder (**W3.4.3**); the rows and a short comparison
(**W3.4.4**); `W3.4.6_probes.pdf` (**W3.4.6**); `W3.4.7_few_labels.pdf` (**W3.4.7**).

**Hints**
- **W3.4.1:** count the negatives of one positive pair first. Write the loss of one positive pair as a softmax over
  $2B - 1$ similarities before you sum over pairs.
- **W3.4.2:** normalise the projections first. The positive pair must not be counted among its own negatives: set the
  diagonal of the $2B \times 2B$ similarity matrix to $-\infty$ before the softmax, not to 0. Stack the two views as
  `torch.cat([z1, z2])`, so that the partner of row $i$ is row $i \pm B$.
- **W3.4.3:** the loss falls fast and then flattens well above its floor. That is expected with these augmentations:
  judge the run by the alignment, the uniformity and the linear evaluation. Evaluate the encoder output (the embedding
  before the projection head), not the projection. `JetCLRAugment` drops the truth columns first, for speed. If the
  linear evaluation of your encoder is far below the reference in [methods.md](docs/methods.md#linear-evaluation), check
  the augmentations first.
- **W3.4.4:** [methods.md](docs/methods.md#linear-evaluation); the random encoder is the control, as in **W3.1.2**. Call
  `linear_eval` three times with the same labels and seed.
- **W3.4.6:** a BDT with 1 000 labels must not early-stop on 15 000 validation labels.

#### Weekly summary

"What our taggers know": one paragraph with numbers, answering the thread question *What do our taggers know about a
jet, and what do they ignore?* (0.5 h with the hand-in). Hand in `answers/week3.md` with the summary and the answers to
every warm-up and core ID. Also hand in the figures `W3.1.1_pca.pdf`, `W3.1.2_probes.pdf` with the probe table,
`W3.2.1_noue.pdf`, `W3.2.2_filters.pdf`, `W3.3.1_soft.pdf`, `W3.3.2_irc.pdf`, `W3.3.3_augmentations.pdf` and
`W3.4.3_ssl_training.pdf`, the checks of your NT-Xent and the rows of **W3.4.4**. In the discussion (0.5 h) we go
through the core IDs.

---

## Week 4: Real jets, and where does a tagger look? (core path about 12.75 h, stretch about 32.75 h)

**Why this week.** Everything so far was a toy. JetClass was simulated with full generators (programs that simulate the
collision, the shower and the hadrons) and a detector simulation. ParT was trained on 100 million of its jets. Which
conclusions survive? An interpretability tool is a method that shows which inputs a score depends on. Each one is
checked on the toy, where the truth is known, before it is used on the real ParT.

**Thread.** Weeks 1-3 covered what a jet is, the taggers and what they learn, all on the toy. This week brings real jets
and a real tagger. Next, the IRC tests of week 3 are applied to the networks. Last, each attribution tool (a method that
gives every particle a share of the score) is tried on the toy, where the prongs are known. Right after that, it is used
on the real ParT. The final summary answers the course question.

**Concepts this week**, in order of use. Real jets: JetClass and how it was made (MadGraph makes the hard collision,
PYTHIA the shower and the hadrons, Delphes the detector; [primer 9](docs/primer.md#what-a-detector-sees),
[jetclass.md](docs/jetclass.md#how-the-events-were-made)). Particle identification (which kind of particle each one is)
and impact parameters (how far a track misses the collision point), in words. Pretrained taggers are models trained by
others, loaded and run as they are. Their ten-class softmax turns the outputs into probabilities that add up to one, and
you renormalise it to three classes. Next come domain shift and transfer ([ml_primer
10](docs/ml_primer.md#domain-shift)). IRC tests of networks: safety by construction against safety in practice.
Attributions: gradients with respect to the inputs; saliency (how strongly the score reacts to each input); integrated
gradients (baseline, completeness; [ml_primer 11](docs/ml_primer.md#gradients-of-the-input)). Faithfulness by deletion:
does the score fall when the top-ranked particles are removed? Proxies for the truth: measured quantities that stand in
for the truth labels that real data lack. Stretch adds reweighting, weaver, attention statistics and probes per block.
It also adds the pair bias with its ablations (switching off one part of a model to see what it contributes).

**This week at a glance** (times in minutes, or hours where marked):

| task | warm-up | core | stretch |
|---|---|---|---|
| start of week | box (15) | | |
| **W4.1** Real jets: JetClass | **W4.1.1** 45 | **W4.1.2** 75 | **W4.1.3** 90, **W4.1.4** 45, **W4.1.5** 105, **W4.1.6** 75, **W4.1.7** 2.25 h |
| **W4.2** A tagger trained on 100 million jets | **W4.2.1** 30 | **W4.2.2** 60, **W4.2.3** 60 | **W4.2.4** 2 h, **W4.2.5** 30, **W4.2.6** 90, **W4.2.7** 3.5 h, **W4.2.8** 60 + GPU, **W4.2.9** 2 h |
| **W4.3** Are the taggers IRC safe? | **W4.3.1** 30 | **W4.3.2** 60 | **W4.3.3** 60, **W4.3.4** 2.25 h |
| **W4.4** Attributions on the toy | **W4.4.1** 45 | **W4.4.2** 2 h, **W4.4.3** 60 | **W4.4.4** 2 h, **W4.4.5** 60 |
| **W4.5** Where does the real ParT look? | **W4.5.1** 30 | **W4.5.2** 45 | **W4.5.3** 75, **W4.5.4** 105, **W4.5.5** 30, **W4.5.6** 30, **W4.5.7** 105, **W4.5.8** 45, **W4.5.9** 2 h |
| final summary, hand-in and discussion | | 1.5 h | |

**Start of week** (15 minutes, then about 2 minutes in the background):

```bash
python examples/make_task_data.py --week 4
```

It skips the downloads that are already complete (they started in week 3). It then writes the split files for weaver
and the two bonus toy files of [Data for week 4](#data-for-week-4). Reading for this week: Jain and Wallace, *Attention
is not Explanation* (arXiv:1902.10186), for the stretch questions of **W4.5**.

**Day plan.** Day 1 (about 6.25 h): the start-of-week
box, **W4.1.1**-**W4.1.2**, **W4.2.1**-**W4.2.3**, **W4.3.1**-**W4.3.2**. Day 2 (about 6.5
h): **W4.4.1**-**W4.4.3**, **W4.5.1**-**W4.5.2**, the final summary and the discussion. Day 3 (optional): the first
stretch picks and catch-up. Three shorter days instead: box, **W4.1**, **W4.2** (4.75 h)
/ **W4.3**, **W4.4.1**-**W4.4.2** (4.25 h) / **W4.4.3**, **W4.5**, final summary, discussion (3.75 h).

**Check-in at the end of day 1:** `ParT_kin` gives a ROC curve (**W4.2.1**).

**If you are behind** (agreed in advance; each saves the time given):
- **W4.1.2** with $m_\mathrm{SD}$ and $N$ only (0.25 h; $\tau_{21}$ is lost).
- *Two-day:* **W4.2.2** with `ParT_kin` and `ParT_full` only (0.25 h; ParticleNet is lost).
- **W4.2.3** with the course PFN only (0.25 h; the EFN and the IRC-safety question of the transfer are lost).
- **W4.3.2** with the toy models only (0.25 h; the scan of `ParT_kin` is lost).
- *Two-day:* **W4.4.2** with completeness checked on 50 jets and the prong shares on 100 W jets (0.25 h; statistics are
  lost).
- **W4.4.3** with ParT-small only (0.25 h; the PFN is lost).
- Never skip **W4.5.2**: it answers the course question.

**First stretch picks:** **W4.5.3** (prong heads), **W4.2.4** (diagnosing the transfer), **W4.2.9** (probes on the real
ParT), and **W4.2.7** (weaver) for every student who continues to the research project.

**Models and data of this week.** The toy models in `models/course/`; the pretrained `ParT_kin`, `ParT_full` and
`ParticleNet_kin` in `models/pretrained/`. JetClass jets: `itj.jetclass.load_jetclass("val", part="test")` (the first 20
% of each file).

### Data for week 4

Three JetClass validation files, the pretrained models of the ParT paper, the split files for weaver and two bonus toy
files. `python examples/make_task_data.py --week 4` runs these commands; it needs `data/t2_detector.parquet` from
week 2. The core questions download about 0.4 GB, the bonus **B.4** downloads 136 MB more, and about 1.2 GB are written.
`--skip-fetch` leaves out the downloads, and `--only fetch` (the week-3 box) runs only the downloads. `jetgen fetch`
resumes an interrupted download. It checks the size and the MD5 checksum of every file
and keeps the files that are already complete ([jetclass.md](docs/jetclass.md#download)).

```bash
# week 4: JetClass validation files (QCD = ZJetsToNuNu, W = WToQQ, top = TTBar), 100 000 jets each, 374 MB
jetgen fetch jetclass --split val --classes QCD,Wqq,Tbqq
# W4.2-W4.5: the pretrained ParT and ParticleNet of the JetClass paper (19 MB)
jetgen fetch models ParT_kin ParT_full ParticleNet_kin
# W4.2.7: the itj split (test 20 %, val 10 %, train 70 % of each file) written as files for weaver
jetgen split data/jetclass/val/ZJetsToNuNu_120.parquet data/jetclass/val/WToQQ_120.parquet data/jetclass/val/TTBar_120.parquet --out-dir data/jetclass/split
jetgen split data/t2_detector.parquet --out-dir data/t2_split
# B.3 (bonus): a known planted shortcut: W jets with more photons from pi0 decays
jetgen generate --override wqq:pi0_fraction=0.50 -n 60000 --seed 72 -o data/t7_shortcut.parquet
# B.3 (bonus): a hidden planted shortcut (do not run jetgen info --reveal on it before you have found it)
jetgen generate --mystery -n 60000 --seed 73 -o data/t7_mystery.parquet
# B.4 (bonus): all ten JetClass classes, 10 000 jets each (ROOT, 136 MB; needs the analysis extra)
jetgen fetch jetclass --split example
```

The JetClass files are read like the toy files. Each file is split in the same way (test = the first 20 000 jets of each
file, val the next 10 000, train the last 70 000):

```python
import itj.jetclass
from itj.io import load

test = itj.jetclass.load_jetclass("val", part="test")  # QCD, W and top, 20 000 jets each
train = load(itj.jetclass.files("val"), part="train", max_jets=20000)  # at most 20 000 jets per file
print(test)  # JetData(60 000 jets: ...; JetClass; ... particles)
```

### W4.1 Real jets: JetClass (core path about 2 h, stretch about 7.5 h)

**Goal:** compare the toy with JetClass, using the same code.

**W4.1.1** *Warm-up, 45 min.* A first look at JetClass. Work through
[notebooks/01_jetclass.ipynb](notebooks/01_jetclass.ipynb), with [docs/jetclass.md](docs/jetclass.md) and [primer
section 9](docs/primer.md#what-a-detector-sees) (Delphes). Then make a short dictionary of the JetClass columns of week
1: `part_px` ... `part_energy`, `part_deta`, `part_dphi`, `jet_pt`, `jet_eta`, `jet_sdmass`, `jet_tau1` to `jet_tau3`,
`jet_nparticles` and the labels. Give the meaning of each in JetClass. Take `part_deta` and `part_dphi` from **W1.1.7**
if you did it, otherwise from [primer section 3](docs/primer.md#collider-coordinates). Also say which of them JetClass
does not have (`jetgen info --columns` on a JetClass file and on a toy file).

**W4.1.2** *Core, 75 min.* Toy against JetClass. Take 10 000 jets per class from each data set. From JetClass, take the
first 10 000 jets of each test part. From `data/t2_detector.parquet`, take the first 10 000 jets per class after
`itj.features.jetclass_selection` ($500 < p_T^\mathrm{jet} < 1000$ GeV, the cut of JetClass). Compare $m_\mathrm{SD}$,
$\tau_{21}$ and $N$ per class with your **W1.1.6** function (about 45 s). Make a table of the medians, and explain the
two largest differences with [What the toy leaves out](docs/physics.md#what-the-toy-leaves-out).

**W4.1.3** *Stretch, 90 min.* The full comparison. Add $\tau_{32}$, $D_2$, the composition by particle type (counts and
energy) and $p_T$ per class to **W4.1.2**. Put the medians next to the validation numbers of
[physics.md](docs/physics.md#calibration-and-validation).

**W4.1.4** *Stretch, 45 min.* Check the stored branches. Recompute `jet_sdmass` from `part_*`
(`itj.observables.compute(data, ["sdmass"])`) and give the fraction of jets within 1 %. Then recompute `part_deta` on a
JetClass jet at negative $\eta$, with your **W1.1.7** code (or from [physics.md](docs/physics.md#units-and-coordinates)
if you skipped it).

**W4.1.5** *Stretch, 105 min.* Which axes did Delphes use? N-subjettiness measures distances to a set of axes, one per
subjet. Recompute `jet_tau1` to `jet_tau3`. Which axis choice (WTA-kT, plain exclusive kT, `onepass_kt` in
`itj.observables.subjet_axes`) reproduces the stored $\tau_N$, and why? Give the fraction of jets within 1 %. About 3
minutes (`onepass_kt` about 70 s).

**W4.1.6** *Stretch, 75 min.* The full dictionary. Use [docs/jetclass.md](docs/jetclass.md),
[docs/data_format.md](docs/data_format.md) and `jetgen info --columns` to make a table. It lists every column of a toy
file with its JetClass meaning. It also lists the JetClass columns that the toy fills trivially, such as the H labels of
the Higgs classes and the aux match code. Last, it lists the columns that only the toy has (`truth_*`). Name the two
truth columns you expect to be most useful for interpretability, and say why.

**W4.1.7** *Stretch, 2.25 h.* Train on JetClass on a CPU. Train the PFN and the EFN on 60 000 jets (20 000 per class
from the train parts, 10 epochs, about 1.5 minutes each):

```bash
python examples/train_tagger.py --model pfn --data data/jetclass/val/ZJetsToNuNu_120.parquet data/jetclass/val/WToQQ_120.parquet data/jetclass/val/TTBar_120.parquet --train-size 20000 --epochs 10 --out models/jetclass/pfn
python examples/train_tagger.py --model efn --data data/jetclass/val/ZJetsToNuNu_120.parquet data/jetclass/val/WToQQ_120.parquet data/jetclass/val/TTBar_120.parquet --train-size 20000 --epochs 10 --out models/jetclass/efn
```

Then train PN-lite (60 000 jets, 5 epochs, about 9-10 minutes, in the background) and the BDT (seconds). *Optional:*
ParT-small (30 000 jets, 5 epochs, about 12 minutes, in the background). Compare with **W4.2.2** (100 M jets and GPUs)
and with the same models on the toy. Is JetClass harder, and for which pair of classes?

**Deliverables:** the dictionary table (**W4.1.1**); `W4.1.2_toy_vs_jetclass.pdf` and the table of medians (**W4.1.2**);
the extended table (**W4.1.3**); the two fractions and the sign check (**W4.1.4**, **W4.1.5**); the full table
(**W4.1.6**); `W4.1.7_roc_jetclass.pdf` and the rows (**W4.1.7**).

**Hints**
- **W4.1.1:** the labels of JetClass are one-hot branches for ten classes (one branch per class, set only for the true
  class); the course uses three.
- **W4.1.2:** `jetclass_selection` has no $\lvert\eta\rvert < 2$ cut, and the $\lvert y\rvert < 2$ of the toy acts on
  the seed parton, the quark or gluon the jet starts from ([data_format.md](docs/data_format.md#jetclass-oddities)).
- **W4.1.5:** guard against $\tau = 0$, and remember that some JetClass particles lie at $\Delta R > 0.8$
  ([data_format.md](docs/data_format.md#jetclass-oddities)). Check your numbers afterwards against
  [observables.md](docs/observables.md#jetclass-conventions).

### W4.2 A tagger trained on 100 million jets (core path about 2.5 h, stretch about 10.5 h)

**Goal:** run the pretrained taggers of the ParT paper, and see how well the toy taggers and the real ParT do on each
other's jets.

**W4.2.1** *Warm-up, 30 min.* Run the pretrained ParT. Load `ParT_kin` (the ParT that sees only the kinematic inputs)
and run it on 5 000 test jets per class (about 70 s). Keep the QCD, W and top outputs with `probs_to_jets3`, and draw
the ROC curves of W and top against QCD. Add the rows to your table.

```python
import itj.jetclass
import itj.models
import itj.train

model = itj.models.load_pretrained("ParT_kin")
test = itj.jetclass.load_jetclass("val", part="test", max_jets_per_class=5000)
probs = itj.train.predict(model, test)  # ten outputs per jet
probs3 = itj.jetclass.probs_to_jets3(probs)  # QCD, W, top
```

**W4.2.2** *Core, 60 min.* Three pretrained taggers. Add `ParT_full` and `ParticleNet_kin` (about 70 s per ParT and 2.5
minutes for ParticleNet on 15 000 jets). (a) Compute the accuracy of the argmax over the ten outputs (the class with the
largest output) on the three classes. Then compute the three-class accuracy after `probs_to_jets3`. Why do they differ?
(b) Give the one-vs-one AUC (W and top against QCD) for all three taggers. What does `ParT_full` gain from the particle
types and the impact parameters, and on which pair?

**W4.2.3** *Core, 60 min.* From the toy to real jets and back. *Predict first:* will the IRC-safe EFN transfer best?
Evaluate the course PFN and EFN on the JetClass test jets, and `ParT_kin` on toy jets after `jetclass_selection`. Make
the table of the W-against-QCD AUC (trained on toy or JetClass, tested on toy or JetClass; about a minute). Does a ParT
trained on real jets recognise toy W jets?

**W4.2.4** *Stretch, 2 h.* Diagnosing the transfer. Make the full AUC matrix per model (with the JetClass models
of **W4.1.7** if you made them). Then diagnose the differences with two tools. First, reweight the toy jets so that
their multiplicity per class matches JetClass ([ml_primer](docs/ml_primer.md#domain-shift)). Second, drop the photons
or the neutral particles at evaluation (`itj.augment.remove_particles`, after the selection). The matrix takes about 6
minutes and the two diagnostics about 6 more: start each in the background, then read. This answers the $N$ question
of **W2.2.4**.

**W4.2.5** *Stretch, 30 min.* More from the pretrained taggers. Compute $R_{50}$ on all 20 000 QCD test jets, and find
where exactly `ParT_full` gains.

**W4.2.6** *Stretch, 90 min.* More transfer. Four studies: the JetClass models of **W4.1.7** on the toy; the score
distributions per class; the probes of **W3.1.2** across the two data sets. The fourth is fine-tuning: train the toy EFN
further on 1 000 or 10 000 JetClass jets, and compare with training from scratch (a look back at **W3.4.7**).

**W4.2.7** *Stretch, 3.5 h.* weaver
([jetclass.md](docs/jetclass.md#weaver-in-one-page)). Install its extra (Python 3.10 to 3.13, not on Intel Macs):
`python -m pip install -e ".[weaver]"`, then `python -c "import weaver"`. If that fails, do this question with `itj`
only and say so.

Start with a short CPU run (about 1 minute, 6-7 GB of memory). Train a ParT of the size of `partsmall` for 30 steps at
a constant learning rate (`--lr-scheduler none`) with `jets3_kin.yaml` (`jetgen configs` prints its path) on the split
toy files. Then run the identical command on the JetClass split files (one named group per class).

Next, run `weaver --predict` with the pretrained `ParT_kin` (`JetClass_kin.yaml`, `--data-fraction 0.05`, `--gpus ''`)
on the split test files of JetClass. In test mode weaver reads the first `--data-fraction` of every file in order. Its
jets are therefore `load(path, max_jets=1000)` of each split test file (e.g.
`data/jetclass/split/test/WToQQ_120.parquet`). Compare its scores with those of `itj` and report the largest
difference.

Read the confusion matrix (true class against predicted class). As a check that the run trained,
after 30 steps expect an accuracy of about 0.6-0.7. The learning rate did not reach the run if you get an
accuracy near 1/3, chance for three classes. weaver 0.5.4 has no seed option, so two such runs differ by a few
hundredths. Finally, load the checkpoint (the saved weights) of the toy run into `itj` with
`load_model(path, name="partsmall")`. Recommended for every student who
continues to the research project. The toy run, as one line (the others are in
[examples/weaver/README.md](examples/weaver/README.md)):

```bash
weaver --data-train 't2:data/t2_split/train/t2_detector.parquet' --data-val data/t2_split/val/t2_detector.parquet --data-test 't2:data/t2_split/test/t2_detector.parquet' --data-config src/itj/configs/jets3_kin.yaml --network-config examples/weaver/networks/example_ParticleTransformer.py -o num_layers 3 -o num_cls_layers 1 -o embed_dims "[64,256,64]" -o pair_embed_dims "[32,32,32]" -o num_heads 4 --model-prefix runs/weaver_partsmall/net --gpus '' --num-workers 0 --batch-size 256 --start-lr 1e-3 --num-epochs 1 --steps-per-epoch 30 --lr-scheduler none --in-memory --fetch-step 1 --predict-output pred.parquet
```

**W4.2.8** *Stretch, 60 min + GPU time.* A full training on a GPU. *Optional (GPU):* a full ParT or ParticleNet training
on the three-class subset ([examples/weaver/README.md](examples/weaver/README.md)).

**W4.2.9** *Stretch, 2 h.* Probes on the real ParT. Run the probes of **W3.1.2** and **W3.1.4** on the class token of
`ParT_kin` (10 000 JetClass test jets) and on the pooled output of each block. The class token is the summary vector
that the last layer reads ([ml_primer](docs/ml_primer.md#networks-on-particles)). The probe targets are $m_\mathrm{SD}$,
$\tau_{21}$, $\tau_{32}$, $D_2$, $N$ and the largest $\ln k_t$ among the primary declusterings of `lund_primary`. Do the
same for the toy ParT-small. Is it the same picture? Compare with arXiv:2605.09881 (the basis of energy correlation
functions, ECF, against the N-subjettiness basis). About 2 minutes.

**Deliverables:** `W4.2.1_roc_part.pdf` and the rows (**W4.2.1**); `W4.2.2_roc_pretrained.pdf` and the rows
(**W4.2.2**); `W4.2.3_transfer.pdf` and the AUC table (**W4.2.3**); the AUC matrix and the two diagnostics (**W4.2.4**);
the agreement between weaver and `itj` and the accuracy of the weaver run (**W4.2.7**); `W4.2.9_part_probes.pdf`
(**W4.2.9**).

**Hints**
- **W4.2.1:** the order of the ten JetClass classes is `itj.jetclass.CLASSES` (QCD 0, ..., Wqq 7, Tbqq 8).
- **W4.2.2:** report the top rejection at $\epsilon_S = 0.8$ with its bootstrap error: $R_{50}$ on 5 000 QCD jets rests
  on a handful of jets. `probs_to_jets3` renormalises; it does not sum. Check your numbers afterwards against
  [jetclass.md](docs/jetclass.md#models-and-numbers).
- **W4.2.3:** use the same selection on both data sets. The toy models have three outputs, `ParT_kin` has ten.
- **W4.2.4:** reweight the jets, not the labels.
- **W4.2.7**, weaver on a laptop. Use `--gpus ''` (the CPU), and `--num-workers` at most the number of files in each
  named group (`0` is fine). Use `--in-memory` for small subsets; weaver 0.5.4 then needs `--steps-per-epoch`, and
  `--steps-per-epoch-val` sets the length of the validation. Use `--fetch-step 1`, because weaver 0.5.4 fails on a
  partial read of a Parquet file. Always use the split files, never an index selection in the YAML: weaver applies
  selections after concatenating the files of a fetch. On Apple Silicon weaver picks MPS (the Apple GPU) by itself: set
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.

### W4.3 Are the taggers IRC safe? (core path about 1.5 h, stretch about 3.25 h)

**Goal:** apply the IRC scans of **W3.3** to the networks, and find which taggers are safe by construction and which
only in practice.

**W4.3.1** *Warm-up, 30 min.* Scan the EFN and the PFN. Run `itj.interpret.irc_scan` on the course EFN and PFN (seconds)
with soft ($n = 1$, $\epsilon = 10^{-1}, \ldots, 10^{-6}$) and collinear ($\lambda = 0.5$, $\delta = 10^{-1}, \ldots,
10^{-5}$) perturbations. Plot the mean $\lvert\Delta s\rvert$ (the change of the score) against $\epsilon$ and $\delta$,
next to your **W3.3.2** curves.

```python
import itj.features
import itj.interpret
import itj.models
from itj.io import load

train, val, test = itj.features.split(load("data/t2_detector.parquet"))
model = itj.models.load_model("models/course/efn.pt")
soft = itj.interpret.irc_scan(model, test, mode="soft", values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
```

**W4.3.2** *Core, 60 min.* Every tagger. *Predict first:* which of your taggers are IRC safe by construction? Then scan
soft ($n = 1, 5$) and collinear ($\lambda = 0.5, 0.1$) perturbations on the EFN, PFN, PN-lite and ParT-small (toy) and
on `ParT_kin` (JetClass). Compare with your prediction. About 1.5 minutes for `ParT_kin`, seconds for the toy models.

**W4.3.3** *Stretch, 60 min.* The numerical pitfalls. How does each pitfall of these scans show in your curves
([observables.md](docs/observables.md#numerical-irc-tests))? There are four. The inputs are truncated at 128 particles.
ParT clamps $k_T$, $\Delta$ and $m^2$ at $10^{-8}$ inside its pair features ($\ln = -18.4$), before a BatchNorm (a layer
that standardises its inputs). Soft particles below the detector thresholds lie outside the training data. The sign of
`part_deta` flips when a soft particle moves $\eta_\mathrm{jet}$ across 0.

**W4.3.4** *Stretch, 2.25 h.* Inherited invariances. Run `irc_scan(..., output="embedding")` on the SSL encoder
of **W3.4.3**, and the score scan on its linear head. Compare both with the course PFN. Is the SSL encoder more
collinear safe? Retrain it without `split_collinear` (about 2 minutes) and compare.

**Deliverables:** `W4.3.1_irc_efn_pfn.pdf` (**W4.3.1**); `W4.3.2_irc_soft.pdf` and `W4.3.2_irc_collinear.pdf`
(**W4.3.2**); one sentence per pitfall (**W4.3.3**); `W4.3.4_irc_ssl.pdf` (**W4.3.4**).

**Hints**
- **W4.3.1:** read [observables.md](docs/observables.md#numerical-irc-tests) and
  [methods.md](docs/methods.md#irc-tests-of-networks).
- **W4.3.2:** `irc_scan` refuses jets that a perturbation would push beyond 128 particles
  (`itj.augment.max_particles_guard` tells you whether it would). It keeps the sign of `part_deta` of the
  unperturbed jet.
- **W4.3.4:** fold the scaler and the logistic regression of the linear head into one linear output layer, so that
  `irc_scan` sees one model. Compare embedding scans only within one encoder.

### W4.4 Attributions on the toy (core path about 3.75 h, stretch about 3 h)

**Goal:** find which particles a tagger's score depends on, and test whether that answer can be trusted. You work on
toy jets, whose prongs are known. Models: the course ParT-small and PFN; data: the test part of
`data/t2_detector.parquet`.

**W4.4.1** *Warm-up, 45 min.* Saliency of one jet. Read [ml_primer section 11](docs/ml_primer.md#gradients-of-the-input)
and [methods.md](docs/methods.md#attributions). Compute the saliency of the W score of the course ParT-small for one W
jet. Draw the jet coloured by it (`itj.plot.attribution_display`; seconds). Do the bright particles sit on the prongs
(`truth_part_prong`)?

```python
import itj.features
import itj.interpret
import itj.models
import itj.train
from itj.io import load

train, val, test = itj.features.split(load("data/t2_detector.parquet"))
model = itj.models.load_model("models/course/partsmall.pt")
jet = test.select([int((test.process_names == "wqq").argmax())])  # the first W jet
values = itj.interpret.saliency(model, itj.train.make_batch(jet), target=1)  # target 1: the W output
```

**W4.4.2** *Core, 2 h.* Your own integrated gradients (write your own). Implement integrated gradients for the course
ParT-small with 32 steps and the zero baseline, using `itj.interpret.input_gradients` for the gradients. Check
completeness (the attributions sum to $s(x) - s(x_0)$), and compare with `itj.interpret.integrated_gradients(model,
batch, target=..., steps=32)`. Then, on 250 W and 250 top jets, give the share of the attribution per prong and on the
underlying event (about a minute).

**W4.4.3** *Core, 60 min.* Do attributions tell the truth? *Predict first:* how much does the W score drop when one W
prong is removed, compared with removing as many random particles? Then do it on the toy W jets for ParT-small and the
PFN (`itj.augment.remove_particles` with `truth_part_prong`, then `predict`). Compare the particles that carry the
largest attributions with the prong you removed.

**W4.4.4** *Stretch, 2 h.* Baselines and completeness. Add a "soft" baseline next to the zero baseline. Completeness
holds for the zero baseline with `wrt="features"`, but for the soft baseline on ParT it needs `wrt=("features",
"vectors")`: why? Then give the completeness gap at 32, 128 and 512 steps, and repeat it with PN-lite as a second
model. Give the share of the attribution on the softest 50 % of the particles, and make five jet displays coloured by
attribution.

**W4.4.5** *Stretch, 60 min.* Deletion curves. Remove particles in the order of their attribution, in the order of their
$p_T$, and at random (several seeds). Plot the score against the fraction removed. Which order ranks the particles best?

**Deliverables:** `W4.4.1_saliency.png`, with a colour bar that says signed or absolute (**W4.4.1**);
`W4.4.2_prong_fraction.pdf`, the completeness gap and the `allclose` line (**W4.4.2**); `W4.4.3_truth_deletion.pdf`
(**W4.4.3**); `W4.4.4_attribution_gallery.pdf` (**W4.4.4**); `W4.4.5_deletion.pdf` (**W4.4.5**).

**Hints**
- **W4.4.1:** saliency is a gradient with respect to the inputs: the change of the score when one input of one particle
  changes a little. `target=1` is the W output of the toy models (QCD 0, W 1, top 2).
- **W4.4.2:** never call `.backward()` on the raw vectors of ParT yourself, because the pair features of a particle with
  itself give NaN gradients; use `input_gradients`. Build it in three steps: the 32 inputs on the straight line from the
  baseline to the jet; the mean of their gradients; times (input minus baseline). Check completeness on one jet before
  you run 500. Say whether a map is signed or absolute.
- **W4.4.3:** use several random draws for the random removal.
- **W4.4.4:** [methods.md](docs/methods.md#attributions) gives realistic completeness gaps; report both baselines.

### W4.5 Where does the real ParT look? (core path about 1.25 h, stretch about 8.5 h)

**Goal:** answer the course question on real jets. Real jets have no truth labels, so you use a proxy for the truth and
check it on the toy first. Then look at attention and the pair bias.

**W4.5.1** *Warm-up, 30 min.* A proxy for the prongs. On JetClass there is no truth of the prongs. The reco-level proxy
`itj.interpret.subjet_membership(data, 2)` assigns each particle to a subjet, using only the reconstructed particles.
Check it against `truth_part_prong` on 250 toy W jets of the test part of `data/t2_detector.parquet`. How often do they
agree? Give the agreement with its error.

**W4.5.2** *Core, 45 min.* Where does the real ParT look? Run integrated gradients (the library version, zero baseline)
with `ParT_kin` on 200 JetClass W jets (`integrated_gradients(model, batch, target=7)`: Wqq among the ten classes, as in
notebook 01; one or two minutes). Give the share of the attribution per subjet of the proxy and outside both. What can
you say without truth, and how does it compare with the toy (**W4.4.2**)? This figure goes into your final summary.

**W4.5.3** *Stretch, 75 min.* Prong heads? First draw one picture: the attention matrix of one head for one toy W jet
(`model.trace(make_batch(jets), attention=True).attention["block2"]`, cut to the real particles, drawn with
`itj.plot.attention_matrix(A, groups=truth_part_prong)`). The matrix shows how much each particle attends to each other
particle. Then run `itj.interpret.attention_stats` on 250 W and 250 top
toy jets, with `truth_part_prong` as the groups. Per block and head, it gives the share of the attention on other
particles that stays within the prong (against uniform attention) and the self-attention $A_{ii}$. Are there "prong
heads", heads that attend mainly within a prong? What can and cannot be concluded from attention (arXiv:1902.10186)?
About a minute.

**W4.5.4** *Stretch, 105 min.* Attention on JetClass. Run `attention_stats` of `ParT_kin` on 250 JetClass W and 250 top
jets with the proxy groups of **W4.5.1**. Put the result next to the toy truth and the toy proxy; *optional:* the rank
correlation of class attention (the attention of the class token) and integrated gradients.

**W4.5.5** *Stretch, 30 min.* What $U$ adds. Compare the course `partsmall` (with the pair bias) against
`partsmall_nopair` (without it), by AUC and $R_{50}$ with errors. This is the ablation by retraining.

**W4.5.6** *Stretch, 30 min.* Setting $U$ to zero. *Predict first:* what happens to the top accuracy of `ParT_kin` when
$U$ is set to 0 (`with itj.interpret.zero_pair_bias(model):`)? Then explain: why does that say little about how useful
$U$ is?

**W4.5.7** *Stretch, 105 min.* What $U$ learns. Regress $U_{ij}$ of each head on $[\ln k_T, \ln z, \ln\Delta, \ln m^2]$
(`itj.interpret.pair_features`) for the toy ParT-small and `ParT_kin`. Which Lund coordinate (**W1.2.8**) does each head
use? Then replace one pair channel at a time by its mean (`itj.interpret.ablate_pair_channels`) and report the change of
the AUC. Does it predict **W4.5.5**? (Replacing all four channels at once by their mean is the same as $U = 0$.)

**W4.5.8** *Stretch, 45 min.* One pair channel. Retrain ParT-small with `pair_input_dim=1` (about 6 minutes, in the
background) and compare with **W4.5.5**.

**W4.5.9** *Stretch, 2 h.* Your own question. One example is SmoothGrad, saliency averaged over noisy copies of the
input. Another is attention rollout, attention multiplied through the blocks (arXiv:2005.00928,
`itj.interpret.attention_rollout`). A third is layer-wise relevance propagation, the score passed back layer by layer
to the inputs.

**Deliverables:** the agreement fraction with its error (**W4.5.1**); `W4.5.2_part_attribution.pdf` (**W4.5.2**);
`W4.5.3_heads.pdf` and `W4.5.3_attention_example.pdf` (**W4.5.3**); `W4.5.4_heads_jetclass.pdf` (**W4.5.4**); the
ablation table (**W4.5.5**, **W4.5.6**, **W4.5.8**); `W4.5.7_pair_bias.pdf` (**W4.5.7**); a figure of your own
(**W4.5.9**).

**Hints**
- **W4.5.1:** compare particle by particle, and count the underlying event ($-1$) separately.
- **W4.5.3:** the rows of an attention matrix sum to 1. Compare with $1/N$ of the real particles, not of the padded
  length, and look at the diagonal ($A_{ii}$) separately. Never `trace` with attention on more than a few hundred jets
  at once (memory); `attention_stats` does the reduction batch by batch.
- **W4.5.6:** the reasons why zeroing $U$ misleads are in [methods.md](docs/methods.md#pair-feature-ablation): read them
  after your prediction.

#### Final summary

Write one page, *What is in a jet, and where does a tagger look?*, with the figures from weeks 1-4 that support it. It
takes 1 h with the hand-in and replaces the weekly summary. Hand in `answers/week4.md` with the summary and the answers
to every warm-up and core ID. Hand in these figures: `W4.1.2_toy_vs_jetclass.pdf` with the table of medians,
`W4.2.1_roc_part.pdf`, `W4.2.2_roc_pretrained.pdf` with the rows, `W4.2.3_transfer.pdf` with the AUC table,
`W4.3.1_irc_efn_pfn.pdf`, `W4.3.2_irc_soft.pdf`, `W4.3.2_irc_collinear.pdf`, `W4.4.1_saliency.png`,
`W4.4.2_prong_fraction.pdf`, `W4.4.3_truth_deletion.pdf` and `W4.5.2_part_attribution.pdf`. In the discussion (0.5 h)
we go through the core IDs.

---

## Bonus: towards a research project (about 24.5 h)

### B Bonus

**Goal:** the first steps of a research project. The bonus covers self-supervision on real jets, a pretraining objective
beyond contrastive learning, and a hunt for shortcuts with the tools of the course. A shortcut is a feature that
separates the classes in the training data without being the physics a tagger should use
([physics.md](docs/physics.md#planted-shortcuts)). The bonus ends with a paper from the current research and a one-page
proposal. It has no tiers: all of it is beyond the core. Each question names the weeks it needs; its data are made by
the week-4 box.

**B.1** *Bonus, 2.75 h.* SSL on JetClass (needs **W3.4** and **W4.1**; *GPU welcome*). Either pretrain on unlabelled
JetClass jets and [evaluate linearly](docs/methods.md#linear-evaluation) on JetClass, or transfer the toy-pretrained
encoder of **W3.4.3** to JetClass.

**B.2** *Bonus, 7.25 h.* Masked particle modelling (needs **W3.4** and **W2.5**; arXiv:2401.13537, arXiv:2409.12589).
The network sees a jet with some particles hidden and learns to predict them
([methods.md](docs/methods.md#masked-particle-modelling)). Hide 30 % of the particles of every jet with
`itj.ssl.mask_particles(batch, frac=0.3)`. Train `partsmall_nopair` together with an
`itj.ssl.MaskedParticleDecoder(64, 3)` on its last particle block (`model.trace(masked).layers["block2"]`). The decoder
predicts the masked ($\ln p_T$, $\Delta\eta$, $\Delta\phi$), each standardised, with a regression loss. Use the first
30 000 jets of `data/t6_pretrain.parquet`, 8 epochs of batch 256 (about 14 minutes).

A masked particle is an empty token, and ParT without $U$ has no notion of order, so every masked slot of a jet gets the
same encoder output. The decoder therefore adds the $p_T$ rank of each slot (a learned slot embedding), and the encoder
has to write what it infers about the hidden particles into that output.

Compare the masked loss of each target with trivial predictors. One is the mean of the visible particles of the jet. The
other, for $\ln p_T$, is the mean of the visible rank neighbours of the slot. Which targets does the model learn beyond
them? Evaluate the encoder linearly as in **W3.4.4**, against the random encoder. Do it on the class token and on the
mean of the last particle block (`extract(..., layers="all")["pooled/block2"]`), because the masked loss never trains
the class-attention blocks. *Optional:* the masked-autoencoder variant, `mask_particles(..., drop=True)` (the encoder
sees only the visible particles), with `MaskedParticleDecoder(64, 3, num_layers=1)` (about 17 minutes). Does its encoder
learn more or less?

**B.3** *Bonus, 6.5 h.* Shortcut hunting (needs **W3.1**, **W3.2** and **W4.4**). (a) A known shortcut. Train a PFN with
the particle types, `build_model("pfn", feature_set="kinpid")` (about a minute), on `data/t7_shortcut.parquet`. These
are W jets with a $\pi^0$ share of 0.50 instead of the default (the neutral pion decays to photons). Evaluate the PFN on
clean W jets (`data/t2_detector.parquet`). Can saliency, integrated gradients summed by particle type or a probe for the
photon fraction find the shortcut without being told? (b) A hidden shortcut. `data/t7_mystery.parquet` holds one of the
five shortcuts of [physics.md](docs/physics.md#planted-shortcuts). A first look at the counts per class (`jetgen info`,
the truth columns) is part of the hunt and may already point at a suspect. The task is to show, with the tools of weeks
3 and 4 and with counterfactual edits, that a tagger trained on the file uses the shortcut. Then remove it, and only
then run `jetgen info data/t7_mystery.parquet --reveal`.

**B.4** *Bonus, 5 h.* A paper from the current research (needs **W4.2.9**). Choose one of arXiv:2412.03673,
arXiv:2512.00210, arXiv:2605.09881 and arXiv:2401.13537. Either reproduce one figure qualitatively on the toy, or probe
every block of `ParT_full` on all ten classes. For the probes use the example file,
`data/jetclass/example/JetClass_example_100k.root`: `itj.jetclass.load_jetclass("example", ("all",), part="test")` gives
2 000 jets of each class. The inference takes about a minute and the probes of every block about 5 minutes. Compare with
arXiv:2605.09881 (probes) and your **W4.2.9**. arXiv:2412.03673 analyses only the attention of the final layer (no
probes). For that paper, compare the attention of the last block with its claim of sparse, nearly binary attention.

**B.5** *Bonus, 3 h.* A research proposal (one page), motivated by one of your results of weeks 1-4. It gives the
question, why it matters, the data, the method, the first experiment and its figure, what would change your mind, the
compute and the risks.

**Deliverables:** one row and one figure (**B.1**, **B.2**); the evidence chain in five sentences and the figure that
gave the shortcut away (**B.3**); one figure (**B.4**); the proposal (**B.5**).

**Hints**
- **B.3:** compare the counts of each particle type between the classes before you train anything. Saliency is
  dominated by the kinematic inputs and may not see a shortcut carried by a particle-type flag. Use the gradient of the
  particle-type channels, or integrated gradients summed by particle type. The provenance of the mystery file is the
  record of how it was made. It holds the override and the statistics of the run only in scrambled form: `jetgen info
  --provenance` shows placeholders, and `jetgen info --reveal` prints them. The mapping from the seed to the shortcut,
  and the scrambling, are in the source code, so do not read `itj.toy.mystery` before you are done.
- **B.4:** the example file is a ROOT file (the standard file format of particle physics); `load` reads it with the
  `analysis` extra.
