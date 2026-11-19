# Week 1: What is a jet? (core path about 13.75 h, stretch about 27.5 h)

*How to work, what to hand in, the figure checklist and the notation are in [TASKS.md](../TASKS.md).*

**Why this week.** A tagger is a function of a jet. Before we ask what a tagger looks at, we need to know what is in a
jet and where each of its particles comes from. We also need to know what the classic observables measure. An
observable is a number computed from the particles of a jet, such as its mass. The toy switches the physics on one step
at a time and keeps the truth.

**Thread.** You start with nothing assumed but first-year physics. A W jet holds the decay products of a W boson, a
heavy unstable particle ([see the primer](../docs/primer.md#the-particles)). By the end of the week you can say what a W
jet is made of at each level of the ladder. You can also say which observables tell it apart from a QCD jet. They are
the mass, the soft-drop mass (the jet mass after soft, wide-angle particles are removed), $\tau_{21}$ and $D_2$. The
last two count the hard clusters of energy in a jet. The first tagger of week 2 is built from these observables.

**Concepts this week**, in order of use. The set-up: a virtual environment and tests, and `JetData`, the object that
holds a file in Python. Units and kinematics: GeV and $c = 1$ ([primer 1](../docs/primer.md#units-and-energies));
four-vectors, invariant mass and a boost along the beam ([primer 2](../docs/primer.md#four-vectors-and-invariant-mass));
$p_T$, $\phi$, $y$, $\eta$ and $\Delta R$ ([primer 3](../docs/primer.md#collider-coordinates)). Particles and jets:
quarks, gluons, W, top, hadrons, photons, neutrinos ([primer 4](../docs/primer.md#the-particles)); jets, anti-kT and $R$
([primer 5](../docs/primer.md#what-a-jet-is)); boosted decays and prongs ([primer
6](../docs/primer.md#boosted-w-and-top-jets)). From the first quark to the detector: the parton shower ([primer
7](../docs/primer.md#the-parton-shower)), hadronisation and the underlying event ([primer
8](../docs/primer.md#hadronisation-and-the-underlying-event)), the detector ([primer
9](../docs/primer.md#what-a-detector-sees)). The toy: the presets, paired samples, padding and masks ([primer
10](../docs/primer.md#the-toy-and-its-files)). Analysis: histograms, densities and Poisson errors ([primer
11](../docs/primer.md#histograms-and-errors)); jet mass, soft drop, N-subjettiness, $D_2$ and multiplicity ([primer
12](../docs/primer.md#a-first-look-at-the-observables)). The stretch questions add the Lund plane as a picture ([primer
7](../docs/primer.md#the-parton-shower), skipped on a first reading). They also add relative coordinates and the
JetClass sign, colour factors, the Lund plane of one jet and its density, the Sudakov factor, leading logarithms and
power counting.

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
[notebooks/00_primer.ipynb](../notebooks/00_primer.ipynb) (about 3.5 hours of **W1.1** in all). This is why the warm-ups
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

## Data for week 1

All files go to `data/` in the repository root. Every path in the week files is relative to the repository root (a
notebook in `notebooks/` reads the same file as `../data/t1_detector.parquet`). `python examples/make_task_data.py
--week 1` runs exactly these commands on any operating system with `--workers auto`. It keeps files that already exist:
`--overwrite` makes them again, `--dry-run` only prints them and `--only TEXT` runs the commands whose command line
contains TEXT. Together they make about 290 000 jets and 0.48 GB, in about 5 minutes on one 2016 core or 1.5-2 minutes
with 8 workers. On a cluster, run them in a job or an interactive session, not on the login node
([examples/cluster](../examples/cluster/README.md)). The file names keep the task numbers of release 0.1 (`t1_`, `t2_`,
`t3_`, `t6_`, `t7_`): they are names, not question IDs.

Files made with the same random seed share their hard processes jet by jet. The hard process is the first step of the
toy: it makes the quark, gluon, W or top that starts the jet. The `t1_*` ladder holds the same events at every preset.
In week 3, `t3_w80_noue`, `t3_w80_centre` and `t3_w80_nocoh` are `t3_w80` with one thing changed
([physics.md](../docs/physics.md#reproducibility)). Colour coherence is the effect that a soft gluon at a wide
angle sees only the total colour of a pair ([the parton shower](../docs/primer.md#the-parton-shower)). Without it, the
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

Every file is read with `load` ([primer section 10](../docs/primer.md#the-toy-and-its-files) explains the columns, the
padding and the chunks):

```python
from itj.io import load

data = load("data/t1_detector.parquet")
data["jet_sdmass"], data["part_px"]  # a per-jet array [N]; a Ragged column (flat values and offsets)
p4, mask = data.head(2000).padded_p4()  # (2000, P, 4) float64 (px, py, pz, E) and (2000, P) bool
for chunk in data.chunks(2000):  # anything that builds (B, N, N) arrays: work on chunks
    ...
```

## W1.0 Set-up (core path about 2 h, stretch about 1.75 h)

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
Silicon and common problems: [docs/install.md](../docs/install.md).

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
from the decay of a W or a top ([primer section 5](../docs/primer.md#what-a-jet-is)). A branch is a column. The `part_*`
columns hold one value per particle and the `jet_*` columns one value per jet. The `truth_*` columns hold what only the
simulation knows, and `label_*` says which process made the jet
([primer section 10](../docs/primer.md#the-toy-and-its-files) explains them). How many jets of each process does the
file hold? Which branches does it have per jet, per particle and as truth? How many particles does the first W jet have?
From here on this file is "the **W1.0.2** file".

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
[physics.md](../docs/physics.md#calibration-and-validation): what is validated, and against what? Read it again after
week 1, when the rows mean something.

**Deliverables:** the summary line of the tests (**W1.0.1**); the number of jets per process, the number of particles of
the first W jet, and the wall time of the week-1 start-of-week box once it has finished (**W1.0.2**);
`W1.0.3_timing.pdf` (**W1.0.3**); two sentences on the validation table (**W1.0.4**).

**Hints**
- **W1.0.1**, troubleshooting. If `jetgen` is not found, the environment is not activated (activate it, or use
  `python -m itj.toy`). If `pytest` is not found, the `dev` extra is missing. About 30 "import file mismatch" errors
  mean a non-editable install (reinstall with `-e`, [install.md](../docs/install.md#common-problems)). On Linux without
  a GPU, install the CPU build of PyTorch first, or pip pulls the multi-GB CUDA build for NVIDIA GPUs. The Windows
  activation line is in [install.md](../docs/install.md#windows). "module 'itj' has no attribute ..." means a submodule
  was not imported (`import itj.toy`, `import itj.observables`).
- **W1.0.2:** jet 0 of the file is a QCD jet, which is why the snippet looks for the first W jet with `label_Wqq`. The
  data script keeps files that exist, and `--dry-run` prints its commands. `particles_frame()` is a pandas table:
  `.to_numpy()` gives a column as an array.
- **W1.0.3:** `--workers auto` uses up to 8 processes, and the output does not depend on the number of workers. In a
  script, `generate(..., workers=4)` needs the `if __name__ == "__main__":` guard (otherwise BrokenProcessPool). Wall
  time, the time on the clock, is not CPU time, the time summed over all cores. `jetgen generate --help` lists every
  option with its default, grouped by stage.

## W1.1 Jets from zero (core path about 7 h, stretch about 4.75 h)

**Goal:** the language of the course, each word tried out on one jet of the toy. Each question names the sections of
[docs/primer.md](../docs/primer.md) to read first. It also names the cell of
[notebooks/00_primer.ipynb](../notebooks/00_primer.ipynb) that shows the tool on another object. The time of both is
inside the question's time. The displayed jets are the W jet of random seed 12, the top jet of random seed 3 and the QCD
jet of random seed 2.

**W1.1.1** *Warm-up, 60 min.* The mass of two particles. Read [primer sections
1-2](../docs/primer.md#units-and-energies) and cells 1-3 of 00_primer. By hand: take two massless particles of 300 GeV
each at an opening angle of 0.27 rad. Compute their invariant mass with the exact formula and with the small-angle form,
and say how much the two differ. In Python: take the W jet of seed 12 at the `partons` level, where it holds only its
two decay quarks. Add their four-vectors and compute the mass of the pair. Compare it with `truth_mass`, the mass the
toy drew for this W (the W mass is not one number; [the primer](../docs/primer.md#the-particles) says why,
and **W1.2.2** returns to it). Then do the same for the top jet of seed 3 at `partons` (three quarks).

```python
import numpy as np

import itj.observables
import itj.toy

jets = itj.toy.show_ladder("wqq", seed=12)  # the W jet of seed 12 at the four presets: one JetData each
p4, mask = jets["partons"].padded_p4()  # four-vectors (px, py, pz, E) and the mask of the real particles
print(p4.shape, mask.shape, jets["partons"]["truth_mass"])
tops = itj.toy.show_ladder("tbqq", seed=3)  # the top jet of seed 3
```

**W1.1.2** *Warm-up, 45 min.* Where is a particle? Read [primer section 3](../docs/primer.md#collider-coordinates). Take
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

**W1.1.3** *Core, 105 min.* Why W and top jets have prongs. Read [primer sections 4-6](../docs/primer.md#the-particles)
and cell 4 of 00_primer. The goal is a formula for the opening angle $\Delta R$ of a boosted two-body decay (a fast
particle that decays into two) with equal momentum sharing. Start from the small-angle mass of **W1.1.1**. Derive the
angle for a W at $\eta = 0$, where $E = p_T$ and the angle between the quarks is $\Delta R$. A boost along the beam does
not change $\Delta R$ ([primer sections 2-3](../docs/primer.md#four-vectors-and-invariant-mass)): use this to carry the
result to every $\eta$. Evaluate it for a W of 600 GeV. The W jets of the **W1.0.2** file have many different $p_T$, so
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

**W1.1.4** *Core, 90 min.* One jet up the ladder. Read [primer sections 7-8](../docs/primer.md#the-parton-shower), the
truth table of [primer section 10](../docs/primer.md#the-toy-and-its-files) and cell 5 of 00_primer. *Predict first:*
how many particles does the W jet have at the `detector` level? Then draw the W jet of seed 12 at the four presets and
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
9-10](../docs/primer.md#what-a-detector-sees). In this toy, why does the number of particles grow by a large factor from
`shower` to `hadrons`, and why does it shrink again at `detector`? Draw the ladder of the QCD jet of seed 2 with the
underlying event coloured. Compare it with the W jet of **W1.1.4**: what is different at `partons`, and what at
`detector`?

```bash
jetgen display --ladder --process qcd --seed 2 --color ue -o W1.1.5_ladder_qcd.png
```

**W1.1.6** *Core, 75 min.* Your first histograms (write your own). Read [primer sections
11-12](../docs/primer.md#histograms-and-errors) and cell 7 of 00_primer. *Predict first:* which of the multiplicity $N$
(the number of particles), the soft-drop mass and $\tau_{21}$ separates W jets from QCD jets best? Which separates top
jets from W jets? Write a small function for a normalised histogram (scaled to unit area) with $\sqrt{n}$ error
bars; you will reuse it all course. With it, draw `jet_nparticles`, `jet_sdmass` and $\tau_{21}$ =
`jet_tau2`/`jet_tau1` for the three processes of the **W1.0.2** file, one panel each. Compare the result with your
prediction in two sentences.

**W1.1.7** *Stretch, 45 min.* Relative coordinates and the JetClass sign. Take the W jet of **W1.1.2**. Compute
$\Delta\eta$ and $\Delta\phi$ of every particle relative to the jet axis, the direction of the whole jet (`jet_eta`,
`jet_phi`, $\Delta\phi$ wrapped), and reproduce `part_deta` and `part_dphi`. Your first attempt will disagree for one of
them: why ([primer section 3](../docs/primer.md#collider-coordinates))? Then take a jet with `jet_eta` above $+1$ from
the **W1.0.2** file and check that the same rule holds. The rule returns in week 4 (**W4.1.4**, **W4.3.2**).

**W1.1.8** *Stretch, 90 min.* A shower by hand. `itj.toy.shower_one` is the readable version of the toy's shower (cell 6
of 00_primer shows one quark shower). Shower a 300 GeV quark and a 300 GeV gluon 2 000 times each from a start angle
of 1. The start angle is the largest angle (in radians) that the first emission may have. This takes about 0.3 s for
4 000 showers. Compare the ratio of the mean numbers of emissions of gluons and quarks, with its error, with $C_A/C_F$,
the ratio of the colour factors of a gluon and a quark. Say why the two need not agree: which emissions does $C_A/C_F$
describe? Histogram $\theta$ and the soft fraction $\min(z, 1 - z)$ of the emissions on log axes, and relate them to the
emission probability of [primer section 7](../docs/primer.md#the-parton-shower). Follow the angles along one chain: do
they shrink?

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
records) and the cell sizes of [physics.md](../docs/physics.md#detector).

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
  paragraph *JetData* of [primer section 10](../docs/primer.md#the-toy-and-its-files) explain this padding. Check your
  result with `itj.observables.mass(p4, mask)`. A difference in the fourth digit comes from storing momenta in float32
  (single precision).
- **W1.1.2:** `np.arctan2` and the wrap to $(-\pi, \pi]$; [physics.md](../docs/physics.md#units-and-coordinates) has the
  formulas.
- **W1.1.3:** `truth_dr_min` is NaN (not a number) for QCD jets. Use `truth_seed_pt`, the $p_T$ of the W itself, not
  `jet_pt`. The jet has lost some radiation and has been through the detector, so its $p_T$ is a little different.
  Equal sharing $z = 1/2$ is a special case of $m^2 \approx z(1 - z)\,p_T^2\,\Delta R^2$: what does unequal sharing do
  to $\Delta R$? Keep this figure: **W1.4.2** and **W3.2.2** come back to it.
- **W1.1.4:** `truth_part_prong` is $-1$ for the underlying event (or an unassigned particle), and `truth_part_is_ue`
  marks the underlying event. The shower level of the same jet answers "what did the shower add". The jets of the
  **W1.0.2** file are other events than the displayed one: use `show_ladder`. For the $p_T$ share, divide by the scalar
  sum of the particle $p_T$. `jet_pt` is the $p_T$ of the summed four-vector, which is smaller.
- **W1.1.5:** see [physics.md](../docs/physics.md#presets) (the shower stops at about 1 GeV, hence "in this toy") and
  [physics.md](../docs/physics.md#detector) (thresholds, cells).
- **W1.1.6:** a density divides by the number of jets and the bin width, and so does its error. Guard against
  $\tau_1 = 0$.
- **W1.1.7:** the JetClass convention is in [physics.md](../docs/physics.md#units-and-coordinates).
- **W1.1.8:** `z` is the momentum fraction of the first daughter (close to 1 for a soft gluon off a quark), hence
  $\min(z, 1 - z)$. See [physics.md](../docs/physics.md#parton-shower) and [its branching
  probability](../docs/physics.md#branching-probability). The cutoff in $k_t$ limits the soft (low-energy) and collinear
  (small-angle) emissions.
- **W1.1.10:** at `partons` the particles of a top jet are its three decay quarks. The small-angle mass of **W1.1.1**
  holds for every pair, with the mass of that pair.

## W1.2 The ladder on many jets (core path about 1.25 h, stretch about 9 h)

**Goal:** what each step of the ladder does to many jets, measured on paired events. Stretch: the emissions inside a jet
as a picture, then the shower against the leading-log formulas (Lund density, Sudakov factor). Leading log is the
approximation behind the emission probability in [the primer](../docs/primer.md#the-parton-shower): only soft,
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
([physics.md](../docs/physics.md#start-angles-and-colour-coherence)). Three short answers with numbers.

**W1.2.3** *Stretch, 30 min.* The emissions of one jet. Draw one QCD jet and one W jet of `data/t1_detector.parquet`
with their primary Lund plane. Every point of this plane is one emission, found by undoing the clustering along the
harder branch ([primer section 7](../docs/primer.md#the-parton-shower)). It sits at its angle $\Delta$ and transverse
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
([physics.md](../docs/physics.md#multiplicity)). If you did **W1.1.8**, you have half of it.

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
(ParT; [models.md](../docs/models.md#particle-transformer)) computes for each pair of particles the pair features
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

([physics.md](../docs/physics.md#sudakov-factor)). Here $E$ is the quark energy, $\theta_0$ the start angle and
$k_{t,\mathrm{cut}}$ the shower cutoff. Double-log means that only the squared-logarithm terms are kept. Where does it
fail, and why? Check your curve with `itj.toy.sudakov_cdf`.

**W1.2.11** *Stretch, 60 min.* Derive it, and the gluons. Derive the Sudakov factor of **W1.2.10**[^sudakov], and repeat
the comparison for the gluon jets of `data/t1_lund_g.parquet`.

**Deliverables:** `W1.2.1_ladder.pdf` (**W1.2.1**); three short answers with numbers (**W1.2.2**); `W1.2.3_lund_qcd.png`
and `W1.2.3_lund_wqq.png` (**W1.2.3**); `W1.2.4_lund_truth.pdf` and the two plateaus with their errors (**W1.2.4**);
`W1.2.5_qg_ratio.pdf` (**W1.2.5**); the `allclose` line (**W1.2.6**); `W1.2.7_lund_reco.pdf` (**W1.2.7**);
`W1.2.10_sudakov.pdf` (**W1.2.10**).

**Hints**
- **W1.2.1:** the four files hold the same events in the same order ([physics.md](../docs/physics.md#reproducibility)).
  The seed parton is the quark or gluon (or the W or top) that the toy starts the jet from ([primer section
  10](../docs/primer.md#the-toy-and-its-files)). Compute the mass by chunks of 2 000 jets, because the padded array of
  30 000 jets does not fit comfortably in memory. Your own mass comes in **W1.3.1**.
- **W1.2.2:** the toy draws the W and top masses from a Breit-Wigner distribution. [Primer section
  4](../docs/primer.md#the-particles) says what that is and why an unstable particle has no single mass.
  [physics.md](../docs/physics.md#hard-process) says how the toy draws it.
- **W1.2.3:** the two figures have different axis ranges, so compare the ranges before the pictures. At `detector` the
  points include splittings between hadrons, which is why some lie below the shower cutoff of about 1 GeV.
  [observables.md](../docs/observables.md#the-lund-plane) explains the plane in words first.
- **W1.2.4**, **W1.2.6:** the error of a histogram density is $\sqrt{n}/(N_\mathrm{jet}\,\mathrm{area})$ per bin
  (`itj.observables.lund_density` returns both).
- **W1.2.7:** check your answer afterwards against [physics.md](../docs/physics.md#lund-density).

## W1.3 The jet mass (core path about 1.5 h, stretch about 5.25 h)

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
distribution of the normalised jet mass (see [Notation](../TASKS.md#notation)). Physicists write $\sigma$ for the rate
of a process (the cross-section), so $d\sigma/d\ln\rho$ is the distribution of the jets in $\ln\rho$, not a standard
deviation. Compare the position of its peak in $\ln(1/\rho)$
with the double-log value and with the value that includes the hard-collinear term. Both are in
[physics.md](../docs/physics.md#jet-mass); this is a two-line check, no derivation needed[^peak]. The peak is broad, so
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
- **W1.3.2:** read [primer section 12](../docs/primer.md#a-first-look-at-the-observables) and
  [observables.md](../docs/observables.md#soft-drop) (the lead *In words*). A standard deviation, or a range between
  quantiles, of the whole distribution is dominated by its tails. Measure the peak itself, then look at the tails
  separately. For (b), think of what soft drop removes first and what a top jet holds.
- **W1.3.4:** use fine bins near the peak, or the median, because a shift of about a GeV hides in bins of 4 GeV. B
  hadrons, the hadrons that carry a b quark, are in [physics.md](../docs/physics.md#hadronisation). A semileptonic decay
  is a decay with a lepton and a neutrino.
- **W1.3.5:** use logarithmic bins in $\rho$ and leave out the jets with $m = 0$ (many fixed-coupling quark jets have a
  single parton inside $R$).

## W1.4 Counting prongs (core path about 1 h, stretch about 6.75 h)

**Goal:** observables that count prongs, ready for the first tagger of week 2.

**W1.4.1** *Warm-up, 30 min.* $\tau_{21}$, $\tau_{32}$ and $D_2$ per process. The first two are N-subjettiness ratios,
the third a ratio of energy correlation functions ([see the primer](../docs/primer.md#a-first-look-at-the-observables)).
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
([observables.md](../docs/observables.md#energy-correlation-functions)). Check them in one `allclose` against
`itj.observables.ecf` on 10 jets.

**W1.4.6** *Stretch, 75 min.* Power counting. Power counting[^power] estimates how each term of an observable scales
with the small momentum fractions and angles in a jet. Use it to explain why $D_2 \gg 1$ for one-prong jets and
$D_2 = O(1)$ for two-prong jets ([physics.md](../docs/physics.md#power-counting)). Then study $D_2$ at $\beta = 2$, and
at the other levels of the ladder.

**W1.4.7** *Stretch, 2 h.* Your own question. A free study with a figure. Some examples: the dead cone of b quarks
(`--dead-cone`), the reduced radiation of a heavy quark at small angles; the W decay angle (`truth_cos_theta_star`);
pull and colour coherence (`--no-coherence`). Two more: the displacement of B-hadron decay products in top jets
(`truth_part_from_b`, `part_d0val`), and angularities for quark and gluon jets
([observables.md](../docs/observables.md#angularities-and-pt_d)).

**Deliverables:** `W1.4.1_prongs.pdf` (**W1.4.1**); `W1.4.2_tau21_vs_dr.pdf` (**W1.4.2**); the counts of the special
cases (**W1.4.3**); the `allclose` lines (**W1.4.4**, **W1.4.5**); a figure of your own (**W1.4.7**).

**Hints**
- **W1.4.1:** read [observables.md](../docs/observables.md#n-subjettiness) and [the energy correlation
  functions](../docs/observables.md#energy-correlation-functions), each with a lead *In words*. Guard against
  $\tau_1 = 0$ (and $\tau_2 = 0$) before taking ratios. Before you read the histograms, decide from the definitions
  whether a jet with a hard two-prong core should give a small or a large $D_2$. Which of the three processes have such
  a core?
- **W1.4.2:** use a 2D histogram, or the median of $\tau_{21}$ in bins of `truth_dr_min`. At the small end, compare the
  separations with the size of the detector cells ([physics.md](../docs/physics.md#detector)).
- **W1.4.3:** write out $\tau_N$ for two particles and two axes.
- **W1.4.4:** WTA-kT axes (winner-take-all axes from kT clustering) from another routine will not pass.
- **W1.4.5:** work on chunks of 2 000 jets for anything that builds $(B, N, N)$ arrays (jets by particles by particles).
  Use `@` (matmul) for $e_3$, not a three-operand `np.einsum`, which needs much more memory and is about 7 times slower.

### Weekly summary

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

- **Next:** [Week 2: Taggers, from cuts to transformers](week2.md)
- **All weeks, the rules and the notation:** [TASKS.md](../TASKS.md#the-weeks)
