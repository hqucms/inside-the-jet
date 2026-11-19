# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-19

The course is re-paced for undergraduates with no prior knowledge of jets who work on it 2-3 days per week: four weeks
instead of two, every task in warm-up, core and stretch questions, a primer on jets and a ladder for machine learning.
The question IDs are renumbered by week (`W<week>.<task>.<question>`, bonus `B.1` to `B.5`);
[docs/id_mapping_0.2.md](docs/id_mapping_0.2.md) maps every question ID of 0.1.x to its 0.2.0 ID or IDs and back, for
remapping answers, solutions and notes. The toy, its default output and the API of `itj` are unchanged: the 26 data
commands are the same, only grouped by week, so every column of every data file is bit-identical (the provenance
stored in a file records the new package version).

### Added

- `docs/primer.md`, *Jets from zero*, the first unit of the course: units, four-vectors and invariant mass, collider
  coordinates, the particles, what a jet is, boosted W and top jets, the parton shower, hadronisation and the
  underlying event, what a detector sees, the toy and its files, histograms and errors and a first look at the
  observables, each section with check-yourself questions. Task W1.1 holds its ten guided exercises on the toy (six
  on the core path), each naming the primer sections to read just before it; `notebooks/00_primer.ipynb` works
  through the same tools on other jets than the exercises.
- `docs/ml_primer.md`, *Machine learning from zero*, the ladder of weeks 2-4 in eleven sections (a tagger as a
  function, train, validation and test, trees and boosting, neural networks, PyTorch in one page, networks on
  particles, embeddings, regression and probes, learning without labels, domain shift, gradients of the input), each
  read just before the question that uses it, with check-yourself questions; and `notebooks/02_first_network.ipynb`, a
  first neural network in two parts on 6 000 generated jets, under a minute of CPU (W2.3.1 tensors, gradients and a
  logistic regression by hand; W2.3.2 the same as an `nn.Module`, an MLP, its training loop, overfitting and the
  learning rate); a one-tree warm-up before the boosted decision trees (W2.2.1).
- Three figures of the primer, made by `examples/make_doc_figures.py` (`--only primer_coordinates`, `primer_boosted`,
  `primer_lund`): the collider coordinates, a boosted W decay and the primary Lund plane of one jet.
- `docs/id_mapping_0.2.md`: the ID scheme, the mapping from 0.1.x to 0.2.0 (split questions part by part) and the
  reverse index.
- Per week, in its file `tasks/week<k>.md`: why this week, the thread, the concepts in order of use, the week at a
  glance, a *Start of week* box, a day plan for two core days (or three shorter ones), a check-in at the end of day 1,
  fallbacks for a student who is behind (with the time saved and what is lost), the first stretch picks and a weekly
  summary; a realistic time for every question, its reading included; a hand-in by ID in `answers/week<k>.md`.
- Fifteen new or partly new questions, the steps of the ladders: W1.1.1-W1.1.3, W1.1.6-W1.1.10, W1.2.3, W2.1.1,
  W2.2.1, W2.3.1, W2.3.2, W2.4.1 and W4.1.1.
- *In words* leads (two or three sentences before the formulas) in observables.md (soft drop, N-subjettiness, energy
  correlation functions, the Lund plane, IRC safety, numerical IRC tests) and methods.md (probes, linear CKA,
  attributions, attention, mass sculpting, contrastive learning, linear evaluation); the shape of the pretraining loop
  in methods.md; "New to jets?" pointers to the primer at the top of physics.md, observables.md and data_format.md, and
  pointers to the machine-learning primer in models.md and methods.md.
- Glossary entries for the basic terms of jet physics and machine learning, each with its primer section.
- Tests: the ID inventory, tier order and times of the week files, the weekly hour guard, the order of the data blocks,
  figure names, the mapping table, the IDs cited in the docs and code, the data commands against a golden list of
  0.1.1, the primer and the four notebooks, and that no page the primer links to before W1.1.3 states its result.

### Changed

- TASKS.md in four weeks: *What is a jet?* (the set-up, jets from zero, the ladder of the toy on many jets, the jet
  mass, counting prongs); *Taggers, from cuts to transformers*; *What do taggers learn?* (representations and probes,
  what the taggers use, IRC safety and symmetries, learning without labels); *Real jets, and where does a tagger
  look?* (JetClass, the pretrained ParT, IRC tests of networks, attributions on the toy, ending on the real ParT). IRC
  safety moves from week 1 to week 3, next to the augmentations that use the same functions; JetClass and
  interpretability share week 4, with every tool checked on the toy before it is used on the real ParT; the JetClass
  dictionary moves from the set-up to week 4. The core path is about 13-14 hours a week (13.75, 12.75, 13.5 and 12.75
  h, reading and writing up included: two days of about 6.5-7 hours or three shorter days, with fallbacks marked
  *two-day* for a week of two shorter days); about 98.5 hours of stretch and 24.5 hours of bonus are on offer.
- TASKS.md is split into an entry page (the rules, the hand-in, the figure checklist, the notation and a table of the
  weeks) and one file per week, `tasks/week1.md` to `tasks/week4.md` and `tasks/bonus.md`, each linking back to it.
- Week 2 trains the five course models (PFN, EFN, PN-lite, ParT-small and ParT-small without the pair bias) with one
  command in its *Start of week* box (`examples/train_tagger.py --course-models --out models/course --threads N`), one
  after the other in the background; they replace the two ParT-small trainings side by side (13 GB together; the
  sequential run peaks at about 9 GB) and are the models of weeks 3 and 4, so a bug in a student's own network cannot
  spoil a later week.
- One "write your own" step per week stays on the core path (the jet mass and a density histogram, a training loop and
  a PFN, the NT-Xent loss, integrated gradients); the others are stretch questions, with the `itj` reference as the
  path forward. The leading-log theory, power counting, Sudakov safety, the extra studies and every proposal are
  stretch questions.
- The JetClass downloads start at the start of week 3 (`python examples/make_task_data.py --week 4 --only fetch`), a
  week before they are needed; the weaver install moves to its question (W4.2.7).
- `examples/make_task_data.py`: `--week 1|2|3|4|all`, one data block per week file (`## Data for week 1` to `4`);
  `block_for(path)` names the block that makes a file, and a missing input names the week's block, as does
  `examples/train_tagger.py`.
- After a dry run of week 1 by a reader without a background in jets: the displayed W jet is that of seed 12, which
  shows a shower step and underlying-event particles at both levels (the jet of seed 3 had no emission); W1.1.3
  compares the opening angle jet by jet and leaves the closest pair of a top to W1.1.10; W1.3.2 names its measures of
  width; W3.3.2 classifies $\tau_2$ instead of $\tau_{21}$, whose Sudakov safety stays in W3.3.4; the IRC
  classification of observables.md sits in a box to open after W3.3.2, and physics.md, methods.md, the glossary and
  the week files no longer state results that a question asks for; the primer explains the width of an unstable particle
  (W1.2.2) and gives the toy's ranges of $p_T$ and of underlying-event particles; `boost` of `itj.toy.kinematics` is
  described with its sign.
- Times: realistic student hours per question and per week, reading included; measured compute times for the
  JetClass PN-lite, `attention_stats`, the `kinpid` PFN, the probes of four models, the pretraining of
  `partsmall_nopair`, the learning curve, the filters, the probes on the real ParT, the label budgets, the inherited
  invariances and the AUC matrix of the transfer.
- README.md (a section on the course: the four weeks, their hours and data, the primers, the tiers, how to plan a week
  of 2-3 days), the docs, the notebooks, the examples, docstrings and help strings cite the new IDs and weeks.
  install.md gives the data per week, the memory of the week-2 box and where to start JupyterLab; jetclass.md says
  that JetClass is first needed in week 4; the reading list of references.md is grouped by week, with a reading to
  start with in each week (two entries added: Salam, *Towards Jetography*; Guest, Cranmer and Whiteson, *Deep Learning
  and its Application to LHC Physics*).
- Version 0.2.0 (`itj.__version__`, CITATION.cff).

### Fixed

- Question wording, after the reference solutions of 0.1.1: the mass-sculpting JSD compares the jets that pass and
  fail the cut, with a random cut of the same size as the floor; the decorrelated tagger is expected to decorrelate in
  part; the non-invariance of CKA is shown with a whitening; the collinear split for the EFN check acts on the hardest
  particle; the transfer table gives the top rejection at a signal efficiency of 0.8; weaver 0.5.4 has no seed option;
  250 W and 250 top jets for the attribution shares; "the largest ln kt among the primary declusterings"; saliency
  cannot see a flag shortcut; the inherited invariances fold the scaler and the logistic regression into one linear
  layer; the NT-Xent loss flattens well above its floor; the opening angle of the W is an explicit probe target; the
  pretraining file is read without its labels; the observable baseline of the probes is linear and contains the
  soft-drop mass; the EFN filters are rings.
- Docs: the $\tau_N$ agreement of each axis choice with JetClass is in observables.md (the pointer between
  jetclass.md, observables.md and TASKS.md was circular); `jet_eta` of JetClass differs from the constituent sum by up
  to about 0.03, and `jetclass_selection` applies the $p_T$ cut only (data_format.md); the integrated-gradient
  completeness gaps are realistic, ParticleNet's stops at about 2 % (methods.md); ParT's $m^2$ clamp is reached from
  $\delta \approx 10^{-3}$ by float32 rounding and its pair features are not clipped (observables.md); imputing all four
  pair channels at their mean is $U = 0$ (methods.md); `import file mismatch` with a non-editable install
  (install.md); hadronisation and the detector on the reconstructed Lund density, and the power counting of $D_2$ for
  two-prong jets (physics.md); `sculpting_jsd` compares pass and fail (methods.md); the glossary's first uses of the
  underlying event and of anti-kT.

### Removed

- The "Minimum path" list (replaced by the tiers) and the assumption of a PyTorch tutorial before the course.

## [0.1.1] - 2026-11-13

The course data were made at commit 3433738, which still reported version 0.1.0 but already held the entries of
each list below up to the `tabulate` extra (Added), the `predict` and `save_model` errors (Changed) and the fixes after
the dry runs (Fixed); the rest is new in 0.1.1.

### Added

- Two experimental hadronisation switches for W and top jets, both off by default and not calibrated
  ([physics.md](docs/physics.md#hadron-level-reshuffle)): `hadron_reshuffle` restores the parton-level four-momentum
  of W and top after the fragmentation chains (the Herwig reshuffle of the shower applied to the hadrons of each
  prong; for a top the W inside it first), and `singlet_frame` runs the chains of the W quarks in the W rest frame.
  The identity `hadron_sum` of `check_identities` and `provenance["hadron_reshuffle"]` (with `failed_w_in_top`)
  record the reshuffle. With both off the columns of the generated files are bit-identical to 0.1.0 (the stored
  configuration gains the two switches and the provenance the reshuffle record), so the calibration, the validation
  table, the planted shortcuts and the figures are unchanged. With `hadron_reshuffle` on, the rows `msd_median_w`
  (66.2 GeV) and `msd_window_w` (0.53) of the validation fail, and `msd_median_top` (157.0 GeV against at least 156)
  and `tau21_w` (0.438 with WTA-kT axes, 0.445 with one-pass kT axes, against at most 0.44) are borderline; with
  `singlet_frame` on, `n_w` (25.3) fails. The calibration knobs cannot recalibrate either
  switch; `kt_sigma` is the parameter that matters for the reshuffle.
- `itj.metrics.RESULT_COLUMNS` has the bootstrap errors `auc_w_err`, `r50_w_err`, `auc_top_err`, `r50_top_err` and
  `r80_top_err`; `examples/train_tagger.py` fills them.
- `itj.plot.efn_filters` draws a colour bar per filter and marks dead (constant) filters.
- The `weaver` extra installs `tabulate` (weaver's FLOP count printed an ERROR line without it).
- Masked particle modelling that can be learned (T7.1): `itj.ssl.MaskedParticleDecoder` adds a learned embedding of
  the pT rank of every slot to the encoder output before a small head. Without it every masked slot of a jet got the
  same input and the same output, and a regression could only learn the mean of the hidden particles. With
  `num_layers > 0` it is the decoder of a masked autoencoder (a learned mask token and transformer layers over the
  slots), for `itj.ssl.mask_particles(..., drop=True)`, which also removes the masked particles from the particle mask
  so that the encoder sees only the visible ones.
- `itj.train.fit(..., val_every=...)` validates every so many optimiser steps instead of after every epoch, so that
  `patience` (which counts validations) becomes a number of steps for small training sets; with `val_every` an epoch
  line is printed only at a validation. `History.steps` records the optimiser steps at the end of every epoch.
- `itj.interpret.attention_stats` returns the mean self-attention `A_ii` per block and head (`"self"`).
- `itj.toy.mystery.plant` makes the provenance of a `--mystery` file from that of its run (`RUN_STATISTICS` lists the
  entries it hides), and `itj.toy.mystery.encode` takes these hidden `statistics`.

### Changed

- `itj.toy.shower.reshuffle` keeps the mass of a system made of one massive particle.
- The planted shortcut of a `--mystery` file is scrambled with a key-derived stream instead of plain base64, and
  `jetgen info --provenance` prints a placeholder for it; files written before the scrambling still decode.
- `itj.jetclass.load_jetclass("example", part=...)` takes the part within each class: the example file holds the
  classes in blocks, so `part="test"` used to return only Hbb and Hcc (and no jet of the course classes). An empty
  selection raises a ValueError.
- `jetgen fetch` flushes its header before the downloads and, when the output is not a terminal, prints the progress
  once per 25 % instead of an updating line.
- `itj.train.predict` raises a clear ValueError for a model without `feature_set` (pass `feature_set=`), and
  `save_model` and `itj.probes.extract` a TypeError for a model that is not an `itj` Tagger.
- `itj.augment.split_collinear` returns the new four-momenta, `part_deta`, `part_dphi` and the recomputed jet columns in
  float64: rounded to float32 piece by piece, the pieces of a TeV particle got a spurious mass of order 0.1 GeV, which
  left a floor of about 1e-3 under the collinear scan of m_SD at lambda = 0.1 and 0.2 (T1.6). With float64 the floor
  at lambda = 0.2 is gone; at lambda <= z_cut = 0.1 a few jets groomed down to one massive hadron still lose the softer
  piece, which is soft drop and not rounding. `float64=False` keeps the old behaviour; `irc_scan` and `JetCLRAugment`
  use it, so their results are unchanged, and `itj.io.save` still writes float32.
- `itj.interpret.attention_stats` no longer counts self-attention as attention within the group: `within_group` is the
  share of the attention on the other particles that stays in the group of the query, and `random_expectation` is
  `(n_same - 1) / (N - 1)`. A head that mostly attends to the particle itself used to look like a prong head.
- A `--mystery` file stores `containment_efficiency`, `hadron_kt_capped` and `stage_times_s` only scrambled with the
  shortcut (the visible provenance holds a placeholder, `jetgen info --reveal` prints them): `hard_w` and `wide_top`
  could be read off `jetgen info --provenance` without any analysis.
- `jetgen validate` (and `itj.toy.validate`) generates the two Lund samples with at least 10 000 jets (`N_LUND_MIN`,
  about a second each), so that `jetgen validate -n 3000` no longer fails `mass_peak_q`.
- `itj.train.fit` refuses `patience` or `val_every` below 1, and its epoch messages give the optimiser step.

### Fixed

- The 30-step weaver commands of T4.6 run with `--lr-scheduler none`: with `--num-epochs 1` weaver's default
  `flat+decay` trained at 1 % of the learning rate and the runs stayed at chance.
- Tasks and docs after the dry runs of both weeks: memory needs (16 GB for T2.4 as written), the CPU-only PyTorch line
  in T0.1, the W jet of T0.2, the data list of T1.4, the planing weights outside the mass range (T3.5), what
  `min_steps` and `patience` do in T2.6, the hand-off of the own PFN and EFN of T2.4 to T3, the times of T1.3 and T4.6,
  the layout of a local JetClass share and the answer of T5.4 moved from jetclass.md to methods.md.
- The GPU weaver commands of T4.6(b) (`examples/weaver`, `examples/cluster`) give `--steps-per-epoch 585` and
  `--steps-per-epoch-val 58`: weaver 0.5.4 refuses `--in-memory` without them. The T4 hint and the weaver options of
  jetclass.md say so.
- Tasks and docs: T7.1 uses the new masked particle modelling with its data size and trivial predictors, and evaluates
  the class token and the mean of the last particle block; T1.6 scans the collinear splits at lambda = 0.5 and 0.2,
  above z_cut, and observables.md explains the jets that change below it; T2.6 and T6.5 validate every 50 steps (T6.5
  every 10 steps at 100 and 300 labels), T2.6 no longer presupposes that the curves cross, and T6.5 counts the
  validation labels in the label budget, explains its T2.6 reference curve and gives a measured time; the few-label
  protocol of
  methods.md no longer claims that pretraining pays off where labels are few; T2.3 no longer presupposes that the BDT
  beats the MLP; the planing weights of T3.5 have a defined scale; T3.4 standardises X_phys for CKA; T3.6 compares
  `t3_w80_nocoh` by its distributions, since only its hard processes are paired; T6.4 adds a BDT on the observables next
  to the linear model; T7.3 compares the block probes with arXiv:2605.09881 and only the last-block attention with
  arXiv:2412.03673; T5.3 and methods.md describe the self-attention split of `attention_stats`.

## [0.1.0] - 2026-11-05

Stage A: the toy "A Spray of Hadrons".

### Added

- `itj.toy`: QCD (also pure quark and gluon), W and top jets from a hard process with Breit-Wigner masses and
  W helicity, a coherent veto-algorithm parton shower with an exact reshuffle and a readable scalar reference
  (`shower_one`), independent fragmentation with pi0 and B-hadron decays, the underlying event and a detector with
  tracks, impact parameters, ECAL and HCAL cells and particle flow; the ladder presets `partons`, `shower`,
  `hadrons`, `detector`, paired samples for "one thing changed" questions and planted shortcuts for the bonus task;
  the `jetgen` command line.
- Parquet data format with the 41 JetClass branches (identical names, types and meaning) and a `truth_*` record,
  configuration and provenance; `merge`, and `write_split` (`jetgen split`) to give weaver the split of the course.
- A wider `itj` package than the earlier toys, because week 2 runs the same analysis on JetClass:
  `itj.data`, `itj.io`, `itj.jetclass`, `itj.features`, `itj.cluster`, `itj.observables`, `itj.augment`,
  `itj.models`, `itj.train`, `itj.metrics`, `itj.probes`, `itj.interpret`, `itj.ssl` and `itj.plot`.
- JetClass downloads from the CERN mirror with pinned sizes and MD5s, and the pretrained ParT and ParticleNet
  checkpoints of the JetClass paper; network inputs identical to weaver 0.5.4 and the packaged weaver data configs
  (`jets3_*.yaml`, `JetClass_*.yaml`).
- `NOTICE.md` for the third-party code, configs and data (weaver-core, particle_transformer, JetClass).
- `tests/data/`: 150 JetClass validation jets (CC BY 4.0) and the weaver scores of the pretrained models on them.
- `examples/weaver/` (data configs in use, network files and single-line weaver commands), `train_tagger.py` and
  `make_doc_figures.py`, and the second notebook `01_jetclass.ipynb`.
- Four documents beyond the earlier toys: `observables.md`, `models.md`, `methods.md` and `glossary.md`.
- CI jobs `extras` (weaver, fastjet and a weaver smoke run on toy split files) and `network` (weekly check of every
  download), and the pytest markers `weaver` and `network`.
- Documentation and student tasks T0-T7.

[Unreleased]: https://github.com/hqucms/inside-the-jet/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/hqucms/inside-the-jet/releases/tag/v0.2.0
[0.1.1]: https://github.com/hqucms/inside-the-jet/releases/tag/v0.1.1
[0.1.0]: https://github.com/hqucms/inside-the-jet/releases/tag/v0.1.0
