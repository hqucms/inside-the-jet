# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/hqucms/inside-the-jet/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/hqucms/inside-the-jet/releases/tag/v0.1.1
[0.1.0]: https://github.com/hqucms/inside-the-jet/releases/tag/v0.1.0
