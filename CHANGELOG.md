# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed

- `itj.toy.shower.reshuffle` keeps the mass of a system made of one massive particle.
- The planted shortcut of a `--mystery` file is scrambled with a key-derived stream instead of plain base64, and
  `jetgen info --provenance` prints a placeholder for it; files of 0.1.0 still decode.
- `itj.jetclass.load_jetclass("example", part=...)` takes the part within each class: the example file holds the
  classes in blocks, so `part="test"` used to return only Hbb and Hcc (and no jet of the course classes). An empty
  selection raises a ValueError.
- `jetgen fetch` flushes its header before the downloads and, when the output is not a terminal, prints the progress
  once per 25 % instead of an updating line.
- `itj.train.predict` raises a clear ValueError for a model without `feature_set` (pass `feature_set=`), and
  `save_model` and `itj.probes.extract` a TypeError for a model that is not an `itj` Tagger.

### Fixed

- The 30-step weaver commands of T4.6 run with `--lr-scheduler none`: with `--num-epochs 1` weaver's default
  `flat+decay` trained at 1 % of the learning rate and the runs stayed at chance.
- Tasks and docs after the dry runs of both weeks: memory needs (16 GB for T2.4 as written), the CPU-only PyTorch line
  in T0.1, the W jet of T0.2, the data list of T1.4, the planing weights outside the mass range (T3.5), what
  `min_steps` and `patience` do in T2.6, the hand-off of the own PFN and EFN of T2.4 to T3, the times of T1.3 and T4.6,
  the layout of a local JetClass share and the answer of T5.4 moved from jetclass.md to methods.md.

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

[0.1.0]: https://github.com/hqucms/inside-the-jet/releases/tag/v0.1.0
