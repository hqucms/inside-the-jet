# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
