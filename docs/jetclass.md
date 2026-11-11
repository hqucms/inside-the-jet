# JetClass, ParT and weaver in one page

JetClass is a large open data set of simulated jets for jet tagging: 125 million jets in ten classes, with the
particles of every jet and the substructure branches. It was released with the Particle Transformer (ParT), and both
are trained with the framework weaver. Everything on this page is taken from the paper (H. Qu, C. Li and S. Qian,
[arXiv:2202.03772](https://arxiv.org/abs/2202.03772)), from the repositories
[jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer) (commit `fa63c86`) and
[hqucms/weaver-core](https://github.com/hqucms/weaver-core) (v0.5.4), or measured on the JetClass files. Week 2 of the
course uses it; the toy of this repository writes the same branches, so that the same code runs on both (task T0.4
writes the map between them).

## The data set

Ten classes, one file prefix each; the numbers are measured on the validation file `<prefix>_120` (100 000 jets).

| class | file prefix | label | process | $\langle n\rangle$ | max $n$ | median $m_\mathrm{SD}$ [GeV] | median $\tau_{21}$ | median $\tau_{32}$ |
|---|---|---|---|---|---|---|---|---|
| QCD | ZJetsToNuNu | `label_QCD` | Z(→ νν) + jets: a quark or gluon jet | 35.3 | 145 | 22.6 | 0.61 | 0.76 |
| Hbb | HToBB | `label_Hbb` | H → bb̄ | 42.0 | 136 | 115 | 0.33 | 0.76 |
| Hcc | HToCC | `label_Hcc` | H → cc̄ | 38.4 | 147 | 119 | 0.31 | 0.73 |
| Hgg | HToGG | `label_Hgg` | H → gg | 56.4 | 129 | 117 | 0.46 | 0.76 |
| H4q | HToWW4Q | `label_H4q` | H → WW* → 4q | 48.5 | 122 | 112 | 0.55 | 0.65 |
| Hqql | HToWW2Q1L | `label_Hqql` | H → WW* → ℓνqq̄′ | 28.3 | 105 | 80 | 0.45 | 0.53 |
| Zqq | ZToQQ | `label_Zqq` | Z → qq̄ | 33.8 | 105 | 86 | 0.37 | 0.74 |
| Wqq | WToQQ | `label_Wqq` | W → qq̄′ | 32.2 | 106 | 77 | 0.37 | 0.73 |
| Tbqq | TTBar | `label_Tbqq` | t → bqq̄′ | 50.9 | 126 | 164 | 0.49 | 0.50 |
| Tbl | TTBarLep | `label_Tbl` | t → bℓν | 27.6 | 125 | 112 | 0.22 | 0.68 |

(The QCD $\tau$ ratios leave out the jets with $\tau = 0$.) About 0.01 % of the jets have more than 128 particles. The
particles are about 47 % charged hadrons, 41 % photons, 10.5 % neutral hadrons and below 3 % leptons. JetClass has no
quark-or-gluon truth: the QCD class is a mixture, and every question about the initiator is toy-only. The toy's three
classes are QCD (ZJetsToNuNu), Wqq (WToQQ) and Tbqq (TTBar), with the labels `label_QCD`, `label_Wqq`, `label_Tbqq`.

## How the events were made

The hard processes were generated with MadGraph5_aMC@NLO, showered and hadronised with PYTHIA 8 (including multiple
parton interactions, without pileup), and passed through Delphes with the CMS card. Jets are clustered from the
particle-flow candidates with anti-kT, $R = 0.8$, and kept for $500 < p_T < 1000$ GeV and $|\eta| < 2$. Signal jets are
kept only if all decay products of the resonance lie inside the jet. The substructure branches follow the Delphes
conventions: soft drop with $\beta = 0$, $z_\mathrm{cut} = 0.1$, and N-subjettiness with $\beta = 1$, most likely with
one-pass minimised kT axes ([observables.md](observables.md#jetclass-conventions)).

## Field names

41 branches per jet, in this order (the toy writes the same, [data_format.md](data_format.md#jetclass-branches)):

| group | branches | dtype |
|---|---|---|
| particles (16, jagged, $p_T$-descending) | `part_px`, `part_py`, `part_pz`, `part_energy`, `part_deta`, `part_dphi`, `part_d0val`, `part_d0err`, `part_dzval`, `part_dzerr`, `part_charge` | float32 |
| | `part_isChargedHadron`, `part_isNeutralHadron`, `part_isPhoton`, `part_isElectron`, `part_isMuon` | int32 |
| labels (10) | `label_QCD` | float32 |
| | `label_Hbb`, `label_Hcc`, `label_Hgg`, `label_H4q`, `label_Hqql` | bool |
| | `label_Zqq`, `label_Wqq`, `label_Tbqq`, `label_Tbl` | int32 |
| jet (10) | `jet_pt`, `jet_eta`, `jet_phi`, `jet_energy`, `jet_nparticles`, `jet_sdmass`, `jet_tau1`, `jet_tau2`, `jet_tau3`, `jet_tau4` | float32 |
| auxiliary (5) | `aux_genpart_eta`, `aux_genpart_phi`, `aux_genpart_pid`, `aux_genpart_pt`, `aux_truth_match` | float32 |

Quirks to know: `part_deta` = sign($\eta_\mathrm{jet}$) ($\eta_i - \eta_\mathrm{jet}$) (checked on every particle of a
file), `part_dphi` is not flipped; the labels have mixed dtypes; `jet_nparticles` is float32; `jet_tau*` = 0 for jets
with too few particles; the impact parameters are 0 for neutral particles; `aux_genpart_*` is 0 for QCD and
`aux_truth_match` holds a code per class (QCD 50, Hbb 41, Hcc 42, Hgg 44, H4q 45, Hqql 46, Zqq 31-33, Wqq 21-22,
Tbqq 11-12, Tbl 15-16). [data_format.md](data_format.md#jetclass-oddities) lists what analysis code must survive.

## Splits and file numbering

| split | jets | files | what the course uses |
|---|---|---|---|
| train | 100 M (10 M per class) | `<prefix>_000` to `<prefix>_099`, 100 000 jets each | optional (`jetgen fetch jetclass --split train`: the `_000` file per class) |
| val | 5 M | `<prefix>_120` to `<prefix>_124` | `<prefix>_120` of QCD, Wqq and Tbqq: all of week 2 |
| test | 20 M | separate archive | not used |

The course splits each downloaded validation file as every other file: test = the first 20 000 jets, val = the next
10 000, train = the last 70 000 (`load(path, part=...)`, `itj.jetclass.load_jetclass(part=...)`; `jetgen split` writes
the parts as files for weaver).

## Download

Sizes, smallest first:

| what | size | source |
|---|---|---|
| one class, 100 000 jets, Parquet (JetClassMini) | 86-166 MB | `https://hqu.web.cern.ch/datasets/JetClassMini/{train_1M_parquet,val_1M_parquet}/<prefix>_<NNN>.parquet` |
| all ten classes, 10 000 jets each, ROOT | 136 MB | `https://hqu.web.cern.ch/datasets/JetClass/example/JetClass_example_100k.root` |
| validation archive (5 M jets, ROOT) | 7.59 GB | Zenodo record 6619768, `JetClass_Pythia_val_5M.tar` |
| one of ten training archives | 15.19 GB | Zenodo, `JetClass_Pythia_train_100M_part{0..9}.tar` |
| test archive (20 M jets) | 30.37 GB | Zenodo, `JetClass_Pythia_test_20M.tar` |

The JetClassMini files on the CERN server are byte-identical to the Parquet files made from the Zenodo archives.
Zenodo serves the archives only whole (it does not answer HTTP range requests), so a partial download of an archive
is not possible; the course therefore fetches single files from the mirror:

```bash
jetgen fetch jetclass --split val --classes QCD,Wqq,Tbqq     # -> data/jetclass/val/*_120.parquet (374 MB)
jetgen fetch jetclass --split example                         # -> data/jetclass/example/JetClass_example_100k.root (136 MB)
jetgen fetch models ParT_kin ParT_full ParticleNet_kin         # -> models/pretrained/*.pt (19 MB)
jetgen fetch jetclass --dry-run                                # only print the URLs, sizes and destinations
```

`jetgen fetch` prints the total size first, resumes interrupted downloads where the server allows, checks the size and
the MD5 of every file, and keeps a file that is already there and correct. The three files of week 2:

| file | size [bytes] | MD5 |
|---|---|---|
| `ZJetsToNuNu_120.parquet` | 112 349 663 | `89e0a6881109781fe0f574121e7d6ea2` |
| `WToQQ_120.parquet` | 103 488 757 | `2e8cd393e8f9c3de2abc8bbb4db41013` |
| `TTBar_120.parquet` | 158 634 524 | `c9b6e734e68fe697ce425789d5c7a9ff` |

`itj.jetclass.MANIFEST` holds the sizes and MD5s of all 21 files (ten classes in two splits, and the example).
**Without access to the mirror**, a teacher can put the files on a local share (or a USB stick) and every student runs
`jetgen fetch jetclass --base-url URL` (an `http://` or `file://` URL of the directory, or the environment variable
`ITJ_JETCLASS_URL`); the MD5s check the copies. The ROOT example needs `uproot` and `awkward` (the `analysis` extra).

## Models and numbers

The repository at commit `fa63c86` holds the trained models of the paper (`models/<name>.pt`, state dicts of the
network wrappers), which `itj` loads without weaver ([models.md](models.md#pretrained-models)): ParT and ParticleNet,
each with the `kin`, `kinpid` and `full` inputs.

| model | ten classes, paper | ten classes, 2 000 val jets per class | QCD, W, top: argmax over 10 | QCD, W, top: 3 classes | AUC W vs QCD | AUC top vs QCD |
|---|---|---|---|---|---|---|
| `ParT_full` | 0.861 | 0.8585 | 0.84 | 0.935 | 0.977 | 0.998 |
| `ParT_kin` | | 0.741 | 0.77 | 0.90 | 0.969 | 0.992 |
| `ParticleNet_kin` | | 0.699 | 0.75 | 0.89 | 0.964 | 0.989 |

The last four columns are measured on the test parts of the three downloaded validation files (5 000 jets per class):
"argmax over 10" counts a jet as correct if the largest of the ten outputs is its class, "3 classes" renormalises the
three outputs QCD, Wqq and Tbqq first (`itj.jetclass.probs_to_jets3`), and the AUCs use the one-vs-one scores
([methods.md](methods.md#metrics)). A ten-class model spends probability on the seven classes that the three files do
not contain, so the two accuracies differ.

For scale: small ParT models (18 k to 0.9 M parameters) trained on a few hundred thousand jets of all ten classes
reach about 0.70-0.76 accuracy (the scaling study of the dl4hep notebooks); the full ParT reaches 0.861 with 100 M
jets. Zeroing the pair bias of `ParT_kin` at inference drops its accuracy on top jets (argmax over the ten outputs)
from about 0.91 to 0.11, most jets becoming `Tbl` (after `probs_to_jets3`: from 0.95 to 0.62), while
retraining without the pair bias costs only about 0.015 in the W against QCD AUC
([methods.md](methods.md#pair-feature-ablation)).

## weaver in one page

[weaver](https://github.com/hqucms/weaver-core) trains and evaluates networks on ROOT, Parquet or HDF5 files from two
configuration files: a **data configuration** (YAML: derived variables, inputs, labels, observers) and a **network
configuration** (Python: the model). The course pins `weaver-core==0.5.4` (`pip install -e ".[weaver]"`,
[install.md](install.md#optional-weaver)); it does not depend on torch, so install torch first.

**Data configuration.** `new_variables` defines derived branches with NumPy and awkward expressions; `inputs` lists the
input arrays, each with `length` (pad or truncate), `pad_mode` (`wrap` or `constant`; the mask must be `constant`) and
`vars` as `[name, centre, scale, min, max]` (`null` centre: no transformation, with `preprocess.method: manual`);
`labels` (`type: simple`: one-hot branches, exactly one per jet); `observers` (copied into the prediction output);
`selection` (applied after the files of a fetch are read and concatenated). The inputs reach `model.forward` in the YAML
order. `jetgen configs` prints the paths of the six packaged files
([data_format.md](data_format.md#split-files-for-weaver)).

**Network configuration.** A Python file with `get_model(data_config, **kwargs)` returning `(model, model_info)`, and
optionally `get_loss`. `-o key value` (repeatable) passes keyword arguments, evaluated as Python literals: quote lists,
`-o embed_dims "[64,256,64]"`. The example wrappers of the ParT repository are in
[examples/weaver/networks](../examples/weaver/README.md).

**Options** that matter on a laptop:

| option | meaning |
|---|---|
| `--data-train`, `--data-val`, `--data-test` | input files; `name:path` makes a named group (one per class or per file); without `--data-val` weaver uses rows (0, 0.8) of every training file to train and (0.8, 1) to validate |
| `--data-config`, `--network-config`, `-o` | the two configurations and the network options |
| `--model-prefix` | training: the prefix of the saved states (`_best_epoch_state.pt`); testing: the `.pt` file |
| `--gpus ''` | run on the CPU (the default `'0'` asks for a GPU); on Apple Silicon weaver then picks MPS (set `PYTORCH_ENABLE_MPS_FALLBACK=1`) |
| `--num-workers` | data-loading processes: at most the number of files in each named group; `0` works and is not slower on a CPU |
| `--in-memory` | load and preprocess the data once: right for small subsets |
| `--fetch-step 1` | read each Parquet file in one piece; weaver 0.5.4 fails on a partial read of a Parquet file ("could not broadcast input array") with the default 0.01 |
| `--batch-size`, `--start-lr`, `--num-epochs`, `--steps-per-epoch` | the training loop (the default optimiser is `ranger`) |
| `--predict`, `--predict-output` | test mode; a `.parquet` output holds `scores` (softmax, $N \times C$), `_label_` and the observers; a bare file name goes to `predict_output/` next to the model; named test groups give one file each |
| `--data-fraction` | the fraction of each file to read; in test mode the first fraction of every file, in order |
| `--load-model-weights`, `--freeze-model-weights`, `--exclude-model-weights` | start from a checkpoint, freeze or skip layers by regex (fine-tuning, linear probes) |

**Splits.** Never split with a `selection` on an index in the YAML: weaver applies it after concatenating the files of
a fetch, so an index computed per batch of files mixes the files and leaks test jets into training. Use the files of
`jetgen split` instead, one named group per file. Then the test jets of weaver and of `itj` are the same:
`--predict --data-fraction 0.05` on a 20 000-jet test file reads its first 1 000 jets, `load(<file>, max_jets=1000)`.
The commands of task T4.6 are in [examples/weaver/README.md](../examples/weaver/README.md).

## Citation

H. Qu, C. Li and S. Qian, *Particle Transformer for Jet Tagging*, Proceedings of the 39th International Conference on
Machine Learning, PMLR 162, 18281-18292 (2022), [arXiv:2202.03772](https://arxiv.org/abs/2202.03772). The data set:
H. Qu, C. Li and S. Qian, *JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics*, Zenodo (2022),
[doi:10.5281/zenodo.6619768](https://doi.org/10.5281/zenodo.6619768), licence CC BY 4.0. weaver: H. Qu,
[hqucms/weaver-core](https://github.com/hqucms/weaver-core) (MIT).
