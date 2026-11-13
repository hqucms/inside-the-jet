# Data format

Every run of `jetgen generate` (or `save()` in Python) writes one Parquet file with one row per jet. This page describes
format version 1, written by `itj` 0.1.0 and 0.1.1. The columns are those of JetClass, with the same names, dtypes and
meaning, followed by the toy's truth columns, so that one reader (`itj.io.load`), one set of network inputs and one
weaver data configuration work on the toy files and on the JetClass files alike. Below, `N` is the number of jets. The
physics behind the stored quantities is in [physics.md](physics.md), the observables in
[observables.md](observables.md).

```text
file.parquet
├── key-value metadata: itj.format_version "1", itj.config (JSON), itj.provenance (JSON)
├── row groups of 1000 jets (one generation chunk each), ZSTD
├── part_*      16 JetClass particle branches, large_list, pT-descending          [N] x var
├── label_*     10 JetClass labels (mixed dtypes as JetClass)                     [N]
├── jet_*       10 JetClass jet branches                                          [N]
├── aux_*        5 JetClass aux branches                                          [N]
└── truth_*     toy only: per jet, per particle, per prong, per parton, per emission, per Lund emission
```

## Reading the data

In Python, `load` returns the same `JetData` object as `generate`: a column store keyed by the branch names, with a
NumPy array per jet column and a `Ragged` array (flat values and offsets) per jagged column.

```python
from itj.io import load

data = load("data/toy_detector_qcd-wqq-tbqq_n3000_seed1.parquet")
print(data)  # JetData(3 000 jets: qcd 1 002, wqq 999, tbqq 999; preset detector; ... particles; with truth)
data["jet_sdmass"]  # [N] float32
data["part_px"]  # Ragged: all particles of all jets
data["part_px"][7]  # the px of the particles of jet 7, pT-descending
data.process_names  # [N] "qcd", "wqq", "tbqq", "q" or "g"
p4, mask = data.head(100).padded_p4()  # (100, P, 4) float64 (px, py, pz, E) and (100, P) bool, P = longest jet
jet = data.jet(7)  # one jet: jet.jet (scalars), jet.particles, jet.prongs, jet.partons, jet.emissions, jet.lund
jet.particles_frame()  # its particles as a pandas DataFrame
data.config, data.provenance["seed"]  # the configuration and how the file was made
```

`load` also reads JetClass files (Parquet, or ROOT with the `analysis` extra), several files at once (a list or a
glob, concatenated in sorted order with a per-jet `file_index`), a subset of the columns (`columns=[...]`), only the
JetClass columns (`truth=False`), one part of the standard split (`part="train"`, `"val"` or `"test"`, see
[Split files for weaver](#split-files-for-weaver)) and at most `max_jets` jets per file. Large files are best read in
chunks: `for chunk in data.chunks(2000): ...`. `data.to_pandas()` gives a DataFrame of the jet columns
(`level="particle"`: one row per particle).

`jetgen info FILE` prints the counts per process, the mean multiplicity, the preset, the seed and the wall time
(`--columns`: every column with its dtype; `--provenance`: the full provenance). The files are plain Parquet, so any
Parquet reader works:

```python
import pyarrow.parquet as pq

table = pq.read_table("data/toy_detector_qcd-wqq-tbqq_n3000_seed1.parquet")
px = table.column("part_px").combine_chunks()  # pyarrow LargeListArray
values, offsets = px.flatten().to_numpy(), px.offsets.to_numpy()  # flat values [n_particles] and offsets [N + 1]
print(table.schema.field("part_px").metadata)  # {b"units": b"GeV", b"description": b"..."}
```

## Ragged columns and offsets

Jets have different numbers of particles, so the values of a jagged column for all jets are stored one after the
other, and `offsets` (int64, `N + 1` entries, `offsets[0] = 0`) says where each jet starts: the particles of jet `i` are
the rows `offsets[i]:offsets[i + 1]`. All columns of one group (particles, prongs, partons, emissions, Lund emissions)
share their offsets. The same holds in pyarrow, and `Ragged` exposes them:

```python
import numpy as np

px = data["part_px"]  # Ragged
counts = np.diff(px.offsets)  # particles per jet, [N] (also px.counts() or data.counts)
jet_of_particle = np.repeat(np.arange(len(data)), counts)  # jet index of every particle (also px.row_index())
pt = np.hypot(data["part_px"].values, data["part_py"].values)  # per particle, flat
sum_pt = np.add.reduceat(pt, px.offsets[:-1])  # per jet (jets with 0 particles need care: use np.bincount instead)
padded = px.padded(128)  # [N, 128], zero-padded and truncated as the network inputs
```

`data.select(mask)` returns a new `JetData` with the selected jets (a boolean mask or indices), `data.with_columns(...)`
adds columns (your own observables), and `data.drop("truth_")` removes columns by name or prefix.

## JetClass branches

The 41 branches of JetClass, in the JetClass order and with the JetClass dtypes (jagged branches are
`large_list<... not null>` in Arrow, every field `not null`):

| column | dtype | units | content |
|---|---|---|---|
| `part_px`, `part_py`, `part_pz`, `part_energy` | large_list<float32> | GeV | four-momentum, sorted by $p_T$, descending |
| `part_deta` | large_list<float32> | | $s(\eta_i - \eta_\mathrm{jet})$, $s = +1$ if $\eta_\mathrm{jet} \ge 0$ else $-1$ |
| `part_dphi` | large_list<float32> | rad | $\mathrm{wrap}(\phi_i - \phi_\mathrm{jet})$ |
| `part_d0val`, `part_d0err`, `part_dzval`, `part_dzerr` | large_list<float32> | mm | impact parameters and their errors; 0 for neutrals |
| `part_charge` | large_list<float32> | | -1, 0, 1 |
| `part_isChargedHadron`, `part_isNeutralHadron`, `part_isPhoton`, `part_isElectron`, `part_isMuon` | large_list<int32> | | one-hot particle type (all 0 at `partons` and `shower`) |
| `label_QCD` | float32 | | 1 for `qcd`, `q`, `g` |
| `label_Hbb`, `label_Hcc`, `label_Hgg`, `label_H4q`, `label_Hqql` | bool | | always false in the toy |
| `label_Zqq`, `label_Wqq`, `label_Tbqq`, `label_Tbl` | int32 | | `label_Wqq` for `wqq`, `label_Tbqq` for `tbqq` |
| `jet_pt`, `jet_eta`, `jet_phi`, `jet_energy` | float32 | GeV, -, rad, GeV | of the four-vector sum of the constituents |
| `jet_nparticles` | float32 | | number of constituents |
| `jet_sdmass` | float32 | GeV | soft-drop mass (C/A, $\beta = 0$, $z_\mathrm{cut} = 0.1$, $R_0 = 0.8$) |
| `jet_tau1`..`jet_tau4` | float32 | | N-subjettiness ($\beta = 1$, $R_0 = 0.8$; toy: WTA-kT axes; JetClass: probably one-pass minimised kT axes, reproduced to about 1 %); 0 if $N \ge$ the number of particles |
| `aux_genpart_eta`, `aux_genpart_phi`, `aux_genpart_pid`, `aux_genpart_pt` | float32 | | the resonance (W: ±24, top: ±6, sign random); all 0 for QCD |
| `aux_truth_match` | float32 | | 50 QCD, 21 W, 11 top |

The label dtypes are mixed exactly as in JetClass; `data.labels()` and `itj.features.labels(data, "jets3")` return the
class index (QCD 0, W 1, top 2) as int64. [jetclass.md](jetclass.md#field-names) describes the same branches from
the JetClass side.

## Meaning at each preset

| preset | `part_*` holds | particle type and impact parameters |
|---|---|---|
| `partons` | 1 (QCD), 2 (W) or 3 (top) partons | PID flags 0, charge 0, impact parameters 0 |
| `shower` | massless showered partons | PID flags 0, charge 0, impact parameters 0 |
| `hadrons` | hadrons, photons from $\pi^0$, e and μ from B decays (neutrinos dropped) | PID from the species; true $d_0$, $d_z$, errors 0 |
| `detector` | tracks and ECAL, HCAL and forward cells | PID from particle flow; measured $d_0$, $d_z$ with errors; 0 for neutrals |

At `partons` and `shower` all log features of the network inputs stay finite (every parton has $E \ge 1$ GeV), and a
one-particle jet has `jet_sdmass` = 0 and `jet_tau*` = 0.

## Truth branches

Columns that only the toy writes; they all start with `truth_`, so `data.drop("truth_")` or `load(..., truth=False)`
removes them. **Never feed a `truth_*` column to a tagger**: it holds the answer.

| column | dtype | content |
|---|---|---|
| `truth_event_id` | int64 | `first_event_id` + row |
| `truth_process` | int8 | index in `itj.toy.PROCESSES` = ("qcd", "wqq", "tbqq", "q", "g") |
| `truth_init_pid` | int32 | initiator: 1 quark (any flavour), 21 gluon, 24 W, 6 top |
| `truth_nprongs` | int8 | 1, 2, 3 |
| `truth_seed_pt`, `truth_seed_eta`, `truth_seed_phi` | float32 | the seed parton or resonance (lab) [GeV, -, rad] |
| `truth_mass` | float32 | Breit-Wigner mass [GeV] (0 for QCD) |
| `truth_cos_theta_star` | float32 | W decay angle (W: in the W frame; top: the helicity angle); NaN for QCD |
| `truth_dr_min`, `truth_dr_max` | float32 | smallest and largest pairwise $\Delta R$ of the decay quarks; NaN for QCD |
| `truth_n_emissions`, `truth_n_primary` | int16 | accepted branchings; primary ones, summed over the prongs |
| `truth_lnkt_max` | float32 | $\ln(k_t/\mathrm{GeV})$ of the hardest primary emission; NaN if none |
| `truth_prong_px`, `_py`, `_pz`, `_energy`, `_deta`, `_dphi` | large_list<float32> | decay partons (QCD: the initiator); top: 0 = b, 1-2 = W quarks by energy; W: by energy |
| `truth_prong_pid` | large_list<int32> | PDG id (±1..5, 21) |
| `truth_part_prong` | large_list<int8> | prong of each particle, -1 for UE |
| `truth_part_parton` | large_list<int16> | index into `truth_parton_*`, -1 for UE |
| `truth_part_is_ue`, `truth_part_from_b` | large_list<int8> | flags: from the underlying event; from a B-hadron decay |
| `truth_part_purity` | large_list<float32> | energy share of the largest contributor (1 for tracks and below `detector`) |
| `truth_parton_px`, `_py`, `_pz`, `_energy` | large_list<float32> | final partons after the shower (lab) [GeV] |
| `truth_parton_pid`, `truth_parton_prong`, `truth_parton_id` | large_list<int32>, <int8>, <int16> | PDG id, prong and shower node id of each final parton |
| `truth_emis_parent`, `truth_emis_daughter1`, `truth_emis_daughter2` | large_list<int16> | shower node ids |
| `truth_emis_energy`, `truth_emis_z`, `truth_emis_theta`, `truth_emis_phi`, `truth_emis_kt` | large_list<float32> | $E_\mathrm{parent}$ [GeV], $z$ of the first daughter (as in $P(z)$), opening angle, azimuth, $z(1 - z)E\theta$ [GeV]; as generated, before the resonance reshuffle |
| `truth_emis_channel`, `truth_emis_prong`, `truth_emis_primary`, `truth_emis_depth` | large_list<int8> | 0 q → qg, 1 g → gg, 2 g → qq̄; prong; on the harder chain; branchings above it |
| `truth_lund_ln_inv_delta`, `truth_lund_ln_kt`, `truth_lund_z` | large_list<float32> | primary emissions: $\ln(1/\theta)$, $\ln(\min(z, 1 - z)E\theta/\mathrm{GeV})$, $\min(z, 1 - z)$ |
| `truth_lund_prong` | large_list<int8> | prong of the primary emission |

The `truth_part_*` columns are aligned with `part_*` (same offsets); `truth_prong_*`, `truth_parton_*`,
`truth_emis_*` and `truth_lund_*` each have their own offsets. No per-jet column starts with one of these five ragged
prefixes, and every column that does is jagged. `itj.data.COLUMNS` holds the dtype, units, description and group of
every column; the descriptions are also written into the file.

## Exact identities

These hold for every generated file (float32 tolerances, relative $10^{-5}$), and
`itj.toy.check_identities(data)` returns one boolean per line. Jets changed by `itj.augment` are exempt.

```text
partons:   sum of truth_prong_* = resonance four-vector; the leading jet holds every decay parton
W, top:    sum of truth_parton_* of the resonance = resonance four-vector (after the reshuffle)
QCD:       sum of parton energies = seed energy; |sum p_perp| / E < 0.05 (a tolerance, not an identity)
chains:    energy conserved per fragmentation chain (in the W rest frame with singlet_frame); the kT kicks of a chain
           sum to 0 (both before the hadron reshuffle)
hadrons:   with hadron_reshuffle, the hadrons of every restored W and top with all decay products, neutrinos included,
           sum to the parton-level resonance (relative 1e-8; the generator measures it, the file lacks the hadrons
           outside the jet and the neutrinos: provenance["hadron_reshuffle"]["max_residual"])
shower:    emission angles decrease along every branch; with coherence the first angle of every W prong <= theta_qq
jet:       jet_nparticles == number of particles; particles pT-descending; |sum pT vector| == jet_pt
           part_deta == s (eta_i - eta_jet) (abs. 1e-5); |part_dphi| <= pi
labels:    exactly one of the 10 labels set, with the JetClass dtypes
types:     PID flags sum to 1 per particle at hadrons and detector, 0 below; d0 = dz = 0 for neutrals
truth:     -1 <= truth_part_prong <= truth_nprongs - 1; truth_part_prong == -1 iff truth_part_is_ue == 1
```

## Provenance and configuration

`data.config` (the key `itj.config`) is the complete configuration, `SimConfig.to_dict()`, read back by
`SimConfig.from_dict`; its parts `hard`, `shower`, `hadron`, `ue`, `detector` and `jet` hold every option of
[physics.md](physics.md). `data.provenance` (the key `itj.provenance`) says how the file was made:

| key | content |
|---|---|
| `seed`, `first_event_id`, `chunk_size` | random seed, the id of the first jet, jets per chunk (1000) |
| `workers` | worker processes (the output does not depend on it) |
| `preset`, `processes`, `overrides`, `process_overrides` | the CLI preset, the process list, the changed options and the per-process changes (`--override`; absent for `--mystery`) |
| `mystery` | only with `--mystery`: the planted shortcut and the run statistics it changes (`containment_efficiency`, `hadron_kt_capped`, `stage_times_s`, which the file stores as a placeholder), scrambled with a key (`jetgen info --provenance` prints a placeholder, `jetgen info --reveal` decodes it) |
| `containment_efficiency` | per process: the fraction of hard processes that passed containment |
| `hadron_kt_capped` | fraction of hadrons whose transverse kick was capped |
| `hadron_reshuffle` | the hadron-level reshuffle (`hadron_reshuffle`, also through a per-process override): `steps` tried (one per W, two per top), `failed` (systems heavier than the resonance, left unchanged), `failed_w_in_top` (the failed first steps of tops: the top is still restored, its W is not) and `max_residual`, the largest $\max_\mu \lvert\sum p^\mu - P^\mu\rvert / E$ of the restored resonances; zeros when it is off. A merged file of different runs has no provenance, so `check_identities` then skips `hadron_sum` |
| `stage_times_s` | seconds per stage, summed over chunks and workers: `hard`, `shower`, `hadronize`, `ue`, `detector`, `jet_building`, `observables`, `write` |
| `package_version`, `git_commit` | version of `itj`; the commit when run from a git checkout (`-dirty` with local changes) |
| `python_version`, `numpy_version`, `pyarrow_version`, `platform` | software environment |
| `created`, `wall_time_s` | creation time (UTC, ISO) and simulation time in seconds |
| `command_line` | the full command line (from `jetgen`) |

A file made by `jetgen merge` has `provenance["merged"]` = one entry (file, seed, config) per input; its config is
`None` if the inputs had different configurations.

## Merging and sharded generation

`jetgen merge FILE... -o OUT` (or `itj.io.merge(paths, out)`) concatenates files in the given order (`--shuffle
--seed S` shuffles the jets). It refuses inputs with the same seed (their jets would repeat the same random numbers)
and inputs whose `truth_event_id` ranges overlap: give every shard its own seed and its own `--first-event-id` range,
or pass `--renumber`. Inputs with different configurations merge (their configurations go to the provenance).
[examples/cluster](../examples/cluster/README.md) shows the workflow.

## Split files for weaver

All of `itj` uses one split of every file: the first 20 % of the rows are the **test** part, the next 10 % the
**val** part and the rest the **train** part (boundaries `int(0.2 n)` and `int(0.3 n)` for a file of `n` rows),
applied per file. `load(path, part="test")` and `itj.features.split(data)` (per `file_index`) give it in Python;
`itj.jetclass.load_jetclass(part=...)` on JetClass.

weaver cannot apply this split itself: it reads all files of a fetch, concatenates them and only then applies a
`selection`, so an index-based selection in the YAML would mix the files and train on test jets of all but the first
file. `jetgen split` therefore writes the three parts as files, row for row what `load(part=...)` returns, with every
column, the metadata and the row groups:

```bash
jetgen split data/t2_detector.parquet --out-dir data/t2_split      # -> data/t2_split/{train,val,test}/t2_detector.parquet
```

The packaged weaver data configurations (`jetgen configs` prints their paths) have no selection and run unchanged on
toy and JetClass files: `jets3_kin.yaml`, `jets3_kinpid.yaml` and `jets3_full.yaml` (three classes, QCD, W and top)
and the stock ten-class `JetClass_kin.yaml`, `JetClass_kinpid.yaml` and `JetClass_full.yaml`. Their inputs are those of
[models.md](models.md#inputs); [examples/weaver](../examples/weaver/README.md) has the commands.

`src/itj/configs/jets3_kin.yaml` in full (in the expanded block style of the JetClassMini configuration of the dl4hep
notebooks, without its index-based split):

```yaml
# Three classes (QCD, W -> qq', t -> bqq') with the kinematic features of the ParT paper.
# Runs unchanged on the toy files and on the JetClass files. Split them first with `jetgen split` and pass the
# train/val/test files with --data-train/--data-val/--data-test (one named group per file). There is no selection:
# weaver applies it after concatenating the files of a fetch, so an index-based split would mix the files.

new_variables:
   part_mask: ak.ones_like(part_energy)
   part_pt: np.hypot(part_px, part_py)
   part_pt_log: np.log(part_pt)
   part_e_log: np.log(part_energy)
   part_logptrel: np.log(part_pt/jet_pt)
   part_logerel: np.log(part_energy/jet_energy)
   part_deltaR: np.hypot(part_deta, part_dphi)

preprocess:
   method: manual
   data_fraction: 0.5

inputs:
   pf_points:
      length: 128
      pad_mode: wrap
      vars:
         - [part_deta, null]
         - [part_dphi, null]
   pf_features:
      length: 128
      pad_mode: wrap
      vars:
         - [part_pt_log, 1.7, 0.7]
         - [part_e_log, 2.0, 0.7]
         - [part_logptrel, -4.7, 0.7]
         - [part_logerel, -4.7, 0.7]
         - [part_deltaR, 0.2, 4.0]
         - [part_deta, null]
         - [part_dphi, null]
   pf_vectors:
      length: 128
      pad_mode: wrap
      vars:
         - [part_px, null]
         - [part_py, null]
         - [part_pz, null]
         - [part_energy, null]
   pf_mask:
      length: 128
      pad_mode: constant
      vars:
         - [part_mask, null]

labels:
   type: simple
   value: [label_QCD, label_Wqq, label_Tbqq]

observers:
   - jet_pt
   - jet_eta
   - jet_phi
   - jet_energy
   - jet_nparticles
   - jet_sdmass
   - jet_tau1
   - jet_tau2
   - jet_tau3
   - jet_tau4

weights:
```

`jets3_kinpid.yaml` and `jets3_full.yaml` differ only in `pf_features` (the particle types, and for `full` the impact
parameters, with `part_d0: np.tanh(part_d0val)` and `part_dz: np.tanh(part_dzval)` added to `new_variables`), exactly
as `JetClass_kinpid.yaml` and `JetClass_full.yaml`, which are byte-for-byte copies of the ParT repository at commit
`fa63c86`. `itj.features.FEATURE_SETS` holds the same variables in the same order (a test checks all six files).

## Storage details

- Columns: the JetClass columns first, in the JetClass order with the exact JetClass Arrow types, then the truth
  columns in the order of `itj.data.COLUMNS`. Every field has the metadata `units` and `description`.
- Row groups of 1000 jets (one generation chunk each), ZSTD compression. About 2.5 kB per jet with truth and 2.1 kB
  with `--no-truth` at `detector`; JetClass files take about 1.0-1.6 kB per jet.
- A file is written as `<name>.part` and renamed when complete, so an interrupted job never leaves a truncated file
  under the final name. `jetgen generate` refuses to overwrite a file without `--overwrite`.
- `load` refuses toy files without `itj.format_version` or with a newer version than the installed package; files
  without `itj` metadata are read as JetClass (`data.source == "jetclass"`, `data.config is None`).

## JetClass oddities

Analysis code must survive these features of the JetClass files (and the toy reproduces the first four):

- **The sign of `part_deta`** flips with the sign of `jet_eta` (see [JetClass branches](#jetclass-branches)).
- **Mixed label dtypes** (float32, bool, int32): stack the labels and take the argmax, as weaver does.
- **`jet_tau*` = 0 for jets with at most $N$ particles**: guard the ratios $\tau_{21}$ and $\tau_{32}$.
- **`jet_nparticles` is float32.**
- **Particles outside the cone**: 0.5-0.7 % of the JetClass particles have $\Delta R > 0.8$ from the jet axis (with
  $|\Delta\eta|$ up to 3.1): never assume $\Delta R < R$.
- **`jet_energy`** differs from the sum of the particle energies by up to 3.8 % and `jet_eta` from the $\eta$ of that
  sum by up to 0.013 (`jet_pt` and `jet_phi` agree to float32 precision); `itj.augment.recompute` and
  `itj.observables.jet_branches` always use the sum of the constituents.
- **The reco cut** $500 < p_T^\mathrm{jet} < 1000$ GeV and $|\eta_\mathrm{jet}| < 2$: toy files have no such cut;
  apply `itj.features.jetclass_selection(data)` before comparing.
