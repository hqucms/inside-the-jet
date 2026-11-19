# Question IDs: 0.1.x to 0.2.0

Release 0.2.0 re-paces the course over four weeks and renumbers its questions. This page maps every question ID of 0.1.x
to the question or questions of 0.2.0 that hold it, and back. It holds no answers. The reference solutions, the
grading notes, the course fixes and the result keys are remapped with it; until then they keep the 0.1.x IDs. The
questions themselves are in one file per week: an ID `W1.*` is in [tasks/week1.md](../tasks/week1.md), `W2.*` in
[tasks/week2.md](../tasks/week2.md), `W3.*` in [tasks/week3.md](../tasks/week3.md), `W4.*` in
[tasks/week4.md](../tasks/week4.md) and `B.*` in [tasks/bonus.md](../tasks/bonus.md). The rules and the notation are
in [TASKS.md](../TASKS.md).

## The ID scheme

- **0.2.0:** `W<week>.<task>.<question>`, for example `W2.3.3` (week 2, task 3, question 3). Weeks 1-4; tasks are
  numbered from 1 in every week (week 1 also has task 0, the set-up); questions from 1 in every task, warm-up first,
  then core, then stretch. A task is `W3.4`, a week "week 3". The bonus questions are `B.1` to `B.5`.
- **The tier is not part of the ID.** The week files print it next to every ID (*Warm-up*, *Core*, *Stretch*). A later
  move of a question across the core/stretch line therefore keeps its ID; any other move will be recorded here.
- **0.1.x:** `T<task>.<question>` (`T0.1` to `T7.4`, 49 questions). The `W` prefix with three numbers cannot be read as
  an old ID, so both can stand side by side.
- **File names.** Figures are named by the new ID (`T2.5_roc_all.pdf` becomes `W2.5.2_roc_all.pdf`). The data files keep
  their 0.1.x stems (`t1_`, `t2_`, `t3_`, `t6_`, `t7_`): they are names, not question IDs. The results table keeps its
  columns; its `note` column may hold the ID.

## From 0.1.x to 0.2.0

Every one of the 49 question IDs of 0.1.x maps; where a question was split, each part has its own row (the part is
named in the wording of 0.1.1). The last row lists the questions with no 0.1.x source.

| 0.1.x | part | 0.2.0 | tier |
|---|---|---|---|
| `T0.1` | the install and the tests | `W1.0.1` | warm-up |
| `T0.1` | the first data set (`jetgen generate -n 3000 --seed 1`) | `W1.0.2` | core |
| `T0.2` | `jetgen info`; the Python look at one jet | `W1.0.2` | core |
| `T0.2` | the ladder display, *Predict first*, (i)-(iv) | `W1.1.4` | core |
| `T0.2` | (v) why $N$ grows and shrinks | `W1.1.5` | core |
| `T0.3` | timing per preset, `--workers 4`, the plot, the dominant stage, the extrapolation | `W1.0.3` | stretch |
| `T0.3` | *Optional* `jetgen validate` | `W1.0.4` | stretch |
| `T0.4` | starting the downloads | week-3 *Start of week* box (`--week 4 --only fetch`), no ID |  |
| `T0.4` | the weaver install | `W4.2.7` | stretch |
| `T0.4` | the sign of `part_deta` at $\eta = -1.2$ | `W1.1.7` (toy); `W4.1.4` (JetClass) | stretch; stretch |
| `T0.4` | the dictionary of the columns in use | `W4.1.1` | warm-up |
| `T0.4` | the full dictionary, trivially filled columns, `truth_*` columns, the two most useful truth columns | `W4.1.6` | stretch |
| `T1.1` | the plots of $N$, $m_J$ and $p_{T,J}/p_{T,\mathrm{seed}}$ per preset and process | `W1.2.1` (mass previewed in `W1.1.1`) | warm-up |
| `T1.1` | (a) the masses at `partons`; (b); (c) | `W1.2.2` | core |
| `T1.1` | (a) the containment check | `W1.2.9` | stretch |
| `T1.1` | (d) $\langle N_g\rangle/\langle N_q\rangle$ | `W1.2.5` (previewed in `W1.1.8`) | stretch |
| `T1.2` | (a) | `W1.2.10` | stretch |
| `T1.2` | (b) *Optional* derivation and gluons | `W1.2.11` | stretch |
| `T1.3` | (a), (b) with the library density | `W1.2.4` | stretch |
| `T1.3` | "write your own density" | `W1.2.6` | stretch |
| `T1.3` | (c) | `W1.2.7` | stretch |
| `T1.3` | *Optional* other levels; the W plane | `W1.2.8` (the W plane previewed in `W1.2.3`) | stretch |
| `T1.4` | own mass | `W1.3.1` | warm-up |
| `T1.4` | (b) $m$ and $m_\mathrm{SD}$ per process | `W1.3.2` | core |
| `T1.4` | (a) the underlying event alone; (b) the shift without the underlying event | `W1.3.3` | stretch |
| `T1.4` | (c) the top $m_\mathrm{SD}$ peak against the top mass; the widths against the Breit-Wigner width | `W1.3.4` | stretch |
| `T1.4` | (a) the fixed-coupling peak and its stability | `W1.3.5` | stretch |
| `T1.4` | (a) the shifts along the paired quark/gluon ladder | `W1.3.6` | stretch |
| `T1.5` | (iii) $\tau_{21}$, $\tau_{32}$, $D_2$ per process | `W1.4.1` | warm-up |
| `T1.5` | (vi) $\tau_{21}$ against `truth_dr_min` | `W1.4.2` | core |
| `T1.5` | (iv) degenerate at `partons` | `W1.4.3` | stretch |
| `T1.5` | (i) own $\tau_N$ | `W1.4.4` | stretch |
| `T1.5` | (ii) own $e_2$, $e_3$, $D_2$ | `W1.4.5` | stretch |
| `T1.5` | (v) power counting; *Optional* $\beta = 2$, other levels | `W1.4.6` | stretch |
| `T1.6` | the soft scan of $m$ and $N$ | `W3.3.1` | warm-up |
| `T1.6` | soft and collinear scans, the classification safe/unsafe (five observables) | `W3.3.2` | core |
| `T1.6` | all nine observables, Sudakov safety, why single jets cannot tell | `W3.3.4` | stretch |
| `T1.6` | *Optional* two-particle jet, $m_\mathrm{SD}$ at $\lambda = 0.05$ | `W3.3.5` | stretch |
| `T1.7` | proposal | `W1.4.7` | stretch |
| `T2.1` | one window and its efficiencies (new steps added) | `W2.1.1` | warm-up |
| `T2.1` | the cut taggers, ROC, AUC, $R_{50}$ on test | `W2.1.2` | core |
| `T2.1` | window optimisation on val; *Predict first* $D_2$ or $\tau_{21}$ | `W2.1.3` | stretch |
| `T2.2` | the BDT on the 12 features (after the new W2.2.1) | `W2.2.2` | core |
| `T2.2` | permutation importance; `jet_phi` | `W2.2.3` | stretch |
| `T2.2` | own $D_2$ added; $N$; the danger on real data | `W2.2.4` | stretch |
| `T2.3` | own loop, standardisation, early stopping, the losses, against the BDT (after the new W2.3.1 and W2.3.2) | `W2.3.3` | core |
| `T2.3` | why a BDT can beat an MLP | `W2.3.4` | stretch |
| `T2.4` | start the two ParT-small trainings; the deliverables `pfn.pt`, `efn.pt` | week-2 *Start of week* box (`--course-models`), used from `W2.4.1` on |  |
| `T2.4` | own PFN, permutation and padding invariance, the key map | `W2.4.2` | core |
| `T2.4` | training the own PFN, the copy-back, `models/toy/pfn.pt` | `W2.4.3` | stretch |
| `T2.4` | own EFN, exact collinear safety, the copy-back; what the PFN sees that the EFN cannot | `W2.4.4` | stretch |
| `T2.5` | train PN-lite; collect the ParT-small checkpoints | week-2 *Start of week* box |  |
| `T2.5` | reading models.md; *Predict first* | `W2.5.1` | warm-up |
| `T2.5` | the ROC curves of all taggers; the cost table (parameters, training time) | `W2.5.2` | core |
| `T2.5` | the inference speeds of the cost table | `W2.5.4` | stretch |
| `T2.5` | *Optional (GPU)* `partlite`, more data | `W2.5.5` | stretch |
| `T2.6` | the BDT and PFN curves with the rule | `W2.5.3` | stretch |
| `T2.6` | seeds, the ParT-small point, the time axis, the crossing | `W2.5.4` | stretch |
| `T2.7` | proposals | `W2.5.6` | stretch |
| `T3.1` | one model, PCA by class and $m_\mathrm{SD}$ | `W3.1.1` | warm-up |
| `T3.1` | all models, t-SNE, UMAP, all colourings | `W3.1.3` | stretch |
| `T3.2` | four models x ($m_\mathrm{SD}$, $N$) with controls and selectivity; *Predict first* (ParT-small) | `W3.1.2` | core |
| `T3.2` | gluon/quark, $\tau_{21}$, $D_2$, `truth_n_primary`; *Optional* probes (MLP, PN-lite, truth targets, every block) | `W3.1.4` | stretch |
| `T3.2` | the $X_\mathrm{IRC}$ baseline; *Predict first* (EFN) | `W3.3.8` | stretch |
| `T3.3` | train, draw, relate to the prong separation | `W3.2.2` | core |
| `T3.3` | $2m/p_T$ for W jets against `truth_dr_min`, jet by jet | `W1.1.3` | core |
| `T3.3` | the top: the $p_T$ of the W inside it, the closest pair of its quarks | `W1.1.10` | stretch |
| `T3.3` | a second seed; Fig. 6 of the paper | `W3.2.7` | stretch |
| `T3.4` | own linear CKA, the invariances | `W3.1.5` | stretch |
| `T3.4` | (a), (b), *Optional* | `W3.1.6` | stretch |
| `T3.5` | the cut at $\epsilon_S = 50\,\%$, the QCD mass, the JSD | `W3.2.3` | stretch |
| `T3.5` | planing weights, retraining, $R_{50}$ against $1/\mathrm{JSD}$ | `W3.2.4` | stretch |
| `T3.5` | the mass scan; *Optional* DisCo | `W3.2.5` | stretch |
| `T3.6` | `t3_w80_noue` with one model (the 0.1.1 minimum path) | `W3.2.1` | warm-up |
| `T3.6` | all taggers, `centre`, `nocoh`, the link to the probe targets | `W3.2.6` | stretch |
| `T3.7` | proposal | `W3.2.8` | stretch |
| `T4.1` | $m_\mathrm{SD}$, $\tau_{21}$, $N$, the medians, the two largest differences | `W4.1.2` | core |
| `T4.1` | $\tau_{32}$, $D_2$, composition, $p_T$, the validation numbers | `W4.1.3` | stretch |
| `T4.2` | `jet_sdmass` and the fraction within 1 %; the sign of `part_deta` | `W4.1.4` | stretch |
| `T4.2` | the $\tau_N$ axes, the fraction within 1 %, why | `W4.1.5` | stretch |
| `T4.3` | `ParT_kin` and its ROC curves | `W4.2.1` | warm-up |
| `T4.3` | (a), (b) | `W4.2.2` | core |
| `T4.3` | *Optional* $R_{50}$ on 20 000 QCD jets; the `ParT_full` gain in detail | `W4.2.5` | stretch |
| `T4.4` | all | `W4.1.7` | stretch |
| `T4.5` | (a), (c), the table; *Predict first* | `W4.2.3` | core |
| `T4.5` | the full matrix, the two diagnostics | `W4.2.4` | stretch |
| `T4.5` | (b) and *Optional* | `W4.2.6` | stretch |
| `T4.6` | (a) | `W4.2.7` | stretch |
| `T4.6` | (b) | `W4.2.8` | stretch |
| `T5.1` | saliency | `W4.4.1` | warm-up |
| `T5.1` | own IG, zero baseline, completeness, the comparison, the prong and underlying-event shares | `W4.4.2` | core |
| `T5.1` | the soft baseline and `wrt`, the steps, PN-lite, the softest 50 %, five displays | `W4.4.4` | stretch |
| `T5.2` | (b) truth deletion | `W4.4.3` | core |
| `T5.2` | (a) deletion curves | `W4.4.5` | stretch |
| `T5.3` | the proxy against the truth | `W4.5.1` | warm-up |
| `T5.3` | IG on JetClass W jets with `ParT_kin` | `W4.5.2` | core |
| `T5.3` | `attention_stats` on toy jets, prong heads, what attention can show | `W4.5.3` | stretch |
| `T5.3` | `ParT_kin` heads with the proxy; *Optional* rank correlation | `W4.5.4` | stretch |
| `T5.4` | (b) the honest ablation | `W4.5.5` | stretch |
| `T5.4` | *Predict first* $U = 0$ | `W4.5.6` | stretch |
| `T5.4` | (a) the regression, the channel ablation | `W4.5.7` | stretch |
| `T5.4` | *Optional* `pair_input_dim=1` | `W4.5.8` | stretch |
| `T5.5` | EFN and PFN | `W4.3.1` | warm-up |
| `T5.5` | all models and `ParT_kin`; *Predict first* | `W4.3.2` | core |
| `T5.5` | the pitfalls | `W4.3.3` | stretch |
| `T5.6` | all | `W4.2.9` | stretch |
| `T5.7` | proposal | `W4.5.9` | stretch |
| `T6.1` | the library augmentations, the table, the classification, what to forget | `W3.3.3` | core |
| `T6.1` | own rotation | `W3.3.6` | stretch |
| `T6.1` | *Optional* `translate`, `drop_soft`, $D_2$ | `W3.3.7` | stretch |
| `T6.2` | the limits on paper | `W3.4.1` | warm-up |
| `T6.2` | own NT-Xent, the checks, the temperature | `W3.4.2` | core |
| `T6.3` | pretraining with monitoring (with a loop skeleton), the saved encoder | `W3.4.3` | core |
| `T6.3` | the loop without skeleton; *Optional* 10 epochs, `partsmall_nopair` | `W3.4.5` | stretch |
| `T6.4` | 1 000 labels, against the random encoder and the supervised PFN | `W3.4.4` | core |
| `T6.4` | all labels, $X_\mathrm{phys}$ with logistic regression and BDT, truth probes | `W3.4.6` | stretch |
| `T6.5` | all | `W3.4.7` | stretch |
| `T6.6` | all | `W4.3.4` | stretch |
| `T6.7` | all | `B.1` | bonus |
| `T7.1` | all | `B.2` | bonus |
| `T7.2` | all | `B.3` | bonus |
| `T7.3` | all | `B.4` | bonus |
| `T7.4` | all | `B.5` | bonus |
| new |  | `W1.1.1`, `W1.1.2`, `W1.1.3`, `W1.1.6`, `W1.1.7`, `W1.1.8`, `W1.1.9`, `W1.1.10`, `W1.2.3`, `W2.1.1`, `W2.2.1`, `W2.3.1`, `W2.3.2`, `W2.4.1`, `W4.1.1` |  |

**New questions** (no 0.1.x source, or new in part; 15): `W1.1.1` (previews `T1.1`(a)), `W1.1.2`, `W1.1.3` (with the
$2m/p_T$ part of `T3.3`), `W1.1.6`, `W1.1.7` (with the sign of `part_deta` of `T0.4`), `W1.1.8` (previews `T1.1`(d) and
`T1.2`), `W1.1.9`, `W1.1.10` (with the top part of `T3.3`), `W1.2.3` (previews an *Optional* part of `T1.3`), `W2.1.1`
(new steps before `T2.1`), `W2.2.1` (one tree as cuts, before `T2.2`), `W2.3.1` and `W2.3.2` (notebook 02, parts A and
B), `W2.4.1` (the invariance checks on the course PFN), `W4.1.1` (with the short dictionary of `T0.4`). `W4.4.1`
(saliency) was a remark in `T5.1` and is now a question.

**Not a question any more.** Two parts of 0.1.x became commands of a *Start of week* box. The start of the downloads
(`T0.4`) runs in the week-3 box (`python examples/make_task_data.py --week 4 --only fetch`). The trainings of
ParT-small and PN-lite (`T2.4`, `T2.5`) are the course models of the week-2 box (`python examples/train_tagger.py
--course-models --out models/course ...`), used from `W2.4.1` on. The two solution scripts that trained them
(`task2_5_*`) map to that box.

## Changes that the remapped reference solutions need

The mapping above is complete. These questions keep their 0.1.x source but changed their set-up in 0.2.0. Their
reference answers must be computed again, not only relabelled (the reference solutions are not part of this release):

- `W1.1.1`, `W1.1.2`, `W1.1.4`: the displayed W jet is the W jet of seed 12 (`show_ladder("wqq", seed=12)`, `jetgen
  display --ladder --process wqq --seed 12`). The 0.1.x reference of `T0.2` used seed 1, a pre-release draft seed 3.
  The top jet of seed 3 (`W1.1.1`) and the QCD jet of seed 2 (`W1.1.5`) are unchanged.
- `W1.1.3`: the W jets are compared jet by jet, as the ratio of `truth_dr_min` to the formula at `truth_seed_pt` and
  `truth_mass`; the top part of `T3.3` moved to `W1.1.10` and is measured on `data/t1_partons.parquet`.
- `W1.1.8`: asks in addition why $\langle N_g\rangle/\langle N_q\rangle$ need not equal $C_A/C_F$.
- `W1.3.2`: names its measures of width (the full width at half maximum or the fraction within $\pm 10$ GeV of the
  peak) and asks about the low tails of $m_\mathrm{SD}$.
- `W3.3.2`: the five observables are $m$, $m_\mathrm{SD}$, $\tau_2$, $N$ and $p_T^D$ ($\tau_{21}$ of `T1.6` moved to
  `W3.3.4`, with the other Sudakov-safe ratios).

## From 0.2.0 back to 0.1.x

| 0.2.0 | 0.1.x source |
|---|---|
| `W1.0.1` | `T0.1` |
| `W1.0.2` | `T0.1` + `T0.2` |
| `W1.0.3`-`W1.0.4` | `T0.3` |
| `W1.1.1`-`W1.1.2` | new |
| `W1.1.3` | new + `T3.3` |
| `W1.1.4`-`W1.1.5` | `T0.2` |
| `W1.1.6` | new |
| `W1.1.7` | new + `T0.4` |
| `W1.1.8`-`W1.1.9` | new |
| `W1.1.10` | new + `T3.3` |
| `W1.2.1`-`W1.2.2` | `T1.1` |
| `W1.2.3` | new |
| `W1.2.4` | `T1.3` |
| `W1.2.5` | `T1.1` |
| `W1.2.6`-`W1.2.8` | `T1.3` |
| `W1.2.9` | `T1.1` |
| `W1.2.10`-`W1.2.11` | `T1.2` |
| `W1.3.1`-`W1.3.6` | `T1.4` |
| `W1.4.1`-`W1.4.6` | `T1.5` |
| `W1.4.7` | `T1.7` |
| `W2.1.1` | new + `T2.1` |
| `W2.1.2`-`W2.1.3` | `T2.1` |
| `W2.2.1` | new |
| `W2.2.2`-`W2.2.4` | `T2.2` |
| `W2.3.1`-`W2.3.2` | new |
| `W2.3.3`-`W2.3.4` | `T2.3` |
| `W2.4.1` | new + `T2.4` |
| `W2.4.2`-`W2.4.4` | `T2.4` |
| `W2.5.1`-`W2.5.2` | `T2.5` |
| `W2.5.3` | `T2.6` |
| `W2.5.4` | `T2.6` + `T2.5` |
| `W2.5.5` | `T2.5` |
| `W2.5.6` | `T2.7` |
| `W3.1.1` | `T3.1` |
| `W3.1.2` | `T3.2` |
| `W3.1.3` | `T3.1` |
| `W3.1.4` | `T3.2` |
| `W3.1.5`-`W3.1.6` | `T3.4` |
| `W3.2.1` | `T3.6` |
| `W3.2.2` | `T3.3` |
| `W3.2.3`-`W3.2.5` | `T3.5` |
| `W3.2.6` | `T3.6` |
| `W3.2.7` | `T3.3` |
| `W3.2.8` | `T3.7` |
| `W3.3.1`-`W3.3.2` | `T1.6` |
| `W3.3.3` | `T6.1` |
| `W3.3.4`-`W3.3.5` | `T1.6` |
| `W3.3.6`-`W3.3.7` | `T6.1` |
| `W3.3.8` | `T3.2` |
| `W3.4.1`-`W3.4.2` | `T6.2` |
| `W3.4.3` | `T6.3` |
| `W3.4.4` | `T6.4` |
| `W3.4.5` | `T6.3` |
| `W3.4.6` | `T6.4` |
| `W3.4.7` | `T6.5` |
| `W4.1.1` | new + `T0.4` |
| `W4.1.2`-`W4.1.3` | `T4.1` |
| `W4.1.4` | `T4.2` + `T0.4` |
| `W4.1.5` | `T4.2` |
| `W4.1.6` | `T0.4` |
| `W4.1.7` | `T4.4` |
| `W4.2.1`-`W4.2.2` | `T4.3` |
| `W4.2.3`-`W4.2.4` | `T4.5` |
| `W4.2.5` | `T4.3` |
| `W4.2.6` | `T4.5` |
| `W4.2.7` | `T4.6` + `T0.4` |
| `W4.2.8` | `T4.6` |
| `W4.2.9` | `T5.6` |
| `W4.3.1`-`W4.3.3` | `T5.5` |
| `W4.3.4` | `T6.6` |
| `W4.4.1`-`W4.4.2` | `T5.1` |
| `W4.4.3` | `T5.2` |
| `W4.4.4` | `T5.1` |
| `W4.4.5` | `T5.2` |
| `W4.5.1`-`W4.5.4` | `T5.3` |
| `W4.5.5`-`W4.5.8` | `T5.4` |
| `W4.5.9` | `T5.7` |
| `B.1` | `T6.7` |
| `B.2`-`B.5` | `T7.1`-`T7.4` |
