# Week 4: Real jets, and where does a tagger look? (core path about 12.75 h, stretch about 32.75 h)

*How to work, what to hand in, the figure checklist and the notation are in [TASKS.md](../TASKS.md).*

**Why this week.** Everything so far was a toy. JetClass was simulated with full generators (programs that simulate the
collision, the shower and the hadrons) and a detector simulation. ParT was trained on 100 million of its jets. Which
conclusions survive? An interpretability tool is a method that shows which inputs a score depends on. Each one is
checked on the toy, where the truth is known, before it is used on the real ParT.

**Thread.** Weeks 1-3 covered what a jet is, the taggers and what they learn, all on the toy. This week brings real jets
and a real tagger. Next, the IRC tests of week 3 are applied to the networks. Last, each attribution tool (a method that
gives every particle a share of the score) is tried on the toy, where the prongs are known. Right after that, it is used
on the real ParT. The final summary answers the course question.

**Concepts this week**, in order of use. Real jets: JetClass and how it was made (MadGraph makes the hard collision,
PYTHIA the shower and the hadrons, Delphes the detector; [primer 9](../docs/primer.md#what-a-detector-sees),
[jetclass.md](../docs/jetclass.md#how-the-events-were-made)). Particle identification (which kind of particle each one
is) and impact parameters (how far a track misses the collision point), in words. Pretrained taggers are models trained
by others, loaded and run as they are. Their ten-class softmax turns the outputs into probabilities that add up to one,
and you renormalise it to three classes. Next come domain shift and transfer ([ml_primer
10](../docs/ml_primer.md#domain-shift)). IRC tests of networks: safety by construction against safety in practice.
Attributions: gradients with respect to the inputs; saliency (how strongly the score reacts to each input); integrated
gradients (baseline, completeness; [ml_primer 11](../docs/ml_primer.md#gradients-of-the-input)). Faithfulness by
deletion: does the score fall when the top-ranked particles are removed? Proxies for the truth: measured quantities that
stand in for the truth labels that real data lack. Stretch adds reweighting, weaver, attention statistics and probes per
block. It also adds the pair bias with its ablations (switching off one part of a model to see what it contributes).

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

## Data for week 4

Three JetClass validation files, the pretrained models of the ParT paper, the split files for weaver and two bonus toy
files. `python examples/make_task_data.py --week 4` runs these commands; it needs `data/t2_detector.parquet` from
week 2. The core questions download about 0.4 GB, the bonus **B.4** downloads 136 MB more, and about 1.2 GB are written.
`--skip-fetch` leaves out the downloads, and `--only fetch` (the week-3 box) runs only the downloads. `jetgen fetch`
resumes an interrupted download. It checks the size and the MD5 checksum of every file
and keeps the files that are already complete ([jetclass.md](../docs/jetclass.md#download)).

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

## W4.1 Real jets: JetClass (core path about 2 h, stretch about 7.5 h)

**Goal:** compare the toy with JetClass, using the same code.

**W4.1.1** *Warm-up, 45 min.* A first look at JetClass. Work through
[notebooks/01_jetclass.ipynb](../notebooks/01_jetclass.ipynb), with [docs/jetclass.md](../docs/jetclass.md) and [primer
section 9](../docs/primer.md#what-a-detector-sees) (Delphes). Then make a short dictionary of the JetClass columns of
week 1: `part_px` ... `part_energy`, `part_deta`, `part_dphi`, `jet_pt`, `jet_eta`, `jet_sdmass`, `jet_tau1` to
`jet_tau3`, `jet_nparticles` and the labels. Give the meaning of each in JetClass. Take `part_deta` and `part_dphi`
from **W1.1.7** if you did it, otherwise from [primer section 3](../docs/primer.md#collider-coordinates). Also say which
of them JetClass does not have (`jetgen info --columns` on a JetClass file and on a toy file).

**W4.1.2** *Core, 75 min.* Toy against JetClass. Take 10 000 jets per class from each data set. From JetClass, take the
first 10 000 jets of each test part. From `data/t2_detector.parquet`, take the first 10 000 jets per class after
`itj.features.jetclass_selection` ($500 < p_T^\mathrm{jet} < 1000$ GeV, the cut of JetClass). Compare $m_\mathrm{SD}$,
$\tau_{21}$ and $N$ per class with your **W1.1.6** function (about 45 s). Make a table of the medians, and explain the
two largest differences with [What the toy leaves out](../docs/physics.md#what-the-toy-leaves-out).

**W4.1.3** *Stretch, 90 min.* The full comparison. Add $\tau_{32}$, $D_2$, the composition by particle type (counts and
energy) and $p_T$ per class to **W4.1.2**. Put the medians next to the validation numbers of
[physics.md](../docs/physics.md#calibration-and-validation).

**W4.1.4** *Stretch, 45 min.* Check the stored branches. Recompute `jet_sdmass` from `part_*`
(`itj.observables.compute(data, ["sdmass"])`) and give the fraction of jets within 1 %. Then recompute `part_deta` on a
JetClass jet at negative $\eta$, with your **W1.1.7** code (or from
[physics.md](../docs/physics.md#units-and-coordinates) if you skipped it).

**W4.1.5** *Stretch, 105 min.* Which axes did Delphes use? N-subjettiness measures distances to a set of axes, one per
subjet. Recompute `jet_tau1` to `jet_tau3`. Which axis choice (WTA-kT, plain exclusive kT, `onepass_kt` in
`itj.observables.subjet_axes`) reproduces the stored $\tau_N$, and why? Give the fraction of jets within 1 %. About 3
minutes (`onepass_kt` about 70 s).

**W4.1.6** *Stretch, 75 min.* The full dictionary. Use [docs/jetclass.md](../docs/jetclass.md),
[docs/data_format.md](../docs/data_format.md) and `jetgen info --columns` to make a table. It lists every column of a
toy file with its JetClass meaning. It also lists the JetClass columns that the toy fills trivially, such as the H
labels of the Higgs classes and the aux match code. Last, it lists the columns that only the toy has (`truth_*`). Name
the two truth columns you expect to be most useful for interpretability, and say why.

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
  the seed parton, the quark or gluon the jet starts from ([data_format.md](../docs/data_format.md#jetclass-oddities)).
- **W4.1.5:** guard against $\tau = 0$, and remember that some JetClass particles lie at $\Delta R > 0.8$
  ([data_format.md](../docs/data_format.md#jetclass-oddities)). Check your numbers afterwards against
  [observables.md](../docs/observables.md#jetclass-conventions).

## W4.2 A tagger trained on 100 million jets (core path about 2.5 h, stretch about 10.5 h)

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
their multiplicity per class matches JetClass ([ml_primer](../docs/ml_primer.md#domain-shift)). Second, drop the photons
or the neutral particles at evaluation (`itj.augment.remove_particles`, after the selection). The matrix takes about 6
minutes and the two diagnostics about 6 more: start each in the background, then read. This answers the $N$ question
of **W2.2.4**.

**W4.2.5** *Stretch, 30 min.* More from the pretrained taggers. Compute $R_{50}$ on all 20 000 QCD test jets, and find
where exactly `ParT_full` gains.

**W4.2.6** *Stretch, 90 min.* More transfer. Four studies: the JetClass models of **W4.1.7** on the toy; the score
distributions per class; the probes of **W3.1.2** across the two data sets. The fourth is fine-tuning: train the toy EFN
further on 1 000 or 10 000 JetClass jets, and compare with training from scratch (a look back at **W3.4.7**).

**W4.2.7** *Stretch, 3.5 h.* weaver
([jetclass.md](../docs/jetclass.md#weaver-in-one-page)). Install its extra (Python 3.10 to 3.13, not on Intel Macs):
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
[examples/weaver/README.md](../examples/weaver/README.md)):

```bash
weaver --data-train 't2:data/t2_split/train/t2_detector.parquet' --data-val data/t2_split/val/t2_detector.parquet --data-test 't2:data/t2_split/test/t2_detector.parquet' --data-config src/itj/configs/jets3_kin.yaml --network-config examples/weaver/networks/example_ParticleTransformer.py -o num_layers 3 -o num_cls_layers 1 -o embed_dims "[64,256,64]" -o pair_embed_dims "[32,32,32]" -o num_heads 4 --model-prefix runs/weaver_partsmall/net --gpus '' --num-workers 0 --batch-size 256 --start-lr 1e-3 --num-epochs 1 --steps-per-epoch 30 --lr-scheduler none --in-memory --fetch-step 1 --predict-output pred.parquet
```

**W4.2.8** *Stretch, 60 min + GPU time.* A full training on a GPU. *Optional (GPU):* a full ParT or ParticleNet training
on the three-class subset ([examples/weaver/README.md](../examples/weaver/README.md)).

**W4.2.9** *Stretch, 2 h.* Probes on the real ParT. Run the probes of **W3.1.2** and **W3.1.4** on the class token of
`ParT_kin` (10 000 JetClass test jets) and on the pooled output of each block. The class token is the summary vector
that the last layer reads ([ml_primer](../docs/ml_primer.md#networks-on-particles)). The probe targets are
$m_\mathrm{SD}$, $\tau_{21}$, $\tau_{32}$, $D_2$, $N$ and the largest $\ln k_t$ among the primary declusterings of
`lund_primary`. Do the same for the toy ParT-small. Is it the same picture? Compare with arXiv:2605.09881 (the basis of
energy correlation functions, ECF, against the N-subjettiness basis). About 2 minutes.

**Deliverables:** `W4.2.1_roc_part.pdf` and the rows (**W4.2.1**); `W4.2.2_roc_pretrained.pdf` and the rows
(**W4.2.2**); `W4.2.3_transfer.pdf` and the AUC table (**W4.2.3**); the AUC matrix and the two diagnostics (**W4.2.4**);
the agreement between weaver and `itj` and the accuracy of the weaver run (**W4.2.7**); `W4.2.9_part_probes.pdf`
(**W4.2.9**).

**Hints**
- **W4.2.1:** the order of the ten JetClass classes is `itj.jetclass.CLASSES` (QCD 0, ..., Wqq 7, Tbqq 8).
- **W4.2.2:** report the top rejection at $\epsilon_S = 0.8$ with its bootstrap error: $R_{50}$ on 5 000 QCD jets rests
  on a handful of jets. `probs_to_jets3` renormalises; it does not sum. Check your numbers afterwards against
  [jetclass.md](../docs/jetclass.md#models-and-numbers).
- **W4.2.3:** use the same selection on both data sets. The toy models have three outputs, `ParT_kin` has ten.
- **W4.2.4:** reweight the jets, not the labels.
- **W4.2.7**, weaver on a laptop. Use `--gpus ''` (the CPU), and `--num-workers` at most the number of files in each
  named group (`0` is fine). Use `--in-memory` for small subsets; weaver 0.5.4 then needs `--steps-per-epoch`, and
  `--steps-per-epoch-val` sets the length of the validation. Use `--fetch-step 1`, because weaver 0.5.4 fails on a
  partial read of a Parquet file. Always use the split files, never an index selection in the YAML: weaver applies
  selections after concatenating the files of a fetch. On Apple Silicon weaver picks MPS (the Apple GPU) by itself: set
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## W4.3 Are the taggers IRC safe? (core path about 1.5 h, stretch about 3.25 h)

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
([observables.md](../docs/observables.md#numerical-irc-tests))? There are four. The inputs are truncated at 128
particles. ParT clamps $k_T$, $\Delta$ and $m^2$ at $10^{-8}$ inside its pair features ($\ln = -18.4$), before a
BatchNorm (a layer that standardises its inputs). Soft particles below the detector thresholds lie outside the training
data. The sign of `part_deta` flips when a soft particle moves $\eta_\mathrm{jet}$ across 0.

**W4.3.4** *Stretch, 2.25 h.* Inherited invariances. Run `irc_scan(..., output="embedding")` on the SSL encoder
of **W3.4.3**, and the score scan on its linear head. Compare both with the course PFN. Is the SSL encoder more
collinear safe? Retrain it without `split_collinear` (about 2 minutes) and compare.

**Deliverables:** `W4.3.1_irc_efn_pfn.pdf` (**W4.3.1**); `W4.3.2_irc_soft.pdf` and `W4.3.2_irc_collinear.pdf`
(**W4.3.2**); one sentence per pitfall (**W4.3.3**); `W4.3.4_irc_ssl.pdf` (**W4.3.4**).

**Hints**
- **W4.3.1:** read [observables.md](../docs/observables.md#numerical-irc-tests) and
  [methods.md](../docs/methods.md#irc-tests-of-networks).
- **W4.3.2:** `irc_scan` refuses jets that a perturbation would push beyond 128 particles
  (`itj.augment.max_particles_guard` tells you whether it would). It keeps the sign of `part_deta` of the
  unperturbed jet.
- **W4.3.4:** fold the scaler and the logistic regression of the linear head into one linear output layer, so that
  `irc_scan` sees one model. Compare embedding scans only within one encoder.

## W4.4 Attributions on the toy (core path about 3.75 h, stretch about 3 h)

**Goal:** find which particles a tagger's score depends on, and test whether that answer can be trusted. You work on
toy jets, whose prongs are known. Models: the course ParT-small and PFN; data: the test part of
`data/t2_detector.parquet`.

**W4.4.1** *Warm-up, 45 min.* Saliency of one jet. Read [ml_primer section
11](../docs/ml_primer.md#gradients-of-the-input) and [methods.md](../docs/methods.md#attributions). Compute the saliency
of the W score of the course ParT-small for one W jet. Draw the jet coloured by it (`itj.plot.attribution_display`;
seconds). Do the bright particles sit on the prongs (`truth_part_prong`)?

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
- **W4.4.4:** [methods.md](../docs/methods.md#attributions) gives realistic completeness gaps; report both baselines.

## W4.5 Where does the real ParT look? (core path about 1.25 h, stretch about 8.5 h)

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
- **W4.5.6:** the reasons why zeroing $U$ misleads are in [methods.md](../docs/methods.md#pair-feature-ablation): read
  them after your prediction.

### Final summary

Write one page, *What is in a jet, and where does a tagger look?*, with the figures from weeks 1-4 that support it. It
takes 1 h with the hand-in and replaces the weekly summary. Hand in `answers/week4.md` with the summary and the answers
to every warm-up and core ID. Hand in these figures: `W4.1.2_toy_vs_jetclass.pdf` with the table of medians,
`W4.2.1_roc_part.pdf`, `W4.2.2_roc_pretrained.pdf` with the rows, `W4.2.3_transfer.pdf` with the AUC table,
`W4.3.1_irc_efn_pfn.pdf`, `W4.3.2_irc_soft.pdf`, `W4.3.2_irc_collinear.pdf`, `W4.4.1_saliency.png`,
`W4.4.2_prong_fraction.pdf`, `W4.4.3_truth_deletion.pdf` and `W4.5.2_part_attribution.pdf`. In the discussion (0.5 h)
we go through the core IDs.

---

- **Previous:** [Week 3: What do taggers learn?](week3.md)
- **Next:** [Bonus: towards a research project](bonus.md)
- **All weeks, the rules and the notation:** [TASKS.md](../TASKS.md#the-weeks)
