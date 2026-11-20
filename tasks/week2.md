# Week 2: Taggers, from cuts to transformers (core path about 12.75 h, stretch about 11.5 h)

*How to work, what to hand in, the figure checklist and the notation are in [TASKS.md](../TASKS.md).*

**Why this week.** You can tell W and top jets from QCD jets by eye in the mass and $\tau_{21}$. Now a computer does it,
one step at a time: one cut, trees on 12 numbers, a neural network on 12 numbers, a network on all particles, graphs and
attention. At each step, ask what it buys and what it costs.

**Thread.** Last week gave the observables that separate the classes. This week you build taggers that use them, then
taggers that see the particles. A tagger is a function that gives each jet a score, large for jets that look like the
signal ([the ml_primer](../docs/ml_primer.md#a-tagger-is-a-function)). Next week looks inside the trained models of
`models/course/`.

**Concepts this week**, in order of use, with the ml_primer section that explains them. From [ml_primer
1](../docs/ml_primer.md#a-tagger-is-a-function): score and threshold, signal and background efficiency, the ROC curve
in the $1/\epsilon_B$ convention, the AUC (area under the ROC curve), $R_{50}$ and
the working point. From [ml_primer 2](../docs/ml_primer.md#train-validation-and-test): train, validation and test parts,
and bootstrap errors. From [ml_primer 3](../docs/ml_primer.md#decision-trees-and-boosting): a decision tree as a set of
cuts, boosting and one-vs-one scores. From [ml_primer 4](../docs/ml_primer.md#neural-networks) and [ml_primer
5](../docs/ml_primer.md#pytorch-in-one-page): tensors and gradients, a linear model, logistic regression and a
multi-layer perceptron (MLP), loss, optimiser, learning rate, mini-batch and epoch, overfitting, standardisation and
early stopping. From [ml_primer 6](../docs/ml_primer.md#networks-on-particles): sets, padding masks and permutation
invariance, Deep Sets and the Particle Flow Network (PFN), and the `state_dict`. The rest of that section is reading
only: graphs and EdgeConv, attention, transformers, the class token and the pair bias. Stretch adds permutation
importance, learning curves, seeds, inference speed and IRC safety by construction in the Energy Flow Network (EFN;
[observables.md](../docs/observables.md#irc-safety)).

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
jets, 10 epochs) and PN-lite, a small ParticleNet (a graph network; see [models.md](../docs/models.md#particlenet)),
with 60 000 jets and 5 epochs. The last two are ParT-small models with and without the pair bias (extra inputs that ParT
computes for each pair of particles; see [models.md](../docs/models.md#particle-transformer)). They are trained on
30 000 jets for 5 epochs. You meet these models in **W2.4** and **W2.5**. For now they are the networks that this week
builds up to, and weeks 3 and 4 use them as well.

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

## Data for week 2

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

## W2.1 Your first tagger: cuts (core path about 2.25 h, stretch about 1 h)

**Goal:** learn what a tagger is and how to measure one, and build the cut-based tagger that every later tagger is
compared with.

**W2.1.1** *Warm-up, 45 min.* One window, two efficiencies. Read [ml_primer sections
1-2](../docs/ml_primer.md#a-tagger-is-a-function). Split `data/t2_detector.parquet`. On the val part, keep the W and QCD
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
  rests on few QCD jets: give the rejection at $\epsilon_S = 0.8$ as well, and say how many QCD jets pass. All jets
  outside the window share the lowest score, so the ROC curve of a window tagger stops at the efficiency of the window
  and goes on as a straight line to (1, 1). That line counts in the AUC: compare cut taggers at a working point such as
  $R_{50}$, not by the AUC alone.

## W2.2 Boosted decision trees (core path about 1.25 h, stretch about 1.5 h)

**Goal:** a boosted decision tree (BDT), a standard machine-learning tagger, on the 12 jet-level numbers. A BDT adds up
many small trees, each correcting the errors of the sum before it. You build it up from one tree, which is nothing but a
set of cuts.

**W2.2.1** *Warm-up, 30 min.* One tree is a set of cuts. Read [ml_primer section
3](../docs/ml_primer.md#decision-trees-and-boosting). You use scikit-learn (sklearn), a Python library of standard
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

## W2.3 A neural network from zero (core path about 3.75 h, stretch about 0.75 h)

**Goal:** your first neural network and training loop in PyTorch, on the same
12 inputs as the BDT. You get there in three steps: the parts of a network by hand, then a network and its loop in the
notebook, then your own loop.

**W2.3.1** *Warm-up, 60 min.* Tensors and gradients. Read [ml_primer section 4](../docs/ml_primer.md#neural-networks) up
to logistic regression, and [section 5](../docs/ml_primer.md#pytorch-in-one-page). Work through part A of
[notebooks/02_first_network.ipynb](../notebooks/02_first_network.ipynb) (its sections 1-3). It starts with the 12 jet
features of a small generated sample (6 000 jets, seconds of CPU). Next come tensors and autograd (PyTorch's automatic
gradients) on a two-line function. Last comes a logistic regression on $m_\mathrm{SD}$ and $\tau_{21}$, written by hand
with gradient descent (repeated small steps against the gradient of the loss), and its AUC. Then change the number of
gradient steps and the step size, and describe what happens to the loss.

**W2.3.2** *Warm-up, 75 min.* A network and its training loop. Read the rest of [ml_primer section
4](../docs/ml_primer.md#neural-networks). Then work through part B of the notebook (its sections 4-8, under a minute of
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
  4](../docs/ml_primer.md#neural-networks): a layer that switches off random units during training). Then compare train
  and val loss in the same mode (`model.eval()` for both); if you do not, say so. Dropout is on in training mode and
  makes the train loss look worse. An MLP far behind the BDT usually
  lacks standardisation, the best-state restore or enough epochs. `fit(..., max_minutes=15)` stops a run that would
  take too long, and `fit` prints an estimate of the time after 20 batches.

## W2.4 Networks that see particles: Deep Sets (core path about 2.5 h, stretch about 2.75 h)

**Goal:** taggers that read all the particles of a jet as a set, with no order, and why the order of the particles must
not matter.

**W2.4.1** *Warm-up, 45 min.* A network that does not care about the order. Read [ml_primer section
6](../docs/ml_primer.md#networks-on-particles) up to graphs, and [models.md](../docs/models.md#pfn-and-efn). Load the
course PFN and make one `JetBatch` (the course's batch of padded jets) of 64 test jets. Check numerically that its
outputs do not change when you shuffle the particles of every jet. Check also that they do not change when you pad the
batch with empty particles.

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
([models.md](../docs/models.md#the-model-api)). The inner network $\Phi$ acts on each particle and the outer network $F$
on the sum. Apply $\Phi$ to the `kin` features of each particle and add the results with a masked sum, a sum over the
real particles only. The seven `kin` features are listed in [models.md](../docs/models.md#inputs). Check that your PFN
is invariant under permutations and under extra padding (as in **W2.4.1**). Next, check that your PFN is the same
network as the course PFN. A key map is a dictionary from the parameter names of the course PFN to yours. Copy the
weights of the course PFN into your PFN with a key map of a few lines (`model.state_dict()`), and check that the outputs
agree on one batch.

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

## W2.5 Graphs and attention (core path about 1.75 h, stretch about 5.5 h)

**Goal:** the network designs of the best published taggers, trained on a CPU, and the ranking of all your taggers.

**W2.5.1** *Warm-up, 45 min.* Read and predict. Read the rest of [ml_primer section
6](../docs/ml_primer.md#networks-on-particles): graphs; attention as a weighted average; heads, the class token, the
pair bias $U$. Then read [models.md](../docs/models.md#particlenet) on ParticleNet, a graph network, and on the
[Particle Transformer](../docs/models.md#particle-transformer) (ParT). ParT-small (0.24 M parameters) is the course's
small transformer. The preset `partlite` is a larger one, the ParT-lite of the dl4hep notebooks (the notebooks this
course grew from; [ml_primer: where to read more](../docs/ml_primer.md#where-to-read-more)). *Predict first:* the
ranking of all your taggers, and what the pair bias buys. Check that `models/course/` holds the five models
(`ls models/course`).

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
well. The time of the course ParT-small comes from the start-of-week box, with its own thread count and with your work
running beside it, so it is not on the same footing as the times of your own runs. Mark it as indicative, or time one
epoch of the same recipe yourself at the thread count of your curves and multiply by its five epochs. Do the curves
cross, against the training size and against the training time, and if so where? Then measure the inference speed of
every tagger (how fast a trained model scores new jets) in jets per second, with `torch.no_grad()`, on one thread and on
all threads. Add it as a column of the **W2.5.2** cost table.

**W2.5.5** *Stretch, 30 min + GPU time.* Larger transformers. *Optional (GPU):* `partlite`, or more training data, on a
GPU ([examples/cluster](../examples/cluster/README.md)).

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

### Weekly summary

The ranking of your taggers with errors and costs, in half a page. It answers the thread question *Which tagger wins, by
how much and at what price?* (0.5 h with the hand-in). Hand in `answers/week2.md` with the summary and the answers to
every warm-up and core ID. Hand in as well the figures `W2.1.1_three_windows.pdf`, `W2.1.2_roc_cuts.pdf`,
`W2.2.1_tree.pdf`, `W2.2.2_roc_bdt.pdf`, `W2.3.1_logistic.pdf`, `W2.3.2_loss.pdf`, `W2.3.3_loss.pdf` and
`W2.5.2_roc_all.pdf` with the cost table, and `results_table.csv` (cuts, BDT, MLP). In the discussion (0.5 h) we go
through the core IDs.

---

- **Previous:** [Week 1: What is a jet?](week1.md)
- **Next:** [Week 3: What do taggers learn?](week3.md)
- **All weeks, the rules and the notation:** [TASKS.md](../TASKS.md#the-weeks)
