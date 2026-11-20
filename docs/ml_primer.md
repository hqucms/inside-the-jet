# Machine learning from zero

This primer is the second unit of the course. The first, *Jets from zero* ([primer.md](primer.md)), explained what is
in a jet. This one explains, from first-year mathematics, how a computer learns to tell jets apart and how you check
what it learned. It assumes Python with NumPy, the mean and spread of a set of numbers, and the derivative of a
function of one variable. It assumes no machine learning, no statistics beyond
[primer section 11](primer.md#histograms-and-errors) and no PyTorch (the Python library for neural networks that the
course uses from the second week on).

**How to read it.** In pieces, each just before the question that uses it (the table below), from week 2 to week 4.
Every section takes about 10 minutes and ends with two or three *Check yourself* questions, whose answers are collapsed
at the end. Sections 4 and 6 are read in two halves. The worked example is
[notebooks/02_first_network.ipynb](../notebooks/02_first_network.ipynb), a first network written step by step (`W2.3.1`,
`W2.3.2`). For depth, every section points to [methods.md](methods.md) (the methods of weeks 2-4) or
[models.md](models.md) (the networks); the [glossary](glossary.md) lists the terms. The Python blocks run as they
stand, in order, in a few seconds and without data.

| § | section | reading | read before |
|---|---|---|---|
| 1 | [A tagger is a function](#a-tagger-is-a-function) | 10 min | `W2.1.1` |
| 2 | [Train, validation and test](#train-validation-and-test) | 10 min | `W2.1.1` |
| 3 | [Decision trees and boosting](#decision-trees-and-boosting) | 10 min | `W2.2.1` |
| 4 | [Neural networks](#neural-networks) | 10 + 10 min | `W2.3.1` (first half), `W2.3.2` (second half) |
| 5 | [PyTorch in one page](#pytorch-in-one-page) | 10 min | `W2.3.1` |
| 6 | [Networks on particles](#networks-on-particles) | 10 + 15 min | `W2.4.1` (first half), `W2.5.1` (second half) |
| 7 | [Embeddings and maps](#embeddings-and-maps) | 10 min | `W3.1.1` |
| 8 | [Regression and probes](#regression-and-probes) | 10 min | `W3.1.2` |
| 9 | [Learning without labels](#learning-without-labels) | 10 min | `W3.4.1` |
| 10 | [Domain shift](#domain-shift) | 10 min | `W4.2.3` |
| 11 | [Gradients of the input](#gradients-of-the-input) | 10 min | `W4.4.1` |
| 12 | [Where to read more](#where-to-read-more) | optional | |
| | [Check yourself: answers](#check-yourself-answers) | | |

## A tagger is a function

*Section 1, about 10 minutes.*

A **tagger** reads a jet $x$ and returns a number, its **score** $s(x)$. The jet is given as a few numbers, such as its
mass, or as the list of its particles. The score is large for jets that look like the **signal** and small for jets that
look like the **background**. Here the signal is W or top jets. These come from the decay of a fast W boson or top quark
and have two or three prongs (hard clusters of energy, one per decay quark; see the
[primer](primer.md#boosted-w-and-top-jets)). The background is QCD
jets: ordinary jets started by a single quark or gluon (QCD, quantum chromodynamics, is the theory of the strong force;
see the [primer](primer.md#the-particles)). A tagger with several classes returns one probability per class. In this
course the classes are QCD, W and top, and their probabilities add up to 1.

**Score and threshold.** To decide, you cut: a jet with $s > c$ is called signal. The cut keeps two fractions,

- the **signal efficiency** $\epsilon_S$, the fraction of signal jets that pass;
- the **background efficiency** $\epsilon_B$, the fraction of background jets that pass (the *mistag rate*).

A cut on one observable (a number computed from the particles of a jet) is a tagger too. For example, take a window in
the soft-drop mass, the jet mass after soft, wide-angle particles are removed (see the
[primer](primer.md#a-first-look-at-the-observables)). The window is a score that is high inside it and low outside it.

**The ROC curve.** Moving the threshold $c$ from high to low traces $\epsilon_B$ against $\epsilon_S$, from $(0, 0)$
(nothing passes) to $(1, 1)$ (everything passes). This is the **ROC curve** (receiver operating characteristic). Jet
physics draws it as the **rejection** $1/\epsilon_B$ on a log axis against $\epsilon_S$, so that a better tagger is a
higher curve. A tagger that guesses at random has $\epsilon_B = \epsilon_S$ at every threshold.

**AUC.** The **AUC** (area under the curve) is the area under $\epsilon_S$ as a function of $\epsilon_B$, with both on
linear axes (not the rejection convention above). It is 0.5 for guessing and 1 for a perfect tagger. It is also the
probability that a random signal jet gets a higher score than a random background jet. It summarises the whole curve in
one number, and so it hides where on the curve a tagger is good.

**Working point and $R_{50}$.** An analysis uses one threshold, its **working point**, for example the cut that keeps
half of the signal. The rejection there is $R_{50} = 1/\epsilon_B$ at $\epsilon_S = 0.5$. A rejection $R$ measured with
$N_B$ background jets rests on only $N_B/R$ jets that pass. A large rejection measured on a small test set is therefore
uncertain.

**Three classes, one pair at a time.** For "W against QCD" you keep the W and QCD jets only. You score them with the
**one-vs-one** score $p_W/(p_W + p_\mathrm{QCD})$ (`itj.metrics.one_vs_one`, from the course's Python package). Top
against QCD works the same way.

```python
import numpy as np

import itj.metrics

rng = np.random.default_rng(0)
s_sig = rng.normal(1.0, 1.0, 5000)  # made-up scores of 5 000 signal jets
s_bkg = rng.normal(0.0, 1.0, 5000)  # and of 5 000 background jets
c = 1.5
eff_s, eff_b = np.mean(s_sig > c), np.mean(s_bkg > c)  # counting is all it takes
print(f"cut at {c}: eff_S = {eff_s:.3f}, eff_B = {eff_b:.4f}, rejection = {1 / eff_b:.1f}")

y = np.concatenate([np.ones(5000), np.zeros(5000)])  # the labels: 1 signal, 0 background
s = np.concatenate([s_sig, s_bkg])
eff_s, eff_b, thresholds = itj.metrics.roc(y, s)  # the whole curve: eff_S first, then eff_B
print(f"at eff_S = 0.5 the curve gives eff_B = {np.interp(0.5, eff_s, eff_b):.4f}")
print(f"AUC = {itj.metrics.auc(y, s):.3f}, R50 = {itj.metrics.rejection(y, s, 0.5):.1f}")
```

**Check yourself.** (1a) A cut keeps 400 of 1 000 signal jets and 20 of 10 000 background jets. What are $\epsilon_S$,
$\epsilon_B$ and the rejection? (1b) Why does the ROC curve of a tagger that guesses lie on $\epsilon_B = \epsilon_S$?
(1c) A rejection of 2 000 is quoted from 10 000 background jets. On how many jets does it rest?

**Used in:** `W2.1.1`, `W2.1.2`; every tagger after them. **Read more:** [methods.md](methods.md#metrics).

## Train, validation and test

*Section 2, about 10 minutes.*

A tagger learns from labelled examples, jets whose true class is known. A learned tagger is always judged on jets it
has not seen, so the labelled jets are split into three sets. TASKS and the course code call them the train, val
and test *parts* of a file:

- **train**, the training set: the jets the model learns from (it learns its weights, the adjustable numbers inside
  it, or its cuts);
- **validation** (val), the validation set: the jets you use to make choices, such as how long to train, how large a
  network, which threshold, or which of two models;
- **test**, the test set: jets that took no part in any choice. You use them once, at the end, to report the result.

A model scored on its own training jets looks better than it is, because it may have learned those particular jets,
their random fluctuations included. The same is true, more mildly, of every choice made by looking at a set of jets.
The test set is the only one that has never influenced the model.

**Leakage** is any route by which information from the test set reaches the training or the choices. Four examples:

- standardising the inputs (shifting and scaling each input to a common range; see
  [Neural networks](#neural-networks)) with the mean of all jets;
- picking the best epoch (one pass over the training set) on the test set;
- the same jet in two sets;
- an input that contains the label, such as the `truth_*` columns of the toy (the course's simulated jets; see the
  [primer](primer.md#the-toy-and-its-files)).

Leakage makes results look better than they are, and it is easy to miss.

In `itj`, `itj.features.split(data)` returns the three sets in a fixed order: test the first 20 %, val the next 10 %,
train the rest. Every student, and every week, therefore uses the same jets.

**Errors of a metric.** A metric measured on a finite test set fluctuates: another test set of the same size would give
a slightly different value. The **bootstrap** estimates that spread without new data. Draw as many jets as the test set
holds *with replacement* (some twice, some not at all) and recompute the metric. Repeat 100 times and take the standard
deviation. The bootstrap covers the statistics of the test set only. Two trainings with different random seeds (the
starting values of the random-number generator) differ as well, and that spread is measured by training again.

**Learning curves.** A metric plotted against the size of the training set, with the same validation and test sets for
every size, says whether more data would help (`W2.5.3`).

```python
auc_mean, auc_err = itj.metrics.bootstrap(itj.metrics.auc, y, s)  # the y and s of section 1
print(f"AUC = {auc_mean:.4f} +- {auc_err:.4f} (bootstrap, {len(y)} test jets)")
```

**Check yourself.** (2a) You standardise the inputs with the mean and standard deviation of all 150 000 jets before
splitting. Is that leakage? (2b) What does a bootstrap error not include? (2c) Why is `truth_dr_min` never an input of
a tagger?

**Used in:** `W2.1.1`, `W2.1.2`; every result from then on. **Read more:** [methods.md](methods.md#metrics).

## Decision trees and boosting

*Section 3, about 10 minutes.*

A **decision tree** asks yes-or-no questions about the inputs, one input and one threshold at a time. It might ask "is
the mass above this value?" and then, on the *yes* branch, "is $\tau_{21}$ below that value?", and so on.
The second question uses the N-subjettiness ratio, which is small for a jet with two prongs (see the
[primer](primer.md#a-first-look-at-the-observables)).
The questions end in **leaves**. Each leaf is a box in the space of the inputs, and it predicts the fraction of signal
among the training jets that fell into it. A tree is a set of nested cuts, and it can be drawn as a picture
(`sklearn.tree.plot_tree`).

Training builds the tree from the top. At each node it tries every input and every threshold and keeps the one that
separates the classes best, so that the two boxes are as pure as possible. It then repeats on each side. The **depth**
is the number of questions on the way to a leaf; a tree of depth $d$ has at most $2^d$ leaves. A deep tree can cut tiny
boxes around single training jets and learn them by heart. It then **overfits**: it does worse on new jets.

A tree only compares each input with thresholds, so it does not care about the units or the scale of an input. The
mass in GeV or in MeV (units of energy and mass; see the [primer](primer.md#units-and-energies)) gives the same tree.

**Boosting.** One small tree is a crude tagger. **Gradient boosting** adds many small trees one after the other. Each
new tree is fitted to the errors that the sum of the earlier trees still makes, and it is added with a small weight
(the learning rate). The sum of a few hundred small trees is a flexible function made of many boxes: a **boosted
decision tree** (BDT). The `HistGradientBoostingClassifier` of scikit-learn (a Python machine-learning library) first
bins every input into at most 255 bins, which makes it fast. For three classes it builds one set of trees per class,
and `predict_proba` returns three probabilities. On more than 10 000 jets it keeps 10 % of the training jets aside and
stops adding trees when they no longer help there.

**Which inputs matter?** The **permutation importance** of an input is the loss of performance when that input is
shuffled across the jets of the validation set. Shuffled, it no longer carries information. Two inputs that carry the
same information share their importance: shuffle one, and the tagger still has the other.

**Check yourself.** (3a) A tree of depth 3: how many leaves at most, and how many cuts on the way to each? (3b) Why
does it not matter for a tree whether the mass is in GeV or in MeV? (3c) Two inputs are almost copies of each other.
What happens to the permutation importance of each?

**Used in:** `W2.2.1`, `W2.2.2`, `W2.2.3`. **Read more:** [methods.md](methods.md#metrics); the scikit-learn user
guide, *Histogram-based gradient boosting*.

## Neural networks

*Section 4, about 20 minutes, in two halves: read the first with `W2.3.1` and the second with `W2.3.2`.*

**First half: a straight line that learns.** The simplest model of a class is a **linear model** followed by a
squashing function (the **sigmoid**). With inputs $x = (x_1, \dots, x_n)$, weights $w$ and a bias $b$,

```math
z = w \cdot x + b , \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}} ,
```

where $p$, between 0 and 1, is the probability that the jet is signal. This model is **logistic regression**. The
weights and the bias are the **parameters**; learning means choosing them. With several classes each class $k$ gets its
own $z_k$. The **softmax** $p_k = e^{z_k}/\sum_j e^{z_j}$ turns them into probabilities that add up to 1.

**The loss.** Learning needs one number that says how wrong the model is: the **loss**. For a classifier it is the
**cross-entropy**, the mean over the jets of $-\ln$ of the probability that the model gives to the true class,

```math
L = -\frac{1}{N}\sum_{i=1}^{N}\bigl[y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\bigr] \qquad (y_i = 1 \text{ signal, } 0
\text{ background}).
```

It is 0 for a model that is certain and right, and large for one that is certain and wrong.

**Gradient descent.** The gradient $\partial L/\partial w$ points in the direction in which the loss grows fastest.
**Gradient descent** takes a small step the other way, again and again:

```math
w \leftarrow w - \eta\,\frac{\partial L}{\partial w} , \qquad b \leftarrow b - \eta\,\frac{\partial L}{\partial b} .
```

The step size $\eta$ is the **learning rate**. For logistic regression the gradient has a short form,
$\partial L/\partial w = \frac{1}{N}\sum_i (p_i - y_i)\,x_i$ and
$\partial L/\partial b = \frac{1}{N}\sum_i (p_i - y_i)$: each jet pulls the line by how wrong its probability is.
Part A of notebook 02 writes this loop by hand. The first experiment of `W2.3.1` asks what a learning rate that is too
large or too small does.

**Second half: from a line to a network.** A **multi-layer perceptron** (MLP) stacks linear layers with a simple
non-linear **activation** between them. The usual activation is the ReLU (rectified linear unit),
$\mathrm{relu}(u) = \max(0, u)$:

```math
h_1 = \mathrm{relu}(W_1 x + b_1), \qquad h_2 = \mathrm{relu}(W_2 h_1 + b_2), \qquad z = W_3 h_2 + b_3 .
```

The vectors $h_1$, $h_2$ are **hidden layers**; their lengths are the widths of the network. Without the activations
the three layers would collapse into one linear map. With them, an MLP can approximate any smooth function of its
inputs, given enough width. Its gradients come from the chain rule, applied layer by layer from the loss backwards
(**backpropagation**); PyTorch does it for you (section 5).

**How a network is trained.**

- **Mini-batches and epochs.** The gradient is computed on a **mini-batch** of, say, 256 jets. This is a noisy but
  cheap estimate, and the weights take one step per batch. One pass over the training set is an **epoch**.
- **The optimiser.** Plain gradient descent uses one learning rate for every weight. **AdamW** adapts the step of each
  weight from running averages of its past gradients. It also pulls the weights gently towards zero (*weight decay*).
  The learning rate is still the main setting.
- **Standardisation.** A network starts from small random weights and learns best when every input is of order 1.
  Subtract the mean of each input and divide by its standard deviation, using the statistics of the training set only
  (section 2).
- **Overfitting and validation.** After each epoch, compute the loss on the validation set as well. While the model
  learns what generalises, both losses fall. When it starts to learn the training jets by heart, the training loss
  keeps falling and the validation loss stops falling or rises: **overfitting**.
- **Early stopping.** Keep the weights of the epoch with the lowest validation loss, and stop when it has not improved
  for a few epochs (the *patience*).
- **Dropout.** A dropout layer sets a random fraction of its inputs to zero during training, so that the network cannot
  rely on any single unit. At evaluation it does nothing. `model.train()` and `model.eval()` switch between the two
  modes, so compare losses in the same mode.

**Check yourself.** (4a) A model gives the probability 0.9 to the true class of one jet and 0.1 to the true class of
another. What is the cross-entropy of the two? (4b) Why is a stack of linear layers without activations no better than
one linear layer? (4c) The training loss falls for 20 epochs; the validation loss is lowest after epoch 6. Which weights
do you keep?

**Used in:** `W2.3.1` (first half), `W2.3.2` (second half), `W2.3.3`. **Read more:** notebook 02;
[models.md](models.md#mlp); Guest, Cranmer and Whiteson (section 12).

## PyTorch in one page

*Section 5, about 10 minutes.*

PyTorch is the Python library in which the networks of this course are written. Five ideas cover what the questions
need.

- **Tensors.** A `torch.Tensor` is an array like a NumPy array, with a type (`float32` by default) and a device, the
  processor that does the arithmetic (the CPU here). `torch.tensor(array)` and `torch.from_numpy(array)` make one;
  `t.numpy()` goes back (after `t.detach()` if it carries gradients).
- **Autograd.** A tensor made with `requires_grad=True` records every operation done with it. After you call
  `f.backward()` on a scalar result, the derivative of `f` with respect to the tensor is in its `.grad`.
- **Modules.** A network is an `nn.Module`: an object that holds its parameters and computes its output in `forward`.
  `nn.Linear(n_in, n_out)` is one linear layer; `nn.Sequential(...)` chains layers. `print(model)` shows the layers and
  the activations. `model.state_dict()` holds only the weights, by name; it is what is saved and copied between
  models. `nn.Linear` acts on the **last** axis of its input, whatever the axes before it.
- **The training step.** Three lines after each mini-batch: `opt.zero_grad()` (forget the old gradients),
  `loss.backward()` (compute the new ones), `opt.step()` (move the weights). They need an optimiser such as
  `torch.optim.AdamW(model.parameters(), lr=1e-3)`.
- **Evaluation.** `with torch.no_grad():` switches the recording off, which is faster and needs less memory. Use it
  with `model.eval()` for predictions. `torch.set_num_threads(2)` limits PyTorch to two CPU cores, so that a long run in
  the background leaves your computer responsive for other work.

```python
import torch
from torch import nn

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # a tensor that records what is done with it
f = (x**2).sum()  # f = x1^2 + x2^2 + x3^2
f.backward()  # autograd: df/dx goes into x.grad
print(x.grad)  # tensor([2., 4., 6.])

layer = nn.Linear(7, 16)  # 7 numbers in, 16 out
particles = torch.randn(64, 30, 7)  # (jets, particles, features per particle)
print(layer(particles).shape)  # torch.Size([64, 30, 16]): the last axis only

model = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 3))
print(model)  # layers and activations
print({name: tuple(w.shape) for name, w in model.state_dict().items()})  # weights only, by name
model.eval()
with torch.no_grad():
    probs = torch.softmax(model(torch.randn(5, 12)), dim=1)  # 5 jets, 12 inputs, 3 classes
print(probs.sum(dim=1))  # every row adds up to 1
```

**Check yourself.** (5a) What is `x.grad` after `f = (3 * x).sum()` and `f.backward()`? (5b) What happens when
`nn.Linear(7, 16)` gets a tensor of the shape (64, 7, 30)? (5c) What does `print(model)` show that the `state_dict`
does not?

**Used in:** `W2.3.1`, `W2.3.2`, `W2.3.3`, `W2.4.2`; every network after them. **Read more:** the PyTorch tutorial
*Learn the basics*; [models.md](models.md#the-model-api).

## Networks on particles

*Section 6, about 25 minutes, in two halves: read the first with `W2.4.1` and the second with `W2.5.1`.*

**First half: a jet is a set.** The networks of section 4 read a fixed list of numbers. A jet is a **set** of
particles: their number varies from jet to jet, and they have no natural order. Sorting them by $p_T$ (the transverse
momentum, the part of the momentum across the beam; see the [primer](primer.md#collider-coordinates)) is a
convention, not physics.

- **Padding and masks.** To put jets of different lengths into one batch, each jet is filled up with empty slots to a
  common length $P$. A **mask** marks the real particles, as `padded_p4()` does in [primer section
  10](primer.md#the-toy-and-its-files).
- **The layout of a batch.** An `itj.features.JetBatch` is **channel-first**, like weaver (the training framework of
  ParT and the JetClass paper; see [jetclass.md](jetclass.md#weaver-in-one-page)). `features` has the shape (jets,
  channels, particles), where the channels are the features of each particle. `points` (jets, 2, particles) holds
  $(\Delta\eta, \Delta\phi)$, the position of each particle relative to the jet axis. `vectors` (jets, 4, particles)
  holds the four-vectors (the energy and momentum of each particle), and `mask` (jets, 1, particles) the mask. The
  particles are on the last axis. A layer that acts on each particle (an `nn.Linear`, section 5) therefore needs the
  features moved to the last axis first.
- **Permutation invariance.** A permutation is a reordering. A tagger on particles should give the same score whatever
  the order of the particles, and the same score whatever the number of empty slots.

**Deep Sets.** A simple way to be invariant is to treat every particle alike and then add:

```math
s(x) = F\Bigl(\sum_{i \in \text{jet}} \Phi(x_i)\Bigr) ,
```

where $\Phi$ is a small network applied to each particle on its own. It maps the features of one particle to a vector
of, say, 128 numbers. The sum runs over the real particles only, and $F$ is an MLP on the summed vector. A sum does not
depend on the order of its terms. Every permutation-invariant function of a set can be written in this form
(arXiv:1703.06114). The **Particle Flow Network** (PFN) is this network with all the particle features in $\Phi$. The
**Energy Flow Network** (EFN) weights each term by the momentum fraction $z_i$ (the share of the jet's transverse
momentum that the particle carries). It gives $\Phi$ only the angles $(\Delta\eta_i, \Delta\phi_i)$, which builds soft
and collinear safety into it (`W2.4.4`). Soft and collinear safety means that adding a very soft particle, or splitting
one particle into two that fly in the same direction, leaves the output unchanged (see the
[primer](primer.md#a-first-look-at-the-observables)).

**Second half: graphs and attention.** In a sum every particle is added on its own. The next networks let particles
see each other.

- **Graphs.** Connect each particle to its $k$ nearest neighbours in $(\Delta\eta, \Delta\phi)$, and the jet becomes a
  **graph**: points joined by links. An **EdgeConv** block computes, for each particle $i$ and each neighbour $j$, a
  small network of $(x_i, x_j - x_i)$ and averages over the neighbours. The next block finds the neighbours again,
  this time in the features it has learned. **ParticleNet** stacks such blocks, then averages over the particles;
  PN-lite is a smaller version.
- **Attention.** Each particle $i$ computes three vectors from its features by linear maps: a **query** $q_i$, a **key**
  $k_i$ and a **value** $v_i$. Particle $i$ then forms a weighted average of the values of all particles,

  ```math
  \text{out}_i = \sum_j a_{ij}\,v_j , \qquad a_{ij} = \operatorname{softmax}_j\Bigl(\frac{q_i\cdot k_j}{\sqrt d}\Bigr) ,
  ```

  with weights $a_{ij}$ that add up to 1 over $j$. A weight is large where the query of $i$ matches the key of $j$
  ($d$ is the length of the vectors). Padded slots get the weight 0. The weights depend on the content of the jet, not
  on fixed positions, so attention treats the particles as a set.
- **Heads and blocks.** Several attention maps work side by side, each with its own maps for queries, keys and values:
  these are the **heads**. A **transformer block** is attention followed by a small MLP on each particle, with
  normalisation layers in between. Each of the two is wrapped so that its input is added to its output (a *residual
  connection*).
- **The class token.** After the particle blocks, the network adds the **class token**: a learned extra entry that is
  treated like a particle but belongs to none. It attends to all particles in one or two *class-attention* blocks. Its
  final vector is the jet's summary, and the last linear layer reads it.
- **The pair bias.** The Particle Transformer (ParT) adds to the attention weight of every pair $(i, j)$, before the
  softmax, a learned function $U_{ij}$ of physics quantities of the pair. These quantities are $\ln k_T$, $\ln z$,
  $\ln\Delta$ and $\ln m^2$, the Lund coordinates of [primer section 7](primer.md#the-parton-shower) for a pair of
  particles. On log scales, they are the pair's relative transverse momentum, momentum share, angle and mass. What
  the pair bias buys is a question of `W2.5` and `W4.5`.

**Check yourself.** (6a) Why is the largest value over the particles (a max instead of a sum) also invariant under
reordering? (6b) The `features` of a batch have the shape (64, 7, 95). How many jets, features per particle and
particle slots? (6c) What do the attention weights $a_{ij}$ of one particle $i$ add up to?

**Used in:** `W2.4.1`, `W2.4.2` (first half); `W2.5.1`, `W4.5` (second half). **Read more:**
[models.md](models.md#pfn-and-efn), [models.md](models.md#particlenet),
[models.md](models.md#particle-transformer), [methods.md](methods.md#attention).

## Embeddings and maps

*Section 7, about 10 minutes.*

**Embeddings.** Every network of the course ends with one linear layer that turns a vector $h$ into the class scores.
That vector is the **embedding** of the jet (`model.embed`), the network's summary of the jet: 64 or 128 numbers per
jet. The final layer is only a linear cut in this space. Week 3 asks what these numbers hold.

**PCA.** Principal component analysis (PCA) rotates the embedding so that the first axis points along the direction in
which the jets differ most (the largest variance). The second axis points along the largest variance of the rest, and
so on. Projecting on the first two components gives a flat picture of the embedding. The *explained variance* of each
component says how much of the spread it keeps. PCA is linear: it keeps straight lines and the order of distances along
its axes. If one coordinate of the embedding has much larger numbers than the others, it dominates the first
component.

**t-SNE and UMAP.** These two maps (t-distributed stochastic neighbour embedding, and uniform manifold approximation
and projection) are non-linear. They try to keep jets that are neighbours in the embedding as neighbours in the
picture. They show clusters well, but the distances between clusters and the sizes of clusters mean little. The
picture also changes with their settings (the *perplexity* of t-SNE, the number of neighbours of UMAP) and with the
seed. Use them to form ideas, then test the ideas with a number (section 8).

**Check yourself.** (7a) An embedding has 128 numbers per jet and a PCA picture keeps 2. What is lost? (7b) Two clusters
of a t-SNE map lie far apart. Are their jets very different?

**Used in:** `W3.1.1`, `W3.1.3`. **Read more:** [methods.md](methods.md#looking-at-embeddings).

## Regression and probes

*Section 8, about 10 minutes.*

**Linear regression** predicts a number $y$ from inputs $x$ with $\hat y = w\cdot x + b$. The number can be a mass, or a
multiplicity (the number of particles in a jet). The fit chooses $w$ and $b$ so that the squared errors are smallest.
**Ridge** regression adds a small penalty on the size of $w$, which keeps the fit stable when inputs are correlated.
Its score is

```math
R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2} :
```

1 for a perfect prediction, 0 for a model that always predicts the mean $\bar y$, negative for a worse one.
**Logistic regression** (section 4) predicts a class, scored with the AUC (section 1).

**Cross-validation.** A score measured on the jets a model was fitted on is too optimistic (section 2). With $k$-fold
cross-validation the jets are split into $k$ parts (folds). The model is fitted on $k - 1$ of them and scored on the one
left out, and each fold takes its turn. The mean over the folds is the score, and the spread over the folds is its
error.

**Probes.** To ask whether a trained network "knows" a quantity, freeze the network (keep its weights fixed) and
extract the embedding $h$ of every jet (section 7). Then fit a linear regression from $h$ to the quantity: this is a
**probe**. A high score says that the quantity can be read from $h$ with a straight line. It does not say that the
network uses it to tag.

**Controls.** A probe score is only meaningful next to the same probe on two kinds of control:

- a **random-weight** network: the same architecture (the same layers and sizes) with its weights reset to random
  values (`reset_parameters()`);
- a **control task**: the same probe with the target shuffled across the jets
  (`itj.probes.control_task(y, kind="shuffle")`), so that it has no relation to the jets.

The **selectivity** is the probe's score on the real target minus its score on the control task
(`itj.probes.selectivity`). What each control rules out is part of `W3.1.2`.

```python
y_true = rng.normal(80.0, 10.0, 1000)  # made-up targets, e.g. masses in GeV
y_pred = y_true + rng.normal(0.0, 5.0, 1000)  # a prediction with an error of 5 GeV
r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
print(f"R^2 = {r2:.2f}; predicting the mean for every jet gives R^2 = 0")
```

**Check yourself.** (8a) What is $R^2$ of a probe that predicts $\bar y$ for every jet? (8b) Why is each fold scored on
jets it was not fitted on? (8c) A probe reaches $R^2 = 0.9$ on a trained network and 0.88 on the same network with
random weights. What did training add to this quantity?

**Used in:** `W3.1.2`, `W3.1.4`; `W4.2.9`. **Read more:** [methods.md](methods.md#probes).

## Learning without labels

*Section 9, about 10 minutes.*

Labels are expensive: real data have none, and a simulation's labels may be wrong in ways nobody knows.
**Self-supervised learning** trains a network on unlabelled jets to give a useful embedding, and uses a few labels only
afterwards.

**Augmentations.** An **augmentation** is a change of a jet that should not change what the jet is. Examples are a
rotation about its axis, a small smearing of its soft (low-energy) particles, and the split of one particle into two
collinear ones (task `W3.3`). Collinear particles fly in the same direction. Applying random augmentations to a jet
twice gives two **views** of the same jet.

**Positives and negatives.** In a batch of jets, the two views of one jet form a **positive pair**; the views of all
other jets in the batch are its **negatives**. The network should map the two views of a jet close together and other
jets further away.

**Cosine similarity and temperature.** The network's output vectors $z$ come from a small **projection head** on top of
the encoder (the network that produces the embedding). They are compared by their **cosine similarity**,
$\mathrm{sim}(u, v) = u\cdot v/(\lVert u\rVert\,\lVert v\rVert)$, which lies between $-1$ and 1 and ignores the lengths
of the vectors. Before the comparison the similarities are divided by a **temperature** $\tau_T$.

**The contrastive loss in words.** For each view, take its similarities to every other view in the batch, divided by
$\tau_T$, and turn them into probabilities with a softmax (section 4). The loss of that view is $-\ln$ of the
probability given to its partner, the other view of the same jet: a cross-entropy whose "true class" is the partner.
The mean over all views is the **NT-Xent** loss (normalised temperature-scaled cross-entropy). It falls when the
partners are more similar to each other than to the negatives. What it does in its limiting cases is the subject of
`W3.4.1`.

**Linear evaluation.** To judge an encoder trained without labels, freeze it and train only a logistic regression on
its embedding with a given number of labelled jets. The comparisons that make such a number meaningful (a random
encoder, a supervised network) are those of section 8.

**Check yourself.** (9a) What is the cosine similarity of $v$ and $2v$, and of $v$ and $-v$? (9b) Why may a rotation of
the jet about its axis be used as an augmentation, but a change of its mass not?

**Used in:** `W3.3.3`, `W3.4.1`-`W3.4.4`; the bonus `B.1`. **Read more:** [methods.md](methods.md#contrastive-learning)
(after `W3.4.1`), [methods.md](methods.md#linear-evaluation).

## Domain shift

*Section 10, about 10 minutes.*

A tagger learns from its training data and silently assumes that the jets it meets later come from the same
distribution. **Domain shift** is any difference between the two. In this course it is the toy against JetClass, a
large public set of simulated jets ([jetclass.md](jetclass.md)). At the LHC (the Large Hadron Collider) it is
simulation against real collisions.

- **The inputs change.** The jets of the new domain have another spectrum of $p_T$, more or fewer particles, other
  particle types or another detector (a *covariate shift*).
- **The class fractions change.** The new data hold another mix of signal and background.
- **The meaning changes.** A pattern that marked a W jet in training is common in the background of the new domain.

A tagger that leans on a feature that differs between the domains does worse in the new domain. This holds even when
it was excellent on its own test set. Domain shift is not overfitting, and a held-out test set from the same domain
cannot reveal it.

**Tools to diagnose it.**

- **Evaluate across.** Train on one domain and test on the other, both ways, and compare with training and testing on
  the same domain.
- **Reweighting.** Give each jet of one domain a weight so that the distribution of one variable matches the other
  domain (for example the multiplicity, per class). If the performance recovers, that variable explains part of the
  difference. Other variables are correlated with it, so reweighting points to a cause but does not prove it.
- **Removing inputs.** Remove one kind of particle (for example the photons) in both domains, or retrain without an
  input, and see how the gap changes.

At the LHC every tagger is trained on simulation and used on data. The difference is measured and corrected with
*scale factors*, the ratios of a tagger's efficiency in data and in simulation.

**Check yourself.** (10a) A tagger trained on jets of 500-1000 GeV is used on jets of 300 GeV. Which kind of shift is
that? (10b) Why can a reweighting in one variable not prove that this variable causes the gap?

**Used in:** `W4.2.3`, `W4.2.4`, `W4.2.6`. **Read more:** [jetclass.md](jetclass.md),
[physics.md](physics.md#what-the-toy-leaves-out).

## Gradients of the input

*Section 11, about 10 minutes.*

Training uses the gradient of the loss with respect to the weights. Interpretation uses the gradient of the score with
respect to the **inputs**: $\partial s/\partial x_{i,c}$, how fast the score changes when the input $c$ of particle $i$
changes a little.

**Saliency.** The size of that gradient, reduced to one number per particle, is the **saliency**. The reduction takes
the length of the vector over the particle's inputs. The saliency is a map of the particles to which the score is most
sensitive. It is local, a statement about small changes. A score that is already close to 1 changes little in every
direction, even when some particles are essential.

**Integrated gradients.** Choose a **baseline** $x'$, an input that carries no information (for example all inputs 0).
Walk along the straight line from $x'$ to the jet $x$ in $m$ steps and compute the gradient at each step. Take the
mean, and multiply by how far each input moved:

```math
\mathrm{IG}_{i,c} = (x_{i,c} - x'_{i,c}) \times \frac{1}{m}\sum_{k=1}^{m}
\frac{\partial s}{\partial x_{i,c}}\Bigl(x' + \tfrac{k}{m}(x - x')\Bigr) .
```

**Completeness.** The attributions are the shares of the score given to each input. They add up to the change of the
score along the path, $\sum_{i,c}\mathrm{IG}_{i,c} = s(x) - s(x')$, up to the error of the sum with $m$ steps. In this
way they share out the score between the particles. Check this before you trust a map. Give the baseline with every map,
because the shares depend on it. A map is **signed** (a particle can push the score down) or **absolute**; say which.

An attribution is a claim about the tagger, and the tools of `W4.4` test it: remove the particles it ranks highest and
see whether the score falls.

```python
w = torch.tensor([1.0, -2.0, 0.5])


def score(x):  # a made-up tagger of three inputs
    return torch.sigmoid((w * x).sum(-1) + 0.3 * x[..., 0] * x[..., 1])


x, x0, m = torch.tensor([1.0, 0.5, 2.0]), torch.zeros(3), 64
path = x0 + (torch.arange(1, m + 1) / m)[:, None] * (x - x0)  # m points on the straight line
path.requires_grad_(True)
score(path).sum().backward()  # the gradient at every point of the path
ig = (x - x0) * path.grad.mean(dim=0)
print("IG per input:", ig.numpy().round(4))
print(f"sum {ig.sum():.4f} against s(x) - s(x0) = {score(x) - score(x0):.4f}")
```

**Check yourself.** (11a) A jet has the score 0.999. Why may its saliency be small everywhere? (11b) With the zero
baseline the integrated gradients of a jet add up to 0.4 and its score is 0.9. What is the score of the baseline?

**Used in:** `W4.4.1`, `W4.4.2`, `W4.5.2`. **Read more:** [methods.md](methods.md#attributions).

## Where to read more

*Section 12, optional.*

- **D. Guest, K. Cranmer and D. Whiteson**, *Deep learning and its application to LHC physics*, Ann. Rev. Nucl. Part.
  Sci. 68 (2018) 161, arXiv:1806.11484: machine learning for jets and collider physics, written for physicists.
- **The PyTorch tutorial** *Learn the basics* (pytorch.org/tutorials): tensors, autograd, modules and a training loop
  in an afternoon.
- **The dl4hep notebooks** 1-3, the notebooks this course grew from: a first MLP, a PFN and a small transformer on
  jets.
- **The course docs.** [methods.md](methods.md): metrics, probes, CKA (centred kernel alignment, which compares two
  embeddings), attributions, attention and contrastive learning, with their papers. [models.md](models.md): the
  networks of `itj` and their costs.

The reading list of the course is in [references.md](references.md#reading-list).

## Check yourself: answers

<details>
<summary>1. A tagger is a function</summary>

(1a) $\epsilon_S = 400/1000 = 0.4$, $\epsilon_B = 20/10\,000 = 0.002$, rejection $1/\epsilon_B = 500$. (1b) A random
score does not depend on the class, so any threshold keeps the same fraction of signal and of background. (1c) On
$10\,000/2000 = 5$ background jets that pass. A count of 5 has a Poisson error of about 2.2 (the square root of the
count), about 45 %.

</details>

<details>
<summary>2. Train, validation and test</summary>

(2a) Yes, in a mild form: the mean and spread of the test jets entered the inputs of the training. Use the statistics
of the training set only. (2b) The spread from the training itself: another random seed (or another training set)
gives another model. (2c) It is the truth of the simulation, which a real detector does not provide. A tagger that used
it would be useless on real data, and on the toy it would hand the tagger the answer.

</details>

<details>
<summary>3. Decision trees and boosting</summary>

(3a) $2^3 = 8$ leaves, and three cuts on the way to each. (3b) A tree compares the mass with thresholds. In MeV every
threshold is simply 1000 times larger, and every jet lands in the same leaf. (3c) Both are low: shuffling one leaves
the information in the other, so the tagger loses little either way.

</details>

<details>
<summary>4. Neural networks</summary>

(4a) $-\frac{1}{2}(\ln 0.9 + \ln 0.1) = \frac{1}{2}(0.105 + 2.303) = 1.20$; the wrong jet contributes almost all of
it. (4b) A linear map of a linear map is linear: $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$. (4c) Those of
epoch 6, the lowest validation loss (early stopping with the best state restored).

</details>

<details>
<summary>5. PyTorch in one page</summary>

(5a) `tensor([3., 3., 3.])`: $f = 3x_1 + 3x_2 + 3x_3$. (5b) An error: the last axis has 30 entries, not 7. Move the
features to the last axis first (`x.transpose(1, 2)`). (5c) The activations and the order of the layers: a ReLU has
no weights, so it is not in the `state_dict`.

</details>

<details>
<summary>6. Networks on particles</summary>

(6a) The largest of a set of numbers does not depend on the order in which they are listed. (6b) 64 jets, 7 features
per particle, 95 particle slots (some of them padding: the mask says which). (6c) To 1: they are a softmax over $j$.

</details>

<details>
<summary>7. Embeddings and maps</summary>

(7a) Every direction beyond the first two components: whatever the jets differ in along those directions is invisible
in the picture. The explained variance says how much that is. (7b) Not necessarily: t-SNE keeps neighbourhoods, not
distances between clusters. Measure the difference with a number before you trust the picture.

</details>

<details>
<summary>8. Regression and probes</summary>

(8a) 0: the numerator equals the denominator. (8b) A model scored on the jets it was fitted on looks better than it is
(section 2); the left-out fold plays the role of a test set. (8c) Little: 0.02 of $R^2$. The quantity was already
readable from the random network, so a high score alone says little about what training learned.

</details>

<details>
<summary>9. Learning without labels</summary>

(9a) 1 for $v$ and $2v$ (the same direction), $-1$ for $v$ and $-v$. (9b) A rotation about the axis changes nothing
that distinguishes a W jet from a QCD jet, because the physics is the same in every direction around the jet. The mass
is one of the things that tell them apart, so a representation must not forget it.

</details>

<details>
<summary>10. Domain shift</summary>

(10a) A shift of the inputs (covariate shift): the same kinds of jets, with another $p_T$. The prongs (the hard cores of
energy in the jet) are therefore narrower, and the observables differ. (10b) Other variables are correlated with the
reweighted one and change with it, so the gain may be theirs.

</details>

<details>
<summary>11. Gradients of the input</summary>

(11a) Near 1 the sigmoid or softmax is flat: small changes of any input barely move the score, so every gradient is
small, whatever the particles' role. (11b) $s(x') = s(x) - \sum\mathrm{IG} = 0.9 - 0.4 = 0.5$.

</details>
