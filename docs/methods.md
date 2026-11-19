# Methods for representations and interpretability

This page collects the methods of weeks 2 to 4 and the bonus. It covers how a tagger is scored, how to find out what a
trained network represents and what it looks at, and how a network can learn without labels. A tagger is a program
that gives each jet a score for how much it looks like signal ([ml_primer.md](ml_primer.md#a-tagger-is-a-function)).
Each section names the `itj` function that implements the method and the paper it comes from
([references.md](references.md#reading-list)). The models are in [models.md](models.md). New to machine learning?
[ml_primer.md](ml_primer.md), *Machine learning from zero*, explains the basic ideas in words: scores and ROC curves,
training and validation, networks, probes, contrastive learning and gradients. The sections that introduce a new idea
start with a short *In words*.

## Metrics

**ROC curve.** A tagger gives every jet a score $s$. The signal jets are the jets you want to find, here W or top
jets. Here the background jets are QCD jets, which start from a single quark or gluon
([primer.md](primer.md#what-a-jet-is)). A cut $s > c$ keeps the fraction $\varepsilon_S(c)$ of the signal jets and the
fraction $\varepsilon_B(c)$ of the background jets. The ROC (receiver operating characteristic) curve is
$\varepsilon_B$ against $\varepsilon_S$. In jet physics it is drawn as the **background rejection** $1/\varepsilon_B$
on a log axis against $\varepsilon_S$. `itj.metrics.roc` computes the curve, and `itj.metrics.auc` the AUC (area under
the curve: the area under $\varepsilon_S(\varepsilon_B)$, 0.5 for a random score, 1 for a perfect one).

**Rejection at a working point.** A working point is one chosen cut on the score. $R_{50} = 1/\varepsilon_B$ at
$\varepsilon_S = 0.5$ (`rejection(y, s, 0.5)`, with linear interpolation), and $R_{80}$ is the same at
$\varepsilon_S = 0.8$. A rejection $R$ measured with $N_B$ background test jets rests on the $N_B/R$ background jets
that pass. Above $R = N_B/10$ fewer than ten pass, and the statistical error exceeds 30 %. Report such numbers with
their bootstrap error (below) and the test-set size.

**One-vs-one scores.** A three-class tagger gives probabilities $(p_\mathrm{QCD}, p_W, p_t)$. To compare one pair of
classes, such as "W against QCD", keep the jets of the two classes and score them with

```math
s_{W/\mathrm{QCD}} = \frac{p_W}{p_W + p_\mathrm{QCD}},
```

which ranks the jets in the same order as the likelihood ratio of the two classes if the probabilities are calibrated
(`one_vs_one(probs, labels, signal=1, background=0)`). The same works for the ten-class pretrained models
([models.md](models.md#pretrained-models)) with the class indices 7 (W), 8 (top) and 0 (QCD).

**Bootstrap.** The bootstrap estimates the statistical error of a metric. Resample the test jets with replacement $n$
times and take the standard deviation of the metric (`bootstrap(metric, y, s, n=100)`). It covers the test-set
statistics only. The spread between training seeds (the random starting points of a training) is a separate error:
train several seeds where the tasks ask for it.

**The results table.** `append_result("results_table.csv", row)` appends one row with the columns
`itj.metrics.RESULT_COLUMNS`. They hold the model, the training and test data and their sizes. They also hold the AUC
and $R_{50}$ for W and top against QCD, $R_{80}$ for top and their bootstrap errors (the `*_err` columns). The last
columns are the accuracy, the parameters, the training time, the random seed and a note. Keep one table from week 2 to
week 4.

**Train, choose, report.** Train on the train part. Choose cuts, epochs and hyperparameters (the settings you fix by
hand) on the validation part. Report on the test part once
([ml_primer.md](ml_primer.md#train-validation-and-test)).

## Probes

*In words.* To find out whether a network "knows" a quantity, freeze it so that its weights no longer change. Then take
its internal representation of each jet and fit the simplest possible model that predicts the quantity from it. If even
a linear fit works, the information is there in an easy-to-read form. The controls below say whether that means
anything.

A **probe** is a simple model trained to predict a quantity from a frozen representation (1610.01644). Suppose a linear
probe predicts the jet mass from the embedding $h$ of a tagger (the vector the network computes for a jet before its
last layer). Then the mass is linearly decodable from $h$, but this does not show that the tagger uses it.
`itj.probes.extract(model, data)` returns the embeddings batch by batch (and, with `layers="all"`, the pooled output
of every layer). `fit_probe(X, y)` standardises $X$ and fits a ridge regression (continuous targets, score $R^2$) or a
logistic regression (classes, score AUC or accuracy); see [ml_primer.md](ml_primer.md#regression-and-probes). It
returns the mean and standard deviation of five-fold cross-validation (five fits, each scored on the fifth of the jets
it did not see).

A probe score means little alone. Compare it with these controls (1909.03368, 2102.12452):

- **Random-weight control.** The same architecture with random weights (`model.reset_parameters()`). A random
  network already maps the inputs non-linearly, so many quantities are decodable from it. What training adds is the
  difference.
- **Control task.** The same probe on a target whose link to the input is destroyed: shuffled labels
  (`control_task(y, kind="shuffle")`, which keeps the label distribution) or a random map of the classes
  (`kind="random_map"`). A probe with many parameters can learn the control too.
- **Selectivity** = score on the real task minus score on the control task (`selectivity(real, control)`). A good
  probe has a high score and a high selectivity.
- **Baselines.** Compare with a probe on known observables, the jet quantities of
  [observables.md](observables.md). Suppose the multiplicity (the number of particles) is as predictable from
  $X_\mathrm{IRC} = [m, m_\mathrm{SD}, \tau_1, \tau_2, \tau_3, e_2, e_3, \text{width}]$ as from the embedding. Then the
  embedding need not encode the multiplicity itself. An IRC-safe network such as the EFN should not beat this baseline
  for an IRC-unsafe target. IRC (infrared and collinear) safe means that an added soft particle or a collinear split
  does not change the output ([observables.md](observables.md#irc-safety)); the EFN (Energy Flow Network) is in
  [models.md](models.md#pfn-and-efn).

Some truth exists only in the toy, the course's own jet simulator ([primer.md](primer.md#the-toy-and-its-files)). It
includes the seed parton of a QCD jet (the quark or gluon that started it), the number of emissions and the opening
angle of the W. Probes of these quantities test what a representation knows beyond its training label.

## Linear CKA

*In words.* CKA asks whether two representations of the same jets are alike: do jets that are close in one also sit
close in the other? It gives one number between 0 and 1, so the layers of two networks can be compared without
matching their neurons one by one.

Centred kernel alignment (1905.00414) compares two representations $X \in \mathbb{R}^{n \times p}$ and
$Y \in \mathbb{R}^{n \times q}$ of the same $n$ jets. With centred columns,

```math
\mathrm{CKA}(X, Y) = \frac{\lVert Y^\mathsf{T} X \rVert_F^2}{\lVert X^\mathsf{T} X \rVert_F\, \lVert Y^\mathsf{T} Y \rVert_F} \in [0, 1].
```

It is invariant under orthogonal transformations (rotations and reflections) and isotropic scaling of either
representation. It is not invariant under general invertible linear maps, which would make every pair of full-rank
representations equal. It is 1 for $X = Y$. `itj.probes.linear_cka` computes it, and `cka_matrix` computes it for every
pair of layers. Compare against a set of physics observables $X_\mathrm{phys}$ to ask how "observable-like" a layer
is. With fewer jets than dimensions CKA is biased upwards: use several thousand jets.

## Looking at embeddings

PCA (principal component analysis: linear, the directions of largest variance), t-SNE and UMAP (1802.03426;
non-linear, they preserve neighbourhoods) project an embedding to two dimensions for plotting
([ml_primer.md](ml_primer.md#embeddings-and-maps)). Colour the points by class and by physics quantities. The
non-linear maps distort distances and cluster sizes, and their pictures change with the hyperparameters and the seed.
Use them to form hypotheses, then test the hypotheses with probes.

## Attributions

*In words.* An attribution asks which particles of one jet made the tagger give its score. The simplest answer is the
gradient: how fast the score changes when a particle changes. Integrated gradients instead add up the gradient along
a path from an "empty" jet to the real one. The shares of all particles then add up to the score of the jet, less the
score of the empty jet.

An attribution gives every input particle a number for its contribution to one output $s(x)$ (a class probability or
logit, the raw network output before the softmax).

- **Saliency** (1312.6034): the gradient $\partial s/\partial x_{i,c}$, reduced to one number per particle by the $L_2$
  norm over the channels $c$, the input features of the particle (`itj.interpret.saliency`). It is local: it says how
  $s$ changes for a small change of $x$, not how much a particle contributes.
- **Integrated gradients** (1703.01365): integrate the gradient along the straight path from a baseline $x'$ (a
  reference input) to $x$,

  ```math
  \mathrm{IG}_{i,c}(x) = (x_{i,c} - x'_{i,c}) \int_0^1 \frac{\partial s}{\partial x_{i,c}}\bigl(x' + \alpha(x - x')\bigr)\, d\alpha
  \approx \frac{x_{i,c} - x'_{i,c}}{m}\sum_{k=1}^{m} \frac{\partial s}{\partial x_{i,c}}\Bigl(x' + \tfrac{k}{m}(x - x')\Bigr).
  ```

  **Completeness**: $\sum_{i,c}\mathrm{IG}_{i,c} = s(x) - s(x')$, up to the error of the sum with $m$ steps; check it.
  The score of ParT (the Particle Transformer, [models.md](models.md#particle-transformer)) wiggles along the path.
  With 32 steps the gap can be a few to 35 % (measured on ParT-small and `ParT_kin`, zero baseline). With 128 steps it
  falls to a few per cent (often carried by one jet), and with 512 steps below 1e-3. Use 128 steps or more for the
  shares. ParticleNet ([models.md](models.md#particlenet)) builds a dynamic kNN graph: it links each particle to its k
  nearest neighbours, recomputed from the features. Its score jumps along the path whenever the neighbours change, so
  the gap stops at about 2 % and more steps do not help. Completeness holds only if every input that differs between
  $x$ and $x'$ is in the gradient. ParT reads the features and the four-vectors (pair features), so a baseline that
  changes both needs `wrt=("features", "vectors")`. Baselines: **zero** (all transformed features 0), or **soft** (every
  $p_T$ and $E$ scaled by $10^{-3}$ and the features recomputed: a jet with the same shape and almost no energy). The
  answer depends on the baseline: report both. `itj.interpret.integrated_gradients` returns the attribution per
  particle (summed over channels) and the completeness gap.
- **SmoothGrad** (1706.03825): average the saliency over copies of the input with small noise added.
- **Gradients on ParT.** Gradients with respect to the four-vectors pass through $\sqrt{\Delta^2}$ of the self-pairs (a
  particle paired with itself) and through padded slots. Padded slots are the empty slots that fill a jet up to a common
  length. Both give NaN. `itj.interpret.input_gradients` handles the padding, `requires_grad` and the finite-gradient
  patch, and it returns zero for padded slots.

**Do attributions tell the truth?** Test them with a **deletion curve**. It removes particles in the order of their
attribution and plots the score against the fraction removed (`itj.augment.remove_particles` recomputes the jet). As
references, remove particles by $p_T$ and at random. A faithful attribution should make the score fall fastest. On the
toy the truth is known: remove the particles of one W prong (`truth_part_prong`) and compare with the same number of
random particles. A prong is the hard subjet that comes from one decay quark.

## Attention

*In words.* In a transformer ([ml_primer.md](ml_primer.md#networks-on-particles)) every particle builds its new
features as a weighted average over the other particles. The network computes these weights, the attention, for every
jet. Looking at them shows which particles read from which, but not necessarily which particles decide the score.

ParT's attention weights $A^{(l,h)}_{ij}$ (block $l$, head $h$; rows sum to 1 over the real particles) say how much
particle $i$ reads from particle $j$. A head is one of several attention patterns that a block computes in parallel.
The class attention says how much the class token, a learned vector that collects the summary of the jet, reads from
each particle (`model.trace(batch, attention=True)`). Attention is not explanation (1902.10186). The weights mix value
vectors that may carry little information, other weights can give the same output, and the residual stream (the skip
connection around each block) bypasses the attention. Treat attention as a description of the computation, test claims
against controls, and compare with attributions.

- **Head statistics.** `itj.interpret.attention_stats(model, data, groups)` measures, per block and head, how much a
  particle attends to its own group (prong). Take the attention that a particle gives to the *other* particles: the
  statistic is the mean fraction of it that goes to particles of its own group. The reference is the same fraction for
  uniform attention over the other particles, $(n_\mathrm{same} - 1)/(N - 1)$. Self-attention ($A_{ii}$) is not
  counted, because a head that attends to the particle itself would otherwise look like a prong head.
  `attention_stats` also returns the mean $A_{ii}$ per block and head (`"self"`). On the toy the groups are the true
  prongs. On JetClass, the public set of simulated jets of the ParT paper ([jetclass.md](jetclass.md)), they come from
  a reco proxy built from the particles alone. This proxy, `subjet_membership(data, n)`, gives the exclusive-kT subjet
  of every particle ([observables.md](observables.md#clustering)) and is validated on the toy first.
- **Rollout** (2005.00928) follows the attention through all layers. Average the attention matrices of each layer over
  heads, mix them with the identity for the residual connection and multiply them,
  $\tilde A = \prod_l \bigl(\tfrac{1}{2}(\bar A^{(l)} + \mathbb{1})\bigr)$. Then read the row of the class token
  (`attention_rollout`).
- **Memory.** A ParT-sized model stores $(B, H, P, P)$ maps per block. Never trace with attention on more than a few
  hundred jets at once; `attention_stats` reduces batch by batch.

## Pair-feature ablation

ParT adds the pair bias $U$ to the attention logits. The pair bias is a learned function of four features of each
particle pair,
$[\ln k_T, \ln z, \ln\Delta, \ln m^2]$ ([models.md](models.md#particle-transformer)). Three ways to ask what it does:

- **Regression.** Regress $U^{(h)}_{ij}$ of each head on the four pair features (`itj.interpret.pair_features` returns
  them exactly as ParT computes them). Which Lund coordinate ([observables.md](observables.md#the-lund-plane)) does
  each head encode?
- **Mean imputation** at inference replaces one pair channel by its mean before the pair embedding
  (`ablate_pair_channels(model, [k], fill="mean")`); then measure the change of the AUC. This is a soft test, but the
  model never saw such inputs. Imputing all four channels at once gives a constant $U$ per head. A constant cancels in
  the softmax, so this is the same as $U = 0$. Imputing one channel moves the inputs off their joint distribution and
  can do more harm than removing $U$.
- **Zeroing** $U$ (`zero_pair_bias(model)`) breaks a trained ParT completely, because the class-attention inputs move
  far from their training distribution. It measures how much the trained model depends on $U$, not how useful $U$ is.
  For `ParT_kin` on JetClass it drops the accuracy on top jets (argmax over the ten outputs) from about 0.91 to 0.11,
  with most jets becoming `Tbl` (the leptonic top class). With the three classes of `probs_to_jets3` it drops from 0.95
  to 0.62. Retraining without the pair bias, in contrast, costs only about 0.015 in the W against QCD AUC.

To measure how useful the pair bias is, retrain. Train the same model without $U$ (`partsmall_nopair`), or with fewer
pair features (`pair_input_dim=1`), in the same way, and compare with errors.

## IRC tests of networks

The protocol of [observables.md](observables.md#numerical-irc-tests) applies to taggers through
`itj.interpret.irc_scan(model, data, mode="soft" | "collinear", values=...)`. It returns the mean $|\Delta s|$ of the
score (and of the logit) against $\epsilon$ or $\delta$. With `output="embedding"` it returns instead the distance
between the embeddings, normalised by the mean embedding norm. An EFN falls linearly to zero; a PFN (Particle Flow
Network), ParticleNet or ParT reaches a plateau.

## Mass sculpting and decorrelation

*In words.* Suppose a tagger has learned that W jets weigh about 80 GeV. Then the QCD jets it lets through will also
weigh about 80 GeV, and the smooth background turns into a fake peak. Decorrelation trains a tagger that cannot use
the mass, so that its cut leaves the background shape alone.

A tagger that has learned the jet mass selects background jets with the signal mass. After a cut, the QCD
$m_\mathrm{SD}$ distribution is **sculpted** into a peak. This is the distribution of the soft-drop mass, the jet mass
after soft, wide-angle particles are removed ([observables.md](observables.md#soft-drop)). A sculpted peak ruins a
search that fits a bump on a smooth background. Measure the sculpting with the Jensen-Shannon divergence (JSD, a measure
of how far apart two distributions are). It compares the background mass distribution of the jets that pass the cut with
that of the jets that fail it (`itj.metrics.sculpting_jsd`). In base 2 it is 0 for identical and 1 for disjoint
distributions. A decorrelated tagger has a large $1/\mathrm{JSD}$ at a given rejection. At a tight cut only about a
hundred QCD jets may pass, and a histogram of so few jets has a large JSD even without any sculpting. Compare with the
JSD of a random cut that keeps as many jets: this is the floor of the measurement.

- **Planing** (1709.10106) weights the training jets so that the mass distribution is flat in every class. The weight
  is $w \propto 1/\hat p_c(m)$, from a histogram per class $c$ (for example 20 bins on [40, 250] GeV). The weights are
  normalised so that the weights of a class sum to its number of jets, and capped (at 10) against empty bins. Pass
  them as `JetDataset(weights=...)`. The network can then no longer gain from the mass shape. The scale of
  $1/\hat p_c$ matters as soon as some jets are outside the histogram, so fix it. In `W3.2.4` a jet in bin $b$ of its
  class gets $w = (n_\mathrm{in}/20)/n_b$, where $n_\mathrm{in}$ is the number of jets of the class in the range and
  $n_b$ the number in the bin. Every filled bin then carries the same total weight, and the in-range weights average
  1. A jet outside the range gets $w = 1$. Both weights are set before the normalisation. Say what the jets outside the
  histogram get: two thirds of the QCD jets have $m_\mathrm{SD} < 40$ GeV. A density per GeV (in-range weights of
  order 200 against 1 outside) or dropping them (weight 0) therefore trains a different tagger.
- **DisCo** (2001.05310) adds to the loss the distance correlation between the score and the mass of background jets.
  Distance correlation is zero only for independent variables, so it also catches non-linear dependence.
- Other methods: designed decorrelation (DDT, 1603.00027), adversaries (1611.01046), MoDe (2010.09745).

A decorrelated W tagger should still tag two-prong jets at other masses. The shifted-mass W samples of `W3.2.5` test
this.

## Contrastive learning

*In words.* Without labels, a network can still learn what makes a jet that jet. Show it two slightly changed copies
of each jet (rotated, with a soft particle smeared or a particle split). Train it so that the two copies of the same
jet land close together and different jets land far apart. The changes define what the representation should ignore.

Self-supervised learning builds a representation from unlabelled jets. Contrastive learning (SimCLR, 2002.05709; JetCLR
for jets, 2108.04253) makes two augmented **views** of every jet (two randomly changed copies) and encodes both,
$h = f(x)$. A small **projection head** maps them to $z = g(h)$ (`itj.ssl.ProjectionHead`). Training adjusts $f$ and
$g$ so that the two views of a jet are close and different jets far apart.

**NT-Xent.** The contrastive loss is NT-Xent (normalised temperature-scaled cross-entropy). For a batch of $B$ jets the
$2B$ normalised projections $\hat z_k$ form $B$ positive pairs; every other projection is a negative. With the
temperature $\tau_T$,

```math
\ell_{k} = -\ln\frac{\exp(\hat z_k \cdot \hat z_{k^+}/\tau_T)}{\sum_{m \ne k} \exp(\hat z_k \cdot \hat z_m/\tau_T)},
\qquad \mathcal{L} = \frac{1}{2B}\sum_{k=1}^{2B} \ell_k,
```

where $k^+$ is the other view of the same jet (`itj.ssl.nt_xent`, $\tau_T = 0.1$). `W3.4.1` works out its limiting
cases.

**Alignment and uniformity** (Wang and Isola) separate the two goals of the loss (`alignment`, `uniformity`). The
alignment $\mathbb{E}\lVert \hat z - \hat z^+ \rVert^2$ is small when the views agree. The uniformity
$\ln\mathbb{E}\, e^{-2\lVert \hat z - \hat z' \rVert^2}$ over pairs of different jets is small when the projections
spread over the sphere.

**Augmentations are symmetries.** The augmentations define what the representation should forget. JetCLR uses:

- rotations of the jet about its axis (a symmetry of the physics, up to detector effects);
- translations in $(\Delta\eta, \Delta\phi)$;
- smearing of soft particles (an approximate symmetry, because soft particles are poorly measured);
- collinear splits (a symmetry of IRC-safe observables only).

`itj.ssl.JetCLRAugment` combines the functions of `itj.augment`, and `two_views` makes the pair.

With these augmentations the loss falls fast in the first epoch or two and then flattens well above its floor. The
floor is the value for perfectly aligned projections spread uniformly over the sphere. Judge a run by the alignment,
the uniformity and the linear evaluation, not by the loss.

**The pretraining loop.** A contrastive pretraining is a few lines of PyTorch around these functions. `itj.ssl` gives
the pieces, not the loop. `W3.4.3` asks for this shape, with three blanks to fill:

```python
for epoch in range(4):
    view1, view2 = ...  # two augmented views of all jets (JetCLRAugment, two_views)
    ds1, ds2 = itj.features.JetDataset(view1, "kin", None), itj.features.JetDataset(view2, "kin", None)
    encoder.train(), head.train()
    for idx in rng.permutation(len(jets))[: len(jets) // 256 * 256].reshape(-1, 256):
        z1, z2 = head(encoder.embed(ds1.batch(idx))), head(encoder.embed(ds2.batch(idx)))
        loss = ...  # the NT-Xent loss
        ...  # the optimiser step
```

Monitor the loss, `alignment`, `uniformity` and a `linear_eval` after each epoch. Compute the last three in evaluation
mode under `torch.no_grad()`, on fixed views of a few thousand jets.

## Linear evaluation

*In words.* To judge a representation learned without labels, freeze it and train only a straight-line classifier
(logistic regression) on top, with a few labelled jets. Whatever that simple classifier can do, the representation
made easy. The comparisons below say how much of it pretraining added.

This is the standard test of a pretrained encoder, the network that maps a jet to its embedding. Freeze it, extract the
embeddings of labelled jets and train a logistic regression on them (`itj.ssl.linear_eval(encoder, train, test,
n_labels=...)`, which returns the accuracy and the one-vs-one AUCs). Always compare with:

- the **random encoder** (the same architecture with random weights): the control every comparison needs;
- the **supervised** model of the same architecture, trained on the same labels;
- the **observables** $X_\mathrm{phys} = [m_\mathrm{SD}, \tau_{21}, \tau_{32}, D_2, N, \ln p_T]$, with a linear model
  (the same probe as for the encoders) and with a BDT (boosted decision trees, as in `W2.2.2`). A linear model cannot
  make a mass window.

**Fine-tuning** trains the encoder further with the labels, starting from the pretrained weights. The **few-label**
protocol repeats both with 100 to 10 000 labelled jets (balanced over the classes, several seeds) and plots the AUC
against the number of labels. Plan at least 2000 optimiser steps for every point (`fit(..., min_steps=2000,
val_every=50, patience=5)`), so that the points with few labels measure data size, not under-training. Early stopping
(ending the training when the validation loss stops falling) needs labelled validation jets. A validation part of
thousands of labelled jets next to 100 training labels gives fine-tuning and training from scratch far more labels
than the linear evaluation, which uses none. Give them a validation part as large as the train part and count it in
the label budget, or train for a fixed number of steps. Compare the pretrained encoder both with the random encoder
and with training from scratch: the two comparisons answer different questions.

Expected values on JetClass (QCD, W and top), W against QCD AUC with 1 000 labels: random PFN encoder 0.860, JetCLR PFN
after 3 epochs 0.902. With all labels the gain over the random encoder is only 0.015-0.03. On the toy (W against QCD,
1 000 labels, three seeds): random PFN encoder 0.85, JetCLR PFN after 4 epochs 0.91-0.92. If your JetCLR encoder is
more than 0.03 below the toy value, check the augmentations first.

## Masked particle modelling

Masked particle modelling (2401.13537, 2409.12589) is the BERT-like alternative (1810.04805); BERT learns language by
filling in hidden words. Hide a fraction of the particles and train the encoder to predict the hidden features, for
example $\ln p_T$, $\Delta\eta$, $\Delta\phi$ with a regression loss. `itj.ssl.mask_particles(batch, frac=0.3)`
replaces their features by zeros and returns the targets and the mask.

The particles have no natural order, and ParT without pair features has no positional information. Every masked slot
of a jet therefore gets the same input and the same output. A regression head on that output alone can only predict
the mean of the hidden particles. The decoder has to know which particle it predicts. `itj.ssl.MaskedParticleDecoder`
adds a learned embedding of the slot (the $p_T$ rank, as stored) to the encoder output before a small head. The encoder
itself then has to infer the hidden particles from the visible ones. The alternative is a set loss that matches
predictions to targets without an order.

With `num_layers > 0` it is the decoder of a masked autoencoder (2111.06377), for `mask_particles(..., drop=True)`,
which removes the masked particles from the encoder input. A learned mask token at each masked slot attends to the
encoded visible particles, and the decoder then does part of the inference itself.

Compare the masked loss with trivial predictors. One is the mean of the visible particles of the jet. For $\ln p_T$,
another is the mean of the visible rank neighbours of the slot, which the $p_T$ order makes almost free. The angles are
the hard part. A regression predicts the conditional mean of each target. The papers instead predict discrete tokens of
the hidden particles (2401.13537), or compare such targets with continuous ones and generative heads (2409.12589).
Evaluate the encoder with the linear evaluation above, on the class token and on the mean of the last particle block.
The masked loss never trains the class-attention blocks.
