# Methods for representations and interpretability

This page collects the methods of tasks T2 to T7: how a tagger is scored, how one finds out what a trained network
represents, what it looks at, and how it can learn without labels. Each section names the `itj` function that
implements the method and the paper it comes from ([references.md](references.md#reading-list)). The models are in
[models.md](models.md).

## Metrics

**ROC curve.** A tagger gives every jet a score $s$; a cut $s > c$ keeps the signal fraction $\varepsilon_S(c)$ and the
background fraction $\varepsilon_B(c)$. The ROC curve is $\varepsilon_B$ against $\varepsilon_S$; in jet physics it is
drawn as the **background rejection** $1/\varepsilon_B$ on a log axis against $\varepsilon_S$. `itj.metrics.roc`,
`itj.metrics.auc` (the area under $\varepsilon_S(\varepsilon_B)$, 0.5 for a random score, 1 for a perfect one).

**Rejection at a working point.** $R_{50} = 1/\varepsilon_B$ at $\varepsilon_S = 0.5$ (`rejection(y, s, 0.5)`, with
linear interpolation), and $R_{80}$ at $\varepsilon_S = 0.8$. A rejection $R$ measured with $N_B$ background test jets
rests on $N_B/R$ background jets that pass: above $R = N_B/10$ fewer than ten pass, and the statistical error exceeds
30 %. Report such numbers with their bootstrap error and the test-set size.

**One-vs-one scores.** A three-class tagger gives probabilities $(p_\mathrm{QCD}, p_W, p_t)$. For the pair "W against
QCD" the jets of the two classes are kept and scored with

```math
s_{W/\mathrm{QCD}} = \frac{p_W}{p_W + p_\mathrm{QCD}},
```

which is the likelihood ratio of the two classes if the probabilities are calibrated (`one_vs_one(probs, labels,
signal=1, background=0)`). The same works for the ten-class pretrained models with the class indices 7 (W), 8 (top)
and 0 (QCD).

**Bootstrap.** Resample the test jets with replacement $n$ times and take the standard deviation of the metric
(`bootstrap(metric, y, s, n=100)`). It covers the test-set statistics only; the spread between training seeds is a
separate error (train several seeds where the tasks ask for it).

**The results table.** `append_result("results_table.csv", row)` appends one row with the columns
`itj.metrics.RESULT_COLUMNS` (model, training and test data, sizes, AUC and $R_{50}$ for W and top against QCD, $R_{80}$
for top, their bootstrap errors in the `*_err` columns, accuracy, parameters, training time, seed, note). Keep one table
from T2 to T6.

**Train, choose, report.** Train on the train part, choose cuts, epochs and hyperparameters on the val part, and
report on the test part once.

## Probes

A **probe** is a simple model trained to predict a quantity from a frozen representation (1610.01644). If a linear
probe predicts the jet mass from the embedding $h$ of a tagger, the mass is linearly decodable from $h$; it does not
show that the tagger uses it. `itj.probes.extract(model, data)` returns the embeddings (and, with `layers="all"`, the
pooled output of every layer) batch by batch; `fit_probe(X, y)` standardises $X$, fits a ridge regression (continuous
targets, score $R^2$) or a logistic regression (classes, score AUC or accuracy) and returns the mean and standard
deviation of five-fold cross-validation.

A probe score means little alone (1909.03368, 2102.12452):

- **Random-weight control.** The same architecture with random weights (`model.reset_parameters()`): a random
  network already maps the inputs non-linearly, so many quantities are decodable from it. What training adds is the
  difference.
- **Control task.** The same probe on a target whose link to the input is destroyed: shuffled labels
  (`control_task(y, kind="shuffle")`, which keeps the label distribution) or a random map of the classes
  (`kind="random_map"`). A powerful probe can learn the control too.
- **Selectivity** = score on the real task minus score on the control task (`selectivity(real, control)`). A good
  probe has a high score and a high selectivity.
- **Baselines.** Compare with a probe from known observables: if the multiplicity is as predictable from
  $X_\mathrm{IRC} = [m, m_\mathrm{SD}, \tau_1, \tau_2, \tau_3, e_2, e_3, \text{width}]$ as from the embedding, the
  embedding need not encode the multiplicity itself. An IRC-safe network (EFN) should not beat this baseline for an
  IRC-unsafe target.

Probes of truth that only the toy has (the initiator of a QCD jet, the number of emissions, the opening angle of the
W) test what a representation knows beyond its training label.

## Linear CKA

Centred kernel alignment (1905.00414) compares two representations $X \in \mathbb{R}^{n \times p}$ and
$Y \in \mathbb{R}^{n \times q}$ of the same $n$ jets. With centred columns,

```math
\mathrm{CKA}(X, Y) = \frac{\lVert Y^\mathsf{T} X \rVert_F^2}{\lVert X^\mathsf{T} X \rVert_F\, \lVert Y^\mathsf{T} Y \rVert_F} \in [0, 1].
```

It is invariant under orthogonal transformations and isotropic scaling of either representation, but not under
general invertible linear maps (which would make every pair of full-rank representations equal), and it is 1 for
$X = Y$. `itj.probes.linear_cka`, `cka_matrix` for every pair of layers. Compare against a set of physics observables
$X_\mathrm{phys}$ to ask how "observable-like" a layer is. With fewer jets than dimensions CKA is biased upwards: use
several thousand jets.

## Looking at embeddings

PCA (linear, the directions of largest variance), t-SNE and UMAP (1802.03426; non-linear, they preserve
neighbourhoods) project an embedding to two dimensions for plotting. Colour the points by class and by physics
quantities. The non-linear maps distort distances and cluster sizes, and their pictures change with the
hyperparameters and the seed: use them to form hypotheses, then test the hypotheses with probes.

## Attributions

An attribution gives every input particle a number for its contribution to one output $s(x)$ (a class probability or
logit).

- **Saliency** (1312.6034): the gradient $\partial s/\partial x_{i,c}$, reduced to one number per particle by the $L_2$
  norm over the channels $c$ (`itj.interpret.saliency`). Local: it says how $s$ changes for a small change of $x$,
  not how much a particle contributes.
- **Integrated gradients** (1703.01365): integrate the gradient along the straight path from a baseline $x'$ to $x$,

  ```math
  \mathrm{IG}_{i,c}(x) = (x_{i,c} - x'_{i,c}) \int_0^1 \frac{\partial s}{\partial x_{i,c}}\bigl(x' + \alpha(x - x')\bigr)\, d\alpha
  \approx \frac{x_{i,c} - x'_{i,c}}{m}\sum_{k=1}^{m} \frac{\partial s}{\partial x_{i,c}}\Bigl(x' + \tfrac{k}{m}(x - x')\Bigr).
  ```

  **Completeness**: $\sum_{i,c}\mathrm{IG}_{i,c} = s(x) - s(x')$, up to the error of the sum with $m$ steps; check it.
  The ParT score wiggles along the path, so 32 steps can leave a gap of a few to 30 % (measured on ParT-small and
  `ParT_kin`, zero baseline); 128 steps bring it to about 1e-2 and 512 steps below 1e-3. It holds only if every input
  that differs between $x$ and $x'$ is in the gradient: ParT reads the features and the four-vectors (pair features), so
  a baseline that changes both needs `wrt=("features", "vectors")`. Baselines: **zero** (all transformed features 0), or
  **soft** (every $p_T$ and $E$ scaled by $10^{-3}$ and the features recomputed: a jet with the same shape and almost no
  energy). The answer depends on the baseline: report both. `itj.interpret.integrated_gradients` returns the attribution
  per particle (summed over channels) and the completeness gap.
- **SmoothGrad** (1706.03825): average the saliency over copies of the input with small noise added.
- **Gradients on ParT.** Gradients with respect to the four-vectors pass through $\sqrt{\Delta^2}$ of the self-pairs and
  through padded slots, which give NaN; `itj.interpret.input_gradients` handles padding, `requires_grad` and the
  finite-gradient patch, and returns zero for padded slots.

**Do attributions tell the truth?** Test them: a **deletion curve** removes particles in the order of their
attribution (and, as references, by $p_T$ and at random) and plots the score against the fraction removed
(`itj.augment.remove_particles` recomputes the jet); a faithful attribution makes the score fall fastest. On the toy
the truth is known: remove the particles of one W prong (`truth_part_prong`) and compare with the same number of random
particles.

## Attention

ParT's attention weights $A^{(l,h)}_{ij}$ (block $l$, head $h$; rows sum to 1 over the real particles) say how much
particle $i$ reads from particle $j$; the class attention says how much the class token reads from each particle
(`model.trace(batch, attention=True)`). Attention is not explanation (1902.10186): the weights mix value vectors that
may carry little information, other weights can give the same output, and the residual stream bypasses the attention.
Treat attention as a description of the computation, test claims against controls, and compare with attributions.

- **Head statistics.** `itj.interpret.attention_stats(model, data, groups)` measures, per block and head, the mean
  fraction of attention that a particle gives to particles of its own group (prong), against the same fraction for
  uniform attention (each particle gives $1/N$ to every particle). On the toy the groups are the true prongs; on
  JetClass a reco proxy, `subjet_membership(data, n)` (the exclusive-kT subjet of every particle), validated on the
  toy first.
- **Rollout** (2005.00928): multiply the attention matrices of the layers, each averaged over heads and mixed with the
  identity for the residual connection, $\tilde A = \prod_l \bigl(\tfrac{1}{2}(\bar A^{(l)} + \mathbb{1})\bigr)$, and
  read the row of the class token (`attention_rollout`).
- **Memory.** A ParT-sized model stores $(B, H, P, P)$ maps per block: never trace with attention on more than a few
  hundred jets at once; `attention_stats` reduces batch by batch.

## Pair-feature ablation

ParT adds the pair bias $U$ (from $[\ln k_T, \ln z, \ln\Delta, \ln m^2]$) to the attention logits. Three ways to ask
what it does:

- **Regression.** Regress $U^{(h)}_{ij}$ of each head on the four pair features (`itj.interpret.pair_features` returns
  them exactly as ParT computes them): which Lund coordinate does each head encode?
- **Mean imputation** at inference: replace one pair channel by its mean before the pair embedding
  (`ablate_pair_channels(model, [k], fill="mean")`) and measure the change of the AUC. Soft, but the model never saw
  such inputs.
- **Zeroing** $U$ (`zero_pair_bias(model)`) breaks a trained ParT completely (the class-attention inputs move far from
  their training distribution); it measures how much the trained model depends on $U$, not how useful $U$ is. For
  `ParT_kin` on JetClass it drops the accuracy on top jets (argmax over the ten outputs) from about 0.91 to 0.11, most
  jets becoming `Tbl` (after `probs_to_jets3`: from 0.95 to 0.62), while retraining without the pair bias costs only
  about 0.015 in the W against QCD AUC.

The honest ablation is retraining: the same model without $U$ (`partsmall_nopair`), or with fewer pair features
(`pair_input_dim=1`), trained in the same way, compared with errors.

## IRC tests of networks

The protocol of [observables.md](observables.md#numerical-irc-tests), applied to taggers by
`itj.interpret.irc_scan(model, data, mode="soft" | "collinear", values=...)`: the mean $|\Delta s|$ of the score (and of
the logit) against $\epsilon$ or $\delta$, or, with `output="embedding"`, the distance between the embeddings
normalised by the mean embedding norm. An EFN falls linearly to zero; a PFN, ParticleNet or ParT reaches a plateau.

## Mass sculpting and decorrelation

A tagger that has learned the jet mass selects background jets with the signal mass: after a cut, the QCD
$m_\mathrm{SD}$ distribution is **sculpted** into a peak, which ruins a search that fits a bump on a smooth background.
Measure it with the Jensen-Shannon divergence between the background mass distributions before and after the cut
(`itj.metrics.sculpting_jsd`, base 2: 0 for identical, 1 for disjoint distributions); a decorrelated tagger has a large
$1/\mathrm{JSD}$ at a given rejection.

- **Planing** (1709.10106): weight the training jets so that the mass distribution is flat in every class:
  $w = 1/\hat p_c(m)$ from a histogram per class $c$ (for example 20 bins on [40, 250] GeV), normalised so that the
  weights of a class sum to its number of jets, capped (at 10) against empty bins; pass them as
  `JetDataset(weights=...)`. The network can no longer gain from the mass shape. Say what the jets outside the
  histogram get: in T3.5 they keep the weight 1 before the normalisation (two thirds of the QCD jets have
  $m_\mathrm{SD} < 40$ GeV, so dropping them instead, with weight 0, trains a different tagger).
- **DisCo** (2001.05310): add the distance correlation between the score and the mass of background jets to the loss.
- Others: designed decorrelation (DDT, 1603.00027), adversaries (1611.01046), MoDe (2010.09745).

A decorrelated W tagger should still tag two-prong jets at other masses: the shifted-mass W samples of T3.5 test this.

## Contrastive learning

Self-supervised learning builds a representation from unlabelled jets. Contrastive learning (SimCLR, 2002.05709; JetCLR
for jets, 2108.04253) makes two augmented **views** of every jet, encodes both, $h = f(x)$, maps them with a small
**projection head** to $z = g(h)$ (`itj.ssl.ProjectionHead`), and trains $f$ and $g$ so that the two views of a jet are
close and different jets far apart.

**NT-Xent.** For a batch of $B$ jets the $2B$ normalised projections $\hat z_k$ form $B$ positive pairs; every other
projection is a negative. With the temperature $\tau_T$,

```math
\ell_{k} = -\ln\frac{\exp(\hat z_k \cdot \hat z_{k^+}/\tau_T)}{\sum_{m \ne k} \exp(\hat z_k \cdot \hat z_m/\tau_T)},
\qquad \mathcal{L} = \frac{1}{2B}\sum_{k=1}^{2B} \ell_k,
```

where $k^+$ is the other view of the same jet (`itj.ssl.nt_xent`, $\tau_T = 0.1$). If all embeddings are equal,
$\mathcal{L} = \ln(2B - 1)$. A batch of 256 jets gives every positive pair 510 negatives.

**Alignment and uniformity** (Wang and Isola): the alignment $\mathbb{E}\lVert \hat z - \hat z^+ \rVert^2$ (small: views
agree) and the uniformity $\ln\mathbb{E}\, e^{-2\lVert \hat z - \hat z' \rVert^2}$ over pairs of different jets (small:
spread over the sphere) separate the two goals of the loss (`alignment`, `uniformity`).

**Augmentations are symmetries.** The augmentations define what the representation should forget. JetCLR uses
rotations of the jet about its axis (a symmetry of the physics, up to detector effects), translations in
$(\Delta\eta, \Delta\phi)$, smearing of soft particles (an approximate symmetry: soft particles are poorly measured),
and collinear splits (a symmetry of IRC-safe observables only). `itj.ssl.JetCLRAugment` combines the functions of
`itj.augment`; `two_views` makes the pair.

With these augmentations the loss falls close to zero within about two epochs: judge a run by the alignment, the
uniformity and the linear evaluation, not by the loss.

## Linear evaluation

The standard test of a pretrained encoder: freeze it, extract the embeddings of labelled jets and train a logistic
regression on them (`itj.ssl.linear_eval(encoder, train, test, n_labels=...)`, which returns the accuracy and the
one-vs-one AUCs). Always compare with:

- the **random encoder** (the same architecture with random weights): a random network gives a surprisingly good
  representation;
- the **supervised** model of the same architecture, trained on the same labels;
- the **observables** $X_\mathrm{phys} = [m_\mathrm{SD}, \tau_{21}, \tau_{32}, D_2, N, \ln p_T]$.

**Fine-tuning** trains the encoder further with the labels, starting from the pretrained weights. The **few-label**
protocol repeats both with 100 to 10 000 labelled jets (balanced over the classes, several seeds) and plots the AUC
against the number of labels; train every point for the same number of optimiser steps (at least 2000,
`fit(..., min_steps=2000)`) with a validation set of at least 5000 jets, so that small points measure data size, not
under-training. Pretraining pays off where labels are few.

Expected values on JetClass (QCD, W and top), W against QCD AUC with 1 000 labels: random PFN encoder 0.860, JetCLR PFN
after 3 epochs 0.902; with all labels the gain over the random encoder is only 0.015-0.03. On the toy (W against QCD, 1
000 labels, three seeds): random PFN encoder 0.85, JetCLR PFN after 4 epochs 0.91-0.92. If your JetCLR encoder is more
than 0.03 below the toy value, check the augmentations first.

## Masked particle modelling

Masked particle modelling (2401.13537, 2409.12589) is the BERT-like alternative (1810.04805): hide a fraction of the
particles (`itj.ssl.mask_particles(batch, frac=0.3)` replaces their features by zeros and returns the targets and the
mask) and train the encoder to predict the hidden features (for example $\ln p_T$, $\Delta\eta$, $\Delta\phi$ with a
regression loss). The particles have no natural order, so the targets must be matched to the masked slots (in $p_T$
order, as stored). Evaluate the encoder with the linear evaluation above.
