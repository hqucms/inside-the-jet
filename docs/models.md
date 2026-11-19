# Inputs and models

This page describes the taggers of `itj.models` (networks that give each jet a score; see
[ml_primer.md](ml_primer.md#a-tagger-is-a-function)): what they read, how they are built and what they return. It also
lists the course presets (named, ready-made model settings) with their CPU and GPU costs, and the pretrained JetClass
models. JetClass is the public set of simulated jets of the ParT paper ([jetclass.md](jetclass.md)). The methods that
study the trained models are in [methods.md](methods.md), and the observables in [observables.md](observables.md). New
to neural networks? [ml_primer.md](ml_primer.md#networks-on-particles), section 6, explains sets of particles, padding
and masks, graphs and attention in words, and [ml_primer.md](ml_primer.md#pytorch-in-one-page) explains PyTorch in one
page.

## Inputs

The networks read the particles of a jet, up to 128 of them sorted by $p_T$ (the transverse momentum, the part of the
momentum across the beam; see [primer.md](primer.md#collider-coordinates)). weaver, the training framework of ParT and
the JetClass paper ([jetclass.md](jetclass.md#weaver-in-one-page)), does the same for JetClass. The three feature sets
(`itj.features.FEATURE_SETS`) are those of the ParT paper. Each per-particle variable $x$ is transformed as
$\mathrm{clip}\bigl((x - c)\,s, -5, 5\bigr)$ with the centre $c$ and scale $s$ of the table, or used raw (no clip) where
the table says "raw":

| feature set | variables | $c$, $s$ |
|---|---|---|
| `kin` (7) | $\ln p_T$ (`part_pt_log`), $\ln E$ (`part_e_log`), $\ln(p_T/p_{T,J})$ (`part_logptrel`), $\ln(E/E_J)$ (`part_logerel`), $\Delta R$ (`part_deltaR`), `part_deta`, `part_dphi` | (1.7, 0.7), (2.0, 0.7), (-4.7, 0.7), (-4.7, 0.7), (0.2, 4.0), raw, raw |
| `kinpid` (13) | `kin` with `part_charge`, `part_isChargedHadron`, `part_isNeutralHadron`, `part_isPhoton`, `part_isElectron`, `part_isMuon` (raw) inserted after `part_deltaR` | |
| `full` (17) | `kinpid` with $\tanh d_0$ (`part_d0`, raw), `part_d0err` (0, 1, clipped to [0, 1]), $\tanh d_z$ (`part_dz`, raw), `part_dzerr` (0, 1, clipped to [0, 1]) inserted after `part_isMuon` | |

with $\Delta R = \sqrt{\texttt{part\_deta}^2 + \texttt{part\_dphi}^2}$ and $p_{T,J}$, $E_J$ the stored `jet_pt`,
`jet_energy`. `itj.features.inputs(data, "kin")` returns the four weaver inputs as float32 arrays. They are identical
to those of weaver 0.5.4 with the YAML files of `jetgen configs`:

| input | shape | content | padding |
|---|---|---|---|
| `pf_points` | $(N, 2, L)$ | `part_deta`, `part_dphi` (ParticleNet's first kNN) | wrap |
| `pf_features` | $(N, F, L)$ | the feature set | wrap |
| `pf_vectors` | $(N, 4, L)$ | $p_x, p_y, p_z, E$ (ParT's pair features, the EFN weights) | wrap |
| `pf_mask` | $(N, 1, L)$ | 1 for a real particle, 0 for padding | constant 0 |

**Padding and masks.** Jets shorter than the length $L$ (128, or the longest jet of the batch) are padded: extra
slots are added so that all jets have the same length. weaver's `wrap` mode fills these slots by repeating each jet's
own particles cyclically. The mask tells the network which slots are real. Wrap padding matters for the four-vectors.
Zero-padded four-vectors give $\Delta = 0$ and $\ln 0$ in ParT's pair features, and their gradients are NaN even where
the mask removes their contribution.

**Batches.** `itj.features.JetBatch` holds one batch as torch tensors, channel-first like weaver. Its fields are
`points` $(B, 2, P)$, `features` $(B, F, P)$, `vectors` $(B, 4, P)$, `mask` $(B, 1, P)$ and `label` $(B)$. It also holds
`weight`, `index` (the row in the source data), `jet` $(B, J)$ (jet-level features for the MLP) and `particle_mask`
$(B, P)$ bool. `JetDataset` keeps the transformed features of a data set as ragged arrays (one array per jet, of its own
length) and pads each batch to its own longest jet. `make_loader` iterates over it.
`itj.models.make_inputs(model, data)` builds the batch a given model needs (its feature set, or its jet features).

## The model API

Every tagger is a `torch.nn.Module` subclass of `itj.models.Tagger`:

```python
from itj.features import JetDataset, make_loader
from itj.models import build_model
from itj.train import fit, predict

model = build_model("pfn", feature_set="kin", n_classes=3)
train_set = JetDataset(train, "kin", "jets3")
history = fit(model, train_set, JetDataset(val, "kin", "jets3"), epochs=10)
probs = predict(model, test)  # (N, 3) softmax probabilities
batch = next(iter(make_loader(JetDataset(test, "kin", "jets3"), 256, shuffle=False)))
logits = model(batch)  # (B, C)
h = model.embed(batch)  # (B, D): the input of the final Linear layer
trace = model.trace(batch)  # Trace: logits, embedding, per-layer outputs; attention=True adds attention maps
```

| method or field | returns |
|---|---|
| `forward(batch)` | logits $(B, C)$, always a tensor (autograd and captum work) |
| `embed(batch)` | the embedding $h$ $(B, D)$: the input of the final Linear layer, for every model |
| `trace(batch, attention=False)` | a `Trace` (below); attention maps only on request |
| `n_parameters()`, `reset_parameters(seed)` | the parameter count; random weights (the control of tasks `W3.1` and `W3.4`) |
| `name`, `feature_set`, `jet_features`, `n_classes`, `config` | what the model reads; `config` rebuilds it with `build_model` |

| `Trace` field | shape | content |
|---|---|---|
| `logits` | $(B, C)$ | |
| `embedding` | $(B, D)$ | as `embed`; `fc(embedding) == logits` |
| `mask` | $(B, P)$ bool | real particles |
| `layers` | $(B, P, D_l)$ each | per-particle outputs: `phi` (PFN, EFN), `edgeconv{i}` (ParticleNet), `embed` and `block{i}` (ParT) |
| `pooled` | $(B, D_l)$ each | masked means of `layers`; PFN and EFN also `latent` (the sum) |
| `attention` | $(B, H, P, P)$ each | ParT `block{i}`: rows sum to 1 over the real particles |
| `class_attention` | $(B, H, 1, 1 + P)$ each | ParT `cls{i}`: the class token's attention over itself and the particles |
| `pair_bias` | $(B, H, P, P)$ | ParT's $U$, the output of the pair embedding |

`save_model(model, path)` writes the weights with the configuration; `load_model(path)` rebuilds the model (also from
a bare weaver state dict when `name` names a preset).

## The models

### MLP

A multi-layer perceptron (a stack of fully connected layers; see [ml_primer.md](ml_primer.md#neural-networks)) on
jet-level features. By default these are the ten JetClass jet branches plus $\tau_{21}$ and $\tau_{32}$ (the
N-subjettiness ratios of [observables.md](observables.md#n-subjettiness)), from `batch.jet`. A `BatchNorm1d` on the
inputs standardises them, as in the dl4hep MLP. Three hidden layers of 128 follow, with batch normalisation, ReLU and
dropout 0.1.

### PFN and EFN

The Particle Flow and Energy Flow Networks (PFN and EFN, 1810.05165) are Deep Sets (1703.06114). Each has a
per-particle network $\Phi$, a permutation-invariant sum over the real particles (the order of the particles does not
matter), and a network $F$ on the sum:

```math
\mathrm{PFN}: \; F\Bigl(\sum_i \Phi(x_i)\Bigr), \qquad \mathrm{EFN}: \; F\Bigl(\sum_i z_i\, \Phi(\Delta\eta_i, \Delta\phi_i)\Bigr),
\qquad z_i = \frac{p_{T,i}}{\sum_j p_{T,j}} .
```

The PFN reads the feature set (Φ: three layers of 128; F: three layers of 128). The EFN reads only the angles
(`batch.points`) and the momentum fractions (from `batch.vectors`). It is therefore IRC (infrared and collinear) safe
by construction: an added soft particle or a collinear split does not change its output
([observables.md](observables.md#irc-safety)). Its sizes are Φ (100, 100, 128) and F (100, 100, 100).
`EFN.filters()` evaluates each latent component of $\Phi$ on a grid in $(\Delta\eta, \Delta\phi)$: these are the
filters the network learned (question `W3.2.2`).

### ParticleNet

ParticleNet (1902.08570) treats the jet as a point cloud, an unordered set of points, and applies EdgeConv blocks
(1801.07829). For each particle $i$ the $k$ nearest neighbours are found: in $(\Delta\eta, \Delta\phi)$ for the first
block, and in the feature space of the previous block after that. Then

```math
x_i' = \frac{1}{k}\sum_{j \in \mathrm{kNN}(i)} h_\Theta\bigl(x_i,\, x_j - x_i\bigr) + \mathrm{shortcut}(x_i),
```

with $h_\Theta$ a stack of 1×1 convolutions, batch normalisation and ReLU. The outputs of the blocks are optionally
fused, averaged over the real particles and passed through fully connected layers. Padded particles are moved far
away, so that they are never neighbours.

### Particle Transformer

ParT (the Particle Transformer, 2202.03772) embeds every particle with an MLP, adds a learned bias for every pair of
particles, and runs transformer blocks followed by class-attention blocks (CaiT, 2103.17239):

1. **Particle embedding**: an MLP (128, 512, 128) on the features.
2. **Pair features** for every pair $(i, j)$ are computed from the four-vectors, in the order
   `[ln kT, ln z, ln Delta, ln m^2]` ([observables.md](observables.md#the-lund-plane)). 1×1 convolutions (64, 64, 64,
   then one output per head) embed them into the **pair bias** $U \in \mathbb{R}^{H \times P \times P}$, computed
   once. A head is one of several attention patterns computed in parallel.
3. **Particle attention blocks** (8, pre-LayerNorm), with the pair bias added to the attention logits of every head:

   ```math
   \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d}} + U\right) V .
   ```

4. **Class attention blocks** (2): a class token, a learned vector that collects the summary of the jet, attends to
   itself and to all particles,
   $\mathrm{softmax}(q_\mathrm{cls}K^\mathsf{T}/\sqrt{d})\,V$, with the particles fixed.
5. **Head**: LayerNorm on the class token (the embedding), then one Linear layer to the logits.

`pair_input_dim` changes the number of pair features, and `pair_embed_dims=None` removes $U$ (a plain transformer:
`partsmall_nopair`). `itj.interpret.ablate_pair_channels` and `zero_pair_bias` change $U$ at inference
([methods.md](methods.md#pair-feature-ablation)).

## Presets

`itj.models.build_model(name)` builds one of these. The training rates are measured end to end, with data handling
and an evaluation per epoch included. They use one 2016 six-core CPU with 6 threads, batch 256 and about 39 particles
per jet, on JetClass QCD, W and top. Laptops from 2022 on are 1.5-3 times faster. GPU numbers are for a GTX 1080 Ti.
The toy is the course's own simulated data ([primer.md](primer.md#the-toy-and-its-files)). On the toy (about 42
particles per jet) the course models train at PFN 8 870, `pnlite` 640, `partsmall` 240 and `partsmall_nopair` 310
jets/s (`examples/train_tagger.py --course-models`, same CPU). The *Start of week* box of week 2 in
[TASKS.md](../TASKS.md) trains the five course models (PFN, EFN, PN-lite, ParT-small and ParT-small without $U$) one
after the other into `models/course/` (`--out models/course`). This takes about 1-1.5 hours in the background, with
`--threads` set to leave two cores free. Weeks 2 to 4 use these models.

| preset | class | inputs | hyperparameters | parameters | CPU training jets/s |
|---|---|---|---|---|---|
| `mlp` | MLP | 12 jet features | BatchNorm input, 3 × 128, BatchNorm, ReLU, dropout 0.1 | about 35 k | > 50 000 |
| `pfn` | PFN | `kin` | Φ 3 × 128, sum, F 3 × 128 | about 85 k | 6 550 |
| `efn` | EFN | $z_i$, $(\Delta\eta, \Delta\phi)$ | Φ (100, 100, 128), F (100, 100, 100) | about 57 k | ≥ 6 550 |
| `pnlite` | ParticleNet | `kin` | EdgeConv [(7, (32, 32, 32)), (7, (64, 64, 64))], fc [(128, 0.1)], fusion on | about 0.05 M | 840 |
| `pn` | ParticleNet | `kin`/`kinpid`/`full` | EdgeConv [(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))], fc [(256, 0.1)], fusion off | 0.37 M | 63 (GPU 1 530) |
| `partsmall` | ParT | `kin` | embed (64, 256, 64), pair (32, 32, 32), 4 heads, 3 + 1 layers | 0.24 M | 290 |
| `partsmall_nopair` | ParT | `kin` | as `partsmall` without $U$ (`pair_embed_dims=None`) | 0.24 M | 410 |
| `partlite` | ParT | `kin` | dl4hep ParT-lite: embed (128, 512, 128), pair (64, 64, 64), 8 heads, 3 + 1 layers | 0.94 M | 156 |
| `part` | ParT | `kin`/`kinpid`/`full` | the JetClass ParT, 8 + 2 layers | 2.14 M | 66 (GPU 1 560) |

ParT-small (`partsmall`) is the course's small transformer ("our ParT-lite"); `partlite` keeps the meaning of the dl4hep
notebooks. `pn` and `part` belong on a GPU. Evaluation is 2-6 times faster than training (on the toy: ParT-small
about 840 jets/s, PFN 13 300). Most of ParT's cost is the pair embedding, which grows as $P^2$, the square of the
number of particles.

## Pretrained models

The ParT and ParticleNet models of the JetClass paper, trained on 100 M jets of all ten classes, load into `itj`
without weaver:

```bash
jetgen fetch models ParT_kin ParT_full ParticleNet_kin    # -> models/pretrained/*.pt (19 MB)
```

```python
from itj.models import load_pretrained

model = load_pretrained("ParT_kin")  # 10 classes, feature set "kin"
```

| name | inputs | size | accuracy, ten classes (paper) |
|---|---|---|---|
| `ParT_kin`, `ParT_kinpid`, `ParT_full` | `kin`, `kinpid`, `full` | 8.65 MB | `ParT_full` 0.861 |
| `ParticleNet_kin`, `ParticleNet_kinpid`, `ParticleNet_full` | `kin`, `kinpid`, `full` | 1.5 MB | `ParticleNet_full` 0.844 |

The checkpoints (the saved weights) are those of the ParT repository at commit `fa63c86`
(`https://github.com/jet-universe/particle_transformer/raw/fa63c86d594ef875c2d25c04bd5afbab5c73b611/models/<name>.pt`),
checked by size and MD5 checksum. The ten outputs are in the JetClass class order: QCD 0, Hbb 1, Hcc 2, Hgg 3, H4q 4,
Hqql 5, Zqq 6, Wqq 7, Tbqq 8, Tbl 9. `itj.jetclass.probs_to_jets3` keeps columns 0, 7 and 8 and renormalises them.
Accuracies measured on the downloaded validation files are in [jetclass.md](jetclass.md#models-and-numbers).
Inference on a CPU runs at about 225 jets/s for `ParT_kin` and 119 jets/s for `ParticleNet_kin`.

## Vendored code and weaver

`itj/models/part.py` and `itj/models/particlenet.py` are copies of `ParticleTransformer.py` and `ParticleNet.py` of
weaver-core v0.5.4 (MIT, [NOTICE.md](../NOTICE.md)). They keep the same parameter names, so that the weaver checkpoints
load (`mod.*` keys, the legacy keys converted as weaver does). Only version 1 of ParT is kept. The differences from
weaver:

- **Attention maps.** weaver uses `scaled_dot_product_attention`, which returns no weights; when traced with
  `attention=True` the vendored ParT computes them explicitly (same outputs).
- **Finite gradients.** The pair features use $\Delta^2 + 10^{-30}$ (`finite_grad=True`), so that the self-pairs
  ($\Delta = 0$, a particle paired with itself) give finite gradients with respect to the four-vectors. The outputs
  change by less than $10^{-5}$. A larger
  constant such as $10^{-16}$ changes the self-pair $\ln k_T$ and the outputs visibly.
- **No trimming.** weaver's `SequenceTrimmer` (which shuffles and trims the particles during training) is off
  (`trim=False`), so particle indices stay fixed.
- **Pair-channel ablation.** `pair_channel_fill` replaces chosen pair features before the pair embedding.

The tests check two things. The vendored models give weaver's outputs with the same weights (to $10^{-6}$). The
pretrained `ParT_kin` and `ParticleNet_kin` reproduce weaver's softmax scores on 150 committed JetClass jets (to
$10^{-5}$). Question
`W4.2.7` checks the same with weaver itself: train with `jets3_kin.yaml` on the split files, predict with the pretrained
model and compare the scores with `itj` ([examples/weaver](../examples/weaver/README.md)).
