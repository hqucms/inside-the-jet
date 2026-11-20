# Bonus: towards a research project (about 24.5 h)

*How to work, what to hand in, the figure checklist and the notation are in [TASKS.md](../TASKS.md).*

## B Bonus

**Goal:** the first steps of a research project. The bonus covers self-supervision on real jets, a pretraining objective
beyond contrastive learning, and a hunt for shortcuts with the tools of the course. A shortcut is a feature that
separates the classes in the training data without being the physics a tagger should use
([physics.md](../docs/physics.md#planted-shortcuts)). The bonus ends with a paper from the current research and a
one-page proposal. It has no tiers: all of it is beyond the core. Each question names the weeks it needs; its data are
made by the week-4 box.

**B.1** *Bonus, 2.75 h.* SSL on JetClass (needs **W3.4** and **W4.1**; *GPU welcome*). Either pretrain on unlabelled
JetClass jets and [evaluate linearly](../docs/methods.md#linear-evaluation) on JetClass, or transfer the toy-pretrained
encoder of **W3.4.3** to JetClass.

**B.2** *Bonus, 7.25 h.* Masked particle modelling (needs **W3.4** and **W2.5**; arXiv:2401.13537, arXiv:2409.12589).
The network sees a jet with some particles hidden and learns to predict them
([methods.md](../docs/methods.md#masked-particle-modelling)). Hide 30 % of the particles of every jet with
`itj.ssl.mask_particles(batch, frac=0.3)`. Train `partsmall_nopair` together with an
`itj.ssl.MaskedParticleDecoder(64, 3)` on its last particle block (`model.trace(masked).layers["block2"]`). The decoder
predicts the masked ($\ln p_T$, $\Delta\eta$, $\Delta\phi$), each standardised, with a regression loss. Use the first
30 000 jets of `data/t6_pretrain.parquet`, 8 epochs of batch 256 (936 steps, about 14 minutes), with AdamW (learning
rate $10^{-3}$, weight decay 0.01). Let the learning rate fall to 0 along half a cosine over the 936 steps (the loop of
the optional part below shows how).

A masked particle is an empty token, and ParT without $U$ has no notion of order, so every masked slot of a jet gets the
same encoder output. The decoder therefore adds the $p_T$ rank of each slot (a learned slot embedding), and the encoder
has to write what it infers about the hidden particles into that output.

Compare the masked loss of each target with trivial predictors. One is the mean of the visible particles of the jet. The
other, for $\ln p_T$, is the mean of the visible rank neighbours of the slot. Which targets does the model learn beyond
them? Evaluate the encoder linearly as in **W3.4.4**, against the random encoder. Do it on the class token and on the
mean of the last particle block (`extract(..., layers="all")["pooled/block2"]`), because the masked loss never trains
the class-attention blocks. Put a simple line next to your encoders: with the same protocol, a logistic regression on
three numbers per jet, the multiplicity $n$, the mass $m$ and $(m - 80\ \mathrm{GeV})^2$
(`itj.observables.compute(jets, ["multiplicity", "mass"])`; 80 GeV is close to the W mass, and the squared distance lets
a linear model cut a window in the mass), gives a W-against-QCD AUC of 0.92 with 1 000 labels. This is a set-up check:
if yours is far from it, check the label draw and the test part before you compare encoders. How does this line change
the way you read the AUCs of the encoders?

*Optional, about 1 hour on top of the 7.25 h (15 minutes of it CPU, in the background):* predict representations instead
of particles. A joint-embedding predictive architecture (JEPA; arXiv:2301.08243 for images; arXiv:2412.05333,
arXiv:2502.03933 and arXiv:2606.14813 for jets;
[methods.md](../docs/methods.md#joint-embedding-predictive-architecture)) hides part of the jet and predicts what a
second network, the *target encoder*, makes of the hidden particles, not their features. The hidden part is one of the
two exclusive kT subjets of every jet (for a W jet, one of its two prongs), found by kT clustering
([observables.md](../docs/observables.md#clustering)) stopped when two subjets are left. The other particles, the
*context*, are rebuilt as a jet of their own, so that nothing of the hidden subjet reaches the encoder. The target
encoder is an exponential moving average (EMA) of the encoder: after every step its weights move a little towards those
of the encoder, and the loss never trains it directly. The loop below is the whole recipe, on the same 30 000 jets with
8 epochs of batch 256 and the optimiser and learning-rate schedule of your MPM run, so that the two compare fairly.

```python
import math

import numpy as np
import torch

import itj.features
import itj.probes
import itj.ssl
from itj.io import load
from itj.models import build_model
from itj.train import set_seed

jets = load("data/t6_pretrain.parquet", max_jets=32_000).drop("label_", "truth_", "aux_")  # never look at the labels
held_out, jets = jets.select(np.arange(30_000, 32_000)), jets.select(np.arange(30_000))
ds = itj.features.JetDataset(jets, "kin", None)
set_seed(0)
model = build_model("partsmall_nopair")
predictor = itj.ssl.JEPAPredictor(64)
ema = itj.ssl.EMATarget(model)  # the target encoder: follows model, never trained by the loss
opt = torch.optim.AdamW([*model.parameters(), *predictor.parameters()], lr=1e-3, weight_decay=0.01)
total = 8 * (len(ds) // 256)  # 936 steps
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
rng, step = np.random.default_rng(0), 0
for epoch in range(8):
    model.train(), predictor.train()
    for idx in rng.permutation(len(ds))[: len(ds) // 256 * 256].reshape(-1, 256):
        batch = ds.batch(idx)
        context, target, _ = itj.ssl.subjet_target_mask(batch, 2, generator=torch.Generator().manual_seed(step))
        view = itj.ssl.context_view(batch, context)  # the context particles as a jet of their own
        trace = model.trace(view.batch)
        pred = predictor(trace.layers["block2"], view, target, cls=trace.embedding)
        loss = itj.ssl.jepa_loss(pred, ema.trace(batch).layers["block2"], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        ema.update(model, itj.ssl.ema_momentum(step, total))
    print(epoch, loss.item(), itj.ssl.effective_rank(itj.probes.extract(model, held_out)["embedding"]))
```

Evaluate `ema.model` and `model` on both readouts as you evaluated the MPM encoder. Compare them with the same network
before pretraining (`set_seed(0)` and `build_model` again). One seed gives one start and one gain: compare the gain of
each run over its own start, and run a second seed (`set_seed(1)` for the start and the pretraining) if time allows
before you answer. Does predicting representations beat predicting particles here? A JEPA can make its loss small by
giving every jet almost the same representation, which is called collapse. The effective rank printed after every epoch
counts how many directions the class tokens of the held-out jets really use: about 1 for a collapsed encoder, at most 64
for ParT-small ([glossary](../docs/glossary.md#machine-learning-and-statistics)). Compute it for your MPM encoder too.
Then ask how much the predictor needs the encoder at all. Train a fresh `JEPAPredictor(64)` for 300 steps (about 2
minutes) against the trained target encoder, with the encoder output replaced by zeros
(`torch.zeros(*view.batch.particle_mask.shape, 64)` and no `cls`), so that it sees only where the particles are. Use the
recipe of the loop so that the loss compares: AdamW with `lr=1e-3` and `weight_decay=0.01` and the cosine schedule of
the loop with `total = 300`. Use the first 300 batches and masks of the loop (`np.random.default_rng(0)` again and
`manual_seed(step)` for the masks). Compare its loss on the held-out jets with that of your trained predictor (a
predictor that always outputs 0 scores 1 on the layer-normalised targets). What do the effective rank and this
positions-only predictor show? The JEPA papers pretrain on far more jets and for far more steps: why might a JEPA need
far more steps or data than this budget? One caveat to think about: in pretraining the encoder never sees a whole jet.
Its input BatchNorm (the layer that standardises the inputs with statistics gathered in training) learns the statistics
of half-jets, and the target encoder copies them. What could that do to the evaluation on full jets? If you try to
repair it by re-estimating the BatchNorm statistics on full jets, judge the repair by the linear evaluation, not by how
close the statistics come. A further idea, not counted in the time: the masked-autoencoder variant of the MPM
(`mask_particles(..., drop=True)` with `MaskedParticleDecoder(64, 3, num_layers=1)`;
[methods.md](../docs/methods.md#masked-particle-modelling)).

**B.3** *Bonus, 6.5 h.* Shortcut hunting (needs **W3.1**, **W3.2** and **W4.4**). (a) A known shortcut. Train a PFN with
the particle types, `build_model("pfn", feature_set="kinpid")` (about a minute; 1.5 minutes with three threads), on
`data/t7_shortcut.parquet`. These are W jets with a $\pi^0$ share of 0.50 instead of the default (the neutral pion
decays to photons). Evaluate the PFN on clean W jets (`data/t2_detector.parquet`). Can saliency, integrated gradients
summed by particle type or a probe for the photon fraction find the shortcut without being told? (b) A hidden shortcut.
`data/t7_mystery.parquet` holds one of the five shortcuts of [physics.md](../docs/physics.md#planted-shortcuts). A first
look at the counts per class (`jetgen info`, the truth columns) is part of the hunt and may already point at a suspect.
The task is to show, with the tools of weeks 3 and 4 and with counterfactual edits, that a tagger trained on the file
uses the shortcut. Then remove it, and only then run `jetgen info data/t7_mystery.parquet --reveal`.

**B.4** *Bonus, 5 h.* A paper from the current research (needs **W4.2.9**). Choose one of arXiv:2412.03673,
arXiv:2512.00210, arXiv:2605.09881 and arXiv:2401.13537. Either reproduce one figure qualitatively on the toy, or probe
every block of `ParT_full` on all ten classes. For the probes use the example file,
`data/jetclass/example/JetClass_example_100k.root`: `itj.jetclass.load_jetclass("example", ("all",), part="test")` gives
2 000 jets of each class. The inference takes about a minute and the probes of every block about 5 minutes. Compare with
arXiv:2605.09881 (probes) and your **W4.2.9**. arXiv:2412.03673 analyses only the attention of the final layer (no
probes). For that paper, compare the attention of the last block with its claim of sparse, nearly binary attention.

**B.5** *Bonus, 3 h.* A research proposal (one page), motivated by one of your results of weeks 1-4. It gives the
question, why it matters, the data, the method, the first experiment and its figure, what would change your mind, the
compute and the risks.

**Deliverables:** one row and one figure (**B.1**, **B.2**); the evidence chain in five sentences and the figure that
gave the shortcut away (**B.3**); one figure (**B.4**); the proposal (**B.5**).

**Hints**
- **B.2:** `subjet_target_mask` also returns the position of the hidden subjet (the `_` of the loop). It is for a
  predictor only: never give it to the encoder. Clustering every batch costs about 0.16 s (about 3 minutes per run);
  the Notes of `help(itj.ssl.subjet_labels)` show how to cluster the 30 000 jets once instead (about 13 s). For the
  positions-only predictor, fix the masks of the held-out jets (one `torch.Generator` seed per batch), so that both
  predictors are scored on the same hidden subjets. For the question of the budget, one thing you could test is the
  EMA schedule (`help(itj.ssl.ema_momentum)`).
- **B.3:** compare the counts of each particle type between the classes before you train anything. Saliency is
  dominated by the kinematic inputs and may not see a shortcut carried by a particle-type flag. Use the gradient of the
  particle-type channels, or integrated gradients summed by particle type. The provenance of the mystery file is the
  record of how it was made. It holds the override and the statistics of the run only in scrambled form: `jetgen info
  --provenance` shows placeholders, and `jetgen info --reveal` prints them. The mapping from the seed to the shortcut,
  and the scrambling, are in the source code, so do not read `itj.toy.mystery` before you are done.
- **B.4:** the example file is a ROOT file (the standard file format of particle physics); `load` reads it with the
  `analysis` extra.

---

- **Previous:** [Week 4: Real jets, and where does a tagger look?](week4.md)
- **All weeks, the rules and the notation:** [TASKS.md](../TASKS.md#the-weeks)
