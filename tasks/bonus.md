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
30 000 jets of `data/t6_pretrain.parquet`, 8 epochs of batch 256 (about 14 minutes).

A masked particle is an empty token, and ParT without $U$ has no notion of order, so every masked slot of a jet gets the
same encoder output. The decoder therefore adds the $p_T$ rank of each slot (a learned slot embedding), and the encoder
has to write what it infers about the hidden particles into that output.

Compare the masked loss of each target with trivial predictors. One is the mean of the visible particles of the jet. The
other, for $\ln p_T$, is the mean of the visible rank neighbours of the slot. Which targets does the model learn beyond
them? Evaluate the encoder linearly as in **W3.4.4**, against the random encoder. Do it on the class token and on the
mean of the last particle block (`extract(..., layers="all")["pooled/block2"]`), because the masked loss never trains
the class-attention blocks. *Optional:* the masked-autoencoder variant, `mask_particles(..., drop=True)` (the encoder
sees only the visible particles), with `MaskedParticleDecoder(64, 3, num_layers=1)` (about 17 minutes). Does its encoder
learn more or less?

**B.3** *Bonus, 6.5 h.* Shortcut hunting (needs **W3.1**, **W3.2** and **W4.4**). (a) A known shortcut. Train a PFN with
the particle types, `build_model("pfn", feature_set="kinpid")` (about a minute), on `data/t7_shortcut.parquet`. These
are W jets with a $\pi^0$ share of 0.50 instead of the default (the neutral pion decays to photons). Evaluate the PFN on
clean W jets (`data/t2_detector.parquet`). Can saliency, integrated gradients summed by particle type or a probe for the
photon fraction find the shortcut without being told? (b) A hidden shortcut. `data/t7_mystery.parquet` holds one of the
five shortcuts of [physics.md](../docs/physics.md#planted-shortcuts). A first look at the counts per class
(`jetgen info`, the truth columns) is part of the hunt and may already point at a suspect. The task is to show, with the
tools of weeks 3 and 4 and with counterfactual edits, that a tagger trained on the file uses the shortcut. Then remove
it, and only then run `jetgen info data/t7_mystery.parquet --reveal`.

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
