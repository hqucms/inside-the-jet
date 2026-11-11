# Glossary

The terms of jet physics and of machine learning that the tasks use, in alphabetical order. Each entry names the task
where the term first appears and links to the section that defines it.

## Jet physics

- **Angular ordering.** In a parton shower, successive branchings along a chain happen at smaller and smaller opening
  angles; this is how the shower builds in colour coherence at leading log. T1.2;
  [physics.md](physics.md#parton-shower).
- **Anti-kT, C/A, kT.** The three generalised-kT clustering algorithms ($p = -1, 0, 1$): anti-kT defines the jets
  (round, hard-centred), Cambridge/Aachen orders the particles by angle (soft drop, Lund plane), kT by relative
  transverse momentum (N-subjettiness axes). T1.3; [observables.md](observables.md#clustering).
- **Colour coherence.** Soft gluons at angles larger than the opening of a colour-connected pair see only the total
  colour of the pair; a colour-singlet W radiates little outside its two prongs. T1.7, T3.6;
  [physics.md](physics.md#start-angles-and-colour-coherence).
- **Colour factor ($C_F$, $C_A$).** The strength of soft-gluon radiation off a quark (4/3) and off a gluon (3); their
  ratio 9/4 sets how much more a gluon jet radiates. T1.1; [physics.md](physics.md#lund-density).
- **Containment.** The requirement that every decay quark of a W or top lies inside the jet; JetClass and the toy keep
  only contained W and top jets. T1.1; [physics.md](physics.md#hard-process).
- **Energy correlation functions ($e_2$, $e_3$, $D_2$, $C_2$).** Sums over pairs and triples of particles weighted by
  their momentum fractions and angles; $D_2 = e_3/e_2^3$ separates one-prong from two-prong jets. T1.5;
  [observables.md](observables.md#energy-correlation-functions).
- **Hadronisation.** The transition from partons to hadrons, which perturbation theory does not describe; the toy uses
  independent Field-Feynman fragmentation. T0.2; [physics.md](physics.md#hadronisation).
- **IRC safety.** An observable is infrared and collinear safe if a soft particle or a collinear split does not change
  it; only IRC-safe observables have finite perturbative predictions. T1.6; [observables.md](observables.md#irc-safety).
- **Jet.** A collimated spray of particles from one energetic quark, gluon or decaying resonance, defined by a
  clustering algorithm and a radius ($R = 0.8$ here). T0.2; [physics.md](physics.md#building-the-jet).
- **JetClass.** The data set of 125 million simulated jets in ten classes behind the Particle Transformer paper;
  the course uses three of its classes. T0.4; [jetclass.md](jetclass.md).
- **Leading log.** The approximation that keeps the largest logarithms of the soft and collinear limits; it gives the
  Lund density, the Sudakov factor and the jet-mass peak that T1 checks. T1.2;
  [physics.md](physics.md#leading-log-results).
- **Lund plane.** The plane of $\ln(1/\Delta)$ and $\ln k_t$ of the emissions of a jet, found by declustering the C/A
  tree along the harder branch; soft-collinear emissions fill it with a constant density. T1.3;
  [observables.md](observables.md#the-lund-plane).
- **Multiplicity ($N$).** The number of particles in a jet: a strong quark/gluon separator, but IRC unsafe and
  sensitive to the detector. T1.1; [physics.md](physics.md#multiplicity).
- **N-subjettiness ($\tau_N$, $\tau_{21}$, $\tau_{32}$).** How well a jet is described by $N$ axes; small $\tau_{21}$
  means two prongs, small $\tau_{32}$ three. T1.5; [observables.md](observables.md#n-subjettiness).
- **Parton shower.** The repeated branching of quarks and gluons ($q \to qg$, $g \to gg$, $g \to q\bar q$) that turns
  one parton into many, generated with Sudakov factors. T0.2; [physics.md](physics.md#parton-shower).
- **Pileup and underlying event.** Soft particles from other proton-proton collisions of the same bunch crossing
  (pileup, not in JetClass or the toy) and from the other partons of the same collision (underlying event). T1.4;
  [physics.md](physics.md#underlying-event).
- **Preset.** One rung of the toy's ladder: `partons`, `shower`, `hadrons`, `detector`. T0.2;
  [physics.md](physics.md#presets).
- **Prong.** A hard subjet that comes from one decay quark: a W jet has two, a top jet three, a QCD jet one. T0.2;
  [data_format.md](data_format.md#truth-branches).
- **Soft drop.** A grooming algorithm that removes soft wide-angle branches of the C/A tree; the soft-drop mass
  $m_\mathrm{SD}$ is the standard tagging mass. T1.4; [observables.md](observables.md#soft-drop).
- **Sudakov factor.** The probability of no emission between two scales; the veto algorithm samples the next emission
  with it. T1.2; [physics.md](physics.md#the-veto-algorithm).
- **Sudakov safety.** A property of observables such as $\tau_{21}$ and $D_2$ that are not IRC safe but have a finite
  distribution once the all-order suppression of the region where they are ill defined is included. T1.6;
  [observables.md](observables.md#irc-safety).
- **Veto algorithm.** Sampling the next emission from a simple overestimate and accepting it with the ratio of the
  true to the overestimated rate; it gives exactly the wanted distribution. T1.2;
  [physics.md](physics.md#the-veto-algorithm).

## Machine learning

- **Attention.** In a transformer, each particle forms a weighted average of all particles, with weights
  softmax($QK^T/\sqrt{d}$); attention maps are not explanations on their own. T2.5, T5.3;
  [methods.md](methods.md#attention).
- **AUC, ROC curve and $R_{50}$.** The ROC curve plots the background efficiency against the signal efficiency of a cut
  on the score; the AUC is the area under it and $R_{50}$ the background rejection at 50 % signal efficiency. T2.1;
  [methods.md](methods.md#metrics).
- **CKA.** Centred kernel alignment, a similarity between two representations of the same jets that is invariant
  under rotations and isotropic scaling. T3.4; [methods.md](methods.md#linear-cka).
- **Contrastive learning (NT-Xent).** Self-supervised training that pulls two augmented views of the same jet together
  and pushes other jets apart. T6.2; [methods.md](methods.md#contrastive-learning).
- **Control task.** A probe target with the structure of the real one but no relation to the input (e.g. shuffled);
  the selectivity of a probe is its score minus that of the control. T3.2; [methods.md](methods.md#probes).
- **Deep Sets, PFN and EFN.** Networks of the form $F(\sum_i \Phi(x_i))$, invariant under permutations of the
  particles; the Energy Flow Network weights each term by $z_i$ and is IRC safe by construction. T2.4;
  [models.md](models.md#pfn-and-efn).
- **Embedding.** The vector a network computes for a jet before its last linear layer (`model.embed`); the object of
  T3 and T6. T3.1; [models.md](models.md#the-model-api).
- **Fine-tuning and linear evaluation.** Two ways to use a pretrained encoder: train all its weights on the labelled
  task, or freeze it and train only a linear classifier on its embedding. T6.4;
  [methods.md](methods.md#linear-evaluation).
- **Integrated gradients.** An attribution method that integrates the gradient of the score along a straight path
  from a baseline to the input; the attributions sum to the change of the score (completeness). T5.1;
  [methods.md](methods.md#attributions).
- **Mass sculpting and decorrelation.** A tagger that learns the mass makes the background look like the signal after
  a cut; planing weights, DisCo or adversaries remove the dependence. T3.5;
  [methods.md](methods.md#mass-sculpting-and-decorrelation).
- **Masked particle modelling.** Self-supervised training that hides some particles and asks the network to predict
  them. T7.1; [methods.md](methods.md#masked-particle-modelling).
- **One-vs-one score.** For a multi-class tagger, $s = p_S/(p_S + p_B)$ for one signal and one background class. T2;
  [methods.md](methods.md#metrics).
- **Pair bias ($U$).** The term of the Particle Transformer that adds learned functions of the pair features
  ($\ln k_T$, $\ln z$, $\ln\Delta$, $\ln m^2$) to the attention logits. T2.5, T5.4;
  [models.md](models.md#particle-transformer).
- **ParticleNet.** A graph network on the particle cloud: EdgeConv blocks on the k nearest neighbours. T2.5;
  [models.md](models.md#particlenet).
- **Particle Transformer (ParT).** A transformer on the particles with the pair bias $U$ and class-attention blocks,
  the state of the art on JetClass. T2.5; [models.md](models.md#particle-transformer).
- **Probe.** A small (usually linear) model trained on frozen embeddings to read out a quantity: what it can read is
  present, in that form. T3.2; [methods.md](methods.md#probes).
- **Saliency.** The gradient of the score with respect to the inputs, the simplest attribution. T5.1;
  [methods.md](methods.md#attributions).
- **Shortcut.** A feature that separates the classes in the training data without being the physics a tagger should
  use (here: planted on purpose). T7.2; [physics.md](physics.md#planted-shortcuts).
- **weaver.** The training framework of ParT and of the JetClass paper, configured by a YAML data configuration and
  a Python network configuration. T0.4, T4.6; [jetclass.md](jetclass.md#weaver-in-one-page).
