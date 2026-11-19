# Glossary

The terms of jet physics, machine learning and statistics that the questions use, in alphabetical order. Each entry
names the question where the term is first needed (in the week files that [TASKS.md](../TASKS.md) links) and links to
the section that defines it.
For the basic terms this is a section of the primers, [primer.md](primer.md) (*Jets from zero*) and
[ml_primer.md](ml_primer.md) (*Machine learning from zero*). For the rest it is a section of the reference docs. A
primer section is read just before the question named.

## Jet physics

- **Angular ordering.** In a parton shower, successive branchings along a chain happen at smaller and smaller opening
  angles; this is how the shower builds in colour coherence at leading log. `W1.1.4`;
  [primer.md](primer.md#the-parton-shower), [physics.md](physics.md#parton-shower).
- **Anti-kT, C/A, kT.** The three generalised-kT clustering algorithms ($p = -1, 0, 1$): anti-kT defines the jets
  (round, hard-centred), Cambridge/Aachen orders the particles by angle (soft drop, Lund plane), kT by relative
  transverse momentum (N-subjettiness axes). Anti-kT `W1.1.3`, C/A `W1.2.7`, kT `W1.4.4`;
  [primer.md](primer.md#what-a-jet-is), [observables.md](observables.md#clustering).
- **b quark and B hadron.** The heaviest quark in a jet (about 4.8 GeV) turns into a B hadron (about 5.3 GeV). The B
  hadron flies a few millimetres before it decays, sometimes semileptonically (with a charged lepton and a neutrino).
  Its decay products have large impact parameters. `W1.1.3`; [primer.md](primer.md#the-particles),
  [physics.md](physics.md#hadronisation).
- **Boost.** The change of energies and momenta seen by an observer who moves with a constant velocity. The mass does
  not change, and a boost along the beam shifts every rapidity by the same constant. `W1.1.1`;
  [primer.md](primer.md#four-vectors-and-invariant-mass).
- **Boosted decay and prong separation.** A heavy particle with $p_T \gg m$ decays into products that fly close
  together, as a narrow pair (W) or triplet (top). Their separation in $\Delta R$ follows from
  $m^2 \approx z(1 - z)\,p_T^2\,\Delta R^2$ and shrinks like $1/p_T$. `W1.1.3`;
  [primer.md](primer.md#boosted-w-and-top-jets).
- **Breit-Wigner.** The distribution of the mass of an unstable particle around its nominal mass, with the width
  $\Gamma \approx \hbar/\tau$ ($\tau$ the lifetime). The toy draws the W and top masses from it (`truth_mass`).
  `W1.2.2`; [primer.md](primer.md#the-particles), [physics.md](physics.md#hard-process).
- **Calorimeter cell.** The smallest unit of a calorimeter, a small square in $(\eta, \phi)$ that reports one energy for
  everything that hit it; the electromagnetic cells are much smaller than the hadronic ones. `W1.1.5`;
  [primer.md](primer.md#what-a-detector-sees), [physics.md](physics.md#detector).
- **Colour and $\alpha_s$.** Colour is the charge of the strong force. Quarks and gluons carry it, and they are never
  seen alone (confinement). $\alpha_s$ is the strength of the strong force, about 0.1 at the energies of a jet and
  growing towards 1 GeV. `W1.1.3`; [primer.md](primer.md#the-particles),
  [physics.md](physics.md#running-coupling-and-cutoff).
- **Colour coherence.** Soft gluons at angles larger than the opening of a colour-connected pair see only the total
  colour of the pair; a colour-singlet W radiates little outside its two prongs. `W1.4.7`, `W3.2.6`;
  [physics.md](physics.md#start-angles-and-colour-coherence).
- **Colour factor ($C_F$, $C_A$).** The strength of soft-gluon radiation off a quark (4/3) and off a gluon (3); at
  leading log a gluon emits soft gluons 9/4 times as often as a quark. `W1.1.4`;
  [primer.md](primer.md#the-parton-shower), [physics.md](physics.md#lund-density).
- **Containment.** The requirement that every decay quark of a W or top lies inside the jet; JetClass and the toy keep
  only contained W and top jets. `W1.1.3`; [primer.md](primer.md#what-a-jet-is), [physics.md](physics.md#hard-process).
- **Delta R ($\Delta R$).** The distance of two particles in the plane of rapidity and azimuth,
  $\Delta R^2 = \Delta y^2 + \Delta\phi^2$ with $\Delta\phi$ wrapped into $(-\pi, \pi]$; a jet is a disc of radius
  $R$ in this plane. `W1.1.2`; [primer.md](primer.md#collider-coordinates).
- **Emission variables ($z$, $\theta$, $k_t$).** The momentum fraction of a daughter, the opening angle and the
  transverse momentum $k_t = z(1 - z)E\theta$ of a shower branching. The files store $z$ of the first daughter, so the
  soft fraction is $\min(z, 1 - z)$. `W1.1.4`; [primer.md](primer.md#the-parton-shower).
- **Energy correlation functions ($e_2$, $e_3$, $D_2$, $C_2$).** Sums over pairs and triples of particles weighted by
  their momentum fractions and angles; $D_2 = e_3/e_2^3$ separates one-prong from two-prong jets. `W1.4.1`;
  [primer.md](primer.md#a-first-look-at-the-observables), [observables.md](observables.md#energy-correlation-functions).
- **Four-vector and invariant mass.** Energy and momentum together, $(p_x, p_y, p_z, E)$ in the files. The invariant
  mass $m^2 = E^2 - \lvert\vec p\rvert^2$ is the same for every observer. The mass of a group of particles is that of
  their summed four-vector. `W1.1.1`; [primer.md](primer.md#four-vectors-and-invariant-mass).
- **GeV and natural units.** Energies, momenta and masses are all given in GeV ($10^9$ electronvolts) by setting
  $c = 1$, so that $E^2 = p^2 + m^2$; the proton weighs about 1 GeV. `W1.1.1`;
  [primer.md](primer.md#units-and-energies).
- **Grooming.** Removing the soft radiation at wide angles from a jet (and with it most of the underlying event) to keep
  its hard core. Soft drop is the groomer of the course. `W1.1.6`;
  [primer.md](primer.md#a-first-look-at-the-observables), [observables.md](observables.md#soft-drop).
- **Hadron, pion.** Hadrons are the colour-neutral bound states of quarks that quarks and gluons turn into; most hadrons
  of a jet are pions ($\pi^\pm$, $\pi^0$), with some kaons, protons and neutrons. `W1.1.3`;
  [primer.md](primer.md#the-particles).
- **Hadronisation.** The transition from partons to hadrons, which perturbation theory does not describe; the toy uses
  independent Field-Feynman fragmentation. `W1.1.4`; [primer.md](primer.md#hadronisation-and-the-underlying-event),
  [physics.md](physics.md#hadronisation).
- **Impact parameter ($d_0$, $d_z$).** The distance by which a track misses the collision point; large for the decay
  products of a B hadron, which decays a few millimetres away. `W1.1.5`; [primer.md](primer.md#what-a-detector-sees),
  [physics.md](physics.md#detector).
- **IRC safety.** An observable is infrared and collinear safe if a soft particle or a collinear split does not change
  it; only IRC-safe observables have finite perturbative predictions. `W3.3.1`;
  [primer.md](primer.md#a-first-look-at-the-observables), [observables.md](observables.md#irc-safety).
- **Jet.** A collimated spray of particles from one energetic quark, gluon or decaying resonance, defined by a
  clustering algorithm and a radius ($R = 0.8$ here). `W1.0.2`; [primer.md](primer.md#what-a-jet-is),
  [physics.md](physics.md#building-the-jet).
- **JetClass.** The data set of 125 million simulated jets in ten classes behind the Particle Transformer paper;
  the course uses three of its classes. `W1.1.7`, `W4.1.1`; [jetclass.md](jetclass.md).
- **Leading log.** The approximation that keeps the largest logarithms of the soft and collinear limits; it gives the
  Lund density, the Sudakov factor and the jet-mass peak that week 1 checks. `W1.2.4`;
  [physics.md](physics.md#leading-log-results).
- **Lund plane.** The plane of $\ln(1/\Delta)$ and $\ln k_t$ of the emissions of a jet. The emissions are found by
  undoing the C/A clustering step by step along the harder branch. Soft-collinear emissions fill the plane with a
  roughly constant density. `W1.1.4` (as a
  picture), `W1.2.3`; [primer.md](primer.md#the-parton-shower), [observables.md](observables.md#the-lund-plane).
- **Multiplicity ($N$).** The number of particles in a jet; it changes at every level of the ladder and with the
  detector. `W1.1.6`; [primer.md](primer.md#a-first-look-at-the-observables),
  [physics.md](physics.md#multiplicity).
- **N-subjettiness ($\tau_N$, $\tau_{21}$, $\tau_{32}$).** How well a jet is described by $N$ axes; small $\tau_{21}$
  means two prongs, small $\tau_{32}$ three. `W1.1.6`; [primer.md](primer.md#a-first-look-at-the-observables),
  [observables.md](observables.md#n-subjettiness).
- **Neutrino.** A neutral lepton that crosses every detector unseen; in the toy it comes from semileptonic B-hadron
  decays and takes energy out of the jet. `W1.1.3`; [primer.md](primer.md#the-particles).
- **Paired samples.** Files made with the same seed hold the same hard events, jet by jet, at every preset or setting;
  they are made for differences, not for pooling. `W1.1.5`; [primer.md](primer.md#the-toy-and-its-files),
  [physics.md](physics.md#reproducibility).
- **Particle flow.** The combination of tracks and calorimeter cells into one list of particles: charged particles
  from tracks, photons from the electromagnetic cells, neutral hadrons from the hadronic cells. `W1.1.5`;
  [primer.md](primer.md#what-a-detector-sees), [physics.md](physics.md#detector).
- **Particle identification.** The type particle flow assigns to each particle (`part_isChargedHadron`,
  `part_isNeutralHadron`, `part_isPhoton`, `part_isElectron`, `part_isMuon`). `W1.1.5`;
  [primer.md](primer.md#what-a-detector-sees), [data_format.md](data_format.md#jetclass-branches).
- **Parton shower.** The repeated branching of quarks and gluons ($q \to qg$, $g \to gg$, $g \to q\bar q$) that turns
  one parton into many, generated with Sudakov factors. `W1.1.4`; [primer.md](primer.md#the-parton-shower),
  [physics.md](physics.md#parton-shower).
- **Pi zero ($\pi^0$) and photon.** The neutral pion decays at once into two photons, which the electromagnetic
  calorimeter measures; most photons of a jet come from $\pi^0$ decays. `W1.1.3`; [primer.md](primer.md#the-particles).
- **Pileup and underlying event.** Soft particles from other proton-proton collisions of the same bunch crossing
  (pileup, not in JetClass or the toy) and from the other partons of the same collision (underlying event). `W1.1.4`;
  [primer.md](primer.md#hadronisation-and-the-underlying-event), [physics.md](physics.md#underlying-event).
- **Preset.** One rung of the toy's ladder: `partons`, `shower`, `hadrons`, `detector`. `W1.0.1`;
  [primer.md](primer.md#the-toy-and-its-files), [physics.md](physics.md#presets).
- **Prong.** A hard subjet that comes from one decay quark: a W jet has two, a top jet three, a QCD jet one. `W1.1.3`;
  [primer.md](primer.md#boosted-w-and-top-jets), [data_format.md](data_format.md#truth-branches).
- **Quark and gluon (partons).** The constituents of the proton: quarks (u, d, s, c, b here) and the gluons that carry
  the strong force between them. A QCD jet starts from one of them. `W1.1.1`; [primer.md](primer.md#the-particles).
- **Rapidity ($y$) and pseudorapidity ($\eta$).** Coordinates of the direction along the beam,
  $y = \frac12\ln\frac{E + p_z}{E - p_z}$ and $\eta = -\ln\tan(\theta/2)$, equal for massless particles; differences of
  rapidity do not change under a boost along the beam. `W1.1.2`; [primer.md](primer.md#collider-coordinates).
- **Soft drop.** A grooming algorithm that removes soft wide-angle branches of the C/A tree; the soft-drop mass
  $m_\mathrm{SD}$ is the standard tagging mass. `W1.1.6`; [primer.md](primer.md#a-first-look-at-the-observables),
  [observables.md](observables.md#soft-drop).
- **Sudakov factor.** The probability of no emission between two scales; the veto algorithm samples the next emission
  with it. `W1.2.10`; [physics.md](physics.md#the-veto-algorithm).
- **Sudakov safety.** A property of observables that are not IRC safe but have a finite distribution once the
  all-order suppression of the region where they are ill defined is included. `W3.3.4`;
  [observables.md](observables.md#irc-safety).
- **Threshold and resolution.** A detector reports a particle or cell only above an energy or momentum threshold, and
  measures it with a resolution, a random error that is relatively smaller at larger energies. `W1.1.5`;
  [primer.md](primer.md#what-a-detector-sees), [physics.md](physics.md#detector).
- **Top quark.** The heaviest known particle (about 173 GeV); it decays before it forms a hadron, almost always into a
  b quark and a W, so a top jet has three prongs. `W1.1.1`; [primer.md](primer.md#the-particles).
- **Track.** The path of a charged particle through the tracker, bent by the magnetic field; it gives the momentum and
  the impact parameters. `W1.1.5`; [primer.md](primer.md#what-a-detector-sees), [physics.md](physics.md#detector).
- **Transverse momentum ($p_T$) and azimuth ($\phi$).** The part of a momentum across the beam,
  $p_T = \sqrt{p_x^2 + p_y^2}$, and its angle around the beam, $\phi$ in $(-\pi, \pi]$. `W1.1.2`;
  [primer.md](primer.md#collider-coordinates).
- **Veto algorithm.** Sampling the next emission from a simple overestimate and accepting it with the ratio of the
  true to the overestimated rate; it gives exactly the wanted distribution. `W1.2.10`;
  [physics.md](physics.md#the-veto-algorithm).
- **W boson.** The charged carrier of the weak force (80.4 GeV). It decays at once, about two thirds of the time into a
  quark and an antiquark, so a W jet has two prongs. `W1.1.1`; [primer.md](primer.md#the-particles).

## Machine learning and statistics

- **Attention.** In a transformer, each particle forms a weighted average of all particles, with weights
  softmax($QK^T/\sqrt{d}$); attention maps are not explanations on their own. `W2.5.1`, `W4.5.3`;
  [ml_primer.md](ml_primer.md#networks-on-particles), [methods.md](methods.md#attention).
- **AUC, ROC curve and $R_{50}$.** The ROC (receiver operating characteristic) curve plots the background efficiency
  against the signal efficiency of a cut on the score. The AUC is the area under it, and $R_{50}$ the background
  rejection at 50 % signal efficiency. `W2.1.2`;
  [ml_primer.md](ml_primer.md#a-tagger-is-a-function), [methods.md](methods.md#metrics).
- **Augmentation.** A change of a jet that should not change what it is (a rotation, a soft smearing, a collinear
  split); contrastive learning trains on two augmented views of each jet. `W3.3.1`;
  [ml_primer.md](ml_primer.md#learning-without-labels), [methods.md](methods.md#contrastive-learning).
- **Bootstrap.** An error estimate from resampling the test part with replacement and recomputing the metric each time.
  `W2.1.2`; [ml_primer.md](ml_primer.md#train-validation-and-test), [methods.md](methods.md#metrics).
- **CKA.** Centred kernel alignment: a similarity between two representations of the same jets that does not change
  under rotations and isotropic scaling. `W3.1.5`; [methods.md](methods.md#linear-cka).
- **Contrastive learning (NT-Xent).** Self-supervised training that pulls two augmented views of the same jet together
  and pushes other jets apart. `W3.4.1`; [ml_primer.md](ml_primer.md#learning-without-labels),
  [methods.md](methods.md#contrastive-learning).
- **Control task.** A probe target with the structure of the real one but no relation to the input (e.g. shuffled).
  The selectivity of a probe is its score minus that of the control. `W3.1.2`;
  [ml_primer.md](ml_primer.md#regression-and-probes), [methods.md](methods.md#probes).
- **Cross-validation.** Splitting the data into folds, training on all but one and scoring on the one left out, in
  turn; the spread over the folds is the error of the score. `W3.1.2`;
  [ml_primer.md](ml_primer.md#regression-and-probes).
- **Decision tree and gradient boosting.** A decision tree is a set of nested cuts. Gradient boosting adds many small
  trees, each correcting the errors of the sum before it (a BDT, boosted decision trees). `W2.2.1`;
  [ml_primer.md](ml_primer.md#decision-trees-and-boosting).
- **Deep Sets, PFN and EFN.** Networks of the form $F(\sum_i \Phi(x_i))$, invariant under permutations of the
  particles; the Energy Flow Network weights each term by $z_i$ and is IRC safe by construction. `W2.4.1`;
  [ml_primer.md](ml_primer.md#networks-on-particles), [models.md](models.md#pfn-and-efn).
- **Domain shift.** A difference between the data a tagger was trained on and the data it is used on (here: the toy and
  JetClass), which can make it fail. `W4.2.3`; [ml_primer.md](ml_primer.md#domain-shift).
- **Dropout.** A layer that switches off random units during training (`model.train()`) and none at evaluation
  (`model.eval()`), against overfitting. `W2.3.3`; [ml_primer.md](ml_primer.md#neural-networks).
- **Effective rank.** The number of directions a set of embeddings really spreads over: $\exp$ of the entropy of the
  normalised singular values of the centred embeddings (`itj.ssl.effective_rank`). It is 1 for embeddings on a line and
  at most the width of the embedding. `B.2`; [methods.md](methods.md#joint-embedding-predictive-architecture).
- **EMA target.** A copy of a network whose weights follow the trained network as an exponential moving average (after
  every step, $\theta_\mathrm{target} \leftarrow m\,\theta_\mathrm{target} + (1 - m)\,\theta$) instead of being trained
  by the loss; the target encoder of a JEPA (`itj.ssl.EMATarget`). `B.2`;
  [methods.md](methods.md#joint-embedding-predictive-architecture).
- **Embedding.** The vector a network computes for a jet before its last linear layer (`model.embed`); the object of
  week 3. `W3.1.1`; [ml_primer.md](ml_primer.md#embeddings-and-maps), [models.md](models.md#the-model-api).
- **Fine-tuning and linear evaluation.** Two ways to use a pretrained encoder: train all its weights on the labelled
  task, or freeze it and train only a linear classifier on its embedding. `W3.4.3` (linear evaluation), `W3.4.7`
  (fine-tuning); [methods.md](methods.md#linear-evaluation).
- **Histogram density and Poisson error.** A histogram divided by the number of entries and the bin width, so that its
  area is 1. A bin of $n$ entries has the error $\sqrt n$, scaled the same way. `W1.1.6`;
  [primer.md](primer.md#histograms-and-errors).
- **Input gradient.** The derivative of a tagger's score with respect to its inputs, the start of every attribution
  method. `W4.4.1`; [ml_primer.md](ml_primer.md#gradients-of-the-input).
- **Integrated gradients.** An attribution method that integrates the gradient of the score along a straight path
  from a baseline to the input; the attributions sum to the change of the score (completeness). `W4.4.2`;
  [ml_primer.md](ml_primer.md#gradients-of-the-input), [methods.md](methods.md#attributions).
- **JEPA.** Joint-embedding predictive architecture: self-supervised training that hides part of the input and predicts
  what a target encoder makes of the hidden part (its representation), not the hidden features themselves. `B.2`;
  [methods.md](methods.md#joint-embedding-predictive-architecture).
- **Learning curve.** A metric of a tagger (here the AUC) against the size of its training set; it shows how much
  more data would help. `W2.5.3`, which defines it; [ml_primer.md](ml_primer.md#train-validation-and-test) for the
  parts it is measured on.
- **Linear and logistic regression.** The simplest models of a number (linear, scored by $R^2$) and of a class
  (logistic: a linear function through a sigmoid, trained with the cross-entropy). `W2.3.1`;
  [ml_primer.md](ml_primer.md#neural-networks), [ml_primer.md](ml_primer.md#regression-and-probes).
- **Loss and cross-entropy.** The loss is the number training makes small; for a classifier it is the cross-entropy,
  the mean of $-\ln$ of the probability the model gives to the true class. `W2.3.1`;
  [ml_primer.md](ml_primer.md#neural-networks).
- **Mass sculpting and decorrelation.** A tagger that learns the mass makes the background look like the signal after
  a cut; planing weights, DisCo or adversaries remove the dependence. `W3.2.3`;
  [methods.md](methods.md#mass-sculpting-and-decorrelation).
- **Masked particle modelling.** Self-supervised training that hides some particles and asks the network to predict
  them. `B.2`; [methods.md](methods.md#masked-particle-modelling).
- **Mini-batch and epoch.** Training updates the weights after each mini-batch of examples; one epoch is one pass over
  the training set. `W2.3.2`; [ml_primer.md](ml_primer.md#neural-networks).
- **MLP.** A multi-layer perceptron: linear layers with a non-linear activation between them. `W2.3.2`;
  [ml_primer.md](ml_primer.md#neural-networks).
- **One-vs-one score.** For a multi-class tagger, $s = p_S/(p_S + p_B)$ for one signal and one background class.
  `W2.2.2`; [ml_primer.md](ml_primer.md#a-tagger-is-a-function), [methods.md](methods.md#metrics).
- **Optimiser and learning rate.** The rule that changes the weights along the negative gradient of the loss (here
  AdamW), and the size of its steps. `W2.3.1`; [ml_primer.md](ml_primer.md#neural-networks).
- **Overfitting and early stopping.** A model that learns its training set by heart does worse on new data; early
  stopping ends the training when the validation loss stops falling and keeps the best weights. `W2.3.2`;
  [ml_primer.md](ml_primer.md#neural-networks).
- **Padding mask.** The boolean array that marks the real particles of jets padded with zeros to a common length; a
  network on particles must ignore the padding. `W1.1.1`; [primer.md](primer.md#the-toy-and-its-files),
  [ml_primer.md](ml_primer.md#networks-on-particles).
- **Pair bias ($U$).** The term of the Particle Transformer that adds learned functions of the pair features
  ($\ln k_T$, $\ln z$, $\ln\Delta$, $\ln m^2$) to the attention logits. `W2.5.1`, `W4.5.5`;
  [models.md](models.md#particle-transformer).
- **ParticleNet.** A graph network on the particle cloud: EdgeConv blocks on the k nearest neighbours. `W2.5.1`;
  [ml_primer.md](ml_primer.md#networks-on-particles), [models.md](models.md#particlenet).
- **Particle Transformer (ParT).** A transformer on the particles with the pair bias $U$ and class-attention blocks;
  the best tagger of the JetClass paper. `W2.5.1`; [models.md](models.md#particle-transformer).
- **PCA, t-SNE and UMAP.** Maps of embeddings into two dimensions: PCA keeps the directions of largest variance
  (linear), t-SNE and UMAP keep neighbourhoods (non-linear, distances between clusters mean little). `W3.1.1` (PCA),
  `W3.1.3` (t-SNE, UMAP); [ml_primer.md](ml_primer.md#embeddings-and-maps).
- **Permutation importance.** The loss of performance when one input is shuffled across the validation part; correlated
  inputs share their importance. `W2.2.3`; [ml_primer.md](ml_primer.md#decision-trees-and-boosting).
- **Permutation invariance.** The output of a network on particles does not depend on the order of the particles.
  `W2.4.2`; [ml_primer.md](ml_primer.md#networks-on-particles).
- **Probe.** A small (usually linear) model trained on frozen embeddings to read out a quantity: what it can read is
  present, in that form. `W3.1.2`; [ml_primer.md](ml_primer.md#regression-and-probes),
  [methods.md](methods.md#probes).
- **Representation collapse.** A failure of self-supervised training in which the network gives (nearly) the same
  representation to every input, which can make the loss small without learning anything. In dimensional collapse the
  representations differ only along a few directions; the effective rank shows it. `B.2`;
  [methods.md](methods.md#joint-embedding-predictive-architecture).
- **Saliency.** The gradient of the score with respect to the inputs, the simplest attribution. `W4.4.1`;
  [ml_primer.md](ml_primer.md#gradients-of-the-input), [methods.md](methods.md#attributions).
- **Selectivity.** The score of a probe minus the score of the same probe on a control task. `W3.1.2`;
  [ml_primer.md](ml_primer.md#regression-and-probes), [methods.md](methods.md#probes).
- **Shortcut.** A feature that separates the classes in the training data without being the physics a tagger should
  use (here: planted on purpose). `B.3`; [physics.md](physics.md#planted-shortcuts).
- **Signal and background efficiency, working point.** The fractions $\epsilon_S$ and $\epsilon_B$ of signal and
  background jets that pass a cut on the score; the working point is the cut chosen for use. `W2.1.1`;
  [ml_primer.md](ml_primer.md#a-tagger-is-a-function).
- **Standardisation.** Shifting and scaling each input to mean 0 and standard deviation 1, with the statistics of the
  train part only. `W2.3.3`; [ml_primer.md](ml_primer.md#neural-networks).
- **Temperature ($\tau_T$).** The scale that divides the cosine similarities in the NT-Xent loss before its softmax;
  `W3.4.1` asks what a small value does. `W3.4.1`; [ml_primer.md](ml_primer.md#learning-without-labels).
- **Tensor and autograd.** A tensor is PyTorch's array; autograd records the operations on tensors and computes the
  gradient of a result with respect to its inputs. `W2.3.1`; [ml_primer.md](ml_primer.md#pytorch-in-one-page).
- **Train, validation and test split; leakage.** Train on the train part, choose settings on the validation part and
  report once on the test part. Leakage is any route by which test information enters the training or the choices.
  `W2.1.1`; [ml_primer.md](ml_primer.md#train-validation-and-test).
- **weaver.** The training framework of ParT and of the JetClass paper, configured by a YAML data configuration and
  a Python network configuration. `W4.2.7`; [jetclass.md](jetclass.md#weaver-in-one-page).
