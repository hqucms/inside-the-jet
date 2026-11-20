# Week 3: What do taggers learn? (core path about 13.5 h, stretch about 26.75 h)

*How to work, what to hand in, the figure checklist and the notation are in [TASKS.md](../TASKS.md).*

**Why this week.** A ROC curve ([ml_primer](../docs/ml_primer.md#a-tagger-is-a-function)) says how well a tagger
separates the classes, not what it has learned. A tagger's representation is the set of numbers the network computes
inside to describe a jet. On the toy we know the truth of every jet. This lets us test what the representation holds,
what the tagger uses, which changes of a jet it should ignore and whether a representation can be learned without
labels.

**Thread.** Last week gave trained taggers and their ranking. This week looks inside them, with controls that keep the
claims honest. Then it turns to the physics of what should *not* matter (soft and collinear particles, rotations), and
to a representation trained only to ignore those changes. Next week takes the same questions to real jets and asks
*where* in a jet a tagger looks.

**Concepts this week**, in order of use: embedding and PCA ([ml_primer 7](../docs/ml_primer.md#embeddings-and-maps));
linear and logistic regression, $R^2$ and AUC as probe scores, cross-validation folds, probes, control tasks,
random-weight baselines and selectivity ([ml_primer 8](../docs/ml_primer.md#regression-and-probes)); counterfactual
samples and paired errors; the filters of a network; soft and collinear perturbations and IRC safety
([primer 12](../docs/primer.md#a-first-look-at-the-observables)); augmentations, exact and approximate symmetries;
cosine similarity, softmax with temperature and the NT-Xent loss; projection head, pretraining, alignment and
uniformity; linear evaluation ([ml_primer 9](../docs/ml_primer.md#learning-without-labels)). Stretch adds t-SNE and
UMAP, the quark/gluon probe, CKA, mass sculpting and the Jensen-Shannon divergence, planing weights, Sudakov safety,
fine-tuning and label budgets (each in the [glossary](../docs/glossary.md)).

**This week at a glance** (times in minutes, or hours where marked):

| task | warm-up | core | stretch |
|---|---|---|---|
| start of week | box (15) | | |
| **W3.1** Inside the representation | **W3.1.1** 60 | **W3.1.2** 105 | **W3.1.3** 90, **W3.1.4** 90, **W3.1.5** 105, **W3.1.6** 90 |
| **W3.2** What do the taggers use? | **W3.2.1** 45 | **W3.2.2** 60 | **W3.2.3** 45, **W3.2.4** 2.25 h, **W3.2.5** 45, **W3.2.6** 2 h, **W3.2.7** 30, **W3.2.8** 2 h |
| **W3.3** Safe observables and symmetries | **W3.3.1** 30 | **W3.3.2** 90, **W3.3.3** 60 | **W3.3.4** 75, **W3.3.5** 45, **W3.3.6** 60, **W3.3.7** 30, **W3.3.8** 2 h |
| **W3.4** Learning without labels | **W3.4.1** 45 | **W3.4.2** 90, **W3.4.3** 105, **W3.4.4** 45 | **W3.4.5** 75, **W3.4.6** 2 h, **W3.4.7** 3.5 h |
| summary, hand-in and discussion | | 1 h | |

**Start of week** (15 minutes; the downloads run in the background all week):

```bash
python examples/make_task_data.py --week 3
python examples/make_task_data.py --week 4 --only fetch
```

The first line makes the seven `t3_*` files and `t6_pretrain` of [Data for week 3](#data-for-week-3) (160 000 jets, 0.39
GB, about a minute with 8 workers). The second line starts the downloads of week 4: three JetClass validation files of
374 MB, and the pretrained models of 19 MB and 136 MB for the bonus **B.4**. `jetgen fetch` resumes an interrupted
download. It checks the size and the MD5 checksum (a fingerprint that changes if a file is damaged) of every file, and
it keeps the files that are already complete ([jetclass.md](../docs/jetclass.md#download)).

**Day plan.** Day 1 (about 6.75 h): the start-of-week box (data and downloads), **W3.1.1**-**W3.1.2**,
then **W3.2.1**-**W3.2.2** and **W3.3.1**-**W3.3.2**. Day 2 (about 6.75 h): **W3.3.3**, **W3.4.1**-**W3.4.4**, the
weekly summary and the discussion. Day 3 (optional): the first stretch picks, catch-up. Three shorter days instead:
box, **W3.1**, **W3.2** (4.75 h) / **W3.3**, **W3.4.1** (3.75 h) / **W3.4.2**-**W3.4.4**, summary, discussion (5 h).

**Check-in at the end of day 1:** the downloads for week 4 are complete, and you can extract an embedding
(**W3.1.1**).

**If you are behind** (agreed in advance; each saves the time given):
- *Two-day:* **W3.1.2** with two models (PFN, ParT-small) and their controls (0.5 h; the comparison across four models
  is lost).
- **W3.3.2** with three observables ($m$, $m_\mathrm{SD}$, $N$) (0.25 h; $\tau_2$ and $p_T^D$ are lost).
- **W3.3.3** with `rotate` and `split_collinear` only (0.25 h; the soft smearing is lost).
- *Two-day:* **W3.4.3** with the loss and the final linear evaluation only (0.25 h; alignment and uniformity are lost).
- **W3.4.4** with the self-supervised (SSL) encoder against the random encoder only (0.25 h; the comparison with the
  supervised PFN is lost).

**First stretch picks:** **W3.2.3** (sculpting), **W3.1.4** (the quark/gluon probe), **W3.1.5** (your own
CKA), **W3.4.7** (few labels, in the background).

**Models and data of this week.** The models are the course models of `models/course/` (or your own PFN of **W2.4.3**:
say which). The control is an untrained ParT-small with random weights (`itj.models.build_model("partsmall")`, then
`model.reset_parameters(seed=0)`; 0 is the default seed, and naming it records the seed for your hand-in). The data
are the test part of `data/t2_detector.parquet` (30 000 jets) unless stated otherwise.

## Data for week 3

`python examples/make_task_data.py --week 3` runs these commands (160 000 jets, 0.39 GB, about a minute with 8 workers).
The `t3_w80_*` files are paired with `t3_w80` (the same hard processes, jet by jet) as described under
[Data for week 1](week1.md#data-for-week-1).

```bash
# W3.2.5: W jets at four masses (independent samples: the mass changes the hard process)
jetgen generate --process wqq --w-mass 60 -n 10000 --seed 35 -o data/t3_w60.parquet
jetgen generate --process wqq -n 10000 --seed 35 -o data/t3_w80.parquet
jetgen generate --process wqq --w-mass 100 -n 10000 --seed 35 -o data/t3_w100.parquet
jetgen generate --process wqq --w-mass 120 -n 10000 --seed 35 -o data/t3_w120.parquet
# W3.2.1, W3.2.6: the jets of data/t3_w80.parquet with one thing changed (paired samples)
jetgen generate --process wqq --no-coherence -n 10000 --seed 35 -o data/t3_w80_nocoh.parquet
jetgen generate --process wqq --no-ue -n 10000 --seed 35 -o data/t3_w80_noue.parquet
jetgen generate --process wqq --cell-position centre -n 10000 --seed 35 -o data/t3_w80_centre.parquet
# W3.4: an unlabelled pretraining set (its labels are used only for the evaluation)
jetgen generate -n 90000 --seed 61 -o data/t6_pretrain.parquet
```

## W3.1 Inside the representation (core path about 2.75 h, stretch about 6.25 h)

**Goal:** what the embedding of a tagger contains, measured with probes and their controls. The embedding is the vector
of numbers the network computes for a jet just before its last layer. A probe is a small model that tries to read one
quantity from the embedding.

**W3.1.1** *Warm-up, 60 min.* The embedding of one tagger. Read [ml_primer section
7](../docs/ml_primer.md#embeddings-and-maps). Extract the embedding of the course PFN on the test part of
`data/t2_detector.parquet`. Project it on its first two principal components (the two directions in which the jets
differ most; `sklearn.decomposition.PCA`), coloured by the class and by $m_\mathrm{SD}$. Which property of the jets does
the largest direction of variation follow?

```python
import itj.features
import itj.models
import itj.probes
from itj.io import load

train, val, test = itj.features.split(load("data/t2_detector.parquet"))
model = itj.models.load_model("models/course/pfn.pt")
h = itj.probes.extract(model, test)["embedding"]  # one row per jet
```

**W3.1.2** *Core, 105 min.* Probes with controls. Read [ml_primer section 8](../docs/ml_primer.md#regression-and-probes)
and [methods.md](../docs/methods.md#probes). (a) One probe first. Fit a linear probe (`itj.probes.fit_probe`) from the
PFN embedding to $m_\mathrm{SD}$. Fit the same probe from the embedding of the untrained ParT-small, and once more with
a shuffled target (`itj.probes.control_task(y, kind="shuffle")`). Then compute the selectivity
(`itj.probes.selectivity`: the score of the trained model minus that of the control). Why are the two controls part of
the answer? (b) Then the map. Fit linear probes to $m_\mathrm{SD}$ and to the multiplicity $N$ from the embeddings of
four models: the PFN, the EFN, ParT-small and the untrained ParT-small. The EFN is a PFN that sees only the angles and
momentum fractions of the particles ([models.md](../docs/models.md#pfn-and-efn)). Give each probe its random-weight and
shuffled controls. *Predict first:* which target does ParT-small encode best? The extraction takes about 2 minutes for
the four models, the linear probes under a minute.

**W3.1.3** *Stretch, 90 min.* Maps of the embedding. Draw PCA, t-SNE and UMAP maps of the four embeddings. Unlike PCA,
t-SNE and UMAP are non-linear: they keep neighbouring jets together. Colour the maps by the class, $m_\mathrm{SD}$,
`truth_init_pid` (QCD jets only), `truth_n_primary` and $N$: what organises the space? UMAP needs `pip install -e
".[ml]"` (on Python 3.14, where umap may not install, use t-SNE). About 3 minutes of CPU for the four models.

**W3.1.4** *Stretch, 90 min.* More probes. Repeat the probes of **W3.1.2** for gluon against quark
(`truth_init_pid == 21`, on QCD jets only; it is never a training label). [Primer section
7](../docs/primer.md#the-parton-shower) and **W1.2.5** say why the two differ. Do the same for $\tau_{21}$, $D_2$ and
`truth_n_primary`. Then extend the map: an MLP probe (a small non-linear network; `kind="mlp"`, about 4 minutes), the
PN-lite model, the targets `truth_mass` and `truth_dr_min`, and probes after every block (`extract(..., layers="all")`).

**W3.1.5** *Stretch, 105 min.* Your own linear CKA (write your own). CKA (centred kernel alignment) is one number that
says how alike two representations of the same jets are ([methods.md](../docs/methods.md#linear-cka)). Check yours
against `itj.probes.linear_cka`. Check that it is invariant under orthogonal maps (rotations and reflections) and
isotropic scaling (multiplying every entry by the same factor), but not under general linear maps. Test the last point
with a whitening map $X\Sigma^{-1/2}$: a random linear map hardly shows it.

**W3.1.6** *Stretch, 90 min.* Which model is most like the observables? Compute the CKA (a) between the blocks of
ParT-small and (b) between each model and the observables
$X_\mathrm{phys} = [m_\mathrm{SD}, \tau_{21}, \tau_{32}, D_2, N, \ln p_T]$. Standardise each column first: in GeV the
mass column would dominate the Gram matrices (the jet-by-jet tables of dot products that CKA compares). Which model is
most "observable-like"? *Optional:* a second EFN seed (about 2 minutes); a second ParT-small (about 12 minutes: start it
in the background first, then read); the four models against each other.

**Deliverables:** `W3.1.1_pca.pdf` (**W3.1.1**); `W3.1.2_probes.pdf` (model against target) and the table of probe
scores with their spread over the folds (**W3.1.2**); `W3.1.3_embeddings.pdf` (**W3.1.3**); the extended table
(**W3.1.4**); the `allclose` line and the invariance checks (**W3.1.5**); `W3.1.6_cka.pdf` (**W3.1.6**).

**Hints**
- **W3.1.1:** [methods.md](../docs/methods.md#looking-at-embeddings); an embedding is the input of the last linear layer
  (`model.embed`). `extract` keeps the memory use low (it never computes attention maps).
- **W3.1.2:** a probe score means little on its own: the random model and the control are part of the answer.
  `fit_probe` standardises the inputs and returns $R^2$ (regression) or AUC (classification) per fold. `control_task(y,
  kind="shuffle")` keeps the marginal distribution of the target: the same values, given to the wrong jets.
- **W3.1.3:** distances between t-SNE clusters mean little.
- **W3.1.5**, **W3.1.6:** the CKA invariances are explained in [methods.md](../docs/methods.md#linear-cka).

## W3.2 What do the taggers use? (core path about 1.75 h, stretch about 8.25 h)

**Goal:** what a tagger uses, found by changing one thing in the jets and by looking at its filters. Stretch: what a cut
on the tagger does to the shape of the background mass distribution (sculpting), and a tagger that avoids it.

**W3.2.1** *Warm-up, 45 min.* Does the tagger use the underlying event
([primer](../docs/primer.md#hadronisation-and-the-underlying-event))? Evaluate the course PFN
(`itj.train.predict`) on `data/t3_w80.parquet` and on the same jets without the underlying event
(`data/t3_w80_noue.parquet`). Plot the change of the W score jet by jet. Is the mean change significant? Pair the jets
by `truth_event_id`: check the IDs, and do not assume that the rows are in the same order.

**W3.2.2** *Core, 60 min.* The filters of an EFN. Each filter is one output of the function that the EFN applies to a
particle, drawn as a picture over the jet ([models.md](../docs/models.md#pfn-and-efn)). Train
`itj.models.build_model("efn", phi_sizes=(100, 100, 8))` with `itj.train.fit` as for the course models (10 epochs, about
1.5 minutes). Draw its 8 filters $\Phi_a(\Delta\eta, \Delta\phi)$ with `itj.plot.efn_filters`. Some filters can die in
training: their ReLU unit (which outputs zero for a negative input) is 0 everywhere, and the plot marks them "dead".
Report these filters as a result. Then relate the size of the structures you see to the prong separation of
your **W1.1.3** figure. Evaluate the separation at the median $p_T$ of the training jets as well as at 600 GeV (as in
Fig. 6 of arXiv:1810.05165).

**W3.2.3** *Stretch, 45 min.* Does a cut on the tagger sculpt the mass? Cut the W-against-QCD score of the PFN at
$\epsilon_S = 50\,\%$ (`itj.metrics.threshold_at`). Compare the $m_\mathrm{SD}$ distribution of the QCD jets that pass
with that of those that fail. Quantify the difference with `itj.metrics.sculpting_jsd`, the Jensen-Shannon divergence
(a measure of how far apart two distributions are) between pass and fail. Why would an analyst who fits a W peak on top
of the QCD background mind?

**W3.2.4** *Stretch, 2.25 h.* A decorrelated tagger (one whose cut leaves the background mass shape alone). Retrain the
PFN with planing weights: weights per jet that make $m_\mathrm{SD}$ flat in each class. Write the weights yourself. Use
20 bins in [40, 250] GeV. A jet in bin $b$ of its class gets $w = (n_\mathrm{in}/20)/n_b$, with $n_\mathrm{in}$ jets of
the class in the range and $n_b$ in the bin. Every filled bin then carries the same total weight. A jet outside the
range gets $w = 1$. Then normalise the weights per class to the number of jets of the class, cap them at 10 and pass
them with `JetDataset(weights=...)`. Plot $R_{50}$ against $1/\mathrm{JSD}$ for both PFNs (one training of about 2-3
minutes: the nominal PFN is the course PFN of `models/course/`). Expect only a partial decorrelation (20 bins with a
cap of 10 flatten the W peak only in part), and say what that does to your answer.

**W3.2.5** *Stretch, 45 min.* Two-prong jets at other masses. Evaluate both PFNs of **W3.2.4** on `data/t3_w60.parquet`,
`data/t3_w80.parquet`, `data/t3_w100.parquet` and `data/t3_w120.parquet`: does the decorrelated tagger still tag these
jets? *Optional:* DisCo, which adds a penalty on the correlation between score and mass to the loss (arXiv:2001.05310).

**W3.2.6** *Stretch, 2 h.* Counterfactual jets (the same jets with one thing changed). Evaluate all your week-2 taggers
on `data/t3_w80.parquet` and on the paired `data/t3_w80_noue.parquet`, `data/t3_w80_centre.parquet` and
`data/t3_w80_nocoh.parquet`. Each file has the same hard processes with one switch changed. Without the underlying
event, and with each calorimeter cell's particle placed at the cell's centre, every track from the jet is identical.
Without coherence, the shower itself changes. Plot the change of the score jet by jet for `noue` and `centre`. `nocoh`
shares only the hard process with `t3_w80` (its whole shower is drawn again), so compare its score distribution and mean
with those of `t3_w80` instead. Which change matters most, and which of the **W3.1.2** probe targets change with it?
What does this say about what the taggers use: the underlying event, the cell grid or the radiation pattern of a colour
singlet? A colour singlet is a colour-neutral source such as the W
([physics.md](../docs/physics.md#start-angles-and-colour-coherence)).

**W3.2.7** *Stretch, 30 min.* A second seed and the paper. Train the EFN of **W3.2.2** with another seed (a different
random start can give a different set of filters). Compare both sets with Fig. 6 of arXiv:1810.05165.

**W3.2.8** *Stretch, 2 h.* Your own question. For example: probes on the outputs of the attention blocks, CKA across
training sizes, probes for `truth_part_from_b`.

**Deliverables:** `W3.2.1_noue.pdf` and the mean change with its paired error (**W3.2.1**); `W3.2.2_filters.pdf`
(**W3.2.2**); `W3.2.3_sculpting.pdf` (**W3.2.3**); `W3.2.4_decorrelation.pdf` (**W3.2.4**); `W3.2.5_mass_scan.pdf`
(**W3.2.5**); `W3.2.6_counterfactual.pdf` (**W3.2.6**); a figure of your own (**W3.2.8**).

**Hints**
- **W3.2.1:** [physics.md](../docs/physics.md#reproducibility) says what the paired files share. The error of a mean of
  differences is not the error of a difference of means.
- **W3.2.2:** the toy jets are not rotated to a common orientation. Before you look at the filters, think about which
  shapes a filter needs to respond to a prong. Also ask how the answer would change if every jet were rotated so that
  its prongs lined up.
- **W3.2.3:** few QCD jets pass the cut, which sets a floor to the divergence. Compare with a random cut that keeps as
  many jets. [methods.md](../docs/methods.md#mass-sculpting-and-decorrelation) explains the planing weights
  of **W3.2.4** as well.

## W3.3 Safe observables and symmetries (core path about 3 h, stretch about 5.5 h)

**Goal:** which observables a soft or collinear particle cannot change, and which changes of a jet a representation
should forget. A soft particle carries a tiny fraction of the jet's momentum. A collinear split replaces one particle by
two that move in almost the same direction.

**W3.3.1** *Warm-up, 30 min.* One soft particle. Read [primer section
12](../docs/primer.md#a-first-look-at-the-observables) again. Add one soft particle to each of 1 000 jets of
`data/t1_detector.parquet`, with a fraction $\epsilon = 10^{-1}, \ldots, 10^{-6}$ of the jet $p_T$. Plot the mean
relative change of the mass and of the multiplicity against $\epsilon$ on log axes. Which one goes to zero?

```python
import itj.augment
import itj.observables
from itj.io import load

jets = load("data/t1_detector.parquet").head(1000)
soft = itj.augment.add_soft(jets, eps=1e-3, substructure=True, rng=0)  # new part_* columns; rng: reproducible
values = itj.observables.compute(soft, ["mass", "multiplicity"])
```

**W3.3.2** *Core, 90 min.* Which observables are IRC safe? *Predict first:* which of $m$, $m_\mathrm{SD}$, $\tau_2$, $N$
and $p_T^D = \sqrt{\sum_i z_i^2}$ (the momentum fractions $z_i$ of the particles squared and summed;
[observables.md](../docs/observables.md#angularities-and-pt_d)) are IRC safe? Then perturb the 1 000 jets in two ways,
both with `substructure=True`: with soft particles (`add_soft`), and with collinear splits
(`itj.augment.split_collinear`, momentum share $\lambda = 0.5$ and 0.2, separation $\delta = 10^{-1}, \ldots, 10^{-5}$;
pass `rng=0` to both, as in **W3.3.1**, so that the scan is reproducible). Plot the mean relative change of the five
observables (`compute(data, ["mass", "sdmass", "tau2", "multiplicity", "ptd"])`, about 25 s) against $\epsilon$ and
$\delta$, and classify each as safe or unsafe. Then check your classification against
[observables.md](../docs/observables.md#irc-safety).

**W3.3.3** *Core, 60 min.* Augmentations are symmetries. An augmentation is a change of a jet that should not change
what the jet is ([ml_primer](../docs/ml_primer.md#learning-without-labels)). Apply `itj.augment.rotate`, `smear_soft`
and `split_collinear` (with `substructure=True`; about 30 s) to 2 000 jets, and tabulate the change of $m_\mathrm{SD}$,
$\tau_{21}$, $N$ and $p_T^D$. Which are symmetries of the physics, which only of IRC-safe observables and which only
approximate? What should a representation trained to ignore them forget?

**W3.3.4** *Stretch, 75 min.* All nine observables and Sudakov safety. Run the scans of **W3.3.2** for $m$,
$m_\mathrm{SD}$, $\tau_2$, $\tau_{21}$, $e_2$, $D_2$, $N$, $p_T^D$ and the leading fraction (the momentum fraction of
the hardest particle). Then sort them into the classes safe, Sudakov safe (a weaker property than IRC safety) or unsafe
([observables.md](../docs/observables.md#irc-safety)). Why can a scan on single jets with many particles not tell an
IRC-safe observable from a Sudakov-safe one? (Hint: which limit does IRC safety concern, and which one does the scan
take?) These curves are the template of **W4.3**.

**W3.3.5** *Stretch, 45 min.* The corner cases. Look at a two-particle jet, where $\tau_{21} = 0/0$. Then scan
$m_\mathrm{SD}$ at $\lambda = 0.05$, below the soft-drop threshold $z_\mathrm{cut} = 0.1$ (see the pitfalls of
[observables.md](../docs/observables.md#numerical-irc-tests)), and find the jets that change.

**W3.3.6** *Stretch, 60 min.* Your own rotation (write your own). Rotate every jet about its $p_T$-weighted axis and
compare with `itj.augment.rotate`.

**W3.3.7** *Stretch, 30 min.* More augmentations. Add `itj.augment.translate`, `drop_soft` and $D_2$ to the table
of **W3.3.3**.

**W3.3.8** *Stretch, 2 h.* Is $N$ more than the safe observables? For $N$, add a baseline to the probes of **W3.1.2**: a
probe from the observables $X_\mathrm{IRC} = [m, m_\mathrm{SD}, \tau_1, \tau_2, \tau_3, e_2, e_3, \text{width}]$
(compare this list with your classes of **W3.3.4**). Is $N$ more predictable from the embeddings than from these
observables alone? *Predict first:* does the EFN beat that baseline? In your answer, say that the baseline is linear and
that its $m_\mathrm{SD}$ column makes an $m_\mathrm{SD}$ probe trivial; the MLP probe is the fairer comparison.

**Deliverables:** `W3.3.1_soft.pdf` (**W3.3.1**); `W3.3.2_irc.pdf` and the classification (**W3.3.2**);
`W3.3.3_augmentations.pdf` or the table (**W3.3.3**); the nine curves and the classes (**W3.3.4**); the jets that change
(**W3.3.5**); the `allclose` line (**W3.3.6**); the baseline next to the probes (**W3.3.8**).

**Hints**
- **W3.3.1:** [primer section 12](../docs/primer.md#a-first-look-at-the-observables) says what IRC safety means; keep
  [observables.md](../docs/observables.md#irc-safety) for after your classification of **W3.3.2**. `itj.augment` returns
  new `part_*` columns, so call the observables again on the result.
- **W3.3.2:** about 0.2 % of the jets have $m_\mathrm{SD} = 0$, which makes the mean of $\lvert\Delta O\rvert/O$
  infinite. Leave out the jets with $O = 0$ before the perturbation, or plot the median of $\lvert\Delta O\rvert/O$ as
  well. Without `substructure=True` the stored $m_\mathrm{SD}$ and $\tau_N$ are not recomputed and look perfectly safe.
  `split_collinear` returns its pieces in double precision (float64). `float64=False` stores them as float32, rounded
  piece by piece, which gives a spurious mass of about 0.1 GeV per piece of a hard particle.
- **W3.3.3:** [methods.md](../docs/methods.md#contrastive-learning), the paragraph "Augmentations are symmetries";
  rotations are about the jet axis.
- **W3.3.6:** rotate about the axis of the summed four-momentum, not about (0, 0).

## W3.4 Learning without labels (core path about 4.75 h, stretch about 6.75 h)

**Goal:** a jet representation learned from unlabelled jets with the augmentations of **W3.3** (JetCLR,
arXiv:2108.04253), and what it knows. The method is contrastive learning: the network learns to place two augmented
copies of the same jet close together and different jets far apart.

The encoder (the network that turns a jet into an embedding) is `itj.models.build_model("pfn")`. On top of it sits a
projection head (a small network whose output enters the loss), `itj.ssl.ProjectionHead(128, 128, 64)` (the fast
default). The pretraining uses `data/t6_pretrain.parquet`. Do not look at its labels: drop the `label_`, `truth_` and
`aux_` columns first and do not print the file. The evaluation uses `data/t2_detector.parquet`.

**W3.4.1** *Warm-up, 45 min.* The contrastive loss on paper. Read [ml_primer section
9](../docs/ml_primer.md#learning-without-labels). Work out on paper the NT-Xent loss (normalised temperature-scaled
cross-entropy), with cosine similarity and a softmax with temperature, for a batch of $B$ pairs in two cases. First, the
two views (augmented copies) of every jet are identical: what does the loss still depend on? Second, all embeddings are
equal: what value does it take? What does a small temperature do?

**W3.4.2** *Core, 90 min.* Your own NT-Xent (write your own). Read
[methods.md](../docs/methods.md#contrastive-learning), which gives the loss as a formula. Write the NT-Xent loss and
compare it with `itj.ssl.nt_xent(z1, z2, temperature=...)` on random tensors (`torch.randn`). Check your two limits
of **W3.4.1** numerically, and plot the loss of random embeddings against the temperature.

**W3.4.3** *Core, 105 min.* Pretrain an encoder. Pretrain the PFN encoder with the projection head on the 90 000 jets of
`data/t6_pretrain.parquet` for 4 epochs (about 3 minutes). Use two views per jet with `itj.ssl.JetCLRAugment` and
`two_views`, batch 256 and $\tau_T = 0.1$. Fill the three blanks of the loop below (the two views, the loss, the step).
After each epoch, record the loss, the alignment, the uniformity and a linear evaluation with 1 000 labels. The
alignment says how close the two views of a jet land, the uniformity how evenly the jets spread. The four monitoring
calls are given, and you decide where they go. Start the run, then read the hints while it trains. Save the encoder as
`models/toy/jetclr_pfn.pt`.

```python
import numpy as np
import torch

import itj.features
import itj.models
import itj.ssl
import itj.train
from itj.io import load

jets = load("data/t6_pretrain.parquet").drop("label_", "truth_", "aux_")  # never look at the labels
train, val, test = itj.features.split(load("data/t2_detector.parquet"))  # labelled jets, for the evaluation only
itj.train.set_seed(0)  # the initial weights; record your seeds
encoder = itj.models.build_model("pfn")
head = itj.ssl.ProjectionHead(128, 128, 64)
opt = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], lr=1e-3)
augment, rng = itj.ssl.JetCLRAugment(), np.random.default_rng(0)
for epoch in range(4):
    view1, view2 = ...  # blank 1: two augmented views of all jets
    ds1, ds2 = itj.features.JetDataset(view1, "kin", None), itj.features.JetDataset(view2, "kin", None)
    encoder.train(), head.train()
    for idx in rng.permutation(len(jets))[: len(jets) // 256 * 256].reshape(-1, 256):
        z1, z2 = head(encoder.embed(ds1.batch(idx))), head(encoder.embed(ds2.batch(idx)))
        loss = ...  # blank 2: the NT-Xent loss at temperature 0.1
        ...  # blank 3: the optimiser step (three lines)
itj.models.save_model(encoder, "models/toy/jetclr_pfn.pt")

# the four monitoring calls (the last three in evaluation mode, under torch.no_grad(),
# on fixed views of a few thousand jets)
loss.item()
itj.ssl.alignment(z1, z2)
itj.ssl.uniformity(z1)
itj.ssl.linear_eval(encoder, train, val, n_labels=1000)
```

**W3.4.4** *Core, 45 min.* What did it learn? Freeze the encoder and train a logistic regression on its embedding with
1 000 labels from the train part of `data/t2_detector.parquet`. Test it on the test part
(`itj.ssl.linear_eval(encoder, train, test, n_labels=1000)`, about a minute). Do the same for the random encoder
(`model.reset_parameters(seed=0)`) and for the course PFN, with the same logistic regression. Where does the pretraining
help, and where not?

**W3.4.5** *Stretch, 75 min.* Your own loop, longer and other encoders. Write the pretraining loop without the skeleton.
Then run it for 10 epochs (about 7 minutes, in the background): does the linear evaluation keep improving? Then
pretrain `partsmall_nopair` for 4 epochs on 15 000 jets, also in the background (on 30 000 jets it takes about 21
minutes). Four epochs of 15 000 jets are only about 230 steps of 256 jets, so this encoder is under-trained by design:
judge it by how its linear evaluation moves over the epochs, not by its final value.

**W3.4.6** *Stretch, 2 h.* Against the observables, and truth probes. Run the linear evaluation with all labels. Add the
baselines of the standardised observables $X_\mathrm{phys} = [m_\mathrm{SD}, \tau_{21}, \tau_{32}, D_2, N, \ln p_T]$,
with the same logistic regression and with a BDT as in **W2.2.2**. Then run the truth probes
of **W3.1.2** and **W3.1.4** on the SSL embedding. Does it know quark from gluon jets, the number of emissions, the
opening angle of the W (`truth_dr_min` of the W jets as the target)?

**W3.4.7** *Stretch, 3.5 h.* Few labels. This takes about 13 minutes of CPU: run it in the background. Plot the AUC (W
against QCD and top against QCD) against the number of labelled jets (100, 300, 1 000, 3 000, 10 000; 3 seeds). Do it
for three methods: SSL + linear evaluation, SSL + fine-tuning (training the whole pretrained encoder on the labels) and
supervised training from scratch. Train every point with the rule of **W2.5.3** (`min_steps=2000`, `val_every=50`,
`patience=5`), but on a validation set of as many labelled jets as the training set. Validation labels are labels too,
so count them in the budget (the linear evaluation needs none). The alternative is to train for a fixed number of steps
without early stopping. At 100 and 300 labels an epoch is one or two steps, and a validation costs almost nothing. Use
`val_every=10`, and if the best state is still the first validation, validate more often. Add the curve of **W2.5.3**
for reference if you made it. It used 5000 validation labels at every size, so its gap to your scratch curve is one
measure of what the validation labels are worth. Where does pretraining help, if anywhere?

**Deliverables:** your two expressions (**W3.4.1**); the `allclose` line, the two numerical limits and the plot against
the temperature (**W3.4.2**); `W3.4.3_ssl_training.pdf` and the encoder (**W3.4.3**); the rows and a short comparison
(**W3.4.4**); `W3.4.6_probes.pdf` (**W3.4.6**); `W3.4.7_few_labels.pdf` (**W3.4.7**).

**Hints**
- **W3.4.1:** count the negatives of one positive pair first. Write the loss of one positive pair as a softmax over
  $2B - 1$ similarities before you sum over pairs.
- **W3.4.2:** normalise the projections first. The positive pair must not be counted among its own negatives: set the
  diagonal of the $2B \times 2B$ similarity matrix to $-\infty$ before the softmax, not to 0. Stack the two views as
  `torch.cat([z1, z2])`, so that the partner of row $i$ is row $i \pm B$.
- **W3.4.3:** the loss falls fast and then flattens well above its floor. That is expected with these augmentations:
  judge the run by the alignment, the uniformity and the linear evaluation. Evaluate the encoder output (the embedding
  before the projection head), not the projection. `JetCLRAugment` drops the truth columns first, for speed. If the
  linear evaluation of your encoder is far below the reference in [methods.md](../docs/methods.md#linear-evaluation),
  check the augmentations first.
- **W3.4.4:** [methods.md](../docs/methods.md#linear-evaluation); the random encoder is the control, as in **W3.1.2**.
  Call `linear_eval` three times with the same labels and seed.
- **W3.4.6:** a BDT with 1 000 labels must not early-stop on 15 000 validation labels.

### Weekly summary

"What our taggers know": one paragraph with numbers, answering the thread question *What do our taggers know about a
jet, and what do they ignore?* (0.5 h with the hand-in). Hand in `answers/week3.md` with the summary and the answers to
every warm-up and core ID. Also hand in the figures `W3.1.1_pca.pdf`, `W3.1.2_probes.pdf` with the probe table,
`W3.2.1_noue.pdf`, `W3.2.2_filters.pdf`, `W3.3.1_soft.pdf`, `W3.3.2_irc.pdf`, `W3.3.3_augmentations.pdf` and
`W3.4.3_ssl_training.pdf`, the checks of your NT-Xent and the rows of **W3.4.4**. In the discussion (0.5 h) we go
through the core IDs.

---

- **Previous:** [Week 2: Taggers, from cuts to transformers](week2.md)
- **Next:** [Week 4: Real jets, and where does a tagger look?](week4.md)
- **All weeks, the rules and the notation:** [TASKS.md](../TASKS.md#the-weeks)
