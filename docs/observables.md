# Jet observables and IRC safety

This page defines the jet observables of `itj.observables` and `itj.cluster` exactly, says which conventions reproduce
the stored JetClass branches, and sorts the observables by infrared and collinear (IRC) safety, with the numerical
tests that check it on observables (task T1.6) and on trained networks (T5.5). The physics behind the scaling of the
observables is in [physics.md](physics.md#leading-log-results).

All functions take padded arrays: `p4` of shape `(B, N, 4)` with $(p_x, p_y, p_z, E)$ in float64 and a boolean `mask`
of shape `(B, N)`, as returned by `data.padded_p4()`. Work on chunks of jets:

```python
from itj import observables
from itj.io import load

data = load("data/t1_detector.parquet")
for chunk in data.chunks(2000):  # never pad a whole data set to its longest jet
    p4, mask = chunk.padded_p4()
    m = observables.mass(p4, mask)  # [B] GeV
    tau = observables.nsubjettiness(p4, mask, n=(1, 2, 3))  # [B, 3]
values = observables.compute(data, ["mass", "sdmass", "tau21", "d2", "multiplicity"])  # dict of [N] arrays
```

## Notation

A jet has the constituents $i = 1, \ldots, N$ with the four-momenta $p_i$, transverse momenta $p_{T,i}$, rapidities
$y_i$ and azimuths $\phi_i$. With $p_{T,J} = \sum_i p_{T,i}$ the momentum fractions are $z_i = p_{T,i}/p_{T,J}$, and

```math
\Delta R_{ij} = \sqrt{(y_i - y_j)^2 + \Delta\phi_{ij}^2}, \qquad \Delta\phi_{ij} \in (-\pi, \pi] .
```

The observables use the rapidity $y$ (as fastjet); the stored `part_deta` uses the pseudorapidity $\eta$ with the
JetClass sign ([physics.md](physics.md#units-and-coordinates)).

## Mass and multiplicity

```math
m_J^2 = \Bigl(\sum_i p_i\Bigr)^2, \qquad \rho = \frac{m_J^2}{p_{T,J}^2 R^2}, \qquad N = \text{number of constituents}.
```

`observables.mass`, `observables.multiplicity`; `jet_p4` gives the summed four-vector and `pt_y_phi_m` the kinematics
of any four-vector array.

## Soft drop

Soft drop grooms the jet by walking down its Cambridge/Aachen (C/A) tree (1402.2657):

1. recluster the constituents with C/A ($p = 0$ below), E-scheme, $R = 0.8$;
2. undo the last merging, $j \to j_1 j_2$;
3. if $\min(p_{T,1}, p_{T,2})/(p_{T,1} + p_{T,2}) > z_\mathrm{cut}\,(\Delta R_{12}/R_0)^\beta$, stop: the groomed jet is
   $j$;
4. otherwise keep the harder branch and go back to 2;
5. if a single particle is reached, the groomed jet is that particle (mass 0 for a massless particle).

`observables.softdrop(p4, mask, zcut=0.1, beta=0.0, R0=0.8)` returns the groomed mass $m_\mathrm{SD}$, $p_T$,
$z_g$ (the momentum fraction of the softer branch at the stop), $R_g$ ($\Delta R_{12}$ at the stop) and the number
of dropped branches. With $\beta = 0$ it is the modified mass-drop tagger (1307.0007), and it reproduces `jet_sdmass`.

## N-subjettiness

N-subjettiness (1011.2268) measures how well the jet is described by $N$ axes $\hat n_1, \ldots, \hat n_N$:

```math
\tau_N^{(\beta)} = \frac{1}{d_0}\sum_i p_{T,i} \min_a \Delta R_{i, a}^\beta, \qquad d_0 = \sum_i p_{T,i} R_0^\beta,
\qquad \tau_{21} = \frac{\tau_2}{\tau_1}, \quad \tau_{32} = \frac{\tau_3}{\tau_2},
```

with $\beta = 1$ and $R_0 = 0.8$. $\tau_N$ is small when the radiation sits close to $N$ axes; a two-prong W jet has a
small $\tau_{21}$, a three-prong top jet a small $\tau_{32}$. The result depends on the axes
(`observables.subjet_axes`):

| `axes` | definition |
|---|---|
| `wta_kt` (default) | exclusive kT clustering to $N$ pseudojets with winner-take-all recombination (the merged pseudojet takes the direction of the harder one and the sum of the $p_T$, 1401.2158) |
| `kt` | exclusive kT clustering with E-scheme recombination (four-vectors add) |
| `onepass_kt` | the kT axes moved to the nearest local minimum of $\tau_N$ by iterated Weiszfeld updates for $\beta = 1$ (up to 1000 steps, accuracy $10^{-4}$, the defaults of fastjet-contrib), the analogue of fastjet's `OnePass_KT_Axes` |

One exclusive-kT pass gives the axes for every $N$ (the pseudojets alive after $N_\mathrm{particles} - N$ mergings).
$\tau_N = 0$ when the jet has at most $N$ particles; guard the ratios.

## Energy correlation functions

The energy correlation functions (1305.0007) need no axes:

```math
e_2^{(\beta)} = \sum_{i<j} z_i z_j \Delta R_{ij}^\beta, \qquad
e_3^{(\beta)} = \sum_{i<j<k} z_i z_j z_k \left(\Delta R_{ij}\Delta R_{ik}\Delta R_{jk}\right)^\beta,
\qquad D_2^{(\beta)} = \frac{e_3^{(\beta)}}{\bigl(e_2^{(\beta)}\bigr)^3}, \qquad C_2^{(\beta)} = \frac{e_3^{(\beta)}}{\bigl(e_2^{(\beta)}\bigr)^2}.
```

$D_2$ is from 1409.6298; $\beta = 2$ is mass-like ($e_2^{(2)} \approx m^2/p_T^2$), $\beta = 1$ the default of
`observables.d2`. A two-prong jet has $D_2 = O(1)$, a one-prong jet $D_2 \gg 1$
([physics.md](physics.md#power-counting)).

**The matrix identity.** With $A_{ij} = \Delta R_{ij}^\beta$ ($A_{ii} = 0$) and $Z = \mathrm{diag}(z)$,

```math
e_2 = \tfrac{1}{2}\, z^\mathsf{T} A z, \qquad e_3 = \tfrac{1}{6}\, \mathrm{tr}\bigl((ZA)^3\bigr),
```

since $\mathrm{tr}((ZA)^3) = \sum_{ijk} z_i A_{ij} z_j A_{jk} z_k A_{ki}$ vanishes whenever two indices coincide
($A_{ii} = 0$) and counts every triple $i < j < k$ six times. In NumPy this is two batched matrix products:
`ZA = z[:, :, None] * A`, then `np.trace(ZA @ ZA @ ZA, axis1=1, axis2=2) / 6`. A three-operand `np.einsum` over
`(B, N, N)` arrays computes the same thing about seven times more slowly, and a whole data set padded to $N = 128$
takes gigabytes: always loop over chunks of about 2000 jets.

## The Lund plane

The primary Lund plane (1807.04758) records the history of the hardest branch of the C/A tree. At each declustering
$j \to a, b$ with $p_{T,a} > p_{T,b}$:

```math
\Delta = \Delta R_{ab}, \qquad k_t = p_{T,b}\,\Delta, \qquad z = \frac{p_{T,b}}{p_{T,a} + p_{T,b}}, \qquad
\psi = \mathrm{atan2}(\Delta y, \Delta\phi),
```

then follow $a$. `observables.lund_primary(p4, mask)` returns $\ln(1/\Delta)$, $\ln(k_t/\mathrm{GeV})$, $z$ and $\psi$
per jet (Ragged), `itj.cluster.decluster_primary` the same from any C/A tree, and `observables.lund_density` the density
per jet per unit area,

```math
\rho(\Delta, k_t) = \frac{1}{N_\mathrm{jet}}\, \frac{dn}{d\ln(1/\Delta)\, d\ln k_t},
```

with its statistical error $\sqrt{n}/(N_\mathrm{jet}\,\text{bin area})$. At leading log the density is
$2\alpha_s C_R/\pi$ ([physics.md](physics.md#lund-density)). The toy stores the same coordinates of the true primary
emissions as `truth_lund_*`.

**Link to ParT.** The pair features of the Particle Transformer are the Lund coordinates of every pair of particles,
in the order `[ln kT, ln z, ln Delta, ln m^2]`:

```math
\ln k_T = \ln\bigl(\min(p_{T,i}, p_{T,j})\,\Delta_{ij}\bigr), \quad
\ln z = \ln\frac{\min(p_{T,i}, p_{T,j})}{p_{T,i} + p_{T,j}}, \quad \ln\Delta_{ij}, \quad \ln m_{ij}^2 = \ln (p_i + p_j)^2,
```

with $\Delta_{ij}$ in rapidity and azimuth. Task T5.4 asks which of them the attention bias $U$ uses
([models.md](models.md#particle-transformer)).

## Angularities and pT_D

The generalised angularities (1408.3122) are

```math
\lambda_\beta^\kappa = \sum_i z_i^\kappa \left(\frac{\Delta R_i}{R}\right)^\beta,
```

with $\Delta R_i$ the distance to the jet axis. Special cases: the width $\lambda_1^1$ (`observables.width`), the
squared $p_T^D$, $(p_T^D)^2 = \lambda_0^2 = \sum_i z_i^2$ (`observables.ptd` returns $p_T^D$), and the multiplicity
$\lambda_0^0 = N$. `observables.leading_fraction` is $\max_i z_i$.

## JetClass conventions

These choices reproduce the stored JetClass branches from `part_*` (the toy uses them for its own branches, through
`itj.observables.jet_branches`):

| branch | convention | agreement on the JetClass sample |
|---|---|---|
| `jet_pt`, `jet_eta`, `jet_phi` | of the four-vector sum of the constituents | exact for `jet_pt` |
| `jet_energy` | sum of the constituent energies (toy) | JetClass differs by up to 3.8 % |
| `jet_sdmass` | soft drop, C/A in $(y, \phi)$, E-scheme, $\beta = 0$, $z_\mathrm{cut} = 0.1$, $R_0 = 0.8$ | within 1 % for 85-92 % of the jets |
| `jet_tau1..4` | $\beta = 1$, $R_0 = 0.8$, $d_0 = \sum p_T R_0$; axes: see below | depends on the axes (task T4.2) |

The CMS card of Delphes (master) computes N-subjettiness with `AxisMode 4`, which is fastjet's `OnePass_KT_Axes`,
$\beta = 1$ and `NormalizedMeasure(1, 0.8)`, and soft drop with $\beta = 0$, $z_\mathrm{cut} = 0.1$, $R_0 = 0.8$;
whether JetClass used exactly this card is not documented. `itj.observables` offers the three axis choices of the
table in [N-subjettiness](#n-subjettiness); which of them reproduces the stored `jet_tau*` is task T4.2. The toy
writes its own `jet_tau1..4` with WTA-kT axes, the fastest of the three (`onepass_kt` costs about 2.6 times more in
`jet_branches`): toy and JetClass $\tau_N$ therefore differ by an axis offset of about 0.5-1.5 % on top of the
physics, which matters only when you compare the two to the per cent.

## Clustering

`itj.cluster` is a pure-NumPy implementation of the generalised kT algorithms (0802.1189, hep-ph/9707323,
hep-ph/9907280; the FastJet manual 1111.6097 is the reference), batched over jets, with no fastjet dependency:

```math
d_{ij} = \min\bigl(p_{T,i}^{2p}, p_{T,j}^{2p}\bigr)\,\frac{\Delta R_{ij}^2}{R^2}, \qquad d_{iB} = p_{T,i}^{2p},
\qquad p = \begin{cases} -1 & \text{anti-kT (the jet definition)} \\ 0 & \text{Cambridge/Aachen (soft drop, Lund plane)} \\ 1 & \text{kT (N-subjettiness axes)} \end{cases}.
```

- **Inclusive mode** (`inclusive=True`): repeatedly take the smallest of all $d_{ij}$ and $d_{iB}$; merge $i$ and $j$
  for a $d_{ij}$, call $i$ a jet for a $d_{iB}$. `leading_jet` returns the members of the hardest anti-kT jet.
- **Exclusive mode** (the default of `cluster`): no beam distance; merge until one pseudojet is left. The tree then
  holds every intermediate state: `exclusive_nodes(n)` gives the $n$ pseudojets alive after
  $N_\mathrm{particles} - n$ mergings and `exclusive_labels(n)` the subjet of every particle (the reco-level proxy of
  `truth_part_prong` in task T5.3).
- **Recombination.** E-scheme: the four-vectors add. WTA (winner-take-all): the $p_T$ add and the merged pseudojet
  takes the direction of the harder one, so the axis follows the hardest radiation and does not recoil against soft
  emissions.
- **Algorithm.** For all jets of a bucket at once: keep the $(B, N, N)$ matrix of distances, take the argmin over the
  flattened matrix, merge, set the row and column of the merged slot to $+\infty$ and recompute the row and column of
  the new one. Jets are sorted into buckets of similar multiplicity (results in the input order). The cost grows about
  as $N^{2.8}$: about 0.14 ms per jet and pass at $N \approx 30$ on one 2016 core (i7-6850K). The tests compare the
  merge sequence with fastjet where it is installed.

## IRC safety

An observable $O(p_1, \ldots, p_n)$ is

- **infrared safe** if adding a soft particle does not change it in the limit: $O(p_1, \ldots, p_n, \epsilon k) \to
  O(p_1, \ldots, p_n)$ as $\epsilon \to 0$, for any direction of $k$;
- **collinear safe** if splitting a particle into two collinear ones does not change it:
  $O(\ldots, p_i, \ldots) = O(\ldots, \lambda p_i, (1 - \lambda)p_i, \ldots)$ for every $\lambda \in [0, 1]$, with a
  continuous limit for nearly collinear splittings.

IRC-safe observables have finite perturbative predictions order by order and, in practice, are insensitive to the
shower cutoff and to hadronisation at leading power. Some observables are not IRC safe but have a finite distribution
once the all-order (Sudakov) suppression of the region where they are ill defined is included: they are **Sudakov
safe** (1307.1699, 1502.01719). The ratios $\tau_{21}$ and $D_2$ are the standard examples: $\tau_{21} = \tau_2/\tau_1$
is $0/0$ when $\tau_1 \to 0$, but jets with $\tau_1 \to 0$ are exponentially rare.

| status | observables |
|---|---|
| IRC safe | $m_J$; $m_\mathrm{SD}$; $\tau_N$ with $\beta > 0$ and IRC-safe axes; $e_2$, $e_3$ ($\beta > 0$); energy flow polynomials (1712.07124); jet $p_T$ and axis; the Lund density in finite bins away from the cutoff |
| Sudakov safe | $\tau_{21}$, $\tau_{32}$, $D_2$, $C_2$; $z_g$ of soft drop with $\beta = 0$ |
| unsafe | the multiplicity $N$ (IR and C); $p_T^D$ and every angularity with $\kappa \ne 1$ (C); the leading fraction (C); counts of particle types and charges (C in principle) |

**A single jet cannot tell safe from Sudakov safe.** A scan that adds a soft particle to one jet with many particles
tests the continuity of the observable at that jet, where $\tau_1 > 0$: there $\tau_{21}$ changes smoothly, like an
IRC-safe observable. Sudakov safety is a statement about the distribution as $\epsilon \to 0$ over the region
$\tau_1 \to 0$, which per-jet scans do not reach. A two-particle jet with $\tau_1 \to 0$ shows the difference.

**Networks.** An Energy Flow Network (1810.05165),

```math
F\Bigl(\sum_i z_i\, \Phi(y_i, \phi_i)\Bigr),
```

with a continuous $\Phi$ is IRC safe by construction: a soft particle enters with the weight $z_i \to 0$, and a
collinear split $\lambda z_i\Phi(y, \phi) + (1 - \lambda) z_i\Phi(y, \phi)$ gives the same sum. A Particle Flow Network
$F(\sum_i \Phi(z_i, y_i, \phi_i, \ldots))$ is not: every particle adds $\Phi$ at least once, whatever its energy.
Neither are ParticleNet and ParT, whose pooling, softmax attention and logarithmic inputs give a soft particle an $O(1)$
weight.

## Numerical IRC tests

`itj.augment` perturbs jets and recomputes every derived quantity (sorting, `part_deta`, `part_dphi`, the jet
kinematics; with `substructure=True` also `jet_sdmass` and `jet_tau*`), so that the observables and the network inputs
can simply be computed again:

- **soft** (`add_soft`): add $n \in \{1, 5\}$ particles with $p_T = \epsilon\, p_{T,J}$, $\epsilon = 10^{-1}, \ldots,
  10^{-6}$, uniformly in the disc $\Delta R < R$, as photons or charged hadrons;
- **collinear** (`split_collinear`): split a particle (the hardest, a random one, all, or chosen ones) into
  $\lambda p$ and $(1 - \lambda)p$, $\lambda \in \{0.5, 0.1\}$, separated by $\delta = 10^{-1}, \ldots, 10^{-5}$ in
  $\Delta R$ at a random azimuth.

Plot the mean $|\Delta O|/O$ (or, for a network, the mean $|\Delta s|$ of the score) against $\epsilon$ and $\delta$
on log axes: an IRC-safe quantity falls linearly to zero, an unsafe one stays at a plateau.
`itj.interpret.irc_scan` runs the same scans on a tagger. Pitfalls:

- **Truncation.** The networks see at most 128 particles, sorted by $p_T$: added soft particles sort last and are
  silently dropped from a full jet, which makes any network look IR safe. `max_particles_guard` (used by `irc_scan`)
  refuses jets that would exceed the length.
- **ParT's pair features.** ParT clamps $k_T$, $\Delta$ and $m^2$ at $10^{-8}$ before the logarithm ($\ln = -18.4$),
  then normalises them with a BatchNorm: an exactly collinear split ($\delta = 0$) hits the clamp. Use finite
  $\delta$ and say where the clamp sets in.
- **Particle types.** A split particle copies its charge, flags and impact parameters to both pieces; an added soft
  particle is a photon or a charged hadron (choose and say which).
- **Standardisation.** Use the same input transformation for every variant (the `FEATURE_SETS` of `itj.features`).
- **Off-support inputs.** Real jets have no photons below the calorimeter thresholds: the networks never saw such soft
  particles, so an IR test probes behaviour outside the training data. This is a result in itself.
- **The sign of `part_deta`.** A soft particle can move $\eta_\mathrm{jet}$ across 0 and flip the sign of every
  `part_deta`; `keep_sign=True` keeps the sign of the unperturbed jet (`irc_scan` does this).
- **Zero denominators.** $|\Delta O|/O$ is infinite for a jet with $O = 0$ (about 0.2 % of the detector-level jets
  have $m_\mathrm{SD} = 0$): leave such jets out, or plot the median of $|\Delta O|/O$ as well.
- **Soft drop below $z_\mathrm{cut}$.** For $\lambda \le z_\mathrm{cut} = 0.1$ the two pieces of a particle fail the
  soft-drop condition against each other. A jet that soft drop grooms down to one massive hadron then loses the softer
  piece, and its $m_\mathrm{SD}$ falls by about the fraction $\lambda$ at any $\delta$: a few jets in a thousand give
  a plateau of the mean. The hadron masses cause it; for massless particles soft drop is collinear safe. Scan
  $m_\mathrm{SD}$ with $\lambda > z_\mathrm{cut}$ (T1.6 uses 0.5 and 0.2).
- **Float32 pieces.** The files store float32 four-momenta. Rounded one by one, the pieces $\lambda p$ and
  $(1 - \lambda)p$ of a TeV particle each get a spurious mass of order 0.1 GeV for $\lambda \ne 0.5$, enough to change
  the groomed mass of a jet whose soft-drop core is one or two particles. `split_collinear` therefore returns the new
  four-momenta in float64 (`float64=False` for the float32 behaviour; `irc_scan` and `JetCLRAugment` use float32,
  since the networks read float32 anyway).

## Chunking and memory

Functions that build $(B, N, N)$ arrays (`ecf`, `d2`, `c2`, `softdrop`, `nsubjettiness`, `lund_primary`) process at
most `chunk_size` = 2048 jets at a time (about 0.6 GB at $N = 123$), and `compute` and `jet_branches` loop over
`data.chunks()`. Do the same in your own code: on an 8 GB laptop a whole data set padded to its longest jet does not
fit.
