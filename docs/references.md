# Literature values, the toy defaults and the reading list

Part 1 lists the physical constants and detector parameters behind the defaults of the toy, the course's own jet
simulator ([primer.md](primer.md#the-toy-and-its-files)). Every number is taken from the cited source or derived from it
with the stated formula ("derived"). The toy is a cartoon, not a tune (a set of parameters fitted to data). The "toy
default" column lists the values of the `detector` preset, and the "comment" column says how they relate to the
literature. Part 2 is the reading list of the course, grouped by week, with the papers to start with in each week.

## 1. Hard process

| quantity | literature value(s) | source | toy default | comment |
|---|---|---|---|---|
| W mass, width | $m_W$ = 80.369 ± 0.013 GeV, $\Gamma_W$ = 2.085 ± 0.042 GeV | [1] | `w_mass` 80.4, `w_width` 2.085 GeV | Breit-Wigner truncated to ± 15 GeV (`w_window`) |
| top mass, width | $m_t$ = 172.57 ± 0.29 GeV (direct measurements), $\Gamma_t$ = 1.42 (+0.19, -0.15) GeV | [1] | `top_mass` 172.5, `top_width` 1.42 GeV | truncated to ± 25 GeV (`top_window`) |
| b mass | $m_b$ about 4.8 GeV (pole mass, scheme dependent) | [1] | `b_mass` 4.8 GeV in the decay kinematics | massless in the shower |
| W helicity in top decays | $F_0$ = 0.693 ± 0.014, $F_L$ = 0.315 ± 0.011 (ATLAS and CMS at 8 TeV); at leading order $F_0 = m_t^2/(m_t^2 + 2m_W^2)$ = 0.697 (derived) | [2] | leading-order $F_0$ per event (`w_helicity`) | no distinction of up- and down-type quarks |
| seed $p_T$ spectrum | JetClass: median jet $p_T$ about 590 GeV after 500 < $p_T$ < 1000 GeV (measured on the val files) | [3] | $p_T^{-4.5}$ on [450, 1100] GeV | chosen to match the median |
| QCD initiators | Z + jet at these $p_T$ is mostly quark-initiated; flavours follow the parton densities | [4] | `qcd_quark_fraction` 0.75; u, d, s, c, b = 0.50, 0.30, 0.08, 0.08, 0.04 | an assumption, not a measurement |

## 2. Parton shower

| quantity | literature value(s) | source | toy default | comment |
|---|---|---|---|---|
| strong coupling | $\alpha_s(M_Z)$ = 0.1180 ± 0.0009 | [1] | `alphas_value` 0.118 | one-loop running with $n_f = 5$ |
| $\Lambda_\mathrm{QCD}$ | about 0.2 GeV ($\overline{\mathrm{MS}}$, five flavours, at higher orders) | [1] | 88 MeV at one loop from $\alpha_s(M_Z)$ = 0.118 (derived); $\alpha_s$(1 GeV) = 0.34 (derived) | a one-loop $\Lambda$ is not the $\overline{\mathrm{MS}}$ one |
| colour factors | $C_F = 4/3$, $C_A = 3$, $T_R = 1/2$ (SU(3)) | [5] | the same | |
| splitting functions | leading-order $P_{qq}$, $P_{gg}$, $P_{qg}$ | [5], [6] | the same, without the virtual terms | [physics.md](physics.md#parton-shower) |
| shower cutoff | PYTHIA 8: $p_{T,\mathrm{min}}$ about 0.5 GeV | [7] | `kt_cut` 1.0 GeV | the shower and hadronisation split differently in the toy |
| ordering variable and recoil | Herwig: angular ordering, reshuffle with the progenitor momenta | [8], [9] | angular ordering, the same reshuffle | QCD jets have no recoil partner |
| veto algorithm | | [10] | as in [physics.md](physics.md#the-veto-algorithm) | |

## 3. Hadronisation

| quantity | literature value(s) | source | toy default | comment |
|---|---|---|---|---|
| Field-Feynman fragmentation | $f(z) = (1 - a) + 3a(1 - z)^2$ with $a$ = 0.77 | [11] | `ff_a` 0.77 | |
| string $k_T$ width | PYTHIA `StringPT:sigma` = 0.335 GeV (Monash tune), given to each quark of a break | [12], [7] | `kt_sigma` 0.24 GeV per component of a hadron | 0.335/$\sqrt{2}$: a hadron is made of two quarks of neighbouring breaks |
| particle composition, JetClass QCD jets (by count) | charged hadrons 0.484, neutral hadrons 0.107, photons 0.403, electrons 0.004, muons 0.001 (measured) | [3] | `pi0_fraction` 0.34, `neutral_fraction` 0.13, `kaon_fraction` 0.15, `klong_fraction` 0.60 | calibrated at the detector level; photons count about twice the $\pi^0$ share |
| particle composition, JetClass top jets (by count) | 0.465, 0.109, 0.414, 0.008, 0.005 (measured) | [3] | | B decays add leptons |
| b fragmentation | $\langle x_B \rangle$ about 0.7 at the Z pole | [1] | `b_frag_mean` 0.7, $f(z) \propto z^\alpha(1 - z)$ | |
| B-hadron lifetime | $c\tau$ = 455.7 µm ($B^0$), 491.1 µm ($B^+$) | [1] | `b_ctau` 0.46 mm, $m_B$ = 5.28 GeV | one species for all B hadrons |
| semileptonic B decays | $\mathcal{B}(B \to X\ell\nu)$ about 0.10-0.11 per lepton flavour; cascades $b \to c \to \ell$ about 0.08-0.10 | [1] | `semileptonic` 0.11, `semileptonic_cascade` 0.10 | no c-hadron decays elsewhere, so fewer leptons than JetClass |
| B decay multiplicity | about 5 charged particles per decay on average | [1] | `b_decay_multiplicity` 7, `b_charged_share` 0.7 (4.9 charged on average, derived) | massless products in flat phase space |

## 4. Underlying event and detector

| quantity | literature value(s) | source | toy default | comment |
|---|---|---|---|---|
| underlying event at 13 TeV | about one charged particle with $p_T > 0.5$ GeV per unit $\eta\phi$ in the transverse region | [13] | `ue_density` 0.7 per unit $(y, \phi)$, $p_T$ = 0.3 GeV + Exp(0.7 GeV) | an order of magnitude only: `ue_density` is a calibrated knob, not a literature value |
| pileup | JetClass has none | [3] | none | |
| track efficiency (Delphes CMS card) | charged hadrons: 0.70 for 0.1 < $p_T \le$ 1 GeV and 0.95 above ($\lvert\eta\rvert \le 1.5$); 0.60 and 0.85 for $1.5 < \lvert\eta\rvert \le 2.5$; 0 beyond | [14], [15] | `track_eff_low` 0.70, `track_eff_high` 0.95, `track_eff_fwd_low` 0.60, `track_eff_fwd_high` 0.85, `track_pt_min` 0.1 GeV | as the card |
| track momentum resolution | card: $\sqrt{0.06^2 + (1.3 \times 10^{-3}\,p_T)^2}$ at $\lvert\eta\rvert < 0.5$ (looser elsewhere) | [15] | $\sigma/p_T = 0.01 \oplus 2 \times 10^{-4}\,p_T$ | a CMS-like cartoon; irrelevant at the course's precision |
| ECAL | cells 0.0174 in $(\eta, \phi)$; $\sigma/E$ with $(1 + 0.64\eta^2)\sqrt{(0.008E)^2 + 0.11^2E + 0.40^2}$ ($\lvert\eta\rvert \le 1.5$), $(2.16 + 5.6(\lvert\eta\rvert - 2)^2)\sqrt{\ldots}$ (1.5-2.5); `EnergyMin` 0.5 GeV, `EnergySignificanceMin` 2.0; log-normal smearing | [15] | the same | |
| HCAL | cells 0.087; $\sqrt{(0.050E)^2 + 1.50^2E}$; `EnergyMin` 1.0 GeV, `EnergySignificanceMin` 1.0 | [15] | the same | |
| forward calorimeter | $\sqrt{(0.107E)^2 + 2.08^2E}$ beyond $\lvert\eta\rvert$ = 2.5 | [15] | the same, with the HCAL cell size | |
| particle flow | tracks for charged particles, the calorimeters for the rest | [14], [15] | perfect for tracked particles | no calorimeter deposit of tracked particles |
| jet substructure in the card | `FatJetFinder`: anti-kT $R$ = 0.8; N-subjettiness with `AxisMode 4` (fastjet `OnePass_KT_Axes`), $\beta$ = 1, `NormalizedMeasure(1, 0.8)`; soft drop $\beta$ = 0, $z_\mathrm{cut}$ = 0.1, $R_0$ = 0.8 | [15], [16] | the same, with WTA-kT axes (`tau_axes`) | whether JetClass used exactly this card is not documented; WTA-kT axes reproduce the stored $\tau_N$ to about 1 % |

## 5. JetClass measured values

Measured with `itj` on 20 000 jets of each validation file of JetClass [3] (QCD = `ZJetsToNuNu_120`, W =
`WToQQ_120`, top = `TTBar_120`). The toy is calibrated to these values at the `detector` level
([physics.md](physics.md#calibration-and-validation)). The statistics of the whole files (100 000 jets, all ten
classes) are in [jetclass.md](jetclass.md#the-data-set); they differ in the last digit.

| quantity | QCD | W | top | toy default |
|---|---|---|---|---|
| mean multiplicity | 35.4 | 32.2 | 50.8 | 35.8 / 38.0 / 53.2, calibrated (`ue_density`, `pi0_fraction`, `neutral_fraction`) |
| median $m_\mathrm{SD}$ [GeV] | 22.9 | 77.3 | 164.6 | 20 / 81.6 / 169.8, from the hard process and the shower |
| fraction in the mass window ([65, 95] GeV for W, [140, 200] GeV for top) | | 0.68 | 0.69 | 0.78 / 0.82 (sharper peaks) |
| median $\tau_{21}$ | 0.608 | 0.367 | | 0.610 / 0.364 |
| median $\tau_{32}$ | 0.763 | | 0.501 | 0.765 / 0.471 |
| smallest $p_T$ per particle type | tracks 0.08-0.13 GeV, neutral hadrons 0.31 GeV, photons 0.45 GeV | | | the thresholds of the detector |
| particles at $\Delta R > 0.8$ from the jet axis | 0.5-0.7 % | | | the toy has none |

## References of part 1

1. Particle Data Group (S. Navas et al.), *Review of particle physics*, Phys. Rev. D 110, 030001 (2024) (2025 update),
   the summary tables and the reviews "Quantum chromodynamics", "Fragmentation functions in e+e-, ep and pp
   collisions" and "Semileptonic b-hadron decays".
2. ATLAS and CMS Collaborations, *Combination of the W boson polarization measurements in top quark decays using ATLAS
   and CMS data at √s = 8 TeV*, JHEP 08 (2020) 051, arXiv:2005.03799.
3. H. Qu, C. Li and S. Qian, *JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics*, Zenodo (2022),
   doi:10.5281/zenodo.6619768; the numbers above were measured on its validation files ([jetclass.md](jetclass.md)).
4. P. Gras et al., *Systematics of quark/gluon tagging*, JHEP 07 (2017) 091, arXiv:1704.03878.
5. R. K. Ellis, W. J. Stirling and B. R. Webber, *QCD and Collider Physics*, Cambridge University Press (1996).
6. S. Höche, *Introduction to parton-shower event generators*, TASI 2014 lectures, arXiv:1411.4085.
7. C. Bierlich et al., *A comprehensive guide to the physics and usage of PYTHIA 8.3*, SciPost Phys. Codebases 8
   (2022), arXiv:2203.11601.
8. S. Gieseke, P. Stephens and B. Webber, *New formalism for QCD parton showers*, JHEP 12 (2003) 045,
   arXiv:hep-ph/0310083.
9. M. Bähr et al., *Herwig++ Physics and Manual*, Eur. Phys. J. C 58, 639 (2008), arXiv:0803.0883.
10. L. Lönnblad, *Fooling Around with the Sudakov Veto Algorithm*, Eur. Phys. J. C 73, 2588 (2013), arXiv:1211.7204.
11. R. D. Field and R. P. Feynman, *A parametrization of the properties of quark jets*, Nucl. Phys. B 136, 1 (1978).
12. P. Skands, S. Carrazza and J. Rojo, *Tuning PYTHIA 8.1: the Monash 2013 Tune*, Eur. Phys. J. C 74, 3024 (2014),
    arXiv:1404.5630.
13. ATLAS Collaboration, *Measurement of charged-particle distributions sensitive to the underlying event in √s = 13
    TeV proton-proton collisions with the ATLAS detector at the LHC*, JHEP 03 (2017) 157, arXiv:1701.05390.
14. J. de Favereau et al., *DELPHES 3, A modular framework for fast simulation of a generic collider experiment*, JHEP
    02 (2014) 057, arXiv:1307.6346.
15. Delphes, `cards/delphes_card_CMS.tcl` and `modules/FastJetFinder.cc` (master branch),
    [github.com/delphes/delphes](https://github.com/delphes/delphes).
16. J. Thaler and K. Van Tilburg, *Maximizing Boosted Top Identification by Minimizing N-subjettiness*, JHEP 02 (2012)
    093, arXiv:1108.2701.

## Reading list

Every entry gives the authors, the title, the journal (or "preprint") and the arXiv number
(`https://arxiv.org/abs/<number>`). The arXiv numbers were checked against the arXiv API on 2026-11-05. The two entries
added in 0.2.0, Salam and Guest, Cranmer and Whiteson, were added from their journal versions. The list is grouped by
the week of the course in which a paper is first needed. The "used in" column names the questions (IDs of the week
files that [TASKS.md](../TASKS.md) links), tasks or weeks and the documentation pages where the paper is needed.

**Start here.** Each week opens with one or two readings to start with, listed first in its table:

- week 1: Salam, *Towards Jetography* (arXiv:0906.1833), chapters 1-2;
- week 2: Guest, Cranmer and Whiteson, *Deep Learning and its Application to LHC Physics* (arXiv:1806.11484);
- week 3: Alain and Bengio, *Understanding intermediate layers using linear classifier probes* (arXiv:1610.01644);
- week 4: Sundararajan, Taly and Yan, *Axiomatic Attribution for Deep Networks* (arXiv:1703.01365), with Jain and
  Wallace, *Attention is not Explanation* (arXiv:1902.10186).

None of them is needed to answer a question: the primers and the docs of this repository are enough.

### Week 1: jets, observables and generators

**Start here:** Salam, *Towards Jetography*, chapters 1-2, then Marzani, Soyez and Spannowsky, *Looking inside jets*,
chapters 1-3, and section 2 of the review by Larkoski, Moult and Nachman. The last section of the primer
([primer.md](primer.md#where-to-read-more)) lists them next to the docs of this repository.

| reference | used in |
|---|---|
| G. P. Salam, *Towards Jetography*, Eur. Phys. J. C 67, 637 (2010), arXiv:0906.1833 | week 1 (start here: chapters 1-2) |
| S. Marzani, G. Soyez and M. Spannowsky, *Looking inside jets: an introduction to jet substructure and boosted-object phenomenology*, Lecture Notes in Physics, Springer (2019), arXiv:1901.10342 | week 1 (start here: chapters 1-3; Lund plane, mass, soft drop), `W4.1.5` |
| A. J. Larkoski, I. Moult and B. Nachman, *Jet Substructure at the Large Hadron Collider: A Review of Recent Advances in Theory and Machine Learning*, Phys. Rept. 841, 1 (2020), arXiv:1709.04464 | week 1 (start here: section 2), week 2 |
| R. Kogler et al., *Jet Substructure at the Large Hadron Collider: Experimental Review*, Rev. Mod. Phys. 91, 045003 (2019), arXiv:1803.06991 | week 1, week 4 |
| M. Cacciari, G. P. Salam and G. Soyez, *The anti-kt jet clustering algorithm*, JHEP 04 (2008) 063, arXiv:0802.1189 | `W1.1.3`, observables.md#clustering |
| M. Cacciari, G. P. Salam and G. Soyez, *FastJet user manual*, Eur. Phys. J. C 72, 1896 (2012), arXiv:1111.6097 | observables.md#clustering |
| Yu. L. Dokshitzer, G. D. Leder, S. Moretti and B. R. Webber, *Better Jet Clustering Algorithms*, JHEP 08 (1997) 001, arXiv:hep-ph/9707323 | `W1.2.7` (C/A) |
| M. Wobisch and T. Wengler, *Hadronization Corrections to Jet Cross Sections in Deep-Inelastic Scattering*, workshop on Monte Carlo generators for HERA physics (1999), arXiv:hep-ph/9907280 | `W1.2.7` (C/A) |
| J. Thaler and K. Van Tilburg, *Identifying Boosted Objects with N-subjettiness*, JHEP 03 (2011) 015, arXiv:1011.2268 | `W1.4`, `W2.1.2` |
| J. Thaler and K. Van Tilburg, *Maximizing Boosted Top Identification by Minimizing N-subjettiness*, JHEP 02 (2012) 093, arXiv:1108.2701 | `W1.4.4`, `W4.1.5` (minimised axes) |
| A. J. Larkoski, D. Neill and J. Thaler, *Jet Shapes with the Broadening Axis*, JHEP 04 (2014) 017, arXiv:1401.2158 | `W1.4.4` (WTA axes) |
| A. J. Larkoski, G. P. Salam and J. Thaler, *Energy Correlation Functions for Jet Substructure*, JHEP 06 (2013) 108, arXiv:1305.0007 | `W1.4.5` (ECFs, C2) |
| A. J. Larkoski, I. Moult and D. Neill, *Power Counting to Better Jet Observables*, JHEP 12 (2014) 009, arXiv:1409.6298 | `W1.4.6` (D2, power counting) |
| A. J. Larkoski, I. Moult and D. Neill, *Analytic Boosted Boson Discrimination*, preprint (2015), arXiv:1507.03018 | `W1.4` |
| I. Moult, L. Necib and J. Thaler, *New Angles on Energy Correlation Functions*, JHEP 12 (2016) 153, arXiv:1609.07483 | `W1.4.7` |
| A. J. Larkoski, S. Marzani, G. Soyez and J. Thaler, *Soft Drop*, JHEP 05 (2014) 146, arXiv:1402.2657 | `W1.3.2`, `W2.1.2` |
| M. Dasgupta, A. Fregoso, S. Marzani and G. P. Salam, *Towards an understanding of jet substructure*, JHEP 09 (2013) 029, arXiv:1307.0007 | `W1.3.5` (mMDT, mass at leading log) |
| F. A. Dreyer, G. P. Salam and G. Soyez, *The Lund Jet Plane*, JHEP 12 (2018) 064, arXiv:1807.04758 | `W1.2.3`-`W1.2.8`, `W4.5.7` |
| ATLAS Collaboration, *Measurement of the Lund jet plane using charged particles in 13 TeV proton-proton collisions with the ATLAS detector*, Phys. Rev. Lett. 124, 222002 (2020), arXiv:2004.03540 | `W1.2.7` |
| M. Dasgupta, L. Magnea and G. P. Salam, *Non-perturbative QCD effects in jets at hadron colliders*, JHEP 02 (2008) 055, arXiv:0712.3014 | `W1.3.3`, `W1.3.6` (hadronisation and UE shifts of the mass) |
| A. J. Larkoski, J. Thaler and W. J. Waalewijn, *Gaining (Mutual) Information about Quark/Gluon Discrimination*, JHEP 11 (2014) 129, arXiv:1408.3122 | `W1.4.7`, `W2.5.6` (angularities, q/g) |
| P. Gras et al., *Systematics of quark/gluon tagging*, JHEP 07 (2017) 091, arXiv:1704.03878 | `W1.2.5`, `W2.5.6` |
| ATLAS Collaboration, *Measurement of the charged-particle multiplicity inside jets from √s = 8 TeV pp collisions with the ATLAS detector*, Eur. Phys. J. C 76, 322 (2016), arXiv:1602.00988 | `W1.2.5` (multiplicity) |
| I. M. Dremin and J. W. Gary, *Hadron Multiplicities*, Phys. Rept. 349, 301 (2001), arXiv:hep-ph/0004215 | `W1.2.5` (C_A/C_F and the measured ratio) |
| S. Höche, *Introduction to parton-shower event generators*, preprint (2014), arXiv:1411.4085 | week 1 (parton showers) |
| A. Buckley et al., *General-purpose event generators for LHC physics*, Phys. Rept. 504, 145 (2011), arXiv:1101.2599 | week 1 (generators) |
| T. Sjöstrand, S. Mrenna and P. Skands, *A Brief Introduction to PYTHIA 8.1*, Comput. Phys. Commun. 178, 852 (2008), arXiv:0710.3820 | physics.md |
| T. Sjöstrand et al., *An Introduction to PYTHIA 8.2*, Comput. Phys. Commun. 191, 159 (2015), arXiv:1410.3012 | physics.md |
| L. Lönnblad, *Fooling Around with the Sudakov Veto Algorithm*, Eur. Phys. J. C 73, 2588 (2013), arXiv:1211.7204 | `W1.2.10` (veto algorithm) |
| S. Gieseke, P. Stephens and B. Webber, *New formalism for QCD parton showers*, JHEP 12 (2003) 045, arXiv:hep-ph/0310083 | physics.md (angular ordering) |
| M. Bähr et al., *Herwig++ Physics and Manual*, Eur. Phys. J. C 58, 639 (2008), arXiv:0803.0883 | physics.md (reshuffle) |
| A. Papaefstathiou, *Pyresias: How to write a parton shower*, preprint (2024), arXiv:2406.03528 | week 1 (a pedagogical toy shower) |
| K. Cranmer, M. Drnevich, S. Macaluso and D. Pappadopulo, *Reframing Jet Physics with New Computational Methods*, preprint (2021), arXiv:2105.10512 | week 1 (the Ginkgo toy shower) |
| J. Brehmer, S. Macaluso, D. Pappadopulo and K. Cranmer, *Hierarchical clustering in particle physics through reinforcement learning*, preprint (2020), arXiv:2011.08191 | week 1 |
| ATLAS Collaboration, *Measurement of charged-particle distributions sensitive to the underlying event in √s = 13 TeV proton-proton collisions with the ATLAS detector at the LHC*, JHEP 03 (2017) 157, arXiv:1701.05390 | physics.md#underlying-event |
| M. Cacciari and G. P. Salam, *Pileup subtraction using jet areas*, Phys. Lett. B 659, 119 (2008), arXiv:0707.1378 | `W1.4.7` (pileup) |
| D. Bertolini, P. Harris, M. Low and N. Tran, *Pileup Per Particle Identification*, JHEP 10 (2014) 059, arXiv:1407.6013 | `W1.4.7` (pileup) |
| M. Cacciari, G. P. Salam and G. Soyez, *On the use of charged-track information to subtract neutral pileup*, Phys. Rev. D 92, 014003 (2015), arXiv:1404.7353 | `W1.4.7` (pileup) |
| ATLAS and CMS Collaborations, *Combination of the W boson polarization measurements in top quark decays using ATLAS and CMS data at √s = 8 TeV*, JHEP 08 (2020) 051, arXiv:2005.03799 | physics.md#hard-process |

### Week 2: taggers

**Start here:** Guest, Cranmer and Whiteson, *Deep Learning and its Application to LHC Physics*. Then read the Energy
Flow Networks of Komiske, Metodiev and Thaler before `W2.4`. Read the ParT paper of Qu, Li and Qian before `W2.5`; it is
the paper of week 4 too.

| reference | used in |
|---|---|
| D. Guest, K. Cranmer and D. Whiteson, *Deep Learning and its Application to LHC Physics*, Ann. Rev. Nucl. Part. Sci. 68, 161 (2018), arXiv:1806.11484 | week 2 (start here) |
| G. Kasieczka et al., *The Machine Learning Landscape of Top Taggers*, SciPost Phys. 7, 014 (2019), arXiv:1902.09914 | week 2 |
| G. Kasieczka, T. Plehn, M. Russell and T. Schell, *Deep-learning Top Taggers or The End of QCD?*, JHEP 05 (2017) 006, arXiv:1701.08784 | week 2 |
| L. de Oliveira et al., *Jet-Images -- Deep Learning Edition*, JHEP 07 (2016) 069, arXiv:1511.05190 | week 2 |
| M. Zaheer et al., *Deep Sets*, NeurIPS 30 (2017), arXiv:1703.06114 | `W2.4` |
| P. T. Komiske, E. M. Metodiev and J. Thaler, *Energy Flow Networks: Deep Sets for Particle Jets*, JHEP 01 (2019) 121, arXiv:1810.05165 | `W2.4`, `W3.2.2`, `W4.3` |
| H. Qu and L. Gouskos, *ParticleNet: Jet Tagging via Particle Clouds*, Phys. Rev. D 101, 056019 (2020), arXiv:1902.08570 | `W2.5`, `W4.2.2` |
| Y. Wang et al., *Dynamic Graph CNN for Learning on Point Clouds*, ACM Trans. Graph. 38, 146 (2019), arXiv:1801.07829 | `W2.5.1` (EdgeConv) |
| H. Qu, C. Li and S. Qian, *Particle Transformer for Jet Tagging*, ICML 2022, PMLR 162, 18281 (2022), arXiv:2202.03772 | `W2.5`, week 4 |
| F. A. Dreyer and H. Qu, *Jet tagging in the Lund plane with graph networks*, JHEP 03 (2021) 052, arXiv:2012.08526 | `W2.5`, `W4.5.7` |
| A. Vaswani et al., *Attention Is All You Need*, NeurIPS 30 (2017), arXiv:1706.03762 | `W2.5` |
| H. Touvron et al., *Going deeper with Image Transformers*, ICCV (2021), arXiv:2103.17239 | `W2.5.1` (class attention) |
| G. Kasieczka, N. Kiefer, T. Plehn and J. M. Thompson, *Quark-Gluon Tagging: Machine Learning vs Detector*, SciPost Phys. 6, 069 (2019), arXiv:1812.09223 | `W2.5.6`, `W4.2.3` |
| A. Butter, G. Kasieczka, T. Plehn and M. Russell, *Deep-learned Top Tagging with a Lorentz Layer*, SciPost Phys. 5, 028 (2018), arXiv:1707.08966 | week 2 |

### Week 3: representations, probes and self-supervision

**Start here:** Alain and Bengio on linear probes before `W3.1`, and Dillon et al., *Symmetries, Safety, and
Self-Supervision* (JetCLR), before `W3.3.3` and `W3.4`. The papers on IRC safety of `W3.3` are at the end of this list.

| reference | used in |
|---|---|
| G. Alain and Y. Bengio, *Understanding intermediate layers using linear classifier probes*, preprint (2016), arXiv:1610.01644 | `W3.1.2` |
| B. M. Dillon et al., *Symmetries, Safety, and Self-Supervision*, SciPost Phys. 12, 188 (2022), arXiv:2108.04253 | `W3.3.3`, `W3.4` |
| J. Hewitt and P. Liang, *Designing and Interpreting Probes with Control Tasks*, EMNLP (2019), arXiv:1909.03368 | `W3.1.2` (control tasks) |
| Y. Belinkov, *Probing Classifiers: Promises, Shortcomings, and Advances*, Comput. Linguist. 48, 207 (2022), arXiv:2102.12452 | `W3.1.2` |
| S. Kornblith, M. Norouzi, H. Lee and G. Hinton, *Similarity of Neural Network Representations Revisited*, ICML 2019, PMLR 97 (2019), arXiv:1905.00414 | `W3.1.5` |
| L. McInnes, J. Healy and J. Melville, *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*, preprint (2018), arXiv:1802.03426 | `W3.1.3` |
| T. Faucett, J. Thaler and D. Whiteson, *Mapping Machine-Learned Physics into a Human-Readable Space*, Phys. Rev. D 103, 036020 (2021), arXiv:2010.11998 | `W3.1.2`, `W3.1.5` |
| K. Datta, A. Larkoski and B. Nachman, *Automating the Construction of Jet Observables with Machine Learning*, Phys. Rev. D 100, 095016 (2019), arXiv:1902.07180 | week 3 |
| R. Das, G. Kasieczka and D. Shih, *Feature Selection with Distance Correlation*, preprint (2022), arXiv:2212.00046 | week 3 |
| J. Dolen et al., *Thinking outside the ROCs: Designing Decorrelated Taggers (DDT) for jet substructure*, JHEP 05 (2016) 156, arXiv:1603.00027 | `W3.2.3`-`W3.2.5` |
| S. Chang, T. Cohen and B. Ostdiek, *What is the Machine Learning?*, Phys. Rev. D 97, 056009 (2018), arXiv:1709.10106 | `W3.2.4` (planing) |
| G. Louppe, M. Kagan and K. Cranmer, *Learning to Pivot with Adversarial Networks*, NeurIPS 30 (2017), arXiv:1611.01046 | `W3.2.3`-`W3.2.5` |
| C. Shimmin et al., *Decorrelated Jet Substructure Tagging using Adversarial Neural Networks*, Phys. Rev. D 96, 074034 (2017), arXiv:1703.03507 | `W3.2.3`-`W3.2.5` |
| L. Bradshaw, R. K. Mishra, A. Mitridate and B. Ostdiek, *Mass Agnostic Jet Taggers*, preprint (2019), arXiv:1908.08959 | `W3.2.3`-`W3.2.5` |
| G. Kasieczka and D. Shih, *DisCo Fever: Robust Networks Through Distance Correlation*, Phys. Rev. Lett. 125, 122001 (2020), arXiv:2001.05310 | `W3.2.5` (DisCo) |
| O. Kitouni, B. Nachman, C. Weisser and M. Williams, *Enhancing searches for resonances with machine learning and moment decomposition*, JHEP 2021, 70 (2021), arXiv:2010.09745 | `W3.2.3`-`W3.2.5` |
| T. Chen, S. Kornblith, M. Norouzi and G. Hinton, *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020, PMLR 119 (2020), arXiv:2002.05709 | `W3.4.1`-`W3.4.2` |
| B. M. Dillon et al., *Anomalies, Representations, and Self-Supervision*, SciPost Phys. Core 7, 056 (2024), arXiv:2301.04660 | `W3.4` |
| P. T. Komiske, E. M. Metodiev and J. Thaler, *Energy flow polynomials: A complete linear basis for jet substructure*, JHEP 04 (2018) 013, arXiv:1712.07124 | `W3.3.2`, observables.md#irc-safety |
| A. J. Larkoski and J. Thaler, *Unsafe but Calculable: Ratios of Angularities in Perturbative QCD*, JHEP 09 (2013) 137, arXiv:1307.1699 | `W3.3.4` (Sudakov safety) |
| A. J. Larkoski, S. Marzani and J. Thaler, *Sudakov Safety in Perturbative QCD*, Phys. Rev. D 91, 111501 (2015), arXiv:1502.01719 | `W3.3.4` (Sudakov safety) |

### Week 4: JetClass, ParT and interpretability

**Start here:** the ParT paper (week 2) for JetClass and `ParT_kin`. Then read Sundararajan, Taly and Yan on integrated
gradients before `W4.4`, and Jain and Wallace, *Attention is not Explanation*, for the stretch questions of `W4.5`.

| reference | used in |
|---|---|
| M. Sundararajan, A. Taly and Q. Yan, *Axiomatic Attribution for Deep Networks*, ICML 2017, PMLR 70 (2017), arXiv:1703.01365 | `W4.4.2` |
| S. Jain and B. C. Wallace, *Attention is not Explanation*, NAACL (2019), arXiv:1902.10186 | `W4.5.3`-`W4.5.4` |
| K. Simonyan, A. Vedaldi and A. Zisserman, *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps*, preprint (2013), arXiv:1312.6034 | `W4.4.1` (saliency) |
| D. Smilkov et al., *SmoothGrad: removing noise by adding noise*, preprint (2017), arXiv:1706.03825 | `W4.5.9` |
| S. Abnar and W. Zuidema, *Quantifying Attention Flow in Transformers*, ACL (2020), arXiv:2005.00928 | `W4.5.9` (attention rollout) |
| F. Mokhtar, R. Kansal and J. Duarte, *Do graph neural networks learn traditional jet substructure?*, preprint (2022), arXiv:2211.09912 | week 4 |
| A. Khot, M. S. Neubauer and A. Roy, *A Detailed Study of Interpretability of Deep Neural Network based Top Taggers*, preprint (2022), arXiv:2210.04371 | week 4 |
| G. Agarwal et al., *Explainable AI for ML jet taggers using expert variables and layerwise relevance propagation*, preprint (2020), arXiv:2011.13466 | `W4.5.9` (LRP) |
| A. Chakraborty, S. H. Lim and M. M. Nojiri, *Interpretable Deep Learning for Two-Prong Jet Classification with Jet Spectra*, JHEP 2019, 135 (2019), arXiv:1904.02092 | week 4 |
| C. Grojean, A. Paul, Z. Qian and I. Strümke, *Interpretable machine learning in Physics*, preprint (2022), arXiv:2203.08021 | week 4 |
| A. Wang et al., *Interpreting Transformers for Jet Tagging*, preprint (2024), arXiv:2412.03673 | `W4.5.3`, `B.4` |
| T. Legge et al., *Why Is Attention Sparse In Particle Transformer?*, preprint (2025), arXiv:2512.00210 | `W4.5.3`, `B.4` |
| S. Rai and S. Ganguly, *Dissecting Jet-Tagger Through Mechanistic Interpretability*, preprint (2026), arXiv:2605.09881 | `W4.2.9`, `B.4` |
| S. Choi, S. J. Lee and M. Perelstein, *Infrared Safety of a Neural-Net Top Tagging Algorithm*, preprint (2018), arXiv:1806.01263 | `W4.3` |
| M. J. Dolan and A. Ore, *Equivariant Energy Flow Networks for Jet Tagging*, Phys. Rev. D 103, 074022 (2021), arXiv:2012.00964 | `W4.3` |
| P. Konar, V. S. Ngairangbam and M. Spannowsky, *Energy-weighted Message Passing: an infra-red and collinear safe graph neural network algorithm*, JHEP 02 (2022) 060, arXiv:2109.14636 | `W4.3` |
| A. Romero et al., *Safety of Quark/Gluon Jet Classification*, preprint (2021), arXiv:2103.09103 | `W4.3` |

### Bonus: foundation models and masked particle modelling

| reference | used in |
|---|---|
| P. T. Komiske, E. M. Metodiev and J. Thaler, *The Metric Space of Collider Events*, Phys. Rev. Lett. 123, 041801 (2019), arXiv:1902.02346 | bonus |
| T. Golling et al., *Masked Particle Modeling on Sets: Towards Self-Supervised High Energy Physics Foundation Models*, Mach. Learn.: Sci. Technol. 5, 035074 (2024), arXiv:2401.13537 | `B.2`, `B.4` |
| M. Leigh et al., *Is Tokenization Needed for Masked Particle Modelling?*, preprint (2024), arXiv:2409.12589 | `B.2` |
| J. Birk, A. Hallin and G. Kasieczka, *OmniJet-α: The first cross-task foundation model for particle physics*, Mach. Learn.: Sci. Technol. 5, 035031 (2024), arXiv:2403.05618 | bonus |
| J. Birk et al., *OmniJet-α_C: Learning point cloud calorimeter simulations using generative transformers*, preprint (2025), arXiv:2501.05534 | bonus |
| V. Mikuni and B. Nachman, *Solving Key Challenges in Collider Physics with Foundation Models*, Phys. Rev. D 111, L051504 (2025), arXiv:2404.16091 | bonus |
| P. Harris et al., *Re-Simulation-based Self-Supervised Learning for Pre-Training Foundation Models*, Phys. Rev. D 111, 032010 (2025), arXiv:2403.07066 | bonus |
| Z. Zhao et al., *Large-Scale Pretraining and Finetuning for Efficient Jet Classification in Particle Physics*, preprint (2024), arXiv:2408.09343 | `B.1`, bonus |
| C. Li et al., *Accelerating Resonance Searches via Signature-Oriented Pre-training*, preprint (2024), arXiv:2405.12972 | bonus |
| J. Bardhan et al., *HEP-JEPA: A foundation model for collider physics using joint embedding predictive architecture*, preprint (2025), arXiv:2502.03933 | bonus |
| K. He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR (2022), arXiv:2111.06377 | `B.2` |
| J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL (2019), arXiv:1810.04805 | `B.2` |

### Outlook: symmetry-aware networks

| reference | used in |
|---|---|
| S. Gong et al., *An Efficient Lorentz Equivariant Graph Neural Network for Jet Tagging*, JHEP 07 (2022) 030, arXiv:2201.08187 | further reading |
| A. Bogatskiy, T. Hoffman, D. W. Miller and J. T. Offermann, *PELICAN: Permutation Equivariant and Lorentz Invariant or Covariant Aggregator Network for Particle Physics*, preprint (2022), arXiv:2211.00454 | further reading |
| A. Bogatskiy et al., *Explainable Equivariant Neural Networks for Particle Physics: PELICAN*, JHEP 2024, 113 (2024), arXiv:2307.16506 | further reading |
| J. Spinner et al., *Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics*, NeurIPS (2024), arXiv:2405.14806 | further reading |

### General

| reference | used in |
|---|---|
| M. Feickert and B. Nachman, *A Living Review of Machine Learning for Particle Physics*, preprint (2021), arXiv:2102.02770 | all (kept up to date at iml-wg.github.io/HEPML-LivingReview) |
