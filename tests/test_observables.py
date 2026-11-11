"""Observables: analytic cases, chunking, IRC numerics, soft drop, the Lund plane and the JetClass conventions."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from itj import observables as ob
from itj.data import JETCLASS_JET_BRANCHES, Ragged
from synthetic import ca_tree_jet, massless, synthetic_jets, three_prong_jet, two_particle_jet


def as_arrays(*jets) -> tuple[np.ndarray, np.ndarray]:
    """Pad lists of four-vectors into (B, N, 4) and (B, N)."""
    n = max(len(j) for j in jets)
    p4 = np.zeros((len(jets), n, 4))
    mask = np.zeros((len(jets), n), dtype=bool)
    for b, jet in enumerate(jets):
        p4[b, : len(jet)] = np.asarray(jet)
        mask[b, : len(jet)] = True
    return p4, mask


def random_jets(n_jets: int = 20, n_max: int = 30, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random massless jets of 2 .. n_max particles around (y, phi) = (0.5, 0.2), padded."""
    rng = np.random.default_rng(seed)
    jets = []
    for _ in range(n_jets):
        k = int(rng.integers(2, n_max + 1))
        pt = rng.exponential(30.0, k) + 0.5
        y = rng.normal(0.5, 0.25, k)
        phi = rng.normal(0.2, 0.25, k)
        jets.append([massless(*v) for v in zip(pt, y, phi, strict=True)])
    return as_arrays(*jets)


def with_split_hardest(p4: np.ndarray, mask: np.ndarray, lam: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """The same jets with the hardest particle replaced by lam p and (1 - lam) p (exactly collinear)."""
    pt = np.where(mask, np.hypot(p4[..., 0], p4[..., 1]), -1.0)
    hard = pt.argmax(axis=1)
    rows = np.arange(len(p4))
    out = np.concatenate([p4, np.zeros((len(p4), 1, 4))], axis=1)
    out_mask = np.concatenate([mask, np.ones((len(p4), 1), dtype=bool)], axis=1)
    out[rows, -1] = (1.0 - lam) * p4[rows, hard]
    out[rows, hard] = lam * p4[rows, hard]
    return out, out_mask


def with_soft(p4: np.ndarray, mask: np.ndarray, eps: float, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The same jets with one massless particle of pT eps * pt_J added within dR < 0.8 of the jet axis."""
    rng = np.random.default_rng(seed)
    total = p4.sum(axis=1)
    pt_j = np.hypot(total[:, 0], total[:, 1])
    _, y, phi, _ = ob.pt_y_phi_m(total)
    r, a = 0.8 * np.sqrt(rng.random(len(p4))), rng.uniform(0, 2 * np.pi, len(p4))
    soft = np.stack(
        [
            massless(eps * p, yy + rr * np.cos(aa), ff + rr * np.sin(aa))
            for p, yy, ff, rr, aa in zip(pt_j, y, phi, r, a, strict=True)
        ]
    )
    out = np.concatenate([p4, soft[:, None]], axis=1)
    return out, np.concatenate([mask, np.ones((len(p4), 1), dtype=bool)], axis=1)


# --------------------------------------------------------------------------------------------------- kinematics


def test_two_particle_mass():
    p4, mask = as_arrays([massless(400.0, 0.1, 0.3), massless(250.0, -0.2, -0.1)])
    expected = np.sqrt(2 * 400.0 * 250.0 * (np.cosh(0.3) - np.cos(0.4)))
    np.testing.assert_allclose(ob.mass(p4, mask), [expected], rtol=1e-12)
    np.testing.assert_allclose(ob.jet_p4(p4, mask), p4.sum(axis=1), rtol=1e-12)
    assert ob.multiplicity(mask).tolist() == [2]
    assert ob.multiplicity(mask).dtype == np.int64


def test_pt_y_phi_m():
    p4 = np.array([massless(10.0, 1.5, -2.0), [0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 0.0, 0.0]])
    p4[0, 3] = np.sqrt(p4[0, 3] ** 2 + 4.0)  # m = 2
    pt, y, phi, m = ob.pt_y_phi_m(p4)
    np.testing.assert_allclose(pt, [10.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(phi[0], -2.0, rtol=1e-12)
    np.testing.assert_allclose(m, [2.0, 0.0, 0.0], atol=1e-6)
    assert y[1] == 1e5
    assert y[2] == 0.0
    np.testing.assert_allclose(y[0], 0.5 * np.log((p4[0, 3] + p4[0, 2]) / (p4[0, 3] - p4[0, 2])), rtol=1e-12)


# ---------------------------------------------------------------------------------------------------------- tau_N


def test_tau_n_vanishes_for_an_exact_n_prong_jet():
    prongs = [(0.0, 0.0), (0.4, 0.1), (-0.1, 0.45)]
    particles = [massless(pt, y, phi) for y, phi in prongs for pt in (100.0, 60.0, 30.0)]
    p4, mask = as_arrays(particles)
    tau = ob.nsubjettiness(p4, mask)[0]
    assert tau[0] > tau[1] > 0.0
    np.testing.assert_allclose(tau[2:], 0.0, atol=1e-12)
    for axes in ("kt", "onepass_kt"):
        np.testing.assert_allclose(ob.nsubjettiness(p4, mask, axes=axes)[0, 2], 0.0, atol=1e-12)


def test_tau_n_of_two_particles():
    p4, mask = as_arrays([massless(400.0, 0.0, 0.0), massless(200.0, 0.4, 0.0)])
    tau = ob.nsubjettiness(p4, mask, n=(1, 2, 3))
    np.testing.assert_allclose(tau, [[200.0 * 0.4 / (600.0 * 0.8), 0.0, 0.0]], rtol=1e-9)  # WTA axis on the harder
    np.testing.assert_allclose(ob.nsubjettiness(p4, mask, n=(1,), beta=2.0), [[200.0 * 0.16 / (600.0 * 0.64)]])
    with pytest.raises(ValueError, match="did you mean 'wta_kt'"):
        ob.nsubjettiness(p4, mask, axes="wta")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="every N must be >= 1"):
        ob.nsubjettiness(p4, mask, n=(0,))


def test_subjet_axes():
    jet = three_prong_jet(dr=0.5)
    p4, mask = jet.padded_p4()
    axes = ob.subjet_axes(p4, mask, 3)
    assert axes.shape == (1, 3, 2)
    _, y, _phi, _ = ob.pt_y_phi_m(p4[0])
    assert sorted(np.round(axes[0, :, 0], 5).tolist()) == sorted(np.round(y, 5).tolist())
    four = ob.subjet_axes(p4, mask, 4)  # three particles: the fourth axis does not exist
    assert np.isnan(four[0, 3]).all()
    assert np.isfinite(four[0, :3]).all()
    two = ob.subjet_axes(p4, mask, 2, axes="kt")
    assert np.isfinite(two).all()


def test_onepass_axes_lower_tau(jetclass_sample):
    p4, mask = jetclass_sample.padded_p4()
    kt = ob.nsubjettiness(p4, mask, n=(1, 2, 3), axes="kt")
    wta = ob.nsubjettiness(p4, mask, n=(1, 2, 3), axes="wta_kt")
    onepass = ob.nsubjettiness(p4, mask, n=(1, 2, 3), axes="onepass_kt")
    assert np.all(onepass <= kt + 1e-9)
    ok = wta > 0
    assert np.median(onepass[ok] / wta[ok]) < 1.0


# ----------------------------------------------------------------------------------------------------------- ECFs


def brute_force_ecf(p4: np.ndarray, mask: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    e2, e3 = np.zeros(len(p4)), np.zeros(len(p4))
    for b in range(len(p4)):
        pt, y, phi, _ = ob.pt_y_phi_m(p4[b, mask[b]])
        z = pt / pt.sum()

        def dr(i, j, y=y, phi=phi):
            return np.hypot(y[i] - y[j], np.angle(np.exp(1j * (phi[i] - phi[j]))))

        n = len(z)
        e2[b] = sum(z[i] * z[j] * dr(i, j) ** beta for i, j in itertools.combinations(range(n), 2))
        e3[b] = sum(
            z[i] * z[j] * z[k] * (dr(i, j) * dr(i, k) * dr(j, k)) ** beta
            for i, j, k in itertools.combinations(range(n), 3)
        )
    return e2, e3


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_matmul_ecf_equals_brute_force(beta):
    p4, mask = random_jets(8, 12, seed=2)
    e2, e3 = ob.ecf(p4, mask, beta=beta)
    b2, b3 = brute_force_ecf(p4, mask, beta)
    np.testing.assert_allclose(e2, b2, rtol=1e-10)
    np.testing.assert_allclose(e3, b3, rtol=1e-10)
    np.testing.assert_allclose(ob.d2(p4, mask, beta=beta), b3 / b2**3, rtol=1e-9)
    np.testing.assert_allclose(ob.c2(p4, mask, beta=beta), b3 / b2**2, rtol=1e-9)


def test_ecf_of_two_and_one_particles():
    jet = two_particle_jet(0.3, 0.4)
    p4, mask = jet.padded_p4()
    e2, e3 = ob.ecf(p4, mask)
    np.testing.assert_allclose(e2, [0.3 * 0.7 * 0.4], rtol=1e-6)
    assert e3.tolist() == [0.0]
    one, one_mask = as_arrays([massless(100.0, 0.0, 0.0)])
    assert ob.d2(one, one_mask).tolist() == [0.0]
    assert ob.c2(one, one_mask).tolist() == [0.0]


# ------------------------------------------------------------------------------------------------------- chunking


def test_chunked_equals_unchunked():
    p4, mask = random_jets(23, 30, seed=4)
    for chunk in (1, 5):
        sd, sd_c = ob.softdrop(p4, mask), ob.softdrop(p4, mask, chunk_size=chunk)
        for name in ("mass", "pt", "zg", "rg", "n_dropped"):
            np.testing.assert_array_equal(getattr(sd, name), getattr(sd_c, name))
        np.testing.assert_allclose(ob.nsubjettiness(p4, mask), ob.nsubjettiness(p4, mask, chunk_size=chunk))
        np.testing.assert_allclose(ob.ecf(p4, mask), ob.ecf(p4, mask, chunk_size=chunk), rtol=1e-12)
        np.testing.assert_allclose(ob.subjet_axes(p4, mask, 2), ob.subjet_axes(p4, mask, 2, chunk_size=chunk))
        a, b = ob.lund_primary(p4, mask), ob.lund_primary(p4, mask, chunk_size=chunk)
        for key in a:
            np.testing.assert_array_equal(a[key].offsets, b[key].offsets)
            np.testing.assert_allclose(a[key].values, b[key].values)
        a_br, b_br = ob.jet_branches(p4, mask), ob.jet_branches(p4, mask, chunk_size=chunk)
        for key in a_br:
            np.testing.assert_array_equal(a_br[key], b_br[key])
    with pytest.raises(ValueError, match="chunk_size"):
        ob.ecf(p4, mask, chunk_size=0)


def test_padding_does_not_change_the_values():
    p4, mask = random_jets(6, 10, seed=6)
    wide = np.concatenate([p4, np.zeros((6, 7, 4))], axis=1)
    wide_mask = np.concatenate([mask, np.zeros((6, 7), dtype=bool)], axis=1)
    np.testing.assert_allclose(ob.nsubjettiness(wide, wide_mask), ob.nsubjettiness(p4, mask))
    np.testing.assert_allclose(ob.ecf(wide, wide_mask), ob.ecf(p4, mask), rtol=1e-12)
    np.testing.assert_allclose(ob.softdrop(wide, wide_mask).mass, ob.softdrop(p4, mask).mass)


# ------------------------------------------------------------------------------------------------------------- IRC


def test_collinear_split_leaves_irc_safe_observables():
    p4, mask = random_jets(20, 30, seed=8)
    split, split_mask = with_split_hardest(p4, mask)
    np.testing.assert_allclose(ob.mass(split, split_mask), ob.mass(p4, mask), rtol=1e-6)
    np.testing.assert_allclose(ob.nsubjettiness(split, split_mask), ob.nsubjettiness(p4, mask), rtol=1e-6, atol=1e-12)
    for got, want in zip(ob.ecf(split, split_mask), ob.ecf(p4, mask), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(ob.softdrop(split, split_mask).mass, ob.softdrop(p4, mask).mass, rtol=1e-6)
    # the IRC-unsafe ones change
    assert np.all(ob.multiplicity(split_mask) == ob.multiplicity(mask) + 1)
    assert np.all(ob.ptd(split, split_mask) < ob.ptd(p4, mask) - 1e-3)
    assert np.all(ob.leading_fraction(split, split_mask) < ob.leading_fraction(p4, mask))


def test_soft_particle_leaves_irc_safe_observables():
    p4, mask = random_jets(20, 30, seed=9)
    soft, soft_mask = with_soft(p4, mask, eps=1e-9)
    for got, want in [
        (ob.mass(soft, soft_mask), ob.mass(p4, mask)),
        (ob.nsubjettiness(soft, soft_mask), ob.nsubjettiness(p4, mask)),
        *zip(ob.ecf(soft, soft_mask), ob.ecf(p4, mask), strict=True),
    ]:
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)
    assert np.all(ob.multiplicity(soft_mask) == ob.multiplicity(mask) + 1)
    # the soft particle changes the count-based angularity (kappa = 0) by exactly one
    np.testing.assert_allclose(
        ob.angularity(soft, soft_mask, kappa=0.0, beta=0.0), ob.angularity(p4, mask, kappa=0.0, beta=0.0) + 1
    )


# ------------------------------------------------------------------------------------------------------- soft drop


def test_softdrop_on_two_particle_jets():
    def pair(z: float, dr: float = 0.4) -> list[np.ndarray]:  # float64, unlike the float32 synthetic jets
        return [massless(z * 600.0, (1 - z) * dr, 0.0), massless((1 - z) * 600.0, -z * dr, 0.0)]

    p4, mask = as_arrays(pair(0.03), pair(0.3), pair(0.07))
    full = ob.mass(p4, mask)
    sd = ob.softdrop(p4, mask)
    np.testing.assert_allclose(sd.mass, [0.0, full[1], 0.0], atol=1e-4)  # sqrt(E^2 - p^2) of a massless particle
    assert sd.n_dropped.tolist() == [1, 0, 1]
    np.testing.assert_allclose(sd.zg, [0.0, 0.3, 0.0], atol=1e-12)
    np.testing.assert_allclose(sd.rg, [0.0, 0.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(sd.pt[0], 0.97 * 600.0, rtol=1e-12)
    # beta = 1: the condition zcut (dR / R0) = 0.05 lets z = 0.07 pass
    sd1 = ob.softdrop(p4, mask, beta=1.0)
    assert sd1.n_dropped.tolist() == [1, 0, 0]
    np.testing.assert_allclose(sd1.mass[2], full[2], rtol=1e-9)


def test_softdrop_on_the_ca_tree():
    p4, mask = ca_tree_jet().padded_p4()
    sd = ob.softdrop(p4, mask)  # the first splitting (a b)(c d) has z = 0.33: nothing is dropped
    np.testing.assert_allclose(sd.mass, ob.mass(p4, mask), rtol=1e-9)
    assert sd.n_dropped.tolist() == [0]
    sd = ob.softdrop(p4, mask, zcut=0.4)  # drops (c d), then b (z = 0.25): a alone
    assert sd.n_dropped.tolist() == [2]
    np.testing.assert_allclose(sd.pt, [300.0], rtol=1e-6)
    np.testing.assert_allclose(sd.mass, [0.0], atol=1e-2)


# ---------------------------------------------------------------------------------------------------------- Lund


def test_lund_primary_of_the_ca_tree():
    p4, mask = ca_tree_jet().padded_p4()
    lund = ob.lund_primary(p4, mask)
    assert set(lund) == {"ln_inv_delta", "ln_kt", "z", "psi"}
    assert lund["z"].counts().tolist() == [2]
    np.testing.assert_allclose(lund["ln_inv_delta"][0][1], np.log(10.0), rtol=1e-6)
    np.testing.assert_allclose(lund["ln_kt"][0][1], np.log(100.0 * 0.1), rtol=1e-6)
    np.testing.assert_allclose(lund["z"][0][1], 0.25, rtol=1e-6)
    assert lund["ln_inv_delta"][0][0] < lund["ln_inv_delta"][0][1]  # largest angle first


def test_lund_density():
    lund = {
        "ln_inv_delta": Ragged.from_counts(np.array([0.5, 1.5, 1.6, np.inf]), [3, 1]),
        "ln_kt": Ragged.from_counts(np.array([0.5, 0.5, 0.7, 0.5]), [3, 1]),
    }
    density, error = ob.lund_density(lund, n_jets=2, x_edges=np.array([0.0, 1.0, 2.0]), y_edges=np.array([0.0, 2.0]))
    np.testing.assert_allclose(density, [[1 / 4], [2 / 4]])  # per jet per unit area (area 2), the inf dropped
    np.testing.assert_allclose(error, [[1 / 4], [np.sqrt(2) / 4]])
    with pytest.raises(ValueError, match="n_jets"):
        ob.lund_density(lund, n_jets=0, x_edges=np.array([0.0, 1.0]), y_edges=np.array([0.0, 1.0]))


# --------------------------------------------------------------------------------------------------- angularities


def test_angularities_and_ptd():
    particles = [massless(100.0, 0.0, 0.0), massless(100.0, 0.2, 0.0), massless(100.0, 0.0, 0.2)]
    p4, mask = as_arrays(particles)
    np.testing.assert_allclose(ob.ptd(p4, mask), [1 / np.sqrt(3)], rtol=1e-12)
    np.testing.assert_allclose(ob.leading_fraction(p4, mask), [1 / 3], rtol=1e-12)
    assert ob.angularity(p4, mask, kappa=0.0, beta=0.0).tolist() == [3.0]
    np.testing.assert_allclose(ob.angularity(p4, mask, kappa=2.0, beta=0.0), ob.ptd(p4, mask) ** 2, rtol=1e-12)
    # the width (girth): sum z dR / R about the jet axis
    _, y, phi, _ = ob.pt_y_phi_m(p4.sum(axis=1))
    dr = np.hypot(np.array([0.0, 0.2, 0.0]) - y[0], np.array([0.0, 0.0, 0.2]) - phi[0])
    np.testing.assert_allclose(ob.width(p4, mask), [dr.sum() / 3 / 0.8], rtol=1e-6)


# ------------------------------------------------------------------------------------------------ JetClass branches


def test_jet_branches_names_dtypes_and_kinematics():
    jets = synthetic_jets(
        [
            np.stack([massless(300.0, 0.5, 0.1), massless(100.0, 0.7, -0.2)]),
            np.stack([massless(250.0, -1.0, 3.0), massless(50.0, -1.2, -3.0)]),
        ]
    )
    p4, mask = jets.padded_p4()
    branches = ob.jet_branches(p4, mask)
    assert tuple(branches) == JETCLASS_JET_BRANCHES
    assert all(v.dtype == np.float32 and v.shape == (2,) for v in branches.values())
    for name in ("jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_nparticles"):
        np.testing.assert_allclose(branches[name], jets[name], rtol=1e-6, err_msg=name)
    assert branches["jet_tau2"].tolist() == [0.0, 0.0]  # two particles: tau_2 = 0 as JetClass
    with pytest.raises(ValueError, match="unknown axes"):
        ob.jet_branches(p4, mask, tau_axes="minimal")  # type: ignore[arg-type]


def test_jetclass_conventions_reproduced(jetclass_sample):
    data = jetclass_sample
    p4, mask = data.padded_p4()
    sd = ob.softdrop(p4, mask)
    stored = np.asarray(data["jet_sdmass"], dtype=np.float64)
    assert np.mean(np.abs(sd.mass - stored) <= 0.01 * stored) >= 0.80
    medians = {}
    for axes in ("wta_kt", "kt", "onepass_kt"):
        tau = ob.nsubjettiness(p4, mask, n=(1, 2, 3), axes=axes)
        medians[axes] = []
        for k in range(3):
            ref = np.asarray(data[f"jet_tau{k + 1}"], dtype=np.float64)
            ok = ref > 0
            medians[axes].append(float(np.median(tau[ok, k] / ref[ok])))
    assert all(0.99 <= m <= 1.05 for m in medians["wta_kt"]), medians
    assert all(m > 1.05 for m in medians["kt"]), medians
    assert all(0.97 <= m <= 1.03 for m in medians["onepass_kt"]), medians
    branches = ob.jet_branches(p4, mask)
    np.testing.assert_array_equal(branches["jet_nparticles"], data["jet_nparticles"])
    np.testing.assert_allclose(branches["jet_pt"], data["jet_pt"], rtol=1e-5)
    np.testing.assert_allclose(branches["jet_sdmass"], sd.mass.astype(np.float32))


# ---------------------------------------------------------------------------------------------------------- compute


def test_compute_matches_the_functions(jetclass_sample):
    data = jetclass_sample.head(40)
    names = list(ob.COMPUTE_NAMES)
    values = ob.compute(data, names, chunk_size=16)
    assert list(values) == names
    p4, mask = data.padded_p4()
    tau = ob.nsubjettiness(p4, mask)
    e2, e3 = ob.ecf(p4, mask)
    e2b, e3b = ob.ecf(p4, mask, beta=2.0)
    expected = {
        "mass": ob.mass(p4, mask),
        "sdmass": ob.softdrop(p4, mask).mass,
        **{f"tau{k + 1}": tau[:, k] for k in range(4)},
        "tau21": tau[:, 1] / tau[:, 0],
        "tau32": tau[:, 2] / tau[:, 1],
        "e2": e2,
        "e3": e3,
        "d2": e3 / e2**3,
        "c2": e3 / e2**2,
        "d2_b2": e3b / e2b**3,
        "c2_b2": e3b / e2b**2,
        "multiplicity": ob.multiplicity(mask),
        "ptd": ob.ptd(p4, mask),
        "width": ob.width(p4, mask),
        "leading_fraction": ob.leading_fraction(p4, mask),
    }
    for name in names:
        np.testing.assert_allclose(values[name], expected[name], rtol=1e-9, err_msg=name)
    assert values["multiplicity"].dtype == np.int64
    with pytest.raises(ValueError, match="did you mean 'tau21'"):
        ob.compute(data, ["tau_21"])
