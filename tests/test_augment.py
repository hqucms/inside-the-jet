"""Perturbations: symmetries, IRC perturbations, recomputed columns, carried truth and the particle guard."""

from __future__ import annotations

import numpy as np
import pytest

from itj import augment
from itj import observables as ob
from itj.data import JetData, Ragged
from synthetic import add_truth_columns, massless, synthetic_jets

_TRUTH_PART = ("truth_part_prong", "truth_part_parton", "truth_part_is_ue", "truth_part_from_b", "truth_part_purity")


def random_jets(n: int = 25, *, seed: int = 0, max_count: int = 30, eta: float = 0.4) -> JetData:
    rng = np.random.default_rng(seed)
    particles = []
    for _ in range(n):
        k = int(rng.integers(2, max_count))
        pt = rng.exponential(30.0, k) + 0.5
        particles.append(
            np.stack(
                [
                    massless(a, b, c)
                    for a, b, c in zip(pt, rng.normal(eta, 0.2, k), rng.normal(0.0, 0.2, k), strict=True)
                ]
            )
        )
    return synthetic_jets(particles)


@pytest.fixture
def jets() -> JetData:
    return add_truth_columns(random_jets())


def flat(data: JetData, name: str) -> np.ndarray:
    col = data[name]
    assert isinstance(col, Ragged)
    return np.asarray(col.values, dtype=np.float64)


def jet_sums(data: JetData) -> np.ndarray:
    p4, _mask = data.padded_p4()
    return p4.sum(axis=1)


def y_phi(data: JetData, i: int) -> tuple[np.ndarray, np.ndarray]:
    p4, mask = data.padded_p4()
    _, y, phi, _ = ob.pt_y_phi_m(p4[i, mask[i]])
    return y, phi


def pairwise_dr(y: np.ndarray, phi: np.ndarray) -> np.ndarray:
    dphi = np.angle(np.exp(1j * (phi[:, None] - phi[None, :])))
    return np.sort(np.hypot(y[:, None] - y[None, :], dphi).ravel())


def check_consistent(data: JetData, sign: np.ndarray | None = None) -> None:
    """Particles pT-descending, jet kinematics from the constituents, deta/dphi with the JetClass convention."""
    px, py, pz = flat(data, "part_px"), flat(data, "part_py"), flat(data, "part_pz")
    pt = np.hypot(px, py)
    row = data["part_px"].row_index()  # type: ignore[union-attr]
    assert np.all((np.diff(pt) <= 0) | (np.diff(row) != 0))
    total = jet_sums(data)
    np.testing.assert_allclose(data["jet_pt"], np.hypot(total[:, 0], total[:, 1]), rtol=1e-5)
    np.testing.assert_allclose(data["jet_energy"], total[:, 3], rtol=1e-5)
    np.testing.assert_array_equal(data["jet_nparticles"], data.counts)
    jet_eta = np.asarray(data["jet_eta"], dtype=np.float64)
    s = np.where(jet_eta >= 0, 1.0, -1.0) if sign is None else sign
    eta = np.arcsinh(pz / pt)
    np.testing.assert_allclose(flat(data, "part_deta"), s[row] * (eta - jet_eta[row]), atol=1e-5)
    dphi = flat(data, "part_dphi")
    assert np.all(np.abs(dphi) <= np.pi)
    np.testing.assert_allclose(
        np.cos(dphi), np.cos(np.arctan2(py, px) - np.asarray(data["jet_phi"], np.float64)[row]), atol=1e-5
    )


# ------------------------------------------------------------------------------------------------------- recompute


def test_recompute_reproduces_exact_jets(jets):
    out = augment.recompute(jets)
    check_consistent(out)
    for name in ("jet_pt", "jet_eta", "jet_phi", "part_px", "part_deta"):
        a, b = out[name], jets[name]
        np.testing.assert_allclose(getattr(a, "values", a), getattr(b, "values", b), rtol=1e-5, atol=1e-6)
    assert out.names == jets.names
    assert all(out[n].dtype == jets[n].dtype for n in jets.names)


def test_recompute_sorts_and_fixes_the_sign():
    jets = random_jets(4, seed=1, eta=-0.5)
    shuffled = {}
    for name in jets.names:
        col = jets[name]
        shuffled[name] = Ragged(col.values[::-1].copy(), col.offsets) if isinstance(col, Ragged) else col
    shuffled_data = JetData(shuffled)  # reversed values: wrong order and wrong rows
    out = augment.recompute(shuffled_data)
    check_consistent(out)
    assert np.all(np.asarray(out["jet_eta"]) < 0)


def test_substructure_only_on_request(jets):
    moved = augment.rotate(jets, angle=0.5)
    np.testing.assert_array_equal(moved["jet_sdmass"], jets["jet_sdmass"])  # stale (zero in the synthetic jets)
    fresh = augment.rotate(jets, angle=0.5, substructure=True)
    p4, mask = fresh.padded_p4()
    branches = ob.jet_branches(p4, mask)
    for name in ("jet_sdmass", "jet_tau1", "jet_tau2", "jet_tau3", "jet_tau4"):
        np.testing.assert_allclose(fresh[name], branches[name], rtol=1e-6, err_msg=name)
    assert np.any(np.asarray(fresh["jet_tau1"]) > 0)


# -------------------------------------------------------------------------------------------------------- symmetries


def test_rotation_keeps_pairwise_delta_r_and_mass(jets):
    out = augment.rotate(jets, rng=3)
    check_consistent(out)
    scalar = Ragged(np.hypot(flat(out, "part_px"), flat(out, "part_py")), out["part_px"].offsets)  # type: ignore
    scalar_in = Ragged(np.hypot(flat(jets, "part_px"), flat(jets, "part_py")), jets["part_px"].offsets)  # type: ignore
    np.testing.assert_allclose(  # the pT of every particle is kept (the vector sum, jet_pt, is not)
        np.add.reduceat(scalar.values, scalar.offsets[:-1]),
        np.add.reduceat(scalar_in.values, scalar_in.offsets[:-1]),
        rtol=1e-5,
    )
    for i in range(len(jets)):
        np.testing.assert_allclose(pairwise_dr(*y_phi(out, i)), pairwise_dr(*y_phi(jets, i)), atol=2e-5)
    m_out = ob.mass(*out.padded_p4())
    m_in = ob.mass(*jets.padded_p4())
    assert np.max(np.abs(m_out / m_in - 1)) < 0.02  # m^2 ~ pt pt dR^2 only to second order in dR
    full = augment.rotate(jets, angle=2 * np.pi)
    np.testing.assert_allclose(flat(full, "part_px"), flat(jets, "part_px"), rtol=1e-4, atol=1e-4)
    angles = np.linspace(0.0, 1.0, len(jets))
    assert len(augment.rotate(jets, angle=angles)) == len(jets)
    with pytest.raises(ValueError, match="one value per jet"):
        augment.rotate(jets, angle=np.zeros(3))


def test_translate_shifts_the_relative_coordinates(jets):
    out = augment.translate(jets, deta=0.1, dphi=-0.2)
    np.testing.assert_array_equal(out["jet_eta"], jets["jet_eta"])  # the stored axis stays
    np.testing.assert_array_equal(out["jet_phi"], jets["jet_phi"])
    np.testing.assert_allclose(flat(out, "part_deta"), flat(jets, "part_deta") + 0.1, atol=1e-5)
    np.testing.assert_allclose(flat(out, "part_dphi"), flat(jets, "part_dphi") - 0.2, atol=1e-5)
    np.testing.assert_allclose(out["jet_pt"], jets["jet_pt"], rtol=1e-3)
    np.testing.assert_allclose(
        np.hypot(flat(out, "part_px"), flat(out, "part_py")),
        np.hypot(flat(jets, "part_px"), flat(jets, "part_py")),
        rtol=1e-5,
    )


def test_smearing(jets):
    soft = augment.smear_soft(jets, scale=0.1, rng=1)
    check_consistent(soft)
    pt_in = np.sort(np.hypot(flat(jets, "part_px"), flat(jets, "part_py")))
    np.testing.assert_allclose(np.sort(np.hypot(flat(soft, "part_px"), flat(soft, "part_py"))), pt_in, rtol=1e-5)
    assert not np.allclose(flat(soft, "part_deta"), flat(jets, "part_deta"))
    hard = augment.smear_pt(jets, sigma=0.1, rng=2)
    check_consistent(hard)
    assert not np.allclose(hard["jet_pt"], jets["jet_pt"])
    np.testing.assert_allclose(
        np.sort(np.arcsinh(flat(hard, "part_pz") / np.hypot(flat(hard, "part_px"), flat(hard, "part_py")))),
        np.sort(np.arcsinh(flat(jets, "part_pz") / np.hypot(flat(jets, "part_px"), flat(jets, "part_py")))),
        atol=1e-5,
    )
    unchanged = augment.smear_pt(jets, sigma=0.0)
    np.testing.assert_allclose(unchanged["jet_pt"], jets["jet_pt"], rtol=1e-6)
    with pytest.raises(ValueError, match="sigma"):
        augment.smear_pt(jets, sigma=-1.0)


# ------------------------------------------------------------------------------------------------- IRC perturbations


def test_collinear_split_conserves_four_momentum_and_irc_safe_observables(jets):
    out = augment.split_collinear(jets, which="hardest", lam=0.3)
    check_consistent(out)
    assert np.all(out.counts == jets.counts + 1)
    np.testing.assert_allclose(jet_sums(out), jet_sums(jets), rtol=1e-5, atol=1e-3)
    p4, mask = jets.padded_p4()
    q4, qmask = out.padded_p4()
    np.testing.assert_allclose(ob.mass(q4, qmask), ob.mass(p4, mask), rtol=1e-4)
    np.testing.assert_allclose(ob.nsubjettiness(q4, qmask), ob.nsubjettiness(p4, mask), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(ob.ecf(q4, qmask)[0], ob.ecf(p4, mask)[0], rtol=1e-4)
    assert np.all(ob.ptd(q4, qmask) < ob.ptd(p4, mask))


def test_collinear_split_variants(jets):
    every = augment.split_collinear(jets, which="all")
    assert np.all(every.counts == 2 * jets.counts)
    some = augment.split_collinear(jets, frac=0.5, rng=0)
    assert np.all(some.counts >= jets.counts)
    assert some.counts.sum() > jets.counts.sum()
    picked = augment.split_collinear(jets, which=np.where(np.arange(len(jets)) % 2 == 0, 1, -1))
    assert (picked.counts - jets.counts).tolist() == [1 - i % 2 for i in range(len(jets))]
    wide = augment.split_collinear(jets, which="hardest", lam=0.5, delta=0.1, rng=4)
    check_consistent(wide)
    p4, _mask = jets.padded_p4()
    hard = p4[:, 0]  # the hardest particle of every jet
    hard_y = ob.pt_y_phi_m(hard)[1]
    q4, qmask = wide.padded_p4()
    _, qy, qphi, _ = ob.pt_y_phi_m(q4)
    # two pieces of pT pt / 2 each, 0.1 apart, centred on the original direction
    pieces = (
        np.isclose(np.hypot(q4[..., 0], q4[..., 1]), 0.5 * np.hypot(hard[:, None, 0], hard[:, None, 1]), rtol=1e-5)
        & qmask
    )
    for i in range(len(jets)):
        k = np.flatnonzero(pieces[i])[:2]
        np.testing.assert_allclose(np.hypot(*np.diff(np.stack([qy[i, k], qphi[i, k]]), axis=1)), 0.1, rtol=1e-3)
        np.testing.assert_allclose(qy[i, k].mean(), hard_y[i], atol=1e-4)
    with pytest.raises(ValueError, match="lam"):
        augment.split_collinear(jets, lam=1.0)
    with pytest.raises(ValueError, match="which"):
        augment.split_collinear(jets, which="softest")  # type: ignore[arg-type]


def test_split_carries_the_particle_columns(jets):
    out = augment.split_collinear(jets, which="hardest", lam=0.4)
    for name in ("part_charge", "part_isChargedHadron", *_TRUTH_PART):
        before = jets[name]
        after = out[name]
        assert isinstance(before, Ragged)
        assert isinstance(after, Ragged)
        assert after.dtype == before.dtype
        for i in (0, 5):
            first = before[i][0]  # both pieces copy the columns of the hardest particle
            assert sorted(after[i].tolist()) == sorted([*before[i].tolist(), first])


def test_add_soft_adds_exactly_eps_jet_pt(jets):
    out = augment.add_soft(jets, n=2, eps=1e-3, kind="photon", R=0.8, rng=5)
    check_consistent(out)
    assert np.all(out.counts == jets.counts + 2)
    added = flat(out, "part_isPhoton") == 1  # the synthetic jets have no PID flags
    assert added.sum() == 2 * len(jets)
    row = out["part_px"].row_index()  # type: ignore[union-attr]
    pt = np.hypot(flat(out, "part_px"), flat(out, "part_py"))
    np.testing.assert_allclose(pt[added], 1e-3 * np.asarray(jets["jet_pt"], np.float64)[row[added]], rtol=1e-5)
    total = jet_sums(jets)
    _, y_j, phi_j, _ = ob.pt_y_phi_m(total)
    _, y, phi, _ = ob.pt_y_phi_m(np.stack([flat(out, n) for n in ("part_px", "part_py", "part_pz", "part_energy")], 1))
    dr = np.hypot(y - y_j[row], np.angle(np.exp(1j * (phi - phi_j[row]))))
    assert np.all(dr[added] < 0.8 + 1e-5)
    # the defaults of added particles
    for name, value in (
        ("truth_part_prong", -1),
        ("truth_part_parton", -1),
        ("truth_part_is_ue", 0),
        ("truth_part_from_b", 0),
        ("truth_part_purity", 1),
        ("part_charge", 0),
        ("part_d0val", 0),
        ("part_dzval", 0),
        ("part_isChargedHadron", 0),
    ):
        assert np.all(flat(out, name)[added] == value), name
    charged = augment.add_soft(jets, n=1, kind="charged", rng=6)
    new = flat(charged, "part_isChargedHadron") == 1
    assert new.sum() == len(jets)
    assert set(np.unique(flat(charged, "part_charge")[new])) <= {-1.0, 1.0}
    with pytest.raises(ValueError, match="kind"):
        augment.add_soft(jets, kind="neutron")  # type: ignore[arg-type]


def test_tiny_soft_particle_is_irc_safe(jets):
    out = augment.add_soft(jets, n=1, eps=1e-9, rng=7)
    p4, mask = jets.padded_p4()
    q4, qmask = out.padded_p4()
    np.testing.assert_allclose(ob.mass(q4, qmask), ob.mass(p4, mask), rtol=1e-6)
    np.testing.assert_allclose(ob.nsubjettiness(q4, qmask), ob.nsubjettiness(p4, mask), rtol=1e-5, atol=1e-9)


def test_drop_remove_keep(jets):
    dropped = augment.drop_soft(jets, pt_max=5.0, p=1.0)
    check_consistent(dropped)
    pt = np.hypot(flat(dropped, "part_px"), flat(dropped, "part_py"))
    lead = dropped["part_px"].offsets[:-1]  # type: ignore[union-attr]
    soft = np.ones(len(pt), dtype=bool)
    soft[lead] = False
    assert np.all(pt[soft] >= 5.0)
    assert np.all(dropped.counts >= 1)
    none = augment.drop_soft(jets, pt_max=5.0, p=0.0)
    np.testing.assert_array_equal(none.counts, jets.counts)
    pt_in = Ragged(np.hypot(flat(jets, "part_px"), flat(jets, "part_py")), jets["part_px"].offsets)  # type: ignore
    removed = augment.remove_particles(jets, pt_in.map(lambda v: v < 2.0))
    kept = augment.keep_particles(jets, pt_in.map(lambda v: v >= 2.0))
    np.testing.assert_array_equal(removed.counts, kept.counts)
    np.testing.assert_array_equal(flat(removed, "truth_part_parton"), flat(kept, "truth_part_parton"))
    assert removed.counts.sum() == (pt_in.values >= 2.0).sum()
    with pytest.raises(ValueError, match="one value per particle"):
        augment.remove_particles(jets, Ragged.from_counts(np.zeros(len(jets), dtype=bool), np.ones(len(jets))))


def test_inputs_are_not_modified(jets):
    before = flat(jets, "part_px").copy()
    augment.split_collinear(jets, which="all")
    augment.rotate(jets, rng=1)
    np.testing.assert_array_equal(flat(jets, "part_px"), before)


# ---------------------------------------------------------------------------------------------------------- sign rule


def test_keep_sign():
    # jet_eta = +0.011; removing the second particle moves it to -0.014
    jets = synthetic_jets(
        [np.stack([massless(300.0, -0.02, 0.0), massless(100.0, 0.1, 0.1), massless(50.0, 0.02, -0.1)])]
    )
    assert float(jets["jet_eta"][0]) > 0
    drop = Ragged.from_counts(np.array([False, True, False]), [3])
    flipped = augment.remove_particles(jets, drop)
    kept = augment.remove_particles(jets, drop, keep_sign=True)
    assert float(flipped["jet_eta"][0]) < 0
    check_consistent(flipped)
    check_consistent(kept, sign=np.array([1.0]))
    np.testing.assert_allclose(flat(kept, "part_deta"), -flat(flipped, "part_deta"), atol=1e-7)
    assert np.any(np.abs(flat(kept, "part_deta")) > 1e-3)
    for fn in (augment.rotate, augment.smear_soft, augment.add_soft, augment.split_collinear, augment.drop_soft):
        out = fn(jets, keep_sign=True)
        check_consistent(out, sign=np.array([1.0]))


# ------------------------------------------------------------------------------------------------------------ guard


def test_max_particles_guard():
    jets = random_jets(3, seed=2, max_count=10)
    augment.max_particles_guard(jets, length=10)
    big = augment.split_collinear(jets, which="all")
    with pytest.raises(ValueError, match="more than 10 particles"):
        augment.max_particles_guard(big, length=10)
    augment.max_particles_guard(big)  # the default length 128


def test_jetclass_sample_round_trip(jetclass_sample):
    out = augment.recompute(jetclass_sample)
    np.testing.assert_allclose(out["jet_pt"], jetclass_sample["jet_pt"], rtol=1e-5)
    np.testing.assert_array_equal(out["jet_nparticles"], jetclass_sample["jet_nparticles"])
    np.testing.assert_allclose(flat(out, "part_dphi"), flat(jetclass_sample, "part_dphi"), atol=1e-2)
    for name in ("part_isPhoton", "part_d0val", "label_QCD", "aux_truth_match"):
        a, b = out[name], jetclass_sample[name]
        np.testing.assert_array_equal(getattr(a, "values", a), getattr(b, "values", b))


def test_split_collinear_conserves_momentum_for_float32_tachyons(jetclass_sample: JetData) -> None:
    # float32 storage leaves many real particles with m^2 slightly below 0; the pieces must still sum to them
    names = ("part_px", "part_py", "part_pz", "part_energy")
    before = np.stack([flat(jetclass_sample, n).astype(np.float64) for n in names], axis=1)
    m2 = before[:, 3] ** 2 - (before[:, :3] ** 2).sum(axis=1)
    assert (m2 < 0).any()
    split = augment.split_collinear(jetclass_sample, which="all", delta=1e-9, rng=0)
    after = np.stack([flat(split, n).astype(np.float64) for n in names], axis=1)
    row_b = np.repeat(np.arange(len(jetclass_sample)), jetclass_sample["part_px"].counts())
    row_a = np.repeat(np.arange(len(split)), split["part_px"].counts())
    sums_b = np.stack([np.bincount(row_b, weights=before[:, k]) for k in range(4)], axis=1)
    sums_a = np.stack([np.bincount(row_a, weights=after[:, k]) for k in range(4)], axis=1)
    np.testing.assert_allclose(sums_a, sums_b, rtol=0, atol=1e-6 * np.abs(sums_b[:, 3:]).max())


def test_split_collinear_in_float64_has_no_rounding_floor(jetclass_sample: JetData, tmp_path) -> None:
    # float32 pieces of a TeV particle get a spurious mass of ~0.1 GeV each; float64 pieces sum to the particle
    from itj.io import load, save

    ref = augment.recompute(jetclass_sample, substructure=True)
    for lam in (0.2, 0.3):
        out = augment.split_collinear(jetclass_sample, lam=lam, delta=0.0, substructure=True)
        assert out["part_px"].dtype == np.float64
        assert out["part_deta"].dtype == np.float64
        assert np.asarray(out["jet_sdmass"]).dtype == np.float64
        assert np.asarray(out["jet_nparticles"]).dtype == np.asarray(jetclass_sample["jet_nparticles"]).dtype
        for name in ("jet_sdmass", "jet_tau2", "jet_pt"):
            np.testing.assert_allclose(np.asarray(out[name]), np.asarray(ref[name], np.float64), rtol=1e-6)
    stored = augment.split_collinear(jetclass_sample, lam=0.2, float64=False)
    assert stored["part_px"].dtype == np.float32
    assert np.asarray(stored["jet_pt"]).dtype == np.float32
    # files keep their dtypes
    path = tmp_path / "split.parquet"
    save(augment.split_collinear(jetclass_sample.head(5), lam=0.2), path)
    assert load(path)["part_px"].dtype == np.float32
