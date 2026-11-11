"""Clustering: hand-built trees, recombination schemes, exclusive subjets, bucketing, cones and fastjet equality."""

from __future__ import annotations

import time

import numpy as np
import pytest

from itj.cluster import ClusterTree, cluster, cone, decluster_primary, leading_jet
from synthetic import ca_tree_jet, massless


def as_arrays(*jets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad lists of four-vectors into (B, N, 4) and (B, N)."""
    n = max(len(j) for j in jets)
    p4 = np.zeros((len(jets), n, 4))
    mask = np.zeros((len(jets), n), dtype=bool)
    for b, jet in enumerate(jets):
        p4[b, : len(jet)] = np.asarray(jet)
        mask[b, : len(jet)] = True
    return p4, mask


def random_jets(n_jets: int, n_max: int, *, seed: int = 0, spread: float = 0.3, masses: bool = False, aligned=True):
    """Random jets with 1 .. n_max particles (in random slots unless ``aligned``) around a random axis."""
    rng = np.random.default_rng(seed)
    p4 = np.zeros((n_jets, n_max, 4))
    mask = np.zeros((n_jets, n_max), dtype=bool)
    for b in range(n_jets):
        k = int(rng.integers(1, n_max + 1))
        pt = rng.exponential(20.0, k) + 0.1
        y = rng.normal(rng.uniform(-2.0, 2.0), spread, k)
        phi = rng.normal(rng.uniform(-np.pi, np.pi), spread, k)
        m = rng.uniform(0.0, 0.5, k) if masses else np.zeros(k)
        mt = np.hypot(pt, m)
        slots = np.arange(k) if aligned else np.sort(rng.choice(n_max, k, replace=False))
        p4[b, slots] = np.stack([pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y), mt * np.cosh(y)], axis=1)
        mask[b, slots] = True
    return p4, mask


def ca_tree_arrays() -> tuple[np.ndarray, np.ndarray]:
    jet = ca_tree_jet()
    return jet.padded_p4()


def merges(tree: ClusterTree, b: int = 0) -> list[list[int]]:
    """The sorted children of every merge of jet b, in merge order."""
    big = tree.n_slots
    return [sorted(tree.children[b, big + k].tolist()) for k in range(int(tree.n_leaves[b]) - 1)]


# ------------------------------------------------------------------------------------------------- hand-built trees


def test_ca_tree_of_the_synthetic_jet():
    p4, mask = ca_tree_arrays()  # pT order: a (300), c (150), b (100), d (50) in slots 0, 1, 2, 3
    tree = cluster(p4, mask, algorithm="ca", R=0.8)
    assert merges(tree) == [[0, 2], [1, 3], [4, 5]]
    assert tree.root.tolist() == [6]
    assert tree.parent[0, :6].tolist() == [4, 5, 4, 5, 6, 6]
    assert tree.parent[0, 6] == -1
    np.testing.assert_allclose(tree.dij[0, 4:6], np.array([0.1, 0.25]) ** 2 / 0.64, rtol=1e-6)  # float32 inputs
    assert np.isnan(tree.dij[0, :4]).all()
    np.testing.assert_allclose(tree.p4[0, 6], p4[0].sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(tree.p4[0, 4], p4[0, 0] + p4[0, 2], rtol=1e-12)
    assert tree.n_leaves.tolist() == [4]
    assert tree.children[0, :4].tolist() == [[-1, -1]] * 4


def test_the_three_algorithms_differ_as_expected():
    # a (300 at 0), b (100 at 0.25), c (10 at 0.6): anti-kt and C/A merge a + b first, kt merges b + c first
    p4, mask = as_arrays([massless(300, 0.0, 0.0), massless(100, 0.25, 0.0), massless(10, 0.6, 0.0)])
    first = {alg: merges(cluster(p4, mask, algorithm=alg))[0] for alg in ("antikt", "ca", "kt")}
    assert first == {"antikt": [0, 1], "ca": [0, 1], "kt": [1, 2]}
    kt = cluster(p4, mask, algorithm="kt", R=1.0)
    np.testing.assert_allclose(kt.dij[0, 3], 10.0**2 * 0.35**2, rtol=1e-9)
    antikt = cluster(p4, mask, algorithm="antikt", R=1.0)
    np.testing.assert_allclose(antikt.dij[0, 3], 0.25**2 / 300.0**2, rtol=1e-9)


def test_e_scheme_and_wta():
    p4, mask = as_arrays([massless(300, 0.0, 0.1), massless(100, 0.3, -0.2)])
    e = cluster(p4, mask, algorithm="ca", recombination="E")
    wta = cluster(p4, mask, algorithm="ca", recombination="WTA")
    np.testing.assert_allclose(e.p4[0, 2], p4[0].sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(wta.pt[0, 2], 400.0, rtol=1e-12)
    np.testing.assert_allclose(wta.direction[0, 2], [0.0, 0.1], atol=1e-12)  # the direction of the harder
    assert e.direction[0, 2, 0] > 0.0
    assert e.direction[0, 2, 1] < 0.1
    np.testing.assert_allclose(wta.direction[0, :2], [[0.0, 0.1], [0.3, -0.2]], atol=1e-12)


def test_single_particle_and_empty_jets():
    p4, mask = as_arrays([massless(100, 0.0, 0.0)], [massless(50, 0.1, 0.0), massless(40, 0.2, 0.0)])
    empty = np.zeros_like(mask)
    tree = cluster(p4, mask, algorithm="kt")
    assert tree.root.tolist() == [0, 2]
    assert tree.exclusive_nodes(1).tolist() == [[0], [2]]
    assert cluster(p4, empty, algorithm="kt").root.tolist() == [-1, -1]
    assert leading_jet(p4, empty).sum() == 0


def test_exclusive_nodes_and_labels():
    p4, mask = ca_tree_arrays()
    tree = cluster(p4, mask, algorithm="ca")
    assert tree.exclusive_nodes(1).tolist() == [[6]]
    assert tree.exclusive_nodes(2).tolist() == [[4, 5]]  # {a, b} has pT 400, {c, d} 200
    assert tree.exclusive_nodes(4).tolist() == [[0, 1, 2, 3]]
    assert tree.exclusive_nodes(5).tolist() == [[0, 1, 2, 3, -1]]
    assert tree.exclusive_labels(1).tolist() == [[0, 0, 0, 0]]
    assert tree.exclusive_labels(2).tolist() == [[0, 1, 0, 1]]
    assert tree.exclusive_labels(3).tolist() == [[0, 1, 0, 2]]  # a + b (400), c (150), d (50)
    with pytest.raises(ValueError, match="n must be >= 1"):
        tree.exclusive_nodes(0)


def test_exclusive_labels_with_padding_and_kt():
    p4, mask = as_arrays(
        [massless(300, 0.0, 0.0), massless(200, 0.5, 0.0), massless(5, 0.52, 0.0)],
        [massless(100, 0.0, 0.0), massless(90, 0.4, 0.4)],
    )
    tree = cluster(p4, mask, algorithm="kt")
    labels = tree.exclusive_labels(2)
    assert labels.tolist() == [[0, 1, 1], [0, 1, -1]]
    # the exclusive kt subjets partition the particles: their four-momenta add up to the jet
    nodes = tree.exclusive_nodes(2)
    np.testing.assert_allclose(tree.p4[0, nodes[0]].sum(axis=0), p4[0].sum(axis=0), rtol=1e-12)


def test_leaves_of():
    p4, mask = ca_tree_arrays()
    tree = cluster(p4, mask, algorithm="ca")
    assert tree.leaves_of(np.array([5])).tolist() == [[False, True, False, True]]
    assert tree.leaves_of(np.array([6])).all()
    assert not tree.leaves_of(np.array([-1])).any()
    with pytest.raises(ValueError, match="one node per jet"):
        tree.leaves_of(np.array([1, 2]))


def test_inclusive_mode_finds_two_jets():
    p4, mask = ca_tree_arrays()
    tree = cluster(p4, mask, algorithm="ca", R=0.4, inclusive=True)
    assert tree.root.tolist() == [-1]
    jets = np.flatnonzero(tree.jets[0])
    assert sorted(tree.leaves_of(np.array([j])).nonzero()[1].tolist() for j in jets) == [[0, 2], [1, 3]]
    np.testing.assert_allclose(tree.p4[0, jets].sum(axis=0), p4[0].sum(axis=0), rtol=1e-12)
    assert leading_jet(p4, mask, algorithm="ca", R=0.4).tolist() == [[True, False, True, False]]
    assert leading_jet(p4, mask, algorithm="ca", R=0.8).all()
    with pytest.raises(ValueError, match="exclusive mode"):
        tree.exclusive_nodes(2)


@pytest.mark.parametrize("algorithm", ["antikt", "ca", "kt"])
@pytest.mark.parametrize("recombination", ["E", "WTA"])
def test_bucketing_preserves_order(algorithm, recombination):
    p4, mask = random_jets(40, 25, seed=3, masses=True, aligned=False)
    small = cluster(p4, mask, algorithm=algorithm, recombination=recombination, bucket_size=3)
    large = cluster(p4, mask, algorithm=algorithm, recombination=recombination, bucket_size=1000)
    for name in ("p4", "direction", "children", "parent", "dij", "root", "n_leaves"):
        np.testing.assert_array_equal(getattr(small, name), getattr(large, name), err_msg=name)
    for b in (0, 7, 39):  # every jet alone gives the same tree
        alone = cluster(p4[b : b + 1], mask[b : b + 1], algorithm=algorithm, recombination=recombination)
        np.testing.assert_allclose(alone.p4, large.p4[b : b + 1], rtol=1e-12, atol=1e-9)
        np.testing.assert_array_equal(alone.children, large.children[b : b + 1])


def test_masked_slots_are_ignored():
    p4, mask = random_jets(10, 12, seed=5, aligned=False)
    garbage = np.where(mask[..., None], p4, 123.0)
    a = cluster(p4, mask, algorithm="kt")
    b = cluster(garbage, mask, algorithm="kt")
    np.testing.assert_array_equal(a.children, b.children)
    np.testing.assert_allclose(a.p4, b.p4)


def test_invalid_arguments():
    p4, mask = ca_tree_arrays()
    with pytest.raises(ValueError, match="did you mean 'antikt'"):
        cluster(p4, mask, algorithm="anti-kt")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="recombination"):
        cluster(p4, mask, algorithm="ca", recombination="pt")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="R must be > 0"):
        cluster(p4, mask, algorithm="ca", R=0.0)
    with pytest.raises(ValueError, match=r"\(B, N, 4\)"):
        cluster(p4[0], mask, algorithm="ca")
    with pytest.raises(ValueError, match="mask must have shape"):
        cluster(p4, mask[:, :2], algorithm="ca")
    with pytest.raises(ValueError, match="axis must have shape"):
        cone(p4, mask, np.zeros(2))


# ---------------------------------------------------------------------------------------------- cones and anti-kt


def test_cone_wraps_phi():
    p4, mask = as_arrays([massless(100, 0.0, 3.1), massless(50, 0.0, -3.1), massless(20, 0.0, 0.0)])
    inside = cone(p4, mask, np.array([[0.0, np.pi]]), R=0.4)
    assert inside.tolist() == [[True, True, False]]


def test_anti_kt_jet_of_a_hard_particle_is_a_cone():
    # one hard particle in a spray of soft ones: the anti-kt jet is exactly the cone of radius R about it
    rng = np.random.default_rng(7)
    events = []
    for _ in range(30):
        y0, phi0 = rng.uniform(-1.0, 1.0), rng.uniform(-np.pi, np.pi)
        soft_y = y0 + rng.uniform(-1.5, 1.5, 60)
        soft_phi = phi0 + rng.uniform(-1.5, 1.5, 60)
        dr = np.hypot(soft_y - y0, np.angle(np.exp(1j * (soft_phi - phi0))))
        keep = np.abs(dr - 0.8) > 0.02
        soft = [
            massless(pt, y, phi)
            for pt, y, phi in zip(rng.uniform(0.1, 1.0, 60)[keep], soft_y[keep], soft_phi[keep], strict=True)
        ]
        events.append([massless(500.0, y0, phi0), *soft])
    p4, mask = as_arrays(*events)
    axis = np.stack([[np.arcsinh(p[0, 2] / np.hypot(p[0, 0], p[0, 1])), np.arctan2(p[0, 1], p[0, 0])] for p in p4])
    np.testing.assert_array_equal(leading_jet(p4, mask, R=0.8), cone(p4, mask, axis, R=0.8))


def test_anti_kt_vs_cone_on_toy_jets_without_ue():
    toy = pytest.importorskip("itj.toy")
    if not hasattr(toy, "generate"):
        pytest.skip("itj.toy.generate is not there yet")
    data = toy.generate(60, config=toy.preset("detector", ue=False), seed=5)
    p4, mask = data.padded_p4()
    total = p4.sum(axis=1)
    pt = np.hypot(total[:, 0], total[:, 1])
    axis = np.stack(
        [0.5 * np.log((total[:, 3] + total[:, 2]) / (total[:, 3] - total[:, 2])), np.arctan2(total[:, 1], total[:, 0])],
        axis=1,
    )
    inside = cone(p4, mask, axis, R=0.8)
    lead = leading_jet(p4, mask, R=0.8)
    # the constituents of an anti-kt jet sit (almost all) inside the cone about its axis and are found again
    assert inside.sum() >= 0.99 * mask.sum()
    assert (lead == mask).mean() > 0.99
    pt_in = np.where(inside[..., None], p4, 0.0).sum(axis=1)
    assert np.median(np.hypot(pt_in[:, 0], pt_in[:, 1]) / pt) > 0.999


# ---------------------------------------------------------------------------------------------------- declustering


def test_decluster_primary_of_the_synthetic_jet():
    p4, mask = ca_tree_arrays()
    chain = decluster_primary(cluster(p4, mask, algorithm="ca"))
    assert chain["delta"].counts().tolist() == [2]
    assert chain["harder"][0].tolist() == [4, 0]
    assert chain["softer"][0].tolist() == [5, 2]
    pt_cd = np.hypot(*(p4[0, 1, :2] + p4[0, 3, :2]))  # E-scheme: the vector sum of c and d
    np.testing.assert_allclose(chain["z"][0], [pt_cd / (400.0 + pt_cd), 100.0 / 400.0], rtol=1e-6)
    np.testing.assert_allclose(chain["delta"][0][1], 0.1, rtol=1e-6)
    np.testing.assert_allclose(chain["kt"][0][1], 10.0, rtol=1e-6)
    np.testing.assert_allclose(chain["psi"][0][1], np.pi / 2, rtol=1e-6)  # b sits at +y from a
    m_root = np.sqrt(max(p4[0].sum(0)[3] ** 2 - (p4[0].sum(0)[:3] ** 2).sum(), 0.0))
    np.testing.assert_allclose(chain["m"][0][0], m_root, rtol=1e-9)
    assert chain["delta"][0][0] > chain["delta"][0][1]
    for key in ("delta", "kt", "z", "psi", "m"):
        assert chain[key].dtype == np.float64
    assert chain["harder"].dtype == np.int64


def test_decluster_primary_of_empty_and_single_jets():
    p4, mask = as_arrays([massless(100, 0.0, 0.0)], [massless(50, 0.1, 0.0)])
    mask[1] = False
    chain = decluster_primary(cluster(p4, mask, algorithm="ca"))
    assert chain["delta"].counts().tolist() == [0, 0]


# ---------------------------------------------------------------------------------------------------------- fastjet


def fastjet_sequence(fj, p4: np.ndarray, mask: np.ndarray, algorithm: str, R: float, recombination: str):
    """Return the fastjet ClusterSequence of one jet and the slot of every input."""
    algorithms = {"antikt": fj.antikt_algorithm, "ca": fj.cambridge_algorithm, "kt": fj.kt_algorithm}
    scheme = fj.WTA_pt_scheme if recombination == "WTA" else fj.E_scheme
    slots = np.flatnonzero(mask)
    particles = []
    for i in slots:
        pj = fj.PseudoJet(*map(float, p4[i]))
        pj.set_user_index(int(i))
        particles.append(pj)
    return fj.ClusterSequence(particles, fj.JetDefinition(algorithms[algorithm], R, scheme)), slots


@pytest.mark.parametrize("algorithm", ["antikt", "ca", "kt"])
@pytest.mark.parametrize("recombination", ["E", "WTA"])
def test_fastjet_merge_sequence_and_four_vectors(algorithm, recombination):
    fj = pytest.importorskip("fastjet._swig")
    p4, mask = random_jets(60, 30, seed=11, masses=True, aligned=False)
    # R = 1000 (fastjet's max_allowable_R, as in N-subjettiness and soft drop): no beam merge before the last
    tree = cluster(p4, mask, algorithm=algorithm, R=1000.0, recombination=recombination)
    big = mask.shape[1]
    for b in range(len(p4)):
        cs, slots = fastjet_sequence(fj, p4[b], mask[b], algorithm, 1000.0, recombination)
        jets = cs.jets()
        n = len(slots)
        assert len(jets) == 2 * n - 1
        for k in range(n, 2 * n - 1):
            node = np.full(len(p4), -1)
            node[b] = big + k - n
            theirs = np.zeros(big, dtype=bool)
            theirs[[c.user_index() for c in jets[k].constituents()]] = True
            np.testing.assert_array_equal(tree.leaves_of(node)[b], theirs)
            v = np.array([jets[k].px(), jets[k].py(), jets[k].pz(), jets[k].E()])
            np.testing.assert_allclose(tree.p4[b, big + k - n], v, rtol=1e-10, atol=1e-10 * v[3])
        if algorithm != "antikt" and n > 1:
            dmerge = [cs.exclusive_dmerge(k) for k in range(n - 1, 0, -1)]
            np.testing.assert_allclose(tree.dij[b, big : big + n - 1], dmerge, rtol=1e-9)


@pytest.mark.parametrize("algorithm", ["antikt", "ca", "kt"])
def test_fastjet_leading_inclusive_jet(algorithm):
    fj = pytest.importorskip("fastjet._swig")
    p4, mask = random_jets(60, 60, seed=12, spread=0.8, aligned=False)
    lead = leading_jet(p4, mask, algorithm=algorithm, R=0.4)
    for b in range(len(p4)):
        cs, _ = fastjet_sequence(fj, p4[b], mask[b], algorithm, 0.4, "E")
        hardest = max(cs.inclusive_jets(), key=lambda j: j.pt())
        theirs = np.zeros(mask.shape[1], dtype=bool)
        theirs[[c.user_index() for c in hardest.constituents()]] = True
        np.testing.assert_array_equal(lead[b], theirs)


# ------------------------------------------------------------------------------------------------------------ speed


@pytest.mark.slow
def test_clustering_speed(capsys):
    rng = np.random.default_rng(0)
    n_jets, n = 4000, 30
    pt = rng.exponential(20.0, (n_jets, n)) + 0.1
    y, phi = rng.normal(0.0, 0.3, (2, n_jets, n))
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(y), pt * np.cosh(y)], axis=-1)
    mask = np.ones((n_jets, n), dtype=bool)
    rates = {}
    for algorithm in ("antikt", "ca", "kt"):
        start = time.perf_counter()
        cluster(p4, mask, algorithm=algorithm)
        rates[algorithm] = n_jets / (time.perf_counter() - start)
    with capsys.disabled():
        print("\nclustering at N = 30, jets/s per pass:", {k: round(v) for k, v in rates.items()})
    assert min(rates.values()) > 1000  # target 2500 on one core of the reference machine (measured about 10 000)
