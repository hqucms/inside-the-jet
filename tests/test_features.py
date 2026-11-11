from __future__ import annotations

import numpy as np
import pytest
import torch

from itj.data import JETCLASS_LABELS, JETS3, JetData, Ragged
from itj.features import (
    CONFIG_NAMES,
    FEATURE_SETS,
    JET_OBSERVERS,
    LABEL_SETS,
    MAX_LENGTH,
    FeatureVar,
    JetBatch,
    JetDataset,
    config_path,
    inputs,
    jet_features,
    jetclass_selection,
    labels,
    make_batch,
    make_loader,
    particle_variables,
    split,
    split_bounds,
)
from synthetic import massless, synthetic_jets


def random_jets(n: int = 30, *, seed: int = 0, max_count: int = 40, labels=None) -> JetData:
    rng = np.random.default_rng(seed)
    particles = []
    for _ in range(n):
        k = int(rng.integers(1, max_count))
        pt = rng.exponential(30.0, k) + 0.1
        eta = rng.normal(0.3, 0.25, k)
        phi = rng.normal(0.0, 0.25, k)
        particles.append(np.stack([massless(a, b, c) for a, b, c in zip(pt, eta, phi, strict=True)]))
    if labels is None:
        labels = [("QCD", "Wqq", "Tbqq")[i % 3] for i in range(n)]
    return synthetic_jets(particles, labels)


def with_impact_parameters(data: JetData) -> JetData:
    px = data["part_px"]
    assert isinstance(px, Ragged)
    rng = np.random.default_rng(5)
    noise = px.map(lambda v: rng.normal(size=len(v)).astype(np.float32))
    return data.with_columns(part_d0val=noise, part_dzval=noise.map(lambda v: 3 * v), part_d0err=noise.map(np.abs))


def test_feature_sets():
    assert [len(FEATURE_SETS[s]) for s in ("kin", "kinpid", "full")] == [7, 13, 17]
    kin = [v.name for v in FEATURE_SETS["kin"]]
    assert kin == [
        "part_pt_log",
        "part_e_log",
        "part_logptrel",
        "part_logerel",
        "part_deltaR",
        "part_deta",
        "part_dphi",
    ]
    full = [v.name for v in FEATURE_SETS["full"]]
    assert full.index("part_d0") == full.index("part_isMuon") + 1
    assert full[-2:] == ["part_deta", "part_dphi"]
    assert FEATURE_SETS["full"][12] == FeatureVar("part_d0err", 0.0, 1.0, 0.0, 1.0)
    assert FEATURE_SETS["kin"][4] == FeatureVar("part_deltaR", 0.2, 4.0)
    assert LABEL_SETS["qg"] == "truth_init_pid"
    assert MAX_LENGTH == 128
    assert len(JET_OBSERVERS) == 10


def test_particle_variables_are_float32_and_weaver_formulas():
    data = with_impact_parameters(random_jets(5))
    pv = particle_variables(data)
    assert set(pv) == {
        "part_mask",
        "part_pt",
        "part_pt_log",
        "part_e_log",
        "part_logptrel",
        "part_logerel",
        "part_deltaR",
        "part_d0",
        "part_dz",
    }
    assert all(r.values.dtype == np.float32 for r in pv.values())
    px = data["part_px"]
    assert isinstance(px, Ragged)
    assert all(r.offsets is px.offsets for r in pv.values())
    pt = np.hypot(px.values, data["part_py"].values)  # type: ignore[union-attr]
    np.testing.assert_array_equal(pv["part_pt"].values, pt)
    jet_pt = np.repeat(np.asarray(data["jet_pt"], np.float32), data.counts)
    np.testing.assert_array_equal(pv["part_logptrel"].values, np.log(pt / jet_pt))
    assert "part_d0" not in particle_variables(random_jets(2).drop("part_d0val", "part_dzval"))


@pytest.mark.parametrize("feature_set", ["kin", "kinpid", "full"])
def test_inputs_shapes_and_padding(feature_set):
    data = with_impact_parameters(random_jets(20, max_count=60))
    x = inputs(data, feature_set)
    f = len(FEATURE_SETS[feature_set])
    assert x["pf_points"].shape == (20, 2, 128)
    assert x["pf_features"].shape == (20, f, 128)
    assert x["pf_vectors"].shape == (20, 4, 128)
    assert x["pf_mask"].shape == (20, 1, 128)
    assert all(a.dtype == np.float32 for a in x.values())
    counts = data.counts
    np.testing.assert_array_equal(x["pf_mask"][:, 0].sum(axis=1), np.minimum(counts, 128))
    # wrap padding: each jet repeats its own particles
    for i in range(3):
        n = counts[i]
        np.testing.assert_array_equal(x["pf_vectors"][i, :, n : 2 * n], x["pf_vectors"][i, :, :n][:, : 128 - n])
    np.testing.assert_array_equal(x["pf_vectors"][:, 0], data["part_px"].padded(128, mode="wrap"))  # type: ignore[union-attr]
    short = inputs(data, feature_set, length=None)
    assert short["pf_features"].shape[2] == min(int(counts.max()), 128)


def test_inputs_transform_clip_and_nan():
    # one particle with E = 0 (log -> -inf -> clipped) and one NaN deta (-> 0, raw variable)
    p4 = np.array([[100.0, 0.0, 0.0, 100.0], [1.0, 0.0, 0.0, 1.0]])
    data = synthetic_jets([p4])
    energy = data["part_energy"]
    deta = data["part_deta"]
    assert isinstance(energy, Ragged)
    assert isinstance(deta, Ragged)
    data = data.with_columns(
        part_energy=Ragged(np.array([100.0, 0.0], np.float32), energy.offsets),
        part_deta=Ragged(np.array([np.nan, 0.5], np.float32), deta.offsets),
    )
    x = inputs(data, "kin", length=2)["pf_features"][0]
    names = [v.name for v in FEATURE_SETS["kin"]]
    e_log = x[names.index("part_e_log")]
    assert e_log[0] == np.float32((np.log(np.float32(100.0)) - np.float32(2.0)) * np.float32(0.7))
    assert e_log[1] == -5.0  # log(0) = -inf, clipped
    assert x[names.index("part_deta"), 0] == 0.0  # NaN -> 0
    assert x[names.index("part_deta"), 1] == np.float32(0.5)


def test_inputs_unknown_feature_set_and_missing_columns():
    data = random_jets(3)
    with pytest.raises(ValueError, match="kinpid"):
        inputs(data, "kinpd")
    with pytest.raises(ValueError, match="part_d0val"):
        inputs(data.drop("part_d0val"), "full")


def test_jet_features():
    data = random_jets(4).with_columns(
        jet_tau1=np.array([0.2, 0.0, 0.5, 1.0], np.float32), jet_tau2=np.array([0.1, 0.0, 0.25, 0.0], np.float32)
    )
    x, names = jet_features(data)
    assert x.dtype == np.float32
    assert x.shape == (4, 12)
    assert names == [*JET_OBSERVERS, "tau21", "tau32"]
    np.testing.assert_allclose(x[:, names.index("tau21")], [0.5, 0.0, 0.5, 0.0])
    x, names = jet_features(data, ["jet_pt", "tau21"], derived=())
    assert names == ["jet_pt", "tau21"]
    assert x.shape == (4, 2)
    with pytest.raises(ValueError, match="per-jet"):
        jet_features(data, ["part_px"])


def test_labels():
    data = random_jets(6)
    assert labels(data).tolist() == [0, 1, 2, 0, 1, 2]
    assert labels(data, "jetclass10").tolist() == [0, 7, 8, 0, 7, 8]
    with pytest.raises(ValueError, match="truth_init_pid"):
        labels(data, "qg")
    qg = data.with_columns(truth_init_pid=np.array([1, 21, 21, 1, 1, 21], np.int32))
    assert labels(qg, "qg").tolist() == [0, 1, 1, 0, 0, 1]
    with pytest.raises(ValueError, match="quark and gluon"):
        labels(data.with_columns(truth_init_pid=np.array([1, 24, 6, 1, 1, 21], np.int32)), "qg")
    with pytest.raises(ValueError, match="jets3"):
        labels(data, "jet3")


def test_split_bounds():
    assert split_bounds(1000) == (200, 300)
    assert split_bounds(10) == (2, 3)
    for n in range(0, 5000, 7):
        assert split_bounds(n) == (int(0.2 * n), int(0.3 * n))
    assert split_bounds(100, test=0.5, val=0.5) == (50, 100)
    with pytest.raises(ValueError, match="test"):
        split_bounds(10, test=0.8, val=0.3)


def test_split_idx_rule_per_file():
    data = random_jets(30).with_columns(
        file_index=np.repeat(np.array([0, 1], np.int16), [10, 20]), jet_eta=np.arange(30, dtype=np.float32)
    )
    train, val, test = split(data)
    assert test["jet_eta"].tolist() == [0, 1, *range(10, 14)]  # type: ignore[union-attr]
    assert val["jet_eta"].tolist() == [2, 14, 15]  # type: ignore[union-attr]
    assert train["jet_eta"].tolist() == [*range(3, 10), *range(16, 30)]  # type: ignore[union-attr]
    train, val, test = split(data.drop("file_index"))
    assert test["jet_eta"].tolist() == list(range(6))  # type: ignore[union-attr]
    assert val["jet_eta"].tolist() == list(range(6, 9))  # type: ignore[union-attr]
    assert len(train) == 21


def test_jetclass_selection():
    data = random_jets(4).with_columns(jet_pt=np.array([450.0, 500.0, 750.0, 1000.0], np.float32))
    assert jetclass_selection(data).tolist() == [False, False, True, False]
    assert jetclass_selection(data, pt_min=400.0, pt_max=2000.0).all()


def test_config_path_names():
    with pytest.raises(ValueError, match="jets3_kin"):
        config_path("jets3_kn")
    for name in CONFIG_NAMES:
        path = config_path(name)
        assert path.is_file()
        assert path.name == f"{name}.yaml"


def _yaml_var(entry) -> FeatureVar:
    if isinstance(entry, str):
        return FeatureVar(entry)
    name, *rest = entry
    if not rest or rest[0] is None:
        return FeatureVar(name)
    center, scale, *clip = rest
    return FeatureVar(name, float(center), float(scale), *(float(c) for c in clip))


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_feature_sets_equal_the_yaml_configs(name):
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(config_path(name).read_text())
    feature_set = name.split("_")[1]
    inputs_ = config["inputs"]
    assert [_yaml_var(v) for v in inputs_["pf_features"]["vars"]] == list(FEATURE_SETS[feature_set])
    assert [_yaml_var(v) for v in inputs_["pf_points"]["vars"]] == [FeatureVar("part_deta"), FeatureVar("part_dphi")]
    vectors = [FeatureVar(n) for n in ("part_px", "part_py", "part_pz", "part_energy")]
    assert [_yaml_var(v) for v in inputs_["pf_vectors"]["vars"]] == vectors
    assert [_yaml_var(v) for v in inputs_["pf_mask"]["vars"]] == [FeatureVar("part_mask")]
    for key in ("pf_points", "pf_features", "pf_vectors"):
        assert inputs_[key]["length"] == MAX_LENGTH
        assert inputs_[key]["pad_mode"] == "wrap"
    assert inputs_["pf_mask"]["pad_mode"] == "constant"
    assert config["preprocess"]["method"] == "manual"
    assert tuple(config["observers"]) == JET_OBSERVERS
    assert config["labels"]["type"] == "simple"
    expected = JETS3 if name.startswith("jets3") else JETCLASS_LABELS
    assert tuple(config["labels"]["value"]) == expected
    assert not config.get("selection")  # never an index selection (weaver applies it after concatenating files)
    new = config["new_variables"]
    assert new["part_logptrel"] == "np.log(part_pt/jet_pt)"
    assert ("part_d0" in new) == (feature_set == "full")
    # every variable that is not a branch is a new variable computed by particle_variables
    data = random_jets(3)
    if feature_set != "full":
        data = data.drop("part_d0val", "part_dzval")
    assert set(new) == set(particle_variables(data))


def test_jets3_yaml_differ_only_in_the_features():
    yaml = pytest.importorskip("yaml")
    kin = yaml.safe_load(config_path("jets3_kin").read_text())
    for other in ("jets3_kinpid", "jets3_full"):
        config = yaml.safe_load(config_path(other).read_text())
        for key in ("preprocess", "labels", "observers", "weights"):
            assert config[key] == kin[key], (other, key)
        for key in ("pf_points", "pf_vectors", "pf_mask"):
            assert config["inputs"][key] == kin["inputs"][key]


# ------------------------------------------------------------------------------------------------ torch batches


def test_jet_dataset_batch_padding():
    data = random_jets(12, max_count=200)
    ds = JetDataset(data, "kin")
    assert len(ds) == 12
    assert ds.feature_names == tuple(v.name for v in FEATURE_SETS["kin"])
    idx = np.array([3, 0, 7])
    batch = ds.batch(idx)
    assert isinstance(batch, JetBatch)
    counts = data.counts[idx]
    p = min(int(counts.max()), 128)
    assert batch.points.shape == (3, 2, p)
    assert batch.features.shape == (3, 7, p)
    assert batch.vectors.shape == (3, 4, p)
    assert batch.mask.shape == (3, 1, p)
    assert batch.features.dtype == torch.float32
    assert batch.label is not None
    assert batch.label.tolist() == [0, 0, 1]
    assert batch.index.tolist() == [3, 0, 7]
    assert batch.index.dtype == torch.int64
    assert batch.weight is None
    assert batch.jet is None
    assert batch.particle_mask.dtype == torch.bool
    assert batch.particle_mask.sum(dim=1).tolist() == np.minimum(counts, p).tolist()
    assert len(batch) == 3
    # the padded batch equals inputs() of the same jets at the same length
    ref = inputs(data.select(idx), "kin", length=p)
    np.testing.assert_array_equal(batch.features.numpy(), ref["pf_features"])
    np.testing.assert_array_equal(batch.vectors.numpy(), ref["pf_vectors"])
    np.testing.assert_array_equal(batch.mask.numpy(), ref["pf_mask"])
    np.testing.assert_array_equal(batch.points.numpy(), ref["pf_points"])
    assert ds.batch(idx, length=MAX_LENGTH).features.shape[2] == 128


def test_jet_dataset_options():
    data = random_jets(9)
    weights = np.linspace(0.5, 1.5, 9)
    ds = JetDataset(data, None, None, weights=weights, jet_feature_names=["jet_pt", "jet_sdmass", "tau21"], length=8)
    batch = ds.batch(np.arange(9))
    assert batch.features.shape[:2] == (9, 0)
    assert batch.points.shape[2] <= 8
    assert batch.label is None
    assert batch.weight is not None
    assert batch.weight.dtype == torch.float32
    assert batch.jet is not None
    assert batch.jet.shape == (9, 3)
    assert ds.jet_feature_names == ("jet_pt", "jet_sdmass", "tau21")
    assert batch.feature_set is None
    with pytest.raises(ValueError, match="weights"):
        JetDataset(data, weights=np.ones(3))
    jet_only = JetDataset(JetData({"jet_pt": np.ones(4, np.float32)}), None, None, jet_feature_names=["jet_pt"])
    assert jet_only.batch(np.arange(4)).jet.shape == (4, 1)  # type: ignore[union-attr]


def test_make_batch():
    data = random_jets(5)
    batch = make_batch(data, "kinpid", label_set="jetclass10")
    assert batch.features.shape[1] == 13
    assert batch.label is not None
    assert batch.label.tolist() == [0, 7, 8, 0, 7]
    assert make_batch(data, length=64).features.shape[2] == 64
    moved = batch.to("cpu")
    assert moved.feature_set == "kinpid"
    torch.testing.assert_close(moved.features, batch.features)


def test_make_loader():
    data = random_jets(23)
    ds = JetDataset(data)
    loader = make_loader(ds, batch_size=5, seed=1)
    epoch1 = [b.index.tolist() for b in loader]
    epoch2 = [b.index.tolist() for b in loader]
    assert len(loader) == 5  # type: ignore[arg-type]
    assert [len(b) for b in epoch1] == [5, 5, 5, 5, 3]
    assert sorted(i for b in epoch1 for i in b) == list(range(23))
    assert epoch1 != epoch2  # a new order every epoch
    assert [b.index.tolist() for b in make_loader(ds, batch_size=5, seed=1)] == epoch1  # reproducible
    fixed = make_loader(ds, batch_size=10, shuffle=False, drop_last=True)
    assert [b.index.tolist() for b in fixed] == [list(range(10)), list(range(10, 20))]


# ------------------------------------------------------------------------------------------- the committed sample


def test_inputs_on_the_jetclass_sample(jetclass_sample):
    data = jetclass_sample
    counts = data.counts
    for feature_set, n_features in (("kin", 7), ("kinpid", 13), ("full", 17)):
        x = inputs(data, feature_set)
        assert x["pf_points"].shape == (150, 2, 128)
        assert x["pf_features"].shape == (150, n_features, 128)
        assert x["pf_vectors"].shape == (150, 4, 128)
        assert x["pf_mask"].shape == (150, 1, 128)
        assert all(v.dtype == np.float32 for v in x.values())
        assert x["pf_mask"].sum(axis=(1, 2)).tolist() == np.minimum(counts, 128).tolist()
        assert np.all(np.isfinite(x["pf_features"]))
        standardised = [k for k, v in enumerate(FEATURE_SETS[feature_set]) if v.center is not None]
        assert np.all(np.abs(x["pf_features"][:, standardised]) <= 5.0)
    # jet 0 by hand: the weaver formulas in float32
    x = inputs(data, "kin", length=None)
    assert x["pf_features"].shape[2] == min(int(counts.max()), 128)
    jet = data.jet(0)
    p = jet.particles
    pt = np.hypot(p["part_px"], p["part_py"])
    n = len(pt)
    expected = np.stack(
        [
            np.clip((np.log(pt) - np.float32(1.7)) * np.float32(0.7), -5, 5),
            np.clip((np.log(p["part_energy"]) - np.float32(2.0)) * np.float32(0.7), -5, 5),
            np.clip((np.log(pt / np.float32(jet.jet["jet_pt"])) + np.float32(4.7)) * np.float32(0.7), -5, 5),
            np.clip(
                (np.log(p["part_energy"] / np.float32(jet.jet["jet_energy"])) + np.float32(4.7)) * np.float32(0.7),
                -5,
                5,
            ),
            np.clip((np.hypot(p["part_deta"], p["part_dphi"]) - np.float32(0.2)) * np.float32(4.0), -5, 5),
            p["part_deta"],
            p["part_dphi"],
        ]
    )
    np.testing.assert_allclose(x["pf_features"][0, :, :n], expected, rtol=1e-6, atol=1e-6)
    # wrap padding repeats the jet's own particles
    length = x["pf_features"].shape[2]
    if n < length:
        np.testing.assert_array_equal(x["pf_vectors"][0, :, n:], x["pf_vectors"][0, :, : length - n][:, : length - n])


def test_labels_and_split_on_the_jetclass_sample(jetclass_sample):
    assert labels(jetclass_sample).tolist() == [0] * 50 + [1] * 50 + [2] * 50
    assert labels(jetclass_sample, "jetclass10").tolist() == [0] * 50 + [7] * 50 + [8] * 50
    train, val, test = split(jetclass_sample)
    assert (len(train), len(val), len(test)) == (105, 15, 30)
    assert jetclass_selection(jetclass_sample).all()  # JetClass has the reco cut 500 < pT < 1000 GeV


# ------------------------------------------------------------------------------------------- weaver 0.5.4 arrays


def _weaver_arrays(path, config_name: str) -> dict[str, np.ndarray]:
    """The arrays weaver 0.5.4 builds from a file with one of the packaged data configs."""
    from weaver.utils.data.config import DataConfig
    from weaver.utils.data.fileio import _read_files
    from weaver.utils.data.preprocess import _build_new_variables
    from weaver.utils.dataset import _finalize_inputs

    config = DataConfig.load(str(config_path(config_name)), load_observers=False)
    table = _read_files([str(path)], config.train_load_branches, load_range=(0, 1), treename=None)
    table = _build_new_variables(table, config.var_funcs)
    out = _finalize_inputs(table, config)
    return {key: out["_" + key] for key in ("pf_points", "pf_features", "pf_vectors", "pf_mask")}


@pytest.fixture(params=["jetclass_sample", "itj_file", "toy_file"])
def weaver_file(request, tmp_path):
    from itj.io import save

    if request.param == "jetclass_sample":
        from conftest import JETCLASS_SAMPLE

        return JETCLASS_SAMPLE
    if request.param == "itj_file":  # a file written by itj.io.save, with extra truth columns
        data = random_jets(40, max_count=150)
        data = with_impact_parameters(data).with_columns(
            part_dzerr=data["part_px"].map(np.abs),  # type: ignore[union-attr]
            truth_process=np.zeros(40, np.int8),
            truth_part_is_ue=data["part_px"].map(lambda v: np.zeros(len(v), np.int8)),  # type: ignore[union-attr]
        )
        return save(
            JetData(data.columns, config={"note": "synthetic"}, provenance={"seed": 0}), tmp_path / "itj.parquet"
        )
    import itj.toy

    if not hasattr(itj.toy, "generate"):
        pytest.skip("itj.toy.generate is not available yet")
    return request.getfixturevalue("tmp_toy_file")


@pytest.mark.weaver
@pytest.mark.parametrize("feature_set", ["kin", "kinpid", "full"])
def test_inputs_equal_weaver(weaver_file, feature_set):
    pytest.importorskip("weaver")
    from itj.io import load

    for config_name in (f"jets3_{feature_set}", f"JetClass_{feature_set}"):
        if config_name.startswith("JetClass") and "label_Hbb" not in load(weaver_file, columns=None).names:
            continue
        expected = _weaver_arrays(weaver_file, config_name)
        ours = inputs(load(weaver_file), feature_set)
        for key, value in expected.items():
            assert ours[key].shape == value.shape, key
            assert ours[key].dtype == value.dtype, key
            np.testing.assert_array_equal(ours[key], value, err_msg=f"{config_name} {key}")
