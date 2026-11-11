"""generate: exact identities on every preset, reproducibility, workers, prefix property, pairing, provenance."""

import numpy as np
import pytest

from helpers import check_identities
from itj.data import JETCLASS_BRANCHES, JetData, Ragged
from itj.toy import CHUNK_SIZE, PRESETS, generate, preset, show_ladder
from itj.toy.generate import STAGES, nearest_preset


def _equal(a: JetData, b: JetData, names=None) -> None:
    names = a.names if names is None else names
    for name in names:
        x, y = a[name], b[name]
        if isinstance(x, Ragged):
            np.testing.assert_array_equal(x.offsets, y.offsets, err_msg=name)
            x, y = x.values, y.values
        np.testing.assert_array_equal(x, y, err_msg=name)


def _all_true(checks):
    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, failed


def test_identities_on_every_preset(ladder_small):
    for name, data in ladder_small.items():
        checks = check_identities(data)
        _all_true(checks)
        assert "jet_nparticles" in checks
        assert "parton_sum" in checks
        assert ("containment" in checks) == (name == "partons")


def test_identities_on_fixtures(toy_small, qg_small):
    _all_true(check_identities(toy_small))
    _all_true(check_identities(qg_small))
    assert len(toy_small) == 300
    assert toy_small.has_truth


@pytest.mark.parametrize(
    "overrides",
    [
        {"ue": False, "coherence": False},
        {"jet_def": "cone"},
        {"cell_position": "centre", "alphas_mode": "fixed", "alphas_value": 0.12},
        {"w_helicity": False, "dead_cone": True, "g_to_qq": False},
        {"hadron_reshuffle": True},
        {"hadron_reshuffle": True, "singlet_frame": True, "ue": False},
    ],
)
def test_identities_on_modified_configurations(overrides):
    data = generate(150, config=preset("detector", **overrides), processes=("q", "g", "wqq", "tbqq"), seed=2)
    checks = check_identities(data)
    _all_true(checks)
    assert ("hadron_sum" in checks) == bool(overrides.get("hadron_reshuffle"))


def test_hadron_reshuffle_identity_and_pairing():
    config = preset("detector", hadron_reshuffle=True, singlet_frame=True)
    data = generate(1200, config=config, processes=("qcd", "wqq", "tbqq"), seed=4, workers=2)
    record = data.provenance["hadron_reshuffle"]
    names = data.process_names
    assert record["steps"] == np.sum(names == "wqq") + 2 * np.sum(names == "tbqq")  # W; W in the top and the top
    assert record["failed"] <= 2  # about 1 in 10 000 W and top systems is too heavy in the singlet frame
    assert 0 <= record["failed_w_in_top"] <= record["failed"]
    assert check_identities(data)["hadron_sum"]
    broken = JetData(data.columns, config=data.config, provenance={**data.provenance, "hadron_reshuffle": {
        "steps": 1, "failed": 0, "max_residual": 1e-3}})  # fmt: skip
    assert not check_identities(broken)["hadron_sum"]
    # the output does not depend on the workers; the hard process and the shower stay paired with the default
    _equal(data, generate(1200, config=config, processes=("qcd", "wqq", "tbqq"), seed=4, workers=1))
    default = generate(1200, processes=("qcd", "wqq", "tbqq"), seed=4)
    _equal(default, data, ["truth_seed_pt", "truth_mass", "truth_parton_px", "truth_emis_theta"])
    qcd = np.flatnonzero(names == "qcd")
    _equal(default.select(qcd), data.select(qcd))  # QCD jets are untouched
    assert not np.array_equal(default["part_px"].values, data["part_px"].values)


def test_hadron_sum_with_a_per_process_override():
    """The identity applies whenever the reshuffle ran, also for one process only (the global switch is off)."""
    overrides = {"wqq": {"hadron_reshuffle": True}}
    data = generate(300, processes=("qcd", "wqq", "tbqq"), process_overrides=overrides, seed=3)
    record = data.provenance["hadron_reshuffle"]
    assert record["steps"] == np.sum(data.process_names == "wqq")
    assert record["failed_w_in_top"] == 0
    assert check_identities(data)["hadron_sum"]


def test_columns_and_labels(toy_small):
    for name in JETCLASS_BRANCHES:
        assert name in toy_small
    names = toy_small.process_names
    assert set(names) == {"qcd", "wqq", "tbqq"}
    np.testing.assert_array_equal(toy_small["label_Wqq"] == 1, names == "wqq")
    np.testing.assert_array_equal(toy_small["label_Tbqq"] == 1, names == "tbqq")
    np.testing.assert_array_equal(toy_small["label_QCD"] == 1, names == "qcd")
    np.testing.assert_array_equal(toy_small["truth_event_id"], np.arange(300))
    pid = toy_small["aux_genpart_pid"]
    assert set(np.abs(pid[names == "wqq"])) == {24}
    assert set(np.abs(pid[names == "tbqq"])) == {6}
    np.testing.assert_array_equal(pid[names == "qcd"], 0)
    np.testing.assert_array_equal(toy_small["aux_truth_match"][names == "tbqq"], 11)
    assert toy_small.config == preset("detector").to_dict()


def test_processes_are_balanced_per_chunk():
    data = generate(1501, config=preset("partons"), processes=("q", "g"), seed=4)
    names = data.process_names
    assert (names[:1000] == "q").sum() == 500
    assert (names[1000:] == "q").sum() == 251  # 501 jets: the remainder goes to the first process
    assert len(data) == 1501


def test_reproducible_and_seed_dependent():
    a = generate(200, seed=5)
    _equal(a, generate(200, seed=5))
    b = generate(200, seed=6)
    assert not np.array_equal(a["jet_pt"], b["jet_pt"])


def test_workers_do_not_change_the_output():
    config = preset("shower")
    a = generate(2100, config=config, seed=7, workers=1)
    b = generate(2100, config=config, seed=7, workers=2)
    _equal(a, b)
    assert b.provenance["workers"] == 2


def test_whole_chunk_prefix_property():
    long = generate(CHUNK_SIZE + 300, config=preset("shower"), seed=8)
    short = generate(CHUNK_SIZE, config=preset("shower"), seed=8)
    _equal(long.head(CHUNK_SIZE), short)


def test_pairing_across_presets(ladder_small):
    hard = [
        "truth_process",
        "truth_seed_pt",
        "truth_seed_eta",
        "truth_seed_phi",
        "truth_mass",
        "truth_cos_theta_star",
        "truth_prong_energy",
        "truth_prong_pid",
        "aux_genpart_pt",
    ]
    shower = ["truth_parton_px", "truth_parton_energy", "truth_emis_theta", "truth_emis_z", "truth_n_primary"]
    reference = ladder_small["detector"]
    for name, data in ladder_small.items():
        _equal(reference, data, hard)
        if name != "partons":
            _equal(reference, data, shower)
    assert len(ladder_small["partons"]["truth_emis_theta"].values) == 0


@pytest.mark.parametrize("extra", [{}, {"hadron_reshuffle": True, "singlet_frame": True}])
def test_jet_origin_tracks_identical_without_ue_and_with_cell_centres(extra):
    base = preset("detector", jet_def="cone", **extra)
    runs = [generate(300, config=base.replace(**o), seed=9) for o in ({}, {"ue": False}, {"cell_position": "centre"})]

    def tracks(data):
        charged = data["part_charge"].values != 0
        jet_origin = data["truth_part_is_ue"].values == 0
        keep = charged & jet_origin
        row = data["part_px"].row_index()[keep]
        return [row] + [data[c].values[keep] for c in ("part_px", "part_py", "part_pz", "part_d0val", "part_dzval")]

    reference = tracks(runs[0])
    for other in runs[1:]:
        for a, b in zip(reference, tracks(other), strict=True):
            np.testing.assert_array_equal(a, b)
    assert not np.array_equal(runs[0]["part_px"].values, runs[1]["part_px"].values)


def test_process_overrides_change_one_process_only():
    clean = generate(300, seed=10)
    changed = generate(300, seed=10, process_overrides={"wqq": {"pi0_fraction": 0.45}})
    names = clean.process_names
    np.testing.assert_array_equal(names, changed.process_names)
    for process in ("qcd", "tbqq"):
        rows = np.flatnonzero(names == process)
        _equal(clean.select(rows), changed.select(rows))
    w = np.flatnonzero(names == "wqq")
    assert not np.array_equal(clean.select(w)["part_px"].values, changed.select(w)["part_px"].values)
    _equal(clean.select(w), changed.select(w), ["truth_seed_pt", "truth_parton_energy"])  # the same partons
    assert changed.provenance["process_overrides"] == {"wqq": {"pi0_fraction": 0.45}}
    assert changed.config == clean.config


def test_provenance():
    data = generate(120, seed=11, first_event_id=500, processes=("wqq", "tbqq"))
    p = data.provenance
    for key in (
        "seed",
        "first_event_id",
        "chunk_size",
        "workers",
        "preset",
        "processes",
        "overrides",
        "process_overrides",
        "containment_efficiency",
        "package_version",
        "git_commit",
        "python_version",
        "numpy_version",
        "pyarrow_version",
        "platform",
        "created",
        "wall_time_s",
        "command_line",
        "stage_times_s",
        "hadron_kt_capped",
        "hadron_reshuffle",
    ):
        assert key in p, key
    assert set(p["stage_times_s"]) == set(STAGES)
    assert p["preset"] == "detector"
    assert p["overrides"] == {}
    assert p["processes"] == ["wqq", "tbqq"]
    assert 0.85 < p["containment_efficiency"]["wqq"] <= 1.0
    assert 0.0 <= p["hadron_kt_capped"] < 0.3
    assert p["hadron_reshuffle"] == {"steps": 0, "failed": 0, "failed_w_in_top": 0, "max_residual": 0.0}  # off
    np.testing.assert_array_equal(data["truth_event_id"], 500 + np.arange(120))
    modified = generate(30, config=preset("shower", kt_cut=2.0), seed=1)
    assert modified.provenance["preset"] == "shower"
    assert modified.provenance["overrides"] == {"kt_cut": 2.0}
    assert nearest_preset(preset("hadrons", ue=False)) == ("hadrons", {"ue": False})


def test_no_truth():
    data = generate(50, seed=12, truth=False)
    assert not data.has_truth
    assert len(data.names) == len(JETCLASS_BRANCHES)


def test_reco_pt_window():
    data = generate(60, config=preset("detector", jet_pt_min=600.0), seed=13)
    assert len(data) == 60
    assert np.all(data["jet_pt"] >= 600.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"processes": ("qcd", "wq")}, "Did you mean 'wqq'"),
        ({"processes": ()}, "no process"),
        ({"workers": 0}, "workers"),
        ({"seed": -1}, "seed"),
        ({"process_overrides": {"g": {"ue": False}}}, "not among the processes"),
    ],
)
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        generate(10, **kwargs)
    with pytest.raises(ValueError, match="n_jets"):
        generate(0)
    with pytest.raises(TypeError, match="unknown configuration parameter"):
        generate(10, process_overrides={"qcd": {"pi0_fractoin": 0.2}})


def test_show_ladder():
    jets = show_ladder("tbqq", seed=3, jet=2)
    assert list(jets) == list(PRESETS)
    for data in jets.values():
        assert len(data) == 1
        assert data.process_names[0] == "tbqq"
    pts = {float(data["truth_seed_pt"][0]) for data in jets.values()}
    assert len(pts) == 1
    assert int(jets["partons"].counts[0]) == 3
    assert int(jets["detector"].counts[0]) > int(jets["shower"].counts[0])
