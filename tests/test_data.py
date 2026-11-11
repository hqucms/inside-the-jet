from __future__ import annotations

import numpy as np
import pytest

from itj.data import (
    COLUMNS,
    GROUPS,
    JETCLASS_AUX_BRANCHES,
    JETCLASS_BRANCHES,
    JETCLASS_JET_BRANCHES,
    JETCLASS_LABELS,
    JETCLASS_PART_BRANCHES,
    JETS3,
    JetData,
    JetView,
    Ragged,
    column_group,
)
from synthetic import jetclass_like_file, synthetic_jets, three_prong_jet, two_particle_jet

# ------------------------------------------------------------------------------------------------------ helpers


def weaver_repeat_pad(values: np.ndarray, offsets: np.ndarray, maxlen: int) -> np.ndarray:
    """Plain-Python copy of weaver 0.5.4 ``_repeat_pad_jagged_kernel`` (utils/data/tools.py)."""
    out = np.zeros((len(offsets) - 1, maxlen), dtype=values.dtype)
    for i in range(len(offsets) - 1):
        start = offsets[i]
        length = offsets[i + 1] - start
        if length == 0:
            continue
        for j in range(maxlen):
            out[i, j] = values[start + j % length]
    return out


def random_ragged(rng: np.random.Generator, n: int = 40, max_count: int = 9) -> Ragged:
    counts = rng.integers(0, max_count, n)
    return Ragged.from_counts(rng.normal(size=int(counts.sum())).astype(np.float32), counts)


def small_data(n: int = 6, *, truth: bool = True) -> JetData:
    rng = np.random.default_rng(1)
    counts = rng.integers(1, 6, n)
    px = Ragged.from_counts(rng.normal(size=int(counts.sum())).astype(np.float32), counts)
    columns: dict[str, np.ndarray | Ragged] = {
        "part_px": px,
        "part_py": px.map(lambda v: v * 2),
        "part_pz": px.map(np.zeros_like),
        "part_energy": px.map(np.abs),
        "jet_pt": np.arange(n, dtype=np.float32),
    }
    for i, name in enumerate(JETCLASS_LABELS):
        columns[name] = (np.arange(n) % 3 == [0, 7, 8].index(i) if i in (0, 7, 8) else np.zeros(n)).astype(
            COLUMNS[name].dtype
        )
    if truth:
        columns["truth_process"] = (np.arange(n) % 3).astype(np.int8)
        prong_counts = (np.arange(n) % 3) + 1
        columns["truth_prong_px"] = Ragged.from_counts(np.ones(int(prong_counts.sum()), np.float32), prong_counts)
        columns["truth_part_prong"] = px.map(lambda v: np.zeros(len(v), np.int8))
    return JetData(columns, provenance={"preset": "detector"})


# ------------------------------------------------------------------------------------------------------ schema


def test_jetclass_branch_constants():
    assert len(JETCLASS_PART_BRANCHES) == 16
    assert len(JETCLASS_LABELS) == 10
    assert len(JETCLASS_JET_BRANCHES) == 10
    assert len(JETCLASS_AUX_BRANCHES) == 5
    assert JETCLASS_BRANCHES == JETCLASS_PART_BRANCHES + JETCLASS_LABELS + JETCLASS_JET_BRANCHES + JETCLASS_AUX_BRANCHES
    assert len(set(JETCLASS_BRANCHES)) == 41
    assert JETS3 == ("label_QCD", "label_Wqq", "label_Tbqq")
    assert [JETCLASS_LABELS.index(name) for name in JETS3] == [0, 7, 8]


def test_columns_cover_jetclass_first_in_file_order():
    assert tuple(COLUMNS)[:41] == JETCLASS_BRANCHES
    assert all(name.startswith("truth_") for name in tuple(COLUMNS)[41:])


def test_jetclass_dtypes():
    # the dtypes of the JetClass files (pyarrow schema of the val files): float labels, bool H labels, int32 others
    assert COLUMNS["label_QCD"].dtype == "float32"
    assert all(COLUMNS[f"label_{s}"].dtype == "bool" for s in ("Hbb", "Hcc", "Hgg", "H4q", "Hqql"))
    assert all(COLUMNS[f"label_{s}"].dtype == "int32" for s in ("Zqq", "Wqq", "Tbqq", "Tbl"))
    assert all(COLUMNS[name].dtype == "int32" for name in JETCLASS_PART_BRANCHES if name.startswith("part_is"))
    others = [n for n in JETCLASS_PART_BRANCHES + JETCLASS_JET_BRANCHES + JETCLASS_AUX_BRANCHES if "part_is" not in n]
    assert all(COLUMNS[name].dtype == "float32" for name in others)


def test_column_info_is_consistent():
    groups = {"jet", *GROUPS}
    for name, info in COLUMNS.items():
        assert info.group in groups, name
        assert np.dtype(info.dtype).name == info.dtype, name
        assert info.jagged == (info.group != "jet"), name
        assert info.description.endswith("."), name
        assert column_group(name) == info.group


def test_ragged_prefix_rule():
    """No per-jet column starts with a ragged prefix, and every column with such a prefix is jagged."""
    prefixes = tuple(p for ps in GROUPS.values() for p in ps)
    for name, info in COLUMNS.items():
        assert info.jagged == name.startswith(prefixes), name
        if info.jagged:
            assert any(name.startswith(p) for p in GROUPS[info.group]), name


def test_truth_columns_of_the_spec():
    expected_jet = {
        "truth_event_id": "int64",
        "truth_process": "int8",
        "truth_init_pid": "int32",
        "truth_nprongs": "int8",
        "truth_seed_pt": "float32",
        "truth_mass": "float32",
        "truth_cos_theta_star": "float32",
        "truth_dr_min": "float32",
        "truth_dr_max": "float32",
        "truth_n_emissions": "int16",
        "truth_n_primary": "int16",
        "truth_lnkt_max": "float32",
    }
    for name, dtype in expected_jet.items():
        assert COLUMNS[name].dtype == dtype
        assert COLUMNS[name].group == "jet"
    counts = {g: sum(1 for c in COLUMNS.values() if c.group == g) for g in GROUPS}
    assert counts == {"particle": 16 + 5, "prong": 7, "parton": 7, "emission": 12, "lund": 4}
    assert COLUMNS["truth_part_parton"].dtype == "int16"
    assert COLUMNS["truth_emis_channel"].dtype == "int8"


def test_column_group_guess():
    assert column_group("part_myfeature") == "particle"
    assert column_group("truth_part_x") == "particle"
    assert column_group("truth_parton_x") == "parton"
    assert column_group("score") == "jet"
    assert column_group("score", jagged=True) == "particle"


def test_toy_processes_match_the_generator():
    toy = pytest.importorskip("itj.toy")
    if not hasattr(toy, "PROCESSES"):
        pytest.skip("itj.toy.PROCESSES not there yet")
    from itj.data import _TOY_PROCESSES

    assert toy.PROCESSES == _TOY_PROCESSES


# ------------------------------------------------------------------------------------------------------ Ragged


def test_ragged_basics():
    r = Ragged.from_counts(np.arange(6.0), [2, 0, 3, 1])
    assert len(r) == 4
    assert r.offsets.dtype == np.int64
    assert r.counts().tolist() == [2, 0, 3, 1]
    assert r.row_index().tolist() == [0, 0, 2, 2, 2, 3]
    assert r[0].tolist() == [0.0, 1.0]
    assert r[1].size == 0
    assert r[-1].tolist() == [5.0]
    with pytest.raises(IndexError):
        r[4]
    assert np.shares_memory(r[2], r.values)  # rows are views


@pytest.mark.parametrize(
    ("values", "offsets", "error"),
    [
        (np.arange(3.0), np.array([1, 3]), ValueError),
        (np.arange(3.0), np.array([0, 2, 1, 3]), ValueError),
        (np.arange(3.0), np.array([0, 2]), ValueError),
        (np.arange(3.0), np.array([0.0, 3.0]), TypeError),
        (np.arange(3.0), np.zeros((2, 2), dtype=np.int64), ValueError),
    ],
)
def test_ragged_validation(values, offsets, error):
    with pytest.raises(error):
        Ragged(values, offsets)


def test_ragged_from_rows_and_concat():
    r = Ragged.from_rows([[1, 2], [], [3]], "int16")
    assert r.values.dtype == np.int16
    assert r.counts().tolist() == [2, 0, 1]
    c = Ragged.concat([r, Ragged.from_rows([[4, 5, 6]], "int16"), r])
    assert c.counts().tolist() == [2, 0, 1, 3, 2, 0, 1]
    assert c.values.tolist() == [1, 2, 3, 4, 5, 6, 1, 2, 3]
    assert len(Ragged.from_rows([], "float32")) == 0
    with pytest.raises(ValueError, match="nothing"):
        Ragged.concat([])


def test_ragged_select():
    rng = np.random.default_rng(0)
    r = random_ragged(rng)
    for rows in (
        np.array([3, 1, 1, 7, 0]),
        np.arange(5, 12),
        rng.random(len(r)) < 0.5,
        np.array([], dtype=np.int64),
        np.array([-1, -2]),
    ):
        s = r.select(rows)
        index = np.flatnonzero(rows) if rows.dtype == bool else rows % len(r)
        assert len(s) == len(index)
        for k, i in enumerate(index):
            np.testing.assert_array_equal(s[k], r[i])
    with pytest.raises(ValueError, match="shape"):
        r.select(np.ones(3, dtype=bool))
    with pytest.raises(IndexError):
        r.select(np.array([len(r)]))
    with pytest.raises(TypeError):
        r.select(np.array([0.5]))


def test_ragged_padded_constant():
    r = Ragged.from_counts(np.arange(1.0, 7.0), [2, 0, 3, 1])
    np.testing.assert_array_equal(r.padded(), [[1, 2, 0], [0, 0, 0], [3, 4, 5], [6, 0, 0]])
    np.testing.assert_array_equal(r.padded(2, fill=-1), [[1, 2], [-1, -1], [3, 4], [6, -1]])
    assert r.padded(4, dtype="float32").dtype == np.float32
    assert r.padded(0).shape == (4, 0)
    empty = Ragged.from_counts(np.zeros(0), [0, 0])
    assert empty.padded().shape == (2, 1)  # at least length 1


def test_ragged_padded_wrap_equals_weaver_kernel():
    rng = np.random.default_rng(3)
    r = random_ragged(rng, n=200, max_count=12)
    for length in (1, 5, 8, 20):
        np.testing.assert_array_equal(r.padded(length, mode="wrap"), weaver_repeat_pad(r.values, r.offsets, length))
    with pytest.raises(ValueError, match="mode"):
        r.padded(3, mode="reflect")  # type: ignore[arg-type]


def test_ragged_with_trailing_dimensions():
    values = np.arange(12.0).reshape(6, 2)
    r = Ragged.from_counts(values, [2, 0, 3, 1])
    assert r[2].shape == (3, 2)
    p = r.padded(3, mode="wrap")
    assert p.shape == (4, 3, 2)
    np.testing.assert_array_equal(p[0], values[[0, 1, 0]])
    np.testing.assert_array_equal(p[3], values[[5, 5, 5]])
    np.testing.assert_array_equal(r.select([3, 0]).values, values[[5, 0, 1]])


def test_ragged_map():
    r = Ragged.from_counts(np.arange(4.0), [1, 3])
    m = r.map(np.square)
    assert m.offsets is r.offsets
    assert m.values.tolist() == [0, 1, 4, 9]
    with pytest.raises(ValueError, match="values"):
        r.map(lambda v: v[:2])


# ------------------------------------------------------------------------------------------------------ JetData


def test_jetdata_validation():
    px = Ragged.from_counts(np.ones(3, np.float32), [1, 2])
    with pytest.raises(ValueError, match="rows"):
        JetData({"part_px": px, "jet_pt": np.zeros(3)})
    with pytest.raises(ValueError, match="share"):
        JetData({"part_px": px, "part_py": Ragged.from_counts(np.ones(3, np.float32), [2, 1])})
    with pytest.raises(TypeError, match="Ragged"):
        JetData({"part_px": np.zeros(2)})
    with pytest.raises(TypeError, match="1-D"):
        JetData({"jet_pt": px})
    with pytest.raises(TypeError, match="Ragged"):
        JetData({"truth_part_x": np.zeros(2)})  # a ragged prefix on a per-jet column
    with pytest.raises(ValueError, match="1-D"):
        JetData({"jet_pt": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="source"):
        JetData({}, source="cms")  # type: ignore[arg-type]
    assert len(JetData({})) == 0


def test_group_offsets_are_shared():
    px = Ragged.from_counts(np.ones(3, np.float32), [1, 2])
    py = Ragged(np.zeros(3, np.float32), px.offsets.copy())
    prong = Ragged.from_counts(np.ones(5, np.float32), [3, 2])  # another group: other lengths are fine
    data = JetData({"part_px": px, "part_py": py, "truth_prong_px": prong})
    assert data["part_py"].offsets is data["part_px"].offsets  # type: ignore[union-attr]
    for sub in (data.select([1, 0]), data.head(1), JetData.concat([data, data])):
        assert sub["part_py"].offsets is sub["part_px"].offsets  # type: ignore[union-attr]


def test_jetdata_access():
    data = small_data()
    assert len(data) == 6
    assert "part_px" in data
    assert "nothing" not in data
    assert data.names[0] == "part_px"
    assert data.counts.dtype == np.int64
    assert data.counts.tolist() == data["part_px"].counts().tolist()  # type: ignore[union-attr]
    assert data.has_truth
    assert not data.without_truth().has_truth
    with pytest.raises(KeyError, match="part_px"):
        data["part_pxx"]
    with pytest.raises(TypeError, match="not iterable"):
        list(data)


def test_select_head_chunks_concat():
    data = small_data(10)
    rows = np.array([9, 0, 4, 4])
    sub = data.select(rows)
    assert len(sub) == 4
    np.testing.assert_array_equal(sub["jet_pt"], data["jet_pt"][rows])  # type: ignore[index]
    for k, i in enumerate(rows):
        np.testing.assert_array_equal(sub["part_px"][k], data["part_px"][i])  # type: ignore[index]
        np.testing.assert_array_equal(sub["truth_prong_px"][k], data["truth_prong_px"][i])  # type: ignore[index]
    mask = np.arange(10) % 2 == 0
    np.testing.assert_array_equal(data.select(mask)["jet_pt"], data["jet_pt"][mask])  # type: ignore[index]
    assert len(data.head(3)) == 3
    assert len(data.head(100)) == 10
    assert len(data.head(0)) == 0
    blocks = list(data.chunks(4))
    assert [len(b) for b in blocks] == [4, 4, 2]
    again = JetData.concat(blocks)
    for name in data.names:
        a, b = data[name], again[name]
        if isinstance(a, Ragged):
            assert isinstance(b, Ragged)
            np.testing.assert_array_equal(a.values, b.values)
            np.testing.assert_array_equal(a.offsets, b.offsets)
        else:
            np.testing.assert_array_equal(a, b)
    assert again.provenance == data.provenance
    with pytest.raises(ValueError, match="size"):
        next(data.chunks(0))


def test_concat_rules():
    a = small_data(4)
    b = JetData(dict(a.columns), config={"x": 1})
    assert JetData.concat([a, a]).config is None
    assert JetData.concat([b, b]).config == {"x": 1}
    assert JetData.concat([a, b]).config is None
    with pytest.raises(ValueError, match="different columns"):
        JetData.concat([a, a.drop("truth_")])
    with pytest.raises(ValueError, match="nothing"):
        JetData.concat([])


def test_with_columns_and_drop():
    data = small_data()
    more = data.with_columns(score=np.ones(6), part_w=data["part_px"].map(np.ones_like))  # type: ignore[union-attr]
    assert more.names[-2:] == ("score", "part_w")
    assert "score" not in data
    replaced = data.with_columns(jet_pt=np.zeros(6))
    assert replaced.names == data.names
    assert not np.any(replaced["jet_pt"])
    with pytest.raises(ValueError, match="rows"):
        data.with_columns(score=np.ones(5))
    assert not data.drop("truth_").has_truth
    truth = tuple(n for n in data.names if n.startswith("truth_"))
    assert data.drop("jet_pt", "label_").names == ("part_px", "part_py", "part_pz", "part_energy", *truth)
    with pytest.raises(ValueError, match="jet_pt"):
        data.drop("jet_ptt")
    assert JetData({"jet_pt": np.ones(2)}).without_truth().names == ("jet_pt",)


def test_labels():
    data = synthetic_jets([np.ones((1, 4))] * 4, labels=["QCD", "Wqq", "Tbqq", "Wqq"])
    assert data.labels().tolist() == [0, 1, 2, 1]
    assert data.labels().dtype == np.int64
    assert data.labels(JETCLASS_LABELS).tolist() == [0, 7, 8, 7]
    higgs = synthetic_jets([np.ones((1, 4))] * 2, labels=["QCD", "Hbb"])
    with pytest.raises(ValueError, match="one-hot"):
        higgs.labels()
    with pytest.raises(ValueError, match="label_Wqq"):
        higgs.drop("label_Wqq").labels()


def test_process_names():
    toy = small_data(6)
    assert toy.process_names.tolist() == ["qcd", "wqq", "tbqq"] * 2
    no_truth = toy.without_truth()
    assert no_truth.process_names.tolist() == ["qcd", "wqq", "tbqq"] * 2
    jc = synthetic_jets([np.ones((1, 4))] * 3, labels=["QCD", "Hbb", "Tbl"])
    assert jc.process_names.tolist() == ["qcd", "hbb", "tbl"]
    with pytest.raises(ValueError, match="processes"):
        _ = JetData({"jet_pt": np.ones(2)}).process_names


def test_repr():
    toy = small_data(6)
    assert repr(toy) == (
        f"JetData(6 jets: qcd 2, wqq 2, tbqq 2; preset detector; {int(toy.counts.sum())} particles; with truth)"
    )
    jc = synthetic_jets([np.ones((2, 4))] * 1500, labels=["QCD"] * 1000 + ["Tbqq"] * 500)
    assert repr(jc) == "JetData(1 500 jets: QCD 1 000, Tbqq 500; JetClass; 3 000 particles)"
    assert repr(JetData({"jet_pt": np.ones(2)})) == "JetData(2 jets)"


def test_jet_view():
    data = small_data()
    view = data.jet(-1)
    assert isinstance(view, JetView)
    assert view.index == 5
    assert set(view.particles) == {"part_px", "part_py", "part_pz", "part_energy", "truth_part_prong"}
    assert set(view.prongs) == {"truth_prong_px"}
    assert view.partons == {}
    assert view.emissions == {}
    assert view.lund == {}
    assert view.jet["jet_pt"] == 5.0
    assert isinstance(view.jet["truth_process"], int)
    np.testing.assert_array_equal(view.particles["part_px"], data["part_px"][5])  # type: ignore[index]
    with pytest.raises(IndexError):
        data.jet(6)
    pytest.importorskip("pandas")
    assert list(view.particles_frame().columns) == list(view.particles)


def test_padded_p4():
    data = two_particle_jet(0.3, 0.4)
    data = JetData.concat([data, three_prong_jet()])
    p4, mask = data.padded_p4()
    assert p4.shape == (2, 3, 4)
    assert p4.dtype == np.float64
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    assert not np.any(p4[0, 2])
    p4, mask = data.padded_p4(2)
    assert p4.shape == (2, 2, 4)
    assert mask.all()
    # the four-vectors sum to the jet
    total = data.padded_p4()[0].sum(axis=1)
    np.testing.assert_allclose(np.hypot(total[:, 0], total[:, 1]), data["jet_pt"], rtol=1e-6)


def test_to_pandas():
    pd = pytest.importorskip("pandas")
    data = small_data()
    jets = data.to_pandas()
    assert isinstance(jets, pd.DataFrame)
    assert "jet_pt" in jets.columns
    assert "part_px" not in jets.columns
    assert len(jets) == 6
    parts = data.to_pandas("particle")
    assert len(parts) == int(data.counts.sum())
    assert list(parts.index.names) == ["jet", "particle"]
    assert set(parts.columns) == {"part_px", "part_py", "part_pz", "part_energy", "truth_part_prong"}
    assert parts.loc[(2, 0), "part_px"] == data["part_px"][2][0]  # type: ignore[index]
    with pytest.raises(ValueError, match="level"):
        data.to_pandas("prong")  # type: ignore[arg-type]


def test_synthetic_jets_have_the_jetclass_columns(tmp_path):
    data = three_prong_jet()
    assert data.names == JETCLASS_BRANCHES
    for name in JETCLASS_BRANCHES:
        col = data[name]
        assert (col.values if isinstance(col, Ragged) else col).dtype == np.dtype(COLUMNS[name].dtype), name
    assert data.process_names.tolist() == ["tbqq"]
    path = jetclass_like_file(tmp_path, n=12)
    assert path.is_file()
