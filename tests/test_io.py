from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from itj.data import COLUMNS, JETCLASS_BRANCHES, JetData, Ragged
from itj.io import FORMAT_VERSION, METADATA_KEYS, describe, load, merge, read_metadata, save, write_split
from synthetic import add_truth_columns, jetclass_like_file, massless, synthetic_jets

#: A JetClass validation file, if this machine has one (the acceptance test of the schema round trip).
_JETCLASS_VAL = [
    Path("data/jetclass/val/WToQQ_120.parquet"),
    Path("/data/hqu/datasets/JetClass/Pythia/val_5M_parquet/WToQQ_120.parquet"),
]


def toy_like(n: int = 30, *, seed: int = 0, first_id: int = 0, config: dict | None = None) -> JetData:
    """Random jets with every column of COLUMNS, as a toy data set (source "itj")."""
    rng = np.random.default_rng(seed)
    particles = []
    for _ in range(n):
        k = int(rng.integers(1, 25))
        particles.append(
            np.stack([massless(pt, eta, phi) for pt, eta, phi in rng.normal([40, 0.3, 0], [10, 0.2, 0.2], (k, 3))])
        )
    data = add_truth_columns(synthetic_jets(particles, [("QCD", "Wqq", "Tbqq")[i % 3] for i in range(n)]), seed=seed)
    data = data.with_columns(truth_event_id=np.arange(first_id, first_id + n, dtype=np.int64))
    config = {"hard": {"w_mass": 80.4}} if config is None else config
    provenance = {"seed": seed, "preset": "detector", "wall_time_s": 1.5, "workers": 1, "package_version": "0.1.0"}
    return JetData(data.columns, config=config, provenance=provenance, source="itj")


def assert_same(a: JetData, b: JetData) -> None:
    assert a.names == b.names
    for name in a.names:
        x, y = a[name], b[name]
        assert isinstance(x, Ragged) == isinstance(y, Ragged), name
        if isinstance(x, Ragged):
            assert isinstance(y, Ragged)
            np.testing.assert_array_equal(x.offsets, y.offsets, err_msg=name)
            x, y = x.values, y.values
        assert x.dtype == y.dtype, name
        np.testing.assert_array_equal(x, y, err_msg=name)


def expected_field(name: str) -> pa.Field:
    info = COLUMNS[name]
    value = pa.from_numpy_dtype(np.dtype(info.dtype))
    type_ = pa.large_list(pa.field("item", value, nullable=False)) if info.jagged else value
    return pa.field(name, type_, nullable=False)


# ----------------------------------------------------------------------------------------------------- save, load


def test_round_trip_of_every_column(tmp_path):
    data = toy_like(40)
    assert set(data.names) == set(COLUMNS)
    path = save(data, tmp_path / "sub" / "toy.parquet")
    assert path == tmp_path / "sub" / "toy.parquet"
    assert not path.with_name("toy.parquet.part").exists()
    back = load(path)
    assert back.source == "itj"
    assert back.config == data.config
    assert back.provenance == data.provenance
    # JetClass columns first in the JetClass order, then the truth columns in COLUMNS order
    assert back.names == tuple(COLUMNS)
    assert_same(data, back)
    assert back.has_truth
    assert repr(back).startswith("JetData(40 jets: qcd 14, wqq 13, tbqq 13; preset detector")


def test_schema_is_exactly_jetclass(tmp_path):
    path = save(toy_like(10), tmp_path / "toy.parquet")
    schema = pq.read_schema(path)
    assert schema.names == list(COLUMNS)
    for name in COLUMNS:
        assert schema.field(name).equals(expected_field(name)), name
        meta = schema.field(name).metadata
        assert meta[b"units"].decode() == COLUMNS[name].units
        assert meta[b"description"].decode() == COLUMNS[name].description
    assert str(schema.field("part_px").type) == "large_list<item: float not null>"
    assert str(schema.field("part_isMuon").type) == "large_list<item: int32 not null>"
    assert [str(schema.field(n).type) for n in ("label_QCD", "label_Hbb", "label_Wqq")] == ["float", "bool", "int32"]


def test_committed_sample_has_the_jetclass_schema(jetclass_sample, tmp_path):
    from conftest import JETCLASS_SAMPLE

    schema = pq.read_schema(JETCLASS_SAMPLE)
    assert schema.names == list(JETCLASS_BRANCHES)
    for name in JETCLASS_BRANCHES:
        assert schema.field(name).equals(expected_field(name)), name
    assert jetclass_sample.source == "jetclass"
    assert jetclass_sample.config is None
    assert len(jetclass_sample) == 150
    back = load(save(jetclass_sample, tmp_path / "copy.parquet"))
    assert back.source == "jetclass"
    copy = pq.read_schema(tmp_path / "copy.parquet")
    assert copy.equals(schema, check_metadata=False)
    assert [str(f.type) for f in copy] == [str(f.type) for f in schema]
    assert pq.read_schema(tmp_path / "copy.parquet").metadata is None  # no itj metadata for JetClass data
    assert_same(jetclass_sample, back)


def test_real_jetclass_file_round_trip(tmp_path):
    path = next((p for p in _JETCLASS_VAL if p.is_file()), None)
    if path is None:
        pytest.skip("no JetClass validation file on this machine")
    data = load(path, max_jets=3000)
    assert data.source == "jetclass"
    out = save(data, tmp_path / "copy.parquet")
    original = pq.read_schema(path)
    copy = pq.read_schema(out)
    assert copy.names == original.names == list(JETCLASS_BRANCHES)
    assert copy.equals(original, check_metadata=False)
    assert [str(copy.field(n).type) for n in copy.names] == [str(original.field(n).type) for n in original.names]
    assert [copy.field(n).nullable for n in copy.names] == [original.field(n).nullable for n in original.names]
    assert_same(data, load(out))


def test_metadata_and_row_groups(tmp_path):
    data = toy_like(25)
    path = save(data, tmp_path / "toy.parquet", row_group_size=10)
    file = pq.ParquetFile(path)
    assert [file.metadata.row_group(g).num_rows for g in range(file.metadata.num_row_groups)] == [10, 10, 5]
    assert file.metadata.row_group(0).column(0).compression == "ZSTD"
    metadata = file.schema_arrow.metadata
    assert {k.decode() for k in metadata} >= set(METADATA_KEYS)
    assert metadata[b"itj.format_version"].decode() == FORMAT_VERSION
    assert json.loads(metadata[b"itj.config"]) == data.config
    info = read_metadata(path)
    assert info["format_version"] == FORMAT_VERSION
    assert info["n_jets"] == 25
    assert info["config"] == data.config
    assert info["provenance"] == data.provenance
    assert info["columns"]["part_px"] == "var * float32"
    assert info["columns"]["label_Hbb"] == "bool"
    assert info["columns"]["truth_event_id"] == "int64"
    save(data, tmp_path / "default.parquet")
    assert pq.ParquetFile(tmp_path / "default.parquet").metadata.num_row_groups == 1


def test_save_options_and_conversions(tmp_path):
    data = toy_like(6)
    data = data.with_columns(
        jet_pt=np.asarray(data["jet_pt"], dtype=np.float64),  # cast back to float32 on save
        file_index=np.zeros(6, np.int16),  # never written
        my_score=np.linspace(0, 1, 6),  # an unknown column keeps its dtype and goes last
    )
    path = save(data, tmp_path / "a.parquet", truth=False)
    schema = pq.read_schema(path)
    assert schema.names == [*JETCLASS_BRANCHES, "my_score"]
    assert schema.field("jet_pt").type == pa.float32()
    assert schema.field("my_score").type == pa.float64()
    back = load(path)
    assert not back.has_truth
    assert "file_index" not in back
    with pytest.raises(TypeError, match="1-D values"):
        save(
            data.with_columns(part_px=Ragged(np.ones((len(data["part_py"].values), 2)), data["part_py"].offsets)),
            tmp_path / "b.parquet",
        )  # type: ignore[union-attr]


def test_refuse_overwrite_and_atomic_write(tmp_path, monkeypatch):
    data = toy_like(5)
    path = save(data, tmp_path / "a.parquet")
    with pytest.raises(FileExistsError, match="--overwrite"):
        save(data, path)
    save(data.head(3), path, overwrite=True)
    assert len(load(path)) == 3

    def broken(table, where, **kwargs):
        Path(where).write_bytes(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(pq, "write_table", broken)
    with pytest.raises(OSError, match="disk full"):
        save(data, path, overwrite=True)
    with pytest.raises(OSError, match="disk full"):
        save(data, tmp_path / "new.parquet")
    monkeypatch.undo()
    assert not (tmp_path / "a.parquet.part").exists()
    assert not (tmp_path / "new.parquet.part").exists()
    assert not (tmp_path / "new.parquet").exists()
    assert len(load(path)) == 3  # the old file is untouched


def test_refuse_newer_or_missing_format(tmp_path):
    table = pq.read_table(save(toy_like(4), tmp_path / "a.parquet"))
    metadata = dict(table.schema.metadata)
    metadata[b"itj.format_version"] = b"2"
    pq.write_table(table.replace_schema_metadata(metadata), tmp_path / "newer.parquet")
    with pytest.raises(ValueError, match="format version 2"):
        load(tmp_path / "newer.parquet")
    with pytest.raises(ValueError, match="format version 2"):
        read_metadata(tmp_path / "newer.parquet")
    del metadata[b"itj.format_version"]
    pq.write_table(table.replace_schema_metadata(metadata), tmp_path / "missing.parquet")
    with pytest.raises(ValueError, match=r"no itj\.format_version"):
        load(tmp_path / "missing.parquet")
    (tmp_path / "junk.parquet").write_text("not parquet")
    with pytest.raises(ValueError, match="not a readable Parquet file"):
        load(tmp_path / "junk.parquet")


def test_load_errors(tmp_path):
    path = save(toy_like(4), tmp_path / "a.parquet")
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.parquet")
    with pytest.raises(FileNotFoundError, match="no file matches"):
        load(str(tmp_path / "*.root"))
    with pytest.raises(ValueError, match="did you mean 'jet_pt'"):
        load(path, columns=["jet_ptt"])
    with pytest.raises(ValueError, match="part"):
        load(path, part="training")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_jets"):
        load(path, max_jets=-1)


def test_columns_and_truth_options(tmp_path):
    path = save(toy_like(8), tmp_path / "a.parquet")
    data = load(path, columns=["jet_sdmass", "part_px", "truth_part_prong"])
    assert data.names == ("jet_sdmass", "part_px", "truth_part_prong")
    assert data["part_px"].offsets is data["truth_part_prong"].offsets  # type: ignore[union-attr]
    assert not load(path, truth=False).has_truth
    assert load(path, columns=["jet_pt", "truth_mass"], truth=False).names == ("jet_pt",)


def test_jetclass_like_file_loads_as_jetclass(tmp_path):
    path = jetclass_like_file(tmp_path, n=12)
    data = load(path)
    assert data.source == "jetclass"
    assert data.config is None
    assert data.provenance is None
    assert data.names == JETCLASS_BRANCHES
    assert data.process_names.tolist()[:3] == ["qcd", "wqq", "tbqq"]
    assert read_metadata(path)["format_version"] is None


# ----------------------------------------------------------------------------------------------------- the idx split


@pytest.mark.parametrize("n", [0, 1, 7, 10, 1003])
def test_part_boundaries(tmp_path, n):
    data = toy_like(max(n, 1)).head(n)
    path = save(data, tmp_path / "a.parquet", row_group_size=100)
    a, b = int(0.2 * n), int(0.3 * n)
    ids = {part: load(path, part=part)["truth_event_id"].tolist() for part in ("train", "val", "test")}  # type: ignore[union-attr]
    assert ids["test"] == list(range(a))
    assert ids["val"] == list(range(a, b))
    assert ids["train"] == list(range(b, n))
    if n:
        assert_same(load(path, part="train"), data.select(np.arange(b, n)))
    assert load(path, part="train", max_jets=5)["truth_event_id"].tolist() == list(range(b, min(b + 5, n)))  # type: ignore[union-attr]


def test_several_files_get_a_file_index(tmp_path):
    a = save(toy_like(10, seed=1), tmp_path / "b_first.parquet")
    b = save(toy_like(20, seed=2, first_id=100), tmp_path / "a_second.parquet")
    data = load([a, b])
    assert len(data) == 30
    assert data["file_index"].dtype == np.int16  # type: ignore[union-attr]
    assert data["file_index"].tolist() == [0] * 10 + [1] * 20  # type: ignore[union-attr]
    assert data.config == {"hard": {"w_mass": 80.4}}
    assert data.provenance is None  # the seeds differ
    globbed = load(str(tmp_path / "*.parquet"))  # sorted: a_second first
    assert globbed["truth_event_id"].tolist()[:2] == [100, 101]  # type: ignore[union-attr]
    test = load([a, b], part="test")
    assert test["truth_event_id"].tolist() == [0, 1, *range(100, 104)]  # type: ignore[union-attr]
    assert test["file_index"].tolist() == [0, 0, 1, 1, 1, 1]  # type: ignore[union-attr]
    assert len(load([a, b], max_jets=3)) == 6
    from itj.features import split

    test_split = split(data)[2]
    assert test_split["truth_event_id"].tolist() == test["truth_event_id"].tolist()  # type: ignore[union-attr]


def test_write_split_equals_load_part(tmp_path):
    from conftest import JETCLASS_SAMPLE

    toy = save(toy_like(1234, seed=3), tmp_path / "in" / "toy.parquet", row_group_size=1000)
    out = write_split([toy, JETCLASS_SAMPLE], tmp_path / "split")
    assert set(out) == {"train", "val", "test"}
    for part, paths in out.items():
        assert paths == [
            tmp_path / "split" / part / "toy.parquet",
            tmp_path / "split" / part / "jetclass_sample.parquet",
        ]
        for source, path in zip([toy, JETCLASS_SAMPLE], paths, strict=True):
            assert pq.read_schema(path).equals(pq.read_schema(source))  # same columns, types and metadata
            assert_same(load(path), load(source, part=part))  # type: ignore[arg-type]
            file = pq.ParquetFile(path)
            assert all(file.metadata.row_group(g).num_rows <= 1000 for g in range(file.metadata.num_row_groups))
    assert len(load(out["train"][0])) == 1234 - int(0.3 * 1234)
    assert load(out["test"][0]).config == load(toy).config
    with pytest.raises(FileExistsError, match="--overwrite"):
        write_split([toy], tmp_path / "split")
    write_split([toy], tmp_path / "split", overwrite=True, test=0.5, val=0.25)
    assert len(load(tmp_path / "split" / "test" / "toy.parquet")) == 617
    other = save(toy_like(5, seed=9), tmp_path / "other" / "toy.parquet")
    with pytest.raises(ValueError, match="'toy'"):
        write_split([toy, other], tmp_path / "split2")
    with pytest.raises(ValueError, match="test"):
        write_split([toy], tmp_path / "split3", test=0.9, val=0.2)


# ----------------------------------------------------------------------------------------------------------- merge


def test_merge_concatenates_and_records_the_inputs(tmp_path):
    a = save(toy_like(10, seed=1), tmp_path / "a.parquet")
    b = save(toy_like(15, seed=2, first_id=10, config={"hard": {"w_mass": 60.0}}), tmp_path / "b.parquet")
    out = merge([a, b], tmp_path / "merged.parquet")
    data = load(out)
    assert len(data) == 25
    assert data["truth_event_id"].tolist() == list(range(25))  # type: ignore[union-attr]
    assert data.config is None  # the configs differ
    merged = data.provenance["merged"]  # type: ignore[index]
    assert [m["file"] for m in merged] == ["a.parquet", "b.parquet"]
    assert [m["seed"] for m in merged] == [1, 2]
    assert merged[1]["config"] == {"hard": {"w_mass": 60.0}}
    assert data.provenance["preset"] == "detector"  # type: ignore[index]
    assert_same(data, load([a, b]).drop("file_index").with_columns())
    assert "merged from 2 files" in describe(out)
    # a merged file merges again; its seeds are followed
    c = save(toy_like(5, seed=3, first_id=50), tmp_path / "c.parquet")
    again = load(merge([out, c], tmp_path / "again.parquet"))
    assert len(again) == 30
    d = save(toy_like(5, seed=2, first_id=90), tmp_path / "d.parquet")
    with pytest.raises(ValueError, match=r"seed\(s\) \[2\]"):
        merge([out, d], tmp_path / "bad.parquet")


def test_merge_rules(tmp_path):
    a = save(toy_like(10, seed=1), tmp_path / "a.parquet")
    same_seed = save(toy_like(10, seed=1, first_id=10), tmp_path / "same_seed.parquet")
    same_ids = save(toy_like(10, seed=2), tmp_path / "same_ids.parquet")
    with pytest.raises(ValueError, match="seed"):
        merge([a, same_seed], tmp_path / "m.parquet")
    with pytest.raises(ValueError, match="event ids"):
        merge([a, same_ids], tmp_path / "m.parquet")
    out = merge([a, same_ids], tmp_path / "m.parquet", renumber=True, shuffle=True, seed=4)
    data = load(out)
    assert sorted(data["truth_event_id"].tolist()) == list(range(20))  # type: ignore[union-attr]
    assert data["truth_event_id"].tolist() != list(range(20))  # type: ignore[union-attr]
    assert data.config == {"hard": {"w_mass": 80.4}}  # equal configs are kept
    unshuffled = load([a, same_ids]).drop("file_index")
    order = np.asarray(data["truth_event_id"])
    np.testing.assert_array_equal(data["jet_pt"], np.asarray(unshuffled["jet_pt"])[order])
    with pytest.raises(FileExistsError):
        merge([a, same_ids], out, renumber=True)
    with pytest.raises(ValueError, match="also an input"):
        merge([a, same_ids], a, renumber=True, overwrite=True)
    no_truth = save(toy_like(4, seed=7, first_id=40), tmp_path / "no_truth.parquet", truth=False)
    with pytest.raises(ValueError, match="different columns"):
        merge([a, no_truth], tmp_path / "x.parquet")
    jetclass = jetclass_like_file(tmp_path, n=5)
    with pytest.raises(ValueError, match="JetClass"):
        merge([a, jetclass], tmp_path / "y.parquet")


# -------------------------------------------------------------------------------------------------------- describe


def test_describe(tmp_path, jetclass_sample):
    from conftest import JETCLASS_SAMPLE

    text = describe(save(toy_like(30), tmp_path / "toy.parquet"))
    assert "toy file, format version 1: 30 jets, 90 columns (41 JetClass, 49 truth)" in text
    assert "preset: detector" in text
    assert "seed: 0" in text
    assert "wall time: 1.5 s (1 worker)" in text
    assert "qcd" in text
    assert "tbqq" in text
    text = describe(JETCLASS_SAMPLE)
    assert "JetClass file (no itj metadata): 150 jets, 41 columns" in text
    lines = [line.split() for line in text.splitlines()]
    rows = {line[0]: line[1:] for line in lines if line and line[0] in ("QCD", "Wqq", "Tbqq")}
    assert [rows[c][0] for c in ("QCD", "Wqq", "Tbqq")] == ["50", "50", "50"]
    qcd = jetclass_sample.counts[:50].mean()
    assert float(rows["QCD"][1]) == pytest.approx(qcd, abs=0.05)


# ------------------------------------------------------------------------------------------------------------- ROOT


def test_root_files(tmp_path):
    uproot = pytest.importorskip("uproot")
    ak = pytest.importorskip("awkward")
    data = load(jetclass_like_file(tmp_path, n=9))
    branches: dict[str, object] = {}
    for name in JETCLASS_BRANCHES:
        col = data[name]
        branches[name] = ak.unflatten(col.values, col.counts()) if isinstance(col, Ragged) else np.asarray(col)
    path = tmp_path / "sample.root"
    with uproot.recreate(path) as f:
        f["tree"] = branches
    back = load(path, columns=list(JETCLASS_BRANCHES))
    assert back.source == "jetclass"
    assert_same(data, back)
    assert len(load(path, columns=["jet_pt"], part="test")) == 1
    assert read_metadata(path)["columns"]["part_px"] == "var * float32"
    assert read_metadata(path)["n_jets"] == 9
    out = write_split([path], tmp_path / "split")
    assert_same(load(out["train"][0], columns=list(JETCLASS_BRANCHES)), back.select(np.arange(2, 9)))
    out = write_split([path], tmp_path / "split_half", test=0.5, val=0.25)
    assert [len(load(out[p][0])) for p in ("train", "val", "test")] == [3, 2, 4]
    with uproot.recreate(tmp_path / "other.root") as f:
        f["events"] = {"x": np.ones(3)}
    with pytest.raises(ValueError, match="no tree 'tree'"):
        load(tmp_path / "other.root")


def test_docs_quote_the_current_version():
    """The pages that name the package version (format, toy model, citation) follow ``itj.__version__``."""
    import re

    import itj
    from helpers import ROOT

    version = itj.__version__
    data_format = " ".join((ROOT / "docs" / "data_format.md").read_text().split())
    assert f"format version {FORMAT_VERSION}, written by `itj` 0.1.0 to {version}." in data_format
    physics = " ".join((ROOT / "docs" / "physics.md").read_text().split())
    assert f"as implemented in `itj.toy` (version {version}," in physics
    citation = (ROOT / "CITATION.cff").read_text()
    assert re.search(rf"^version: {re.escape(version)}$", citation, re.MULTILINE)
