"""Planted shortcuts of T7.2: the choice from the seed, one process changed, hidden from describe, reveal."""

import numpy as np
import pytest

import itj.toy
from itj.io import describe, load, read_metadata
from itj.toy import generate, preset
from itj.toy.cli import main
from itj.toy.mystery import SHORTCUTS, decode, encode, mystery_overrides, reveal


def test_shortcuts_are_valid_overrides():
    assert len(SHORTCUTS) == 5
    assert len({name for name, *_ in SHORTCUTS}) == 5
    for _, process, field, value in SHORTCUTS:
        assert process in ("qcd", "wqq", "tbqq")
        preset("detector", **{field: value})  # a valid configuration


def test_mystery_overrides_are_deterministic_and_cover_every_shortcut(monkeypatch):
    monkeypatch.delenv("ITJ_MYSTERY_KEY", raising=False)
    assert mystery_overrides(73) == mystery_overrides(73)
    seen = {(p, f, v) for seed in range(200) for p, fv in mystery_overrides(seed).items() for f, v in fv.items()}
    assert seen == {(p, f, v) for _, p, f, v in SHORTCUTS}
    public = [mystery_overrides(seed) for seed in range(50)]
    monkeypatch.setenv("ITJ_MYSTERY_KEY", "12345")
    private = [mystery_overrides(seed) for seed in range(50)]
    assert public != private
    monkeypatch.setenv("ITJ_MYSTERY_KEY", "abc")
    with pytest.raises(ValueError, match="ITJ_MYSTERY_KEY"):
        mystery_overrides(1)


def test_not_reexported():
    for name in ("mystery_overrides", "SHORTCUTS", "reveal"):
        assert name not in itj.toy.__all__
        assert not hasattr(itj.toy, name)


def test_encode_decode():
    payload = decode(encode(5))
    overrides = mystery_overrides(5)
    assert {payload["process"]: {payload["field"]: payload["value"]}} == overrides
    with pytest.raises(ValueError, match="cannot decode"):
        decode("not base64!")


def test_mystery_file_differs_from_a_clean_file_in_one_process(tmp_path, monkeypatch):
    monkeypatch.delenv("ITJ_MYSTERY_KEY", raising=False)
    seed = 73
    path = tmp_path / "mystery.parquet"
    assert main(["generate", "--mystery", "-n", "150", "--seed", str(seed), "-o", str(path), "-q"]) == 0
    data = load(path)
    clean = generate(150, seed=seed)
    ((process, changes),) = mystery_overrides(seed).items()
    names = clean.process_names
    np.testing.assert_array_equal(names, data.process_names)
    for other in {"qcd", "wqq", "tbqq"} - {process}:
        rows = np.flatnonzero(names == other)
        for column in ("part_px", "part_energy", "jet_sdmass", "truth_seed_pt"):
            a, b = clean.select(rows)[column], data.select(rows)[column]
            np.testing.assert_array_equal(getattr(a, "values", a), getattr(b, "values", b))
    rows = np.flatnonzero(names == process)
    assert not np.array_equal(clean.select(rows)["part_px"].values, data.select(rows)["part_px"].values)
    # hidden: the stored configuration is the preset, no process_overrides, nothing in jetgen info
    meta = read_metadata(path)
    assert meta["config"] == preset("detector").to_dict()
    assert "process_overrides" not in meta["provenance"]
    text = describe(path)
    ((field, _),) = changes.items()
    assert field not in text
    assert process not in text.split("process")[0]  # only in the table of jet counts
    assert "mystery" in meta["provenance"]
    # revealed on request
    revealed = reveal(path)
    assert field in revealed
    assert process in revealed
    assert reveal(data) == revealed


def test_reveal_of_a_clean_file(tmp_toy_file, toy_small):
    assert reveal(tmp_toy_file) == "this file has no planted shortcut"
    assert reveal(toy_small) == "this file has no planted shortcut"
