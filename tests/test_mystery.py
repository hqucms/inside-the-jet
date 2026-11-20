"""Planted shortcuts of B.3: the choice from the seed, one process changed, hidden from describe, reveal."""

import base64
import json

import numpy as np
import pytest

import itj.toy
from itj.data import JetData
from itj.io import describe, load, read_metadata
from itj.toy import generate, preset
from itj.toy.cli import main
from itj.toy.mystery import (
    HIDDEN,
    RUN_STATISTICS,
    SHORTCUTS,
    decode,
    encode,
    hide,
    mystery_overrides,
    reveal,
)


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


def test_encode_decode(monkeypatch):
    monkeypatch.delenv("ITJ_MYSTERY_KEY", raising=False)
    text = encode(5)
    payload = decode(text)
    overrides = mystery_overrides(5)
    assert {payload["process"]: {payload["field"]: payload["value"]}} == overrides
    with pytest.raises(ValueError, match="cannot decode"):
        decode("not base64!")
    # a base64 decoder alone does not reveal it
    raw = base64.b64decode(text.split(":", 1)[1])
    for name, process, field, _ in SHORTCUTS:
        for word in (name, process, field):
            assert word.encode() not in raw
    # files written before the scrambling (plain base64 of JSON) still decode
    legacy = base64.b64encode(json.dumps(payload).encode()).decode()
    assert decode(legacy) == payload
    # an instructor's key scrambles differently and is needed to reveal
    monkeypatch.setenv("ITJ_MYSTERY_KEY", "123457")
    private = encode(5)
    assert decode(private)["key_set"] is True
    monkeypatch.delenv("ITJ_MYSTERY_KEY")
    with pytest.raises(ValueError, match="ITJ_MYSTERY_KEY"):
        decode(private)


#: Entries written by itj 0.2.0 ("x1:", the key itself), for ``encode(5)`` with the public key and, with
#: ``ITJ_MYSTERY_KEY=123457``, for ``encode(5, {"containment_efficiency": 0.99})``.
X1_PUBLIC = (
    "x1:xGIsc4MCAWoEjqinXSbu7ZZVHbiORoz82snEk4I8E5ZHrUAxAooA9okvPoTfqgdXuR4PMiBLew74oAvDhqkNtz6EPttHr+5QaJAMIdr1aaJcKeOWw05L"
    "zFs="
)
X1_PRIVATE = (
    "x1:Y2b105WJb3OzZrrATRLag9xzfXgHD/Q4uuSg63kN6/v7Op7fYRMg24Ry8wVxeZhpa/fyOfBYaxu0NYFug3TMcQtvYZgOAEZ+RpKgvmCs1KMm0saRe6s5"
    "lxNnOoqiAMcaUPDOHeX09SuFWJT6UQYCWbaE9CiR4YD/UFK1qOVPi0wR7PeNBhXU/Q=="
)


def test_entries_of_earlier_releases_still_decode(monkeypatch):
    monkeypatch.delenv("ITJ_MYSTERY_KEY", raising=False)
    payload = decode(X1_PUBLIC)
    assert {payload["process"]: {payload["field"]: payload["value"]}} == mystery_overrides(5)
    assert payload["key_set"] is False
    with pytest.raises(ValueError, match="ITJ_MYSTERY_KEY"):
        decode(X1_PRIVATE)
    monkeypatch.setenv("ITJ_MYSTERY_KEY", "123457")
    private = decode(X1_PRIVATE)
    assert private["key_set"] is True
    assert private["statistics"] == {"containment_efficiency": 0.99}
    assert {private["process"]: {private["field"]: private["value"]}} == mystery_overrides(5)


def test_the_key_is_stretched(monkeypatch, caplog):
    """A new entry is scrambled with the PBKDF2-stretched key, so that every key a student tries costs a KDF."""
    import hashlib

    from itj.toy import mystery

    monkeypatch.setenv("ITJ_MYSTERY_KEY", "123457")
    text = encode(5, {"containment_efficiency": 0.99})
    assert text.startswith("x2:")
    raw = base64.b64decode(text[3:])
    plain = json.dumps(decode(text), sort_keys=True).encode()
    assert mystery._scramble(raw, 123457) == plain
    assert mystery._scramble(raw, 123457, stretch=False) != plain  # not the stream of the bare key
    stretched = hashlib.pbkdf2_hmac("sha256", b"123457", b"itj-mystery", mystery.KDF_ITERATIONS).hex()
    assert mystery._stretched(123457) == stretched
    assert mystery.KDF_ITERATIONS >= 100_000
    # a short private key is worth a warning, a long random one and the public default are not
    with caplog.at_level("WARNING", logger="itj.toy.mystery"):
        encode(5)
    assert "6 digits" in caplog.text
    assert "secrets.randbits(64)" in caplog.text
    caplog.clear()
    for key in (str(2**63 + 12345), "0"):
        monkeypatch.setenv("ITJ_MYSTERY_KEY", key)
        with caplog.at_level("WARNING", logger="itj.toy.mystery"):
            assert decode(encode(7))["key_set"] is (key != "0")
    assert caplog.text == ""


def test_hide():
    provenance = {"seed": 3, "mystery": "x1:abc", "merged": [{"path": "a", "provenance": {"mystery": "x1:def"}}, 4]}
    hidden = hide(provenance)
    assert hidden["mystery"] == HIDDEN
    assert hidden["merged"][0]["provenance"]["mystery"] == HIDDEN
    assert hidden["merged"][1] == 4
    assert provenance["mystery"] == "x1:abc"  # a copy


def test_mystery_file_differs_from_a_clean_file_in_one_process(tmp_path, monkeypatch, capsys):
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
    capsys.readouterr()
    assert main(["info", str(path), "--provenance"]) == 0
    printed = capsys.readouterr().out
    assert HIDDEN in printed
    assert meta["provenance"]["mystery"] not in printed
    # revealed on request
    revealed = reveal(path)
    assert field in revealed
    assert process in revealed
    assert reveal(data) == revealed


def test_reveal_of_a_clean_file(tmp_toy_file, toy_small):
    assert reveal(tmp_toy_file) == "this file has no planted shortcut"
    assert reveal(toy_small) == "this file has no planted shortcut"


def test_visible_provenance_of_a_mystery_file_is_that_of_a_clean_file(tmp_path, monkeypatch):
    """The run statistics that a shortcut changes (containment, capped kT, stage times) are scrambled too."""
    monkeypatch.delenv("ITJ_MYSTERY_KEY", raising=False)
    seed = 5  # hard_w: raises the W containment efficiency
    assert mystery_overrides(seed) == {"wqq": {"pt_min": 650.0}}
    clean_path, mystery_path = tmp_path / "clean.parquet", tmp_path / "mystery.parquet"
    assert main(["generate", "-n", "120", "--seed", str(seed), "-o", str(clean_path), "-q"]) == 0
    assert main(["generate", "--mystery", "-n", "120", "--seed", str(seed), "-o", str(mystery_path), "-q"]) == 0
    clean = read_metadata(clean_path)["provenance"]
    planted = read_metadata(mystery_path)["provenance"]
    volatile = {"command_line", "created", "wall_time_s", "mystery", "process_overrides"}
    for key in RUN_STATISTICS:
        assert planted[key] == HIDDEN
        assert clean[key] != HIDDEN
    differ = {key for key in (set(clean) | set(planted)) - volatile if clean.get(key) != planted.get(key)}
    assert differ == set(RUN_STATISTICS)
    payload = decode(planted["mystery"])
    assert set(payload["statistics"]) == set(RUN_STATISTICS)
    assert payload["statistics"]["containment_efficiency"]["wqq"] != clean["containment_efficiency"]["wqq"]
    assert payload["statistics"]["containment_efficiency"]["qcd"] == clean["containment_efficiency"]["qcd"]
    assert "hidden run statistics" in reveal(mystery_path)
    # the entry of an older file (no statistics) still reveals in one line
    assert "\n" not in reveal(JetData({"jet_pt": np.ones(1, np.float32)}, provenance={"mystery": encode(seed)}))
