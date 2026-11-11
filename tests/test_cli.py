"""The command line jetgen: every subcommand, the config flags, default names and the error format."""

import subprocess
import sys

import numpy as np
import pytest

from itj.io import load, read_metadata
from itj.toy import TOY_NAME, config_fields, preset
from itj.toy.cli import build_parser, default_output, main, parse_override


def _help(capsys, *argv):
    with pytest.raises(SystemExit) as info:
        main([*argv, "--help"])
    assert info.value.code == 0
    return capsys.readouterr().out


def test_help_lists_every_config_flag_and_group(capsys):
    text = _help(capsys, "generate")
    for _, name, f in config_fields():
        flag = "--" + name.replace("_", "-")
        assert flag in text, flag
        if isinstance(f.default, bool):
            assert "--no-" + name.replace("_", "-") in text
    for title in ("hard process", "shower", "hadronisation", "underlying event", "detector", "jet"):
        assert f"{title} options" in text
    assert "(on in: hadrons, detector)" in " ".join(text.split())
    assert "(detector: 0.77)" in text
    for flag in ("-n N, --n-jets N", "--override", "--mystery", "--workers", "--first-event-id", "--no-truth"):
        assert flag in text
    top = _help(capsys)
    assert TOY_NAME in " ".join(top.split())
    for command in ("presets", "generate", "info", "display", "merge", "split", "configs", "fetch", "validate"):
        assert command in top


def test_every_subcommand_has_verbosity_flags(capsys):
    for argv in (["presets"], ["info", "x"], ["merge", "x", "-o", "y"], ["split", "x", "--out-dir", "d"], ["configs"],
                 ["fetch", "jetclass"], ["fetch", "models"], ["validate"], ["display"]):  # fmt: skip
        text = _help(capsys, *argv[:1] if argv[0] != "fetch" else argv[:2])
        assert "-v, --verbose" in text
        assert "-q, --quiet" in text


def test_presets(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    for name in ("partons", "shower", "hadrons", "detector"):
        assert name in out


def test_generate_overrides_reach_the_stored_config(tmp_path, capsys):
    path = tmp_path / "out.parquet"
    argv = ["generate", "-n", "40", "--seed", "3", "--preset", "hadrons", "--no-ue", "--pi0-fraction", "0.2"]
    argv += ["--qcd-flavour-weights", "1", "1", "1", "1", "1", "--process", "q,g", "-o", str(path), "--no-progress"]
    assert main(argv) == 0
    err = capsys.readouterr().err
    assert "wrote" in err
    assert "ms/jet" in err
    meta = read_metadata(path)
    expected = preset("hadrons", ue=False, pi0_fraction=0.2, qcd_flavour_weights=(1, 1, 1, 1, 1))
    assert meta["config"] == expected.to_dict()
    provenance = meta["provenance"]
    assert provenance["command_line"].startswith("jetgen generate -n 40")
    assert provenance["processes"] == ["q", "g"]
    assert provenance["overrides"] == {"ue": False, "pi0_fraction": 0.2, "qcd_flavour_weights": [1.0] * 5}
    assert provenance["stage_times_s"]["write"] > 0
    assert meta["n_jets"] == 40
    # refuses to overwrite
    assert main(argv) == 1
    assert "use --overwrite" in capsys.readouterr().err
    assert main([*argv, "--overwrite", "-q"]) == 0


def test_generate_default_name_override_and_no_truth(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["generate", "--n-jets", "30", "--seed", "2", "-q"]) == 0
    assert (tmp_path / "data" / "toy_detector_qcd-wqq-tbqq_n30_seed2.parquet").exists()
    argv = ["generate", "-n", "30", "--seed", "2", "--override", "wqq:pi0_fraction=0.4", "--no-truth", "-q"]
    assert main(argv) == 0
    (custom,) = [p for p in (tmp_path / "data").iterdir() if "custom" in p.name]
    assert custom.name.startswith("toy_detector-custom-")
    data = load(custom)
    assert not data.has_truth
    assert read_metadata(custom)["provenance"]["process_overrides"] == {"wqq": {"pi0_fraction": 0.4}}


def test_default_output():
    assert default_output(preset("shower"), ["q", "g"], 2000, 3).as_posix() == "data/toy_shower_q-g_n2000_seed3.parquet"
    custom = default_output(preset("shower", alphas_mode="fixed"), ["q"], 10, 0).name
    assert custom.startswith("toy_shower-custom-")
    assert len(custom.split("-custom-")[1].split("_")[0]) == 6
    a = default_output(preset(), ["wqq"], 10, 0, process_overrides={"wqq": {"ue": False}})
    assert "custom" in a.name


def test_parse_override():
    assert parse_override("wqq:pi0_fraction=0.40") == ("wqq", "pi0_fraction", 0.4)
    assert parse_override("tbqq:cell-position=centre") == ("tbqq", "cell_position", "centre")
    assert parse_override("qcd:ue=off") == ("qcd", "ue", False)
    assert parse_override("g:n_flavours=4") == ("g", "n_flavours", 4)
    assert parse_override("qcd:qcd_flavour_weights=1,1,1,1,0") == (
        "qcd",
        "qcd_flavour_weights",
        (1.0, 1.0, 1.0, 1.0, 0.0),
    )
    assert parse_override("qcd:jet_pt_min=none") == ("qcd", "jet_pt_min", None)
    for bad, match in (("wqq-pi0=1", "PROC:KEY=VALUE"), ("w:ue=1", "unknown process"), ("wqq:ue=maybe", "switch")):
        with pytest.raises(ValueError, match=match):
            parse_override(bad)


def test_info(tmp_toy_file, capsys):
    assert main(["info", str(tmp_toy_file), "--columns", "--provenance", "--reveal"]) == 0
    out = capsys.readouterr().out
    assert "300 jets" in out
    assert "part_d0val" in out
    assert "stage_times_s" in out
    assert "this file has no planted shortcut" in out


def test_merge_and_split(tmp_path, capsys):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    assert main(["generate", "-n", "20", "--seed", "1", "-o", str(a), "-q"]) == 0
    assert main(["generate", "-n", "20", "--seed", "2", "--first-event-id", "20", "-o", str(b), "-q"]) == 0
    merged = tmp_path / "ab.parquet"
    assert main(["merge", str(a), str(b), "-o", str(merged), "--shuffle", "--seed", "4"]) == 0
    data = load(merged)
    assert len(data) == 40
    assert sorted(data["truth_event_id"].tolist()) == list(range(40))
    out_dir = tmp_path / "split"
    assert main(["split", str(merged), "--out-dir", str(out_dir), "--test", "0.25", "--val", "0.25"]) == 0
    sizes = {part: len(load(out_dir / part / "ab.parquet")) for part in ("train", "val", "test")}
    assert sizes == {"train": 20, "val": 10, "test": 10}
    test = load(out_dir / "test" / "ab.parquet")
    np.testing.assert_array_equal(test["truth_event_id"], data["truth_event_id"][:10])
    assert main(["merge", str(a), str(a), "-o", str(tmp_path / "aa.parquet")]) == 1
    assert "jetgen merge: error:" in capsys.readouterr().err


def test_configs(capsys):
    assert main(["configs"]) == 0
    lines = capsys.readouterr().out.split()
    assert len(lines) == 6
    assert any(line.endswith("jets3_kin.yaml") for line in lines)


def test_fetch_dry_run(tmp_path, capsys):
    assert main(["fetch", "jetclass", "--dry-run", "--dest", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ZJetsToNuNu_120.parquet" in out
    assert "TTBar_120.parquet" in out
    assert not any(tmp_path.iterdir())
    assert main(["fetch", "models", "ParT_kin", "--dry-run", "--dest", str(tmp_path)]) == 0
    assert "ParT_kin.pt" in capsys.readouterr().out
    assert main(["fetch", "jetclass", "--classes", "QCDD", "--dry-run"]) == 1
    assert "jetgen fetch jetclass: error:" in capsys.readouterr().err


def test_validate_small(capsys):
    assert main(["validate", "-n", "200", "--only", "containment,lund_q", "--only", "n_primary"]) == 0
    out = capsys.readouterr().out
    assert "containment_w" in out
    assert "lund_q" in out
    assert "n_primary_q" in out
    assert "tau21" not in out
    assert main(["validate", "--only", "nonsense"]) == 1
    assert "jetgen validate: error:" in capsys.readouterr().err


def test_errors_use_the_fnt_format(tmp_path, capsys):
    assert main(["info", str(tmp_path / "missing.parquet")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("jetgen info: error:")
    assert main(["generate", "-n", "5", "--override", "wqq:pi0_fractoin=0.3", "-o", str(tmp_path / "x.parquet")]) == 1
    assert "Did you mean 'pi0_fraction'" in capsys.readouterr().err
    assert main(["generate", "-n", "5", "--pi0-fraction", "2", "-o", str(tmp_path / "x.parquet")]) == 1
    assert "jetgen generate: error: pi0_fraction" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["generate", "--process", "qcd,wq"])
    assert "unknown process" in capsys.readouterr().err
    assert main(["display"]) == 1
    assert "give either a FILE or --ladder" in capsys.readouterr().err


def test_display(tmp_toy_file, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("itj.plot")
    fig = tmp_path / "jet.png"
    assert main(["display", str(tmp_toy_file), "--jet", "3", "--color", "prong", "--lund", "-o", str(fig)]) == 0
    assert fig.stat().st_size > 1000
    ladder = tmp_path / "ladder.png"
    assert main(["display", "--ladder", "--process", "tbqq", "--seed", "1", "-o", str(ladder)]) == 0
    assert ladder.stat().st_size > 1000
    assert main(["display", str(tmp_toy_file), "--jet", "1000"]) == 1


def test_display_gif(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    pytest.importorskip("itj.plot")
    gif = tmp_path / "ladder.gif"
    assert main(["display", "--ladder", "-o", str(gif)]) == 0
    assert gif.read_bytes()[:3] == b"GIF"


def test_python_m_itj_toy():
    result = subprocess.run([sys.executable, "-m", "itj.toy", "--help"], capture_output=True, text=True, check=True)
    assert "jetgen" in result.stdout
    assert build_parser().prog == "jetgen"
