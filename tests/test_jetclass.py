from __future__ import annotations

import functools
import hashlib
import http.server
import re
import threading
import urllib.request
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from itj import jetclass
from itj.data import JETCLASS_LABELS
from itj.jetclass import (
    CLASSES,
    JETS3_INDEX,
    MANIFEST,
    MODELS,
    FileInfo,
    class_names,
    fetch,
    fetch_models,
    files,
    load_jetclass,
    probs_to_jets3,
)


def test_classes():
    assert [c.label for c in CLASSES] == list(JETCLASS_LABELS)
    assert [c.index for c in CLASSES] == list(range(10))
    assert [CLASSES[i].short for i in JETS3_INDEX] == ["QCD", "Wqq", "Tbqq"]
    assert {c.short: c.aux_pid for c in CLASSES}["Tbqq"] == 6
    assert {c.short: c.truth_match for c in CLASSES}["Zqq"] == (31, 32, 33)


def test_class_names():
    assert class_names(["qcd", "wqq", "TBQQ"]) == ["QCD", "Wqq", "Tbqq"]
    assert class_names(["ZJetsToNuNu", "Wqq", "wqq"]) == ["QCD", "Wqq"]
    assert class_names(("all",)) == [c.short for c in CLASSES]
    assert class_names("Hbb") == ["Hbb"]
    with pytest.raises(ValueError, match="did you mean 'Tbqq'"):
        class_names(["Tbbq"])


def test_manifest_consistency():
    keys = set(MANIFEST)
    assert keys == {(s, c.short) for s in ("train", "val") for c in CLASSES} | {("example", "all")}
    assert len(MANIFEST) == 21
    filenames = [info.filename for info in MANIFEST.values()]
    assert len(set(filenames)) == len(filenames)
    assert len({info.url for info in MANIFEST.values()}) == len(MANIFEST)
    for (split, short), info in MANIFEST.items():
        assert re.fullmatch(r"[0-9a-f]{32}", info.md5), info
        assert info.url.endswith("/" + info.filename)
        assert info.url.startswith("https://hqu.web.cern.ch/datasets/")
        if split == "example":
            assert 130e6 < info.size < 140e6
            continue
        prefix = next(c.prefix for c in CLASSES if c.short == short)
        number = {"train": "000", "val": "120"}[split]
        assert info.filename == f"{prefix}_{number}.parquet"
        assert f"/JetClassMini/{split}_1M_parquet/" in info.url
        assert 80e6 < info.size < 180e6
    core = sum(MANIFEST["val", c].size for c in ("QCD", "Wqq", "Tbqq"))
    assert round(core / 1e6, 1) == 374.5
    assert len({info.md5 for info in MANIFEST.values()}) == len(MANIFEST)


def test_models_manifest():
    assert list(MODELS) == [
        "ParT_kin",
        "ParT_kinpid",
        "ParT_full",
        "ParticleNet_kin",
        "ParticleNet_kinpid",
        "ParticleNet_full",
    ]
    for name, info in MODELS.items():
        assert info.filename == f"{name}.pt"
        assert "fa63c86d594ef875c2d25c04bd5afbab5c73b611" in info.url
        assert re.fullmatch(r"[0-9a-f]{32}", info.md5)
    assert round(sum(m.size for m in MODELS.values()) / 1e6) == 31


def test_probs_to_jets3():
    probs = np.zeros((2, 10))
    probs[0, [0, 7, 8]] = [0.1, 0.2, 0.1]
    probs[0, 1] = 0.6
    probs[1, 3] = 1.0  # no probability on the three classes
    out = probs_to_jets3(probs)
    np.testing.assert_allclose(out, [[0.25, 0.5, 0.25], [1 / 3, 1 / 3, 1 / 3]])
    with pytest.raises(ValueError, match="10"):
        probs_to_jets3(np.ones((3, 3)))


# --------------------------------------------------------------------------------------------- downloads (offline)


@pytest.fixture
def share(tmp_path, monkeypatch):
    """A teacher's share (file:// URL) with three small fake val files and the manifest pinned to them."""
    root = tmp_path / "share"
    (root / "val").mkdir(parents=True)
    fake = dict(MANIFEST)
    for k, short in enumerate(("QCD", "Wqq", "Tbqq")):
        info = MANIFEST["val", short]
        content = (f"{short} " * (1000 + 1000 * k)).encode()
        (root / "val" / info.filename).write_bytes(content)
        fake["val", short] = FileInfo(info.url, info.filename, len(content), hashlib.md5(content).hexdigest())
    monkeypatch.setattr(jetclass, "MANIFEST", fake)
    monkeypatch.delenv("ITJ_JETCLASS_URL", raising=False)
    return root


def test_fetch_from_a_file_url(share, tmp_path, capsys):
    dest = tmp_path / "data"
    paths = fetch("val", ("qcd", "Wqq", "TTBar"), dest=dest, base_url=share.as_uri(), progress=False)
    assert paths == [dest / "val" / n for n in ("ZJetsToNuNu_120.parquet", "WToQQ_120.parquet", "TTBar_120.parquet")]
    for path in paths:
        assert path.read_bytes() == (share / "val" / path.name).read_bytes()
        assert not path.with_name(path.name + ".part").exists()
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("JetClass val from file://")
    assert "3 files" in out
    # a matching file is kept (not downloaded again)
    before = paths[0].stat().st_mtime_ns
    fetch("val", ("QCD",), dest=dest, base_url=share.as_uri(), progress=False)
    assert paths[0].stat().st_mtime_ns == before
    assert "kept" in capsys.readouterr().out
    # a damaged file is downloaded again; overwrite always downloads
    paths[1].write_bytes(b"damaged")
    fetch("val", ("Wqq",), dest=dest, base_url=share.as_uri(), progress=True)
    assert paths[1].read_bytes() == (share / "val" / paths[1].name).read_bytes()
    fetch("val", ("QCD",), dest=dest, base_url=share.as_uri(), overwrite=True, progress=False)
    assert "kept" not in capsys.readouterr().out


def test_fetch_environment_variable_and_resume(share, tmp_path, monkeypatch):
    monkeypatch.setenv("ITJ_JETCLASS_URL", share.as_uri() + "/")
    dest = tmp_path / "data"
    target = dest / "val" / "WToQQ_120.parquet"
    target.parent.mkdir(parents=True)
    content = (share / "val" / "WToQQ_120.parquet").read_bytes()
    target.with_name(target.name + ".part").write_bytes(content[:100])  # an interrupted download
    assert fetch("val", ("Wqq",), dest=dest, progress=False) == [target]
    assert target.read_bytes() == content  # file:// has no ranges: downloaded again from the start


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """A file server that answers ``Range: bytes=N-`` requests with 206 (like the CERN mirror)."""

    requests: ClassVar[list[str | None]] = []

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        header = self.headers.get("Range")
        type(self).requests.append(header)
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        content = path.read_bytes()
        start = int(header.removeprefix("bytes=").rstrip("-")) if header else 0
        self.send_response(206 if header else 200)
        self.send_header("Content-Length", str(len(content) - start))
        self.end_headers()
        self.wfile.write(content[start:])


@pytest.fixture
def http_share(share):
    """The share of ``share`` served over HTTP with range support."""
    handler = functools.partial(_RangeHandler, directory=str(share))
    _RangeHandler.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join()


def test_fetch_resumes_with_a_range_request(http_share, share, tmp_path):
    dest = tmp_path / "data"
    target = dest / "val" / "TTBar_120.parquet"
    target.parent.mkdir(parents=True)
    content = (share / "val" / target.name).read_bytes()
    target.with_name(target.name + ".part").write_bytes(content[:1234])  # an interrupted download
    assert fetch("val", ("Tbqq",), dest=dest, base_url=http_share, progress=False) == [target]
    assert target.read_bytes() == content
    assert _RangeHandler.requests == ["bytes=1234-"]
    assert not target.with_name(target.name + ".part").exists()


def test_fetch_refuses_a_wrong_md5(share, tmp_path, monkeypatch):
    info = jetclass.MANIFEST["val", "QCD"]
    monkeypatch.setitem(jetclass.MANIFEST, ("val", "QCD"), FileInfo(info.url, info.filename, info.size, "0" * 32))
    dest = tmp_path / "data"
    with pytest.raises(OSError, match="MD5"):
        fetch("val", ("QCD",), dest=dest, base_url=share.as_uri(), progress=False)
    assert not (dest / "val" / info.filename).exists()
    assert not (dest / "val" / (info.filename + ".part")).exists()
    monkeypatch.setitem(jetclass.MANIFEST, ("val", "QCD"), FileInfo(info.url, info.filename, info.size + 1, info.md5))
    with pytest.raises(OSError, match="bytes"):
        fetch("val", ("QCD",), dest=dest, base_url=share.as_uri(), progress=False)
    with pytest.raises(OSError, match="could not download"):
        fetch("val", ("Hbb",), dest=dest, base_url=share.as_uri(), progress=False)


def test_fetch_dry_run_and_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ITJ_JETCLASS_URL", raising=False)
    paths = fetch("val", dest=tmp_path, dry_run=True)
    assert [p.name for p in paths] == ["ZJetsToNuNu_120.parquet", "WToQQ_120.parquet", "TTBar_120.parquet"]
    assert not any(p.exists() for p in paths)
    out = capsys.readouterr().out
    assert "JetClass val: 3 files, 374.5 MB" in out
    assert MANIFEST["val", "Wqq"].url in out
    example = fetch("example", ("ignored",), dest=tmp_path, dry_run=True)
    assert example == [tmp_path / "example" / "JetClass_example_100k.root"]
    assert len(fetch("train", ("all",), dest=tmp_path, dry_run=True)) == 10
    models = fetch_models(["ParT_kin", "ParticleNet_kin"], dest=tmp_path / "models", dry_run=True)
    assert [p.name for p in models] == ["ParT_kin.pt", "ParticleNet_kin.pt"]
    assert len(fetch_models(dry_run=True, dest=tmp_path)) == 6
    with pytest.raises(ValueError, match="did you mean 'ParT_kin'"):
        fetch_models(["ParT_kn"], dry_run=True)
    with pytest.raises(ValueError, match="split"):
        fetch("test", dry_run=True)  # type: ignore[arg-type]


def test_files_and_load_jetclass(tmp_path):
    from conftest import JETCLASS_SAMPLE
    from itj.io import load, save

    with pytest.raises(FileNotFoundError, match="jetgen fetch jetclass --split val --classes QCD,Wqq,Tbqq"):
        files("val", root=tmp_path / "nothing")
    with pytest.raises(FileNotFoundError, match=re.escape(f"--dest {tmp_path / 'nothing'}")):
        files("val", root=tmp_path / "nothing")
    with pytest.raises(FileNotFoundError, match=r"fetch jetclass --split example --dest"):
        files("example", root=tmp_path / "nothing")
    sample = load(JETCLASS_SAMPLE)
    root = tmp_path / "jetclass"
    for k, short in enumerate(("QCD", "Wqq", "Tbqq")):
        save(sample.select(np.arange(50 * k, 50 * k + 50)), root / "val" / MANIFEST["val", short].filename)
    assert files("val", ("tbqq",), root=root) == [root / "val" / "TTBar_120.parquet"]
    data = load_jetclass("val", root=root)
    assert len(data) == 150
    assert data.source == "jetclass"
    assert data["file_index"].tolist() == [0] * 50 + [1] * 50 + [2] * 50  # type: ignore[union-attr]
    test = load_jetclass("val", ("Tbqq", "QCD"), part="test", max_jets_per_class=4, columns=["jet_pt"], root=root)
    assert test.names == ("jet_pt", "file_index")
    assert len(test) == 8
    np.testing.assert_array_equal(test["jet_pt"][:4], sample["jet_pt"][100:104])


def test_load_jetclass_example(tmp_path):
    uproot = pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    from conftest import JETCLASS_SAMPLE
    from itj.data import JETCLASS_BRANCHES
    from itj.io import load

    sample = load(JETCLASS_SAMPLE)
    path = tmp_path / "jetclass" / "example" / "JetClass_example_100k.root"
    path.parent.mkdir(parents=True)
    with uproot.recreate(path) as f:
        f["tree"] = {name: np.asarray(sample[name]) for name in JETCLASS_BRANCHES if not name.startswith("part_")}
    data = load_jetclass("example", ("Wqq", "Tbqq"), max_jets_per_class=3, root=tmp_path / "jetclass")
    assert data.process_names.tolist() == ["wqq"] * 3 + ["tbqq"] * 3
    only = load_jetclass("example", ("QCD",), columns=["jet_pt"], root=tmp_path / "jetclass")
    assert only.names == ("jet_pt",)
    assert len(only) == 50
    # the file holds the classes in blocks: the part is taken within each class, not of the whole file
    test = load_jetclass("example", ("all",), part="test", root=tmp_path / "jetclass")
    counts = {name: int((test.process_names == name).sum()) for name in set(test.process_names.tolist())}
    assert len(counts) == len(set(sample.process_names.tolist()))
    assert set(counts.values()) == {10}
    qcd_test = load_jetclass("example", ("QCD",), part="test", columns=["jet_pt"], root=tmp_path / "jetclass")
    np.testing.assert_array_equal(qcd_test["jet_pt"], only["jet_pt"][:10])
    with pytest.raises(ValueError, match="no jets of the classes"):
        load_jetclass("example", ("QCD",), part="test", max_jets_per_class=0, root=tmp_path / "jetclass")


# ------------------------------------------------------------------------------------------------------- network


def _content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "itj"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


@pytest.mark.network
@pytest.mark.parametrize("key", sorted(MANIFEST), ids=lambda k: f"{k[0]}-{k[1]}")
def test_mirror_sizes(key):
    info = MANIFEST[key]
    assert _content_length(info.url) == info.size


@pytest.mark.network
@pytest.mark.parametrize("name", list(MODELS))
def test_checkpoint_sizes(name):
    assert _content_length(MODELS[name].url) == MODELS[name].size


@pytest.mark.network
def test_checkpoint_download(tmp_path):
    (path,) = fetch_models(["ParticleNet_kin"], dest=tmp_path, progress=False)
    assert path == tmp_path / "ParticleNet_kin.pt"
    assert path.stat().st_size == MODELS["ParticleNet_kin"].size
    assert hashlib.md5(Path(path).read_bytes()).hexdigest() == MODELS["ParticleNet_kin"].md5
