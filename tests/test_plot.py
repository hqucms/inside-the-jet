"""Every figure of ``itj.plot`` draws headless (Agg) on synthetic jets, toy jets and the JetClass sample."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from itj.data import JetData, Ragged
from synthetic import massless, synthetic_jets

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from itj import plot  # noqa: E402


def toy_fixture(request: pytest.FixtureRequest, name: str) -> object:
    """Return a toy fixture of conftest, or skip while the generator is not there."""
    try:
        from itj.toy import generate  # noqa: F401
    except ImportError:
        pytest.skip("itj.toy.generate is not available yet")
    return request.getfixturevalue(name)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _with_truth(data: JetData, seed: int = 0) -> JetData:
    """Add the per-particle and per-prong truth columns a toy file has (random but consistent)."""
    rng = np.random.default_rng(seed)
    counts = data.counts
    offsets = data["part_px"].offsets
    n = int(counts.sum())
    prong = rng.integers(-1, 3, n).astype(np.int8)
    parton = np.where(prong < 0, -1, rng.integers(0, 6, n)).astype(np.int16)
    n_prongs = np.full(len(data), 3)
    deta = Ragged.from_counts(rng.normal(0, 0.2, n_prongs.sum()).astype(np.float32), n_prongs)
    dphi = Ragged(rng.normal(0, 0.2, n_prongs.sum()).astype(np.float32), deta.offsets)
    return data.with_columns(
        truth_part_prong=Ragged(prong, offsets),
        truth_part_parton=Ragged(parton, offsets),
        truth_part_is_ue=Ragged((prong < 0).astype(np.int8), offsets),
        truth_prong_deta=deta,
        truth_prong_dphi=dphi,
    )


@pytest.fixture(scope="module")
def jets() -> JetData:
    rng = np.random.default_rng(1)
    particles = []
    for _ in range(6):
        k = int(rng.integers(5, 40))
        pt = np.sort(rng.exponential(30.0, k) + 1.0)[::-1]
        particles.append(
            np.stack(
                [massless(p, e, f) for p, e, f in zip(pt, rng.normal(0, 0.3, k), rng.normal(0, 0.3, k), strict=True)]
            )
        )
    data = synthetic_jets(particles, ["QCD", "Wqq", "Tbqq"] * 2)
    flags = data["part_isPhoton"].map(lambda v: (np.arange(len(v)) % 3 == 0).astype(np.int32))
    charged = data["part_isChargedHadron"].map(lambda v: (np.arange(len(v)) % 3 == 1).astype(np.int32))
    return _with_truth(data.with_columns(part_isPhoton=flags, part_isChargedHadron=charged))


@pytest.mark.parametrize("color", ["type", "prong", "parton", "ue"])
def test_jet_display_colourings(jets: JetData, color: str) -> None:
    ax = plot.jet_display(jets, 1, color=color)
    assert ax.get_aspect() == 1.0
    assert ax.get_xlim() == (-0.8, 0.8)
    assert ax.get_legend() is not None


def test_jet_display_values_and_options(jets: JetData) -> None:
    n = int(jets.counts[2])
    values = np.linspace(-1, 1, n + 5)  # a padded row is accepted
    ax = plot.jet_display(jets, 2, color=values, size="energy", lim=0.5, label="IG", show_prongs=False)
    assert ax.get_xlim() == (-0.5, 0.5)
    assert len(ax.figure.axes) == 2  # the colour bar
    with pytest.raises(ValueError, match="truth column"):
        plot.jet_display(jets.drop("truth_"), 0, color="prong")
    with pytest.raises(ValueError, match="color must be"):
        plot.jet_display(jets, 0, color="pid")  # type: ignore[arg-type]
    # no PID flags: one colour
    plain = jets.drop("part_isPhoton", "part_isChargedHadron")
    plot.jet_display(plain, 0)


def test_jet_display_axes(jets: JetData) -> None:
    pytest.importorskip("itj.observables")
    plot.jet_display(jets, 2, show_axes=3)


def test_ladder_and_animation(jets: JetData, tmp_path: Path) -> None:
    levels = {"partons": jets.drop("truth_"), "shower": jets, "detector": jets}
    fig = plot.ladder(levels, i=1)
    assert len(fig.axes) == 3
    pytest.importorskip("PIL")
    path = plot.animate_ladder(levels, tmp_path / "sub" / "ladder.gif", width=240, fade=0.2, fps=10)
    from PIL import Image

    with Image.open(path) as im:
        assert im.n_frames == 3 * (1 + 2)
    assert path.stat().st_size < 2_900_000


def test_ladder_on_toy(request: pytest.FixtureRequest) -> None:
    ladder_small = toy_fixture(request, "ladder_small")
    assert isinstance(ladder_small, dict)
    fig = plot.ladder(ladder_small)
    assert len(fig.axes) == len(ladder_small)


def test_jetclass_displays(jetclass_sample: JetData) -> None:
    ax = plot.jet_display(jetclass_sample, 60, show_axes=2)
    assert "wqq" in ax.get_title()
    plot.jet_image(jetclass_sample)


def test_toy_displays(request: pytest.FixtureRequest) -> None:
    toy_small = toy_fixture(request, "toy_small")
    assert isinstance(toy_small, JetData)
    for color in ("type", "prong", "parton", "ue"):
        plot.jet_display(toy_small, 0, color=color)  # type: ignore[arg-type]


def test_jet_image(jets: JetData) -> None:
    ax = plot.jet_image(jets, bins=17, extent=0.6, average=False, log=False)
    assert ax.get_aspect() == 1.0
    plot.jet_image(jets)


def test_lund_plane() -> None:
    x = np.linspace(0, 5, 11)
    y = np.linspace(-3, 6, 10)
    rho = np.random.default_rng(0).random((10, 9))
    ax = plot.lund_plane(rho, x, y, kt_cut=1.0, n_jets=100)
    assert "100 jets" in ax.get_title()
    with pytest.raises(ValueError, match="shape"):
        plot.lund_plane(rho.T, x, y)


def test_roc_curves() -> None:
    eff_s = np.linspace(0.01, 1, 50)
    ax = plot.roc_curves({"a": (eff_s, eff_s**2), "b": (eff_s, eff_s**3)}, title="W vs QCD")
    assert ax.get_yscale() == "log"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "R_{50}" in labels[0]
    assert "4" in labels[0]  # 1 / 0.5^2


def test_curves_and_history() -> None:
    x = np.logspace(-6, -1, 6)
    ax = plot.curves(x, {"EFN": x * 0, "PFN": (x, x / 2)}, xlabel="eps", ylabel="|ds|", logx=True)
    assert ax.get_xscale() == "log"
    with pytest.raises(ValueError, match="shape"):
        plot.curves(x, {"bad": x[:3]}, xlabel="x", ylabel="y")

    @dataclass
    class History:
        train_loss: list[float] = field(default_factory=lambda: [1.0, 0.7, 0.6])
        val_loss: list[float] = field(default_factory=lambda: [1.1, 0.8, 0.75])
        train_acc: list[float] = field(default_factory=lambda: [0.5, 0.7, 0.75])
        val_acc: list[float] = field(default_factory=lambda: [0.5, 0.68, 0.7])
        best_epoch: int = 2

    ax = plot.history(History())
    assert ax.get_legend() is not None


def test_attention_and_attribution(jets: JetData) -> None:
    rng = np.random.default_rng(0)
    a = rng.random((7, 7))
    a /= a.sum(1, keepdims=True)
    groups = np.array([1, 0, 2, 0, 1, -1, 2])
    plot.attention_matrix(a, groups=groups, title="block 0, head 1")
    plot.attention_matrix(a, order=np.arange(7)[::-1])
    with pytest.raises(ValueError, match="square"):
        plot.attention_matrix(a[:3])
    attr = rng.normal(size=(len(jets), 64))
    ax = plot.attribution_display(jets, 3, attr)
    assert any("signed" in a.get_ylabel() for a in ax.figure.axes)
    plot.attribution_display(jets, 3, attr[3], absolute=True)


def test_efn_filters() -> None:
    class FakeEFN:
        def filters(self, *, extent: float = 0.8, bins: int = 64) -> np.ndarray:
            g = np.linspace(-extent, extent, bins)
            r = np.hypot(*np.meshgrid(g, g, indexing="ij"))
            return np.stack([np.exp(-((r - c) ** 2) / 0.01) for c in np.linspace(0, 0.6, 10)])

    fig = plot.efn_filters(FakeEFN(), n=6)
    assert sum(ax.get_visible() for ax in fig.axes) == 6


def test_cka_heatmap() -> None:
    m = np.random.default_rng(0).random((3, 4))
    ax = plot.cka_heatmap(m, ["a", "b", "c"], ["w", "x", "y", "z"])
    assert len(ax.texts) == 12
    with pytest.raises(ValueError, match="shape"):
        plot.cka_heatmap(m, ["a"], ["w"])


def test_palette() -> None:
    for key in ("qcd", "quark", "gluon", "wqq", "tbqq", "prong0", "prong1", "prong2", "ue", "charged", "neutral",
                "photon", "lepton"):  # fmt: skip
        assert plot.palette[key].startswith("#")
