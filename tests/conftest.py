"""Shared fixtures of the test suite (all simulations run on the CPU)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")

if TYPE_CHECKING:
    from itj.data import JetData

#: The committed JetClass sample: rows 0-49 of ZJetsToNuNu_120, WToQQ_120 and TTBar_120 (tests/data/README.md).
JETCLASS_SAMPLE = Path(__file__).parent / "data" / "jetclass_sample.parquet"


@pytest.fixture(scope="session")
def toy_small() -> JetData:
    """300 jets of the detector preset (qcd, wqq, tbqq)."""
    from itj.toy import generate

    return generate(300, seed=11)


@pytest.fixture(scope="session")
def ladder_small() -> dict[str, JetData]:
    """90 jets at every preset of the ladder (the same events)."""
    from itj.toy import PRESETS, generate, preset

    return {name: generate(90, config=preset(name), seed=11) for name in PRESETS}


@pytest.fixture(scope="session")
def qg_small() -> JetData:
    """200 quark and gluon jets at the shower preset."""
    from itj.toy import generate, preset

    return generate(200, config=preset("shower"), processes=("q", "g"), seed=13)


@pytest.fixture(scope="session")
def jetclass_sample() -> JetData:
    """The 150 committed JetClass validation jets (50 QCD, 50 W, 50 top)."""
    from itj.io import load

    return load(JETCLASS_SAMPLE)


@pytest.fixture(scope="session")
def tmp_toy_file(toy_small: JetData, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``toy_small`` saved as a Parquet file."""
    from itj.io import save

    return save(toy_small, tmp_path_factory.mktemp("toy") / "toy_small.parquet")
