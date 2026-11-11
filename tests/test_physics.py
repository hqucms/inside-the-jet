"""Slow physics validation: the validation table, the speed of every preset and fresh-process reproducibility.

Run with ``pytest -m slow tests/test_physics.py -s``. Every two-sided row of the validation table passes within
max(5 sigma, 3 %) of the value calibrated against JetClass (``itj.toy.validate.ROWS``) and inside its acceptance range.
"""

import hashlib
import subprocess
import sys
import time

import numpy as np
import pytest

from itj.data import Ragged
from itj.toy import PRESETS, format_validation, generate, preset, validate
from itj.toy.validate import REL_WINDOW, ROWS

pytestmark = pytest.mark.slow

#: Rows that test the physics of the toy (leading-log formulas, containment, the toy's own estimates).
PHYSICS_ROWS = (
    "lund_q",
    "lund_g",
    "lund_ratio",
    "n_primary_q",
    "mass_peak_q",
    "mass_peak_g",
    "ng_nq_shower",
    "ng_nq_hadrons",
    "ng_nq_detector",
    "containment_w",
    "containment_top",
    "d2_ratio",
    "hadron_cap_fraction",
    "speed_detector",
)


@pytest.fixture(scope="module")
def rows():
    rows = validate(n=10_000, seed=0, workers="auto")
    return {row.name: row for row in rows}


def test_print_validation_table(rows, capsys):
    with capsys.disabled():
        print("\n" + format_validation(list(rows.values())))
    assert len(rows) == 29


def test_physics_rows(rows):
    failed = [
        f"{n}: {rows[n].value:.4g} not in [{rows[n].low}, {rows[n].high}]" for n in PHYSICS_ROWS if not rows[n].ok
    ]
    assert not failed, failed


def test_jetclass_rows(rows):
    failed = [
        f"{n}: {r.value:.4g} not in [{r.low}, {r.high}]" for n, r in rows.items() if n not in PHYSICS_ROWS and not r.ok
    ]
    assert not failed, failed


def test_windows_inside_acceptance_ranges():
    for name, (_, _, calibrated, low, high) in ROWS.items():
        if np.isnan(calibrated):
            continue
        # the calibrated value +- 3 % lies inside the acceptance range (the 5 sigma part depends on the run)
        width = REL_WINDOW * abs(calibrated)
        assert low <= calibrated - width, name
        assert calibrated + width <= high, name


def test_speed_of_every_preset(capsys):
    n = 2000
    rates = {}
    for name in PRESETS:
        generate(100, config=preset(name), seed=1)  # warm up
        start = time.perf_counter()
        data = generate(n, config=preset(name), seed=1)
        rates[name] = n / (time.perf_counter() - start)
        stages = data.provenance["stage_times_s"]
        with capsys.disabled():
            share = ", ".join(f"{k} {v / sum(stages.values()):.0%}" for k, v in stages.items() if v > 0)
            print(f"\n{name:<9} {rates[name]:7.0f} jets/s/core  ({share})", end="")
    with capsys.disabled():
        print()
    assert rates["detector"] >= 100.0
    assert rates["partons"] >= 1000.0


def _hashes(data):
    out = {}
    for name in data.names:
        column = data[name]
        values = column.values if isinstance(column, Ragged) else column
        h = hashlib.sha256(np.ascontiguousarray(values).tobytes())
        if isinstance(column, Ragged):
            h.update(np.ascontiguousarray(column.offsets).tobytes())
        out[name] = h.hexdigest()
    return out


_SCRIPT = """
import hashlib, json, numpy as np
from itj.data import Ragged
from itj.toy import generate
data = generate(1200, seed=3)
out = {}
for name in data.names:
    column = data[name]
    values = column.values if isinstance(column, Ragged) else column
    h = hashlib.sha256(np.ascontiguousarray(values).tobytes())
    if isinstance(column, Ragged):
        h.update(np.ascontiguousarray(column.offsets).tobytes())
    out[name] = h.hexdigest()
print(json.dumps(out))
"""


def test_fresh_process_reproducibility():
    import json

    runs = [
        json.loads(subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=True).stdout)
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    assert runs[0] == _hashes(generate(1200, seed=3))
    assert len(runs[0]) == 90
