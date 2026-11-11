"""Statistical helpers and common checks of the test suite."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

#: Tolerance of the statistical checks in standard deviations (the tests must pass for any seed and platform: a
#: correct implementation fails a 5 sigma check with probability 6e-7 per check).
N_SIGMA = 5.0

#: sqrt(n) D of the Kolmogorov-Smirnov test exceeds this value with probability below 0.1 % for a correct pdf
#: (the asymptotic Kolmogorov distribution: P(K > 1.95) = 0.001).
KS_LIMIT = 1.95


def assert_binomial(k: float, n: float, p: float, n_sigma: float = N_SIGMA) -> None:
    """Assert that ``k`` successes in ``n`` independent trials are compatible with the probability ``p``."""
    sigma = math.sqrt(p * (1 - p) / n)
    assert abs(k / n - p) < n_sigma * sigma, f"fraction {k / n:.5f}, expected {p:.5f} (sigma {sigma:.2g})"


def ks_distance(samples: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    """Kolmogorov-Smirnov distance ``sqrt(n) D`` between samples and a cumulative distribution function."""
    x = np.sort(np.asarray(samples, dtype=np.float64))
    n = len(x)
    f = cdf(x)
    d = max(np.max(np.arange(1, n + 1) / n - f), np.max(f - np.arange(n) / n))
    return math.sqrt(n) * float(d)


@dataclass
class RSSGrowth:
    """Growth of the peak resident set size inside a :func:`rss_growth` block, in bytes."""

    bytes: int = 0

    @property
    def gb(self) -> float:
        return self.bytes / 1e9


def _peak_rss() -> int:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # bytes on macOS, kB on Linux


@contextmanager
def rss_growth() -> Iterator[RSSGrowth]:
    """Measure how much the peak RSS of the process grows inside the block (skips the test on Windows).

    The peak RSS never decreases, so run the memory test first in a fresh process (or accept that an earlier peak
    hides growth); the measurement is an upper bound for the block when the earlier peak was lower.
    """
    if sys.platform == "win32":
        pytest.skip("rss_growth needs the resource module (not on Windows)")
    result = RSSGrowth()
    start = _peak_rss()
    try:
        yield result
    finally:
        result.bytes = _peak_rss() - start


def check_identities(data: Any) -> dict[str, bool]:
    """The exact identities of generated data (``itj.toy.validate.check_identities``, owner WP2)."""
    from itj.toy.validate import check_identities as _check

    return _check(data)


def sudakov_cdf(theta: np.ndarray, **kwargs: Any) -> np.ndarray:
    """First-emission CDF of a quark at fixed coupling (``itj.toy.validate.sudakov_cdf``, owner WP2)."""
    from itj.toy.validate import sudakov_cdf as _cdf

    return _cdf(theta, **kwargs)


def lund_ll(cr: float, alphas: float) -> float:
    """Leading-log primary Lund density ``2 alpha_s C_R / pi`` (``itj.toy.validate.lund_ll``, owner WP2)."""
    from itj.toy.validate import lund_ll as _lund

    return _lund(cr, alphas)
