"""Return series used by the conformance vectors.

Two kinds:

* ``bacon``: the 24 monthly portfolio returns from Carl Bacon, *Practical
  Portfolio Performance Measurement and Attribution* (2nd ed., Wiley, 2008),
  as shipped with R PerformanceAnalytics (``data(portfolio_bacon)``). Many
  published textbook values exist for this series, so it anchors the suite to
  numbers nobody in this project computed.
* Synthetic daily series chosen for where real library bugs have lived: fat
  tails, volatility regimes, skew, an all-positive series (empty downside), a
  near-constant series, and a seven-observation sample.

Every series is deterministic (fixed seeds) and is written to
``src/vetted/data/fixtures.json`` so the R oracle and every library see identical
bytes.
"""

from __future__ import annotations

import numpy as np

BACON_PORTFOLIO = [
    0.003, 0.026, 0.011, -0.010, 0.015, 0.025, 0.016, 0.067, -0.014, 0.040,
    -0.005, 0.081, 0.040, -0.037, -0.061, 0.017, -0.049, -0.022, 0.070, 0.058,
    -0.065, 0.024, -0.005, -0.009,
]
BACON_BENCHMARK = [
    0.002, 0.025, 0.018, -0.011, 0.014, 0.018, 0.014, 0.065, -0.015, 0.042,
    -0.006, 0.083, 0.039, -0.038, -0.062, 0.015, -0.048, 0.021, 0.060, 0.056,
    -0.067, 0.019, -0.003, 0.000,
]


def _gaussian_neg(n=1512, seed=0):
    return np.random.default_rng(seed).normal(0.0004, 0.011, n)


def _gaussian_pos(n=1512, seed=1):
    return np.random.default_rng(seed).normal(0.0012, 0.009, n)


def _student_t3(n=1512, seed=2):
    rng = np.random.default_rng(seed)
    return 0.0004 + 0.011 * rng.standard_t(3, n) / np.sqrt(3.0)


def _regime(n=1512, seed=3):
    rng = np.random.default_rng(seed)
    hi = rng.random(n) < 0.15
    return rng.normal(0.0004, np.where(hi, 0.035, 0.007))


def _left_skewed(n=1512, seed=4):
    rng = np.random.default_rng(seed)
    return 0.002 - rng.gamma(2.0, 0.008, n)


def _all_positive(n=756, seed=6):
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(0.001, 0.004, n)) + 1e-5


def _gaussian_n(n, seed):
    """Sizes where 0.05 * n is a whole number, so the lower and upper order
    statistics differ (1260 = five years of daily data)."""
    return np.random.default_rng(seed).normal(0.0004, 0.011, n)


def _tiny(n=7, seed=8):
    return np.random.default_rng(seed).normal(0.0004, 0.011, n)


# name -> (returns, periods per year)
FIXTURES: dict[str, tuple[list[float], int]] = {
    "bacon_monthly": (BACON_PORTFOLIO, 12),
    "gaussian_neg_mean": (_gaussian_neg().round(12).tolist(), 252),
    "gaussian_pos_mean": (_gaussian_pos().round(12).tolist(), 252),
    "student_t3": (_student_t3().round(12).tolist(), 252),
    "regime_switch": (_regime().round(12).tolist(), 252),
    "left_skewed": (_left_skewed().round(12).tolist(), 252),
    "all_positive": (_all_positive().round(12).tolist(), 252),
    "gaussian_n1260": (_gaussian_n(1260, 9).round(12).tolist(), 252),
    "gaussian_n500": (_gaussian_n(500, 10).round(12).tolist(), 252),
    "tiny_sample_n7": (_tiny().round(12).tolist(), 252),
}
