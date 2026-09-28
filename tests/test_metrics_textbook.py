"""Values printed in Bacon (2008), Practical Portfolio Performance Measurement
and Attribution, for the 24-month example portfolio, as quoted in the
R PerformanceAnalytics documentation. These are checked at the precision the
book prints, and were not produced by any code in this project."""

import numpy as np

from vetted.fixtures import BACON_PORTFOLIO as R
from vetted.metrics import _downside_dev, _omega, _pain, _ulcer


def rounds_to(x, printed):
    decimals = len(str(printed).split(".")[1])
    return round(x, decimals) == printed


def test_downside_deviation_mar_half_percent():
    assert rounds_to(_downside_dev("full", mar=0.005)(R), 0.0255)


def test_downside_frequency_mar_half_percent():
    r = np.asarray(R)
    assert rounds_to((r < 0.005).mean(), 0.458)


def test_omega_at_zero_is_bernardo_ledoit():
    assert rounds_to(_omega(0.0)(R), 1.78)


def test_pain_index():
    assert rounds_to(_pain(R), 0.04)


def test_martin_ratio_uses_ulcer_index():
    # Martin ratio = annualised return / Ulcer index; book value 1.70
    cagr = np.prod(1 + np.asarray(R)) ** (12 / len(R)) - 1
    assert rounds_to(cagr / _ulcer(R), 1.70)
