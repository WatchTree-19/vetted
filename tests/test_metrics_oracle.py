"""Every convention that names an R PerformanceAnalytics call must reproduce
that call's output, generated independently by oracle/gen_metrics_oracle.R."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vetted.metrics import CONVENTIONS, compute

DATA = Path(__file__).resolve().parents[1] / "src" / "vetted" / "data"
FIX = json.loads((DATA / "fixtures.json").read_text())
ORACLE = json.loads((DATA / "oracle.json").read_text())["values"]

CASES = [
    (c, f)
    for c in CONVENTIONS
    if c.oracle
    for f in FIX
    if ORACLE[f][c.oracle] is not None
]


@pytest.mark.parametrize("conv,fixture", CASES, ids=lambda x: getattr(x, "name", x))
def test_matches_performanceanalytics(conv, fixture):
    d = FIX[fixture]
    got = conv.fn(d["returns"], d["periods_per_year"])
    assert np.isclose(got, ORACLE[fixture][conv.oracle], rtol=1e-10, atol=1e-14)


def test_every_metric_has_an_oracle_backed_convention():
    metrics = {c.metric for c in CONVENTIONS}
    backed = {c.metric for c in CONVENTIONS if c.oracle}
    # sortino_annual has no PerformanceAnalytics annualised form; its
    # conventions are sortino_period times sqrt(periods), checked below.
    assert metrics - backed == {"sortino_annual"}


@pytest.mark.parametrize("fixture", list(FIX))
def test_sortino_annual_is_period_times_sqrt_periods(fixture):
    d = FIX[fixture]
    per = ORACLE[fixture]["SortinoRatio(R, MAR=0)"]
    if per is None:
        pytest.skip("no downside in this series")
    ann = compute("sortino_annual", "full_ddof0", d["returns"], d["periods_per_year"])
    assert np.isclose(ann, per * np.sqrt(d["periods_per_year"]), rtol=1e-10)


@pytest.mark.parametrize("fixture", list(FIX))
def test_pandas_skew_is_adjusted_g1(fixture):
    r = FIX[fixture]["returns"]
    assert np.isclose(compute("skewness", "adjusted_g1", r, 1), pd.Series(r).skew(), rtol=1e-10)


@pytest.mark.parametrize("fixture", list(FIX))
def test_pandas_kurt_is_sample_excess(fixture):
    r = FIX[fixture]["returns"]
    assert np.isclose(compute("kurtosis", "excess_sample", r, 1), pd.Series(r).kurt(), rtol=1e-10)
