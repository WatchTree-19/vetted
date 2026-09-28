"""The classifier must actually catch defects: every guard is shown to fail
on a deliberately broken estimator."""

import json
from pathlib import Path

import numpy as np

from vetted.conformance import classify
from vetted.metrics import compute

FIX = json.loads((Path(__file__).resolve().parents[1] / "src" / "vetted" / "data" / "fixtures.json").read_text())


def _all(metric, fn):
    return [classify(metric, fn(d["returns"], d["periods_per_year"]), d["returns"], d["periods_per_year"])
            for d in FIX.values()]


def test_correct_estimator_is_named():
    res = _all("sharpe_annual", lambda r, a: compute("sharpe_annual", "arithmetic_ddof1", r, a))
    assert all(("arithmetic_ddof1", False) in {tuple(m) for m in x[0]} for x in res)


def test_sign_flip_is_recognised_not_flagged():
    res = _all("var_95", lambda r, a: -compute("var_95", "historical_interpolated", r, a))
    assert all(("historical_interpolated", True) in {tuple(m) for m in x[0]} for x in res)


def test_double_annualisation_matches_nothing():
    bad = lambda r, a: compute("sharpe_annual", "arithmetic_ddof1", r, a) * np.sqrt(a)
    assert all(not x[0] for x in _all("sharpe_annual", bad))


def test_off_by_one_percentile_matches_nothing_on_real_samples():
    # one rank above the upper order statistic (itself one above the lower
    # one when 0.05 n is whole), so it is no published quantile on any size
    def bad(r, a):
        s = np.sort(np.asarray(r))
        return float(s[int(np.floor(0.05 * len(s))) + 1])
    res = [classify("var_95", bad(d["returns"], 1), d["returns"], 1)
           for n, d in FIX.items() if len(d["returns"]) > 100]
    assert all(not x[0] for x in res)


def test_fillna_zero_is_not_a_convention():
    d = FIX["gaussian_neg_mean"]
    r = np.asarray(d["returns"]).copy()
    r[::40] = 0.0  # what fillna(0) does to missing observations
    v = compute("volatility_annual", "std_ddof1", r, 252)
    assert not classify("volatility_annual", v, d["returns"], 252)[0]


def test_negated_ratio_is_not_a_convention():
    res = _all("sharpe_annual", lambda r, a: -compute("sharpe_annual", "arithmetic_ddof1", r, a))
    # zero-valued or undefined cases aside, nothing matches
    assert all(not x[0] for x in res)


def _summary_for(fn, metric):
    from vetted.conformance import summarise
    rows = []
    for name, d in FIX.items():
        try:
            v = float(fn(name, d["returns"], d["periods_per_year"]))
        except Exception:
            v = float("nan")
        row = dict(library="x", metric=metric, fixture=name, value=v, error=None)
        if np.isfinite(v):
            m, nearest, dev = classify(metric, v, d["returns"], d["periods_per_year"])
            row.update(matches=sorted([list(t) for t in m], key=str), nearest=nearest, deviation=dev)
        rows.append(row)
    return summarise(rows, FIX)[0]


def test_switching_sign_between_inputs_is_inconsistent():
    def flaky(name, r, a):
        v = compute("var_95", "historical_interpolated", r, a)
        return -v if name == "student_t3" else v
    assert _summary_for(flaky, "var_95")["status"] == "INCONSISTENT SIGN"


def test_failing_where_the_convention_is_defined_is_reported():
    def crashes_on_small(name, r, a):
        if len(r) < 30:
            raise ZeroDivisionError
        return compute("sharpe_annual", "arithmetic_ddof1", r, a)
    assert _summary_for(crashes_on_small, "sharpe_annual")["status"] == "FAILS ON INPUT"


def test_nan_where_the_estimator_is_undefined_is_fine():
    # Sortino has no downside on the all-positive fixture
    def sortino(name, r, a):
        return compute("sortino_annual", "full_ddof0", r, a)
    assert _summary_for(sortino, "sortino_annual")["status"] == "conformant"
