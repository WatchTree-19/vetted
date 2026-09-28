"""Conventions without an R PerformanceAnalytics counterpart, each checked
against an independent implementation (numpy, scipy, or a brute-force
definition), plus a check that no two conventions of a metric are secretly
the same estimator on every fixture."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize_scalar

from vetted.metrics import CONVENTIONS, compute, conventions_for, metrics

FIX = json.loads((Path(__file__).resolve().parents[1] / "src" / "vetted" / "data" / "fixtures.json").read_text())
NAMES = list(FIX)


def arr(f):
    return np.asarray(FIX[f]["returns"], dtype=float)


@pytest.mark.parametrize("f", NAMES)
def test_population_vol(f):
    assert np.isclose(compute("volatility_annual", "std_ddof0", arr(f), 252), np.std(arr(f)) * np.sqrt(252), rtol=1e-12)


@pytest.mark.parametrize("f", NAMES)
def test_sharpe_ddof0(f):
    r = arr(f)
    assert np.isclose(compute("sharpe_annual", "arithmetic_ddof0", r, 12), r.mean() / r.std() * np.sqrt(12), rtol=1e-12)


@pytest.mark.parametrize("f", NAMES)
def test_lower_order_statistic_is_inverted_cdf_quantile(f):
    assert np.isclose(compute("var_95", "historical_order_statistic", arr(f), 1),
                      np.quantile(arr(f), 0.05, method="inverted_cdf"), rtol=0, atol=0)


@pytest.mark.parametrize("f", NAMES)
def test_upper_order_statistic_definition(f):
    s = np.sort(arr(f))
    k = int(np.floor(0.05 * len(s)))  # rank k+1
    assert compute("var_95", "historical_order_statistic_upper", arr(f), 1) == s[k]


def test_lower_and_upper_order_statistics_differ_only_on_whole_a_n():
    for f in NAMES:
        n = len(arr(f))
        lo = compute("var_95", "historical_order_statistic", arr(f), 1)
        hi = compute("var_95", "historical_order_statistic_upper", arr(f), 1)
        whole = abs(0.05 * n - round(0.05 * n)) < 1e-9
        assert (lo != hi) == whole, f


@pytest.mark.parametrize("f", NAMES)
def test_gaussian_ddof1(f):
    r = arr(f)
    assert np.isclose(compute("var_95", "gaussian_ddof1", r, 1),
                      stats.norm.ppf(0.05, loc=r.mean(), scale=r.std(ddof=1)), rtol=1e-12)
    assert np.isclose(compute("es_95", "gaussian_ddof1", r, 1),
                      r.mean() - r.std(ddof=1) * stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05, rtol=1e-12)


@pytest.mark.parametrize("f", [n for n in NAMES if len(FIX[n]["returns"]) > 20])
def test_rockafellar_uryasev_is_the_minimum_of_its_objective(f):
    # CVaR as the minimum over t of t + E[max(loss - t, 0)] / a, losses = -r
    loss = -arr(f)
    obj = lambda t: t + np.maximum(loss - t, 0).mean() / 0.05
    grid = np.sort(loss)
    best = min(obj(t) for t in grid)  # the optimum sits at a data point
    assert np.isclose(-compute("es_95", "rockafellar_uryasev", arr(f), 1), best, rtol=1e-12)
    assert minimize_scalar(obj, bounds=(grid[0], grid[-1]), method="bounded").fun >= best - 1e-12


@pytest.mark.parametrize("f", NAMES)
def test_pearson_kurtosis_is_scipy_non_fisher(f):
    assert np.isclose(compute("kurtosis", "pearson_moment", arr(f), 1),
                      stats.kurtosis(arr(f), fisher=True, bias=True) + 3, rtol=1e-12)
    assert np.isclose(compute("kurtosis", "pearson_moment", arr(f), 1),
                      stats.kurtosis(arr(f), fisher=False, bias=True), rtol=1e-12)


@pytest.mark.parametrize("f", NAMES)
def test_adjusted_g1_is_scipy_unbiased_skew(f):
    assert np.isclose(compute("skewness", "adjusted_g1", arr(f), 1), stats.skew(arr(f), bias=False), rtol=1e-10)


def _brute_mdd(r, initial_peak):
    wealth = 1.0
    peak = 1.0 if initial_peak else None
    worst = 0.0
    for x in r:
        wealth *= 1 + x
        peak = wealth if peak is None else max(peak, wealth)
        worst = max(worst, 1 - wealth / peak)
    return worst


@pytest.mark.parametrize("f", NAMES)
def test_drawdown_variants_by_brute_force(f):
    assert np.isclose(compute("max_drawdown", "geometric_no_initial_peak", arr(f), 1), _brute_mdd(arr(f), False), rtol=1e-12, atol=0)
    assert np.isclose(compute("max_drawdown", "geometric_initial_peak", arr(f), 1), _brute_mdd(arr(f), True), rtol=1e-12, atol=0)


@pytest.mark.parametrize("f", [n for n in NAMES if n != "all_positive"])
def test_sortino_denominators(f):
    r = arr(f)
    d = np.minimum(r, 0.0)
    assert np.isclose(compute("sortino_annual", "full_ddof1", r, 252), r.mean() / np.sqrt((d ** 2).sum() / (len(r) - 1)) * np.sqrt(252), rtol=1e-12)
    assert np.isclose(compute("sortino_annual", "subset", r, 252), r.mean() / np.sqrt((d ** 2).sum() / (r < 0).sum()) * np.sqrt(252), rtol=1e-12)


@pytest.mark.parametrize("f", NAMES)
def test_naive_tail_mean(f):
    s = np.sort(arr(f))
    assert np.isclose(compute("es_95", "naive_tail_mean_floor", arr(f), 1), s[: int(np.floor(0.05 * len(s))) + 1].mean(), rtol=1e-12)


@pytest.mark.parametrize("metric", metrics())
def test_every_pair_of_conventions_is_distinguishable(metric):
    convs = conventions_for(metric)
    for a, b in itertools.combinations(convs, 2):
        differs = False
        for d in FIX.values():
            va, vb = a.fn(d["returns"], d["periods_per_year"]), b.fn(d["returns"], d["periods_per_year"])
            if np.isfinite(va) and np.isfinite(vb) and not np.isclose(va, vb, rtol=1e-9):
                differs = True
                break
        assert differs, f"{metric}: {a.name} and {b.name} agree on every fixture"


def test_every_convention_is_tested_somewhere():
    tested_here = {
        ("volatility_annual", "std_ddof0"), ("sharpe_annual", "arithmetic_ddof0"),
        ("var_95", "historical_order_statistic"), ("var_95", "historical_order_statistic_upper"),
        ("var_95", "gaussian_ddof1"), ("es_95", "gaussian_ddof1"), ("es_95", "rockafellar_uryasev"),
        ("kurtosis", "pearson_moment"), ("skewness", "adjusted_g1"),
        ("max_drawdown", "geometric_no_initial_peak"), ("max_drawdown", "geometric_initial_peak"),
        ("sortino_annual", "full_ddof1"), ("sortino_annual", "subset"), ("es_95", "naive_tail_mean_floor"),
        ("sortino_annual", "full_ddof0"),  # test_oracle: period x sqrt(periods)
    }
    untested = [(c.metric, c.name) for c in CONVENTIONS if not c.oracle and (c.metric, c.name) not in tested_here]
    # deliberate slips and hybrids are defined by their docstring, not a published source
    assert set(untested) <= {("sortino_annual", "losers_std_slip"), ("es_95", "hybrid_parametric_threshold"),
                             ("var_95", "cornish_fisher_ddof1")}
