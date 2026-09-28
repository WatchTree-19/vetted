"""vetted's backtests against established R implementations.

Expected values are in tests/data/backtest_oracle.json, written by
oracle/gen_backtest_oracle.R from tests/data/backtest_fixtures.json. The
package versions are recorded in the file's "meta" block. Nothing in that
file was computed by vetted.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from vetted import compare, es, overfitting, var

DATA = Path(__file__).parent / "data"
FX = json.loads((DATA / "backtest_fixtures.json").read_text())
OR = json.loads((DATA / "backtest_oracle.json").read_text())
SCENARIOS = [k for k in OR if k.endswith(("_250", "_1000")) and k != "compare_1000"]
LEVELS = [("var99", 0.01), ("var975", 0.025)]


def arr(x):
    return np.asarray(x, dtype=float)


def scalar(x):
    return float(np.asarray(x, dtype=float).ravel()[0])


@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("key, alpha", LEVELS)
def test_kupiec_matches_rugarch(name, key, alpha):
    f, o = FX[name], OR[name][key]
    r = var.kupiec(f["pnl"], f[key], alpha)
    assert r.detail["exceptions"] == o["exceptions"]
    if o["rugarch_uc_stat"] is None or not np.isfinite(o["rugarch_uc_stat"]):
        pytest.skip("rugarch cannot compute this case (no exceptions)")
    assert r.statistic == pytest.approx(o["rugarch_uc_stat"], rel=1e-9, abs=1e-12)
    assert r.p_value == pytest.approx(o["rugarch_uc_p"], rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("key, alpha", LEVELS)
def test_christoffersen_matches_rugarch(name, key, alpha):
    f, o = FX[name], OR[name][key]
    if o["rugarch_cc_stat"] is None:
        pytest.skip("rugarch VaRTest fails without consecutive exceptions in the transition table")
    r = var.christoffersen(f["pnl"], f[key], alpha)
    assert r.statistic == pytest.approx(o["rugarch_cc_stat"], rel=1e-9, abs=1e-12)
    assert r.p_value == pytest.approx(o["rugarch_cc_p"], rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("key, alpha", LEVELS)
def test_dq_matches_gas(name, key, alpha):
    f, o = FX[name], OR[name][key]
    if o["gas_dq_stat"] is None:
        pytest.skip("GAS failed on this case")
    r = var.dynamic_quantile(f["pnl"], f[key], alpha, lags=4, squared_pnl_lag=True)
    assert r.statistic == pytest.approx(scalar(o["gas_dq_stat"]), rel=1e-8)
    if r.detail["df"] == 7:
        assert r.p_value == pytest.approx(scalar(o["gas_dq_p"]), rel=1e-7, abs=1e-12)
    else:
        # A constant VaR (and, with no exceptions, constant hit lags) is
        # collinear with the intercept. GAS still counts seven degrees of
        # freedom; vetted uses the rank of the regressors.
        assert name.startswith("static") and r.detail["df"] < 7
        assert r.p_value == pytest.approx(float(stats.chi2.sf(r.statistic, r.detail["df"])), rel=1e-12)


@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("key, alpha", LEVELS)
def test_quantile_loss_matches_gas(name, key, alpha):
    f, o = FX[name], OR[name][key]
    loss = var.quantile_loss(f["pnl"], f[key], alpha)
    assert loss.mean() == pytest.approx(o["gas_quantile_loss"], rel=1e-10)


@pytest.mark.parametrize("name", SCENARIOS)
def test_fz0_loss_matches_gas(name):
    f, o = FX[name], OR[name]
    loss = es.fz0_loss(f["pnl"], f["var975"], f["es975"], 0.025)
    assert loss.mean() == pytest.approx(o["fz0_loss_mean"], rel=1e-10)
    np.testing.assert_allclose(loss[:5], arr(o["fz0_loss_first5"]), rtol=1e-10)


@pytest.mark.parametrize("name", SCENARIOS)
def test_conditional_calibration_matches_esback(name):
    f, o = FX[name], OR[name]["esback_cc"]
    r = es.conditional_calibration(f["pnl"], f["var975"], f["es975"], 0.025, scale=f["scale"])
    assert r.p_value == pytest.approx(o["pvalue_twosided_simple"], rel=1e-7, abs=1e-12)
    assert r.detail["p_simple_one_sided_esback"] == pytest.approx(o["pvalue_onesided_simple"], rel=1e-7, abs=1e-12)
    assert r.detail["p_general_two_sided"] == pytest.approx(o["pvalue_twosided_general"], rel=1e-7, abs=1e-12)
    assert r.detail["p_general_one_sided_esback"] == pytest.approx(o["pvalue_onesided_general"], rel=1e-7, abs=1e-12)


def _qloss(pnl, v, a=0.01):
    return var.quantile_loss(pnl, v, a)


@pytest.mark.parametrize("key", list(OR["dm"]))
def test_diebold_mariano_matches_forecast(key):
    h, ve, alt = key.split("_")
    f = FX["compare_1000"]
    la = _qloss(f["pnl"], f["var_garch"])
    lb = _qloss(f["pnl"], f["var_static"])
    r = compare.diebold_mariano(
        la, lb, horizon=int(float(h)), varestimator=ve, alternative={"two.sided": "two-sided", "less": "less"}[alt]
    )
    assert r.statistic == pytest.approx(OR["dm"][key]["statistic"], rel=1e-9)
    assert r.p_value == pytest.approx(OR["dm"][key]["p"], rel=1e-8, abs=1e-14)


def test_pbo_matches_r_package():
    M = arr(FX["trials_1000x12"])
    o = OR["pbo"]
    # The R package divides the out-of-sample rank by N, not N + 1.
    r = overfitting.pbo(M, n_blocks=8, rank_denominator="n")
    assert r.detail["splits"] == 70 == len(o["os_rank"])
    assert r.statistic == pytest.approx(o["phi"], abs=1e-12)
    np.testing.assert_allclose(r.detail["oos_rank_of_selected"], arr(o["os_rank"]), rtol=1e-12)
    assert len(set(o["os_rank"])) > 3  # the fixture exercises more than one rank


def test_probabilistic_sharpe_and_mintrl_match_performanceanalytics():
    M = arr(FX["trials_1000x12"])
    r = M[:, 3]
    o = OR["psr"]
    sr, sk, ku, n = overfitting._moments(r)
    assert (sr, sk, ku, n) == pytest.approx((o["sr"], o["skew"], o["kurt"], o["n"]), rel=1e-10)
    # benchmark is annualised at the interface; the oracle's 0.02 is per period
    assert overfitting.probabilistic_sharpe(r, 0.0, periods=1) == pytest.approx(o["psr_0"], rel=1e-10)
    assert overfitting.probabilistic_sharpe(r, 0.02, periods=1) == pytest.approx(o["psr_002"], rel=1e-10)
    assert overfitting.min_track_record(r, 0.0, periods=1) == pytest.approx(o["mintrl_0"], rel=1e-10)
    assert overfitting.min_track_record(r, 0.02, 0.99, periods=1) == pytest.approx(o["mintrl_002_99"], rel=1e-10)


@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("standardised", [False, True])
def test_exceedance_residuals_match_esback(name, standardised):
    # Both p-values come from a bootstrap (esback with R's generator, vetted
    # with numpy's), so they agree to Monte Carlo error, not to machine
    # precision: 20,000 resamples on each side put the standard error near
    # 0.004.
    f, o = FX[name], OR[name]["esback_er"]
    if o is None:
        pytest.skip("esback failed on this case")
    suffix = "standardized" if standardised else "simple"
    try:
        r = es.exceedance_residuals(
            f["pnl"], f["var975"], f["es975"], scale=f["scale"] if standardised else None, n_boot=20_000
        )
    except ValueError:
        pytest.skip("fewer than five exceptions, the bootstrap minimum")
    assert r.p_value == pytest.approx(o[f"pvalue_onesided_{suffix}"], abs=0.02)
    assert r.detail["p_two_sided"] == pytest.approx(o[f"pvalue_twosided_{suffix}"], abs=0.02)
