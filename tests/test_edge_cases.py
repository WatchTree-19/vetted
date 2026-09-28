"""Regression tests for problems found in review: each would have given a
wrong answer or a crash before it was fixed."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vetted import compare, conformal, es, frtb, overfitting, var


# --------------------------------------------------------------------- VaR


def test_dq_does_not_depend_on_units():
    rng = np.random.default_rng(1)
    p = rng.standard_normal(500)
    v = 2.33 * np.abs(1 + 0.3 * rng.standard_normal(500))
    base = var.dynamic_quantile(p, v, 0.01, squared_pnl_lag=True).statistic
    for scale in (1e-3, 1e3, 1e6):
        r = var.dynamic_quantile(p * scale, v * scale, 0.01, squared_pnl_lag=True)
        assert r.statistic == pytest.approx(base, rel=1e-9)
    # and equals the projection computed with a plain least-squares solve
    hit = (p < -v).astype(float) - 0.01
    X = np.column_stack([np.ones(496)] + [hit[4 - k : 500 - k] for k in range(1, 5)] + [v[4:], p[3:499] ** 2])
    b, *_ = np.linalg.lstsq(X, hit[4:], rcond=None)
    assert base == pytest.approx(float(b @ X.T @ X @ b) / (0.01 * 0.99), rel=1e-9)


def test_dq_degrees_of_freedom_follow_the_rank():
    p = np.random.default_rng(2).standard_normal(500)
    assert var.dynamic_quantile(p, 2.33, 0.01).detail["df"] == 5  # constant VaR adds nothing
    assert var.dynamic_quantile(p, 2.33 + 0.1 * np.sin(np.arange(500)), 0.01).detail["df"] == 6


def test_dq_with_missing_day_and_squared_lag():
    p = np.random.default_rng(3).standard_normal(300)
    p[10] = np.nan
    r = var.dynamic_quantile(p, 2.33, 0.01, missing="exception", squared_pnl_lag=True)
    assert np.isfinite(r.statistic)


def test_pandas_inputs_must_share_an_index():
    idx = pd.bdate_range("2024-01-01", periods=300)
    pnl = pd.Series(np.zeros(300), index=idx)
    var_same = pd.Series(np.ones(300), index=idx)
    var_shifted = pd.Series(np.ones(300), index=idx + pd.offsets.BDay(1))
    assert var.kupiec(pnl, var_same, 0.01).detail["exceptions"] == 0
    with pytest.raises(ValueError, match="different indexes"):
        var.kupiec(pnl, var_shifted, 0.01)
    with pytest.raises(ValueError, match="different indexes"):
        frtb.traffic_light(var_shifted, pnl)


def test_traffic_light_input_checks():
    with pytest.raises(ValueError, match="length"):
        frtb.traffic_light(np.ones(300), np.zeros(260))
    with pytest.raises(ValueError, match="same days"):
        frtb.traffic_light(np.ones(250), np.zeros(250), np.zeros(100))
    with pytest.raises(ValueError, match="window"):
        frtb.traffic_light(np.ones(300), np.zeros(300), window=0)


@pytest.mark.parametrize("n, alpha", [(1, 1e-4), (5, 0.01), (20, 0.001), (250, 0.025)])
def test_zero_exceptions_is_always_green(n, alpha):
    z = frtb.binomial_zones(n, alpha)
    assert z["green"][0] == 0 and z["green"][1] >= 0
    assert z["red_from"] >= 1
    assert frtb.traffic_light(1.0, np.zeros(n), alpha=alpha, window=None).detail["zone"] == "green"


def test_pla_rejects_constant_pnl():
    with pytest.raises(ValueError, match="constant"):
        frtb.pla_test(np.ones(250), np.random.default_rng(4).standard_normal(250))


# ---------------------------------------------------------------------- ES


def _t(nu=5):
    s = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(0.025, nu)
    return -q * s, stats.t.pdf(q, nu) / 0.025 * (nu + q**2) / (nu - 1) * s, s


def test_mcneil_frey_needs_enough_exceptions():
    pnl = np.zeros(100)
    pnl[[3, 7]] = [-2.5, -3.0]
    with pytest.raises(ValueError, match="exceptions"):
        es.exceedance_residuals(pnl, 2.0, 2.4)


def test_mcneil_frey_size_on_short_samples():
    v, e, s = _t()
    rng = np.random.default_rng(5)
    rej = tried = 0
    for i in range(300):
        x = rng.standard_t(5, 250) * s
        try:
            rej += es.exceedance_residuals(x, v, e, n_boot=1000, seed=i).reject
            tried += 1
        except ValueError:
            pass
    assert rej / tried <= 0.08


def test_nolde_ziegel_one_sided_points_at_underestimation():
    v, e, s = _t()
    rng = np.random.default_rng(6)
    under = [
        es.conditional_calibration(rng.standard_t(5, 1000) * s * 1.3, v, e).detail["p_simple_one_sided"]
        for _ in range(30)
    ]
    conservative = [
        es.conditional_calibration(rng.standard_t(5, 1000) * s * 0.7, v, e).detail["p_simple_one_sided"]
        for _ in range(30)
    ]
    assert np.mean(np.array(under) < 0.05) > 0.9
    assert np.mean(np.array(conservative) < 0.05) == 0.0
    # esback's variant rejects the conservative model too
    r = es.conditional_calibration(rng.standard_t(5, 1000) * s * 0.7, v, e)
    assert r.detail["p_simple_one_sided_esback"] < 0.05


def test_du_escanciano_hand_computed():
    # alpha = 0.1, PITs chosen so H can be written down:
    # u <= 0.1 gives H = (0.1 - u) / 0.1: 0.05 -> 0.5, 0.02 -> 0.8, others 0
    u = np.array([0.05, 0.5, 0.02, 0.9, 0.3, 0.7, 0.6, 0.4, 0.8, 0.2])
    h = np.array([0.5, 0, 0.8, 0, 0, 0, 0, 0, 0, 0])
    z = np.sqrt(10) * (h.mean() - 0.05) / np.sqrt(0.1 * (1 / 3 - 0.1 / 4))
    d = h - 0.05
    rho1 = np.sum(d[1:] * d[:-1]) / 9 / np.mean(d * d)
    r = es.du_escanciano(u, 0.1, lags=1)
    assert r.statistic == pytest.approx(z, rel=1e-12)
    assert r.detail["conditional_statistic"] == pytest.approx(10 * rho1**2, rel=1e-12)


def test_du_escanciano_monte_carlo_size():
    rng = np.random.default_rng(7)
    rej_u = rej_c = 0
    for i in range(200):
        r = es.du_escanciano(rng.random(250), 0.025, pvalue="monte_carlo", n_sims=1000, seed=i)
        rej_u += r.reject
        rej_c += r.detail["reject_conditional"]
    assert rej_u / 200 <= 0.08 and rej_c / 200 <= 0.08


def test_fz0_rejects_bad_forecasts():
    with pytest.raises(ValueError):
        es.fz0_loss([-3.0], [2.0], [1.0])  # ES below VaR
    with pytest.raises(ValueError):
        es.fz0_loss([np.nan], [2.0], [3.0])


def test_diebold_mariano_means_use_the_same_days():
    a = np.arange(10.0)
    b = np.arange(10.0) + 1
    a[0] = np.nan
    r = compare.diebold_mariano(a, b + np.random.default_rng(8).normal(0, 0.1, 10))
    assert r.detail["observations"] == 9
    assert np.isfinite(r.detail["mean_loss_a"])


def test_sampler_with_dropped_days_is_refused():
    v, e, s = _t()
    pnl = np.random.default_rng(9).standard_t(5, 250) * s
    pnl[5] = np.nan
    with pytest.raises(ValueError, match="sampler"):
        es.acerbi_szekely(pnl, v, e, sampler=lambda g, n: g.standard_normal((n, 250)), missing="drop")


# -------------------------------------------------------------- overfitting


def test_expected_max_sharpe_never_negative():
    for n in (1.0, 1.0001, 1.1, 1.2, 1.3, 2.0):
        assert overfitting.expected_max_sharpe(n, 0.5) >= 0.0
    r = np.random.default_rng(10).normal(0.0003, 0.01, 500)
    assert overfitting.deflated_sharpe(r, 1.02, sharpe_variance=0.3).statistic <= overfitting.probabilistic_sharpe(r)


def test_deflated_sharpe_holds_its_size_on_correlated_trials():
    # 50 correlated null strategies (pairwise correlation 0.7): the best one
    # should be called significant at most about 5% of the time.
    rng = np.random.default_rng(11)
    hits = 0
    reps = 200
    for _ in range(reps):
        common = rng.standard_normal((500, 1))
        M = 0.01 * (np.sqrt(0.7) * common + np.sqrt(0.3) * rng.standard_normal((500, 50)))
        best = M[:, np.argmax(M.mean(0) / M.std(0, ddof=1))]
        hits += overfitting.deflated_sharpe(best, M).reject
    assert hits / reps <= 0.07


def test_haircut_adds_the_strategy_when_missing():
    others = np.array([0.1, 0.2, 0.3, 0.4])
    with_it = overfitting.haircut_sharpe(1.0, 1260, np.append(others, 1.0), method="holm")
    without = overfitting.haircut_sharpe(1.0, 1260, others, method="holm")
    assert without["adjusted_p_value"] == pytest.approx(with_it["adjusted_p_value"])
    assert without["trials"] == 5
    with pytest.raises(ValueError):
        overfitting.haircut_sharpe(1.0, 1260, 0, method="bonferroni")


def test_deflated_sharpe_argument_checks():
    rng = np.random.default_rng(12)
    M = rng.normal(0, 0.01, (300, 5))
    with pytest.raises(ValueError, match="only with a count"):
        overfitting.deflated_sharpe(M[:, 0], M, sharpe_variance=0.3)
    assert overfitting.deflated_sharpe(M[:, 0], 10.0, sharpe_variance=0.3).detail["trials"] == 10


def test_flat_series_and_flat_configurations():
    with pytest.raises(ValueError, match="zero variance"):
        overfitting.probabilistic_sharpe(np.full(100, 0.1))
    M = np.random.default_rng(13).normal(0, 0.01, (800, 6))
    M[:, 2] = 0.0003
    with pytest.raises(ValueError, match="constant"):
        overfitting.pbo(M, n_blocks=8)
    assert overfitting.effective_trials(M) > 1


# ---------------------------------------------------------------- conformal


def test_conformal_keeps_infinite_scores_and_checks_scale():
    assert conformal.conformal_quantile(np.append(np.arange(1.0, 10.0), np.inf), 0.1) == np.inf
    with pytest.raises(ValueError):
        conformal.conformal_var(np.zeros(50), 0.05, scale=-np.ones(50))


def test_conformal_var_is_not_too_conservative():
    rng = np.random.default_rng(14)
    b = []
    for _ in range(4000):
        x = rng.standard_t(2, 100)
        b.append(x[99] < -conformal.conformal_var(x[:99], 0.05))
    # exact finite-sample coverage: P(breach) is in [alpha - 1/(n+1), alpha]
    se = np.sqrt(0.05 * 0.95 / 4000)
    assert 0.05 - 1 / 100 - 3 * se <= np.mean(b) <= 0.05 + 3 * se


def test_adaptive_var_reports_when_the_guarantee_is_lost():
    rng = np.random.default_rng(15)
    p = -np.abs(rng.normal(0.01, 0.002, 1500)) - np.linspace(0, 0.02, 1500)  # a steady sell-off
    capped = conformal.adaptive_var(p, 0.01, window=250, gamma=0.005)
    exact = conformal.adaptive_var(p, 0.01, window=250, gamma=0.005, unbounded="inf")
    assert capped["capped_exceptions"] > 0 and not capped["guarantee_holds"]
    assert exact["guarantee_holds"]
    assert abs(exact["exception_rate"] - 0.01) <= exact["bound"]
    with pytest.raises(ValueError):
        conformal.adaptive_var(np.append(p, np.nan), 0.01)


# ------------------------------------------------------------ second review


def test_index_checks_cover_losses_and_comparisons():
    idx = pd.bdate_range("2024-01-01", periods=50)
    a = pd.Series(np.random.default_rng(20).random(50), index=idx)
    b = pd.Series(np.random.default_rng(21).random(50), index=idx + pd.offsets.BDay(1))
    with pytest.raises(ValueError, match="different indexes"):
        compare.diebold_mariano(a, b)
    with pytest.raises(ValueError, match="different indexes"):
        var.quantile_loss(a, b, 0.01)
    with pytest.raises(ValueError, match="different indexes"):
        conformal.conformal_var(a, 0.05, scale=b)


def test_dq_df_does_not_depend_on_where_exceptions_fall():
    dfs = set()
    for pos in (100, 248, 249):
        p = np.zeros(250)
        p[pos] = -5.0
        dfs.add(var.dynamic_quantile(p, 2.33, 0.01).detail["df"])
    assert dfs == {5}


def test_regulatory_counts_refuse_dropping_days():
    with pytest.raises(ValueError, match="cannot drop"):
        frtb.traffic_light(np.ones(250), np.zeros(250), missing="drop")
    assert frtb.traffic_light(np.ones(300), np.zeros(300), window=250.0).detail["observations"] == 250
    assert frtb.desk_backtest(np.ones(300), np.ones(300), np.zeros(300), window=None).detail["observations_counted"] == 300


def test_conditional_calibration_without_exceptions_and_a_scale():
    r = es.conditional_calibration(np.zeros(250), 2.0, 2.5, scale=np.ones(250))
    assert np.isnan(r.detail["p_general_two_sided"]) or r.detail["p_general_two_sided"] >= 0


def test_report_includes_du_escanciano_conditional():
    import vetted

    rng = np.random.default_rng(22)
    u = rng.random(500)
    u[[100, 101, 200, 201, 300, 301, 400, 401, 450, 451]] = 0.001  # tail events in pairs
    pnl = stats.norm.ppf(u)
    rep = vetted.validate_risk_model(pnl, -stats.norm.ppf(0.01), pit=u, n_sims=1000)
    cond = rep["Du-Escanciano conditional"]
    assert cond.p_value < 0.01
    assert cond in rep.rejected


def test_validate_strategy_refuses_variance_with_trial_returns():
    import vetted

    M = np.random.default_rng(23).normal(0, 0.01, (400, 8))
    with pytest.raises(ValueError, match="sharpe_variance"):
        vetted.validate_strategy(M[:, 0], M, sharpe_variance=0.5)
