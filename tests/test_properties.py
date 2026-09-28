"""Behaviour that must hold whatever the implementation: sizes under the
null, power against known failures, edge cases, input validation, and
cross-checks against independent Python implementations."""

import numpy as np
import pytest
from scipy import stats

import vetted
from vetted import compare, conformal, es, frtb, overfitting, var


def _normal_model(rng, n, scale=1.0):
    pnl = rng.standard_normal(n)
    v99 = -stats.norm.ppf(0.01) * scale * np.ones(n)
    return pnl, v99


# ----------------------------------------------------------------- VaR tests


@pytest.mark.parametrize("test", ["kupiec", "christoffersen", "dq"])
def test_var_tests_hold_their_size(test):
    # A correct 97.5% VaR over 1000 days should be rejected about 5% of the
    # time at the 5% level.
    rng = np.random.default_rng({"kupiec": 1, "christoffersen": 2, "dq": 3}[test])
    rejections = 0
    reps = 400
    q = -stats.norm.ppf(0.025)
    for _ in range(reps):
        pnl = rng.standard_normal(1000)
        if test == "kupiec":
            r = var.kupiec(pnl, q, 0.025)
        elif test == "christoffersen":
            r = var.christoffersen(pnl, q, 0.025)
        else:
            r = var.dynamic_quantile(pnl, q * np.ones(1000), 0.025)
        rejections += r.reject
    assert 0.02 <= rejections / reps <= 0.10


def test_var_tests_detect_underestimation():
    rng = np.random.default_rng(4)
    pnl = rng.standard_normal(1000) * 1.4  # true volatility 40% higher than modelled
    q = -stats.norm.ppf(0.01)
    assert var.kupiec(pnl, q, 0.01).reject
    assert var.christoffersen(pnl, q, 0.01).reject


def test_christoffersen_detects_clustering_kupiec_misses():
    # Exactly the right number of exceptions, all on consecutive days.
    pnl = np.zeros(1000)
    pnl[500:510] = -5.0
    r_uc = var.kupiec(pnl, 1.0, 0.01)
    r_cc = var.christoffersen(pnl, 1.0, 0.01)
    assert not r_uc.reject
    assert r_cc.detail["reject_ind"]
    assert r_cc.reject


def test_kupiec_edge_cases():
    assert var.kupiec(np.zeros(250), 1.0, 0.01).detail["exceptions"] == 0
    r0 = var.kupiec(np.zeros(250), 1.0, 0.01)
    assert r0.statistic == pytest.approx(-2 * 250 * np.log(0.99))
    r_all = var.kupiec(-np.ones(10) * 5, 1.0, 0.01)
    assert r_all.statistic == pytest.approx(-2 * 10 * np.log(0.01))
    assert np.isfinite(var.christoffersen(np.zeros(250), 1.0, 0.01).statistic)


def test_negative_var_is_rejected():
    with pytest.raises(ValueError, match="positive loss"):
        var.kupiec(np.zeros(10), -1.0, 0.01)


def test_missing_data_policies():
    pnl = np.array([0.0, np.nan, -3.0, 0.5])
    with pytest.raises(ValueError, match="missing"):
        var.exceptions(pnl, 1.0)
    assert var.exceptions(pnl, 1.0, missing="drop").tolist() == [False, True, False]
    assert var.exceptions(pnl, 1.0, missing="exception").tolist() == [False, True, True, False]
    # the regulatory count treats a missing VaR the same way (MAR32.5)
    v = np.array([1.0, 1.0, np.nan, 1.0])
    assert frtb.traffic_light(v, np.zeros(4)).statistic == 1


def test_a_loss_equal_to_var_is_not_an_exception():
    # MAR32.5: an exception is a loss that *exceeds* the VaR
    pnl = np.array([-1.0, -1.0 - 1e-12, 0.0])
    assert var.exceptions(pnl, 1.0).tolist() == [False, True, False]
    assert frtb.traffic_light(np.ones(3), pnl).statistic == 1


def test_traffic_light_takes_greater_of_actual_and_hypothetical():
    v = np.ones(250)
    hpl = np.zeros(250)
    apl = np.zeros(250)
    hpl[:3] = -2
    apl[:6] = -2
    r = frtb.traffic_light(v, hpl, apl)
    assert r.statistic == 6
    assert r.detail["exceptions"] == {"hypothetical": 3, "actual": 6}
    assert r.detail["zone"] == "amber"


@pytest.mark.parametrize("n99, n975, fail", [(12, 30, False), (13, 0, True), (0, 31, True)])
def test_desk_backtest_limits(n99, n975, fail):
    hpl = np.zeros(250)
    v99 = np.full(250, 10.0)
    v975 = np.full(250, 10.0)
    hpl[:n99] = -20
    v975[:n99] = 30  # keep the 99% exceptions from also counting at 97.5%
    hpl[100 : 100 + n975] = -20
    v99[100 : 100 + n975] = 30
    r = frtb.desk_backtest(v99, v975, hpl)
    assert r.detail["99"]["exceptions"] == n99
    assert r.detail["97.5"]["exceptions"] == n975
    assert r.reject == fail


def test_pla_perfect_and_broken_models():
    rng = np.random.default_rng(5)
    hpl = rng.standard_normal(250)
    assert frtb.pla_test(hpl + 0.01 * rng.standard_normal(250), hpl).detail["zone"] == "green"
    assert frtb.pla_test(rng.standard_normal(250), hpl).detail["zone"] == "red"
    # KS metric per MAR32.39-32.41 is the two-sample KS distance
    a, b = rng.standard_normal(250), rng.standard_normal(250) + 0.3
    assert frtb.pla_test(a, b).detail["ks"] == pytest.approx(stats.ks_2samp(a, b).statistic)


# ------------------------------------------------------------------ ES tests


def _t_model(nu=5):
    s = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(0.025, nu)
    var_ = -q * s
    es_ = stats.t.pdf(q, nu) / 0.025 * (nu + q**2) / (nu - 1) * s
    return var_, es_, s


def test_acerbi_szekely_monte_carlo_pvalue_is_calibrated():
    nu = 5
    var_, es_, s = _t_model(nu)
    rng = np.random.default_rng(6)

    def sampler(g, n):
        return g.standard_t(nu, (n, 250)) * s

    pvals = []
    for _ in range(60):
        pnl = rng.standard_t(nu, 250) * s
        r = es.acerbi_szekely(pnl, var_, es_, sampler=sampler, n_sims=2000, seed=int(rng.integers(1e9)))
        pvals.append(r.p_value)
    # p-values under the null are roughly uniform
    assert 0.3 < np.mean(pvals) < 0.7
    # and a model that understates volatility by 30% is caught
    pnl = rng.standard_t(nu, 250) * s * 1.3
    assert es.acerbi_szekely(pnl, var_, es_, sampler=sampler, n_sims=4000).p_value < 0.05


@pytest.mark.parametrize("T", [125, 500, 1000])
@pytest.mark.parametrize("nu", [5, 100])
def test_scaled_z2_threshold_keeps_its_size(T, nu):
    s = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(0.025, nu)
    var_, es_ = -q * s, stats.t.pdf(q, nu) / 0.025 * (nu + q**2) / (nu - 1) * s
    rng = np.random.default_rng(T + nu)
    x = rng.standard_t(nu, (10_000, T)) * s
    z2 = np.sum(x * (x < -var_), axis=1) / (T * 0.025 * es_) + 1
    amber = es.acerbi_szekely(x[0], var_, es_).detail["thresholds"]["amber"]
    assert 0.035 <= np.mean(z2 <= amber) <= 0.07


def test_z2_relationship_to_z1():
    # Acerbi and Szekely (2014), eq. 7: Z2 = 1 - (1 - Z1) N / (T alpha) when
    # ES is constant.
    rng = np.random.default_rng(7)
    var_, es_, s = _t_model()
    pnl = rng.standard_t(5, 500) * s * 1.2
    r = es.acerbi_szekely(pnl, var_, es_)
    n = r.detail["exceptions"]
    assert r.detail["z2"] == pytest.approx(1 - (1 - r.detail["z1"]) * n / (500 * 0.025), rel=1e-12)


def test_es_below_var_is_rejected():
    with pytest.raises(ValueError, match="at least var"):
        es.acerbi_szekely(np.zeros(10), 2.0, 1.0)


def test_du_escanciano_moments_and_size():
    # E[H] = alpha / 2 and Var[H] = alpha (1/3 - alpha/4) under U(0,1)
    a = 0.025
    u = np.linspace(0, 1, 2_000_001)
    h = (a - u) * (u <= a) / a
    assert h.mean() == pytest.approx(a / 2, rel=1e-4)
    assert h.var() == pytest.approx(a * (1 / 3 - a / 4), rel=1e-4)
    rng = np.random.default_rng(8)
    rej_u = rej_c = 0
    for _ in range(300):
        r = es.du_escanciano(rng.random(1000), a)
        rej_u += r.reject
        rej_c += r.detail["reject_conditional"]
    assert 0.02 <= rej_u / 300 <= 0.10
    assert 0.01 <= rej_c / 300 <= 0.12


def test_du_escanciano_detects_heavy_tail():
    # the model says N(0,1); the truth is t(3)
    rng = np.random.default_rng(9)
    pnl = rng.standard_t(3, 1000)
    assert es.du_escanciano(stats.norm.cdf(pnl), 0.025).reject


def test_fz0_prefers_the_true_model():
    rng = np.random.default_rng(10)
    var_, es_, s = _t_model()
    pnl = rng.standard_t(5, 5000) * s
    good = es.fz0_loss(pnl, var_, es_)
    bad = es.fz0_loss(pnl, var_ * 0.7, es_ * 0.7)
    r = compare.diebold_mariano(good, bad, alternative="less")
    assert r.reject and r.detail["better"] == "a"


def test_diebold_mariano_matches_statsmodels_hac():
    # With horizon 1 the statistic before the HLN factor is the t-ratio of the
    # mean loss difference with the plain variance.
    import statsmodels.api as sm

    rng = np.random.default_rng(11)
    a, b = rng.random(300), rng.random(300) + 0.05
    r = compare.diebold_mariano(a, b)
    d = a - b
    ols = sm.OLS(d, np.ones_like(d)).fit()
    n = d.size
    raw = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / n)
    assert r.statistic == pytest.approx(raw * np.sqrt((n - 1) / n), rel=1e-12)
    assert np.sign(r.statistic) == np.sign(ols.params[0])


# --------------------------------------------------------------- overfitting


def test_selected_noise_is_not_evidence():
    rng = np.random.default_rng(12)
    psr_hits = dsr_hits = 0
    for _ in range(60):
        M = rng.normal(0, 0.01, (504, 30))
        best = M[:, np.argmax(M.mean(0) / M.std(0, ddof=1))]
        psr_hits += overfitting.probabilistic_sharpe(best) > 0.95
        dsr_hits += overfitting.deflated_sharpe(best, M).reject
    assert psr_hits / 60 > 0.5
    assert dsr_hits / 60 <= 0.05


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(13)
    best, v = [], []
    for _ in range(150):
        x = rng.normal(0, 0.01, (1000, 100))
        s = x.mean(0) / x.std(0, ddof=1)
        best.append(s.max())
        v.append(s.var(ddof=1))
    assert np.mean(best) == pytest.approx(overfitting.expected_max_sharpe(100, np.mean(v)), rel=0.03)


def test_deflated_sharpe_from_trial_matrix_equals_count_form():
    rng = np.random.default_rng(24)
    M = rng.normal(0.0003, 0.01, (750, 25))
    srs = M.mean(0) / M.std(0, ddof=1) * np.sqrt(252)
    a = overfitting.deflated_sharpe(M[:, 0], M)
    b = overfitting.deflated_sharpe(M[:, 0], 25, sharpe_variance=srs.var(ddof=1))
    assert a.statistic == pytest.approx(b.statistic, rel=1e-12)
    assert a.detail["hurdle_sharpe"] == pytest.approx(b.detail["hurdle_sharpe"], rel=1e-12)


def test_one_trial_is_the_probabilistic_sharpe():
    rng = np.random.default_rng(14)
    r = rng.normal(0.001, 0.01, 500)
    assert overfitting.deflated_sharpe(r, 1).statistic == pytest.approx(overfitting.probabilistic_sharpe(r))


def test_min_track_record_inverts_psr():
    rng = np.random.default_rng(15)
    r = rng.standard_t(4, 800) * 0.01 + 0.001
    sr, sk, ku, _ = overfitting._moments(r)
    for c in (0.9, 0.95, 0.99):
        n = overfitting.min_track_record(r, confidence=c)
        assert overfitting._psr_from_moments(sr, 0.0, sk, ku, n) == pytest.approx(c, rel=1e-12)
    assert overfitting.min_track_record(-np.abs(r)) == float("inf")


def test_pbo_noise_and_skill():
    rng = np.random.default_rng(16)
    noise = rng.normal(0, 0.01, (1600, 20))
    assert 0.3 < overfitting.pbo(noise).statistic < 0.7
    skill = noise.copy()
    skill[:, 7] += 0.002
    assert overfitting.pbo(skill).statistic < 0.1


@pytest.mark.parametrize("method, sm_method", [("holm", "holm"), ("bhy", "fdr_by"), ("bonferroni", "bonferroni")])
def test_haircut_adjustments_match_statsmodels(method, sm_method):
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(17)
    srs = rng.normal(0.3, 0.4, 40)
    srs[5] = 1.6
    n, periods = 1260, 252
    p = 2 * stats.norm.sf(np.abs(srs * np.sqrt(n / periods)))
    adj = multipletests(p, method=sm_method)[1]
    for i in range(srs.size):  # every strategy, not only the best
        out = overfitting.haircut_sharpe(srs[i], n, srs, method=method, periods=periods)
        assert out["adjusted_p_value"] == pytest.approx(adj[i], rel=1e-10), i
    out = overfitting.haircut_sharpe(srs[5], n, srs, method=method, periods=periods)
    assert 0 < out["haircut_sharpe"] < srs[5]


def test_effective_trials():
    rng = np.random.default_rng(18)
    base = rng.normal(size=(2000, 1))
    same = np.repeat(base, 10, axis=1) + 1e-9 * rng.normal(size=(2000, 10))
    assert overfitting.effective_trials(same) == pytest.approx(1.0, abs=0.01)
    indep = rng.normal(size=(2000, 10))
    assert overfitting.effective_trials(indep) == pytest.approx(10.0, rel=0.05)


# ----------------------------------------------------------------- conformal


def test_conformal_var_coverage_guarantee():
    # exchangeable heavy-tailed data: P(loss > VaR) <= alpha, averaged over
    # many calibrations, with 99 calibration points at alpha = 5%.
    rng = np.random.default_rng(19)
    breaches = []
    for _ in range(4000):
        x = rng.standard_t(2, 100)
        v = conformal.conformal_var(x[:99], 0.05)
        breaches.append(x[99] < -v)
    assert np.mean(breaches) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 4000)


def test_conformal_quantile_index():
    s = np.arange(1, 20)  # n = 19
    # ceil(20 * 0.9) = 18th smallest
    assert conformal.conformal_quantile(s, 0.1) == 18
    assert conformal.conformal_quantile(np.arange(5), 0.1) == float("inf")


def test_adaptive_var_keeps_coverage_through_a_regime_change():
    rng = np.random.default_rng(20)
    pnl = np.concatenate([rng.standard_normal(1500), 3 * rng.standard_normal(1500)])
    static = conformal.conformal_var(pnl[:1500], 0.01)
    static_rate = np.mean(pnl[1500:] < -static)
    assert static_rate > 0.05  # a VaR calibrated before the shift breaks after it
    # a 1000-day rolling window adapts too slowly on its own (gamma = 0) ...
    rolling = conformal.adaptive_var(pnl, 0.01, window=1000, gamma=0.0)
    assert rolling["exception_rate"] > 0.02
    # ... and ACI steers it back to the target, inside its guarantee
    out = conformal.adaptive_var(pnl, 0.01, window=1000, gamma=0.02)
    assert abs(out["exception_rate"] - 0.01) <= out["bound"]
    assert abs(out["exception_rate"] - 0.01) < 0.002


# -------------------------------------------------------------------- report


def test_validate_risk_model_report():
    rng = np.random.default_rng(21)
    var_, es_, s = _t_model()
    v99 = -stats.t.ppf(0.01, 5) * s
    pnl = rng.standard_t(5, 1000) * s
    good = vetted.validate_risk_model(pnl, v99, var_, es_, pit=stats.t.cdf(pnl / s, 5))
    assert good.status == "pass", str(good)
    md = good.to_markdown()
    assert "Kupiec POF (99%)" in md and "Acerbi-Szekely Z2" in md and "References" in md
    bad = vetted.validate_risk_model(pnl * 1.5, v99, var_, es_)
    assert not bad.passed
    assert bad["Basel traffic light"].detail["zone"] in ("amber", "red")
    assert "FAILED" in str(bad)


def test_validate_strategy_report():
    rng = np.random.default_rng(22)
    M = rng.normal(0, 0.01, (1600, 40))
    best = M[:, np.argmax(M.mean(0) / M.std(0, ddof=1))]
    rep = vetted.validate_strategy(best, M)
    assert not rep.passed
    assert rep.facts["trials"] == 40
    M[:, 0] += 0.003
    rep2 = vetted.validate_strategy(M[:, 0], M)
    assert rep2.passed, str(rep2)


def test_nolde_ziegel_two_sided_is_oversized_on_short_samples():
    # Pins the warning in vetted.es.conditional_calibration: on a correct
    # model the two-sided simple test rejects far more than 5% at 250 days,
    # while the one-sided version holds its level.
    var_, es_, s = _t_model()
    rng = np.random.default_rng(25)
    two = one = 0
    reps = 200
    for _ in range(reps):
        x = rng.standard_t(5, 250) * s
        r = es.conditional_calibration(x, var_, es_)
        two += r.reject
        one += r.detail["p_simple_one_sided"] < 0.05
    assert two / reps > 0.15
    assert one / reps < 0.09


def test_report_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(26)
    var_, es_, s = _t_model()
    v99 = -stats.t.ppf(0.01, 5) * s
    pnl = rng.standard_t(5, 1000) * s * 1.1
    rep = vetted.validate_risk_model(pnl, v99, var_, es_)
    adj = rep.adjusted_p
    names = list(adj)
    raw = [rep[n].p_value for n in names]
    expected = multipletests(raw, method="holm")[1]
    assert [adj[n] for n in names] == pytest.approx(list(expected), rel=1e-12)
    assert "Basel traffic light" not in adj  # zones are not in the family


def test_report_false_alarm_rate_on_correct_models():
    # With about eight statistical tests, "any test rejects" would flag a
    # correct model far more than 5% of the time; the Holm verdict should not.
    rng = np.random.default_rng(27)
    var_, es_, s = _t_model()
    v99 = -stats.t.ppf(0.01, 5) * s
    raw_any = holm_any = 0
    reps = 60
    for i in range(reps):
        pnl = rng.standard_t(5, 1000) * s
        rep = vetted.validate_risk_model(pnl, v99, var_, es_, n_sims=500, seed=i)
        stat = [r for r in rep.results if "zone" not in r.detail]
        raw_any += any(r.reject for r in stat)
        holm_any += any(rep.counts_against(r) for r in stat)
    assert raw_any / reps > holm_any / reps
    assert holm_any / reps <= 0.10


def test_pbo_fast_path_equals_generic_loop():
    from vetted.overfitting import _sharpe_cols

    rng = np.random.default_rng(28)
    M = rng.normal(0.0002, 0.01, (801, 15))
    M[:, 4] += 0.0006
    fast = overfitting.pbo(M, n_blocks=10)
    slow = overfitting.pbo(M, n_blocks=10, metric=_sharpe_cols)
    assert fast.statistic == slow.statistic
    np.testing.assert_allclose(fast.detail["oos_rank_of_selected"], slow.detail["oos_rank_of_selected"])
    np.testing.assert_allclose(fast.detail["logits"], slow.detail["logits"], rtol=1e-9)


def test_dq_monte_carlo_fixes_small_sample_size():
    # A correct 99% VaR over 250 days: the chi-squared DQ p-value over-rejects,
    # the Monte Carlo one holds its level.
    rng = np.random.default_rng(29)
    q = -stats.norm.ppf(0.01) * np.ones(250)
    asym = mc = 0
    reps = 300
    for i in range(reps):
        pnl = rng.standard_normal(250)
        asym += var.dynamic_quantile(pnl, q, 0.01).reject
        mc += var.dynamic_quantile(pnl, q, 0.01, pvalue="monte_carlo", n_sims=400, seed=i).reject
    assert asym / reps > 0.06
    assert mc / reps <= 0.06


def test_monte_carlo_pvalues_agree_with_chi2_in_large_samples():
    rng = np.random.default_rng(30)
    pnl = rng.standard_normal(5000) * 1.05
    q = -stats.norm.ppf(0.025)
    for fn in (var.kupiec, var.christoffersen):
        a = fn(pnl, q, 0.025)
        m = fn(pnl, q, 0.025, pvalue="monte_carlo", n_sims=20_000)
        assert m.statistic == a.statistic
        assert m.p_value == pytest.approx(a.p_value, abs=0.02)
    # seeded, so repeatable
    assert var.kupiec(pnl, q, 0.025, pvalue="monte_carlo", seed=3).p_value == var.kupiec(
        pnl, q, 0.025, pvalue="monte_carlo", seed=3
    ).p_value


def test_amber_zone_flags_but_does_not_fail():
    # MAR32.11: amber can come from an accurate model. Five exceptions in the
    # last 250 days of an otherwise well-behaved record is amber, not a fail.
    rng = np.random.default_rng(31)
    q = -stats.norm.ppf(0.01)
    pnl = rng.standard_normal(1000)
    pnl[pnl < -q] = 0.0  # clear the random exceptions
    pnl[[760, 820, 880, 940, 990]] = -4.0  # five, spread out, in the last 250 days
    pnl[[100, 300, 500, 700]] = -4.0
    rep = vetted.validate_risk_model(pnl, q, n_sims=2000)
    assert rep["Basel traffic light"].detail["zone"] == "amber"
    assert rep.flagged and rep.passed
    assert rep.status == "amber"
    assert "amber flags" in str(rep)
