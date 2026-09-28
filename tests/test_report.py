"""The report layer: verdict logic, multiple testing, and every branch of
validate_risk_model and validate_strategy."""

import numpy as np
import pytest
from scipy import stats

import vetted
from vetted import TestResult, frtb
from vetted.report import Report


def _r(name, p, reject=None, **detail):
    return TestResult(
        name=name,
        statistic=0.0,
        p_value=p,
        reject=(p < 0.05) if reject is None else reject,
        significance=0.05,
        null="",
        reference=name,
        detail=detail,
    )


def test_holm_adjustment_on_known_values():
    rep = Report("t", [_r("a", 0.01), _r("b", 0.02), _r("c", 0.03), _r("d", 0.2)])
    assert rep.holm_p() == pytest.approx([0.04, 0.06, 0.06, 0.2])
    # Bonferroni would give 0.04, 0.08, 0.12, 0.8; raw would reject a, b, c
    assert [x.name for x in rep.rejected] == ["a"]
    assert rep.status == "fail"


def test_significance_is_respected():
    results = [_r("a", 0.01), _r("b", 0.02), _r("c", 0.03), _r("d", 0.2)]
    assert [x.name for x in Report("t", results, significance=0.07).rejected] == ["a", "b", "c"]
    assert Report("t", results, significance=0.01).rejected == []


def test_zones_are_outside_the_family_and_amber_only_flags():
    tl = _r("traffic", 0.001, reject=True, zone="amber")
    desk = _r("desk", float("nan"), reject=True, zone="ineligible")
    rep = Report("t", [tl, _r("a", 0.2)])
    assert rep.holm_p()[0] != rep.holm_p()[0]  # nan: not in the family
    assert rep.holm_p()[1] == pytest.approx(0.2)  # a family of one, unadjusted
    assert rep.passed and rep.status == "amber" and rep.flagged == [tl]
    rep2 = Report("t", [desk])
    assert not rep2.passed and rep2.rejected == [desk]
    rep3 = Report("t", [_r("red", float("nan"), reject=True, zone="red")])
    assert rep3.status == "fail"


def test_duplicate_names_are_handled_by_position():
    a, b = _r("x", 0.001), _r("x", 0.9)
    for order in ([a, b], [b, a]):
        rep = Report("t", order)
        assert rep.rejected == [a]


def test_missing_p_values_and_empty_reports():
    rep = Report("t", [_r("x", None, reject=False)])
    assert rep.holm_p() != [0.0]
    assert Report("t").passed is False
    assert Report("t").status == "no tests"


def test_to_dict_carries_both_raw_and_verdict_fields():
    rep = Report("t", [_r("a", 0.03), _r("b", 0.2)])
    d = rep.to_dict()
    row = d["results"][0]
    assert row["rejects_null"] is True and row["counts_against"] is False
    assert row["holm_p_value"] == pytest.approx(0.06)
    assert d["status"] == "pass"


# ------------------------------------------------------- validate_risk_model


def _t_model(nu=5):
    s = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(0.025, nu)
    return -stats.t.ppf(0.01, nu) * s, -q * s, stats.t.pdf(q, nu) / 0.025 * (nu + q**2) / (nu - 1) * s, s


def test_every_input_adds_its_tests():
    v99, v975, e975, s = _t_model()
    rng = np.random.default_rng(40)
    pnl = rng.standard_t(5, 1000) * s
    full = vetted.validate_risk_model(
        pnl,
        v99,
        v975,
        e975,
        actual_pnl=pnl * 1.01,
        risk_theoretical_pnl=pnl + 0.01 * rng.standard_normal(1000),
        pit=stats.t.cdf(pnl / s, 5),
        n_sims=500,
    )
    names = [r.name for r in full.results]
    for expected in (
        "Basel traffic light",
        "Kupiec POF (99%)",
        "Christoffersen conditional coverage (97.5%)",
        "Engle-Manganelli DQ (97.5%)",
        "FRTB desk backtest",
        "Acerbi-Szekely Z2",
        "Nolde-Ziegel conditional calibration (one-sided)",
        "McNeil-Frey exceedance residuals",
        "Du-Escanciano unconditional",
        "FRTB PLA test",
    ):
        assert expected in names, expected
    assert set(full["Basel traffic light"].detail["exceptions"]) == {"hypothetical", "actual"}
    assert full.status == "pass", str(full)
    assert "Kupiec POF (99%)" in full.to_markdown()


def test_an_ineligible_desk_fails_the_report():
    pnl = np.zeros(250)
    pnl[:13] = -5.0
    rep = vetted.validate_risk_model(pnl, np.full(250, 1.0), np.full(250, 0.5), n_sims=200)
    assert rep["FRTB desk backtest"].detail["zone"] == "ineligible"
    assert rep["FRTB desk backtest"] in rep.rejected


def test_a_red_pla_fails_the_report():
    rng = np.random.default_rng(41)
    pnl = rng.standard_normal(250)
    rep = vetted.validate_risk_model(pnl, 2.33, risk_theoretical_pnl=rng.standard_normal(250), n_sims=200)
    assert rep["FRTB PLA test"].detail["zone"] == "red"
    assert not rep.passed


def test_es_without_var_975_is_an_error():
    with pytest.raises(ValueError, match="var_975"):
        vetted.validate_risk_model(np.zeros(250), 1.0, es_975=1.5)


def test_underestimated_model_fails():
    v99, v975, e975, s = _t_model()
    pnl = np.random.default_rng(42).standard_t(5, 1000) * s * 1.5
    rep = vetted.validate_risk_model(pnl, v99, v975, e975, n_sims=500)
    assert rep.status == "fail"
    assert rep["Basel traffic light"].detail["zone"] in ("amber", "red")


# --------------------------------------------------------- validate_strategy


def test_strategy_with_a_count_needs_the_sharpe_variance():
    r = np.random.default_rng(43).normal(0.0005, 0.01, 1260)
    with pytest.raises(ValueError, match="sharpe_variance"):
        vetted.validate_strategy(r, 1000)
    rep = vetted.validate_strategy(r, 1000, sharpe_variance=0.3)
    assert rep["Deflated Sharpe ratio"].detail["trials"] == 1000
    assert rep.results and rep.status in ("pass", "fail")


def test_strategy_verdicts():
    rng = np.random.default_rng(22)
    M = rng.normal(0, 0.01, (1600, 40))
    best = M[:, np.argmax(M.mean(0) / M.std(0, ddof=1))]
    rep = vetted.validate_strategy(best, M)
    assert not rep.passed
    d = rep.to_dict()["results"][0]
    assert d["name"] == "Deflated Sharpe ratio" and d["counts_against"] and not d["rejects_null"]
    M[:, 0] += 0.003
    rep2 = vetted.validate_strategy(M[:, 0], M)
    assert rep2.passed, str(rep2)
    # the deflated Sharpe uses every trial, not an effective count
    assert rep2["Deflated Sharpe ratio"].detail["trials"] == 40


def test_strategy_without_trials_uses_psr():
    r = np.random.default_rng(44).normal(0.002, 0.01, 1000)
    rep = vetted.validate_strategy(r)
    assert rep.passed and rep.results[0].name == "Probabilistic Sharpe ratio"
    assert not vetted.validate_strategy(-r).passed


def test_frtb_traffic_light_counts_last_250_by_default():
    pnl = np.zeros(500)
    pnl[:20] = -5.0  # old exceptions, outside the regulatory window
    assert frtb.traffic_light(np.ones(500), pnl).detail["zone"] == "green"
    assert frtb.traffic_light(np.ones(500), pnl, window=None).statistic == 20
