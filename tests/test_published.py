"""Numbers printed in the papers and regulatory texts vetted implements.

None of the expected values below was computed by vetted.
"""

import numpy as np
import pytest
from scipy import stats

from vetted import es, frtb, overfitting


def test_deflated_sharpe_worked_example():
    # Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio", worked
    # example: N = 100 trials, variance of annualised Sharpe ratios 1/2,
    # 250 periods a year give a hurdle of 0.1132 per period; an annualised
    # Sharpe ratio of 2.5 over 1250 days with skewness -3 and kurtosis 10
    # then has DSR = 0.9004.
    hurdle = overfitting.expected_max_sharpe(100, 0.5 / 250)
    assert hurdle == pytest.approx(0.1132, abs=5e-5)
    dsr = overfitting._psr_from_moments(2.5 / np.sqrt(250), hurdle, -3.0, 10.0, 1250)
    assert dsr == pytest.approx(0.9004, abs=5e-5)


def test_basel_table_1_zones():
    # MAR32.9 Table 1: 250 days at 99%, green 0-4, amber 5-9, red 10 or more.
    z = frtb.binomial_zones(250, 0.01)
    assert z["green"] == (0, 4)
    assert z["amber"] == (5, 9)
    assert z["red_from"] == 10


def test_basel_cumulative_probabilities():
    # The cumulative probabilities printed beside the traffic light in the
    # Basel Committee's 1996 supervisory framework for backtesting (the basis
    # of MAR99): 4 exceptions 89.22%, 5 95.88%, 6 98.63%, 7 99.60%,
    # 8 99.89%, 9 99.97%, 10 99.99%.
    z = frtb.binomial_zones(250, 0.01)["cumulative_probability"]
    printed = {4: 89.22, 5: 95.88, 6: 98.63, 7: 99.60, 8: 99.89, 9: 99.97, 10: 99.99}
    for k, pct in printed.items():
        assert round(100 * z[k], 2) == pytest.approx(pct, abs=0.006), k


@pytest.mark.parametrize(
    "count, zone, multiplier",
    [(0, "green", 1.50), (4, "green", 1.50), (5, "amber", 1.70), (6, "amber", 1.76), (7, "amber", 1.83),
     (8, "amber", 1.88), (9, "amber", 1.92), (10, "red", 2.00), (17, "red", 2.00)],
)
def test_basel_table_1_multipliers(count, zone, multiplier):
    var = np.ones(250)
    pnl = np.zeros(250)
    pnl[:count] = -2.0
    r = frtb.traffic_light(var, pnl)
    assert r.detail["zone"] == zone
    assert r.detail["multiplier"] == multiplier


@pytest.mark.parametrize(
    "rho, ks, zone",
    [
        (0.81, 0.08, "green"),
        (0.80, 0.08, "amber"),  # "above 0.80" is needed for green
        (0.81, 0.09, "amber"),  # "below 0.09" is needed for green
        (0.70, 0.12, "amber"),  # red needs "less than 0.7" or "above 0.12"
        (0.69, 0.05, "red"),
        (0.95, 0.121, "red"),
    ],
)
def test_pla_table_2_boundaries(rho, ks, zone):
    # MAR32.42 Table 2
    assert frtb.pla_zone(rho, ks) == zone


def test_acerbi_szekely_z2_threshold_gaussian():
    # Acerbi and Szekely (2014), Table 4: for Gaussian P&L, T = 250 and
    # alpha = 2.5%, the 5% significance threshold of Z2 is -0.70. Simulate Z2
    # under a correct model and recover it.
    rng = np.random.default_rng(2014)
    T, alpha, n = 250, 0.025, 40_000
    var = -stats.norm.ppf(alpha)
    es_ = stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    x = rng.standard_normal((n, T))
    z2 = np.sum(x * (x < -var), axis=1) / (T * alpha * es_) + 1
    assert np.quantile(z2, 0.05) == pytest.approx(es.Z2_AMBER, abs=0.02)
    # and the package computes the same statistic path by path
    r = es.acerbi_szekely(x[0], var, es_, alpha)
    assert r.detail["z2"] == pytest.approx(z2[0], rel=1e-12)
