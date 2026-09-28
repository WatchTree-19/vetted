"""Named reference estimators.

Most portfolio metrics have several published estimators. A library that
disagrees with one formula is not wrong; a library whose output matches *no*
published estimator, or matches one its documentation does not claim, is.
This module therefore implements every common convention side by side, with
a name a maintainer can put in a docstring.

Canonical signs, used for every value in this package:

* VaR and ES are returned as *returns* (a loss is negative), as in R
  PerformanceAnalytics.
* Maximum drawdown and the drawdown-based indices are returned as positive
  magnitudes, as in R PerformanceAnalytics.

The conformance checker accepts a library that uses the opposite sign and
reports the flip separately, so sign is never confused with the estimator.

Every convention marked ``PA`` below is checked in the test suite against
R PerformanceAnalytics output to 1e-10 relative tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm


def _clean(r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return r[np.isfinite(r)]


# --------------------------------------------------------------------------
# Drawdown path
# --------------------------------------------------------------------------

def drawdown_path(r, geometric: bool = True, initial_peak: bool = True) -> np.ndarray:
    """Drawdown at each period, as non-negative magnitudes.

    geometric=True compounds wealth, prod(1 + r); geometric=False uses simple
    cumulative wealth, 1 + cumsum(r), as R PerformanceAnalytics does.
    initial_peak=True counts the starting capital (1.0) as the first peak, so
    a loss in the very first period is a drawdown. initial_peak=False starts
    the running peak at the first period's wealth.
    """
    r = _clean(r)
    wealth = np.cumprod(1.0 + r) if geometric else 1.0 + np.cumsum(r)
    if initial_peak:
        peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    else:
        peak = np.maximum.accumulate(wealth)
    return 1.0 - wealth / peak


# --------------------------------------------------------------------------
# Estimators. Each takes (r, ann) and returns a float in canonical sign.
# --------------------------------------------------------------------------

def _vol(ddof):
    def f(r, ann):
        r = _clean(r)
        return float(r.std(ddof=ddof) * np.sqrt(ann))
    return f


def _ret_geometric(r, ann):
    r = _clean(r)
    return float(np.prod(1.0 + r) ** (ann / len(r)) - 1.0)


def _ret_arithmetic(r, ann):
    return float(_clean(r).mean() * ann)


def _sharpe_arith(ddof):
    def f(r, ann):
        r = _clean(r)
        sd = r.std(ddof=ddof)
        return float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")
    return f


def _sharpe_geometric(r, ann):
    sd = _vol(1)(r, ann)
    return _ret_geometric(r, ann) / sd if sd > 0 else float("nan")


def _downside_dev(method: str, mar: float = 0.0, ddof: int = 0):
    """method 'full': divide by all n; 'subset': divide by the count below MAR."""
    def f(r, ann=1):
        r = _clean(r)
        d = np.minimum(r - mar, 0.0)
        if method == "full":
            denom = len(r) - ddof
        else:
            denom = int((r < mar).sum()) - ddof
        if denom <= 0:
            # No observation below MAR: the downside is empty. R
            # PerformanceAnalytics reports 0 here rather than 0/0.
            return 0.0 if method == "subset" and (d ** 2).sum() == 0 else float("nan")
        return float(np.sqrt((d ** 2).sum() / denom))
    return f


def _sortino(method: str, ddof: int = 0, annualise: bool = True, mar: float = 0.0):
    dd = _downside_dev(method, mar=mar, ddof=ddof)

    def f(r, ann):
        r = _clean(r)
        d = dd(r)
        if not np.isfinite(d) or d == 0:
            return float("nan")
        v = (r.mean() - mar) / d
        return float(v * np.sqrt(ann)) if annualise else float(v)
    return f


def _sortino_losers_std(r, ann):
    """Standard deviation of the losing periods only: a common slip, named so
    it can be identified rather than flagged forever."""
    r = _clean(r)
    losers = r[r < 0]
    if len(losers) < 2:
        return float("nan")
    sd = losers.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")


def _mdd(geometric, initial_peak):
    def f(r, ann=1):
        return float(drawdown_path(r, geometric, initial_peak).max())
    return f


def _calmar(r, ann):
    mdd = _mdd(True, True)(r)
    return _ret_geometric(r, ann) / mdd if mdd > 0 else float("nan")


def _var_hist_interp(a):
    def f(r, ann=1):
        return float(np.quantile(_clean(r), a, method="linear"))
    return f


def _var_order_stat(a):
    def f(r, ann=1):
        s = np.sort(_clean(r))
        return float(s[max(0, int(np.ceil(a * len(s))) - 1)])
    return f


def _var_order_stat_upper(a):
    """The (floor(a n) + 1)-th smallest return. Equals the lower order
    statistic unless a n is a whole number, where it is one rank higher."""
    def f(r, ann=1):
        s = np.sort(_clean(r))
        return float(s[min(len(s) - 1, int(np.floor(a * len(s))))])
    return f


def _var_gaussian(a, ddof=1):
    def f(r, ann=1):
        r = _clean(r)
        return float(r.mean() + r.std(ddof=ddof) * norm.ppf(a))
    return f


def _cornish_fisher_z(r, a):
    r = _clean(r)
    s = _skew_moment(r)
    k = _kurt_excess_moment(r)
    z = norm.ppf(a)
    return (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36)


def _var_cornish_fisher(a, ddof=1):
    def f(r, ann=1):
        r = _clean(r)
        return float(r.mean() + r.std(ddof=ddof) * _cornish_fisher_z(r, a))
    return f


def _es_hist_below_interp_var(a):
    """Mean of returns at or below the interpolated historical VaR."""
    def f(r, ann=1):
        r = _clean(r)
        v = np.quantile(r, a, method="linear")
        return float(r[r <= v].mean())
    return f


def _es_gaussian(a, ddof=1):
    def f(r, ann=1):
        r = _clean(r)
        return float(r.mean() - r.std(ddof=ddof) * norm.pdf(norm.ppf(a)) / a)
    return f


def _es_rockafellar_uryasev(a):
    def f(r, ann=1):
        s = np.sort(_clean(r))
        v = s[max(0, int(np.ceil(a * len(s))) - 1)]
        return float(v - np.maximum(v - s, 0.0).mean() / a)
    return f


def _es_naive_floor(a):
    """Mean of the floor(a*n)+1 smallest returns."""
    def f(r, ann=1):
        s = np.sort(_clean(r))
        k = int(np.floor(a * len(s)))
        return float(s[: k + 1].mean())
    return f


def _es_hybrid_parametric_threshold(a):
    """Empirical mean below a *parametric Gaussian* VaR threshold. Internally
    inconsistent as an estimator; named so it can be recognised."""
    def f(r, ann=1):
        r = _clean(r)
        v = r.mean() + r.std(ddof=1) * norm.ppf(a)
        tail = r[r < v]
        return float(tail.mean()) if len(tail) else float("nan")
    return f


def _omega(L=0.0):
    def f(r, ann=1):
        r = _clean(r)
        up = np.maximum(r - L, 0.0).sum()
        dn = np.maximum(L - r, 0.0).sum()
        return float(up / dn) if dn > 0 else float("nan")
    return f


def _ulcer(r, ann=1):
    d = drawdown_path(r, True, True)
    return float(np.sqrt((d ** 2).mean()))


def _pain(r, ann=1):
    return float(drawdown_path(r, True, True).mean())


def _skew_moment(r, ann=1):
    r = _clean(r)
    m = r.mean()
    m2 = ((r - m) ** 2).mean()
    return float(((r - m) ** 3).mean() / m2 ** 1.5)


def _skew_sample(r, ann=1):
    r = _clean(r)
    n = len(r)
    return float(_skew_moment(r) * np.sqrt(n * (n - 1)) / (n - 2))


def _skew_pa_sample(r, ann=1):
    """What R PerformanceAnalytics 2.1.0 computes for skewness(method="sample"):
    n / ((n-1)(n-2)) x sum(((x - mean) / sd_pop)^3). Its documentation states
    the sample standard deviation, but the code divides by the population one,
    so it matches neither g1 nor the standard adjusted G1."""
    r = _clean(r)
    n = len(r)
    return float(_skew_moment(r) * n * n / ((n - 1) * (n - 2)))


def _kurt_excess_moment(r, ann=1):
    r = _clean(r)
    m = r.mean()
    m2 = ((r - m) ** 2).mean()
    return float(((r - m) ** 4).mean() / m2 ** 2 - 3.0)


def _kurt_sample_excess(r, ann=1):
    r = _clean(r)
    n = len(r)
    g2 = _kurt_excess_moment(r)
    return float(((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3)))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Convention:
    metric: str
    name: str
    fn: Callable
    description: str
    oracle: str | None = None  # R PerformanceAnalytics call it must reproduce
    signed: str = "return"     # "return": loss negative; "magnitude": loss positive; "ratio"


A = 0.05

CONVENTIONS: list[Convention] = [
    Convention("volatility_annual", "std_ddof1", _vol(1), "sample std (n-1) x sqrt(periods)", "StdDev.annualized(R, scale)"),
    Convention("volatility_annual", "std_ddof0", _vol(0), "population std (n) x sqrt(periods)"),

    Convention("return_annual", "geometric", _ret_geometric, "prod(1+r)^(periods/n) - 1 (CAGR)", "Return.annualized(R, scale, geometric=TRUE)"),
    Convention("return_annual", "arithmetic", _ret_arithmetic, "mean(r) x periods", "Return.annualized(R, scale, geometric=FALSE)"),

    Convention("sharpe_annual", "arithmetic_ddof1", _sharpe_arith(1), "mean/std(n-1) x sqrt(periods)", "SharpeRatio.annualized(R, 0, scale, geometric=FALSE)", "ratio"),
    Convention("sharpe_annual", "arithmetic_ddof0", _sharpe_arith(0), "mean/std(n) x sqrt(periods)", None, "ratio"),
    Convention("sharpe_annual", "geometric", _sharpe_geometric, "CAGR / (std(n-1) x sqrt(periods))", "SharpeRatio.annualized(R, 0, scale, geometric=TRUE)", "ratio"),

    Convention("downside_deviation", "full", _downside_dev("full"), "sqrt(sum(min(r,0)^2) / n), all periods (Sortino and Price 1994)", "DownsideDeviation(R, MAR=0, method='full')", "ratio"),
    Convention("downside_deviation", "subset", _downside_dev("subset"), "sqrt(sum(min(r,0)^2) / #losing periods)", "DownsideDeviation(R, MAR=0, method='subset')", "ratio"),

    Convention("sortino_period", "full", _sortino("full", annualise=False), "mean / full downside deviation, per period", "SortinoRatio(R, MAR=0)", "ratio"),
    Convention("sortino_annual", "full_ddof0", _sortino("full", 0), "mean / sqrt(sum(min(r,0)^2)/n) x sqrt(periods)", None, "ratio"),
    Convention("sortino_annual", "full_ddof1", _sortino("full", 1), "mean / sqrt(sum(min(r,0)^2)/(n-1)) x sqrt(periods)", None, "ratio"),
    Convention("sortino_annual", "subset", _sortino("subset", 0), "mean / sqrt(sum(min(r,0)^2)/#losers) x sqrt(periods)", None, "ratio"),
    Convention("sortino_annual", "losers_std_slip", _sortino_losers_std, "mean / std of losing periods only (a slip, not a published estimator)", None, "ratio"),

    Convention("max_drawdown", "geometric_initial_peak", _mdd(True, True), "compounded wealth, starting capital counts as a peak", "maxDrawdown(R, geometric=TRUE)", "magnitude"),
    Convention("max_drawdown", "geometric_no_initial_peak", _mdd(True, False), "compounded wealth, first peak is the first period's wealth", None, "magnitude"),
    Convention("max_drawdown", "simple_initial_peak", _mdd(False, True), "simple wealth 1+cumsum(r), starting capital counts as a peak", "maxDrawdown(R, geometric=FALSE)", "magnitude"),

    Convention("calmar", "cagr_over_mdd", _calmar, "CAGR / geometric max drawdown (initial peak)", "CalmarRatio(R, scale)", "ratio"),

    Convention("var_95", "historical_interpolated", _var_hist_interp(A), "linear-interpolated 5% quantile (R type 7, numpy default)", "VaR(R, p=0.95, method='historical')"),
    Convention("var_95", "historical_order_statistic", _var_order_stat(A), "the ceil(0.05 n)-th smallest return (lower quantile, inverted CDF)"),
    Convention("var_95", "historical_order_statistic_upper", _var_order_stat_upper(A), "the (floor(0.05 n)+1)-th smallest return (upper quantile)"),
    Convention("var_95", "gaussian_ddof0", _var_gaussian(A, 0), "mean + z(0.05) x std(n)", "VaR(R, p=0.95, method='gaussian')"),
    Convention("var_95", "gaussian_ddof1", _var_gaussian(A, 1), "mean + z(0.05) x std(n-1)"),
    Convention("var_95", "cornish_fisher_ddof0", _var_cornish_fisher(A, 0), "mean + z_CF x std(n), moment skew and excess kurtosis", "VaR(R, p=0.95, method='modified')"),
    Convention("var_95", "cornish_fisher_ddof1", _var_cornish_fisher(A, 1), "mean + z_CF x std(n-1), moment skew and excess kurtosis"),

    Convention("es_95", "historical_below_interpolated_var", _es_hist_below_interp_var(A), "mean of returns at or below the interpolated 5% quantile", "ES(R, p=0.95, method='historical')"),
    Convention("es_95", "gaussian_ddof0", _es_gaussian(A, 0), "mean - std(n) x phi(z)/0.05", "ES(R, p=0.95, method='gaussian')"),
    Convention("es_95", "gaussian_ddof1", _es_gaussian(A, 1), "mean - std(n-1) x phi(z)/0.05"),
    Convention("es_95", "rockafellar_uryasev", _es_rockafellar_uryasev(A), "Rockafellar-Uryasev (2002) at the order-statistic VaR"),
    Convention("es_95", "naive_tail_mean_floor", _es_naive_floor(A), "mean of the floor(0.05 n)+1 smallest returns"),
    Convention("es_95", "hybrid_parametric_threshold", _es_hybrid_parametric_threshold(A), "empirical mean below the Gaussian VaR (inconsistent hybrid)"),

    Convention("omega_0", "simple", _omega(0.0), "sum(max(r,0)) / sum(max(-r,0))", "Omega(R, L=0, method='simple')", "ratio"),
    Convention("ulcer_index", "martin", _ulcer, "sqrt(sum(drawdown^2) / n), Martin (1987)", "UlcerIndex(R)", "magnitude"),
    Convention("pain_index", "geometric_initial_peak", _pain, "mean(drawdown)", "PainIndex(R)", "magnitude"),

    Convention("skewness", "moment", _skew_moment, "population moment coefficient g1", "skewness(R, method='moment')", "ratio"),
    Convention("skewness", "adjusted_g1", _skew_sample, "adjusted Fisher-Pearson G1 (Excel SKEW, pandas .skew(); PerformanceAnalytics 'fisher')", "skewness(R, method='fisher')", "ratio"),
    Convention("skewness", "pa_sample", _skew_pa_sample, "R PerformanceAnalytics 'sample': n/((n-1)(n-2)) x sum(((x-mean)/sd_pop)^3)", "skewness(R, method='sample')", "ratio"),
    Convention("kurtosis", "excess_moment", _kurt_excess_moment, "population moment coefficient minus 3 (g2)", "kurtosis(R, method='excess')", "ratio"),
    Convention("kurtosis", "excess_sample", _kurt_sample_excess, "adjusted excess G2 (Excel KURT, pandas .kurt())", "kurtosis(R, method='sample_excess')", "ratio"),
    Convention("kurtosis", "pearson_moment", lambda r, ann=1: _kurt_excess_moment(r) + 3.0, "population moment coefficient, not excess (normal = 3)", None, "ratio"),
]


def conventions_for(metric: str) -> list[Convention]:
    return [c for c in CONVENTIONS if c.metric == metric]


def metrics() -> list[str]:
    seen: list[str] = []
    for c in CONVENTIONS:
        if c.metric not in seen:
            seen.append(c.metric)
    return seen


def compute(metric: str, convention: str, r, periods_per_year: int) -> float:
    for c in CONVENTIONS:
        if c.metric == metric and c.name == convention:
            return c.fn(r, periods_per_year)
    raise KeyError(f"{metric}/{convention}")
