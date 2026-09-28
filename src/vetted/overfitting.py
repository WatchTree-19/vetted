"""Is a backtest evidence of skill, or the best of many lucky tries?

* ``probabilistic_sharpe``: probability the true Sharpe ratio exceeds a
  benchmark, allowing for sample length, skewness and fat tails.
* ``expected_max_sharpe``: the Sharpe ratio the best of N skill-less
  strategies shows on average (the "false strategy theorem").
* ``deflated_sharpe``: the probabilistic Sharpe ratio against that hurdle.
* ``min_track_record``: how long a track record has to be before its Sharpe
  ratio is significant.
* ``pbo``: probability of backtest overfitting by combinatorially symmetric
  cross-validation.
* ``haircut_sharpe``: the Sharpe ratio after a multiple-testing correction
  (Bonferroni, Holm or Benjamini-Hochberg-Yekutieli).
* ``effective_trials``: how many independent trials a set of correlated
  strategies amounts to.

Sharpe ratios are annualised with ``periods`` (252 for daily data) at the
interface and computed per period inside, as the formulas require. Skewness
and kurtosis are the population moments used by Bailey and Lopez de Prado
and by R PerformanceAnalytics ``ProbSharpeRatio``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Optional, Union

import numpy as np
from scipy import stats

from ._inputs import as_1d, check_level
from ._result import TestResult

EULER_GAMMA = float(np.euler_gamma)


def _clean(returns) -> np.ndarray:
    r = as_1d(returns, "returns")
    r = r[np.isfinite(r)]
    if r.size < 3:
        raise ValueError("need at least three non-missing returns")
    return r


def _is_flat(sd, mean) -> bool:
    return not sd > 1e-12 * max(abs(mean), 1e-300) or sd < 1e-300


def _moments(r: np.ndarray) -> tuple[float, float, float, int]:
    sd = r.std(ddof=1)
    if _is_flat(sd, r.mean()):
        raise ValueError("returns have (numerically) zero variance")
    sr = r.mean() / sd
    z = (r - r.mean()) / r.std(ddof=0)
    return float(sr), float(np.mean(z**3)), float(np.mean(z**4)), int(r.size)


def sharpe_std_error(sr: float, skew: float, kurt: float, n: int) -> float:
    """Standard error of a per-period Sharpe ratio for non-normal returns
    (Mertens 2002; Bailey and Lopez de Prado 2012, eq. 3). ``kurt`` is raw
    kurtosis (3 for a normal distribution)."""
    return float(np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1)))


def _psr_from_moments(sr, benchmark, skew, kurt, n) -> float:
    return float(stats.norm.cdf((sr - benchmark) / sharpe_std_error(sr, skew, kurt, n)))


def probabilistic_sharpe(returns, benchmark: float = 0.0, periods: int = 252) -> float:
    """Probability that the true Sharpe ratio exceeds ``benchmark`` (annualised).

    Reference: Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio
    efficient frontier. Journal of Risk, 15(2), 3-44.
    """
    sr, sk, ku, n = _moments(_clean(returns))
    return _psr_from_moments(sr, benchmark / np.sqrt(periods), sk, ku, n)


def expected_max_sharpe(trials: float, sharpe_variance: float) -> float:
    """Expected maximum Sharpe ratio of ``trials`` independent strategies
    whose true Sharpe ratio is zero, when their estimated Sharpe ratios have
    variance ``sharpe_variance``. Units follow the input: pass the variance of
    annualised Sharpe ratios to get an annualised hurdle.

    E[max] ~ sqrt(V) ((1 - gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - 1/(N e))),
    with gamma the Euler-Mascheroni constant.

    Reference: Bailey, D. H. and Lopez de Prado, M. (2014). The deflated
    Sharpe ratio: correcting for selection bias, backtest overfitting and
    non-normality. Journal of Portfolio Management, 40(5), 94-107.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance cannot be negative")
    if trials == 1:
        return 0.0
    z = (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / trials) + EULER_GAMMA * stats.norm.ppf(
        1 - 1 / (trials * np.e)
    )
    # The approximation is for large N and turns negative for N below about
    # 1.28 (only reachable with a non-integer N); the best of one or more
    # skill-less strategies never has a negative expected Sharpe, so clamp.
    return float(np.sqrt(sharpe_variance) * max(z, 0.0))


def _trial_sharpes(trials) -> np.ndarray:
    M = np.asarray(trials, dtype=float)
    if M.ndim != 2:
        raise ValueError("trials must be a (T, N) array or DataFrame of returns, one column per strategy")
    out = []
    for j in range(M.shape[1]):
        c = M[:, j]
        c = c[np.isfinite(c)]
        if c.size >= 3 and not _is_flat(c.std(ddof=1), c.mean()):
            out.append(c.mean() / c.std(ddof=1))
    return np.asarray(out)


def deflated_sharpe(
    returns,
    trials: Union[int, np.ndarray],
    sharpe_variance: Optional[float] = None,
    periods: int = 252,
    significance: float = 0.05,
    independent_trials: Optional[float] = None,
) -> TestResult:
    """Deflated Sharpe ratio: the probability the selected strategy's true
    Sharpe ratio is above zero, allowing for how many strategies were tried
    and for non-normal returns.

    ``trials`` is either the number of strategies tried, with
    ``sharpe_variance`` the variance of their **annualised** Sharpe ratios,
    or a (T, N) array or DataFrame holding every tried strategy's returns,
    from which both are measured.

    Correlated trials: measuring the Sharpe variance on the trials already
    reflects their correlation (correlated strategies have similar Sharpe
    ratios, so the variance is smaller and so is the hurdle). Do not also
    shrink the count to an effective number of trials: that discounts the
    correlation twice, and in simulations with pairwise correlation 0.7 it
    raised the false positive rate from about 3.5% to 23% at a nominal 5%.
    ``independent_trials`` is for the Bailey and Lopez de Prado recipe where
    both the count and ``sharpe_variance`` come from clusters of strategies,
    not from the raw trials.

    ``statistic`` is the DSR (a probability); ``reject`` means the Sharpe
    ratio is significant after deflation (DSR > 1 - significance). The
    hurdle and the undeflated PSR are in ``detail``.

    Reference: Bailey and Lopez de Prado (2014), Journal of Portfolio
    Management 40(5).
    """
    r = _clean(returns)
    sr, sk, ku, n = _moments(r)
    if np.ndim(trials) == 0:
        n_trials = float(trials)
        if n_trials < 1:
            raise ValueError("trials must be at least 1")
        if n_trials > 1 and sharpe_variance is None:
            raise ValueError("sharpe_variance is required when trials is a count")
        var_per_period = (sharpe_variance or 0.0) / periods
    else:
        if sharpe_variance is not None:
            raise ValueError("pass sharpe_variance only with a count; with a matrix it is measured")
        srs = _trial_sharpes(trials)
        n_trials = srs.size
        if n_trials == 0:
            raise ValueError("no trial has a defined Sharpe ratio")
        var_per_period = float(srs.var(ddof=1)) if n_trials > 1 else 0.0
    if independent_trials is not None:
        if independent_trials < 1:
            raise ValueError("independent_trials must be at least 1")
        n_trials = independent_trials
    hurdle = expected_max_sharpe(n_trials, var_per_period)
    dsr = _psr_from_moments(sr, hurdle, sk, ku, n)
    return TestResult(
        name="Deflated Sharpe ratio",
        statistic=dsr,
        p_value=1.0 - dsr,
        reject=dsr > 1.0 - significance,
        significance=significance,
        null=f"true Sharpe ratio <= 0, given the best of {n_trials:.4g} trials was selected",
        reference="Bailey and Lopez de Prado (2014), Journal of Portfolio Management 40(5)",
        detail={
            "sharpe": sr * np.sqrt(periods),
            "hurdle_sharpe": hurdle * np.sqrt(periods),
            "trials": n_trials if n_trials != int(n_trials) else int(n_trials),
            "sharpe_variance": var_per_period * periods,
            "psr": _psr_from_moments(sr, 0.0, sk, ku, n),
            "skewness": sk,
            "kurtosis": ku,
            "observations": n,
        },
    )


def min_track_record(
    returns, benchmark: float = 0.0, confidence: float = 0.95, periods: int = 252
) -> float:
    """Minimum number of observations for the Sharpe ratio to be significantly
    above ``benchmark`` (annualised) at ``confidence``. ``inf`` if the
    observed Sharpe ratio does not exceed the benchmark.

    Reference: Bailey and Lopez de Prado (2012), Journal of Risk 15(2).
    """
    confidence = check_level(confidence, "confidence")
    sr, sk, ku, _ = _moments(_clean(returns))
    b = benchmark / np.sqrt(periods)
    if sr <= b:
        return float("inf")
    return float(1 + (1 - sk * sr + (ku - 1) / 4 * sr**2) * (stats.norm.ppf(confidence) / (sr - b)) ** 2)


def _sharpe_cols(M: np.ndarray) -> np.ndarray:
    sd = M.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, M.mean(axis=0) / sd, np.nan)


def pbo(
    trials,
    n_blocks: int = 16,
    metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rank_denominator: str = "n+1",
) -> TestResult:
    """Probability of backtest overfitting (CSCV).

    Splits the (T, N) matrix of every configuration's returns into
    ``n_blocks`` equal blocks and, for every way of choosing half of them as
    in-sample, picks the configuration with the best in-sample ``metric``
    (Sharpe by default) and records its relative rank out of sample. PBO is
    the share of splits in which the in-sample winner ranks at or below the
    out-of-sample median. Near 0: selection found something real. Near 0.5
    or above: the selection process is overfitting.

    ``metric(M)`` receives a (t, N) block and returns N scores. Rows beyond a
    multiple of ``n_blocks`` are dropped from the end.

    ``rank_denominator``: the relative rank is rank / (N + 1) in Bailey et al.
    (2017); the R package pbo divides by N. Pass "n" to reproduce it.

    Reference: Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J.
    (2017). The probability of backtest overfitting. Journal of Computational
    Finance, 20(4), 39-69.
    """
    M = np.asarray(trials, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("trials must be (T, N) with at least two configurations")
    if not np.all(np.isfinite(M)):
        raise ValueError("trials must not contain missing values")
    flat = M.std(axis=0) <= 1e-12 * np.maximum(np.abs(M.mean(axis=0)), 1e-300)
    if flat.any():
        raise ValueError(
            f"configuration(s) {np.flatnonzero(flat).tolist()} have constant returns, so no Sharpe ratio; drop them"
        )
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be even")
    T, N = M.shape
    size = T // n_blocks
    if size < 2:
        raise ValueError("not enough observations for the requested number of blocks")
    M = M[: size * n_blocks]
    denom = {"n+1": N + 1.0, "n": float(N)}[rank_denominator]
    combos = np.array(list(combinations(range(n_blocks), n_blocks // 2)))
    if metric is None:
        # Sharpe ratio from per-block sums, so every split costs one matrix
        # product instead of a pass over the data. Same numbers as the loop
        # below (tests compare the two).
        B = M.reshape(n_blocks, size, N)
        s1, s2 = B.sum(axis=1), (B**2).sum(axis=1)
        ind = np.zeros((len(combos), n_blocks))
        np.put_along_axis(ind, combos, 1.0, axis=1)

        def sharpe_from(w):
            cnt = w.sum(axis=1, keepdims=True) * size
            mu = (w @ s1) / cnt
            var = ((w @ s2) - cnt * mu**2) / (cnt - 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(var > 0, mu / np.sqrt(np.maximum(var, 0)), np.nan)

        R_is, R_oos = sharpe_from(ind), sharpe_from(1.0 - ind)
    else:
        blocks = np.arange(size * n_blocks) // size
        R_is = np.empty((len(combos), N))
        R_oos = np.empty((len(combos), N))
        for k, ins in enumerate(combos):
            mask = np.isin(blocks, ins)
            R_is[k] = np.asarray(metric(M[mask]), dtype=float)
            R_oos[k] = np.asarray(metric(M[~mask]), dtype=float)
    star = np.nanargmax(R_is, axis=1)
    rows = np.arange(len(combos))
    sel_oos = R_oos[rows, star]
    # average rank, as scipy.stats.rankdata and R's rank() give on ties
    less = np.sum(R_oos < sel_oos[:, None], axis=1)
    equal = np.sum(R_oos == sel_oos[:, None], axis=1)
    rank_arr = less + (equal + 1) / 2.0
    ranks = rank_arr.tolist()
    w = rank_arr / denom
    with np.errstate(divide="ignore"):
        logits = np.where(w < 1, np.log(w / (1 - w)), np.inf)
    perf_is = R_is[rows, star]
    perf_oos = sel_oos
    p = float(np.mean(logits <= 0))
    slope = float(np.polyfit(perf_is, perf_oos, 1)[0]) if np.ptp(perf_is) > 0 else float("nan")
    return TestResult(
        name="Probability of backtest overfitting",
        statistic=p,
        p_value=float("nan"),
        reject=p > 0.5,
        significance=float("nan"),
        null="selecting the in-sample best configuration carries information out of sample",
        reference="Bailey, Borwein, Lopez de Prado and Zhu (2017), Journal of Computational Finance 20(4)",
        detail={
            "splits": int(logits.size),
            "logits": logits.tolist(),
            "oos_rank_of_selected": ranks,
            "performance_degradation_slope": slope,
            "prob_oos_loss": float(np.mean(perf_oos < 0)),
            "median_oos_metric_of_selected": float(np.median(perf_oos)),
        },
    )


def haircut_sharpe(
    sharpe: float,
    observations: int,
    trials: Union[int, np.ndarray],
    method: str = "holm",
    periods: int = 252,
) -> dict:
    """Sharpe ratio after correcting its p-value for multiple testing.

    The annualised ``sharpe`` over ``observations`` periods is converted to a
    t-statistic, t = SR sqrt(observations / periods); its two-sided p-value is
    adjusted for the number of strategies tried and converted back. With
    ``trials`` as a count, only "bonferroni" is possible; pass the annualised
    Sharpe ratios of every trial (same length of history, including this
    strategy, which is added if it is missing) to use "holm" or "bhy", which
    are less conservative.

    Returns the haircut Sharpe ratio, the haircut (share of the Sharpe ratio
    lost), and the single and adjusted p-values.

    Reference: Harvey, C. R. and Liu, Y. (2015). Backtesting. Journal of
    Portfolio Management, 42(1), 13-28.
    """
    years = observations / periods
    t = sharpe * np.sqrt(years)
    p_single = float(2 * stats.norm.sf(abs(t)))
    if np.ndim(trials) == 0:
        m = float(trials)
        if m < 1:
            raise ValueError("trials must be at least 1")
        if method != "bonferroni":
            raise ValueError("holm and bhy need every trial's Sharpe ratio; pass an array or use method='bonferroni'")
        p_adj = min(p_single * m, 1.0)
    else:
        others = np.asarray(trials, dtype=float).ravel()
        others = others[np.isfinite(others)]
        if others.size == 0:
            raise ValueError("trials is empty")
        # The tested strategy is one of the trials; add it if it is missing.
        if not np.any(np.isclose(others, sharpe, rtol=1e-12, atol=1e-15)):
            others = np.append(others, sharpe)
        m = others.size
        pv = np.sort(2 * stats.norm.sf(np.abs(others * np.sqrt(years))))
        k = int(np.searchsorted(pv, p_single, side="left"))  # 0-based rank of our p
        if method == "bonferroni":
            p_adj = min(p_single * m, 1.0)
        elif method == "holm":
            adj = np.maximum.accumulate(np.minimum((m - np.arange(m)) * pv, 1.0))
            p_adj = float(adj[min(k, m - 1)])
        elif method == "bhy":
            c = np.sum(1.0 / np.arange(1, m + 1))
            raw = pv * m * c / np.arange(1, m + 1)
            adj = np.minimum.accumulate(raw[::-1])[::-1]
            p_adj = float(min(adj[min(k, m - 1)], 1.0))
        else:
            raise ValueError("method must be 'bonferroni', 'holm' or 'bhy'")
    t_adj = stats.norm.isf(p_adj / 2)
    sr_adj = float(np.sign(sharpe) * max(t_adj, 0.0) / np.sqrt(years))
    return {
        "sharpe": float(sharpe),
        "haircut_sharpe": sr_adj,
        "haircut": float(1 - sr_adj / sharpe) if sharpe else float("nan"),
        "p_value": p_single,
        "adjusted_p_value": float(p_adj),
        "method": method,
        "trials": int(m) if float(m).is_integer() else m,
    }


def effective_trials(trials) -> float:
    """Effective number of independent strategies among correlated ones.

    Uses the eigenvalues of the correlation matrix of the trials' returns:
    N_eff = (sum lambda)^2 / sum lambda^2, which is N for uncorrelated
    strategies and 1 when they are all the same strategy.

    This is the participation ratio of the spectrum, a descriptive measure
    of how many distinct bets a search made. It is biased low when the
    history is short relative to the number of strategies (100 independent
    series of 250 days give about 71), and it is not an input to
    ``deflated_sharpe`` when the Sharpe variance is measured on the same
    trials (see that function).
    """
    M = np.asarray(trials, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("trials must be (T, N) with at least two strategies")
    M = M[np.all(np.isfinite(M), axis=1)]
    sd = M.std(axis=0)
    M = M[:, sd > 1e-12 * np.maximum(np.abs(M.mean(axis=0)), 1e-300)]
    if M.shape[1] < 2:
        raise ValueError("fewer than two strategies with non-zero variance")
    C = np.corrcoef(M, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(lam.sum() ** 2 / np.sum(lam**2))
