"""Backtests for Value-at-Risk forecasts.

Every function takes the realised P&L and the VaR that was forecast for each
day (a positive loss amount, see ``vetted._inputs``), plus the VaR tail
probability ``alpha`` (0.01 for a 99% VaR).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats
from scipy.special import xlogy

from ._inputs import align, as_1d, broadcast, check_level, check_same_index
from ._result import TestResult


def exceptions(pnl, var, missing: str = "raise") -> np.ndarray:
    """Boolean array, True on days where the loss exceeded the VaR
    (``pnl < -var``). See ``vetted._inputs.align`` for ``missing``."""
    p, v = align(pnl, var, names=("var",), missing=missing)
    return p < -v


def _kupiec_lr(x, n, alpha):
    """LR_uc for exception counts ``x`` (scalar or array) out of ``n``."""
    x = np.asarray(x, dtype=float)
    phat = x / n
    ll0 = xlogy(n - x, 1.0 - alpha) + xlogy(x, alpha)
    ll1 = xlogy(n - x, 1.0 - phat) + xlogy(x, phat)
    return np.maximum(-2.0 * (ll0 - ll1), 0.0)


def _ind_lr(hits):
    """Christoffersen LR_ind for a (..., n) boolean array of hit sequences."""
    prev, curr = hits[..., :-1], hits[..., 1:]
    n00 = np.sum(~prev & ~curr, axis=-1).astype(float)
    n01 = np.sum(~prev & curr, axis=-1).astype(float)
    n10 = np.sum(prev & ~curr, axis=-1).astype(float)
    n11 = np.sum(prev & curr, axis=-1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pi01 = np.where(n00 + n01 > 0, n01 / (n00 + n01), 0.0)
        pi11 = np.where(n10 + n11 > 0, n11 / (n10 + n11), 0.0)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    l_restricted = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    l_markov = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    return np.maximum(-2.0 * (l_restricted - l_markov), 0.0), (n00, n01, n10, n11, pi01, pi11)


def _mc_pvalue(observed, simulated):
    # (1 + #{sim >= obs}) / (1 + B): exact-level Monte Carlo test (Dufour
    # 2006), conservative when the statistic is discrete.
    simulated = np.asarray(simulated)
    return float((1 + np.sum(simulated >= observed - 1e-12)) / (1 + simulated.size))


def _null_hits(rng, n_sims, n, alpha):
    return rng.random((n_sims, n)) < alpha


def _check_pvalue(pvalue):
    if pvalue not in ("asymptotic", "monte_carlo"):
        raise ValueError("pvalue must be 'asymptotic' or 'monte_carlo'")


_MC_NOTE = (
    "pvalue='monte_carlo' simulates the statistic under the null (independent "
    "exceptions with probability alpha, the forecasts held fixed) instead of "
    "using the chi-squared limit, which is unreliable when few exceptions are "
    "expected (Dufour 2006; Dumitrescu, Hurlin and Pham 2012)."
)


def kupiec(
    pnl,
    var,
    alpha: float,
    significance: float = 0.05,
    missing="raise",
    pvalue: str = "asymptotic",
    n_sims: int = 10_000,
    seed: Optional[int] = 0,
) -> TestResult:
    """Kupiec (1995) proportion-of-failures test.

    H0: the probability of an exception is ``alpha``. Two-sided: too many
    exceptions and too few both reject. LR ~ chi2(1), or a Monte Carlo
    p-value with ``pvalue="monte_carlo"``.

    Reference: Kupiec, P. (1995). Techniques for verifying the accuracy of risk
    measurement models. Journal of Derivatives, 3(2), 73-84.
    """
    alpha = check_level(alpha)
    _check_pvalue(pvalue)
    hits = exceptions(pnl, var, missing=missing)
    n, x = hits.size, int(hits.sum())
    lr = float(_kupiec_lr(x, n, alpha))
    if pvalue == "asymptotic":
        p = float(stats.chi2.sf(lr, 1))
    else:
        sims = np.random.default_rng(seed).binomial(n, alpha, n_sims)
        p = _mc_pvalue(lr, _kupiec_lr(sims, n, alpha))
    return TestResult(
        name="Kupiec POF",
        statistic=lr,
        p_value=p,
        reject=p < significance,
        significance=significance,
        null=f"exception probability is {alpha:g}",
        reference="Kupiec (1995), Journal of Derivatives 3(2)",
        detail={"observations": n, "exceptions": x, "expected": n * alpha, "rate": x / n, "p_value_method": pvalue},
    )


def christoffersen(
    pnl,
    var,
    alpha: float,
    significance: float = 0.05,
    missing="raise",
    pvalue: str = "asymptotic",
    n_sims: int = 10_000,
    seed: Optional[int] = 0,
) -> TestResult:
    """Christoffersen (1998) independence and conditional coverage tests.

    ``statistic`` and ``p_value`` are for conditional coverage,
    LR_cc = LR_uc + LR_ind ~ chi2(2): exceptions happen with probability
    ``alpha`` *and* do not cluster. The independence part alone (first-order
    Markov alternative, LR_ind ~ chi2(1)) is in ``detail``. The
    decomposition matches R rugarch ``VaRTest``.

    Reference: Christoffersen, P. (1998). Evaluating interval forecasts.
    International Economic Review, 39(4), 841-862.
    """
    alpha = check_level(alpha)
    _check_pvalue(pvalue)
    hits = exceptions(pnl, var, missing=missing)
    if hits.size < 2:
        raise ValueError("need at least two observations")
    n = hits.size
    lr_ind, (n00, n01, n10, n11, pi01, pi11) = _ind_lr(hits)
    lr_ind = float(lr_ind)
    lr_uc = float(_kupiec_lr(int(hits.sum()), n, alpha))
    lr_cc = lr_uc + lr_ind
    if pvalue == "asymptotic":
        p_cc = float(stats.chi2.sf(lr_cc, 2))
        p_ind = float(stats.chi2.sf(lr_ind, 1))
        p_uc = float(stats.chi2.sf(lr_uc, 1))
    else:
        sim = _null_hits(np.random.default_rng(seed), n_sims, n, alpha)
        ind_sim, _ = _ind_lr(sim)
        uc_sim = _kupiec_lr(sim.sum(axis=1), n, alpha)
        p_cc = _mc_pvalue(lr_cc, uc_sim + ind_sim)
        p_ind = _mc_pvalue(lr_ind, ind_sim)
        p_uc = _mc_pvalue(lr_uc, uc_sim)
    return TestResult(
        name="Christoffersen conditional coverage",
        statistic=lr_cc,
        p_value=p_cc,
        reject=p_cc < significance,
        significance=significance,
        null=f"exceptions are independent with probability {alpha:g}",
        reference="Christoffersen (1998), International Economic Review 39(4)",
        detail={
            "lr_uc": lr_uc,
            "p_uc": p_uc,
            "lr_ind": lr_ind,
            "p_ind": p_ind,
            "reject_ind": p_ind < significance,
            "transitions": {"n00": int(n00), "n01": int(n01), "n10": int(n10), "n11": int(n11)},
            "pi01": float(pi01),
            "pi11": float(pi11),
            "exceptions": int(hits.sum()),
            "observations": int(n),
            "p_value_method": pvalue,
        },
    )


def _dq_design(hit_c, v, psq, lags, squared_pnl_lag):
    """Regressors for the DQ test. ``hit_c`` is (..., n) demeaned hits.

    The VaR and squared P&L columns are divided by their root mean square,
    which leaves the projection (and so the statistic) unchanged but keeps
    the pseudo-inverse's tolerance from depending on the units of the P&L.
    """
    n = hit_c.shape[-1]
    lead = hit_c.shape[:-1]
    cols = [np.ones(lead + (n - lags,))]
    cols += [hit_c[..., lags - k : n - k] for k in range(1, lags + 1)]

    def unit(x):
        rms = np.sqrt(np.mean(x**2))
        return x / rms if rms > 0 else x

    cols.append(np.broadcast_to(unit(v[lags:]), lead + (n - lags,)))
    if squared_pnl_lag:
        cols.append(np.broadcast_to(unit(psq[lags - 1 : n - 1]), lead + (n - lags,)))
    return np.stack(cols, axis=-1)


_RCOND = np.sqrt(np.finfo(float).eps)  # the tolerance MASS::ginv uses


def _dq_stat(hit_c, v, psq, lags, squared_pnl_lag, alpha):
    """DQ statistic y'X(X'X)^+X'y / (alpha (1 - alpha)), batched over leading
    axes. A constant VaR is collinear with the intercept, so X'X can be
    singular; the pseudo-inverse then projects onto the column space."""
    X = _dq_design(hit_c, v, psq, lags, squared_pnl_lag)
    y = hit_c[..., lags:]
    XtX = np.einsum("...ti,...tj->...ij", X, X)
    Xty = np.einsum("...ti,...t->...i", X, y)
    beta = np.einsum("...ij,...j->...i", np.linalg.pinv(XtX, rcond=_RCOND), Xty)
    return np.einsum("...i,...i->...", beta, Xty) / (alpha * (1 - alpha)), beta, X


def dynamic_quantile(
    pnl,
    var,
    alpha: float,
    lags: int = 4,
    significance: float = 0.05,
    missing="raise",
    squared_pnl_lag: bool = False,
    pvalue: str = "asymptotic",
    n_sims: int = 5_000,
    seed: Optional[int] = 0,
) -> TestResult:
    """Engle and Manganelli (2004) dynamic quantile (DQ) test.

    Regresses the demeaned hit sequence Hit_t = 1{exception} - alpha on a
    constant, ``lags`` of its own lags and the VaR forecast itself. Under a
    correct model none of these predict the next hit, so
    DQ = b' X'X b / (alpha (1 - alpha)) ~ chi2(lags + 2), with the degrees of
    freedom reduced by one when the VaR is constant (it is then collinear
    with the intercept).

    Catches what the Christoffersen test cannot: clustering beyond one day and
    exceptions that depend on the level of the forecast.

    ``squared_pnl_lag=True`` adds yesterday's squared P&L as a regressor
    (df = lags + 3), the variant implemented in the R package GAS
    (``BacktestVaR``).

    With few expected exceptions (a 99% VaR over one year expects 2.5) the
    chi-squared p-value is oversized: in vetted's simulations it rejected a
    correct model 6.4% of the time at 5% over 250 days (tools/simulations.py).
    Use ``pvalue="monte_carlo"`` there; ``validate_risk_model`` does.

    Reference: Engle, R. F. and Manganelli, S. (2004). CAViaR: conditional
    autoregressive value at risk by regression quantiles. Journal of Business
    and Economic Statistics, 22(4), 367-381.
    """
    alpha = check_level(alpha)
    _check_pvalue(pvalue)
    p, v = align(pnl, var, names=("var",), missing=missing)
    hit = (p < -v).astype(float) - alpha
    n = hit.size
    if n <= lags + 2:
        raise ValueError("not enough observations for the requested lags")
    # squared P&L regressor: a day counted as an exception because its P&L
    # was missing (missing="exception") contributes 0, not inf
    psq = np.where(np.isfinite(p), p, 0.0) ** 2
    dq, beta, X = _dq_stat(hit, v, psq, lags, squared_pnl_lag, alpha)
    dq = float(dq)
    # Degrees of freedom: the hit lags count in full; the intercept, VaR and
    # squared P&L count by their rank, so a constant VaR (collinear with the
    # intercept) adds nothing. Only fixed collinearity reduces df, never
    # where exceptions happened to fall.
    fixed = X[:, [0] + list(range(lags + 1, X.shape[-1]))]
    df = lags + int(np.linalg.matrix_rank(fixed, tol=_RCOND * np.linalg.norm(fixed, 2)))
    if pvalue == "asymptotic":
        pv = float(stats.chi2.sf(dq, df))
    else:
        rng = np.random.default_rng(seed)
        sims = []
        for start in range(0, n_sims, 1000):
            b = min(1000, n_sims - start)
            h = _null_hits(rng, b, n, alpha).astype(float) - alpha
            sims.append(_dq_stat(h, v, psq, lags, squared_pnl_lag, alpha)[0])
        pv = _mc_pvalue(dq, np.concatenate(sims))
    return TestResult(
        name="Engle-Manganelli DQ",
        statistic=dq,
        p_value=pv,
        reject=pv < significance,
        significance=significance,
        null="hits are unpredictable from past hits and the VaR level",
        reference="Engle and Manganelli (2004), JBES 22(4)",
        detail={
            "lags": lags,
            "df": df,
            "regressors": int(X.shape[-1]),
            "p_value_method": pvalue,
        },
    )


def exception_durations(pnl, var, alpha: float, missing="raise") -> dict:
    """Days between exceptions, and how they compare with the geometric
    distribution a correct model implies (mean 1 / alpha). Descriptive; for a
    formal test use ``christoffersen`` or ``dynamic_quantile``."""
    alpha = check_level(alpha)
    hits = exceptions(pnl, var, missing=missing)
    idx = np.flatnonzero(hits)
    d = np.diff(idx)
    return {
        "exception_days": idx.tolist(),
        "durations": d.tolist(),
        "mean_duration": float(d.mean()) if d.size else float("nan"),
        "expected_mean_duration": 1.0 / alpha,
    }


def quantile_loss(pnl, var, alpha: float) -> np.ndarray:
    """Per-day quantile (tick) loss of a VaR forecast, the strictly consistent
    scoring function for a quantile. Lower is better; compare models with
    ``vetted.compare.diebold_mariano``."""
    alpha = check_level(alpha)
    check_same_index(pnl, var, names=("pnl", "var"))
    p = as_1d(pnl, "pnl")
    q = -broadcast(var, p.size, "var")  # the return quantile
    hit = (p < q).astype(float)
    return (alpha - hit) * (p - q)
