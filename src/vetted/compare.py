"""Compare two risk models on the same days.

A backtest says whether one model is acceptable; a comparison says which of
two is better. Score each model with a strictly consistent loss
(``vetted.var.quantile_loss`` for VaR, ``vetted.es.fz0_loss`` for VaR and ES
together) and test whether the mean loss difference is zero.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._inputs import as_1d, check_same_index
from ._result import TestResult


def diebold_mariano(
    loss_a,
    loss_b,
    horizon: int = 1,
    alternative: str = "two-sided",
    varestimator: str = "acf",
    significance: float = 0.05,
) -> TestResult:
    """Diebold-Mariano test of equal predictive accuracy, with the Harvey,
    Leybourne and Newbold (1997) small-sample correction and Student-t
    p-values.

    ``loss_a`` and ``loss_b`` are the per-day losses of two models. The
    statistic is negative when model A has the lower loss.
    ``alternative="less"`` tests that A is better. The long-run variance uses
    autocovariances up to ``horizon - 1``, unweighted ("acf") or with
    Bartlett weights ("bartlett"). This matches R forecast::dm.test.

    References: Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive
    accuracy. Journal of Business and Economic Statistics, 13(3), 253-263.
    Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of
    prediction mean squared errors. International Journal of Forecasting,
    13(2), 281-291.
    """
    check_same_index(loss_a, loss_b, names=("loss_a", "loss_b"))
    a = as_1d(loss_a, "loss_a")
    b = as_1d(loss_b, "loss_b")
    if a.size != b.size:
        raise ValueError("the two loss series must cover the same days")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    d = a - b
    n = d.size
    h = int(horizon)
    if h < 1 or h > n:
        raise ValueError("horizon must be between 1 and the number of observations")
    dc = d - d.mean()
    acov = np.array([np.sum(dc[k:] * dc[: n - k]) / n for k in range(h)])
    if varestimator == "acf" or h == 1:
        var = (acov[0] + 2 * acov[1:].sum()) / n
    elif varestimator == "bartlett":
        w = 1 - np.arange(1, h) / h
        var = (acov[0] + 2 * np.sum(w * acov[1:])) / n
    else:
        raise ValueError("varestimator must be 'acf' or 'bartlett'")
    if var <= 0:
        raise ValueError("long-run variance is not positive; try varestimator='bartlett' or horizon=1")
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm = float(d.mean() / np.sqrt(var) * k)
    if alternative == "two-sided":
        p = 2 * stats.t.cdf(-abs(dm), n - 1)
    elif alternative == "less":
        p = stats.t.cdf(dm, n - 1)
    elif alternative == "greater":
        p = stats.t.sf(dm, n - 1)
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    return TestResult(
        name="Diebold-Mariano",
        statistic=dm,
        p_value=float(p),
        reject=bool(p < significance),
        significance=significance,
        null="both models have the same expected loss",
        reference="Diebold and Mariano (1995); Harvey, Leybourne and Newbold (1997)",
        detail={
            "mean_loss_a": float(np.mean(a)),
            "mean_loss_b": float(np.mean(b)),
            "better": "a" if d.mean() < 0 else "b",
            "observations": n,
            "horizon": h,
            "alternative": alternative,
        },
    )
