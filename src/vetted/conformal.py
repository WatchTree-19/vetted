"""VaR with a coverage guarantee, from conformal prediction.

Historical-simulation VaR takes an empirical quantile and hopes. Split
conformal prediction takes the same quantile with a (n + 1) correction and
gets a finite-sample guarantee: if the calibration P&L and the next day are
exchangeable, the next loss exceeds the VaR with probability at most alpha,
with no distributional assumption. Markets are not exchangeable for long, so
``adaptive_var`` adds Adaptive Conformal Inference, which steers the level
online and guarantees the long-run exception rate converges to alpha even
when volatility regimes change.

Both accept an optional ``scale`` (a volatility forecast for each day, from
GARCH, EWMA or anything else), in which case conformal calibration is done
on standardised losses and the result is a conformalised version of that
model: its shape, with coverage that is guaranteed rather than assumed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._inputs import as_1d, check_level, check_same_index


def conformal_quantile(scores, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest of ``n`` scores, the
    quantile that makes split conformal valid in finite samples. ``inf`` when
    n is too small for the level (fewer than 1 / alpha - 1 scores)."""
    alpha = check_level(alpha)
    s = as_1d(scores, "scores")
    if np.any(np.isnan(s)):
        raise ValueError("scores contain NaN")
    s = np.sort(s)  # +inf scores are kept: they are real, unbounded scores
    n = s.size
    if n == 0:
        raise ValueError("no calibration scores")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n else float(s[k - 1])


def conformal_var(calibration_pnl, alpha: float = 0.01, scale=None, next_scale: float = 1.0) -> float:
    """One-day VaR (positive loss amount) with P(loss > VaR) <= alpha when
    the calibration days and the next day are exchangeable.

    With a volatility forecast ``scale`` for the calibration days and
    ``next_scale`` for the next day, calibration uses standardised losses
    -pnl / scale and the VaR is their conformal quantile times
    ``next_scale``.

    Reference: Vovk, V., Gammerman, A. and Shafer, G. (2005). Algorithmic
    Learning in a Random World. Springer. Lei, J. et al. (2018).
    Distribution-free predictive inference for regression. JASA 113(523).
    """
    check_same_index(calibration_pnl, scale, names=("calibration_pnl", "scale"))
    p = as_1d(calibration_pnl, "calibration_pnl")
    s = np.ones_like(p) if scale is None else as_1d(scale, "scale")
    if s.size != p.size:
        raise ValueError("scale must have one value per calibration day")
    if np.any(~np.isfinite(s) | (s <= 0)) or not (np.isfinite(next_scale) and next_scale > 0):
        raise ValueError("scale and next_scale must be positive")
    q = conformal_quantile(-p / s, alpha)
    return float(q * next_scale)


def adaptive_var(
    pnl,
    alpha: float = 0.01,
    window: int = 250,
    gamma: float = 0.005,
    scale=None,
    alpha_start: Optional[float] = None,
    unbounded: str = "max_loss",
) -> dict:
    """Out-of-sample VaR forecasts that adapt to regime changes, with a
    long-run coverage guarantee.

    For each day t >= ``window`` the VaR is the conformal quantile of the
    previous ``window`` (standardised) losses at level alpha_t, using only
    data before t. After seeing day t, the level moves by
    alpha_{t+1} = alpha_t + gamma (alpha - err_t), err_t = 1{exception}.
    With ``unbounded="inf"``, whatever the data do, the realised exception
    rate over T days is within (max(alpha_1, 1 - alpha_1) + gamma) / (gamma T)
    of alpha.

    When alpha_t falls below 1 / (window + 1), the conformal quantile is
    infinite: no loss in the window is large enough. ``unbounded="inf"``
    keeps the infinite forecast, which is what the guarantee assumes.
    ``unbounded="max_loss"`` (the default) uses the largest loss in the
    window instead, so every forecast is a usable number; the guarantee then
    holds only if no exception falls on such a capped day, which the result
    reports as ``guarantee_holds`` (with ``capped_days`` and
    ``capped_exceptions``). In a steady sell-off the capped VaR can be
    breached repeatedly and the guarantee is lost; check the flag.

    Returns the VaR path (NaN for the first ``window`` days), the exception
    indicators, the level path and the realised exception rate.

    Reference: Gibbs, I. and Candes, E. (2021). Adaptive conformal inference
    under distribution shift. NeurIPS 34.
    """
    alpha = check_level(alpha)
    check_same_index(pnl, scale, names=("pnl", "scale"))
    p = as_1d(pnl, "pnl")
    n = p.size
    s = np.ones(n) if scale is None else as_1d(scale, "scale")
    if s.size != n:
        raise ValueError("scale must have one value per day")
    if np.any(~np.isfinite(p)):
        raise ValueError("pnl must not contain missing values")
    if np.any(~np.isfinite(s) | (s <= 0)):
        raise ValueError("scale must be positive")
    if window < 2 or window >= n:
        raise ValueError("window must be at least 2 and shorter than the series")
    z = -p / s
    if unbounded not in ("max_loss", "inf"):
        raise ValueError("unbounded must be 'max_loss' or 'inf'")
    a_t = a0 = alpha if alpha_start is None else alpha_start
    capped = 0
    var = np.full(n, np.nan)
    hit = np.zeros(n, dtype=bool)
    levels = np.full(n, np.nan)
    capped_hits = 0
    for t in range(window, n):
        capped_now = False
        levels[t] = a_t
        hist = np.sort(z[t - window : t])
        if a_t >= 1:
            q = -np.inf
        else:
            k = int(np.ceil((window + 1) * (1 - a_t))) if a_t > 0 else window + 1
            q = np.inf if k > window else hist[max(k, 1) - 1]
            if not np.isfinite(q) and unbounded == "max_loss":
                q = hist[-1]
                capped += 1
                capped_now = True
        var[t] = q * s[t]
        hit[t] = p[t] < -var[t]
        capped_hits += int(capped_now and hit[t])
        a_t = a_t + gamma * (alpha - float(hit[t]))
    live = slice(window, n)
    return {
        "var": var,
        "exceptions": hit,
        "alpha_path": levels,
        "exception_rate": float(hit[live].mean()),
        "target": alpha,
        "capped_days": capped,
        "capped_exceptions": capped_hits,
        "guarantee_holds": capped_hits == 0,
        "bound": (max(a0, 1 - a0) + gamma) / (gamma * (n - window)) if gamma > 0 else float("inf"),
    }
