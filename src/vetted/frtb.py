"""Regulatory backtesting under the Basel market risk framework (FRTB).

Implements the tests in chapter MAR32 of the Basel Framework ("Internal
models approach: backtesting and P&L attribution test requirements"), in
force from 1 January 2023, and the binomial logic of MAR99 that sets the
backtesting zones:

* ``traffic_light``: bank-wide VaR backtesting zone and multiplier (MAR32.5,
  MAR32.9 Table 1), counting a missing P&L or VaR as an exception, and taking
  the greater of the actual-P&L and hypothetical-P&L counts.
* ``binomial_zones``: the same green / amber / red cut-offs for any sample
  size and VaR level, from the cumulative binomial probabilities (MAR99).
* ``desk_backtest``: trading desk eligibility, at most 12 exceptions at the
  99th and 30 at the 97.5th percentile over 250 days (MAR32.18-32.19).
* ``pla_test``: the P&L attribution test, Spearman correlation and the
  Kolmogorov-Smirnov distance between risk-theoretical and hypothetical P&L,
  and the resulting PLA zone (MAR32.34-32.42).

Source: https://www.bis.org/basel_framework/chapter/MAR/32.htm
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from ._inputs import align, as_1d, broadcast, check_level, check_same_index
from ._result import TestResult

MAR32 = "Basel Framework MAR32 (effective 1 January 2023)"

#: MAR32.9 Table 1: backtesting-dependent multiplier by number of exceptions
#: in 250 days at the 99th percentile.
MULTIPLIER = {0: 1.50, 1: 1.50, 2: 1.50, 3: 1.50, 4: 1.50, 5: 1.70, 6: 1.76, 7: 1.83, 8: 1.88, 9: 1.92}
MULTIPLIER_RED = 2.00

#: MAR32.42 Table 2: PLA test thresholds.
PLA_SPEARMAN_GREEN = 0.80
PLA_SPEARMAN_RED = 0.70
PLA_KS_GREEN = 0.09
PLA_KS_RED = 0.12


def binomial_zones(n: int = 250, alpha: float = 0.01, green: float = 0.95, red: float = 0.9999) -> dict:
    """Green, amber and red exception counts for ``n`` days at VaR tail
    probability ``alpha``.

    A count k is green while the probability of seeing k or fewer exceptions
    from a correct model is below ``green`` (95%), red once it reaches ``red``
    (99.99%), and amber in between, the rule MAR99 uses to set Table 1. For
    n = 250 and alpha = 1% this reproduces Table 1 exactly: green 0-4, amber
    5-9, red 10 or more.
    """
    alpha = check_level(alpha)
    if n < 1:
        raise ValueError("n must be at least 1")
    cdf = stats.binom.cdf(np.arange(n + 1), n, alpha)
    # Zero exceptions is always green, and red needs at least one exception,
    # even for windows so short that the binomial cut-offs fall below 1.
    green_max = max(int(np.sum(cdf < green) - 1), 0)
    red_min = int(np.argmax(cdf >= red)) if np.any(cdf >= red) else n + 1
    red_min = max(red_min, green_max + 1)
    return {
        "green": (0, green_max),
        "amber": (green_max + 1, red_min - 1),
        "red_from": red_min,
        "cumulative_probability": {int(k): float(cdf[k]) for k in range(min(n, red_min + 2) + 1)},
    }


def _zone(count: int, zones: dict) -> str:
    if count <= zones["green"][1]:
        return "green"
    if count < zones["red_from"]:
        return "amber"
    return "red"


def traffic_light(
    var,
    hypothetical_pnl,
    actual_pnl=None,
    alpha: float = 0.01,
    missing: str = "exception",
    window: Optional[int] = 250,
) -> TestResult:
    """Bank-wide VaR backtesting zone (MAR32.5 and MAR32.9).

    Exceptions against hypothetical and actual P&L are counted separately and
    the greater count is used (MAR32.5); a day on which the P&L or the VaR is
    missing counts as an exception (``missing="exception"``, the default).
    The zone comes from ``binomial_zones`` for the sample size and level
    given, and the capital multiplier from Table 1 when the window is the
    standard 250 days at the 99th percentile.

    ``window``: the regulatory count uses the most recent 12 months, so by
    default only the last 250 days are counted. Pass ``window=None`` to use
    the whole series (the zones then come from the binomial rule for that
    length).

    ``statistic`` is the exception count; ``reject`` is True outside the
    green zone.
    """
    alpha = check_level(alpha)
    if missing == "drop":
        raise ValueError("regulatory counts cannot drop days: MAR32.5 counts a missing P&L or VaR as an exception")
    if window is not None:
        if int(window) != window or int(window) < 1:
            raise ValueError("window must be a positive whole number of days, or None")
        window = int(window)
    hpl = as_1d(hypothetical_pnl, "hypothetical_pnl")
    v_all = broadcast(var, hpl.size, "var")
    series = {"hypothetical": hpl}
    if actual_pnl is not None:
        apl = as_1d(actual_pnl, "actual_pnl")
        if apl.size != hpl.size:
            raise ValueError("actual and hypothetical P&L must cover the same days")
        series["actual"] = apl
    check_same_index(hypothetical_pnl, var, actual_pnl, names=("hypothetical_pnl", "var", "actual_pnl"))
    counts = {}
    for name, p in series.items():
        v = v_all
        if window is not None and p.size > window:
            p, v = p[-window:], v[-window:]
        p, v = align(p, v, names=("var",), missing=missing)
        counts[name] = int(np.sum(p < -v))
        n = p.size
    count = max(counts.values())
    zones = binomial_zones(n, alpha)
    zone = _zone(count, zones)
    standard = n == 250 and abs(alpha - 0.01) < 1e-12
    multiplier: Optional[float] = None
    if standard:
        multiplier = MULTIPLIER.get(count, MULTIPLIER_RED)
    return TestResult(
        name="Basel traffic light",
        statistic=float(count),
        p_value=float(stats.binom.sf(count - 1, n, alpha)),
        reject=zone != "green",
        significance=0.05,
        null=f"the VaR is a correct {100 * (1 - alpha):g}% VaR",
        reference=f"{MAR32}, MAR32.5 and MAR32.9 Table 1; zones per MAR99",
        detail={
            "zone": zone,
            "exceptions": counts,
            "observations": n,
            "zones": {k: zones[k] for k in ("green", "amber", "red_from")},
            "multiplier": multiplier,
        },
    )


def desk_backtest(
    var_99,
    var_975,
    hypothetical_pnl,
    actual_pnl=None,
    missing: str = "exception",
    window: Optional[int] = 250,
) -> TestResult:
    """Trading desk backtesting requirement (MAR32.18-32.19).

    A desk stays on the internal models approach only if, over the most
    recent 250 days (``window``), it has at most 12 exceptions of its 99th percentile VaR
    and at most 30 of its 97.5th percentile VaR, counting against actual and
    hypothetical P&L separately and taking the greater. ``reject`` is True
    when the desk fails and must move to the standardised approach
    (``detail["zone"]`` is "ineligible"). ``statistic`` is the 99% count; both
    counts are in ``detail``.
    """
    out = {}
    for label, var, limit, a in (("99", var_99, 12, 0.01), ("97.5", var_975, 30, 0.025)):
        tl = traffic_light(var, hypothetical_pnl, actual_pnl, alpha=a, missing=missing, window=window)
        out[label] = {"exceptions": int(tl.statistic), "limit": limit, "breach": tl.statistic > limit}
    fail = out["99"]["breach"] or out["97.5"]["breach"]
    n = int(as_1d(hypothetical_pnl, "hypothetical_pnl").size)
    if n < 250:
        out["note"] = f"MAR32.18 asks for at least 250 observations; {n} given"
    out["observations_counted"] = n if window is None else min(n, int(window))
    out["zone"] = "ineligible" if fail else "eligible"
    return TestResult(
        name="FRTB desk backtest",
        statistic=float(out["99"]["exceptions"]),
        p_value=float("nan"),
        reject=bool(fail),
        significance=float("nan"),
        null="the desk meets the MAR32.19 exception limits",
        reference=f"{MAR32}, MAR32.18-32.19",
        detail=out,
    )


def pla_zone(spearman: float, ks: float) -> str:
    """PLA zone from the two test metrics (MAR32.42 Table 2): green if the
    correlation is above 0.80 and the KS distance below 0.09; red if the
    correlation is below 0.70 or the KS distance above 0.12; amber otherwise.
    The boundaries themselves (0.80, 0.09) fall in amber."""
    if spearman > PLA_SPEARMAN_GREEN and ks < PLA_KS_GREEN:
        return "green"
    if spearman < PLA_SPEARMAN_RED or ks > PLA_KS_RED:
        return "red"
    return "amber"


def pla_test(risk_theoretical_pnl, hypothetical_pnl) -> TestResult:
    """P&L attribution test (MAR32.34-32.42).

    Spearman rank correlation between risk-theoretical P&L (RTPL, the P&L the
    risk model's pricing produces) and hypothetical P&L (HPL, front office
    pricing), and the Kolmogorov-Smirnov distance between their empirical
    distributions. Green needs correlation above 0.80 and KS below 0.09; red
    is correlation below 0.70 or KS above 0.12; anything else is amber
    (Table 2). A red desk must use the standardised approach (MAR32.43); an
    amber desk pays a capital surcharge (MAR32.44).

    ``statistic`` is the Spearman correlation, the KS distance is in
    ``detail``, and ``reject`` is True outside the green zone. MAR32.35 asks
    for the most recent 250 days.
    """
    check_same_index(risk_theoretical_pnl, hypothetical_pnl, names=("risk_theoretical_pnl", "hypothetical_pnl"))
    rtpl = as_1d(risk_theoretical_pnl, "risk_theoretical_pnl")
    hpl = as_1d(hypothetical_pnl, "hypothetical_pnl")
    if rtpl.size != hpl.size:
        raise ValueError("RTPL and HPL must cover the same days")
    if not (np.all(np.isfinite(rtpl)) and np.all(np.isfinite(hpl))):
        raise ValueError("RTPL and HPL must be complete; MAR32 has no rule for filling gaps")
    if np.ptp(rtpl) == 0 or np.ptp(hpl) == 0:
        raise ValueError("RTPL or HPL is constant, so the Spearman correlation is undefined")
    rho = float(stats.spearmanr(rtpl, hpl)[0])
    ks = float(stats.ks_2samp(rtpl, hpl).statistic)
    zone = pla_zone(rho, ks)
    detail = {"spearman": rho, "ks": ks, "zone": zone, "observations": int(rtpl.size)}
    if rtpl.size != 250:
        detail["note"] = f"MAR32.35 uses the most recent 250 days; {rtpl.size} given"
    return TestResult(
        name="FRTB PLA test",
        statistic=rho,
        p_value=float("nan"),
        reject=zone != "green",
        significance=float("nan"),
        null="risk-theoretical P&L tracks hypothetical P&L closely enough (green zone)",
        reference=f"{MAR32}, MAR32.34-32.42 Table 2",
        detail=detail,
    )
