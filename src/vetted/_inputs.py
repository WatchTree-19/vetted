"""Input handling shared by the tests.

Sign conventions, used everywhere in vetted:

* ``pnl`` is profit and loss or returns: a gain is positive, a loss negative.
* ``var`` and ``es`` are reported the way risk reports and the Basel text
  state them: as **positive** loss amounts. A VaR of 2.1 means a loss of 2.1.
* An exception (breach) on day t is ``pnl[t] < -var[t]``, a loss larger than
  the VaR (MAR32.5).
"""

from __future__ import annotations

import numpy as np

MISSING_POLICIES = ("raise", "drop", "exception")


def as_1d(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return a.reshape(1)
    if a.ndim != 1:
        a = a.squeeze()
        if a.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    return a


def broadcast(x, n: int, name: str) -> np.ndarray:
    a = as_1d(x, name)
    if a.size == 1:
        return np.full(n, float(a[0]))
    if a.size != n:
        raise ValueError(f"{name} has length {a.size}, expected {n}")
    return a


def check_level(alpha: float, name: str = "alpha") -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {alpha}")
    return alpha


def check_same_index(*objs, names=None) -> None:
    """If two or more inputs carry a pandas index, they must be identical.

    vetted works by position. Two Series covering different dates would
    otherwise be compared day-for-day by position, silently pairing each
    forecast with the wrong P&L."""
    idx = [(i, getattr(o, "index", None)) for i, o in enumerate(objs)]
    idx = [(i, x) for i, x in idx if x is not None and hasattr(x, "equals")]
    for (i, a), (j, b) in zip(idx, idx[1:]):
        if not a.equals(b):
            ni = names[i] if names else f"input {i}"
            nj = names[j] if names else f"input {j}"
            raise ValueError(
                f"{ni} and {nj} have different indexes; align them first "
                "(for example pnl, var = pnl.align(var, join='inner'))"
            )


def align(pnl, *forecasts, names=("var", "es"), missing: str = "raise"):
    """Align P&L with one or more forecast series and apply a missing-data
    policy.

    missing:
        "raise": any NaN is an error (the default for statistical tests,
            which assume a complete record).
        "drop": days where anything is missing are removed.
        "exception": a day where the P&L or a forecast is missing counts as an
            exception, as MAR32.5 requires for regulatory backtesting. The
            returned P&L for that day is set to -inf so every downstream
            comparison treats it as a breach.
    """
    if missing not in MISSING_POLICIES:
        raise ValueError(f"missing must be one of {MISSING_POLICIES}")
    check_same_index(pnl, *forecasts, names=("pnl",) + tuple(names))
    p = as_1d(pnl, "pnl")
    fs = [broadcast(f, p.size, nm) for f, nm in zip(forecasts, names)]
    for f, nm in zip(fs, names):
        if np.any(np.isfinite(f) & (f < 0)):
            raise ValueError(
                f"{nm} must be reported as a positive loss amount "
                f"(a VaR of 2 means a loss of 2); got negative values"
            )
    bad = ~np.isfinite(p)
    for f in fs:
        bad |= ~np.isfinite(f)
    if bad.any():
        if missing == "raise":
            raise ValueError(
                f"{int(bad.sum())} day(s) have a missing or non-finite P&L or "
                "forecast; pass missing='drop' or missing='exception'"
            )
        if missing == "drop":
            keep = ~bad
            p = p[keep]
            fs = [f[keep] for f in fs]
        else:
            p = p.copy()
            p[bad] = -np.inf
            fs = [np.where(np.isfinite(f), f, 0.0) for f in fs]
    if p.size == 0:
        raise ValueError("no observations")
    return (p, *fs)
