"""Write tests/data/backtest_fixtures.json: deterministic P&L and forecast
series that oracle/gen_backtest_oracle.R and the Python tests both read, so
the R packages and vetted see identical numbers.

    python oracle/make_fixtures.py

Scenarios, each a GARCH(1,1) P&L path with Student-t(5) shocks, so the true
daily distribution is known:

* correct: forecasts from the true conditional distribution.
* underestimated: the same forecasts scaled by 0.8 (risk too low).
* static: a constant VaR / ES from the unconditional volatility, so
  exceptions cluster in high-volatility spells.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "backtest_fixtures.json"
NU = 5.0


def t_std_quantile(p):
    # quantile of a Student-t scaled to unit variance
    return stats.t.ppf(p, NU) * np.sqrt((NU - 2) / NU)


def t_std_es(alpha):
    # ES (positive loss) of a unit-variance Student-t at tail probability alpha
    q = stats.t.ppf(alpha, NU)
    es_raw = stats.t.pdf(q, NU) / alpha * (NU + q**2) / (NU - 1)
    return es_raw * np.sqrt((NU - 2) / NU)


def garch_path(n, seed, omega=0.02, a=0.08, b=0.9):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(NU, n) * np.sqrt((NU - 2) / NU)
    sig = np.empty(n)
    x = np.empty(n)
    s2 = omega / (1 - a - b)
    for t in range(n):
        sig[t] = np.sqrt(s2)
        x[t] = sig[t] * z[t]
        s2 = omega + a * x[t] ** 2 + b * s2
    return x, sig


def scenario(n, seed, kind):
    pnl, sig = garch_path(n, seed)
    if kind == "static":
        sig_f = np.full(n, np.sqrt(0.02 / (1 - 0.08 - 0.9)))
    elif kind == "underestimated":
        sig_f = 0.8 * sig
    else:
        sig_f = sig
    var99 = -t_std_quantile(0.01) * sig_f
    var975 = -t_std_quantile(0.025) * sig_f
    es975 = t_std_es(0.025) * sig_f
    pit = stats.t.cdf(pnl / sig_f / np.sqrt((NU - 2) / NU), NU)
    r = lambda a: np.round(a, 12).tolist()  # noqa: E731
    return {
        "pnl": r(pnl),
        "var99": r(var99),
        "var975": r(var975),
        "es975": r(es975),
        "scale": r(sig_f),
        "pit": r(pit),
    }


def main():
    fixtures = {}
    for n, seed in ((250, 11), (1000, 12)):
        for kind in ("correct", "underestimated", "static"):
            fixtures[f"{kind}_{n}"] = scenario(n, seed, kind)
    # two competing VaR models on the same P&L, for Diebold-Mariano
    rng = np.random.default_rng(21)
    pnl, sig = garch_path(1000, 22)
    fixtures["compare_1000"] = {
        "pnl": np.round(pnl, 12).tolist(),
        "var_garch": np.round(-t_std_quantile(0.01) * sig, 12).tolist(),
        "var_static": np.round(np.full(1000, -t_std_quantile(0.01) * np.sqrt(0.02 / 0.02)), 12).tolist(),
    }
    # a strategy search for PBO: 1000 days x 12 configurations, one with skill
    M = rng.normal(0.0, 0.01, (1000, 12))
    M[:, 3] += 0.0004
    fixtures["trials_1000x12"] = np.round(M, 12).tolist()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixtures))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
