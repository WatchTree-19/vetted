"""Validate four VaR / ES models for one trading desk.

A desk's daily P&L follows a GARCH(1,1) process with Student-t(5) shocks, so
volatility clusters and tails are fat, as in real P&L. Four models forecast
its 99% and 97.5% VaR and 97.5% ES each day, using only data up to the day
before:

1. Historical simulation over the last 250 days, the most common bank model.
2. RiskMetrics: an EWMA volatility (lambda = 0.94) with normal quantiles.
3. Conformal EWMA: the same volatility, with the quantile taken from past
   standardised losses by adaptive conformal inference (vetted.conformal),
   so no distribution is assumed.
4. The true conditional distribution, which no real model has.

Each model is backtested on 1250 days with vetted.validate_risk_model, first
in detail on one simulated history and then on 20 independent histories.

Run:  python examples/risk_model_validation.py
"""

import numpy as np
from scipy import stats

import vetted
from vetted import compare, conformal, es

NU = 5
S = np.sqrt((NU - 2) / NU)
N_LIVE = 1250
WARMUP = 250


def simulate(n, seed, omega=0.02, a=0.10, b=0.88):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(NU, n) * S
    sig, x = np.empty(n), np.empty(n)
    s2 = omega / (1 - a - b)
    for t in range(n):
        sig[t] = np.sqrt(s2)
        x[t] = sig[t] * z[t]
        s2 = omega + a * x[t] ** 2 + b * s2
    return x, sig


def ewma_vol(pnl, lam=0.94):
    s2 = np.empty(pnl.size)
    s2[0] = np.var(pnl[:WARMUP])
    for t in range(1, pnl.size):
        s2[t] = lam * s2[t - 1] + (1 - lam) * pnl[t - 1] ** 2
    return np.sqrt(s2)


def tail_mean(losses, var_level):
    tail = losses[losses >= var_level]
    return tail.mean() if tail.size else var_level


def models(pnl, sig):
    n = pnl.size
    out = {}

    # 1. historical simulation
    v99, v975, e975 = (np.full(n, np.nan) for _ in range(3))
    for t in range(WARMUP, n):
        h = -pnl[t - WARMUP : t]
        v99[t], v975[t] = np.quantile(h, 0.99), np.quantile(h, 0.975)
        e975[t] = tail_mean(h, v975[t])
    out["Historical simulation"] = (v99, v975, e975)

    # 2. RiskMetrics
    vol = ewma_vol(pnl)
    z99, z975 = -stats.norm.ppf(0.01), -stats.norm.ppf(0.025)
    out["RiskMetrics EWMA"] = (z99 * vol, z975 * vol, stats.norm.pdf(z975) / 0.025 * vol)

    # 3. conformal EWMA
    a99 = conformal.adaptive_var(pnl, 0.01, window=WARMUP, gamma=0.005, scale=vol)
    a975 = conformal.adaptive_var(pnl, 0.025, window=WARMUP, gamma=0.005, scale=vol)
    c99, c975 = a99["var"], a975["var"]
    guarantee_held = a99["guarantee_holds"] and a975["guarantee_holds"]
    ce = np.full(n, np.nan)
    z = -pnl / vol
    for t in range(WARMUP, n):
        ce[t] = tail_mean(z[t - WARMUP : t], c975[t] / vol[t]) * vol[t]
    out["Conformal EWMA"] = (c99, c975, np.maximum(ce, c975))

    # 4. the truth
    q99, q975 = -stats.t.ppf(0.01, NU) * S, -stats.t.ppf(0.025, NU) * S
    qes = stats.t.pdf(stats.t.ppf(0.025, NU), NU) / 0.025 * (NU + stats.t.ppf(0.025, NU) ** 2) / (NU - 1) * S
    out["True model"] = (q99 * sig, q975 * sig, qes * sig)
    return out, guarantee_held


def main():
    pnl, sig = simulate(WARMUP + N_LIVE, seed=0)
    live = slice(WARMUP, None)
    y = pnl[live]
    losses = {}
    for name, (v99, v975, e975) in models(pnl, sig)[0].items():
        print(vetted.validate_risk_model(y, v99[live], v975[live], e975[live], title=name), "\n")
        losses[name] = es.fz0_loss(y, v975[live], e975[live], 0.025)

    print("Which model forecasts the 97.5% tail best? FZ0 loss, Diebold-Mariano against the historical model:")
    for name in list(losses)[1:]:
        r = compare.diebold_mariano(losses[name], losses["Historical simulation"])
        print(f"  {name:<22} DM {r.statistic:6.2f}  p = {r.p_value:.3g}  lower loss: "
              f"{name if r.detail['better'] == 'a' else 'Historical simulation'}")

    print("\nOn 20 independent histories of 1250 days, how often does each model fail?")
    fails = {}
    guarantee_lost = 0
    for seed in range(1, 21):
        pnl, sig = simulate(WARMUP + N_LIVE, seed=seed)
        y = pnl[live]
        forecasts, held = models(pnl, sig)
        guarantee_lost += not held
        for name, (v99, v975, e975) in forecasts.items():
            rep = vetted.validate_risk_model(y, v99[live], v975[live], e975[live], n_sims=1000, seed=seed)
            fails.setdefault(name, 0)
            fails[name] += rep.status == "fail"
    for name, k in fails.items():
        print(f"  {name:<22} {k:2d} / 20")
    print(
        f"  (the conformal VaR hit its cap and a capped forecast was breached in {guarantee_lost} of 20 "
        "histories, so its formal coverage guarantee did not apply there)"
    )


if __name__ == "__main__":
    main()
