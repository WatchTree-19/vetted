"""Reproduce every simulated number quoted in README.md.

    python tools/simulations.py          # a few minutes

Each block prints the figure the README quotes. Seeds are fixed, so the
output is identical from run to run on the same numpy version.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

import vetted
from vetted import es, overfitting, var

NU = 5
S = np.sqrt((NU - 2) / NU)
Q975 = stats.t.ppf(0.025, NU)
VAR975 = -Q975 * S
ES975 = stats.t.pdf(Q975, NU) / 0.025 * (NU + Q975**2) / (NU - 1) * S
VAR99 = -stats.t.ppf(0.01, NU) * S


def z2_paths(rng, T, n, nu):
    s = np.sqrt((nu - 2) / nu) if nu else 1.0
    q = stats.t.ppf(0.025, nu) if nu else stats.norm.ppf(0.025)
    v = -q * s
    e = (stats.t.pdf(q, nu) / 0.025 * (nu + q**2) / (nu - 1) * s) if nu else stats.norm.pdf(q) / 0.025
    x = (rng.standard_t(nu, (n, T)) * s) if nu else rng.standard_normal((n, T))
    return np.sum(x * (x < -v), axis=1) / (T * 0.025 * e) + 1


def acerbi_szekely_thresholds():
    print("1. Acerbi-Szekely Z2 amber threshold, false alarm rate on correct models")
    rng = np.random.default_rng(1)
    z = z2_paths(rng, 1000, 300, NU)
    print(f"   unscaled -0.70 at T=1000, Student-t(5): {int(np.sum(z <= es.Z2_AMBER))} of 300")
    rates = []
    for nu in (NU, None):
        for T in (125, 250, 500, 1000, 2500):
            z = z2_paths(rng, T, 20_000, nu)
            rate = float(np.mean(z <= es.Z2_AMBER * np.sqrt(250 / T)))
            rates.append(rate)
            print(f"   scaled, T={T:5d}, {'t(5)' if nu else 'normal'}: {rate:.3%}")
    print(f"   range: {min(rates):.1%} to {max(rates):.1%}")


def nolde_ziegel_size():
    print("2. Nolde-Ziegel two-sided simple test, rejection rate of a correct t(5) model at 5%")
    rng = np.random.default_rng(2)
    for n in (250, 500, 1000):
        rej = sum(es.conditional_calibration(rng.standard_t(NU, n) * S, VAR975, ES975).reject for _ in range(300))
        print(f"   T={n}: {rej / 300:.0%}")


def nolde_ziegel_one_sided():
    print("2b. Nolde-Ziegel one-sided (paper's direction), rejection rate at 5%")
    rng = np.random.default_rng(6)
    for n in (250, 1000):
        out = []
        for mult in (1.0, 1.3, 0.7):
            rej = sum(
                es.conditional_calibration(rng.standard_t(NU, n) * S * mult, VAR975, ES975).detail["p_simple_one_sided"]
                < 0.05
                for _ in range(300)
            )
            out.append(rej / 300)
        print(f"   T={n}: correct {out[0]:.1%}, risk understated 30% {out[1]:.0%}, overstated 30% {out[2]:.0%}")


def dq_size():
    print("3. DQ test on a correct 99% VaR, 250 days, rejection rate at 5%")
    rng = np.random.default_rng(3)
    q = -stats.norm.ppf(0.01) * np.ones(250)
    a = m = 0
    for i in range(1000):
        p = rng.standard_normal(250)
        a += var.dynamic_quantile(p, q, 0.01).reject
        m += var.dynamic_quantile(p, q, 0.01, pvalue="monte_carlo", n_sims=1000, seed=i).reject
    print(f"   chi-squared: {a / 1000:.1%}   Monte Carlo: {m / 1000:.1%}")
    print("3b. Du-Escanciano conditional test on correct (uniform) PITs, rejection rate at 5%")
    for n in (250, 1000):
        a = m = 0
        for i in range(1000):
            u = rng.random(n)
            a += es.du_escanciano(u, 0.025).detail["reject_conditional"]
            m += es.du_escanciano(u, 0.025, pvalue="monte_carlo", n_sims=1000, seed=i).detail["reject_conditional"]
        print(f"   T={n}: asymptotic {a / 1000:.1%}   Monte Carlo {m / 1000:.1%}")


def report_false_alarms():
    print("4. validate_risk_model on correct t(5) models: 'any statistical test rejects' vs the report verdict")
    rng = np.random.default_rng(4)
    for n in (250, 1000):
        raw = verdict = 0
        reps = 300
        for i in range(reps):
            x = rng.standard_t(NU, n) * S
            rep = vetted.validate_risk_model(x, VAR99, VAR975, ES975, n_sims=1000, seed=i)
            stat = [r for r in rep.results if "zone" not in r.detail]
            raw += any(r.reject for r in stat)
            verdict += rep.status == "fail"
        print(f"   T={n}: any test {raw / reps:.0%}, report fails the model {verdict / reps:.1%}")


def dsr_correlated():
    print("5. Deflated Sharpe on the best of 50 null strategies with pairwise correlation 0.7")
    rng = np.random.default_rng(5)
    raw = eff = 0
    reps = 400
    for _ in range(reps):
        common = rng.standard_normal((500, 1))
        M = 0.01 * (np.sqrt(0.7) * common + np.sqrt(0.3) * rng.standard_normal((500, 50)))
        best = M[:, np.argmax(M.mean(0) / M.std(0, ddof=1))]
        raw += overfitting.deflated_sharpe(best, M).reject
        n_eff = overfitting.effective_trials(M)
        eff += overfitting.deflated_sharpe(best, M, independent_trials=max(n_eff, 1.0)).reject
    print(f"   all trials: {raw / reps:.1%} false positives; also shrinking N to the effective number: {eff / reps:.1%}")


if __name__ == "__main__":
    acerbi_szekely_thresholds()
    nolde_ziegel_size()
    nolde_ziegel_one_sided()
    dq_size()
    report_false_alarms()
    dsr_correlated()
