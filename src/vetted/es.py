"""Backtests for Expected Shortfall forecasts.

Expected Shortfall replaced VaR as the capital measure under FRTB, but it is
not elicitable on its own (Gneiting 2011), so there is no single loss
function that backtests it the way the quantile loss backtests VaR. This
module collects the tests that work anyway:

* ``acerbi_szekely``: the Z1 and Z2 statistics, with the fixed Z2 traffic
  light the paper shows is stable across distributions, and exact Monte Carlo
  p-values when you can simulate from your model.
* ``conditional_calibration``: Nolde and Ziegel (2017), a deterministic test
  of VaR and ES jointly, built on their identification function.
* ``exceedance_residuals``: McNeil and Frey (2000), bootstrap test that the
  losses beyond VaR average the forecast ES.
* ``du_escanciano``: unconditional and conditional (clustering) tests from
  the probability integral transforms of the P&L.

Sign conventions as in ``vetted._inputs``: P&L positive for a gain, VaR and
ES positive loss amounts, ``alpha`` the tail probability (0.025 for the FRTB
97.5% ES).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import stats

from ._inputs import align, as_1d, check_level, check_same_index
from ._result import TestResult

#: Z2 thresholds from Acerbi and Szekely (2014), Table 4: 5% and 0.01%
#: significance, stable across Student-t tails from nu = 10 to Gaussian
#: (T = 250, alpha = 2.5%).
Z2_AMBER = -0.70
Z2_RED = -1.8


def _es_inputs(pnl, var, es, missing):
    if missing == "exception":
        raise ValueError("missing='exception' only applies to counting tests; use 'raise' or 'drop'")
    p, v, e = align(pnl, var, es, names=("var", "es"), missing=missing)
    if np.any(e < v - 1e-12 * np.maximum(1.0, np.abs(v))):
        raise ValueError("es must be at least var on every day (ES is the mean loss beyond VaR)")
    if np.any(e <= 0):
        raise ValueError("es must be strictly positive")
    return p, v, e


def _z1(p, v, e):
    hit = p < -v
    n = hit.sum()
    if n == 0:
        return float("nan")
    return float(np.sum(p[hit] / e[hit]) / n + 1.0)


def _z2(p, v, e, alpha):
    hit = p < -v
    return float(np.sum(p * hit / (p.size * alpha * e)) + 1.0)


def acerbi_szekely(
    pnl,
    var,
    es,
    alpha: float = 0.025,
    sampler: Optional[Callable[[np.random.Generator, int], np.ndarray]] = None,
    n_sims: int = 10_000,
    seed: Optional[int] = 0,
    significance: float = 0.05,
    missing: str = "raise",
) -> TestResult:
    """Acerbi and Szekely (2014) tests of Expected Shortfall.

    Z2 = sum_t P&L_t 1{exception_t} / (T alpha ES_t) + 1 tests the frequency
    and the size of tail losses together; Z1 averages P&L_t / ES_t over the
    exceptions only and so tests size given the VaR. Both are 0 in
    expectation under a correct model and negative when risk is
    underestimated. ``statistic`` is Z2; Z1 is in ``detail``.

    Without a ``sampler``, the decision uses the paper's fixed Z2 thresholds
    (amber below -0.70, red below -1.8), which hold across tail shapes for
    T = 250 and alpha = 2.5%, and ``p_value`` is ``nan``. For other sample
    sizes the thresholds are scaled by sqrt(250 / T). That scaling is
    vetted's, not the paper's: unscaled, the 5% threshold would almost never
    fire on 1000 days (a false alarm rate near 0% instead of 5%). Scaled, the
    amber threshold keeps a false alarm rate of about 4-6.5% from T = 60 to
    2500 in simulations. The red (0.01%) threshold is only approximate away
    from T = 250, because the far tail of Z2 is skewed: its false alarm rate
    ranged from about 0.001% to 0.1% over the same span. For an exact level,
    pass a ``sampler``. With a
    ``sampler(rng, n)`` returning an (n, T) array of P&L paths drawn from the
    model's own daily predictive distributions, p-values for Z1 and Z2 are
    exact Monte Carlo p-values (one-sided, small when risk is underestimated).

    Reference: Acerbi, C. and Szekely, B. (2014). Backtesting expected
    shortfall. Risk, December 2014.
    """
    alpha = check_level(alpha)
    p, v, e = _es_inputs(pnl, var, es, missing)
    z1 = _z1(p, v, e)
    z2 = _z2(p, v, e, alpha)
    # The published thresholds are for T = 250. Z2's spread shrinks like
    # 1 / sqrt(T), so scale them; tests/test_properties.py checks by
    # simulation that this keeps the amber false-alarm rate near 5% from
    # T = 125 to T = 2500.
    scale_t = np.sqrt(250.0 / p.size)
    amber, red = Z2_AMBER * scale_t, Z2_RED * scale_t
    zone = "green" if z2 > amber else ("amber" if z2 > red else "red")
    detail = {
        "z1": z1,
        "z2": z2,
        "zone": zone,
        "exceptions": int(np.sum(p < -v)),
        "observations": int(p.size),
        "thresholds": {"amber": amber, "red": red},
    }
    if sampler is None:
        pv = float("nan")
        reject = z2 <= amber
        detail["p_value_source"] = (
            "fixed thresholds (Acerbi and Szekely 2014, Table 4)"
            if p.size == 250
            else f"Acerbi and Szekely (2014) thresholds scaled by sqrt(250 / {p.size})"
        )
    else:
        n_in = as_1d(pnl, "pnl").size
        if p.size != n_in:
            raise ValueError(
                "a sampler cannot be combined with dropped days; the simulated paths must cover "
                "the same days as the P&L"
            )
        rng = np.random.default_rng(seed)
        sims = np.asarray(sampler(rng, n_sims), dtype=float)
        if sims.shape != (n_sims, p.size):
            raise ValueError(f"sampler must return shape ({n_sims}, {p.size}), got {sims.shape}")
        hit = sims < -v
        z2_sim = np.sum(sims * hit / (p.size * alpha * e), axis=1) + 1.0
        nh = hit.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z1_sim = np.sum(np.where(hit, sims / e, 0.0), axis=1) / nh + 1.0
        z1_sim = z1_sim[nh > 0]
        # (1 + count) / (1 + n): a valid p-value that is never exactly zero.
        pv = float((1 + np.sum(z2_sim <= z2)) / (1 + n_sims))
        detail["p_z1"] = (
            float((1 + np.sum(z1_sim <= z1)) / (1 + z1_sim.size)) if z1 == z1 else float("nan")
        )
        detail["p_value_source"] = f"Monte Carlo, {n_sims} paths"
        reject = pv < significance
    return TestResult(
        name="Acerbi-Szekely Z2",
        statistic=z2,
        p_value=pv,
        reject=bool(reject),
        significance=significance,
        null="the tail of the P&L distribution is as forecast",
        reference="Acerbi and Szekely (2014), Risk",
        detail=detail,
    )


def conditional_calibration(
    pnl,
    var,
    es,
    alpha: float = 0.025,
    scale=None,
    significance: float = 0.05,
    missing: str = "raise",
) -> TestResult:
    """Nolde and Ziegel (2017) conditional calibration test of VaR and ES.

    Uses the identification function of the (VaR, ES) pair,
    V1 = alpha - 1{exception} and
    V2 = ES_r - VaR_r + 1{exception} (VaR_r - r) / alpha
    (in return space, where VaR_r = -var and ES_r = -es), which has mean zero
    when both forecasts are right. ``statistic`` and ``p_value`` are the
    two-sided Wald test of the "simple" version. With a volatility forecast
    ``scale``, the "general" versions are added to ``detail``. The two-sided
    tests follow the R package esback (Bayer and Dimitriadis) exactly.

    One-sided tests (``detail["p_simple_one_sided"]`` and
    ``["p_general_one_sided"]``) are the paper's: they reject only when risk
    is underestimated, that is when there are too many exceptions or the
    losses beyond VaR are larger than the ES. esback's one-sided p-values use
    the opposite sign for V1, so they also reject a model that is too
    conservative; they are reported as ``p_simple_one_sided_esback`` and
    ``p_general_one_sided_esback``.

    Use the two-sided test with care on short samples: it is asymptotic, and
    on a correct Student-t(5) model it rejected 24% of the time at 250 days,
    17% at 500 and 13% at 1000 at the 5% level in vetted's simulations (300
    paths each, tools/simulations.py). ``validate_risk_model`` uses the paper's one-sided version.

    Reference: Nolde, N. and Ziegel, J. F. (2017). Elicitability and
    backtesting: perspectives for banking regulation. Annals of Applied
    Statistics, 11(4), 1833-1874.
    """
    alpha = check_level(alpha)
    r, v, e = _es_inputs(pnl, var, es, missing)
    q, es_r = -v, -e
    n = r.size
    hit = (r < q).astype(float)  # an exception is a loss beyond VaR (MAR32.5)
    V = np.column_stack([alpha - hit, es_r - q + hit * (q - r) / alpha])

    def wald(hv, df):
        m = hv.mean(axis=0)
        omega = hv.T @ hv / n
        try:
            t = float(n * m @ np.linalg.solve(omega, m))
        except np.linalg.LinAlgError:
            # e.g. no exceptions with a scale given: the covariance is singular
            return float("nan"), float("nan")
        return t, float(stats.chi2.sf(t, df))

    def hommel(tstats):
        k = len(tstats)
        pv = np.sort(1 - stats.norm.cdf(tstats))
        return float(min(k * np.sum(1.0 / np.arange(1, k + 1)) * np.min(pv / np.arange(1, k + 1)), 1.0))

    def tstats(hv):
        omega = hv.T @ hv / n
        return np.sqrt(n) * np.diag(omega) ** -0.5 * hv.mean(axis=0)

    # Risk underestimation makes V2 positive and V1 negative; the paper's
    # one-sided test flips V1 so that both point the same way.
    flip2 = np.array([-1.0, 1.0])
    t1, p1 = wald(V, 2)
    t3 = tstats(V)
    detail = {
        "p_simple_one_sided": hommel(t3 * flip2),
        # largest one-sided t-ratio (positive when risk looks underestimated)
        "t_simple": float(np.max(t3 * flip2)),
        "p_simple_one_sided_esback": hommel(t3),
        "mean_identification": V.mean(axis=0).tolist(),
    }
    if scale is not None:
        check_same_index(pnl, scale, names=("pnl", "scale"))
        s = as_1d(scale, "scale")
        if s.size != n:
            raise ValueError("scale must have one value per day")
        if np.any(~np.isfinite(s) | (s <= 0)):
            raise ValueError("scale must be positive")
        hv2 = ((q - es_r) / alpha / s * V[:, 0] + V[:, 1] / s)[:, None]
        t2, p2 = wald(hv2, 1)
        hv3 = np.column_stack([V[:, 0], np.abs(q) * V[:, 0], V[:, 1], V[:, 1] / s])
        t4 = tstats(hv3)
        flip4 = np.array([-1.0, -1.0, 1.0, 1.0])
        detail.update(
            {
                "statistic_general": t2,
                "p_general_two_sided": p2,
                "p_general_one_sided": hommel(t4 * flip4),
                "p_general_one_sided_esback": hommel(t4),
            }
        )
    return TestResult(
        name="Nolde-Ziegel conditional calibration",
        statistic=t1,
        p_value=p1,
        reject=p1 < significance,
        significance=significance,
        null="VaR and ES are both correctly calibrated",
        reference="Nolde and Ziegel (2017), Annals of Applied Statistics 11(4)",
        detail=detail,
    )


def exceedance_residuals(
    pnl,
    var,
    es,
    scale=None,
    n_boot: int = 10_000,
    seed: Optional[int] = 0,
    significance: float = 0.05,
    missing: str = "raise",
    min_exceptions: int = 5,
) -> TestResult:
    """McNeil and Frey (2000) exceedance residual test.

    On exception days the realised return minus the forecast ES (in return
    space) should average zero. The statistic is the t-ratio of those
    residuals (divided by ``scale`` if a volatility forecast is given) and
    the one-sided p-value, small when losses beyond VaR are worse than the
    ES said, comes from a centred bootstrap, as in esback's er_backtest.
    The two-sided p-value is in ``detail``.

    The bootstrap needs a handful of exceptions to mean anything: with two,
    every resample reproduces the observed t-ratio or is degenerate, and a
    correct model is rejected whenever that ratio is negative (esback has the
    same behaviour). Below ``min_exceptions`` the test raises ValueError.

    Reference: McNeil, A. J. and Frey, R. (2000). Estimation of tail-related
    risk measures for heteroscedastic financial time series: an extreme value
    approach. Journal of Empirical Finance, 7(3-4), 271-300.
    """
    r, v, e = _es_inputs(pnl, var, es, missing)
    hit = r < -v
    x = r[hit] + e[hit]  # r - ES_r with ES_r = -es
    if scale is not None:
        check_same_index(pnl, scale, names=("pnl", "scale"))
        s = as_1d(scale, "scale")
        if s.size != r.size:
            raise ValueError("scale must have one value per day")
        if np.any(~np.isfinite(s) | (s <= 0)):
            raise ValueError("scale must be positive")
        x = x / s[hit]
    if x.size < max(int(min_exceptions), 3):
        raise ValueError(f"{x.size} exceptions; the bootstrap needs at least {max(int(min_exceptions), 3)}")

    def tstat(a):
        with np.errstate(divide="ignore", invalid="ignore"):
            return a.mean(axis=-1) / a.std(axis=-1, ddof=1) * np.sqrt(a.shape[-1])

    t0 = float(tstat(x))
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, x.size, size=(n_boot, x.size))]
    t = tstat(boot)
    t = t[np.isfinite(t)]
    centred = t - t.mean()
    p1 = float(np.mean(centred <= t0))
    p2 = float(np.mean(np.abs(centred) >= abs(t0)))
    return TestResult(
        name="McNeil-Frey exceedance residuals",
        statistic=t0,
        p_value=p1,
        reject=p1 < significance,
        significance=significance,
        null="losses beyond VaR average the forecast ES",
        reference="McNeil and Frey (2000), Journal of Empirical Finance 7(3-4)",
        detail={"p_two_sided": p2, "exceptions": int(x.size), "mean_residual": float(x.mean())},
    )


def _de_stats(u, alpha, lags):
    """Unconditional z and Box-Pierce C for (..., n) PIT arrays."""
    n = u.shape[-1]
    h = (alpha - u) * (u <= alpha) / alpha
    z = np.sqrt(n) * (h.mean(axis=-1) - alpha / 2) / np.sqrt(alpha * (1 / 3 - alpha / 4))
    d = h - alpha / 2
    g0 = np.mean(d * d, axis=-1)
    rho = np.stack([np.sum(d[..., j:] * d[..., :-j], axis=-1) / (n - j) / g0 for j in range(1, lags + 1)], axis=-1)
    c = n * np.sum(rho**2, axis=-1)
    return z, c, h, rho


def du_escanciano(
    pit,
    alpha: float = 0.025,
    lags: int = 5,
    significance: float = 0.05,
    pvalue: str = "asymptotic",
    n_sims: int = 10_000,
    seed: Optional[int] = 0,
) -> TestResult:
    """Du and Escanciano (2017) backtests from probability integral transforms.

    ``pit`` is u_t = F_t(P&L_t), the model's own forecast CDF evaluated at the
    realised P&L (small u is a large loss). The cumulative violation
    H_t = (alpha - u_t) 1{u_t <= alpha} / alpha has mean alpha / 2 and
    variance alpha (1/3 - alpha/4) under a correct model, whatever the
    distribution.

    ``statistic`` and ``p_value`` are the unconditional test (one-sided,
    rejecting when the tail is heavier than forecast). The conditional test in
    ``detail`` is a Box-Pierce statistic on the first ``lags``
    autocorrelations of H_t - alpha/2. Its null is E[H_t | past] = alpha/2,
    so it rejects clustering of tail losses but also a miscalibrated tail
    (a model that is too conservative has few tail events and fails it too).

    ``pvalue="monte_carlo"`` simulates both statistics under the null
    (independent uniform PITs), which keeps its level at any sample size. The
    asymptotic conditional test is oversized: in vetted's simulations it
    rejected a correct model 12.8% of the time at 250 days and 9.6% at 1000,
    at a nominal 5% (tools/simulations.py).

    Reference: Du, Z. and Escanciano, J. C. (2017). Backtesting expected
    shortfall: accounting for tail risk. Management Science, 63(4), 940-958.
    """
    alpha = check_level(alpha)
    if pvalue not in ("asymptotic", "monte_carlo"):
        raise ValueError("pvalue must be 'asymptotic' or 'monte_carlo'")
    u = as_1d(pit, "pit")
    if np.any(~np.isfinite(u)) or np.any((u < 0) | (u > 1)):
        raise ValueError("pit must be finite and in [0, 1]")
    n = u.size
    if n <= lags + 1:
        raise ValueError("not enough observations for the requested lags")
    z, c, h, rho = _de_stats(u, alpha, lags)
    z, c = float(z), float(c)
    if pvalue == "asymptotic":
        p_u = float(stats.norm.sf(z))
        p_c = float(stats.chi2.sf(c, lags))
    else:
        rng = np.random.default_rng(seed)
        zs, cs = [], []
        for start in range(0, n_sims, 2000):
            b = min(2000, n_sims - start)
            zz, cc, _, _ = _de_stats(rng.random((b, n)), alpha, lags)
            zs.append(zz)
            cs.append(cc)
        zs, cs = np.concatenate(zs), np.concatenate(cs)
        p_u = float((1 + np.sum(zs >= z - 1e-12)) / (1 + n_sims))
        p_c = float((1 + np.sum(cs >= c - 1e-12)) / (1 + n_sims))
    return TestResult(
        name="Du-Escanciano unconditional",
        statistic=z,
        p_value=p_u,
        reject=p_u < significance,
        significance=significance,
        null="the forecast tail is correct on average",
        reference="Du and Escanciano (2017), Management Science 63(4)",
        detail={
            "conditional_statistic": c,
            "p_conditional": p_c,
            "reject_conditional": p_c < significance,
            "lags": lags,
            "autocorrelations": np.asarray(rho).tolist(),
            "mean_h": float(h.mean()),
            "expected_mean_h": alpha / 2,
            "p_value_method": pvalue,
        },
    )


def fz0_loss(pnl, var, es, alpha: float = 0.025) -> np.ndarray:
    """Per-day FZ0 loss of a joint (VaR, ES) forecast.

    VaR and ES are not separately elicitable for comparison purposes but the
    pair is (Fissler and Ziegel 2016). FZ0 is the zero-degree homogeneous
    member of that family, recommended by Patton, Ziegel and Chen (2019)
    because loss differences do not depend on the scale of the P&L. Lower is
    better; compare two models with ``vetted.compare.diebold_mariano``.

    In return space with v = -var, e = -es (both negative):
    L = -1{r <= v} (v - r) / (alpha e) + v / e + log(-e) - 1.

    Reference: Patton, A. J., Ziegel, J. F. and Chen, R. (2019). Dynamic
    semiparametric models for expected shortfall (and value-at-risk). Journal
    of Econometrics, 211(2), 388-413.
    """
    alpha = check_level(alpha)
    r, var_, es_ = _es_inputs(pnl, var, es, "raise")
    v, e = -var_, -es_
    hit = (r <= v).astype(float)
    return -hit * (v - r) / (alpha * e) + v / e + np.log(-e) - 1.0
