# Conformal VaR (`vetted.conformal`)

Historical-simulation VaR takes an empirical quantile and hopes. Split
conformal prediction takes the same quantile with a (n + 1) correction and
gets a guarantee: if the calibration days and the next day are
exchangeable, the next loss exceeds the VaR with probability at most alpha,
with no distributional assumption.

```python
from vetted import conformal
v = conformal.conformal_var(last_250_days_pnl, alpha=0.01)
```

With a volatility forecast (GARCH, EWMA, anything), calibration runs on
standardised losses and the result is that model's shape with a coverage
that is guaranteed rather than assumed:

```python
v = conformal.conformal_var(pnl, 0.01, scale=vol_history, next_scale=vol_tomorrow)
```

## Adaptive conformal VaR

Markets are not exchangeable for long. `adaptive_var` produces
out-of-sample forecasts that steer their level online (Gibbs and Candes
2021), so the long-run exception rate converges to alpha whatever the data
do:

```python
out = conformal.adaptive_var(pnl, alpha=0.01, window=250, gamma=0.005, scale=ewma_vol)
out["var"], out["exception_rate"], out["bound"], out["guarantee_holds"]
```

* With `unbounded="inf"` the formal bound holds for any sequence.
* With the default `unbounded="max_loss"`, a forecast that would be infinite
  is capped at the window's largest loss so it stays usable. If a capped
  forecast is then breached, the guarantee no longer applies, and the result
  says so in `guarantee_holds` (with `capped_days` and `capped_exceptions`).

## Example

In `examples/risk_model_validation.py`, 20 simulated 1250-day histories of a
GARCH desk with fat tails: historical simulation failed validation 17 times,
RiskMetrics EWMA 16 times, the same EWMA volatility with an adaptive conformal
quantile 0 times, and the true model once. In 4 of the 20 histories the
conformal forecast hit its cap and was breached, so its formal guarantee did
not apply there. This is one simulated process, not a general result.
