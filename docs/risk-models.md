# Risk models

## The one-call report

`vetted.validate_risk_model` runs every applicable test and returns a
`Report`:

| input | tests it adds |
|---|---|
| `pnl`, `var_99` | Basel traffic light; Kupiec, Christoffersen and DQ at 99% |
| `var_975` | Kupiec, Christoffersen and DQ at 97.5%; FRTB desk backtest |
| `es_975` (with `var_975`) | Acerbi-Szekely Z2; Nolde-Ziegel one-sided; McNeil-Frey |
| `pit` | Du-Escanciano unconditional and conditional |
| `actual_pnl` | counted in the traffic light and desk backtest (the greater count is used) |
| `risk_theoretical_pnl` | FRTB PLA test |
| `scale` | the standardised McNeil-Frey residuals |

### How the verdict is reached

Running about eight tests at 5% each would flag a correct model far more
often than 5% of the time. The report therefore:

1. adjusts the p-values of the statistical tests together with Holm's method
   (`report.holm_p()`), and counts a test against the model only if its
   adjusted p-value is below `significance`;
2. keeps regulatory zones outside that family: a red zone or an ineligible
   desk fails the model, and an amber zone is reported as a flag
   (`report.flagged`) but does not fail it, because MAR32.11 says an accurate
   model can land in amber;
3. uses Monte Carlo p-values for every VaR test and for Du-Escanciano, because
   the chi-squared limits are unreliable with the handful of exceptions a year
   of 99% VaR produces.

`report.status` is `"pass"`, `"amber"` or `"fail"`. On correct Student-t(5)
models, "any test rejects" happened 14% of the time at 250 days and 20% at
1000; the report failed 2.3% and 3.7% of them.

### Missing data

Statistical tests default to `missing="raise"`; pass `"drop"` to remove
incomplete days. The regulatory counts always treat a missing P&L or VaR as
an exception, as MAR32.5 requires, and refuse `"drop"`.

## VaR tests (`vetted.var`)

| function | what it tests | checked against |
|---|---|---|
| `kupiec` | exception rate equals alpha (two-sided) | R rugarch `VaRTest` |
| `christoffersen` | exceptions are independent and at rate alpha | R rugarch `VaRTest` |
| `dynamic_quantile` | past hits and the VaR level do not predict the next hit | R GAS `BacktestVaR` (`squared_pnl_lag=True`) |
| `quantile_loss` | per-day loss for comparing VaR models | R GAS |

Each test takes `pvalue="asymptotic"` (default, matches the R packages) or
`pvalue="monte_carlo"`, which simulates independent exceptions with the
forecasts held fixed. The DQ statistic does not depend on the units of the
P&L, and its degrees of freedom drop by one when the VaR is constant.

## Expected Shortfall tests (`vetted.es`)

ES is not elicitable on its own, so no single loss function backtests it.
These tests work anyway:

| function | what it tests | checked against |
|---|---|---|
| `acerbi_szekely` | Z2 (frequency and size of tail losses), Z1 in `detail` | Acerbi and Szekely (2014) Table 4 |
| `conditional_calibration` | Nolde-Ziegel identification function of (VaR, ES) | R esback `cc_backtest` |
| `exceedance_residuals` | losses beyond VaR average the forecast ES | R esback `er_backtest` |
| `du_escanciano` | cumulative violations from the PITs | hand calculation, simulated size |
| `fz0_loss` | joint (VaR, ES) loss for model comparison | R GAS `FZLoss` |

Notes that matter in practice:

* **Acerbi-Szekely thresholds.** The published -0.70 and -1.8 are for 250
  days. vetted scales them by sqrt(250 / T); the amber threshold then keeps a
  false alarm rate of 4.5-6.1% from T = 125 to 2500. The red threshold is
  approximate away from 250 days. Pass a `sampler` for Monte Carlo p-values.
* **Nolde-Ziegel.** The two-sided simple test is oversized on short samples
  (24% at 250 days on a correct model). The paper's one-sided test, which
  rejects only risk underestimation, is in `detail["p_simple_one_sided"]`
  and is what the report uses. esback's one-sided variant, which also
  rejects conservative models, is reported under the `_esback` keys.
* **McNeil-Frey** needs at least five exceptions; below that the bootstrap is
  degenerate and the test raises.

## Comparing two models (`vetted.compare`)

```python
from vetted import compare, es
la = es.fz0_loss(pnl, var_a, es_a, 0.025)
lb = es.fz0_loss(pnl, var_b, es_b, 0.025)
compare.diebold_mariano(la, lb)   # negative statistic: model A is better
```

`diebold_mariano` applies the Harvey-Leybourne-Newbold correction and
matches R forecast `dm.test`.
