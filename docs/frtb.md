# Basel FRTB (`vetted.frtb`)

Implements chapter MAR32 of the Basel Framework, "Internal models approach:
backtesting and P&L attribution test requirements", in force from
1 January 2023, and the binomial logic of MAR99 behind its zones.

## Traffic light

```python
from vetted import frtb
r = frtb.traffic_light(var_99, hypothetical_pnl, actual_pnl)
r.detail["zone"], r.detail["multiplier"], r.detail["exceptions"]
```

* Counts the most recent 250 days by default (`window=250`); pass
  `window=None` for the whole series.
* Counts exceptions against hypothetical and actual P&L separately and uses
  the greater (MAR32.5).
* Counts a missing P&L or VaR as an exception (MAR32.5).
* The zone comes from `binomial_zones(n, alpha)`, which reproduces MAR32.9
  Table 1 exactly for 250 days at 99% (green 0-4, amber 5-9, red 10+) and
  extends the same rule to any window and level.
* The multiplier is Table 1's (1.50 to 2.00) for the standard window.

## Trading desk backtest

```python
frtb.desk_backtest(var_99, var_975, hypothetical_pnl, actual_pnl)
```

A desk stays on the internal models approach only with at most 12
exceptions at the 99th percentile and 30 at the 97.5th over the most recent
250 days (MAR32.19). `detail["zone"]` is `"eligible"` or `"ineligible"`.

## P&L attribution test

```python
frtb.pla_test(risk_theoretical_pnl, hypothetical_pnl)
```

Spearman correlation and the Kolmogorov-Smirnov distance between RTPL and
HPL, zoned per MAR32.42 Table 2: green needs correlation above 0.80 and KS
below 0.09; red is correlation below 0.70 or KS above 0.12; amber otherwise.
`pla_zone(spearman, ks)` exposes the rule, and the boundaries are tested
exactly (0.80 itself is amber).

!!! note
    A regulatory result from vetted is an implementation of the published
    text, not a supervisor's decision.
