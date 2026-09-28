# How it is verified

A validation library is only worth using if its own numbers are right.

## R packages

`oracle/gen_backtest_oracle.R` runs rugarch 1.5.6, GAS 0.3.4, esback 0.3.1,
forecast 9.0.2, pbo 1.3.5 and PerformanceAnalytics 2.1.0 on fixed P&L paths
and writes their output to `tests/data/backtest_oracle.json`. The tests
require vetted to reproduce the closed-form statistics to a relative 1e-8 or
better, their p-values to 1e-7 or better, and esback's bootstrap test to
within bootstrap error. `oracle/gen_metrics_oracle.R` does the same for the
metric conventions, to 1e-10.

## Published numbers

* Bailey and Lopez de Prado (2014): hurdle 0.1132 and DSR 0.9004 from the
  worked example.
* MAR32 Table 1 zones and the cumulative binomial probabilities printed
  beside them (89.22% at 4 exceptions, 95.88% at 5, 99.99% at 10).
* MAR32 Table 2 PLA boundaries, tested at the exact thresholds.
* Acerbi and Szekely (2014) Table 4: simulated Z2 recovers the -0.70
  threshold.

## Behaviour

Simulations check that the backtests hold their size on correct models,
reject broken ones, and that the conformal VaR keeps its coverage guarantee.
Every simulated figure quoted in these docs is printed by
`tools/simulations.py` with fixed seeds.

## The tests test

`tools/mutation_check.py` plants 56 bugs in the source one at a time (a
wrong Euler constant, a missing Harvey-Leybourne-Newbold factor, a PLA
threshold of 0.10 instead of 0.09, Bonferroni in place of Holm, and so on)
and checks that the suite fails on every one. CI runs it on every push,
alongside the tests on Python 3.9 to 3.13 and on the oldest supported numpy
and scipy.

## What building it found

1. The Acerbi-Szekely Z2 thresholds are calibrated for 250 days; unscaled,
   the 5% threshold fired 0 times in 300 correct 1000-day simulations.
2. The Nolde-Ziegel two-sided test rejected a correct model 24% of the time
   at 250 days, 17% at 500 and 13% at 1000.
3. The chi-squared DQ test rejected a correct 99% VaR 6.4% of the time on 250
   days; the Du-Escanciano conditional test 12.8% at 250 days.
4. Running every test at 5% flagged correct models 14-20% of the time; the
   Holm verdict with amber flags brought that to 2.3-3.7%.
5. Reference implementations differ in small ways: R pbo divides the rank by
   N rather than N + 1; GAS adds yesterday's squared return to DQ and keeps
   seven degrees of freedom under collinearity; rugarch's `VaRTest` errors
   when no two exceptions are consecutive; esback's exceedance bootstrap
   rejects a correct model with exactly two exceptions and a negative
   t-ratio.
