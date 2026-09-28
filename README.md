# vetted

**Independent validation for risk models and trading strategies.**

`vetted` answers the two questions every quant result has to survive:

* **Is this risk model right?** VaR and Expected Shortfall backtests, the
  Basel FRTB traffic light, desk eligibility and P&L attribution tests, and a
  one-call validation report for a model risk file.
* **Is this backtest real?** The deflated Sharpe ratio, the probability of
  backtest overfitting, the minimum track record length and multiple-testing
  haircuts, for results that were the best of many tries, including tries
  made by grid searches, genetic algorithms and AI research agents.

Every statistic is checked against an R package, a number printed in the
paper or regulation that defines it, or a hand calculation; the few with no
reference implementation (the one-sided Nolde-Ziegel test, the conformal
VaR) are checked by simulating the property that defines them. The test
that checks each one says which. A validation library is only worth using if its
own numbers are right.

```bash
pip install git+https://github.com/WatchTree-19/vetted
```

Requires only numpy and scipy.

## Validate a risk model

```python
import vetted

report = vetted.validate_risk_model(
    pnl,                  # daily hypothetical P&L, gains positive
    var_99,               # 99% one-day VaR per day, as a positive loss
    var_975, es_975,      # 97.5% VaR and ES (the FRTB measure)
    actual_pnl=apl,       # optional: MAR32.5 counts the greater of APL and HPL
    risk_theoretical_pnl=rtpl,  # optional: runs the PLA test
)
print(report)
report.to_markdown()      # for the validation file
```

```
Historical simulation
=====================
observations: 1250

test                                               statistic         p    Holm p  result
Basel traffic light                                        5    0.1078       n/a  AMBER
Kupiec POF (99%)                                       4.848   0.03648    0.2039  PASS
Christoffersen conditional coverage (99%)              5.677   0.03398    0.2039  PASS
Engle-Manganelli DQ (99%)                              30.56   0.01199   0.08396  PASS
Kupiec POF (97.5%)                                     2.312    0.1464     0.375  PASS
Christoffersen conditional coverage (97.5%)            4.137    0.1069     0.375  PASS
Engle-Manganelli DQ (97.5%)                            30.65  0.0009995  0.007996  FAIL
FRTB desk backtest                                         5       n/a       n/a  ELIGIBLE
Acerbi-Szekely Z2                                    -0.3909       n/a       n/a  AMBER
Nolde-Ziegel conditional calibration (one-sided)       1.863   0.09375     0.375  PASS
McNeil-Frey exceedance residuals                      -1.092    0.1265     0.375  PASS

FAILED: Engle-Manganelli DQ (97.5%)
note: The Basel traffic light and desk backtest count the most recent 250 days (MAR32.3, MAR32.19); the statistical tests use all 1250.
```

That is a 250-day historical simulation model on a desk whose volatility
clusters ([`examples/risk_model_validation.py`](examples/risk_model_validation.py)).
The Basel traffic light, which only counts exceptions, says amber, and an
amber zone alone does not fail a model (MAR32.11). The dynamic quantile test
sees what counting cannot: the exceptions arrive in bunches, because the
model reacts to a volatility spike a year too late. Eight statistical tests
run, so their p-values are adjusted together (Holm) before any of them can
fail the model.

## Validate a backtested strategy

```python
report = vetted.validate_strategy(best_returns, trials=all_trial_returns)
```

```
Search 1 (no skill)
===================
observations: 1260
annualised Sharpe ratio: 1.046
probabilistic Sharpe ratio (vs 0): 0.9901
minimum track record (observations): 627.9
trials: 200
effective independent bets (descriptive): 54.68
hurdle Sharpe ratio (best of noise): 1.141
haircut Sharpe ratio (Holm): 0

test                                  statistic         p    Holm p  result
Deflated Sharpe ratio                    0.4159    0.5841       n/a  FAIL
Probability of backtest overfitting      0.6321       n/a       n/a  FAIL

FAILED: Deflated Sharpe ratio, Probability of backtest overfitting
```

That is the best of 200 strategies with no skill at all
([`examples/strategy_search.py`](examples/strategy_search.py)). Its Sharpe
ratio of 1.05 is 99% significant on its own. Allowing for the 200 tries, it
is noise: the best of that many skill-less strategies is expected to show
1.14. When one of the 200 ideas is real (a true Sharpe ratio near 3), the
same report passes it.

The trials in the example are correlated, like real variations on ten
ideas. The deflation measures the spread of Sharpe ratios across the trials
themselves, so correlated trials, which have similar Sharpe ratios, raise
the hurdle less than independent ones would. On the best of 50 null
strategies with pairwise correlation 0.7, this gives 2% false positives at
a nominal 5%. Also shrinking the count to an "effective number of trials"
would discount the correlation twice: 19%.

## What is in it

| module | what | checked against |
|---|---|---|
| `vetted.var` | Kupiec, Christoffersen, Engle-Manganelli DQ, with asymptotic or Monte Carlo p-values; quantile loss | R rugarch `VaRTest`, R GAS `BacktestVaR` |
| `vetted.es` | Acerbi-Szekely Z1 and Z2, Nolde-Ziegel conditional calibration, McNeil-Frey exceedance residuals, Du-Escanciano (asymptotic or Monte Carlo), FZ0 loss | R esback, R GAS `FZLoss`, Acerbi and Szekely (2014) Table 4, a hand calculation for Du-Escanciano |
| `vetted.frtb` | Basel traffic light and multiplier, trading desk backtest, PLA test (Spearman and KS) | Basel Framework MAR32 Tables 1 and 2, the 1996 cumulative probabilities |
| `vetted.compare` | Diebold-Mariano with the Harvey-Leybourne-Newbold correction | R forecast `dm.test` |
| `vetted.conformal` | VaR with a finite-sample coverage guarantee; adaptive conformal VaR | coverage checked by simulation |
| `vetted.overfitting` | Deflated and probabilistic Sharpe, minimum track record, PBO, Holm and BHY haircuts, effective trials | Bailey and Lopez de Prado (2014) worked example, R PerformanceAnalytics, R pbo, statsmodels |
| `vetted.metrics` | 40 conventions for 15 performance and risk metrics | R PerformanceAnalytics, Bacon (2008) |
| `vetted.report` | `validate_risk_model`, `validate_strategy` | |

Conventions: P&L is positive for a gain; VaR and ES are positive loss
amounts, as risk reports and the Basel text state them; an exception is a
loss larger than the VaR (MAR32.5). Every test returns a `TestResult` with
the statistic, p-value, decision, null hypothesis and the reference that
defines it.

## Every number has a receipt

* **R packages.** `oracle/gen_backtest_oracle.R` runs rugarch 1.5.6,
  GAS 0.3.4, esback 0.3.1, forecast 9.0.2, pbo 1.3.5 and
  PerformanceAnalytics 2.1.0 on fixed P&L paths and writes their output to
  `tests/data/backtest_oracle.json`. The tests require vetted to reproduce
  the closed-form statistics to a relative 1e-8 or better and their p-values
  to 1e-7 or better, and esback's bootstrap test to within bootstrap error.
  The 40 metric conventions reproduce PerformanceAnalytics to 1e-10
  (`oracle/gen_metrics_oracle.R`).
* **Published numbers.** The deflated Sharpe ratio reproduces the worked
  example in Bailey and Lopez de Prado (2014) (hurdle 0.1132, DSR 0.9004).
  The traffic light reproduces MAR32 Table 1 and the cumulative
  probabilities beside it (89.22% at 4 exceptions, 95.88% at 5, 99.99% at
  10). The PLA zones are tested at the exact boundaries of MAR32 Table 2.
  Simulated Z2 reproduces the -0.70 threshold in Acerbi and Szekely's Table 4.
* **Behaviour.** Simulations check that the backtests hold their size on
  correct models, reject broken ones, and that the conformal VaR keeps its
  coverage guarantee.
* **The tests test.** [`tools/mutation_check.py`](tools/mutation_check.py)
  plants 56 bugs in the source one at a time, from a wrong Euler constant to
  a missing Harvey-Leybourne-Newbold factor to a PLA threshold of 0.10
  instead of 0.09 to Bonferroni in place of Holm, and checks that the suite
  fails on every one. CI runs it on every push.
* **The claims below are reproducible.** Every simulated figure in this
  README is printed by [`tools/simulations.py`](tools/simulations.py) with
  fixed seeds.

## What building it found

1. **The Acerbi-Szekely Z2 thresholds are for 250 days.** Applied unscaled to
   1000 days of a correct model, the 5% threshold fired 0 times in 300
   simulations, so the test has almost no power there. vetted scales the
   amber threshold by sqrt(250 / T), which kept the false alarm rate between
   4.5% and 6.1% from T = 125 to 2500 for Student-t(5) and Gaussian P&L. The
   scaling is vetted's, not the paper's; the red threshold is scaled the
   same way but is only approximate away from 250 days, and a `sampler`
   gives Monte Carlo p-values at any length.
2. **The Nolde-Ziegel two-sided test is oversized on short samples.** On a
   correct Student-t(5) model it rejected 24% of the time at 250 days, 17%
   at 500 and 13% at 1000, at a nominal 5%. The one-sided version from the
   paper, which rejects only when risk is underestimated, never exceeded its
   level (0.3% at 250 days, 2% at 1000) and caught risk understated by 30%
   in 40% and 99% of cases.
   esback's one-sided variant also rejects models that are too
   conservative; vetted reports both and uses the paper's in its report.
3. **Asymptotic p-values are unreliable with few exceptions.** The
   chi-squared DQ test rejected a correct 99% VaR over 250 days 6.4% of the
   time at 5%; the Du-Escanciano conditional test 12.8% at 250 days and 9.6%
   at 1000. Monte Carlo p-values (simulating the null with the forecasts
   held fixed) held their level within simulation error (Du-Escanciano 5.5%
   and 4.6%, 1000 paths each) and are conservative when the statistic is
   discrete (3.5% for the DQ case above); the report uses them.
4. **Running every test at 5% fails correct models.** On correct Student-t(5)
   models, "any statistical test rejects" happened 14% of the time at 250
   days and 20% at 1000. The report adjusts the p-values together with
   Holm's method and treats amber zones as flags, as MAR32.11 allows (a
   correct 99% VaR lands in amber about 11% of the time). It failed 2.3% of
   correct models at 250 days and 3.7% at 1000.
5. **A conformalised EWMA passed where the standard models failed.** In the
   example, on 20 simulated 1250-day histories of a GARCH desk with fat
   tails, historical simulation failed validation 17 times and RiskMetrics
   EWMA 16 times. The same EWMA volatility with its quantile set by adaptive
   conformal inference failed 0 times, and the true model once. Adaptive
   conformal inference steers the exception rate toward its target by
   design, so passing the counting tests is expected; the DQ, ES and
   clustering tests, which it does not target, passed too. In 4 of the 20
   histories its 99% level fell far enough that the forecast was capped at
   the window's largest loss and then breached, so its formal coverage
   guarantee did not apply there (`adaptive_var` reports this as
   `guarantee_holds`). This is one simulated process, not a general result.
6. **Reference implementations differ in small ways.** The R package pbo
   divides the out-of-sample rank by N where Bailey et al. (2017) divide by
   N + 1; GAS adds yesterday's squared return to the DQ regression and
   counts seven degrees of freedom even when a constant VaR makes the
   regressors collinear; rugarch's `VaRTest` stops with an error when no
   two exceptions are consecutive; esback's exceedance residual bootstrap
   rejects a correct model whenever there are exactly two exceptions and
   their t-ratio is negative. vetted follows the papers by default,
   reproduces each package's choice as an option where it is a choice, and
   refuses the bootstrap below five exceptions.

## What the Python performance libraries compute

`python -m vetted.conformance` calls each installed library through its
public API on ten fixtures and names the convention it matches (full detail
in [CONFORMANCE.md](CONFORMANCE.md)). (-) means the library reports a loss as
a positive number.

| metric | quantstats 0.0.86 | empyrical-reloaded 0.5.12 | ffn 1.2.2 | skfolio 1.4.5 | riskfolio-lib 7.3.0 |
|---|---|---|---|---|---|
| annual volatility | std (n-1) | std (n-1) | | | |
| annual return | geometric | geometric | geometric over calendar time | | |
| Sharpe, annualised | arithmetic, std (n-1) | same | same | | |
| Sortino | full-sample downside deviation (n) | same | same | | |
| maximum drawdown | geometric, starting capital as a peak (-) | same (-) | same (-) | same | same |
| Calmar | CAGR / max drawdown | same | calendar CAGR / max drawdown | | |
| VaR 95% | **Gaussian** | interpolated quantile | | upper order statistic (-) | lower order statistic (-) |
| ES 95% | **Gaussian** | mean at or below the interpolated quantile | | Rockafellar-Uryasev (-) | Rockafellar-Uryasev (-) |
| Omega at 0 | simple | simple | | | |
| Ulcer index | **matches none (divides by n - 1)** | | | Martin (n) | Martin (n) |
| skewness | adjusted G1 | | | moment g1 | |
| kurtosis | excess, adjusted | | | **not excess** (Pearson, normal = 3) | |

On identical returns, quantstats' Gaussian VaR differs from the empirical
VaR of the other libraries by 19% on Student t(3) returns and 30% on a
regime-switching series; kurtosis means excess in one library and raw in
another; riskfolio-lib's drawdown measures ignore everything after the first
missing return (78% too small on one fixture). Fixes for the quantstats
Ulcer index and the riskfolio-lib missing-data handling are open upstream,
and two earlier findings were fixed in quantstats 0.0.83.

A library can pin its own estimators to these values with no dependency:

```python
from vetted.testing import cases

@pytest.mark.parametrize("returns, periods, expected", cases("sortino_annual", "full_ddof0"))
def test_sortino(returns, periods, expected):
    assert my_sortino(returns, periods) == pytest.approx(expected, rel=1e-9)
```

## Reproducing

```bash
pip install -e ".[test]"
pytest                                   # about 30 seconds
python tools/simulations.py              # the README's simulated figures, a few minutes
python tools/mutation_check.py           # plants 56 bugs, about 10 minutes

# regenerate the oracles (R with the packages named above)
Rscript oracle/gen_backtest_oracle.R tests/data/backtest_fixtures.json tests/data/backtest_oracle.json
Rscript oracle/gen_metrics_oracle.R src/vetted/data/fixtures.json src/vetted/data/oracle.json

python examples/risk_model_validation.py
python examples/strategy_search.py
```

## Scope

One-day horizons and single P&L series. Multi-day overlapping backtests,
multivariate and portfolio-level tests, the FRTB risk factor eligibility
test and the default risk charge are not covered. A regulatory result from
vetted is an implementation of the published text, not a supervisor's
decision. The conformal VaR guarantee assumes exchangeability (split
conformal) or holds on average over time (adaptive); neither is a guarantee
for any single day.

## References

* Acerbi, C. and Szekely, B. (2014). Backtesting expected shortfall. Risk.
* Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier. Journal of Risk 15(2).
* Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio. Journal of Portfolio Management 40(5).
* Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The probability of backtest overfitting. Journal of Computational Finance 20(4).
* Basel Committee on Banking Supervision. Basel Framework, MAR32: backtesting and P&L attribution test requirements.
* Christoffersen, P. (1998). Evaluating interval forecasts. International Economic Review 39(4).
* Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. JBES 13(3).
* Du, Z. and Escanciano, J. C. (2017). Backtesting expected shortfall: accounting for tail risk. Management Science 63(4).
* Engle, R. F. and Manganelli, S. (2004). CAViaR. JBES 22(4).
* Gibbs, I. and Candes, E. (2021). Adaptive conformal inference under distribution shift. NeurIPS.
* Harvey, C. R. and Liu, Y. (2015). Backtesting. Journal of Portfolio Management 42(1).
* Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. Journal of Derivatives 3(2).
* McNeil, A. J. and Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series. Journal of Empirical Finance 7.
* Nolde, N. and Ziegel, J. F. (2017). Elicitability and backtesting. Annals of Applied Statistics 11(4).
* Patton, A. J., Ziegel, J. F. and Chen, R. (2019). Dynamic semiparametric models for expected shortfall. Journal of Econometrics 211(2).

MIT licence.
