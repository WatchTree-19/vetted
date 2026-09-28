# vetted

**Independent validation for risk models and trading strategies.**

vetted answers the two questions every quant result has to survive:

* **Is this risk model right?** VaR and Expected Shortfall backtests, the
  Basel FRTB traffic light, desk eligibility and P&L attribution tests, and a
  one-call validation report for a model risk file.
* **Is this backtest real?** The deflated Sharpe ratio, the probability of
  backtest overfitting, the minimum track record length and multiple-testing
  haircuts, for results that were the best of many tries.

Every statistic is checked against an R package, a number printed in the
paper or regulation that defines it, or a hand calculation, and the test that
checks it says which. See [How it is verified](verification.md).

## Install

```bash
pip install vetted
```

It needs only numpy and scipy.

## Validate a risk model in one call

```python
import vetted

report = vetted.validate_risk_model(
    pnl,                  # daily hypothetical P&L, gains positive
    var_99,               # 99% one-day VaR per day, as a positive loss
    var_975, es_975,      # 97.5% VaR and ES (the FRTB measure)
    actual_pnl=apl,       # optional: MAR32.5 counts the greater of APL and HPL
    risk_theoretical_pnl=rtpl,  # optional: runs the PLA test
)
print(report)             # a table with a pass, amber or fail verdict
report.to_markdown()      # for the validation file
report.to_dict()          # for your own reporting
```

More in [Risk models](risk-models.md) and [Basel FRTB](frtb.md).

## Validate a backtested strategy

```python
report = vetted.validate_strategy(best_returns, trials=all_trial_returns)
```

`trials` is every strategy that was tried, one column each. More in
[Strategies and overfitting](strategies.md).

## Conventions used everywhere

* P&L and returns are positive for a gain.
* VaR and ES are positive loss amounts, as risk reports and the Basel text
  state them: a VaR of 2.1 means a loss of 2.1.
* An exception is a loss larger than the VaR (`pnl < -var`, MAR32.5).
* `alpha` is the tail probability: 0.01 for a 99% VaR, 0.025 for the FRTB
  97.5% ES.
* Every test returns a `TestResult` with the statistic, p-value, decision,
  null hypothesis, the reference that defines the test, and a `detail`
  dictionary with everything else it computed.
* pandas Series are accepted; when two inputs carry an index, the indexes
  must match, so a forecast is never paired with the wrong day by position.
