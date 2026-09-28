# Strategies and overfitting (`vetted.overfitting`)

A Sharpe ratio found after testing one idea and the same Sharpe ratio found
after testing a thousand are different evidence. Grid searches, genetic
algorithms and AI research agents all inflate the number of tries.

## The one-call report

```python
report = vetted.validate_strategy(best_returns, trials=all_trial_returns)
```

* With a (T, N) matrix of every strategy tried: deflated Sharpe ratio, PBO,
  and a Holm haircut Sharpe ratio.
* With a count instead: pass `sharpe_variance` (the variance of the tried
  strategies' annualised Sharpe ratios) and the deflated Sharpe ratio runs.
* With no trials: the probabilistic Sharpe ratio only, which does not correct
  for selection (the report says so).

The strategy passes when the deflated Sharpe ratio is significant and PBO is
at most 0.5.

## Functions

| function | what it gives | checked against |
|---|---|---|
| `probabilistic_sharpe` | P(true Sharpe > benchmark), allowing for skew and fat tails | R PerformanceAnalytics `ProbSharpeRatio` |
| `min_track_record` | observations needed for significance | R PerformanceAnalytics `MinTrackRecord` |
| `expected_max_sharpe` | the Sharpe ratio the best of N skill-less strategies shows | Bailey and Lopez de Prado (2014) worked example |
| `deflated_sharpe` | PSR against that hurdle | the same worked example (DSR 0.9004) |
| `pbo` | probability of backtest overfitting by CSCV | R pbo (with `rank_denominator="n"`) |
| `haircut_sharpe` | Sharpe ratio after Bonferroni, Holm or BHY | statsmodels `multipletests` |
| `effective_trials` | participation ratio of the trials' correlation spectrum | descriptive |

## Correlated trials

Measuring the Sharpe variance on the trials themselves already reflects
their correlation: correlated strategies have similar Sharpe ratios, so the
hurdle is lower. Do not also shrink the count to an effective number of
trials. On the best of 50 null strategies with pairwise correlation 0.7,
deflating by all trials gave 2% false positives at a nominal 5%; also
shrinking the count gave 19%. `effective_trials` is reported as a
descriptive figure only.

## Example

`examples/strategy_search.py` takes the best of 200 skill-less strategies:
a Sharpe ratio of 1.05, 99% significant on its own. The deflated Sharpe
ratio is 0.42 against a hurdle of 1.14, and PBO is 0.63: it fails. When one
of the 200 ideas is real, the same report passes it.
