# Metrics and conformance

Most performance metrics have several published estimators. A library that
disagrees with one formula is not wrong; one whose output matches no
published estimator, or one its documentation does not state, is.

## Named conventions (`vetted.metrics`)

40 conventions for 15 metrics: annual volatility and return, Sharpe, Sortino
and downside deviation, maximum drawdown, Calmar, 95% VaR and ES, Omega,
Ulcer and Pain indices, skewness and kurtosis.

```python
from vetted import metrics
metrics.metrics()                                  # metric names
[c.name for c in metrics.conventions_for("var_95")]
metrics.compute("sharpe_annual", "arithmetic_ddof1", returns, 252)
```

The 24 conventions R PerformanceAnalytics implements reproduce it to 1e-10;
the rest are checked against numpy, scipy or a brute-force definition.

## Pinning a library to the reference values

```python
import pytest
from vetted.testing import cases

@pytest.mark.parametrize("returns, periods, expected", cases("sortino_annual", "full_ddof0"))
def test_sortino(returns, periods, expected):
    assert my_sortino(returns, periods) == pytest.approx(expected, rel=1e-9)
```

Or copy the relevant block of `src/vetted/data/expected.json` into your own
test data, with no dependency on vetted. quantstats, ffn and skfolio test
suites built this way are in review upstream.

## Conformance report

`python -m vetted.conformance` calls each installed library through its
public API, names the convention it matches, and checks that missing
observations are not treated as data and that distributional metrics do not
depend on the order of the returns. The current report is in
[CONFORMANCE.md](https://github.com/WatchTree-19/vetted/blob/main/CONFORMANCE.md).
