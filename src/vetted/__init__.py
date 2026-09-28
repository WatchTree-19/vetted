"""vetted: independent validation for risk models and trading strategies.

Every statistic in this package is checked against an independent
implementation (R PerformanceAnalytics, rugarch, esback, GAS, pbo,
forecast, statsmodels) or a published worked example, and the test that
does the checking records which one.

Modules:
    vetted.var          VaR backtests (Kupiec, Christoffersen, DQ)
    vetted.es           Expected Shortfall backtests (Acerbi-Szekely,
                        Nolde-Ziegel, McNeil-Frey, Du-Escanciano) and FZ0 loss
    vetted.frtb         Basel MAR32 traffic light, desk backtest, PLA test
    vetted.compare      Diebold-Mariano comparison of two models
    vetted.conformal    VaR with a finite-sample coverage guarantee
    vetted.overfitting  Deflated Sharpe, PBO, minimum track record, haircuts
    vetted.metrics      Performance and risk metrics under named conventions
    vetted.report       One-call validation reports
"""

from . import compare, conformal, es, frtb, metrics, overfitting, report, var
from ._result import TestResult
from .report import validate_risk_model, validate_strategy

__version__ = "0.1.0"

__all__ = [
    "TestResult",
    "compare",
    "conformal",
    "es",
    "frtb",
    "metrics",
    "overfitting",
    "report",
    "validate_risk_model",
    "validate_strategy",
    "var",
]
