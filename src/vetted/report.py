"""One-call validation reports.

``validate_risk_model`` runs every applicable VaR, ES and FRTB test on a
model's forecasts and realised P&L; ``validate_strategy`` runs every
overfitting check on a backtested strategy. Both return a ``Report`` that
prints as a table, exports to Markdown for a validation file, and keeps each
underlying ``TestResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from . import es as _es
from . import frtb as _frtb
from . import overfitting as _of
from . import var as _var
from ._result import TestResult


@dataclass
class Report:
    """A set of test results with one overall verdict.

    Running ten tests at 5% each would flag a correct model about 40% of the
    time, so the verdict does not simply ask whether any test rejected. The
    p-values of the statistical tests are adjusted together with Holm's
    method (``holm_p``), and a test counts against the model only if its
    adjusted p-value is below ``significance``. Tests that report a zone are
    outside that family: red (or an ineligible desk) counts against the
    model, amber is flagged but does not fail it. Tests without a p-value or
    zone (PBO) count whenever they reject.

    For a strategy, passing means showing skill: a test named in
    ``pass_requires_rejection`` (the deflated Sharpe ratio) counts against
    the strategy when it does *not* reject its null.
    """

    title: str
    results: list[TestResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    significance: float = 0.05
    multiple_testing: bool = True
    pass_requires_rejection: tuple = ()

    def _family(self) -> list[int]:
        # Regulatory zones are rules, not members of the statistical family.
        return [
            i
            for i, r in enumerate(self.results)
            if r.p_value is not None and r.p_value == r.p_value and "zone" not in r.detail
        ]

    def holm_p(self) -> list[float]:
        """Holm-adjusted p-value for each result, by position (``nan`` for
        results outside the family)."""
        out = [float("nan")] * len(self.results)
        fam = self._family() if self.multiple_testing else []
        if not fam:
            return out
        ps = np.array([float(self.results[i].p_value) for i in fam])
        order = np.argsort(ps, kind="stable")
        m = len(fam)
        running = 0.0
        for rank, j in enumerate(order):
            running = max(running, min((m - rank) * ps[j], 1.0))
            out[fam[j]] = running
        return out

    @property
    def adjusted_p(self) -> dict[str, float]:
        """Holm-adjusted p-values by test name (for display)."""
        return {r.name: a for r, a in zip(self.results, self.holm_p()) if a == a}

    def _against(self) -> list[bool]:
        adj = self.holm_p()
        out = []
        for r, a in zip(self.results, adj):
            zone = r.detail.get("zone")
            if r.name in self.pass_requires_rejection:
                out.append(not bool(r.reject))
            elif a == a:
                out.append(bool(a < self.significance))
            elif zone is not None:
                out.append(bool(zone in ("red", "ineligible")))
            else:
                out.append(bool(r.reject))
        return out

    def counts_against(self, r: TestResult) -> bool:
        for x, flag in zip(self.results, self._against()):
            if x is r:
                return flag
        raise ValueError("result is not part of this report")

    @property
    def rejected(self) -> list[TestResult]:
        """Tests that count against the model (see the class docstring)."""
        return [r for r, f in zip(self.results, self._against()) if f]

    @property
    def flagged(self) -> list[TestResult]:
        """Amber zones. MAR32.11: amber outcomes "could result from either
        accurate or inaccurate models"; a correct 99% VaR lands in the amber
        zone about 11% of the time over 250 days. They are reported, not
        counted as failures."""
        return [r for r in self.results if r.detail.get("zone") == "amber"]

    @property
    def passed(self) -> bool:
        """True when at least one test ran and nothing counts against the
        model. Amber zones do not fail a model; check ``flagged``."""
        return bool(self.results) and not self.rejected

    @property
    def status(self) -> str:
        if not self.results:
            return "no tests"
        if self.rejected:
            return "fail"
        return "amber" if self.flagged else "pass"

    def __getitem__(self, name: str) -> TestResult:
        for r in self.results:
            if r.name == name:
                return r
        raise KeyError(name)

    def _outcome(self, r: TestResult, against: bool) -> str:
        zone = r.detail.get("zone")
        return zone if zone else ("fail" if against else "pass")

    def _verdict(self) -> str:
        if not self.results:
            return "No tests ran."
        if self.rejected:
            return "Failed: " + ", ".join(r.name for r in self.rejected) + "."
        if self.flagged:
            return "Passed, with amber flags: " + ", ".join(r.name for r in self.flagged) + "."
        return "Passed: no test counts against the model."

    def to_dict(self) -> dict:
        adj, against = self.holm_p(), self._against()
        return {
            "title": self.title,
            "passed": self.passed,
            "status": self.status,
            "facts": self.facts,
            "notes": self.notes,
            "results": [
                {
                    "name": r.name,
                    "statistic": r.statistic,
                    "p_value": r.p_value,
                    "holm_p_value": None if a != a else a,
                    "counts_against": f,
                    "rejects_null": r.reject,
                    "significance": r.significance,
                    "null": r.null,
                    "reference": r.reference,
                    "zone": r.detail.get("zone"),
                }
                for r, a, f in zip(self.results, adj, against)
            ],
        }

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for k, v in self.facts.items():
            lines.append(f"- **{k}**: {_fmt(v)}")
        lines += [
            "",
            "| test | statistic | p-value | Holm p-value | result | null hypothesis |",
            "|---|---|---|---|---|---|",
        ]
        for r, a, f in zip(self.results, self.holm_p(), self._against()):
            lines.append(
                f"| {r.name} | {_fmt(r.statistic)} | {_fmt(r.p_value)} | {_fmt(a)} "
                f"| {self._outcome(r, f)} | {r.null} |"
            )
        lines += ["", f"**{self._verdict()}**", ""]
        for n in self.notes:
            lines.append(f"- {n}")
        lines += ["", "References:", ""]
        for ref in dict.fromkeys(r.reference for r in self.results):
            lines.append(f"- {ref}")
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        width = max((len(r.name) for r in self.results), default=10)
        out = [self.title, "=" * len(self.title)]
        for k, v in self.facts.items():
            out.append(f"{k}: {_fmt(v)}")
        out.append("")
        out.append(f"{'test':<{width}}  {'statistic':>10}  {'p':>8}  {'Holm p':>8}  result")
        for r, a, f in zip(self.results, self.holm_p(), self._against()):
            out.append(
                f"{r.name:<{width}}  {_fmt(r.statistic):>10}  {_fmt(r.p_value):>8}  "
                f"{_fmt(a):>8}  {self._outcome(r, f).upper()}"
            )
        out.append("")
        if not self.results:
            out.append("NO TESTS RAN")
        elif self.rejected:
            out.append("FAILED: " + ", ".join(r.name for r in self.rejected))
        elif self.flagged:
            out.append("PASSED, with amber flags: " + ", ".join(r.name for r in self.flagged))
        else:
            out.append("PASSED")
        out += [f"note: {n}" for n in self.notes]
        return "\n".join(out)

    _repr_markdown_ = to_markdown


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return "n/a"
        return f"{v:.4g}"
    return str(v)


def validate_risk_model(
    pnl,
    var_99,
    var_975=None,
    es_975=None,
    actual_pnl=None,
    risk_theoretical_pnl=None,
    pit=None,
    scale=None,
    significance: float = 0.05,
    missing: str = "raise",
    title: str = "Risk model validation",
    n_sims: int = 2_000,
    seed: int = 0,
) -> Report:
    """Run every applicable backtest on a VaR / ES model.

    Args:
        pnl: Daily hypothetical P&L (gain positive).
        var_99: 99% one-day VaR forecast for each day (positive loss amount).
        var_975: 97.5% VaR, needed for the desk backtest and the ES tests.
        es_975: 97.5% Expected Shortfall forecast (positive loss amount).
        actual_pnl: Actual P&L, if the traffic light should count against it
            too (MAR32.5 takes the greater count).
        risk_theoretical_pnl: The risk model's own P&L, for the PLA test.
        pit: The model's forecast CDF at the realised P&L, for the
            Du-Escanciano unconditional and conditional tests.
        scale: A volatility forecast, for the standardised exceedance
            residual test.
        significance: Level for the statistical tests.
        missing: How statistical tests treat gaps ("raise" or "drop"). The
            regulatory counts always treat a gap as an exception (MAR32.5).
        n_sims: Monte Carlo draws for the VaR tests' p-values and the
            bootstrap of the exceedance residual test.
        seed: Random seed for those draws.
    """
    rep = Report(title=title, significance=significance)
    n = int(np.asarray(pnl, dtype=float).size)
    rep.facts["observations"] = n

    rep.results.append(_frtb.traffic_light(var_99, pnl, actual_pnl, alpha=0.01, window=250))
    # Monte Carlo p-values throughout: with 2.5 exceptions expected a year at
    # 99%, the chi-squared limits are unreliable (see vetted.var).
    mc = dict(significance=significance, missing=missing, pvalue="monte_carlo", n_sims=n_sims, seed=seed)
    levels = [(var_99, 0.01, "99%")]
    if var_975 is not None:
        levels.append((var_975, 0.025, "97.5%"))
    for v, a, label in levels:
        rep.results.append(_rename(_var.kupiec(pnl, v, a, **mc), label))
        rep.results.append(_rename(_var.christoffersen(pnl, v, a, **mc), label))
        rep.results.append(_rename(_var.dynamic_quantile(pnl, v, a, **mc), label))
    if var_975 is not None:
        rep.results.append(_frtb.desk_backtest(var_99, var_975, pnl, actual_pnl))

    if es_975 is not None:
        if var_975 is None:
            raise ValueError("es_975 needs var_975: the ES tests condition on the 97.5% VaR")
        rep.results.append(_es.acerbi_szekely(pnl, var_975, es_975, 0.025, significance=significance, missing=missing))
        # Nolde-Ziegel: the paper's one-sided test (risk underestimation).
        # Its two-sided Wald form is oversized on short samples and is left
        # out; see vetted.es.conditional_calibration.
        nz = _es.conditional_calibration(pnl, var_975, es_975, 0.025, significance=significance, missing=missing)
        p1 = nz.detail["p_simple_one_sided"]
        rep.results.append(
            replace(
                nz,
                name="Nolde-Ziegel conditional calibration (one-sided)",
                statistic=float(nz.detail["t_simple"]),
                p_value=p1,
                reject=p1 < significance,
                null="VaR and ES do not underestimate risk",
            )
        )
        try:
            rep.results.append(
                _es.exceedance_residuals(
                    pnl, var_975, es_975, scale=scale, significance=significance, missing=missing, n_boot=n_sims, seed=seed
                )
            )
        except ValueError as exc:
            rep.notes.append(f"McNeil-Frey test skipped: {exc}")

    if pit is not None:
        de = _es.du_escanciano(pit, 0.025, significance=significance, pvalue="monte_carlo", n_sims=n_sims, seed=seed)
        rep.results.append(de)
        pc = de.detail["p_conditional"]
        rep.results.append(
            replace(
                de,
                name="Du-Escanciano conditional",
                statistic=float(de.detail["conditional_statistic"]),
                p_value=pc,
                reject=pc < significance,
                null="tail losses do not cluster and are calibrated",
            )
        )

    if risk_theoretical_pnl is not None:
        rep.results.append(_frtb.pla_test(risk_theoretical_pnl, pnl))

    if n < 250:
        rep.notes.append(f"{n} observations; the Basel tests are defined on 250 days")
    elif n > 250:
        rep.notes.append(
            "The Basel traffic light and desk backtest count the most recent 250 days (MAR32.3, MAR32.19); "
            f"the statistical tests use all {n}."
        )
    return rep


def _rename(r: TestResult, suffix: str) -> TestResult:
    return replace(r, name=f"{r.name} ({suffix})")


def validate_strategy(
    returns,
    trials=None,
    sharpe_variance: Optional[float] = None,
    periods: int = 252,
    significance: float = 0.05,
    n_blocks: int = 16,
    title: str = "Strategy validation",
) -> Report:
    """Run every overfitting check on a backtested strategy.

    Args:
        returns: Returns of the selected strategy.
        trials: Every strategy that was tried, as a (T, N) array or DataFrame
            of returns (including the selected one), or the number tried.
            Without it, only the probabilistic Sharpe ratio runs, which does
            not correct for selection.
        sharpe_variance: When ``trials`` is a number, the variance of the
            tried strategies' annualised Sharpe ratios (required then).
        periods: Periods per year.

    The strategy passes when the deflated Sharpe ratio is significant and
    PBO is at most 0.5.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    rep = Report(
        title=title,
        significance=significance,
        multiple_testing=False,
        pass_requires_rejection=("Deflated Sharpe ratio", "Probabilistic Sharpe ratio"),
    )
    sr = float(r.mean() / r.std(ddof=1) * np.sqrt(periods))
    rep.facts["observations"] = int(r.size)
    rep.facts["annualised Sharpe ratio"] = sr
    psr = _of.probabilistic_sharpe(r, periods=periods)
    rep.facts["probabilistic Sharpe ratio (vs 0)"] = psr
    rep.facts["minimum track record (observations)"] = _of.min_track_record(r, periods=periods)

    if trials is None:
        rep.notes.append("No trials given: the Sharpe ratio is not deflated for selection.")
        rep.results.append(
            TestResult(
                name="Probabilistic Sharpe ratio",
                statistic=psr,
                p_value=1 - psr,
                reject=psr > 1 - significance,
                significance=significance,
                null="true Sharpe ratio <= 0",
                reference="Bailey and Lopez de Prado (2012), Journal of Risk 15(2)",
            )
        )
        return rep

    if np.ndim(trials) == 0:
        if sharpe_variance is None:
            raise ValueError(
                "with trials given as a count, pass sharpe_variance (the variance of the tried "
                "strategies' annualised Sharpe ratios), or pass every trial's returns"
            )
        dsr = _of.deflated_sharpe(r, float(trials), sharpe_variance=sharpe_variance, periods=periods,
                                  significance=significance)
        rep.facts["trials"] = trials
        rep.facts["hurdle Sharpe ratio (best of noise)"] = dsr.detail["hurdle_sharpe"]
        rep.results.append(dsr)
        rep.notes.append("Trials given as a count: pass every trial's returns to also run PBO.")
        return rep

    if sharpe_variance is not None:
        raise ValueError("pass sharpe_variance only with a trial count; with trial returns it is measured")
    M = np.asarray(trials, dtype=float)
    rep.facts["trials"] = int(M.shape[1])
    try:
        rep.facts["effective independent bets (descriptive)"] = _of.effective_trials(M)
    except ValueError:
        pass
    dsr = _of.deflated_sharpe(r, M, periods=periods, significance=significance)
    rep.facts["hurdle Sharpe ratio (best of noise)"] = dsr.detail["hurdle_sharpe"]
    rep.results.append(dsr)
    try:
        rep.results.append(_of.pbo(M, n_blocks=n_blocks))
    except ValueError as exc:
        rep.notes.append(f"PBO skipped: {exc}")
    srs = _of._trial_sharpes(M) * np.sqrt(periods)
    hc = _of.haircut_sharpe(sr, r.size, srs, method="holm", periods=periods)
    rep.facts["haircut Sharpe ratio (Holm)"] = hc["haircut_sharpe"]
    return rep
