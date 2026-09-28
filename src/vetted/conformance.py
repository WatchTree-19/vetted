"""Ask each installed library for each metric and name the convention it uses.

    python -m vetted.conformance            # writes CONFORMANCE.md and conformance.json

A (library, metric) pair is:

* CONFORMANT to convention X if it matches X on every fixture where X is
  defined (up to the documented sign convention);
* INCONSISTENT if it matches different conventions on different fixtures;
* NO MATCH on the fixtures where it matches no published convention. That is
  the case worth a maintainer's attention.

Every library call is the documented public API with the documented
annualisation argument set explicitly.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .metrics import conventions_for

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent / "data"
RTOL = 1e-7


def _series(r, ann):
    r = np.asarray(r, dtype=float)
    if ann == 12:
        idx = pd.date_range("2000-01-31", periods=len(r), freq="ME")
    else:
        idx = pd.bdate_range("2000-01-03", periods=len(r))
    return pd.Series(r, index=idx)


def _prices(r, ann):
    """Price path with the starting capital (1.0) dated one period before the
    first return, which is what a price-based API expects."""
    s = _series(r, ann)
    base = s.index[0] - (pd.offsets.MonthEnd(1) if ann == 12 else pd.offsets.BDay(1))
    return pd.concat([pd.Series([1.0], index=[base]), (1 + s).cumprod()])


def _calendar_cagr(r, ann):
    """CAGR over calendar time, 365.25-day years, on the harness's own dates."""
    p = _prices(r, ann)
    years = (p.index[-1] - p.index[0]).total_seconds() / 31557600
    return float(p.iloc[-1] ** (1 / years) - 1)


def _calendar_calmar(r, ann):
    from .metrics import compute
    mdd = compute("max_drawdown", "geometric_initial_peak", r, ann)
    return _calendar_cagr(r, ann) / mdd if mdd > 0 else float("nan")


# Conventions that depend on dates rather than a periods-per-year count. They
# are defined on the harness's deterministic index.
CALENDAR = {
    "return_annual": {"geometric_calendar_365.25": _calendar_cagr},
    "calmar": {"cagr_calendar_over_mdd": _calendar_calmar},
}


def _scalar(v) -> float:
    if isinstance(v, (pd.Series, pd.DataFrame, np.ndarray)):
        v = np.asarray(v, dtype=float).ravel()[0]
    return float(v)


def build_adapters() -> dict[str, dict[str, Callable]]:
    ad: dict[str, dict[str, Callable]] = {}

    try:
        import quantstats as qs
        ad["quantstats " + qs.__version__] = {
            "volatility_annual": lambda r, a: qs.stats.volatility(_series(r, a), periods=a, annualize=True),
            "return_annual": lambda r, a: qs.stats.cagr(_series(r, a), periods=a),
            "sharpe_annual": lambda r, a: qs.stats.sharpe(_series(r, a), periods=a, annualize=True),
            "sortino_annual": lambda r, a: qs.stats.sortino(_series(r, a), periods=a, annualize=True),
            "max_drawdown": lambda r, a: qs.stats.max_drawdown(_series(r, a)),
            "calmar": lambda r, a: qs.stats.calmar(_series(r, a), periods=a),
            "var_95": lambda r, a: qs.stats.value_at_risk(_series(r, a), confidence=0.95),
            "es_95": lambda r, a: qs.stats.conditional_value_at_risk(_series(r, a), confidence=0.95),
            "omega_0": lambda r, a: qs.stats.omega(_series(r, a), periods=a),
            "ulcer_index": lambda r, a: qs.stats.ulcer_index(_series(r, a)),
            "skewness": lambda r, a: qs.stats.skew(_series(r, a)),
            "kurtosis": lambda r, a: qs.stats.kurtosis(_series(r, a)),
        }
    except ImportError:
        pass

    try:
        import empyrical as ep
        ver = getattr(ep, "__version__", "")
        ad["empyrical-reloaded " + ver] = {
            "volatility_annual": lambda r, a: ep.annual_volatility(np.asarray(r), annualization=a),
            "return_annual": lambda r, a: ep.annual_return(np.asarray(r), annualization=a),
            "sharpe_annual": lambda r, a: ep.sharpe_ratio(np.asarray(r), annualization=a),
            "sortino_annual": lambda r, a: ep.sortino_ratio(np.asarray(r), annualization=a),
            "max_drawdown": lambda r, a: ep.max_drawdown(np.asarray(r)),
            "calmar": lambda r, a: ep.calmar_ratio(np.asarray(r), annualization=a),
            "var_95": lambda r, a: ep.value_at_risk(np.asarray(r), cutoff=0.05),
            "es_95": lambda r, a: ep.conditional_value_at_risk(np.asarray(r), cutoff=0.05),
            "omega_0": lambda r, a: ep.omega_ratio(np.asarray(r), annualization=a),
        }
    except ImportError:
        pass

    try:
        import ffn
        ad["ffn " + ffn.__version__] = {
            "return_annual": lambda r, a: ffn.core.calc_cagr(_prices(r, a)),
            "sharpe_annual": lambda r, a: ffn.core.calc_sharpe(_series(r, a), rf=0.0, nperiods=a, annualize=True),
            "sortino_annual": lambda r, a: ffn.core.calc_sortino_ratio(_series(r, a), rf=0.0, nperiods=a, annualize=True),
            "max_drawdown": lambda r, a: ffn.core.calc_max_drawdown(_prices(r, a)),
            "calmar": lambda r, a: ffn.core.calc_calmar_ratio(_prices(r, a)),
        }
    except ImportError:
        pass

    try:
        import skfolio
        from skfolio import measures as skm
        ad["skfolio " + skfolio.__version__] = {
            "max_drawdown": lambda r, a: skm.max_drawdown(skm.get_drawdowns(np.asarray(r), compounded=True)),
            "var_95": lambda r, a: skm.value_at_risk(np.asarray(r), beta=0.95),
            "es_95": lambda r, a: skm.cvar(np.asarray(r), beta=0.95),
            "ulcer_index": lambda r, a: skm.ulcer_index(skm.get_drawdowns(np.asarray(r), compounded=True)),
            "skewness": lambda r, a: skm.skew(np.asarray(r)),
            "kurtosis": lambda r, a: skm.kurtosis(np.asarray(r)),
        }
    except ImportError:
        pass

    try:
        import riskfolio as rp
        from importlib.metadata import version
        rf = rp.RiskFunctions
        col = lambda r: np.asarray(r, dtype=float).reshape(-1, 1)
        ad["riskfolio-lib " + version("riskfolio-lib")] = {
            "max_drawdown": lambda r, a: rf.MDD_Rel(col(r)),
            "var_95": lambda r, a: rf.VaR_Hist(col(r), alpha=0.05),
            "es_95": lambda r, a: rf.CVaR_Hist(col(r), alpha=0.05),
            "ulcer_index": lambda r, a: rf.UCI_Rel(col(r)),
        }
    except ImportError:
        pass

    return ad


# Loss measures, where "loss as a negative return" and "loss as a positive
# number" are both common. A sign flip is accepted (and reported) only here;
# a negated Sharpe ratio or skewness is not a convention.
FLIPPABLE = {"var_95", "es_95", "max_drawdown", "ulcer_index", "pain_index"}


def _reference(metric: str, name: str, r, ann: int) -> float:
    for c in conventions_for(metric):
        if c.name == name:
            return c.fn(r, ann)
    return CALENDAR[metric][name](r, ann)


def classify(metric: str, value: float, r, ann: int):
    """-> (set of matching (convention, sign_flipped), nearest name, rel deviation).

    Several conventions can coincide on a given series (for example two tail
    estimators that happen to average the same observations), so matching is
    a set, and a library is judged on the intersection across fixtures.
    """
    cands = [(c.name, c.fn) for c in conventions_for(metric)]
    cands += list(CALENDAR.get(metric, {}).items())
    matches, nearest, best = set(), None, np.inf
    for name, fn in cands:
        ref = fn(r, ann)
        if not np.isfinite(ref):
            continue
        for flip in ((False, True) if metric in FLIPPABLE else (False,)):
            v = -value if flip else value
            dev = abs(v - ref) / max(abs(ref), 1e-12)
            if dev < best:
                nearest, best = name, dev
            if dev <= RTOL:
                # a zero value carries no sign information
                matches.add((name, None if value == 0 else flip))
    return matches, nearest, best


def run(fixtures: dict) -> list[dict]:
    rows = []
    for lib, metrics in build_adapters().items():
        for metric, fn in metrics.items():
            for fname, d in fixtures.items():
                r, ann = d["returns"], d["periods_per_year"]
                try:
                    value = _scalar(fn(r, ann))
                    err = None
                except Exception as e:  # a crash is a finding too
                    value, err = float("nan"), f"{type(e).__name__}: {e}"[:160]
                row = dict(library=lib, metric=metric, fixture=fname, value=value, error=err)
                if np.isfinite(value):
                    matches, nearest, dev = classify(metric, value, r, ann)
                    row.update(matches=sorted([list(m) for m in matches], key=str),
                               nearest=nearest, deviation=dev)
                rows.append(row)
    return rows


def summarise(rows: list[dict], fixtures: dict) -> list[dict]:
    by = defaultdict(list)
    for row in rows:
        by[(row["library"], row["metric"])].append(row)
    out = []
    for (lib, metric), rs in by.items():
        valued = [x for x in rs if np.isfinite(x["value"])]
        nomatch = [x for x in valued if not x["matches"]]
        names = None
        for x in valued:
            if x["matches"]:
                s = {m[0] for m in x["matches"]}
                names = s if names is None else names & s
        flips = {m[1] for x in valued for m in x["matches"] if m[1] is not None}
        # a failure only counts where the convention it otherwise follows has
        # a value: returning NaN where the estimator is undefined is correct
        failed = []
        for x in rs:
            if np.isfinite(x["value"]):
                continue
            d = fixtures[x["fixture"]]
            defined = [n for n in (names or []) if np.isfinite(_reference(metric, n, d["returns"], d["periods_per_year"]))]
            x["undefined_in_reference"] = not defined
            failed.append(x)
        real_failures = [x for x in failed if not x["undefined_in_reference"]]
        if nomatch:
            status = "NO MATCH"
        elif names is None:
            status = "no value"
        elif not names:
            status = "INCONSISTENT"
        elif len(flips) > 1:
            status = "INCONSISTENT SIGN"
        elif real_failures:
            status = "FAILS ON INPUT"
        else:
            status = "conformant"
        out.append(dict(
            library=lib, metric=metric, status=status, conventions=sorted(names or []),
            sign={frozenset([False]): "canonical", frozenset([True]): "flipped"}.get(frozenset(flips), "mixed" if flips else "-"),
            no_match=[dict(fixture=x["fixture"], value=x["value"], nearest=x["nearest"],
                           deviation=x["deviation"]) for x in nomatch],
            failed=[dict(fixture=x["fixture"], value=x["value"], error=x["error"],
                         undefined_in_reference=x["undefined_in_reference"]) for x in failed],
        ))
    return out


def to_markdown(summary: list[dict]) -> str:
    lines = [
        "# Conformance report",
        "",
        "Generated by `python -m vetted.conformance`. Each library is called through its",
        "public API on the fixtures in `src/vetted/data/fixtures.json`; its output is matched",
        f"against every named convention in `vetted/metrics.py` (relative tolerance {RTOL:g}).",
        "",
        "| library | metric | status | convention | sign | notes |",
        "|---|---|---|---|---|---|",
    ]
    for s in sorted(summary, key=lambda s: (s["metric"], s["library"])):
        notes = []
        for m in s["no_match"]:
            notes.append(f"{m['fixture']}: nearest {m['nearest']} off by {m['deviation']:.2%}")
        for f in s["failed"]:
            tag = " (undefined here)" if f["undefined_in_reference"] else ""
            notes.append(f"{f['fixture']}: {f['error'] or 'NaN'}{tag}")
        lines.append(
            f"| {s['library']} | {s['metric']} | {s['status']} | {', '.join(s['conventions']) or '-'} "
            f"| {s['sign']} | {'; '.join(notes)[:400]} |"
        )
    return "\n".join(lines) + "\n"


def main():
    fixtures = json.loads((DATA / "fixtures.json").read_text())
    rows = run(fixtures)
    summary = summarise(rows, fixtures)
    (Path.cwd() / "conformance.json").write_text(json.dumps(dict(rows=rows, summary=summary), indent=1, default=float))
    (Path.cwd() / "CONFORMANCE.md").write_text(to_markdown(summary))
    for s in summary:
        if s["status"] != "conformant":
            print(s["library"], s["metric"], s["status"], s["conventions"], len(s["no_match"]), len(s["failed"]))




# --------------------------------------------------------------------------
# Invariance checks: properties any estimator of the metric must have,
# whichever convention it follows.
# --------------------------------------------------------------------------

# Annualised over elapsed time: treating a missing period as time that passed
# is a defensible convention rather than a defect.
TIME_BASED = {"return_annual", "calmar"}

DISTRIBUTIONAL = {"volatility_annual", "sharpe_annual", "sortino_annual", "var_95",
                  "es_95", "omega_0", "skewness", "kurtosis"}


def _with_nans(r, k=12, seed=11):
    r = np.asarray(r, dtype=float)
    pos = np.random.default_rng(seed).integers(0, len(r), k)
    return np.insert(r, np.sort(pos), np.nan)


def invariance(fixtures: dict) -> list[dict]:
    """Missing observations should be dropped (or reported); a permutation of
    the returns must not change a distributional metric."""
    base = fixtures["gaussian_neg_mean"]
    r, ann = base["returns"], base["periods_per_year"]
    perm = np.random.default_rng(3).permutation(np.asarray(r))
    out = []
    for lib, metrics in build_adapters().items():
        for metric, fn in metrics.items():
            try:
                clean = _scalar(fn(r, ann))
            except Exception:
                continue
            try:
                nan_v = _scalar(fn(_with_nans(r), ann))
            except Exception as e:
                nan_v = f"error: {type(e).__name__}"
            if isinstance(nan_v, str):
                nan_result = nan_v
            elif not np.isfinite(nan_v):
                nan_result = "propagates NaN"
            elif np.isclose(nan_v, clean, rtol=1e-9):
                nan_result = "drops NaN (value unchanged)"
            elif metric in TIME_BASED:
                nan_result = f"counts missing periods as elapsed time ({abs(nan_v / clean - 1):.3%}; a convention)"
            else:
                nan_result = f"CHANGES VALUE by {abs(nan_v / clean - 1):.3%} (missing data used as data)"
            perm_result = None
            if metric in DISTRIBUTIONAL:
                pv = _scalar(fn(perm, ann))
                perm_result = "invariant" if np.isclose(pv, clean, rtol=1e-9) else f"ORDER-DEPENDENT ({abs(pv / clean - 1):.3%})"
            out.append(dict(library=lib, metric=metric, nan=nan_result, permutation=perm_result))
    return out


def invariance_markdown(rows: list[dict]) -> str:
    lines = ["", "## Invariance", "",
             "12 NaNs inserted into `gaussian_neg_mean` (n = 1512); and the same series permuted.",
             "", "| library | metric | missing data | permutation |", "|---|---|---|---|"]
    for x in sorted(rows, key=lambda x: (x["metric"], x["library"])):
        lines.append(f"| {x['library']} | {x['metric']} | {x['nan']} | {x['permutation'] or 'n/a'} |")
    return "\n".join(lines) + "\n"


def _main_with_invariance():
    fixtures = json.loads((DATA / "fixtures.json").read_text())
    rows = run(fixtures)
    summary = summarise(rows, fixtures)
    inv = invariance(fixtures)
    (Path.cwd() / "conformance.json").write_text(json.dumps(dict(rows=rows, summary=summary, invariance=inv), indent=1, default=float))
    (Path.cwd() / "CONFORMANCE.md").write_text(to_markdown(summary) + invariance_markdown(inv))
    for s in summary:
        if s["status"] != "conformant":
            print(s["library"], s["metric"], s["status"])
    for x in inv:
        if "CHANGES" in x["nan"] or (x["permutation"] or "").startswith("ORDER") or x["nan"].startswith("error"):
            print(x)


if __name__ == "__main__":
    _main_with_invariance()
