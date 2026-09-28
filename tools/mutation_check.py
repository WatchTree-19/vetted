"""Plant known bugs in vetted one at a time and check the test suite fails.

    python tools/mutation_check.py            # every mutation
    python tools/mutation_check.py report     # only mutations in files matching "report"

Each entry replaces one exact string in one source file. A mutation the
suite does not catch is a gap in the tests; the script exits non-zero if
any survive. The source files are restored after every run.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

MUTATIONS = [
    # var.py
    ("src/vetted/var.py", "return np.maximum(-2.0 * (ll0 - ll1), 0.0)", "return np.maximum(-2.0 * (ll0 - ll1), 0.0) * 1.01"),
    ("src/vetted/var.py", "n11 = np.sum(prev & curr, axis=-1).astype(float)", "n11 = np.sum(prev & curr, axis=-1).astype(float) + 1"),
    ("src/vetted/var.py", "cols.append(np.broadcast_to(unit(v[lags:]), lead + (n - lags,)))", "cols.append(np.broadcast_to(unit(v[lags - 1 : n - 1]), lead + (n - lags,)))"),
    ("src/vetted/var.py", "cols.append(np.broadcast_to(unit(v[lags:]), lead + (n - lags,)))", "cols.append(np.broadcast_to(v[lags:], lead + (n - lags,)))"),
    ("src/vetted/var.py", "df = lags + int(np.linalg.matrix_rank(fixed, tol=_RCOND * np.linalg.norm(fixed, 2)))", "df = int(X.shape[-1])"),
    ("src/vetted/var.py", "df = lags + int(np.linalg.matrix_rank(fixed, tol=_RCOND * np.linalg.norm(fixed, 2)))", "df = int(np.linalg.matrix_rank(X))"),
    ("src/vetted/var.py", "return float((1 + np.sum(simulated >= observed - 1e-12)) / (1 + simulated.size))", "return float(np.sum(simulated >= observed - 1e-12) / simulated.size) * 0.5"),
    ("src/vetted/var.py", "return rng.random((n_sims, n)) < alpha", "return rng.random((n_sims, n)) < alpha * 1.5"),
    ("src/vetted/var.py", "psq = np.where(np.isfinite(p), p, 0.0) ** 2", "psq = p ** 2"),
    ("src/vetted/var.py", "return (alpha - hit) * (p - q)", "return (alpha - hit) * (p - q) * 1.001"),
    # es.py
    ("src/vetted/es.py", "return float(np.sum(p * hit / (p.size * alpha * e)) + 1.0)", "return float(np.sum(p * hit / (p.size * alpha * e)) + 1.01)"),
    ("src/vetted/es.py", "scale_t = np.sqrt(250.0 / p.size)", "scale_t = 1.0"),
    ("src/vetted/es.py", "V = np.column_stack([alpha - hit, es_r - q + hit * (q - r) / alpha])", "V = np.column_stack([alpha - hit, es_r - q + hit * (q - r)])"),
    ("src/vetted/es.py", "flip2 = np.array([-1.0, 1.0])", "flip2 = np.array([1.0, 1.0])"),
    ("src/vetted/es.py", "x = r[hit] + e[hit]  # r - ES_r with ES_r = -es", "x = r[hit] + 0.95 * e[hit]"),
    ("src/vetted/es.py", "if x.size < max(int(min_exceptions), 3):", "if x.size < 2:"),
    ("src/vetted/es.py", "h = (alpha - u) * (u <= alpha) / alpha", "h = (alpha - u) * (u < alpha / 2) / alpha"),
    ("src/vetted/es.py", "np.sqrt(alpha * (1 / 3 - alpha / 4))", "np.sqrt(1.3 * alpha * (1 / 3 - alpha / 4))"),
    ("src/vetted/es.py", "c = n * np.sum(rho**2, axis=-1)", "c = 0.7 * n * np.sum(rho**2, axis=-1)"),
    ("src/vetted/es.py", "return -hit * (v - r) / (alpha * e) + v / e + np.log(-e) - 1.0", "return -hit * (v - r) / (alpha * e) + v / e + np.log(-e)"),
    # frtb.py
    ("src/vetted/frtb.py", "PLA_KS_GREEN = 0.09", "PLA_KS_GREEN = 0.10"),
    ("src/vetted/frtb.py", "5: 1.70, 6: 1.76", "5: 1.70, 6: 1.75"),
    ("src/vetted/frtb.py", "counts[name] = int(np.sum(p < -v))", "counts[name] = int(np.sum(p <= -v - 1e-9))"),
    ("src/vetted/frtb.py", 'fail = out["99"]["breach"] or out["97.5"]["breach"]', 'fail = out["99"]["breach"]'),
    ("src/vetted/frtb.py", "green_max = max(int(np.sum(cdf < green) - 1), 0)", "green_max = int(np.sum(cdf <= green))"),
    ("src/vetted/frtb.py", "green_max = max(int(np.sum(cdf < green) - 1), 0)", "green_max = int(np.sum(cdf < green) - 1)"),
    # compare.py
    ("src/vetted/compare.py", "k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)", "k = 1.0"),
    ("src/vetted/compare.py", "w = 1 - np.arange(1, h) / h", "w = 1 - np.arange(0, h - 1) / h"),
    # overfitting.py
    ("src/vetted/overfitting.py", "EULER_GAMMA = float(np.euler_gamma)", "EULER_GAMMA = 0.5"),
    ("src/vetted/overfitting.py", "return float(np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1)))", "return float(np.sqrt((1 - skew * sr + (kurt - 3) / 4 * sr**2) / (n - 1)))"),
    ("src/vetted/overfitting.py", "var_per_period = float(srs.var(ddof=1)) if n_trials > 1 else 0.0", "var_per_period = float(srs.var(ddof=0)) if n_trials > 1 else 0.0"),
    ("src/vetted/overfitting.py", "return float(np.sqrt(sharpe_variance) * max(z, 0.0))", "return float(np.sqrt(sharpe_variance) * z)"),
    ("src/vetted/overfitting.py", "star = np.nanargmax(R_is, axis=1)", "star = np.nanargmin(R_is, axis=1)"),
    ("src/vetted/overfitting.py", "adj = np.maximum.accumulate(np.minimum((m - np.arange(m)) * pv, 1.0))", "adj = np.minimum((m - np.arange(m)) * pv, 1.0)"),
    ("src/vetted/overfitting.py", "raw = pv * m * c / np.arange(1, m + 1)", "raw = pv * m / np.arange(1, m + 1)"),
    ("src/vetted/overfitting.py", "others = np.append(others, sharpe)", "pass"),
    ("src/vetted/overfitting.py", "return float(lam.sum() ** 2 / np.sum(lam**2))", "return float(lam.sum() / lam.max())"),
    # conformal.py
    ("src/vetted/conformal.py", "k = int(np.ceil((n + 1) * (1 - alpha)))", "k = int(np.ceil(n * (1 - alpha)))"),
    ("src/vetted/conformal.py", "a_t = a_t + gamma * (alpha - float(hit[t]))", "a_t = a_t"),
    ("src/vetted/conformal.py", "capped_hits += int(capped_now and hit[t])", "capped_hits += 0"),
    # _inputs.py
    ("src/vetted/_inputs.py", "p[bad] = -np.inf", "p[bad] = 0.0"),
    ("src/vetted/_inputs.py", 'check_same_index(pnl, *forecasts, names=("pnl",) + tuple(names))', "pass"),
    # report.py
    ("src/vetted/report.py", "running = max(running, min((m - rank) * ps[j], 1.0))", "running = min(m * ps[j], 1.0)"),
    ("src/vetted/report.py", "out.append(bool(a < self.significance))", "out.append(bool(a < 0.05))"),
    ("src/vetted/report.py", 'out.append(bool(zone in ("red", "ineligible")))', 'out.append(bool(zone in ("red",)))'),
    ("src/vetted/report.py", 'out.append(bool(zone in ("red", "ineligible")))', 'out.append(bool(zone in ("red", "ineligible", "amber")))'),
    ("src/vetted/report.py", "out.append(not bool(r.reject))", "out.append(bool(r.reject))"),
    ("src/vetted/report.py", "rep.results.append(_frtb.desk_backtest(var_99, var_975, pnl, actual_pnl))", "pass"),
    ("src/vetted/report.py", "rep.results.append(_frtb.pla_test(risk_theoretical_pnl, pnl))", "pass"),
    ("src/vetted/report.py", "_frtb.traffic_light(var_99, pnl, actual_pnl, alpha=0.01, window=250)", "_frtb.traffic_light(var_99, pnl, None, alpha=0.01, window=250)"),
    ("src/vetted/report.py", '"holm_p_value": None if a != a else a,', '"holm_p_value": r.p_value,'),
    ("src/vetted/report.py", "return bool(self.results) and not self.rejected", "return not self.rejected"),
    ("src/vetted/report.py", "rep.results.append(de)", "pass"),
    ("src/vetted/report.py", 'name="Du-Escanciano conditional",', 'name="Du-Escanciano unconditional",'),
    ("src/vetted/frtb.py", 'if missing == "drop":', 'if missing == "never":'),
    ("src/vetted/compare.py", 'check_same_index(loss_a, loss_b, names=("loss_a", "loss_b"))', "pass"),
]


def main(argv: list[str]) -> int:
    pick = argv[1] if len(argv) > 1 else ""
    todo = [m for m in MUTATIONS if pick in m[0]]
    survived = []
    for i, (rel, old, new) in enumerate(todo, 1):
        path = ROOT / rel
        src = path.read_text()
        if src.count(old) != 1:
            print(f"STALE   {rel}: target found {src.count(old)} times: {old[:60]}")
            survived.append((rel, old))
            continue
        path.write_text(src.replace(old, new))
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        finally:
            path.write_text(src)
        caught = r.returncode != 0
        print(f"{'caught ' if caught else 'SURVIVED'} [{i}/{len(todo)}] {rel}: {new[:70]}", flush=True)
        if not caught:
            survived.append((rel, new))
    print(f"\n{len(todo) - len(survived)} of {len(todo)} mutations caught")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
