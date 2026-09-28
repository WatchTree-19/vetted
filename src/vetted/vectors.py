"""Write src/vetted/data/expected.json: every (fixture, metric, convention) value.

    python -m vetted.vectors

Conventions that name an R PerformanceAnalytics call are, by the test suite,
identical to that call's output to 1e-10; the rest are the named formulas in
vetted/metrics.py. A library can vendor the slice it needs (one metric,
one convention) without depending on this package.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import CONVENTIONS

DATA = Path(__file__).resolve().parent / "data"


def build() -> dict:
    fixtures = json.loads((DATA / "fixtures.json").read_text())
    out: dict = {}
    for c in CONVENTIONS:
        m = out.setdefault(c.metric, {}).setdefault(c.name, {
            "description": c.description,
            "oracle": f"R PerformanceAnalytics: {c.oracle}" if c.oracle else None,
            "sign": c.signed,
            "values": {},
        })
        for f, d in fixtures.items():
            v = c.fn(d["returns"], d["periods_per_year"])
            m["values"][f] = float(v) if np.isfinite(v) else None
    return out


if __name__ == "__main__":
    (DATA / "expected.json").write_text(json.dumps(build(), indent=1))
    print("wrote src/vetted/data/expected.json")
