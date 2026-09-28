"""Helpers for a library's own test suite.

    from vetted.testing import cases

    @pytest.mark.parametrize("returns, periods, expected", cases("sortino_annual", "full_ddof0"))
    def test_sortino_matches_documented_convention(returns, periods, expected):
        assert my_sortino(returns, periods) == pytest.approx(expected, rel=1e-9)
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def cases(metric: str, convention: str, skip_undefined: bool = True):
    fixtures = json.loads((DATA / "fixtures.json").read_text())
    expected = json.loads((DATA / "expected.json").read_text())
    values = expected[metric][convention]["values"]
    out = []
    for name, d in fixtures.items():
        v = values[name]
        if v is None and skip_undefined:
            continue
        out.append((d["returns"], d["periods_per_year"], v))
    return out
