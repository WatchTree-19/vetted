"""The result object every test in vetted returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TestResult:
    """Outcome of a statistical test.

    Attributes:
        name: Short name of the test.
        statistic: The test statistic.
        p_value: p-value under the null hypothesis (``nan`` when the test has
            no p-value, for example a regulatory zone).
        reject: Whether the null is rejected at ``significance``.
        significance: Level the decision was taken at.
        null: The null hypothesis, in words.
        reference: Where the test is defined.
        detail: Anything else the test computed (counts, components, zones).
    """

    name: str
    statistic: float
    p_value: float
    reject: bool
    significance: float
    null: str
    reference: str
    detail: dict[str, Any] = field(default_factory=dict)

    __test__ = False  # not a pytest test class

    def __getitem__(self, key: str) -> Any:
        return self.detail[key]

    def summary(self) -> str:
        verdict = "reject" if self.reject else "do not reject"
        p = "n/a" if self.p_value != self.p_value else f"{self.p_value:.4g}"
        return (
            f"{self.name}: statistic {self.statistic:.4g}, p-value {p}, "
            f"{verdict} at {self.significance:g} ({self.null})"
        )

    def __str__(self) -> str:
        return self.summary()
