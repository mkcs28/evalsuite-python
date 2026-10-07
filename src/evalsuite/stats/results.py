"""Result types for intervals and statistical tests."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, cast

from ..core.result import _json_safe

__all__ = ["ConfidenceInterval", "TestResult"]


def _f(x: float, digits: int = 4) -> str:
    return "NaN" if math.isnan(x) else f"{x:.{digits}f}"


@dataclass(frozen=True, eq=False)
class ConfidenceInterval:
    """A point estimate with a confidence interval.

    ``method`` names the procedure (for example ``"bootstrap-bca"``, ``"wilson"``, ``"delong"``) and
    ``params`` records everything needed to reproduce it (resamples, random state, failed resamples...).
    """

    estimate: float
    low: float
    high: float
    level: float
    method: str
    metric: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("estimate", "low", "high", "level"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __float__(self) -> float:
        return self.estimate

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Unpack as ``low, high``."""
        return iter((self.low, self.high))

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def __repr__(self) -> str:
        pct = round(self.level * 100, 6)
        label = f"{self.metric} " if self.metric else ""
        return f"{label}{_f(self.estimate)} [{pct:g}% CI {_f(self.low)}, {_f(self.high)}] ({self.method})"

    def format(self, digits: int = 3) -> str:
        """``0.812 (95% CI 0.774–0.846)``: the form most journals expect."""
        pct = round(self.level * 100, 6)
        return f"{self.estimate:.{digits}f} ({pct:g}% CI {self.low:.{digits}f}–{self.high:.{digits}f})"

    def to_dict(self) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            _json_safe(
                {
                    "metric": self.metric,
                    "estimate": self.estimate,
                    "low": self.low,
                    "high": self.high,
                    "level": self.level,
                    "method": self.method,
                    "params": dict(self.params),
                }
            ),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


@dataclass(frozen=True, eq=False)
class TestResult:
    """Outcome of a statistical test: statistic, p-value and what was tested."""

    __test__ = False  # not a pytest test class

    test: str
    statistic: float
    p_value: float
    alternative: str = "two-sided"
    estimate: Optional[float] = None
    ci: Optional[ConfidenceInterval] = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "statistic", float(self.statistic))
        object.__setattr__(self, "p_value", float(self.p_value))
        if self.estimate is not None:
            object.__setattr__(self, "estimate", float(self.estimate))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def __repr__(self) -> str:
        est = f", estimate={_f(self.estimate)}" if self.estimate is not None else ""
        p = "p<0.0001" if self.p_value < 1e-4 else f"p={self.p_value:.4f}"
        return f"TestResult({self.test}: statistic={_f(self.statistic)}, {p}{est})"

    def to_dict(self) -> dict[str, Any]:
        out = {
            "test": self.test,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "alternative": self.alternative,
            "estimate": self.estimate,
            "params": dict(self.params),
        }
        if self.ci is not None:
            out["ci"] = self.ci.to_dict()
        return cast("dict[str, Any]", _json_safe(out))

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)
