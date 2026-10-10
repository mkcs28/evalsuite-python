"""Shared input handling for the v0.5.0 LLM-systems metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.result import MetricResult

_Z95 = 1.959963984540054
Sized = Any  # anything with __len__


def seq(x: Any, name: str) -> list[Any]:
    """A non-empty list, one entry per example."""
    if x is None:
        raise InputValidationError(f"{name} is required but was None.")
    if isinstance(x, (str, bytes)):
        raise InputValidationError(f"{name} must be a list (one entry per example), not a single string.")
    if isinstance(x, Mapping):
        raise InputValidationError(f"{name} must be a list, not a mapping.")
    items = x.tolist() if isinstance(x, np.ndarray) else list(x)
    if not items:
        raise InputValidationError(f"{name} is empty. Provide at least one example.")
    return items


def bools(x: Any, name: str) -> np.ndarray:
    """A 1-D boolean array from booleans or 0/1 values (anything else is rejected, never coerced)."""
    items = seq(x, name)
    out = np.empty(len(items), dtype=bool)
    for i, v in enumerate(items):
        if isinstance(v, (bool, np.bool_)) or (
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and v in (0, 1)
        ):
            out[i] = bool(v)
        else:
            raise InputValidationError(f"{name}[{i}] must be a boolean (or 0/1); got {v!r}.")
    return out


def floats(x: Any, name: str, *, lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    """A finite 1-D float array, optionally bounded."""
    items = seq(x, name)
    try:
        a = np.asarray(items, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name} must contain numbers.") from exc
    if a.ndim != 1:
        raise InputValidationError(f"{name} must be one-dimensional (one number per example).")
    if not np.all(np.isfinite(a)):
        raise InputValidationError(f"{name} contains NaN or infinite values.")
    if lo is not None and np.any(a < lo):
        raise InputValidationError(f"{name} must be >= {lo}.")
    if hi is not None and np.any(a > hi):
        raise InputValidationError(f"{name} must be <= {hi}.")
    return a


def same_length(*pairs: tuple[str, Sized]) -> int:
    n = len(pairs[0][1])
    for name, v in pairs[1:]:
        if len(v) != n:
            raise InputValidationError(
                f"{pairs[0][0]} and {name} must contain the same number of examples. Received {n} and {len(v)}."
            )
    return n


def wilson(k: float, n: float) -> tuple[float, float]:
    """95% Wilson interval for a proportion (Wilson 1927)."""
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    z2 = _Z95**2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(metric: str, name: str, flags: np.ndarray, extra: Optional[dict[str, Any]] = None) -> MetricResult:
    """A proportion with its count and 95% Wilson interval in ``params``."""
    n = int(flags.size)
    k = int(flags.sum())
    lo, hi = wilson(k, n)
    params = {"count": k, "n": n, "ci_low": lo, "ci_high": hi, **(extra or {})}
    return MetricResult(metric, name, k / n, params)


def by_group(values: np.ndarray, groups: Optional[Sequence[Any]], name: str = "groups") -> dict[str, float]:
    """Mean of ``values`` per group label (keys as strings, sorted)."""
    if groups is None:
        return {}
    g = seq(groups, name)
    if len(g) != values.size:
        raise InputValidationError(f"{name} must have one label per example ({values.size}); got {len(g)}.")
    out: dict[str, float] = {}
    for label in sorted({str(v) for v in g}):
        mask = np.array([str(v) == label for v in g])
        out[label] = float(values[mask].mean())
    return out


def summary(a: np.ndarray) -> dict[str, float]:
    """Mean, standard deviation, min, max and the p50/p90/p95/p99 percentiles (linear interpolation)."""
    p50, p90, p95, p99 = np.percentile(a, [50, 90, 95, 99])
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
    }


def lists_of(x: Any, name: str) -> list[list[Any]]:
    """One (possibly empty) list per example."""
    rows = seq(x, name)
    out = []
    for i, r in enumerate(rows):
        if isinstance(r, (str, bytes)) or not hasattr(r, "__iter__"):
            raise InputValidationError(f"{name}[{i}] must be a list.")
        out.append(list(r))
    return out
