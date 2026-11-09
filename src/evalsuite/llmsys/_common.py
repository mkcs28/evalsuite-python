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
    if isinstance(x, np.ndarray) and x.ndim == 1 and x.size and x.dtype.kind in "biuf":
        if x.dtype.kind == "b":
            return x.astype(bool, copy=False)
        bad = ~np.isin(x, (0, 1))
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise InputValidationError(f"{name}[{i}] must be a boolean (or 0/1); got {x[i]!r}.")
        return x.astype(bool)
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
    items: Any = x if isinstance(x, np.ndarray) and x.size and x.dtype.kind in "biuf" else seq(x, name)
    try:
        a = np.asarray(items, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name} must contain numbers.") from exc
    if a.ndim != 1:
        raise InputValidationError(f"{name} must be one-dimensional (one number per example).")
    if a.size:
        # min and max propagate NaN and ±inf, so two reductions check finiteness and both bounds.
        mn, mx = float(a.min()), float(a.max())
        if not (math.isfinite(mn) and math.isfinite(mx)):
            raise InputValidationError(f"{name} contains NaN or infinite values.")
        if lo is not None and mn < lo:
            raise InputValidationError(f"{name} must be >= {lo}.")
        if hi is not None and mx > hi:
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


def _order_stats_partition(a: np.ndarray, ks: list[int]) -> dict[int, float]:
    part = np.partition(a, ks)
    return {k: float(part[k]) for k in ks}


def _order_stats_bracketed(a: np.ndarray, ks: list[int]) -> Optional[dict[int, float]]:
    """Exact order statistics of a large array without partitioning all of it.

    A sorted strided sample (about 8k values) brackets each wanted rank; one pass counts the values below the
    bracket and extracts those inside it, and only that small subset is partitioned. Overlapping brackets are
    merged. Returns None if a bracket misses its rank (possible only for adversarially ordered data), and the
    caller falls back to a full partition, so the result is always exact."""
    n = a.size
    s = np.sort(a[:: max(1, n // 8192)])
    ns = s.size
    brackets: list[list[Any]] = []
    for k in ks:
        r = k / (n - 1) * (ns - 1)
        margin = 6 * math.sqrt(max(r * (ns - r) / ns, 1.0)) + 4
        lo_i, hi_i = max(0, math.floor(r - margin)), min(ns - 1, math.ceil(r + margin))
        if brackets and lo_i <= brackets[-1][1]:
            brackets[-1][1] = max(brackets[-1][1], hi_i)
            brackets[-1][2].append(k)
        else:
            brackets.append([lo_i, hi_i, [k]])
    out: dict[int, float] = {}
    for lo_i, hi_i, group in brackets:
        mask = np.ones(n, dtype=bool) if lo_i == 0 else a >= s[lo_i]
        below = 0 if lo_i == 0 else n - int(np.count_nonzero(mask))
        if hi_i < ns - 1:
            mask &= a <= s[hi_i]
        window = a[mask]
        rel = [k - below for k in group]
        if rel[0] < 0 or rel[-1] >= window.size:
            return None
        part = np.partition(window, rel)
        for k, r_ in zip(group, rel):
            out[k] = float(part[r_])
    return out


def quantiles(a: np.ndarray, qs: Sequence[float]) -> list[float]:
    """Percentiles ``qs`` (0-100, linear interpolation, as ``np.percentile``). Large arrays use bracketed
    selection on a small window per percentile; small ones one partition. Both are exact."""
    n = a.size
    pos = [(n - 1) * (q / 100.0) for q in qs]  # NumPy's order of operations
    lo = [math.floor(h) for h in pos]
    ks = sorted({k for i in lo for k in (i, min(i + 1, n - 1))})
    stats = _order_stats_bracketed(a, ks) if n >= 65536 else None
    if stats is None:
        stats = _order_stats_partition(a, ks)
    out = []
    for h, i in zip(pos, lo):
        v0, v1 = stats[i], stats[min(i + 1, n - 1)]
        t = h - i
        d = v1 - v0
        # NumPy's lerp, so the result is bit-for-bit np.percentile(..., method="linear")
        out.append(v1 - d * (1 - t) if t >= 0.5 else v0 + d * t)
    return out


def summary(a: np.ndarray, extra: Sequence[float] = ()) -> dict[str, float]:
    """Mean, standard deviation, min, max and the p50/p90/p95/p99 percentiles (linear interpolation), plus any
    ``extra`` percentiles under keys ``p<q>``; all order statistics come from one partition."""
    qs = [50.0, 90.0, 95.0, 99.0, *[q for q in extra if q not in (0.0, 50.0, 90.0, 95.0, 99.0, 100.0)]]
    v = quantiles(a, qs)
    by_q = dict(zip(qs, v))
    mean = float(a.mean())
    if a.size > 1:
        c = a - mean
        std = math.sqrt(float(np.dot(c, c)) / (a.size - 1))
    else:
        std = 0.0
    out = {
        "mean": mean,
        "std": std,
        "min": float(a.min()),
        "max": float(a.max()),
        "p50": by_q[50.0],
        "p90": by_q[90.0],
        "p95": by_q[95.0],
        "p99": by_q[99.0],
    }
    for q in extra:
        out[f"p{q:g}"] = float(a.min()) if q == 0 else float(a.max()) if q == 100 else by_q[q]
    return out


def lists_of(x: Any, name: str) -> list[list[Any]]:
    """One (possibly empty) list per example."""
    rows = seq(x, name)
    out = []
    for i, r in enumerate(rows):
        if isinstance(r, (str, bytes)) or not hasattr(r, "__iter__"):
            raise InputValidationError(f"{name}[{i}] must be a list.")
        out.append(list(r))
    return out
