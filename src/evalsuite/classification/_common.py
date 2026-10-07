"""Shared averaging logic for count-based classification metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from ..core.context import ClassificationContext
from ..core.exceptions import InputValidationError, UnsupportedTaskError
from ..core.result import MetricResult
from ..core.types import FloatArray, ZeroDivision
from ..core.validation import safe_divide, validate_zero_division

CountFn = Callable[[FloatArray, FloatArray, FloatArray, FloatArray, ZeroDivision], FloatArray]

AVERAGES = ("auto", "binary", "micro", "macro", "weighted", "samples", None)


def resolve_average(ctx: ClassificationContext, average: Optional[str]) -> Optional[str]:
    if average not in AVERAGES:
        raise UnsupportedTaskError(
            f"average={average!r} is not supported. Use one of: 'binary', 'micro', 'macro', 'weighted', "
            "'samples' (multilabel) or None (per-class values)."
        )
    if average == "auto":
        return "binary" if ctx.target_type == "binary" else "macro"
    if average == "binary" and ctx.target_type != "binary":
        raise UnsupportedTaskError(
            f"average='binary' needs a binary target, but this target is {ctx.target_type} "
            f"({len(ctx.labels)} labels). Choose average='macro', 'micro' or 'weighted', "
            "or None for per-class values."
        )
    if average == "samples" and ctx.target_type != "multilabel":
        raise UnsupportedTaskError("average='samples' is only defined for multilabel targets.")
    return average


def positive_index(ctx: ClassificationContext, pos_label: Any) -> Optional[int]:
    """Index of the positive class in ctx.labels, or None if it never occurs (all-negative data)."""
    labels = ctx.labels
    if pos_label is None:
        if set(labels.tolist()) <= {0, 1, -1}:
            pos_label = 1
        else:
            raise InputValidationError(
                f"Labels are {labels.tolist()}; specify which one is positive with pos_label=..."
            )
    hits = np.flatnonzero(labels == pos_label)
    if hits.size:
        return int(hits[0])
    if labels.shape[0] <= 1:
        return None
    raise InputValidationError(f"pos_label={pos_label!r} is not one of the labels {labels.tolist()}.")


def averaged(
    ctx: ClassificationContext,
    fn: CountFn,
    *,
    metric: str,
    name: str,
    average: Optional[str],
    pos_label: Any = None,
    zero_division: ZeroDivision = "warn",
    extra_params: Optional[dict[str, Any]] = None,
) -> MetricResult:
    """Apply a vectorised count metric fn(tp, fp, fn, tn) with the requested averaging."""
    zd = validate_zero_division(zero_division)
    avg = resolve_average(ctx, average)
    c = ctx.counts
    params: dict[str, Any] = {"average": avg, "zero_division": zero_division, **(extra_params or {})}

    if avg == "binary":
        idx = positive_index(ctx, pos_label)
        if idx is None:
            tp = fp = fnn = np.zeros(1)
            tn = np.array([ctx.total_weight])
        else:
            tp, fp, fnn, tn = (c[k][idx : idx + 1] for k in ("tp", "fp", "fn", "tn"))
        params["pos_label"] = 1 if pos_label is None else pos_label
        return MetricResult(metric, name, float(fn(tp, fp, fnn, tn, zd)[0]), params)

    if avg == "micro":
        tp_s, fp_s, fn_s, tn_s = (np.array([c[k].sum()]) for k in ("tp", "fp", "fn", "tn"))
        return MetricResult(metric, name, float(fn(tp_s, fp_s, fn_s, tn_s, zd)[0]), params)

    if avg == "samples":
        w = ctx.weights
        t = ctx.y_true.astype(bool)
        if ctx.y_pred is None:
            raise InputValidationError("This metric needs y_pred (predicted labels).")
        p = ctx.y_pred.astype(bool)
        per_sample = fn(
            (t & p).sum(1).astype(float),
            (~t & p).sum(1).astype(float),
            (t & ~p).sum(1).astype(float),
            (~t & ~p).sum(1).astype(float),
            zd,
        )
        return MetricResult(metric, name, float(np.average(per_sample, weights=w)), params)

    per_class = fn(c["tp"], c["fp"], c["fn"], c["tn"], zd)
    if avg is None:
        return MetricResult(metric, name, per_class, params, labels=tuple(ctx.labels.tolist()))
    if avg == "macro":
        return MetricResult(metric, name, float(np.mean(per_class)), params)
    support = c["support"]
    value = safe_divide((per_class * support).sum(), support.sum(), zero_division=zd, metric=name)
    return MetricResult(metric, name, float(value), params)
