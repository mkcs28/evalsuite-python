"""Diagnostic-accuracy metrics for binary tests and risk models.

All functions take a binary reference standard ``y_true`` and binary test results ``y_pred`` (or predicted
risks ``y_prob`` for decision curves). The positive class is ``pos_label`` (default 1 for 0/1 labels).

Ratios that divide by zero are reported honestly: ``x / 0`` with ``x > 0`` is ``inf`` and ``0 / 0`` is NaN,
always with an :class:`~evalsuite.core.exceptions.UndefinedMetricWarning`. They are never replaced by 0,
because a likelihood ratio of 0 would mean the opposite of an infinite one.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np

from ..classification._common import positive_index
from ..core.context import ClassificationContext
from ..core.exceptions import InputValidationError, UndefinedMetricWarning, UnsupportedTaskError
from ..core.registry import register
from ..core.result import MetricResult
from ..core.types import ArrayLike, ZeroDivision
from ..core.validation import safe_divide, validate_zero_division

__all__ = [
    "diagnostic_odds_ratio",
    "lr_negative",
    "lr_positive",
    "net_benefit",
    "ppv",
    "sensitivity",
    "youden_j",
]

_C = "clinical"
_REF_ALTMAN = "Altman DG, Bland JM. Diagnostic tests 1: sensitivity and specificity. BMJ. 1994;308(6943):1552."
_REF_ALTMAN2 = "Altman DG, Bland JM. Diagnostic tests 2: predictive values. BMJ. 1994;309(6947):102."
_REF_DEEKS = "Deeks JJ, Altman DG. Diagnostic tests 4: likelihood ratios. BMJ. 2004;329(7458):168-169."
_REF_SIMEL = (
    "Simel DL, Samsa GP, Matchar DB. Likelihood ratios with confidence: sample size estimation for "
    "diagnostic test studies. J Clin Epidemiol. 1991;44(8):763-770."
)
_REF_GLAS = (
    "Glas AS, Lijmer JG, Prins MH, Bonsel GJ, Bossuyt PMM. The diagnostic odds ratio: a single indicator "
    "of test performance. J Clin Epidemiol. 2003;56(11):1129-1135."
)
_REF_YOUDEN = "Youden WJ. Index for rating diagnostic tests. Cancer. 1950;3(1):32-35."
_REF_VICKERS = (
    "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. "
    "Med Decis Making. 2006;26(6):565-574."
)


def binary_counts(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> tuple[float, float, float, float]:
    """``(tp, fp, fn, tn)`` for the positive class (weighted if ``sample_weight`` is given)."""
    ctx = ClassificationContext(y_true, y_pred, sample_weight=sample_weight)
    if ctx.target_type not in ("binary",):
        raise UnsupportedTaskError(
            f"Diagnostic metrics are defined for binary outcomes; this target is {ctx.target_type}."
        )
    idx = positive_index(ctx, pos_label)
    if idx is None:
        return 0.0, 0.0, 0.0, float(ctx.total_weight)
    c = ctx.counts
    return tuple(float(c[k][idx]) for k in ("tp", "fp", "fn", "tn"))  # type: ignore[return-value]


def _ratio(num: float, den: float, what: str) -> float:
    if den != 0:
        return num / den
    value = math.inf if num > 0 else math.nan
    warnings.warn(
        f"{what} is {'infinite' if num > 0 else 'undefined (0/0)'} for this input because its denominator is "
        "zero; returning " + ("inf." if num > 0 else "NaN."),
        UndefinedMetricWarning,
        stacklevel=3,
    )
    return value


def _params(pos_label: Any) -> dict[str, Any]:
    return {"pos_label": 1 if pos_label is None else pos_label}


@register(
    category=_C,
    task="binary",
    name="Sensitivity",
    definition="Of the people with the condition, the proportion the test calls positive (true positive rate).",
    formula="TP / (TP + FN)",
    range="[0, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_ALTMAN,),
)
def sensitivity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> MetricResult:
    """Sensitivity (recall of the positive class)."""
    tp, _fp, fn, _tn = binary_counts(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
    zd = validate_zero_division(zero_division)
    value = float(safe_divide(tp, tp + fn, zero_division=zd, metric="Sensitivity"))
    return MetricResult(
        "sensitivity", "Sensitivity", value, {**_params(pos_label), "zero_division": zero_division}
    )


@register(
    category=_C,
    task="binary",
    name="Positive predictive value",
    definition="Of the people the test calls positive, the proportion who have the condition. Depends on "
    "prevalence.",
    formula="TP / (TP + FP)",
    range="[0, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_ALTMAN2,),
)
def ppv(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> MetricResult:
    """Positive predictive value (precision of the positive class)."""
    tp, fp, _fn, _tn = binary_counts(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
    zd = validate_zero_division(zero_division)
    value = float(safe_divide(tp, tp + fp, zero_division=zd, metric="PPV"))
    return MetricResult("ppv", "PPV", value, {**_params(pos_label), "zero_division": zero_division})


def _sens_spec(tp: float, fp: float, fn: float, tn: float) -> tuple[float, float]:
    sens = tp / (tp + fn) if tp + fn else math.nan
    spec = tn / (tn + fp) if tn + fp else math.nan
    return sens, spec


def _need_both_classes(tp: float, fp: float, fn: float, tn: float, what: str) -> None:
    if tp + fn == 0 or tn + fp == 0:
        raise InputValidationError(
            f"{what} needs both people with and without the condition in y_true; "
            f"got {tp + fn:g} positive and {tn + fp:g} negative."
        )


@register(
    category=_C,
    task="binary",
    name="Positive likelihood ratio",
    definition="How much a positive result raises the odds of the condition: sensitivity / (1 − specificity).",
    formula="LR+ = sensitivity / (1 − specificity)",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_DEEKS, _REF_SIMEL),
)
def lr_positive(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """Positive likelihood ratio. ``inf`` (with a warning) when specificity is 1 and sensitivity > 0."""
    tp, fp, fn, tn = binary_counts(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
    _need_both_classes(tp, fp, fn, tn, "LR+")
    sens, spec = _sens_spec(tp, fp, fn, tn)
    return MetricResult("lr_positive", "LR+", _ratio(sens, 1 - spec, "LR+"), _params(pos_label))


@register(
    category=_C,
    task="binary",
    name="Negative likelihood ratio",
    definition="How much a negative result lowers the odds of the condition: (1 − sensitivity) / specificity.",
    formula="LR− = (1 − sensitivity) / specificity",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_DEEKS, _REF_SIMEL),
    higher_is_better=False,
)
def lr_negative(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """Negative likelihood ratio. ``inf`` (with a warning) when specificity is 0 and sensitivity < 1."""
    tp, fp, fn, tn = binary_counts(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
    _need_both_classes(tp, fp, fn, tn, "LR−")
    sens, spec = _sens_spec(tp, fp, fn, tn)
    return MetricResult("lr_negative", "LR−", _ratio(1 - sens, spec, "LR−"), _params(pos_label))


@register(
    category=_C,
    task="binary",
    name="Diagnostic odds ratio",
    definition="Odds of a positive test in people with the condition divided by the odds in people without it.",
    formula="DOR = (TP · TN) / (FP · FN) = LR+ / LR−",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_GLAS,),
)
def diagnostic_odds_ratio(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    correction: float = 0.0,
) -> MetricResult:
    """Diagnostic odds ratio. ``correction=0.5`` adds 0.5 to every cell when any cell is zero (Haldane–Anscombe),
    which keeps the estimate finite; the default 0 reports ``inf`` or NaN with a warning instead."""
    tp, fp, fn, tn = binary_counts(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
    _need_both_classes(tp, fp, fn, tn, "The diagnostic odds ratio")
    if correction < 0:
        raise InputValidationError("correction must be 0 or positive (0.5 is the usual choice).")
    applied = bool(correction) and min(tp, fp, fn, tn) == 0
    if applied:
        tp, fp, fn, tn = (v + correction for v in (tp, fp, fn, tn))
    value = _ratio(tp * tn, fp * fn, "The diagnostic odds ratio")
    return MetricResult(
        "diagnostic_odds_ratio",
        "DOR",
        value,
        {**_params(pos_label), "correction": correction, "correction_applied": applied},
    )


@register(
    category=_C,
    task="binary",
    name="Youden's J",
    definition="Sensitivity + specificity − 1: 0 for a test no better than chance, 1 for a perfect test.",
    formula="J = sensitivity + specificity − 1",
    range="[−1, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_YOUDEN,),
)
def youden_j(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """Youden's J statistic (informedness) for a binary test."""
    tp, fp, fn, tn = binary_counts(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
    _need_both_classes(tp, fp, fn, tn, "Youden's J")
    sens, spec = _sens_spec(tp, fp, fn, tn)
    return MetricResult("youden_j", "Youden's J", sens + spec - 1, _params(pos_label))


def _binary_risk(
    y_true: ArrayLike, y_prob: ArrayLike, pos_label: Any, sample_weight: Optional[ArrayLike]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = ClassificationContext(y_true, None, y_prob=y_prob, sample_weight=sample_weight)
    if ctx.target_type != "binary":
        raise UnsupportedTaskError("Decision curves are defined for binary outcomes.")
    idx = positive_index(ctx, pos_label)
    pos = ctx.labels[idx] if idx is not None else None
    y = (ctx.y_true == pos).astype(np.float64) if pos is not None else np.zeros(ctx.n)
    return y, np.asarray(ctx.y_prob, dtype=np.float64), np.asarray(ctx.weights, dtype=np.float64)


def _check_thresholds(t: ArrayLike) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if arr.ndim != 1 or arr.size == 0 or np.any(~np.isfinite(arr)) or np.any((arr <= 0) | (arr >= 1)):
        raise InputValidationError("Threshold probabilities must be strictly between 0 and 1.")
    return arr


def net_benefit_curve(
    y: np.ndarray, p: np.ndarray, w: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Net benefit of the model and of treating everyone at each threshold."""
    n = w.sum()
    treat = p[None, :] >= thresholds[:, None]
    tp = (treat * (w * y)[None, :]).sum(axis=1)
    fp = (treat * (w * (1 - y))[None, :]).sum(axis=1)
    odds = thresholds / (1 - thresholds)
    model = tp / n - fp / n * odds
    prevalence = float((w * y).sum() / n)
    treat_all = prevalence - (1 - prevalence) * odds
    return model, treat_all


@register(
    category=_C,
    task="binary",
    name="Net benefit",
    definition="Clinical value of acting on the model at a threshold probability: true positives per person "
    "minus false positives per person weighted by the odds of the threshold.",
    formula="NB(pₜ) = TP/n − FP/n · pₜ / (1 − pₜ)",
    range="(−∞, prevalence]",
    input_requirements=("y_true", "y_prob", "threshold"),
    references=(_REF_VICKERS,),
)
def net_benefit(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    threshold: float,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """Net benefit at one threshold probability (treat when predicted risk ≥ threshold)."""
    y, p, w = _binary_risk(y_true, y_prob, pos_label, sample_weight)
    t = _check_thresholds(threshold)
    if t.size != 1:
        raise InputValidationError("threshold must be a single probability; use decision_curve() for many.")
    model, _ = net_benefit_curve(y, p, w, t)
    return MetricResult(
        "net_benefit", "Net benefit", float(model[0]), {**_params(pos_label), "threshold": float(t[0])}
    )
