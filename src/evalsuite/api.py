"""High-level API: ``evaluate()``."""

from __future__ import annotations

import platform
from collections.abc import Sequence
from typing import Any, Callable, Optional

import numpy as np

from .classification import metrics as clf
from .classification._common import averaged
from .core.context import ClassificationContext
from .core.exceptions import InputValidationError, UnsupportedTaskError
from .core.result import EvaluationResult, MetricResult
from .core.types import ArrayLike, ZeroDivision
from .core.validation import target_type, to_numpy
from .regression import metrics as reg
from .version import __version__

__all__ = ["evaluate"]

CtxMetric = Callable[[ClassificationContext], MetricResult]


def _classification_registry(
    average: Optional[str], pos_label: Any, zero_division: ZeroDivision
) -> dict[str, tuple[CtxMetric, bool]]:
    """metric id -> (function of a shared context, needs probabilities)."""

    def avg(fn: Any, metric: str, name: str) -> CtxMetric:
        return lambda ctx: averaged(
            ctx,
            fn,
            metric=metric,
            name=name,
            average=average,
            pos_label=pos_label,
            zero_division=zero_division,
        )

    return {
        "accuracy": (clf._accuracy, False),
        "balanced_accuracy": (clf._balanced_accuracy, False),
        "precision": (avg(clf._precision, "precision", "Precision"), False),
        "recall": (avg(clf._recall, "recall", "Recall"), False),
        "f1": (avg(clf._fbeta_fn(1.0), "f1", "F1"), False),
        "specificity": (avg(clf._specificity, "specificity", "Specificity"), False),
        "npv": (avg(clf._npv, "npv", "NPV"), False),
        "jaccard": (avg(clf._jaccard, "jaccard", "Jaccard"), False),
        "mcc": (lambda ctx: clf._mcc(ctx, zero_division), False),
        "cohen_kappa": (lambda ctx: clf._cohen_kappa(ctx, None, zero_division), False),
        "hamming_loss": (_hamming, False),
        "roc_auc": (lambda ctx: clf._roc_auc(ctx, average=average, pos_label=pos_label), True),
        "average_precision": (
            lambda ctx: clf._average_precision(ctx, average=average, pos_label=pos_label),
            True,
        ),
        "log_loss": (lambda ctx: clf._log_loss(ctx, pos_label), True),
        "brier_score": (lambda ctx: clf._brier(ctx, pos_label), True),
    }


def _hamming(ctx: ClassificationContext) -> MetricResult:
    if ctx.target_type == "multilabel":
        assert ctx.y_pred is not None  # noqa: S101
        wrong = (ctx.y_true != ctx.y_pred).mean(axis=1)
    else:
        wrong = (ctx.true_idx != ctx.pred_idx).astype(float)
    return MetricResult("hamming_loss", "Hamming loss", float(np.average(wrong, weights=ctx.weights)))


_DEFAULT_SINGLE = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "mcc",
    "cohen_kappa",
]
_DEFAULT_MULTILABEL = ["accuracy", "hamming_loss", "precision", "recall", "f1", "jaccard"]
_DEFAULT_PROB = ["roc_auc", "average_precision", "log_loss", "brier_score"]
_DEFAULT_PROB_MULTILABEL = ["roc_auc", "average_precision"]

_REGRESSION: dict[str, Callable[..., MetricResult]] = {
    "mae": reg.mae,
    "mse": reg.mse,
    "rmse": reg.rmse,
    "r2": reg.r2,
    "explained_variance": reg.explained_variance,
    "median_absolute_error": reg.median_absolute_error,
    "max_error": reg.max_error,
    "mean_bias_error": reg.mean_bias_error,
    "mape": reg.mape,
    "smape": reg.smape,
    "msle": reg.msle,
    "rmsle": reg.rmsle,
    "rae": reg.rae,
    "rse": reg.rse,
    "huber_loss": reg.huber_loss,
    "quantile_loss": reg.quantile_loss,
}
_DEFAULT_REGRESSION = [
    "mae",
    "mse",
    "rmse",
    "r2",
    "explained_variance",
    "median_absolute_error",
    "max_error",
    "mean_bias_error",
]
_NO_WEIGHTS = {"max_error"}


def _infer_task(y_true: ArrayLike, y_prob: Optional[ArrayLike]) -> str:
    if y_prob is not None:
        return "classification"
    return (
        "regression"
        if target_type(to_numpy(y_true, "y_true", allow_2d=True)) == "continuous"
        else "classification"
    )


def evaluate(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    *,
    y_prob: Optional[ArrayLike] = None,
    task: Optional[str] = None,
    metrics: Optional[Sequence[str]] = None,
    average: Optional[str] = "auto",
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> EvaluationResult:
    """Evaluate predictions with a standard set of metrics (or the ``metrics`` you name).

    The task is inferred when omitted: probabilities or integer/string labels mean classification, non-integer
    numbers mean regression. Inputs are validated once and shared intermediate results (the confusion matrix)
    are computed once for all metrics.

    >>> import evalsuite as es
    >>> r = es.evaluate([0, 1, 1, 0], [0, 1, 0, 0])
    >>> round(r["accuracy"], 2)
    0.75
    """
    task = task or _infer_task(y_true, y_prob)
    if task == "classification":
        return _evaluate_classification(
            y_true, y_pred, y_prob, metrics, average, labels, pos_label, sample_weight, zero_division
        )
    if task == "regression":
        if y_pred is None:
            raise InputValidationError("Regression evaluation needs y_pred.")
        return _evaluate_regression(y_true, y_pred, metrics, sample_weight)
    raise UnsupportedTaskError("task must be 'classification' or 'regression' in EvalSuite 0.1.")


def _metadata(**extra: Any) -> dict[str, Any]:
    return {
        "evalsuite_version": __version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        **extra,
    }


def _evaluate_classification(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike],
    y_prob: Optional[ArrayLike],
    metrics: Optional[Sequence[str]],
    average: Optional[str],
    labels: Optional[ArrayLike],
    pos_label: Any,
    sample_weight: Optional[ArrayLike],
    zero_division: ZeroDivision,
) -> EvaluationResult:
    if y_pred is None and y_prob is None:
        raise InputValidationError("Classification evaluation needs y_pred, y_prob, or both.")
    ctx = ClassificationContext(y_true, y_pred, y_prob=y_prob, labels=labels, sample_weight=sample_weight)
    registry = _classification_registry(average, pos_label, zero_division)
    if metrics is None:
        multilabel = ctx.target_type == "multilabel"
        names = [] if y_pred is None else list(_DEFAULT_MULTILABEL if multilabel else _DEFAULT_SINGLE)
        if y_prob is not None:
            names += _DEFAULT_PROB_MULTILABEL if multilabel else _DEFAULT_PROB
    else:
        names = list(metrics)
        unknown = [m for m in names if m not in registry]
        if unknown:
            raise InputValidationError(
                f"Unknown classification metric(s): {', '.join(unknown)}. Available: {', '.join(registry)}."
            )
    out: dict[str, MetricResult] = {}
    for name in names:
        fn, needs_prob = registry[name]
        if needs_prob and y_prob is None:
            raise InputValidationError(f"Metric '{name}' needs y_prob (predicted probabilities).")
        if not needs_prob and y_pred is None:
            raise InputValidationError(f"Metric '{name}' needs y_pred (predicted labels).")
        out[name] = fn(ctx)
    cm = None if ctx.target_type == "multilabel" or y_pred is None else ctx.confusion_matrix
    return EvaluationResult(
        task="classification",
        metrics=out,
        n_samples=ctx.n,
        target_type=ctx.target_type,
        labels=tuple(ctx.labels.tolist()),
        confusion_matrix=cm,
        metadata=_metadata(average=average, zero_division=zero_division, weighted=sample_weight is not None),
    )


def _evaluate_regression(
    y_true: ArrayLike, y_pred: ArrayLike, metrics: Optional[Sequence[str]], sample_weight: Optional[ArrayLike]
) -> EvaluationResult:
    yt = to_numpy(y_true, "y_true", allow_2d=True)
    multi = yt.ndim == 2
    if metrics is None:
        names = [m for m in _DEFAULT_REGRESSION if not (multi and m == "max_error")]
    else:
        names = list(metrics)
        unknown = [m for m in names if m not in _REGRESSION]
        if unknown:
            raise InputValidationError(
                f"Unknown regression metric(s): {', '.join(unknown)}. Available: {', '.join(_REGRESSION)}."
            )
    out: dict[str, MetricResult] = {}
    for name in names:
        fn = _REGRESSION[name]
        if name in _NO_WEIGHTS:
            if sample_weight is not None:
                raise InputValidationError(f"Metric '{name}' does not support sample_weight.")
            out[name] = fn(y_true, y_pred)
        else:
            out[name] = fn(y_true, y_pred, sample_weight=sample_weight)
    return EvaluationResult(
        task="regression",
        metrics=out,
        n_samples=yt.shape[0],
        target_type="continuous",
        metadata=_metadata(weighted=sample_weight is not None, outputs=yt.shape[1] if multi else 1),
    )
