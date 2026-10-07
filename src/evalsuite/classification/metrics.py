"""Classification metrics for binary, multiclass and multilabel targets.

Conventions (stated once, applied everywhere):

* ``labels`` order defines per-class outputs and the columns of 2-D ``y_prob``. By default it is the sorted
  set of labels present in ``y_true`` and ``y_pred``.
* ``average="auto"`` means ``"binary"`` for binary targets and ``"macro"`` otherwise; the resolved value is
  recorded in ``result.params["average"]``.
* Undefined ratios (zero denominators) follow ``zero_division``: ``"warn"`` returns 0 with an
  :class:`~evalsuite.core.exceptions.UndefinedMetricWarning`; ``np.nan`` propagates NaN.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np

from ..core.context import ClassificationContext
from ..core.exceptions import InputValidationError, MetricInputError, UnsupportedTaskError
from ..core.registry import register
from ..core.result import MetricResult
from ..core.types import ArrayLike, FloatArray, ZeroDivision
from ..core.validation import safe_divide, validate_zero_division
from ._common import averaged, positive_index, resolve_average

__all__ = [
    "accuracy",
    "average_precision",
    "balanced_accuracy",
    "brier_score",
    "cohen_kappa",
    "confusion_matrix",
    "f1",
    "fbeta",
    "hamming_loss",
    "jaccard",
    "log_loss",
    "mcc",
    "npv",
    "pr_curve",
    "precision",
    "recall",
    "roc_auc",
    "roc_curve",
    "specificity",
    "top_k_accuracy",
]

_C = "classification"
_REF_SOKOLOVA = "Sokolova M, Lapalme G. A systematic analysis of performance measures for classification tasks. Information Processing & Management. 2009;45(4):427-437."
_REF_POWERS = "Powers DMW. Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation. Journal of Machine Learning Technologies. 2011;2(1):37-63."


def _ctx(y_true: ArrayLike, y_pred: Optional[ArrayLike] = None, **kw: Any) -> ClassificationContext:
    return ClassificationContext(y_true, y_pred, **kw)


# ---- count formulas (vectorised; shared by every averaging mode) ----------------------------------
def _precision(tp: FloatArray, fp: FloatArray, fn: FloatArray, tn: FloatArray, zd: ZeroDivision) -> FloatArray:
    return safe_divide(tp, tp + fp, zero_division=zd, metric="Precision")


def _recall(tp: FloatArray, fp: FloatArray, fn: FloatArray, tn: FloatArray, zd: ZeroDivision) -> FloatArray:
    return safe_divide(tp, tp + fn, zero_division=zd, metric="Recall")


def _specificity(tp: FloatArray, fp: FloatArray, fn: FloatArray, tn: FloatArray, zd: ZeroDivision) -> FloatArray:
    return safe_divide(tn, tn + fp, zero_division=zd, metric="Specificity")


def _npv(tp: FloatArray, fp: FloatArray, fn: FloatArray, tn: FloatArray, zd: ZeroDivision) -> FloatArray:
    return safe_divide(tn, tn + fn, zero_division=zd, metric="NPV")


def _jaccard(tp: FloatArray, fp: FloatArray, fn: FloatArray, tn: FloatArray, zd: ZeroDivision) -> FloatArray:
    return safe_divide(tp, tp + fp + fn, zero_division=zd, metric="Jaccard index")


def _fbeta_fn(beta: float) -> Any:
    b2 = beta * beta

    def fn(tp: FloatArray, fp: FloatArray, fn_: FloatArray, tn: FloatArray, zd: ZeroDivision) -> FloatArray:
        return safe_divide((1 + b2) * tp, (1 + b2) * tp + b2 * fn_ + fp, zero_division=zd, metric=f"F{beta:g}")

    return fn


_AVG_DOC = "average, labels, pos_label, sample_weight, zero_division"


# ---- label-based metrics --------------------------------------------------------------------------
@register(
    category=_C,
    task="binary, multiclass, multilabel",
    name="Accuracy",
    definition="Proportion of observations whose predicted label equals the true label (exact match for multilabel).",
    formula="(1/n) Σ 1[ŷᵢ = yᵢ]",
    range="[0, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_SOKOLOVA,),
)
def accuracy(y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: Optional[ArrayLike] = None) -> MetricResult:
    """Accuracy. For multilabel targets this is subset accuracy (all labels of a row must match)."""
    return _accuracy(_ctx(y_true, y_pred, sample_weight=sample_weight))


def _accuracy(ctx: ClassificationContext) -> MetricResult:
    if ctx.target_type == "multilabel":
        assert ctx.y_pred is not None  # noqa: S101 - ensured by context for label metrics
        correct = (ctx.y_true == ctx.y_pred).all(axis=1)
    else:
        correct = ctx.true_idx == ctx.pred_idx
    return MetricResult("accuracy", "Accuracy", float(np.average(correct, weights=ctx.weights)))


@register(
    category=_C,
    task="binary, multiclass, multilabel",
    name="Precision",
    definition="Of the observations predicted positive, the proportion that are truly positive.",
    formula="TP / (TP + FP)",
    range="[0, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_SOKOLOVA,),
)
def precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "auto",
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> MetricResult:
    """Precision (positive predictive value)."""
    ctx = _ctx(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    return averaged(
        ctx,
        _precision,
        metric="precision",
        name="Precision",
        average=average,
        pos_label=pos_label,
        zero_division=zero_division,
    )


@register(
    category=_C,
    task="binary, multiclass, multilabel",
    name="Recall (sensitivity)",
    definition="Of the truly positive observations, the proportion predicted positive.",
    formula="TP / (TP + FN)",
    range="[0, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_SOKOLOVA,),
)
def recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "auto",
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> MetricResult:
    """Recall (sensitivity, true positive rate)."""
    ctx = _ctx(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    return averaged(
        ctx,
        _recall,
        metric="recall",
        name="Recall",
        average=average,
        pos_label=pos_label,
        zero_division=zero_division,
    )


@register(
    category=_C,
    task="binary, multiclass, multilabel",
    name="Specificity",
    definition="Of the truly negative observations, the proportion predicted negative.",
    formula="TN / (TN + FP)",
    range="[0, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_SOKOLOVA,),
)
def specificity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "auto",
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> MetricResult:
    """Specificity (true negative rate). Per class, the class is treated as positive versus the rest."""
    ctx = _ctx(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    return averaged(
        ctx,
        _specificity,
        metric="specificity",
        name="Specificity",
        average=average,
        pos_label=pos_label,
        zero_division=zero_division,
    )


@register(
    category=_C,
    task="binary, multiclass, multilabel",
    name="Negative predictive value",
    definition="Of the observations predicted negative, the proportion that are truly negative.",
    formula="TN / (TN + FN)",
    range="[0, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_POWERS,),
)
def npv(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "auto",
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> MetricResult:
    """Negative predictive value."""
    ctx = _ctx(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    return averaged(
        ctx, _npv, metric="npv", name="NPV", average=average, pos_label=pos_label, zero_division=zero_division
    )


@register(
    category=_C,
    task="binary, multiclass, multilabel",
    name="F1 score",
    definition="Harmonic mean of precision and recall.",
    formula="2·TP / (2·TP + FP + FN)",
    range="[0, 1]",
    input_requirements=("y_true", "y_pred"),
    references=("van Rijsbergen CJ. Information Retrieval. 2nd ed. Butterworths; 1979.", _REF_SOKOLOVA),
)
def f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "auto",
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> MetricResult:
    """F1 score."""
    ctx = _ctx(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    return averaged(
        ctx,
        _fbeta_fn(1.0),
        metric="f1",
        name="F1",
        average=average,
        pos_label=pos_label,
        zero_division=zero_division,
    )


@register(
    category=_C,
    task="binary, multiclass, multilabel",
    name="F-beta score",
    definition="Weighted harmonic mean of precision and recall; beta > 1 favours recall, beta < 1 precision.",
    formula="(1+β²)·TP / ((1+β²)·TP + β²·FN + FP)",
    range="[0, 1]",
    input_requirements=("y_true", "y_pred"),
    references=("van Rijsbergen CJ. Information Retrieval. 2nd ed. Butterworths; 1979.",),
)
def fbeta(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    beta: float,
    average: Optional[str] = "auto",
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> MetricResult:
    """F-beta score."""
    if not (isinstance(beta, (int, float)) and beta > 0 and np.isfinite(beta)):
        raise InputValidationError("beta must be a positive finite number.")
    ctx = _ctx(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    return averaged(
        ctx,
        _fbeta_fn(float(beta)),
        metric="fbeta",
        name=f"F{beta:g}",
        average=average,
        pos_label=pos_label,
        zero_division=zero_division,
        extra_params={"beta": float(beta)},
    )


@register(
    category=_C,
    task="binary, multiclass, multilabel",
    name="Jaccard index",
    definition="Size of the intersection over the size of the union of predicted and true positives.",
    formula="TP / (TP + FP + FN)",
    range="[0, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(
        "Jaccard P. The distribution of the flora in the alpine zone. New Phytologist. 1912;11(2):37-50.",
    ),
)
def jaccard(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "auto",
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> MetricResult:
    """Jaccard index (intersection over union)."""
    ctx = _ctx(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    return averaged(
        ctx,
        _jaccard,
        metric="jaccard",
        name="Jaccard",
        average=average,
        pos_label=pos_label,
        zero_division=zero_division,
    )


@register(
    category=_C,
    task="binary, multiclass",
    name="Balanced accuracy",
    definition="Mean recall over classes present in y_true; robust to class imbalance.",
    formula="(1/K) Σₖ TPₖ / (TPₖ + FNₖ)",
    range="[0, 1] (adjusted: chance = 0)",
    input_requirements=("y_true", "y_pred"),
    references=(
        "Brodersen KH, Ong CS, Stephan KE, Buhmann JM. The balanced accuracy and its posterior distribution. ICPR 2010:3121-3124.",
    ),
)
def balanced_accuracy(
    y_true: ArrayLike, y_pred: ArrayLike, *, adjusted: bool = False, sample_weight: Optional[ArrayLike] = None
) -> MetricResult:
    """Balanced accuracy. With ``adjusted=True`` chance performance scores 0 and perfect performance 1."""
    return _balanced_accuracy(_ctx(y_true, y_pred, sample_weight=sample_weight), adjusted=adjusted)


def _balanced_accuracy(ctx: ClassificationContext, *, adjusted: bool = False) -> MetricResult:
    if ctx.target_type == "multilabel":
        raise UnsupportedTaskError("Balanced accuracy is defined for binary and multiclass targets only.")
    c = ctx.counts
    present = c["support"] > 0
    score = float(np.mean(c["tp"][present] / c["support"][present]))
    if adjusted:
        k = int(present.sum())
        chance = 1.0 / k
        score = (score - chance) / (1 - chance) if k > 1 else 0.0
    return MetricResult("balanced_accuracy", "Balanced accuracy", score, {"adjusted": adjusted})


@register(
    category=_C,
    task="binary, multiclass",
    name="Matthews correlation coefficient",
    definition="Correlation between true and predicted labels using all confusion-matrix cells; 0 is chance level.",
    formula="(c·s − Σₖ pₖtₖ) / √((s² − Σₖ pₖ²)(s² − Σₖ tₖ²))",
    range="[-1, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(
        "Matthews BW. Comparison of the predicted and observed secondary structure of T4 phage lysozyme. Biochim Biophys Acta. 1975;405(2):442-451.",
        "Gorodkin J. Comparing two K-category assignments by a K-category correlation coefficient. Comput Biol Chem. 2004;28(5-6):367-374.",
        "Chicco D, Jurman G. The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics. 2020;21:6.",
    ),
)
def mcc(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> MetricResult:
    """Matthews correlation coefficient (Gorodkin's multiclass generalisation)."""
    return _mcc(_ctx(y_true, y_pred, sample_weight=sample_weight), zero_division)


def _mcc(ctx: ClassificationContext, zero_division: ZeroDivision = "warn") -> MetricResult:
    if ctx.target_type == "multilabel":
        raise UnsupportedTaskError("MCC is defined for binary and multiclass targets; use it per label instead.")
    cm = ctx.confusion_matrix
    t, p = cm.sum(1), cm.sum(0)
    c, s = np.trace(cm), cm.sum()
    cov_ytyp = c * s - (t * p).sum()
    cov_ypyp = s * s - (p * p).sum()
    cov_ytyt = s * s - (t * t).sum()
    value = safe_divide(
        cov_ytyp,
        np.sqrt(cov_ytyt * cov_ypyp),
        zero_division=validate_zero_division(zero_division),
        metric="MCC",
    )
    return MetricResult("mcc", "MCC", float(value), {"zero_division": zero_division})


@register(
    category=_C,
    task="binary, multiclass",
    name="Cohen's kappa",
    definition="Agreement between true and predicted labels corrected for agreement expected by chance.",
    formula="κ = 1 − Σ wᵢⱼ Oᵢⱼ / Σ wᵢⱼ Eᵢⱼ",
    range="[-1, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(
        "Cohen J. A coefficient of agreement for nominal scales. Educ Psychol Meas. 1960;20(1):37-46.",
        "Cohen J. Weighted kappa. Psychol Bull. 1968;70(4):213-220.",
    ),
)
def cohen_kappa(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    weights: Optional[Literal["linear", "quadratic"]] = None,
    labels: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> MetricResult:
    """Cohen's kappa, optionally linearly or quadratically weighted (for ordinal labels, in label order)."""
    return _cohen_kappa(_ctx(y_true, y_pred, labels=labels, sample_weight=sample_weight), weights, zero_division)


def _cohen_kappa(
    ctx: ClassificationContext, weights: Optional[str] = None, zero_division: ZeroDivision = "warn"
) -> MetricResult:
    if ctx.target_type == "multilabel":
        raise UnsupportedTaskError("Cohen's kappa is defined for binary and multiclass targets.")
    cm = ctx.confusion_matrix
    k = cm.shape[0]
    expected = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    if weights is None:
        w = 1.0 - np.eye(k)
    elif weights in ("linear", "quadratic"):
        grid = np.abs(np.subtract.outer(np.arange(k), np.arange(k))).astype(float)
        w = grid if weights == "linear" else grid**2
    else:
        raise InputValidationError("weights must be None, 'linear' or 'quadratic'.")
    ratio = safe_divide(
        (w * cm).sum(),
        (w * expected).sum(),
        zero_division=validate_zero_division(zero_division),
        metric="Cohen's kappa",
    )
    return MetricResult("cohen_kappa", "Cohen's kappa", float(1 - ratio), {"weights": weights})


@register(
    category=_C,
    task="binary, multiclass, multilabel",
    name="Hamming loss",
    definition="Fraction of labels predicted incorrectly (for single-label targets equal to 1 − accuracy).",
    formula="(1/(n·L)) Σᵢ Σₗ 1[ŷᵢₗ ≠ yᵢₗ]",
    range="[0, 1]",
    input_requirements=("y_true", "y_pred"),
    references=(
        "Tsoumakas G, Katakis I. Multi-label classification: an overview. Int J Data Warehousing and Mining. 2007;3(3):1-13.",
    ),
    higher_is_better=False,
)
def hamming_loss(
    y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: Optional[ArrayLike] = None
) -> MetricResult:
    """Hamming loss."""
    ctx = _ctx(y_true, y_pred, sample_weight=sample_weight)
    if ctx.target_type == "multilabel":
        assert ctx.y_pred is not None  # noqa: S101
        wrong = (ctx.y_true != ctx.y_pred).mean(axis=1)
    else:
        wrong = (ctx.true_idx != ctx.pred_idx).astype(float)
    return MetricResult("hamming_loss", "Hamming loss", float(np.average(wrong, weights=ctx.weights)))


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
    normalize: Optional[Literal["true", "pred", "all"]] = None,
) -> np.ndarray:
    """Confusion matrix: rows are true labels, columns predicted labels, both in ``labels`` order
    (sorted by default). Counts are integers unless ``sample_weight`` or ``normalize`` is given."""
    ctx = _ctx(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    cm = ctx.confusion_matrix
    if normalize is not None:
        if normalize not in ("true", "pred", "all"):
            raise InputValidationError("normalize must be None, 'true', 'pred' or 'all'.")
        den = (
            cm.sum(1, keepdims=True)
            if normalize == "true"
            else cm.sum(0, keepdims=True)
            if normalize == "pred"
            else cm.sum()
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(cm / den)
    return cm if sample_weight is not None else cm.astype(np.int64)


# ---- probability-based metrics --------------------------------------------------------------------
def _binary_curve(y: FloatArray, score: FloatArray, w: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Cumulative weighted false/true positives at each distinct score threshold (descending)."""
    order = np.argsort(-score, kind="mergesort")
    score, y, w = score[order], y[order], w[order]
    distinct = np.flatnonzero(np.diff(score))
    thr_idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y * w)[thr_idx]
    fps = np.cumsum((1 - y) * w)[thr_idx]
    return fps, tps, score[thr_idx]


def _require_both_classes(tps: FloatArray, fps: FloatArray, what: str) -> None:
    if tps[-1] == 0 or fps[-1] == 0:
        missing = "positive" if tps[-1] == 0 else "negative"
        raise MetricInputError(
            f"{what} is undefined because y_true contains no {missing} observations. "
            "It needs at least one positive and one negative example."
        )


def _binary_auc(y: FloatArray, score: FloatArray, w: FloatArray) -> float:
    fps, tps, _ = _binary_curve(y, score, w)
    _require_both_classes(tps, fps, "ROC AUC")
    fpr = np.r_[0.0, fps / fps[-1]]
    tpr = np.r_[0.0, tps / tps[-1]]
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # noqa: B009 - NumPy 1.x/2.x
    return float(integrate(tpr, fpr))


def _binary_ap(y: FloatArray, score: FloatArray, w: FloatArray) -> float:
    fps, tps, _ = _binary_curve(y, score, w)
    if tps[-1] == 0:
        raise MetricInputError("Average precision is undefined because y_true contains no positive observations.")
    prec = tps / (tps + fps)
    rec = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def _binary_target(ctx: ClassificationContext, pos_label: Any) -> FloatArray:
    idx = positive_index(ctx, pos_label)
    if idx is None:
        return np.zeros(ctx.n)
    target: FloatArray = (ctx.true_idx == idx).astype(np.float64)
    return target


@register(
    category=_C,
    task="binary, multiclass, multilabel",
    name="ROC AUC",
    definition="Area under the receiver operating characteristic curve: the probability that a random positive "
    "is scored above a random negative (ties count half).",
    formula="∫₀¹ TPR d(FPR)",
    range="[0, 1] (0.5 = chance)",
    input_requirements=("y_true", "y_prob"),
    references=(
        "Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology. 1982;143(1):29-36.",
        "Fawcett T. An introduction to ROC analysis. Pattern Recognit Lett. 2006;27(8):861-874.",
        "Hand DJ, Till RJ. A simple generalisation of the area under the ROC curve for multiple class classification problems. Mach Learn. 2001;45:171-186.",
    ),
)
def roc_auc(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    average: Optional[str] = "auto",
    multi_class: Literal["ovr", "ovo"] = "ovr",
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """ROC AUC. Binary: ``y_prob`` is P(positive). Multiclass: one column per class (label order), averaged
    one-vs-rest (``"ovr"``, macro or weighted) or one-vs-one (``"ovo"``, Hand & Till, macro).
    Multilabel: one column per label, macro/micro/weighted or per-label (``average=None``)."""
    ctx = _ctx(y_true, None, y_prob=y_prob, labels=labels, sample_weight=sample_weight)
    return _roc_auc(ctx, average=average, multi_class=multi_class, pos_label=pos_label)


def _per_column(
    ctx: ClassificationContext,
    fn: Any,
    average: Optional[str],
    metric: str,
    name: str,
    params: dict[str, Any],
) -> MetricResult:
    """Apply a binary score metric per class/label column with macro, weighted, micro or no averaging."""
    p = ctx.y_prob
    w = ctx.weights
    y = (
        ctx.y_true.astype(float)
        if ctx.target_type == "multilabel"
        else (ctx.true_idx[:, None] == np.arange(ctx.labels.shape[0])[None, :]).astype(float)
    )
    if average == "micro":
        return MetricResult(metric, name, fn(y.ravel(), p.ravel(), np.repeat(w, y.shape[1])), params)
    scores = np.array([fn(y[:, j], p[:, j], w) for j in range(y.shape[1])])
    if average is None:
        return MetricResult(metric, name, scores, params, labels=tuple(ctx.labels.tolist()))
    if average == "macro":
        return MetricResult(metric, name, float(scores.mean()), params)
    support = (y * w[:, None]).sum(0)
    return MetricResult(metric, name, float(np.average(scores, weights=support)), params)


def _roc_auc(
    ctx: ClassificationContext,
    *,
    average: Optional[str] = "auto",
    multi_class: str = "ovr",
    pos_label: Any = None,
) -> MetricResult:
    avg = resolve_average(ctx, average)
    params: dict[str, Any] = {"average": avg}
    if ctx.target_type == "binary":
        params["pos_label"] = 1 if pos_label is None else pos_label
        return MetricResult(
            "roc_auc", "ROC AUC", _binary_auc(_binary_target(ctx, pos_label), ctx.y_prob, ctx.weights), params
        )
    if ctx.target_type == "multiclass":
        params["multi_class"] = multi_class
        if multi_class == "ovo":
            if avg != "macro":
                raise UnsupportedTaskError("One-vs-one ROC AUC supports average='macro' only.")
            return MetricResult("roc_auc", "ROC AUC", _ovo_auc(ctx), params)
        if multi_class != "ovr":
            raise UnsupportedTaskError("multi_class must be 'ovr' or 'ovo'.")
        if avg == "micro":
            raise UnsupportedTaskError(
                "average='micro' is not defined for multiclass ROC AUC; use 'macro', 'weighted' or None."
            )
    return _per_column(ctx, _binary_auc, avg, "roc_auc", "ROC AUC", params)


def _ovo_auc(ctx: ClassificationContext) -> float:
    """Hand & Till (2001) multiclass AUC: mean over class pairs of the two directional AUCs."""
    k = ctx.labels.shape[0]
    p, t, w = ctx.y_prob, ctx.true_idx, ctx.weights
    pair_scores = []
    for a in range(k):
        for b in range(a + 1, k):
            mask = (t == a) | (t == b)
            ya = (t[mask] == a).astype(float)
            a_vs_b = _binary_auc(ya, p[mask, a], w[mask])
            b_vs_a = _binary_auc(1 - ya, p[mask, b], w[mask])
            pair_scores.append((a_vs_b + b_vs_a) / 2)
    return float(np.mean(pair_scores))


@register(
    category=_C,
    task="binary, multiclass, multilabel",
    name="Average precision (PR AUC)",
    definition="Precision averaged over recall levels, weighting each precision by the increase in recall "
    "(step interpolation, no trapezoidal optimism).",
    formula="AP = Σₙ (Rₙ − Rₙ₋₁) Pₙ",
    range="[0, 1] (chance = prevalence)",
    input_requirements=("y_true", "y_prob"),
    references=(
        "Davis J, Goadrich M. The relationship between precision-recall and ROC curves. ICML 2006:233-240.",
        "Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE. 2015;10(3):e0118432.",
    ),
)
def average_precision(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    average: Optional[str] = "auto",
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """Average precision, the recommended summary of the precision-recall curve."""
    ctx = _ctx(y_true, None, y_prob=y_prob, labels=labels, sample_weight=sample_weight)
    return _average_precision(ctx, average=average, pos_label=pos_label)


def _average_precision(
    ctx: ClassificationContext, *, average: Optional[str] = "auto", pos_label: Any = None
) -> MetricResult:
    avg = resolve_average(ctx, average)
    params: dict[str, Any] = {"average": avg}
    if ctx.target_type == "binary":
        params["pos_label"] = 1 if pos_label is None else pos_label
        return MetricResult(
            "average_precision",
            "Average precision",
            _binary_ap(_binary_target(ctx, pos_label), ctx.y_prob, ctx.weights),
            params,
        )
    return _per_column(ctx, _binary_ap, avg, "average_precision", "Average precision", params)


def roc_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Binary ROC curve: ``(fpr, tpr, thresholds)``, starting at (0, 0) with threshold +inf."""
    ctx = _ctx(y_true, None, y_prob=y_prob, sample_weight=sample_weight)
    if ctx.target_type != "binary":
        raise UnsupportedTaskError(
            "roc_curve is binary; for multiclass, compute one curve per class (one-vs-rest)."
        )
    fps, tps, thr = _binary_curve(_binary_target(ctx, pos_label), ctx.y_prob, ctx.weights)
    _require_both_classes(tps, fps, "The ROC curve")
    return np.r_[0.0, fps / fps[-1]], np.r_[0.0, tps / tps[-1]], np.r_[np.inf, thr]


def pr_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Binary precision-recall curve: ``(precision, recall, thresholds)`` ordered by increasing threshold
    (one point per distinct score) and ending at (precision=1, recall=0)."""
    ctx = _ctx(y_true, None, y_prob=y_prob, sample_weight=sample_weight)
    if ctx.target_type != "binary":
        raise UnsupportedTaskError("pr_curve is binary; for multiclass, compute one curve per class.")
    fps, tps, thr = _binary_curve(_binary_target(ctx, pos_label), ctx.y_prob, ctx.weights)
    if tps[-1] == 0:
        raise MetricInputError("The PR curve is undefined because y_true contains no positive observations.")
    prec = tps / (tps + fps)
    rec = tps / tps[-1]
    rev = slice(None, None, -1)  # increasing thresholds; every threshold is kept
    return np.r_[prec[rev], 1.0], np.r_[rec[rev], 0.0], thr[rev]


@register(
    category=_C,
    task="binary, multiclass",
    name="Log loss (cross-entropy)",
    definition="Negative mean log-likelihood of the true labels under the predicted probabilities.",
    formula="−(1/n) Σᵢ log p̂ᵢ,yᵢ",
    range="[0, ∞)",
    input_requirements=("y_true", "y_prob"),
    references=("Good IJ. Rational decisions. J R Stat Soc B. 1952;14(1):107-114.",),
    higher_is_better=False,
)
def log_loss(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """Log loss. Probabilities are clipped to [ε, 1 − ε] (ε = float64 machine epsilon) to keep it finite."""
    ctx = _ctx(y_true, None, y_prob=y_prob, labels=labels, sample_weight=sample_weight)
    return _log_loss(ctx, pos_label)


def _log_loss(ctx: ClassificationContext, pos_label: Any = None) -> MetricResult:
    if ctx.target_type == "multilabel":
        raise UnsupportedTaskError("Log loss here is for binary and multiclass targets.")
    eps = np.finfo(np.float64).eps
    p = np.clip(ctx.y_prob, eps, 1 - eps)
    if ctx.target_type == "binary":
        y = _binary_target(ctx, pos_label)
        losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    else:
        losses = -np.log(p[np.arange(ctx.n), ctx.true_idx])
    return MetricResult(
        "log_loss", "Log loss", float(np.average(losses, weights=ctx.weights)), {"eps": float(eps)}
    )


@register(
    category=_C,
    task="binary, multiclass",
    name="Brier score",
    definition="Mean squared difference between predicted probabilities and the outcome; for multiclass, the "
    "squared error summed over classes (Brier's original definition).",
    formula="(1/n) Σᵢ Σₖ (p̂ᵢₖ − yᵢₖ)²  (binary: (1/n) Σᵢ (p̂ᵢ − yᵢ)²)",
    range="[0, 1] binary; [0, 2] multiclass",
    input_requirements=("y_true", "y_prob"),
    references=(
        "Brier GW. Verification of forecasts expressed in terms of probability. Mon Weather Rev. 1950;78(1):1-3.",
    ),
    higher_is_better=False,
)
def brier_score(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    labels: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """Brier score."""
    ctx = _ctx(y_true, None, y_prob=y_prob, labels=labels, sample_weight=sample_weight)
    return _brier(ctx, pos_label)


def _brier(ctx: ClassificationContext, pos_label: Any = None) -> MetricResult:
    if ctx.target_type == "multilabel":
        raise UnsupportedTaskError("Brier score here is for binary and multiclass targets.")
    if ctx.target_type == "binary":
        err = (ctx.y_prob - _binary_target(ctx, pos_label)) ** 2
    else:
        onehot = np.eye(ctx.labels.shape[0])[ctx.true_idx]
        err = ((ctx.y_prob - onehot) ** 2).sum(axis=1)
    return MetricResult("brier_score", "Brier score", float(np.average(err, weights=ctx.weights)))


@register(
    category=_C,
    task="multiclass",
    name="Top-k accuracy",
    definition="Proportion of observations whose true class is among the k classes with the highest probability.",
    formula="(1/n) Σᵢ 1[yᵢ ∈ top-k(p̂ᵢ)]",
    range="[0, 1]",
    input_requirements=("y_true", "y_prob"),
    references=(
        "Russakovsky O, et al. ImageNet Large Scale Visual Recognition Challenge. IJCV. 2015;115:211-252.",
    ),
)
def top_k_accuracy(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    k: int = 2,
    labels: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """Top-k accuracy for multiclass probabilities. Ties at the k-th place count as a hit only if the true
    class's probability is strictly greater than the (k+1)-th largest (no credit for unbroken ties)."""
    ctx = _ctx(y_true, None, y_prob=y_prob, labels=labels, sample_weight=sample_weight)
    if ctx.target_type != "multiclass":
        raise UnsupportedTaskError(
            "top_k_accuracy needs a multiclass target with one probability column per class."
        )
    n_classes = ctx.labels.shape[0]
    if not (isinstance(k, (int, np.integer)) and 1 <= k <= n_classes):
        raise InputValidationError(f"k must be an integer between 1 and the number of classes ({n_classes}).")
    p = ctx.y_prob
    true_p = p[np.arange(ctx.n), ctx.true_idx]
    n_higher = (p > true_p[:, None]).sum(axis=1)
    hit = n_higher < k
    return MetricResult(
        "top_k_accuracy", f"Top-{k} accuracy", float(np.average(hit, weights=ctx.weights)), {"k": int(k)}
    )
