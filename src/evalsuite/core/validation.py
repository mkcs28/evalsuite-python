"""Centralised input validation.

Inputs are converted to NumPy once, checked once, and never modified in place.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .exceptions import InputValidationError, UndefinedMetricWarning
from .types import ArrayLike, FloatArray, ZeroDivision

__all__ = [
    "TargetType",
    "check_consistent_length",
    "resolve_labels",
    "safe_divide",
    "target_type",
    "to_numpy",
    "validate_probabilities",
    "validate_sample_weight",
    "validate_zero_division",
]

TargetType = Literal["binary", "multiclass", "multilabel", "continuous"]


def to_numpy(x: ArrayLike, name: str, *, allow_2d: bool = False) -> NDArray[Any]:
    """Convert array-likes (lists, NumPy arrays, pandas objects) to a NumPy array without copying
    when possible. Raises :class:`InputValidationError` for None, scalars, empty input or bad shapes."""
    if x is None:
        raise InputValidationError(f"{name} is required but was None.")
    values = getattr(x, "to_numpy", None)
    arr = np.asarray(values() if callable(values) else x)
    if arr.ndim == 0:
        raise InputValidationError(f"{name} must be one-dimensional array-like; received a scalar.")
    if arr.ndim == 2 and arr.shape[1] == 1 and not allow_2d:
        arr = arr.ravel()
    if arr.ndim > (2 if allow_2d else 1):
        expected = "1-D or 2-D" if allow_2d else "1-D"
        raise InputValidationError(f"{name} must be {expected}; received an array with shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise InputValidationError(f"{name} is empty. Provide at least one observation.")
    if (
        arr.ndim == 1
        and arr.dtype.kind in "US"
        and isinstance(x, (list, tuple))
        and not all(isinstance(v, (str, bytes)) for v in x)
    ):
        # NumPy silently turns [0, "a"] into ["0", "a"]; a mix of numbers and strings is almost always a bug.
        raise InputValidationError(f"{name} mixes numbers and strings; use one label type throughout.")
    return arr


def _label_kind(arr: NDArray[Any]) -> str:
    if arr.dtype.kind in "US":
        return "string"
    if arr.dtype.kind in "biuf":
        return "number"
    return "other"


def check_finite(arr: NDArray[Any], name: str) -> None:
    if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise InputValidationError(
            f"{name} contains {n_nan} NaN and {n_inf} infinite value(s). Remove or impute them before evaluating."
        )


def check_consistent_length(**arrays: Optional[NDArray[Any]]) -> int:
    """All given arrays must have the same number of observations; returns that number."""
    lengths = {name: a.shape[0] for name, a in arrays.items() if a is not None}
    if len(set(lengths.values())) > 1:
        detail = " and ".join(f"{k}={v}" for k, v in lengths.items())
        names = " and ".join(lengths)
        raise InputValidationError(f"{names} must contain the same number of observations. Received {detail}.")
    return int(next(iter(lengths.values())))


def validate_sample_weight(sample_weight: Optional[ArrayLike], n: int) -> Optional[FloatArray]:
    if sample_weight is None:
        return None
    w = to_numpy(sample_weight, "sample_weight").astype(np.float64, copy=False)
    check_finite(w, "sample_weight")
    if w.shape[0] != n:
        raise InputValidationError(
            f"sample_weight must contain one weight per observation. Received {w.shape[0]} for {n} observations."
        )
    if np.any(w < 0):
        raise InputValidationError("sample_weight must be non-negative.")
    if w.sum() == 0:
        raise InputValidationError("sample_weight must not sum to zero.")
    return w


def unique_labels(y: NDArray[Any]) -> NDArray[Any]:
    """Sorted unique values, like ``np.unique``. Small-range integer labels (the usual case for class labels)
    are found with one marking pass instead of a sort, which is several times faster on large arrays."""
    if y.size and np.issubdtype(y.dtype, np.integer):
        lo, hi = int(y.min()), int(y.max())
        if lo > -(2**62) and hi < 2**62 and hi - lo <= max(1024, y.size):
            present = np.zeros(hi - lo + 1, dtype=bool)
            present[(y.astype(np.int64) - lo).ravel()] = True  # int64 offsets: no overflow for small dtypes
            return (np.flatnonzero(present).astype(np.int64) + lo).astype(y.dtype)
    return np.unique(y)


def target_type(y: NDArray[Any]) -> TargetType:
    """Infer the type of a label array."""
    if y.ndim == 2:
        if y.shape[1] > 1 and np.isin(y, (0, 1)).all():
            return "multilabel"
        if np.issubdtype(y.dtype, np.floating) and not np.all(np.mod(y, 1) == 0):
            return "continuous"  # multi-output regression
        raise InputValidationError(
            "2-D targets must be a binary indicator matrix (0/1, one column per label) for multilabel tasks; "
            f"received shape {y.shape} with values other than 0 and 1."
        )
    if np.issubdtype(y.dtype, np.floating) and not np.all(np.mod(y, 1) == 0):
        return "continuous"
    try:
        n_unique = unique_labels(y).shape[0]
    except TypeError as exc:
        raise InputValidationError(
            "Labels must be mutually comparable (for example all integers or all strings); found a mix of types."
        ) from exc
    return "binary" if n_unique <= 2 else "multiclass"


def resolve_labels(
    y_true: NDArray[Any], y_pred: Optional[NDArray[Any]], labels: Optional[ArrayLike]
) -> NDArray[Any]:
    """Sorted labels present in y_true or y_pred, or the user's explicit list (order kept)."""
    if labels is not None:
        lab = to_numpy(labels, "labels")
        if np.unique(lab).shape[0] != lab.shape[0]:
            raise InputValidationError("labels contains duplicates.")
        return lab
    if y_pred is not None and {_label_kind(y_true), _label_kind(y_pred)} == {"string", "number"}:
        raise InputValidationError(
            f"y_true has {_label_kind(y_true)} labels but y_pred has {_label_kind(y_pred)} labels; "
            "use the same label type in both (for example map class names to integers first)."
        )
    present = y_true if y_pred is None else np.concatenate([y_true, y_pred])
    try:
        return unique_labels(present)
    except TypeError as exc:
        raise InputValidationError(
            "Labels in y_true and y_pred must be mutually comparable (for example all integers or all strings)."
        ) from exc


def validate_probabilities(
    y_prob: ArrayLike,
    n: int,
    *,
    n_classes: Optional[int] = None,
    name: str = "y_prob",
    rows_sum_to_one: bool = True,
    unit: str = "class",
) -> FloatArray:
    """Probabilities in [0, 1]. For multiclass (2-D), one column per class and rows summing to 1.
    Multilabel probabilities are independent per label (``rows_sum_to_one=False``)."""
    p = to_numpy(y_prob, name, allow_2d=True).astype(np.float64, copy=False)
    check_finite(p, name)
    if p.shape[0] != n:
        raise InputValidationError(
            f"{name} must contain one row per observation. Received {p.shape[0]} for {n} observations."
        )
    if np.any(p < 0) or np.any(p > 1):
        raise InputValidationError(
            f"{name} must contain probabilities in [0, 1]; found values from {float(np.min(p)):.4g} to "
            f"{float(np.max(p)):.4g}. "
            "If these are scores or logits, convert them to probabilities first."
        )
    if p.ndim == 2:
        if n_classes is not None and p.shape[1] != n_classes:
            raise InputValidationError(
                f"{name} has {p.shape[1]} columns but there are {n_classes} {unit}es. "
                f"Provide one probability column per {unit}, in label order."
                if unit == "class"
                else f"{name} has {p.shape[1]} columns but there are {n_classes} {unit}s. "
                f"Provide one probability column per {unit}."
            )
        sums = p.sum(axis=1)
        if rows_sum_to_one and not np.allclose(sums, 1.0, atol=1e-6):
            worst = float(np.abs(sums - 1).max())
            raise InputValidationError(
                f"Each row of {name} must sum to 1 for multiclass probabilities (largest deviation {worst:.3g})."
            )
    return p


def validate_zero_division(zero_division: ZeroDivision) -> ZeroDivision:
    if zero_division == "warn":
        return zero_division
    if isinstance(zero_division, (int, float)) and (np.isnan(zero_division) or 0 <= zero_division <= 1):
        return float(zero_division)
    raise InputValidationError('zero_division must be "warn", 0, 1 or np.nan.')


def safe_divide(
    num: NDArray[np.float64] | float,
    den: NDArray[np.float64] | float,
    *,
    zero_division: ZeroDivision = "warn",
    metric: str = "metric",
) -> NDArray[np.float64]:
    """Element-wise num/den. Where den == 0 the result is ``zero_division``.

    ``"warn"`` (default) returns 0 and emits :class:`UndefinedMetricWarning`, matching scikit-learn's
    convention; pass ``zero_division=np.nan`` to propagate undefined values instead. Never silent.
    """
    num_a = np.asarray(num, dtype=np.float64)
    den_a = np.asarray(den, dtype=np.float64)
    zero = den_a == 0
    out = np.divide(num_a, den_a, out=np.zeros(np.broadcast(num_a, den_a).shape), where=~zero)
    if np.any(zero):
        if zero_division == "warn":
            warnings.warn(
                f"{metric} is undefined for {int(np.sum(zero))} case(s) because the denominator is zero; "
                "using 0. Set zero_division=0, 1 or np.nan to choose the value explicitly.",
                UndefinedMetricWarning,
                stacklevel=4,
            )
            out[zero] = 0.0
        else:
            out[zero] = float(zero_division)
    return out
