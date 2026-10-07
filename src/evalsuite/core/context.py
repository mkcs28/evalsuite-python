"""EvaluationContext: validate inputs once and cache shared intermediate results.

Every classification metric derives from per-class counts (TP, FP, FN, TN). The context computes the
confusion matrix once; ``evaluate()`` passes one context to all requested metrics, so ten metrics cost one
pass over the data.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .exceptions import InputValidationError, UnsupportedTaskError
from .types import ArrayLike, FloatArray
from .validation import (
    TargetType,
    check_consistent_length,
    check_finite,
    resolve_labels,
    target_type,
    to_numpy,
    validate_probabilities,
    validate_sample_weight,
)

__all__ = ["ClassificationContext"]


class ClassificationContext:
    """Validated classification inputs plus cached counts.

    For binary and multiclass targets, ``labels`` defines the class order used by every per-class output and
    by the columns of 2-D ``y_prob``. For multilabel targets, labels are column indices.
    """

    def __init__(
        self,
        y_true: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        *,
        y_prob: Optional[ArrayLike] = None,
        labels: Optional[ArrayLike] = None,
        sample_weight: Optional[ArrayLike] = None,
    ) -> None:
        yt = to_numpy(y_true, "y_true", allow_2d=True)
        yp = None if y_pred is None else to_numpy(y_pred, "y_pred", allow_2d=True)
        check_finite(yt, "y_true")
        if yp is not None:
            check_finite(yp, "y_pred")
        self.n = check_consistent_length(y_true=yt, y_pred=yp)
        kind = target_type(yt)
        if kind == "continuous":
            raise UnsupportedTaskError(
                "y_true contains non-integer numbers, which looks like a regression target. "
                'Use evalsuite.regression metrics or evaluate(..., task="regression").'
            )
        if yp is not None and yt.ndim != yp.ndim:
            raise InputValidationError(
                f"y_true has {yt.ndim} dimension(s) but y_pred has {yp.ndim}. Multilabel targets need both as "
                "indicator matrices; single-label targets need both as 1-D label arrays."
            )
        if kind == "multilabel" and yp is not None:
            if yp.shape[1] != yt.shape[1]:
                raise InputValidationError(f"y_true has {yt.shape[1]} label columns but y_pred has {yp.shape[1]}.")
            if not np.isin(yp, (0, 1)).all():
                raise InputValidationError("Multilabel y_pred must be a 0/1 indicator matrix.")
        self.target_type: TargetType = kind
        self.y_true = yt
        self.y_pred = yp
        self.sample_weight = validate_sample_weight(sample_weight, self.n)
        if kind == "multilabel":
            self.labels: NDArray[Any] = np.arange(yt.shape[1])
        else:
            self.labels = resolve_labels(yt, yp, labels)
            if labels is not None:
                observed = np.unique(yt if yp is None else np.concatenate([yt, yp]))
                missing = np.setdiff1d(observed, self.labels)
                if missing.size:
                    raise InputValidationError(
                        f"y_true/y_pred contain label(s) {missing.tolist()} that are not in labels. "
                        "Include every label that occurs, or filter the data first."
                    )
            if kind == "binary" and self.labels.shape[0] > 2:
                self.target_type = "multiclass"
        self._y_prob_raw = y_prob

    # ---- encodings -------------------------------------------------------------------------------
    @cached_property
    def _index(self) -> dict[Any, int]:
        return {lab.item() if hasattr(lab, "item") else lab: i for i, lab in enumerate(self.labels)}

    def encode(self, y: NDArray[Any]) -> NDArray[np.int64]:
        sorted_labels = np.all(self.labels[:-1] <= self.labels[1:]) if self.labels.dtype != object else False
        if sorted_labels:
            return np.searchsorted(self.labels, y).astype(np.int64)
        idx = self._index
        return np.fromiter((idx[v.item() if hasattr(v, "item") else v] for v in y), dtype=np.int64, count=len(y))

    @cached_property
    def true_idx(self) -> NDArray[np.int64]:
        return self.encode(self.y_true)

    @cached_property
    def pred_idx(self) -> NDArray[np.int64]:
        if self.y_pred is None:
            raise InputValidationError("This metric needs y_pred (predicted labels).")
        return self.encode(self.y_pred)

    @cached_property
    def weights(self) -> FloatArray:
        return np.ones(self.n) if self.sample_weight is None else self.sample_weight

    # ---- cached counts ---------------------------------------------------------------------------
    @cached_property
    def confusion_matrix(self) -> FloatArray:
        """Rows: true labels; columns: predicted labels (weighted counts). Single-label targets only."""
        if self.target_type == "multilabel":
            raise UnsupportedTaskError(
                "A single confusion matrix is not defined for multilabel targets; use multilabel_counts."
            )
        k = self.labels.shape[0]
        flat = self.true_idx * k + self.pred_idx
        return np.bincount(flat, weights=self.weights, minlength=k * k).reshape(k, k).astype(np.float64)

    @cached_property
    def counts(self) -> dict[str, FloatArray]:
        """Per-class (or per-label) weighted tp, fp, fn, tn and support."""
        if self.target_type == "multilabel":
            if self.y_pred is None:
                raise InputValidationError("This metric needs y_pred (predicted labels).")
            w = self.weights[:, None]
            t = self.y_true.astype(bool)
            p = self.y_pred.astype(bool)
            tp = ((t & p) * w).sum(0)
            fp = ((~t & p) * w).sum(0)
            fn = ((t & ~p) * w).sum(0)
            tn = ((~t & ~p) * w).sum(0)
        else:
            cm = self.confusion_matrix
            tp = np.diag(cm).copy()
            fp = cm.sum(0) - tp
            fn = cm.sum(1) - tp
            tn = cm.sum() - tp - fp - fn
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": tp + fn}

    @cached_property
    def total_weight(self) -> float:
        return float(self.weights.sum())

    # ---- probabilities ---------------------------------------------------------------------------
    @cached_property
    def y_prob(self) -> FloatArray:
        if self._y_prob_raw is None:
            raise InputValidationError(
                "This metric needs predicted probabilities (y_prob): P(positive class) for binary tasks, "
                "or one column per class in label order for multiclass tasks."
            )
        k = self.labels.shape[0]
        p = validate_probabilities(
            self._y_prob_raw,
            self.n,
            n_classes=None if self.target_type == "binary" else k,
            rows_sum_to_one=self.target_type == "multiclass",
            unit="label" if self.target_type == "multilabel" else "class",
        )
        if self.target_type == "binary" and p.ndim == 2:
            if p.shape[1] != 2:
                raise InputValidationError(
                    f"For binary tasks y_prob must be P(positive) or two columns; received {p.shape[1]} columns."
                )
            p = p[:, 1]
        if self.target_type == "multiclass" and p.ndim == 1:
            raise InputValidationError(
                f"Multiclass tasks need y_prob with one column per class ({k} columns, in label order)."
            )
        if self.target_type == "multilabel" and (p.ndim != 2 or p.shape[1] != self.y_true.shape[1]):
            raise InputValidationError("Multilabel y_prob must have one probability column per label.")
        return p
