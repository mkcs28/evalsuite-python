"""Resolve metric names to functions and call them on index subsets."""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..core.exceptions import InputValidationError
from ..core.validation import target_type, to_numpy

MetricFn = Callable[..., Any]


def resolve_metric(metric: Any) -> tuple[MetricFn, str]:
    """A metric function (``es.f1``) or its name (``"f1"``, ``"classification.f1"``)."""
    if callable(metric):
        return metric, getattr(metric, "__name__", "metric")
    if isinstance(metric, str):
        import evalsuite

        name = metric.split(".")[-1]
        fn = getattr(evalsuite, name, None)
        if callable(fn) and name in {m.split(".")[-1] for m in evalsuite.list_metrics()}:
            return fn, name
        raise InputValidationError(
            f"Unknown metric '{metric}'. Use a metric function or a name from list_metrics()."
        )
    raise InputValidationError("metric must be a metric function (e.g. es.f1) or its name (e.g. 'f1').")


def as_observations(x: Any, name: str) -> NDArray[Any]:
    """Observations to resample along the first axis: 1-D/2-D arrays (labels, values, probabilities), stacks of
    segmentation masks (3-D or more: images first), or per-image detection dicts / differently sized masks
    (held in a 1-D object array)."""
    if (
        isinstance(x, (list, tuple))
        and x
        and (isinstance(x[0], Mapping) or (np.ndim(x[0]) >= 2 and len({np.shape(m) for m in x}) > 1))
    ):
        out = np.empty(len(x), dtype=object)
        for i, v in enumerate(x):
            out[i] = v
        return out
    arr = np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x)
    if arr.ndim > 2:
        if arr.shape[0] == 0:
            raise InputValidationError(f"{name} is empty. Provide at least one observation.")
        return arr
    return to_numpy(x, name, allow_2d=True)


def is_categorical(y: NDArray[Any]) -> bool:
    if y.ndim > 2 or (y.dtype == object and y.size and not isinstance(y.flat[0], (str, bytes))):
        return False
    try:
        return target_type(y) in ("binary", "multiclass", "multilabel")
    except Exception:
        return False


class MetricCall:
    """Evaluate ``fn(y_true[idx], second[idx], **kwargs)`` repeatedly on index subsets."""

    def __init__(
        self,
        fn: MetricFn,
        y_true: Any,
        y_pred: Any = None,
        y_prob: Any = None,
        sample_weight: Any = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if (y_pred is None) == (y_prob is None):
            raise InputValidationError(
                "Pass exactly one of y_pred (label/value metrics) or y_prob (probability metrics)."
            )
        self.fn = fn
        self.y_true = as_observations(y_true, "y_true")
        second = y_pred if y_pred is not None else y_prob
        self.second = as_observations(second, "y_pred" if y_pred is not None else "y_prob")
        if self.second.shape[0] != self.y_true.shape[0]:
            raise InputValidationError(
                f"y_true and {'y_pred' if y_pred is not None else 'y_prob'} must contain the same number of "
                f"observations. Received {self.y_true.shape[0]} and {self.second.shape[0]}."
            )
        self.n = self.y_true.shape[0]
        self.weight = None if sample_weight is None else to_numpy(sample_weight, "sample_weight")
        self.kwargs = dict(kwargs or {})
        params = inspect.signature(fn).parameters
        self.categorical = is_categorical(self.y_true)
        # Keep every class in every resample so per-class and averaged metrics stay comparable.
        if self.categorical and "labels" in params and "labels" not in self.kwargs and self.y_true.ndim == 1:
            present = self.y_true if y_pred is None else np.concatenate([self.y_true, self.second])
            with contextlib.suppress(TypeError):  # incomparable labels: the metric reports it
                self.kwargs["labels"] = np.unique(present)
        if self.weight is not None and "sample_weight" not in params:
            raise InputValidationError(f"{getattr(fn, '__name__', 'This metric')} does not accept sample_weight.")

    def __call__(self, idx: Optional[NDArray[np.int64]] = None) -> float:
        if idx is None:
            yt, s, w = self.y_true, self.second, self.weight
        else:
            yt, s = self.y_true[idx], self.second[idx]
            w = None if self.weight is None else self.weight[idx]
        kw = dict(self.kwargs)
        if w is not None:
            kw["sample_weight"] = w
        return float(self.fn(yt, s, **kw))
