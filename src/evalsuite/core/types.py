"""Shared type aliases."""

from __future__ import annotations

from typing import Any, Literal, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Average", "ArrayLike", "FloatArray", "IntArray", "Task", "ZeroDivision"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: How per-class results are combined. ``None`` returns per-class values.
Average = Literal["binary", "micro", "macro", "weighted", "samples", None]

#: Value used when a metric's denominator is zero. ``"warn"`` returns NaN and warns.
ZeroDivision = Union[Literal["warn"], float]

Task = Literal["classification", "regression"]

JSONValue = Union[None, bool, int, float, str, list[Any], dict[str, Any]]
