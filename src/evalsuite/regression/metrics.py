"""Regression metrics.

Single- and multi-output targets (2-D arrays, one column per output) are supported. ``multioutput``
is ``"uniform_average"`` (default), ``"raw_values"`` (one value per output) or an array of output weights.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np

from ..core.exceptions import InputValidationError, MetricInputError, UndefinedMetricWarning
from ..core.registry import register
from ..core.result import MetricResult
from ..core.types import ArrayLike, FloatArray
from ..core.validation import check_consistent_length, check_finite, to_numpy, validate_sample_weight

__all__ = [
    "adjusted_r2",
    "explained_variance",
    "huber_loss",
    "mae",
    "mape",
    "max_error",
    "mean_bias_error",
    "median_absolute_error",
    "mse",
    "msle",
    "quantile_loss",
    "r2",
    "rae",
    "rmse",
    "rmsle",
    "rse",
    "smape",
]

_R = "regression"
Multioutput = Union[str, ArrayLike]
_REF_HYNDMAN = (
    "Hyndman RJ, Koehler AB. Another look at measures of forecast accuracy. Int J Forecast. 2006;22(4):679-688."
)
_REF_WILLMOTT = "Willmott CJ, Matsuura K. Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance. Clim Res. 2005;30:79-82."


class _Inputs:
    """Validated regression arrays as float64 (n, k)."""

    def __init__(self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike]) -> None:
        yt = to_numpy(y_true, "y_true", allow_2d=True)
        yp = to_numpy(y_pred, "y_pred", allow_2d=True)
        for arr, name in ((yt, "y_true"), (yp, "y_pred")):
            if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
                raise InputValidationError(
                    f"{name} must be numeric for regression metrics; got dtype {arr.dtype}."
                )
        yt = yt.astype(np.float64, copy=False)
        yp = yp.astype(np.float64, copy=False)
        check_finite(yt, "y_true")
        check_finite(yp, "y_pred")
        self.n = check_consistent_length(y_true=yt, y_pred=yp)
        if yt.ndim != yp.ndim or (yt.ndim == 2 and yt.shape[1] != yp.shape[1]):
            raise InputValidationError(
                f"y_true has shape {yt.shape} but y_pred has shape {yp.shape}; they must match."
            )
        self.multi = yt.ndim == 2
        self.y_true: FloatArray = yt if self.multi else yt[:, None]
        self.y_pred: FloatArray = yp if self.multi else yp[:, None]
        self.weight = validate_sample_weight(sample_weight, self.n)

    @property
    def w(self) -> FloatArray:
        return np.ones(self.n) if self.weight is None else self.weight

    @property
    def error(self) -> FloatArray:
        return self.y_pred - self.y_true

    def mean(self, values: FloatArray) -> FloatArray:
        """Weighted mean over observations, per output column."""
        return np.average(values, axis=0, weights=self.w)


def _finish(
    per_output: FloatArray,
    inp: _Inputs,
    multioutput: Multioutput,
    metric: str,
    name: str,
    params: Optional[dict[str, Any]] = None,
) -> MetricResult:
    params = {**(params or {})}
    if inp.multi:
        params["multioutput"] = multioutput if isinstance(multioutput, str) else "weights"
    if not inp.multi:
        return MetricResult(metric, name, float(per_output[0]), params)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return MetricResult(metric, name, per_output, params, labels=tuple(range(per_output.shape[0])))
        if multioutput == "uniform_average":
            return MetricResult(metric, name, float(per_output.mean()), params)
        raise InputValidationError("multioutput must be 'uniform_average', 'raw_values' or an array of weights.")
    weights = to_numpy(multioutput, "multioutput").astype(float)
    if weights.shape[0] != per_output.shape[0] or np.any(weights < 0) or weights.sum() == 0:
        raise InputValidationError("multioutput weights must be non-negative, one per output, not all zero.")
    return MetricResult(metric, name, float(np.average(per_output, weights=weights)), params)


def _weighted_median(values: FloatArray, w: FloatArray) -> float:
    """Median minimising Σ wᵢ|xᵢ − m|: the smallest value whose cumulative weight reaches half the total."""
    order = np.argsort(values, kind="mergesort")
    v, cw = values[order], np.cumsum(w[order])
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


# ---- metrics --------------------------------------------------------------------------------------
@register(
    category=_R,
    task="regression",
    name="Mean absolute error",
    definition="Average absolute difference between predictions and targets, in the target's units.",
    formula="(1/n) Σ |ŷᵢ − yᵢ|",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_WILLMOTT,),
    higher_is_better=False,
)
def mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Mean absolute error."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    return _finish(inp.mean(np.abs(inp.error)), inp, multioutput, "mae", "MAE")


@register(
    category=_R,
    task="regression",
    name="Mean squared error",
    definition="Average squared difference between predictions and targets; penalises large errors.",
    formula="(1/n) Σ (ŷᵢ − yᵢ)²",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_WILLMOTT,),
    higher_is_better=False,
)
def mse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Mean squared error."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    return _finish(inp.mean(inp.error**2), inp, multioutput, "mse", "MSE")


@register(
    category=_R,
    task="regression",
    name="Root mean squared error",
    definition="Square root of the mean squared error, in the target's units.",
    formula="√((1/n) Σ (ŷᵢ − yᵢ)²)",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred"),
    references=(
        _REF_WILLMOTT,
        "Chai T, Draxler RR. Root mean square error (RMSE) or mean absolute error (MAE)? Geosci Model Dev. 2014;7:1247-1250.",
    ),
    higher_is_better=False,
)
def rmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Root mean squared error (square root taken per output, then averaged)."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    return _finish(np.sqrt(inp.mean(inp.error**2)), inp, multioutput, "rmse", "RMSE")


def _r2_per_output(inp: _Inputs) -> FloatArray:
    w = inp.w[:, None]
    ss_res = (w * inp.error**2).sum(0)
    mean = inp.mean(inp.y_true)
    ss_tot = (w * (inp.y_true - mean) ** 2).sum(0)
    out = np.empty_like(ss_res)
    const = ss_tot == 0
    out[~const] = 1 - ss_res[~const] / ss_tot[~const]
    if np.any(const):
        warnings.warn(
            "R² is undefined when y_true is constant; returning 1.0 for perfect predictions and 0.0 otherwise "
            "(the scikit-learn convention).",
            UndefinedMetricWarning,
            stacklevel=3,
        )
        out[const] = np.where(ss_res[const] == 0, 1.0, 0.0)
    return out


@register(
    category=_R,
    task="regression",
    name="Coefficient of determination (R²)",
    definition="Proportion of the variance in the target explained by the predictions; can be negative for "
    "models worse than predicting the mean.",
    formula="1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²",
    range="(−∞, 1]",
    input_requirements=("y_true", "y_pred"),
    references=("Kvålseth TO. Cautionary note about R². Am Stat. 1985;39(4):279-285.",),
)
def r2(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Coefficient of determination."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    return _finish(_r2_per_output(inp), inp, multioutput, "r2", "R²")


@register(
    category=_R,
    task="regression",
    name="Adjusted R²",
    definition="R² penalised for the number of predictors, for comparing models with different numbers of features.",
    formula="1 − (1 − R²)(n − 1)/(n − p − 1)",
    range="(−∞, 1]",
    input_requirements=("y_true", "y_pred", "n_features"),
    references=("Theil H. Economic Forecasts and Policy. North-Holland; 1961.",),
)
def adjusted_r2(
    y_true: ArrayLike, y_pred: ArrayLike, *, n_features: int, sample_weight: Optional[ArrayLike] = None
) -> MetricResult:
    """Adjusted R². Needs ``n_features`` (number of predictors, excluding the intercept)."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    if inp.multi:
        raise InputValidationError("adjusted_r2 supports single-output targets only.")
    if not isinstance(n_features, (int, np.integer)) or n_features < 0:
        raise InputValidationError("n_features must be a non-negative integer.")
    if inp.n - n_features - 1 <= 0:
        raise MetricInputError(
            f"Adjusted R² needs more observations than predictors + 1 (n={inp.n}, n_features={n_features})."
        )
    r = float(_r2_per_output(inp)[0])
    value = 1 - (1 - r) * (inp.n - 1) / (inp.n - n_features - 1)
    return MetricResult("adjusted_r2", "Adjusted R²", value, {"n_features": int(n_features)})


@register(
    category=_R,
    task="regression",
    name="Mean absolute percentage error",
    definition="Average absolute error relative to the true value, as a fraction (multiply by 100 for percent).",
    formula="(1/n) Σ |ŷᵢ − yᵢ| / |yᵢ|",
    range="[0, ∞)",
    input_requirements=("y_true ≠ 0", "y_pred"),
    references=(_REF_HYNDMAN,),
    higher_is_better=False,
)
def mape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """MAPE as a fraction. Refuses zero targets (division by zero) instead of silently using a tiny epsilon."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    zeros = int((inp.y_true == 0).sum())
    if zeros:
        raise MetricInputError(
            f"MAPE is undefined because y_true contains {zeros} zero value(s). "
            "Use smape, mae or rae instead, or exclude zero targets explicitly."
        )
    return _finish(inp.mean(np.abs(inp.error) / np.abs(inp.y_true)), inp, multioutput, "mape", "MAPE")


@register(
    category=_R,
    task="regression",
    name="Symmetric mean absolute percentage error",
    definition="Absolute error relative to the mean magnitude of target and prediction; a pair that is both "
    "zero contributes 0.",
    formula="(1/n) Σ 2|ŷᵢ − yᵢ| / (|yᵢ| + |ŷᵢ|)",
    range="[0, 2]",
    input_requirements=("y_true", "y_pred"),
    references=(
        _REF_HYNDMAN,
        "Makridakis S. Accuracy measures: theoretical and practical concerns. Int J Forecast. 1993;9(4):527-529.",
    ),
    higher_is_better=False,
)
def smape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Symmetric MAPE (fraction, range [0, 2])."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    den = np.abs(inp.y_true) + np.abs(inp.y_pred)
    terms = np.divide(2 * np.abs(inp.error), den, out=np.zeros_like(den), where=den != 0)
    return _finish(inp.mean(terms), inp, multioutput, "smape", "sMAPE")


def _check_log_domain(inp: _Inputs, name: str) -> None:
    if np.any(inp.y_true < 0) or np.any(inp.y_pred < 0):
        raise MetricInputError(f"{name} needs non-negative y_true and y_pred (it compares log(1 + y)).")


@register(
    category=_R,
    task="regression",
    name="Mean squared logarithmic error",
    definition="Mean squared difference of log(1 + y); emphasises relative error and penalises under-prediction.",
    formula="(1/n) Σ (log(1 + ŷᵢ) − log(1 + yᵢ))²",
    range="[0, ∞)",
    input_requirements=("y_true ≥ 0", "y_pred ≥ 0"),
    references=(_REF_HYNDMAN,),
    higher_is_better=False,
)
def msle(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Mean squared logarithmic error."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    _check_log_domain(inp, "MSLE")
    return _finish(inp.mean((np.log1p(inp.y_pred) - np.log1p(inp.y_true)) ** 2), inp, multioutput, "msle", "MSLE")


@register(
    category=_R,
    task="regression",
    name="Root mean squared logarithmic error",
    definition="Square root of the mean squared logarithmic error.",
    formula="√MSLE",
    range="[0, ∞)",
    input_requirements=("y_true ≥ 0", "y_pred ≥ 0"),
    references=(_REF_HYNDMAN,),
    higher_is_better=False,
)
def rmsle(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Root mean squared logarithmic error."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    _check_log_domain(inp, "RMSLE")
    per = np.sqrt(inp.mean((np.log1p(inp.y_pred) - np.log1p(inp.y_true)) ** 2))
    return _finish(per, inp, multioutput, "rmsle", "RMSLE")


@register(
    category=_R,
    task="regression",
    name="Median absolute error",
    definition="Median of the absolute errors; robust to outliers.",
    formula="median(|ŷᵢ − yᵢ|)",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred"),
    references=("Rousseeuw PJ, Leroy AM. Robust Regression and Outlier Detection. Wiley; 1987.",),
    higher_is_better=False,
)
def median_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Median absolute error (weighted median when ``sample_weight`` is given)."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    abs_err = np.abs(inp.error)
    if inp.weight is None:
        per = np.median(abs_err, axis=0)
    else:
        per = np.array([_weighted_median(abs_err[:, j], inp.w) for j in range(abs_err.shape[1])])
    return _finish(per, inp, multioutput, "median_absolute_error", "Median absolute error")


@register(
    category=_R,
    task="regression",
    name="Explained variance",
    definition="Proportion of target variance explained, ignoring systematic bias (unlike R²).",
    formula="1 − Var(y − ŷ) / Var(y)",
    range="(−∞, 1]",
    input_requirements=("y_true", "y_pred"),
    references=("Draper NR, Smith H. Applied Regression Analysis. 3rd ed. Wiley; 1998.",),
)
def explained_variance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Explained variance score."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    resid = inp.y_true - inp.y_pred
    var_res = inp.mean((resid - inp.mean(resid)) ** 2)
    var_true = inp.mean((inp.y_true - inp.mean(inp.y_true)) ** 2)
    out = np.empty_like(var_res)
    const = var_true == 0
    out[~const] = 1 - var_res[~const] / var_true[~const]
    if np.any(const):
        warnings.warn(
            "Explained variance is undefined when y_true is constant; returning 1.0 for zero residual "
            "variance and 0.0 otherwise.",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        out[const] = np.where(var_res[const] == 0, 1.0, 0.0)
    return _finish(out, inp, multioutput, "explained_variance", "Explained variance")


@register(
    category=_R,
    task="regression",
    name="Maximum error",
    definition="Largest absolute error: the worst case.",
    formula="maxᵢ |ŷᵢ − yᵢ|",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred"),
    references=("Hyndman RJ, Athanasopoulos G. Forecasting: Principles and Practice. 3rd ed. OTexts; 2021.",),
    higher_is_better=False,
)
def max_error(y_true: ArrayLike, y_pred: ArrayLike) -> MetricResult:
    """Maximum absolute error (single output, unweighted)."""
    inp = _Inputs(y_true, y_pred, None)
    if inp.multi:
        raise InputValidationError("max_error supports single-output targets only.")
    return MetricResult("max_error", "Max error", float(np.abs(inp.error).max()))


@register(
    category=_R,
    task="regression",
    name="Mean bias error",
    definition="Average signed error: positive when the model over-predicts on average.",
    formula="(1/n) Σ (ŷᵢ − yᵢ)",
    range="(−∞, ∞) (0 = unbiased)",
    input_requirements=("y_true", "y_pred"),
    references=(_REF_WILLMOTT,),
    higher_is_better=None,
)
def mean_bias_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Mean bias error (prediction minus truth)."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    return _finish(inp.mean(inp.error), inp, multioutput, "mean_bias_error", "Mean bias error")


@register(
    category=_R,
    task="regression",
    name="Quantile (pinball) loss",
    definition="Asymmetric absolute loss for evaluating a predicted alpha-quantile.",
    formula="(1/n) Σ max(α(yᵢ − ŷᵢ), (α − 1)(yᵢ − ŷᵢ))",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred", "alpha"),
    references=("Koenker R, Bassett G. Regression quantiles. Econometrica. 1978;46(1):33-50.",),
    higher_is_better=False,
)
def quantile_loss(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    alpha: float = 0.5,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Pinball loss for the ``alpha`` quantile (alpha=0.5 gives half the MAE)."""
    if not (isinstance(alpha, (int, float)) and 0 < alpha < 1):
        raise InputValidationError("alpha must be strictly between 0 and 1.")
    inp = _Inputs(y_true, y_pred, sample_weight)
    diff = inp.y_true - inp.y_pred
    loss = np.maximum(alpha * diff, (alpha - 1) * diff)
    return _finish(inp.mean(loss), inp, multioutput, "quantile_loss", "Quantile loss", {"alpha": float(alpha)})


@register(
    category=_R,
    task="regression",
    name="Huber loss",
    definition="Squared error for small residuals and linear error beyond delta; robust to outliers.",
    formula="½e² if |e| ≤ δ, else δ(|e| − ½δ)",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred", "delta"),
    references=("Huber PJ. Robust estimation of a location parameter. Ann Math Stat. 1964;35(1):73-101.",),
    higher_is_better=False,
)
def huber_loss(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    delta: float = 1.0,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Mean Huber loss."""
    if not (isinstance(delta, (int, float)) and delta > 0):
        raise InputValidationError("delta must be positive.")
    inp = _Inputs(y_true, y_pred, sample_weight)
    a = np.abs(inp.error)
    loss = np.where(a <= delta, 0.5 * a**2, delta * (a - 0.5 * delta))
    return _finish(inp.mean(loss), inp, multioutput, "huber_loss", "Huber loss", {"delta": float(delta)})


def _relative(inp: _Inputs, power: int, name: str) -> FloatArray:
    num = (inp.w[:, None] * np.abs(inp.error) ** power).sum(0)
    den = (inp.w[:, None] * np.abs(inp.y_true - inp.mean(inp.y_true)) ** power).sum(0)
    if np.any(den == 0):
        raise MetricInputError(f"{name} is undefined when y_true is constant (zero baseline error).")
    ratio: FloatArray = num / den
    return ratio


@register(
    category=_R,
    task="regression",
    name="Relative absolute error",
    definition="Total absolute error relative to that of always predicting the mean; below 1 beats the mean.",
    formula="Σ|ŷᵢ − yᵢ| / Σ|yᵢ − ȳ|",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred"),
    references=("Witten IH, Frank E, Hall MA. Data Mining. 3rd ed. Morgan Kaufmann; 2011.",),
    higher_is_better=False,
)
def rae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Relative absolute error."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    return _finish(_relative(inp, 1, "RAE"), inp, multioutput, "rae", "RAE")


@register(
    category=_R,
    task="regression",
    name="Relative squared error",
    definition="Total squared error relative to that of always predicting the mean (equals 1 − R²).",
    formula="Σ(ŷᵢ − yᵢ)² / Σ(yᵢ − ȳ)²",
    range="[0, ∞)",
    input_requirements=("y_true", "y_pred"),
    references=("Witten IH, Frank E, Hall MA. Data Mining. 3rd ed. Morgan Kaufmann; 2011.",),
    higher_is_better=False,
)
def rse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Multioutput = "uniform_average",
) -> MetricResult:
    """Relative squared error."""
    inp = _Inputs(y_true, y_pred, sample_weight)
    return _finish(_relative(inp, 2, "RSE"), inp, multioutput, "rse", "RSE")
