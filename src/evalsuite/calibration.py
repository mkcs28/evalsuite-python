"""Calibration of predicted probabilities: maximum calibration error, calibration slope and intercept
(logistic recalibration) and the Hosmer–Lemeshow goodness-of-fit test.

The calibration curve, expected calibration error and Brier score live in :mod:`evalsuite.classification`
and are re-exported here for convenience.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .classification.metrics import (
    _binary_target,
    _ctx,
    brier_score,
    calibration_curve,
    expected_calibration_error,
)
from .core.exceptions import InputValidationError, StatisticalTestError, UnsupportedTaskError
from .core.export import PathLike, csv_text, html_document, html_table, save_as
from .core.registry import register
from .core.result import MetricResult, _fmt, _json_safe, _latex_escape, _latex_table
from .core.types import ArrayLike
from .stats.results import TestResult

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "CalibrationReport",
    "brier_score",
    "calibration_report",
    "calibration_curve",
    "calibration_intercept",
    "calibration_slope",
    "expected_calibration_error",
    "hosmer_lemeshow",
    "maximum_calibration_error",
]

_C = "calibration"
_EPS = 1e-15
_REF_VC = (
    "Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of "
    "predictive analytics. BMC Med. 2019;17(1):230."
)
_REF_COX = "Cox DR. Two further applications of a model for binary regression. Biometrika. 1958;45(3-4):562-565."
_REF_HL = (
    "Hosmer DW, Lemeshow S. Goodness of fit tests for the multiple logistic regression model. "
    "Commun Stat Theory Methods. 1980;9(10):1043-1069."
)
_REF_NAEINI = (
    "Naeini MP, Cooper GF, Hauskrecht M. Obtaining well calibrated probabilities using Bayesian binning. "
    "AAAI 2015:2901-2907."
)


def _binary(
    y_true: ArrayLike, y_prob: ArrayLike, pos_label: Any, sample_weight: Optional[ArrayLike]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    ctx = _ctx(y_true, None, y_prob=y_prob, sample_weight=sample_weight)
    if ctx.target_type != "binary":
        raise UnsupportedTaskError("Calibration measures here are for binary outcomes; use one-vs-rest per class.")
    y = _binary_target(ctx, pos_label)
    return y, np.asarray(ctx.y_prob, dtype=np.float64), np.asarray(ctx.weights, dtype=np.float64)


@register(
    category=_C,
    task="binary",
    name="Maximum calibration error",
    definition="Largest gap between observed frequency and mean predicted probability over probability bins.",
    formula="MCE = max_b |acc_b − conf_b|",
    range="[0, 1]",
    input_requirements=("y_true", "y_prob"),
    references=(_REF_NAEINI,),
    higher_is_better=False,
)
def maximum_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """Maximum calibration error over the non-empty bins of :func:`calibration_curve`."""
    prob_true, prob_pred, _ = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy, pos_label=pos_label, sample_weight=sample_weight
    )
    return MetricResult(
        "maximum_calibration_error",
        "MCE",
        float(np.max(np.abs(prob_true - prob_pred))),
        {"n_bins": int(n_bins), "strategy": strategy},
    )


def _logit(p: NDArray[np.float64]) -> NDArray[np.float64]:
    q = np.clip(p, _EPS, 1 - _EPS)
    return np.asarray(np.log(q / (1 - q)), dtype=np.float64)


def logistic_fit(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    w: NDArray[np.float64],
    *,
    offset: bool,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> NDArray[np.float64]:
    """Weighted maximum-likelihood logistic regression by Newton–Raphson.

    ``offset=True`` fits ``logit P(y=1) = a + x`` (returns ``[a]``); otherwise ``logit P(y=1) = a + b·x``
    (returns ``[a, b]``).
    """
    design = np.ones((x.shape[0], 1)) if offset else np.column_stack([np.ones_like(x), x])
    off = x if offset else np.zeros_like(x)
    beta = np.zeros(design.shape[1])
    for _ in range(max_iter):
        eta = design @ beta + off
        mu = 1 / (1 + np.exp(-eta))
        grad = design.T @ (w * (y - mu))
        hess = (design * (w * mu * (1 - mu))[:, None]).T @ design
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            raise StatisticalTestError(
                "The logistic recalibration model could not be fitted (singular information matrix); "
                "the predictions may be constant or the outcome may be perfectly separated."
            ) from None
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            return beta
    raise StatisticalTestError(
        "The logistic recalibration model did not converge; the outcome may be perfectly separated by the "
        "predictions."
    )


def _need_both(y: NDArray[np.float64], what: str) -> None:
    if np.min(y) == np.max(y):
        raise InputValidationError(f"{what} needs both outcomes (0 and 1) in y_true.")


@register(
    category=_C,
    task="binary",
    name="Calibration slope",
    definition="Slope b of the logistic recalibration model logit P(y=1) = a + b · logit(p). 1 is ideal; below "
    "1 means predictions are too extreme (overfitting), above 1 too moderate.",
    formula="logit P(y=1) = a + b · logit(p̂)",
    range="(−∞, ∞), ideal 1",
    input_requirements=("y_true", "y_prob"),
    references=(_REF_COX, _REF_VC),
    higher_is_better=None,
)
def calibration_slope(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """Calibration slope (Cox 1958). Probabilities are clipped to [1e-15, 1 − 1e-15] before the logit."""
    y, p, w = _binary(y_true, y_prob, pos_label, sample_weight)
    _need_both(y, "The calibration slope")
    a, b = logistic_fit(y, _logit(p), w, offset=False)
    return MetricResult("calibration_slope", "Calibration slope", float(b), {"intercept": float(a)})


@register(
    category=_C,
    task="binary",
    name="Calibration intercept",
    definition="Calibration-in-the-large: intercept a of logit P(y=1) = a + logit(p) with the slope fixed at 1. "
    "0 is ideal; negative means risks are overestimated on average, positive underestimated.",
    formula="logit P(y=1) = a + logit(p̂)  (offset)",
    range="(−∞, ∞), ideal 0",
    input_requirements=("y_true", "y_prob"),
    references=(_REF_COX, _REF_VC),
    higher_is_better=None,
)
def calibration_intercept(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> MetricResult:
    """Calibration intercept (calibration-in-the-large) with the logit of the predictions as an offset."""
    y, p, w = _binary(y_true, y_prob, pos_label, sample_weight)
    _need_both(y, "The calibration intercept")
    (a,) = logistic_fit(y, _logit(p), w, offset=True)
    return MetricResult("calibration_intercept", "Calibration intercept", float(a))


def hosmer_lemeshow(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_groups: int = 10,
    pos_label: Any = None,
) -> TestResult:
    """Hosmer–Lemeshow goodness-of-fit test for predicted risks.

    Observations are sorted by predicted risk and split into ``n_groups`` groups of (nearly) equal size
    ("deciles of risk"). The statistic Σ (O − E)² / (E (1 − E/nₖ)) is compared with χ² on ``n_groups − 2``
    degrees of freedom (use ``n_groups`` degrees of freedom for an external validation set by reading
    ``params["statistic"]``). A large p-value does not show that calibration is good, especially in small
    samples; report the calibration curve, slope and intercept as well.
    """
    y, p, _ = _binary(y_true, y_prob, pos_label, None)
    if not (isinstance(n_groups, (int, np.integer)) and n_groups >= 3):
        raise InputValidationError("n_groups must be an integer of at least 3.")
    if y.shape[0] < n_groups:
        raise InputValidationError(f"Need at least n_groups={n_groups} observations; got {y.shape[0]}.")
    order = np.argsort(p, kind="mergesort")
    groups = np.array_split(order, n_groups)
    observed = np.array([y[g].sum() for g in groups])
    expected = np.array([p[g].sum() for g in groups])
    sizes = np.array([g.shape[0] for g in groups], dtype=np.float64)
    var = expected * (1 - expected / sizes)
    if np.any(var <= 0):
        raise StatisticalTestError(
            "Hosmer–Lemeshow is undefined: a risk group has every predicted risk at 0 or 1. Use fewer groups."
        )
    stat = float(np.sum((observed - expected) ** 2 / var))
    df = n_groups - 2
    return TestResult(
        "hosmer-lemeshow",
        stat,
        float(stats.chi2.sf(stat, df)),
        params={
            "df": df,
            "n_groups": int(n_groups),
            "observed": observed.tolist(),
            "expected": expected.tolist(),
            "group_sizes": sizes.astype(int).tolist(),
        },
    )


_REPORT_ROWS = (
    ("brier_score", "Brier score", "0"),
    ("expected_calibration_error", "ECE", "0"),
    ("maximum_calibration_error", "MCE", "0"),
    ("calibration_intercept", "Calibration intercept", "0"),
    ("calibration_slope", "Calibration slope", "1"),
    ("hosmer_lemeshow_statistic", "Hosmer–Lemeshow χ²", "–"),
    ("hosmer_lemeshow_p", "Hosmer–Lemeshow p", "–"),
)


@dataclass(frozen=True, eq=False)
class CalibrationReport:
    """Calibration of one model's predicted risks: Brier score, ECE, MCE, intercept, slope and Hosmer–Lemeshow,
    plus the binned calibration curve (``curve``: prob_true, prob_pred, bin_weight)."""

    values: Any
    curve: Any
    params: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __getitem__(self, name: str) -> float:
        return float(self.values[name])

    def _rows(self, digits: int) -> list[list[str]]:
        return [[label, _fmt(self.values[k], digits), ideal] for k, label, ideal in _REPORT_ROWS]

    def summary(self, *, digits: int = 4) -> str:
        p = self.params
        lines = [
            f"EvalSuite calibration (n={p['n']}, {p['n_bins']} {p['strategy']} bins, HL {p['n_groups']} groups)"
        ]
        width = max(len(label) for _, label, _ in _REPORT_ROWS)
        for k, label, ideal in _REPORT_ROWS:
            lines.append(f"  {label:<{width}}  {_fmt(self.values[k], digits)}   (ideal {ideal})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        prob_true, prob_pred, weight = self.curve
        return cast(
            "dict[str, Any]",
            _json_safe(
                {
                    "values": dict(self.values),
                    "curve": {"prob_true": prob_true, "prob_pred": prob_pred, "bin_weight": weight},
                    "params": dict(self.params),
                }
            ),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        frame: pd.DataFrame = pd.DataFrame({"value": dict(self.values)})
        return frame

    def to_markdown(self, *, digits: int = 4) -> str:
        lines = ["| Measure | Value | Ideal |", "| --- | ---: | ---: |"]
        lines += ["| " + " | ".join(r) + " |" for r in self._rows(digits)]
        return "\n".join(lines)

    def to_latex(self, *, digits: int = 4, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        rows = [[_latex_escape(c) for c in r] for r in self._rows(digits)]
        return _latex_table(["Measure", "Value", "Ideal"], rows, caption, label)

    def to_csv(self, path: Optional[PathLike] = None) -> str:
        text = csv_text(["measure", "value"], [[k, float(v)] for k, v in self.values.items()])
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def to_html(self, *, digits: int = 4, full: bool = True) -> str:
        body = html_table(["Measure", "Value", "Ideal"], self._rows(digits), caption="Calibration")
        prob_true, prob_pred, weight = self.curve
        body += "\n" + html_table(
            ["Mean predicted", "Observed", "Weight"],
            [[_fmt(a, digits), _fmt(b, digits), f"{w:g}"] for a, b, w in zip(prob_pred, prob_true, weight)],
            caption="Calibration curve",
            numeric=[True, True, True],
        )
        return html_document("EvalSuite calibration", body) if full else body

    def save(self, path: PathLike) -> str:
        return save_as(
            path,
            {
                "json": self.to_json,
                "csv": self.to_csv,
                "markdown": self.to_markdown,
                "latex": self.to_latex,
                "html": self.to_html,
                "text": self.summary,
            },
        )


def calibration_report(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    n_groups: int = 10,
    pos_label: Any = None,
) -> CalibrationReport:
    """Every calibration measure for binary predicted risks in one object (see :class:`CalibrationReport`)."""
    kw: dict[str, Any] = {"n_bins": n_bins, "strategy": strategy, "pos_label": pos_label}
    hl = hosmer_lemeshow(y_true, y_prob, n_groups=n_groups, pos_label=pos_label)
    values = {
        "brier_score": float(brier_score(y_true, y_prob, pos_label=pos_label)),
        "expected_calibration_error": float(expected_calibration_error(y_true, y_prob, **kw)),
        "maximum_calibration_error": float(maximum_calibration_error(y_true, y_prob, **kw)),
        "calibration_intercept": float(calibration_intercept(y_true, y_prob, pos_label=pos_label)),
        "calibration_slope": float(calibration_slope(y_true, y_prob, pos_label=pos_label)),
        "hosmer_lemeshow_statistic": hl.statistic,
        "hosmer_lemeshow_p": hl.p_value,
    }
    curve = calibration_curve(y_true, y_prob, **kw)
    n = int(np.asarray(y_true).shape[0])
    return CalibrationReport(
        values,
        curve,
        {"n": n, "n_bins": n_bins, "strategy": strategy, "n_groups": n_groups, "hl_df": hl.params["df"]},
    )
