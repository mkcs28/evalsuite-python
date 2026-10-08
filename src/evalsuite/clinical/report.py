"""Diagnostic accuracy summary with confidence intervals, and decision curve analysis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy import special

from ..core.exceptions import InputValidationError
from ..core.export import PathLike, csv_text, html_document, html_table, save_as
from ..core.result import _fmt, _json_safe, _latex_escape, _latex_table
from ..core.types import ArrayLike
from ..stats.intervals import _check_level, proportion_ci
from ..stats.results import ConfidenceInterval
from .metrics import _binary_risk, _check_thresholds, _need_both_classes, binary_counts, net_benefit_curve

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["DecisionCurve", "DiagnosticReport", "decision_curve", "diagnostic_report"]

_ROWS = (
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("ppv", "PPV"),
    ("npv", "NPV"),
    ("lr_positive", "LR+"),
    ("lr_negative", "LR−"),
    ("diagnostic_odds_ratio", "Diagnostic odds ratio"),
    ("youden_j", "Youden's J"),
    ("accuracy", "Accuracy"),
    ("prevalence", "Prevalence"),
)


@dataclass(frozen=True, eq=False)
class DiagnosticReport:
    """Every diagnostic accuracy measure for one binary test, each with a confidence interval.

    Index it by name: ``report["lr_positive"]`` is a :class:`ConfidenceInterval`. ``counts`` holds the 2×2
    table (``tp``, ``fp``, ``fn``, ``tn``).
    """

    estimates: Any
    counts: Any
    level: float
    params: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimates", MappingProxyType(dict(self.estimates)))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __getitem__(self, name: str) -> ConfidenceInterval:
        try:
            return cast("ConfidenceInterval", self.estimates[name])
        except KeyError:
            raise KeyError(f"No measure {name!r}. Available: {', '.join(self.estimates)}.") from None

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.estimates)

    def _rows(self, digits: int) -> list[list[str]]:
        out = []
        for key, label in _ROWS:
            ci = self.estimates[key]
            out.append([label, _fmt(ci.estimate, digits), _fmt(ci.low, digits), _fmt(ci.high, digits), ci.method])
        return out

    def _header(self) -> list[str]:
        pct = f"{round(self.level * 100, 6):g}%"
        return ["Measure", "Estimate", f"{pct} CI low", f"{pct} CI high", "Method"]

    def summary(self, *, digits: int = 3) -> str:
        c = self.counts
        lines = [
            f"EvalSuite diagnostic accuracy (n={int(c['tp'] + c['fp'] + c['fn'] + c['tn'])}, "
            f"{round(self.level * 100, 6):g}% CIs)",
            f"  2×2 table: TP={c['tp']:g}  FP={c['fp']:g}  FN={c['fn']:g}  TN={c['tn']:g}",
        ]
        width = max(len(label) for _, label in _ROWS)
        for key, label in _ROWS:
            ci = self.estimates[key]
            lines.append(
                f"  {label:<{width}}  {_fmt(ci.estimate, digits)}  "
                f"({_fmt(ci.low, digits)}–{_fmt(ci.high, digits)})  {ci.method}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            _json_safe(
                {
                    "counts": dict(self.counts),
                    "level": self.level,
                    "measures": {k: v.to_dict() for k, v in self.estimates.items()},
                    "params": dict(self.params),
                }
            ),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        rows = [
            {"measure": k, "estimate": v.estimate, "low": v.low, "high": v.high, "method": v.method}
            for k, v in self.estimates.items()
        ]
        frame: pd.DataFrame = pd.DataFrame(rows).set_index("measure")
        return frame

    def to_markdown(self, *, digits: int = 3) -> str:
        header = self._header()
        lines = ["| " + " | ".join(header) + " |", "| --- | ---: | ---: | ---: | --- |"]
        lines += ["| " + " | ".join(r) + " |" for r in self._rows(digits)]
        return "\n".join(lines)

    def to_latex(self, *, digits: int = 3, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        rows = [[_latex_escape(c) for c in r] for r in self._rows(digits)]
        return _latex_table([_latex_escape(h) for h in self._header()], rows, caption, label)

    def to_csv(self, path: Optional[PathLike] = None) -> str:
        rows = [[k, v.estimate, v.low, v.high, v.level, v.method] for k, v in self.estimates.items()]
        text = csv_text(["measure", "estimate", "low", "high", "level", "method"], rows)
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def to_html(self, *, digits: int = 3, full: bool = True) -> str:
        c = self.counts
        body = html_table(self._header(), self._rows(digits), caption="Diagnostic accuracy")
        body += "\n" + html_table(
            ["", "Condition present", "Condition absent"],
            [["Test positive", f"{c['tp']:g}", f"{c['fp']:g}"], ["Test negative", f"{c['fn']:g}", f"{c['tn']:g}"]],
            caption="2×2 table",
        )
        return html_document("EvalSuite diagnostic accuracy", body) if full else body

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


def _log_ratio_ci(est: float, se: float, z: float, level: float, method: str, name: str) -> ConfidenceInterval:
    if not (math.isfinite(est) and est > 0 and math.isfinite(se)):
        return ConfidenceInterval(est, math.nan, math.nan, level, method, name)
    lo, hi = math.exp(math.log(est) - z * se), math.exp(math.log(est) + z * se)
    return ConfidenceInterval(est, lo, hi, level, method, name, {"se_log": se})


def diagnostic_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = None,
    level: float = 0.95,
    proportion_method: str = "wilson",
) -> DiagnosticReport:
    """Sensitivity, specificity, PPV, NPV, likelihood ratios, diagnostic odds ratio, Youden's J, accuracy and
    prevalence for a binary test, each with a confidence interval.

    Intervals: proportions use ``proportion_method`` (Wilson by default, or ``"clopper-pearson"``); likelihood
    ratios use the log method of Simel et al. (1991); the diagnostic odds ratio uses Woolf's log method with
    0.5 added to every cell when one is zero; Youden's J uses the Wald interval from the independent
    sensitivity and specificity variances, clipped to [−1, 1].
    """
    level = _check_level(level)
    tp, fp, fn, tn = binary_counts(y_true, y_pred, pos_label=pos_label)
    _need_both_classes(tp, fp, fn, tn, "A diagnostic report")
    a, b, c, d = (round(v) for v in (tp, fp, fn, tn))
    z = float(special.ndtri(1 - (1 - level) / 2))
    method: Any = proportion_method

    def prop(k: int, n: int, name: str) -> ConfidenceInterval:
        if n == 0:
            return ConfidenceInterval(math.nan, math.nan, math.nan, level, method, name)
        ci = proportion_ci(k, n, level=level, method=method)
        return ConfidenceInterval(ci.estimate, ci.low, ci.high, level, ci.method, name, ci.params)

    n1, n0, n = a + c, b + d, a + b + c + d
    sens, spec = a / n1, d / n0
    est: dict[str, ConfidenceInterval] = {
        "sensitivity": prop(a, n1, "sensitivity"),
        "specificity": prop(d, n0, "specificity"),
        "ppv": prop(a, a + b, "ppv"),
        "npv": prop(d, c + d, "npv"),
    }
    lr_pos = sens / (1 - spec) if spec < 1 else (math.inf if sens > 0 else math.nan)
    lr_neg = (1 - sens) / spec if spec > 0 else (math.inf if sens < 1 else math.nan)
    se_pos = math.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n0) if a and b else math.nan
    se_neg = math.sqrt(1 / c - 1 / n1 + 1 / d - 1 / n0) if c and d else math.nan
    est["lr_positive"] = _log_ratio_ci(lr_pos, se_pos, z, level, "log (Simel 1991)", "lr_positive")
    est["lr_negative"] = _log_ratio_ci(lr_neg, se_neg, z, level, "log (Simel 1991)", "lr_negative")
    corrected = min(a, b, c, d) == 0
    ca, cb, cc, cd = (v + 0.5 for v in (a, b, c, d)) if corrected else (a, b, c, d)
    dor = ca * cd / (cb * cc)
    se_dor = math.sqrt(1 / ca + 1 / cb + 1 / cc + 1 / cd)
    est["diagnostic_odds_ratio"] = _log_ratio_ci(
        dor, se_dor, z, level, "log (Woolf" + (", +0.5 cells)" if corrected else ")"), "diagnostic_odds_ratio"
    )
    j = sens + spec - 1
    se_j = math.sqrt(sens * (1 - sens) / n1 + spec * (1 - spec) / n0)
    est["youden_j"] = ConfidenceInterval(
        j, max(-1.0, j - z * se_j), min(1.0, j + z * se_j), level, "wald", "youden_j", {"se": se_j}
    )
    est["accuracy"] = prop(a + d, n, "accuracy")
    est["prevalence"] = prop(n1, n, "prevalence")
    return DiagnosticReport(
        est,
        {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        level,
        {"pos_label": 1 if pos_label is None else pos_label, "proportion_method": proportion_method},
    )


@dataclass(frozen=True, eq=False)
class DecisionCurve:
    """Net benefit of one or more models, of treating everyone and of treating no one, per threshold."""

    thresholds: NDArray[np.float64]
    net_benefit: Any  # model name -> array
    treat_all: NDArray[np.float64]
    prevalence: float
    n: int

    def __post_init__(self) -> None:
        for name in ("thresholds", "treat_all"):
            arr = np.array(getattr(self, name), dtype=np.float64)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        models = {}
        for k, v in dict(self.net_benefit).items():
            arr = np.array(v, dtype=np.float64)
            arr.setflags(write=False)
            models[k] = arr
        object.__setattr__(self, "net_benefit", MappingProxyType(models))

    @property
    def treat_none(self) -> NDArray[np.float64]:
        return np.zeros_like(self.thresholds)

    def useful_range(self, model: Optional[str] = None) -> list[tuple[float, float]]:
        """Threshold intervals where the model beats both treating everyone and treating no one."""
        name = model if model is not None else next(iter(self.net_benefit))
        nb = self.net_benefit[name]
        better = (nb > self.treat_all) & (nb > 0)
        ranges: list[tuple[float, float]] = []
        start: Optional[int] = None
        for i, ok in enumerate(better):
            if ok and start is None:
                start = i
            if not ok and start is not None:
                ranges.append((float(self.thresholds[start]), float(self.thresholds[i - 1])))
                start = None
        if start is not None:
            ranges.append((float(self.thresholds[start]), float(self.thresholds[-1])))
        return ranges

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        data: dict[str, Any] = {"threshold": self.thresholds}
        for k, v in self.net_benefit.items():
            data[f"net_benefit[{k}]"] = v
        data["treat_all"] = self.treat_all
        data["treat_none"] = self.treat_none
        frame: pd.DataFrame = pd.DataFrame(data)
        return frame

    def to_dict(self) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            _json_safe(
                {
                    "thresholds": self.thresholds,
                    "net_benefit": dict(self.net_benefit),
                    "treat_all": self.treat_all,
                    "prevalence": self.prevalence,
                    "n": self.n,
                }
            ),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_csv(self, path: Optional[PathLike] = None) -> str:
        names = list(self.net_benefit)
        rows = [
            [float(t), *(float(self.net_benefit[k][i]) for k in names), float(self.treat_all[i]), 0.0]
            for i, t in enumerate(self.thresholds)
        ]
        text = csv_text(["threshold", *names, "treat_all", "treat_none"], rows)
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def __repr__(self) -> str:
        return (
            f"DecisionCurve(models={list(self.net_benefit)}, thresholds={self.thresholds[0]:g}..."
            f"{self.thresholds[-1]:g} ({self.thresholds.size}), prevalence={self.prevalence:.3f}, n={self.n})"
        )


def decision_curve(
    y_true: ArrayLike,
    y_prob: Any,
    *,
    thresholds: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
) -> DecisionCurve:
    """Decision curve analysis (Vickers & Elkin 2006).

    ``y_prob`` is one array of predicted risks, or a dict ``{"model name": risks}`` to compare models.
    ``thresholds`` defaults to 0.01, 0.02, …, 0.99. Treat-all net benefit is prevalence − (1 − prevalence) ·
    pₜ / (1 − pₜ); treat-none is 0.
    """
    t = _check_thresholds(np.arange(1, 100) / 100 if thresholds is None else thresholds)
    if not np.all(np.diff(t) > 0):
        raise InputValidationError("thresholds must be strictly increasing.")
    models = dict(y_prob) if isinstance(y_prob, dict) else {"model": y_prob}
    if not models:
        raise InputValidationError("Give at least one model's predicted risks.")
    curves: dict[str, NDArray[np.float64]] = {}
    treat_all: Optional[NDArray[np.float64]] = None
    prevalence, n = 0.0, 0
    for name, probs in models.items():
        y, p, w = _binary_risk(y_true, probs, pos_label, sample_weight)
        model, all_ = net_benefit_curve(y, p, w, t)
        curves[str(name)] = model
        treat_all = all_
        prevalence, n = float((w * y).sum() / w.sum()), int(y.shape[0])
    if treat_all is None:  # pragma: no cover - models is non-empty
        raise InputValidationError("Give at least one model's predicted risks.")
    return DecisionCurve(t, curves, treat_all, prevalence, n)
