"""compare(): evaluate several models on the same test set with intervals and paired tests."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import metric_info
from ..core.result import _json_safe, _latex_escape, _latex_table
from ..core.types import ArrayLike
from ._resolve import MetricCall, as_observations, is_categorical, resolve_metric
from .effect import adjust_pvalues
from .intervals import _check_level, _check_resamples, bootstrap_distribution, resample_indices
from .paired import delong_test, mcnemar_test

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["ComparisonResult", "compare"]

_DEFAULT_CLF = ["accuracy", "f1", "mcc"]
_DEFAULT_CLF_PROB = ["roc_auc", "brier_score"]
_DEFAULT_REG = ["mae", "rmse", "r2"]


def _p(value: float) -> str:
    return "<0.0001" if value < 1e-4 else f"={value:.4f}"


def _info(name: str) -> Any:
    for cat in ("classification", "regression"):
        try:
            return metric_info(f"{cat}.{name}")
        except KeyError:
            continue
    return None


def _needs_prob(name: str) -> bool:
    info = _info(name)
    return info is not None and "y_prob" in info.input_requirements


@dataclass(frozen=True, eq=False)
class ComparisonResult:
    """Per-model estimates with confidence intervals, and pairwise tests with adjusted p-values."""

    models: tuple[str, ...]
    metrics: tuple[str, ...]
    estimates: tuple[Mapping[str, Any], ...]
    tests: tuple[Mapping[str, Any], ...]
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimates", tuple(MappingProxyType(dict(r)) for r in self.estimates))
        object.__setattr__(self, "tests", tuple(MappingProxyType(dict(r)) for r in self.tests))
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))

    def estimate(self, model: str, metric: str) -> Mapping[str, Any]:
        for row in self.estimates:
            if row["model"] == model and row["metric"] == metric:
                return row
        raise KeyError(f"No estimate for model={model!r}, metric={metric!r}.")

    def best(self, metric: str) -> str:
        """Model with the best point estimate (direction from the metric registry)."""
        rows = [r for r in self.estimates if r["metric"] == metric]
        if not rows:
            raise KeyError(f"Metric {metric!r} was not compared. Compared: {', '.join(self.metrics)}.")
        info = _info(metric)
        higher = True if info is None or info.higher_is_better is None else info.higher_is_better
        pick = max if higher else min
        return str(pick(rows, key=lambda r: r["estimate"])["model"])

    def summary(self, *, digits: int = 3) -> str:
        level = round(self.settings["level"] * 100, 6)
        lines = [
            f"EvalSuite model comparison (n={self.settings['n_samples']}, {level:g}% CIs, "
            f"{self.settings['n_resamples']} paired bootstrap resamples, {self.settings['correction']} correction)"
        ]
        width = max(len(m) for m in self.models)
        for metric in self.metrics:
            lines.append(f"\n{metric}  (best: {self.best(metric)})")
            for row in (r for r in self.estimates if r["metric"] == metric):
                lines.append(
                    f"  {row['model']:<{width}}  {row['estimate']:.{digits}f} "
                    f"({row['low']:.{digits}f}–{row['high']:.{digits}f})"
                )
            for t in (t for t in self.tests if t["metric"] == metric):
                mark = " *" if t["significant"] else ""
                lines.append(
                    f"  {t['model_a']} − {t['model_b']}: {t['difference']:+.{digits}f} "
                    f"[{t['test']}, p={t['p_value']:.4f}, adjusted p={t['p_adjusted']:.4f}]{mark}"
                )
        lines.append(f"\n* significant at α = {self.settings['alpha']} after correction")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            _json_safe(
                {
                    "models": list(self.models),
                    "metrics": list(self.metrics),
                    "estimates": [dict(r) for r in self.estimates],
                    "tests": [dict(r) for r in self.tests],
                    "settings": dict(self.settings),
                }
            ),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_dataframe(self, which: str = "estimates") -> pd.DataFrame:
        """``"estimates"`` (one row per model and metric) or ``"tests"`` (one row per comparison)."""
        import pandas as pd

        if which not in ("estimates", "tests"):
            raise InputValidationError("which must be 'estimates' or 'tests'.")
        frame: pd.DataFrame = pd.DataFrame(
            [dict(r) for r in (self.estimates if which == "estimates" else self.tests)]
        )
        return frame

    def _cell(self, model: str, metric: str, digits: int) -> str:
        r = self.estimate(model, metric)
        return f"{r['estimate']:.{digits}f} ({r['low']:.{digits}f}–{r['high']:.{digits}f})"

    def to_markdown(self, *, digits: int = 3) -> str:
        """Models as rows, metrics as columns: ``estimate (low–high)``; best value per metric in bold."""
        header = "| Model | " + " | ".join(self.metrics) + " |"
        sep = "| --- |" + " ---: |" * len(self.metrics)
        rows = []
        for model in self.models:
            cells = []
            for metric in self.metrics:
                c = self._cell(model, metric, digits)
                cells.append(f"**{c}**" if self.best(metric) == model else c)
            rows.append(f"| {model} | " + " | ".join(cells) + " |")
        return "\n".join([header, sep, *rows])

    def to_latex(self, *, digits: int = 3, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        """Publication table: models × metrics with confidence intervals; best value per metric in bold."""
        header = ["Model", *(_latex_escape(m) for m in self.metrics)]
        rows = []
        for model in self.models:
            cells = [_latex_escape(model)]
            for metric in self.metrics:
                c = self._cell(model, metric, digits).replace("–", "--")
                cells.append(rf"\textbf{{{c}}}" if self.best(metric) == model else c)
            rows.append(cells)
        level = round(self.settings["level"] * 100, 6)
        cap = caption or f"Model comparison: estimate ({level:g}% CI)."  # escaped once by _latex_table
        return _latex_table(header, rows, caption=cap, label=label)

    def _csv_rows(self) -> tuple[list[str], list[list[Any]]]:
        header = ["model", "metric", "estimate", "low", "high"]
        return header, [[r["model"], r["metric"], r["estimate"], r["low"], r["high"]] for r in self.estimates]

    def to_csv(self, path: Optional[str] = None, *, which: str = "estimates") -> str:
        """CSV of the estimates (``which="estimates"``) or the pairwise tests (``which="tests"``)."""
        from ..core.export import csv_text

        if which == "estimates":
            header, rows = self._csv_rows()
        elif which == "tests":
            header = [
                "metric",
                "model_a",
                "model_b",
                "difference",
                "ci_low",
                "ci_high",
                "test",
                "statistic",
                "p_value",
                "p_adjusted",
                "significant",
            ]
            rows = [[t[h] for h in header] for t in self.tests]
        else:
            raise InputValidationError("which must be 'estimates' or 'tests'.")
        text = csv_text(header, rows)
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def to_html(self, *, digits: int = 3, full: bool = False) -> str:
        """Estimates table (best value per metric in bold) and the pairwise tests table."""
        from ..core.export import html_document, html_table

        rows = [[m, *(self._cell(m, metric, digits) for metric in self.metrics)] for m in self.models]
        bold = {
            (i, j + 1)
            for i, m in enumerate(self.models)
            for j, metric in enumerate(self.metrics)
            if self.best(metric) == m
        }
        level = round(self.settings["level"] * 100, 6)
        parts = [
            html_table(
                ["Model", *self.metrics], rows, caption=f"Estimate ({level:g}% CI); best in bold", bold=bold
            )
        ]
        test_rows = [
            [
                t["metric"],
                f"{t['model_a']} − {t['model_b']}",
                f"{t['difference']:+.{digits}f}",
                f"{t['ci_low']:.{digits}f}–{t['ci_high']:.{digits}f}",
                t["test"],
                f"{t['p_value']:.4g}",
                f"{t['p_adjusted']:.4g}",
                "yes" if t["significant"] else "no",
            ]
            for t in self.tests
        ]
        parts.append(
            html_table(
                ["Metric", "Comparison", "Difference", f"{level:g}% CI", "Test", "p", "Adjusted p", "Significant"],
                test_rows,
                caption=f"Pairwise tests ({self.settings['correction']} correction, α = {self.settings['alpha']})",
                numeric=[False, False, True, True, False, True, True, False],
            )
        )
        body = "\n".join(parts)
        meta = f"n = {self.settings['n_samples']}, {self.settings['n_resamples']} paired bootstrap resamples"
        return html_document("EvalSuite model comparison", body, meta) if full else body

    def save(self, path: str, *, digits: int = 3) -> str:
        """Save as .json .csv .md .tex .html or .txt (chosen by the extension)."""
        from ..core.export import save_as

        return save_as(
            path,
            {
                "json": self.to_json,
                "csv": self.to_csv,
                "markdown": lambda: self.to_markdown(digits=digits) + "\n",
                "latex": lambda: self.to_latex(digits=digits) + "\n",
                "html": lambda: self.to_html(digits=digits, full=True),
                "text": lambda: self.summary(digits=digits) + "\n",
            },
        )


def compare(
    y_true: ArrayLike,
    predictions: Optional[Mapping[str, ArrayLike]] = None,
    *,
    probabilities: Optional[Mapping[str, ArrayLike]] = None,
    metrics: Optional[Sequence[str]] = None,
    baseline: Optional[str] = None,
    level: float = 0.95,
    alpha: float = 0.05,
    n_resamples: int = 1000,
    correction: str = "holm",
    random_state: Optional[int] = None,
    stratify: Optional[bool] = None,
    sample_weight: Optional[ArrayLike] = None,
    metric_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> ComparisonResult:
    """Compare models evaluated on the same test set.

    ``predictions`` and/or ``probabilities`` map model names to that model's predicted labels/values or
    probabilities. Every model is resampled on the same bootstrap indices (paired), giving a percentile
    confidence interval per model and metric and a paired test per pair of models:

    * accuracy: McNemar's test (exact when there are fewer than 25 discordant pairs);
    * binary ROC AUC: DeLong's test;
    * everything else: paired bootstrap test.

    Pairs are every model against ``baseline`` if given, otherwise all pairs. p-values are adjusted within
    each metric (``correction``: "holm", "bonferroni", "hochberg", "bh", "by"); ``significant`` uses the adjusted
    p-value and ``alpha``. ``metric_kwargs`` passes options per metric, e.g. ``{"f1": {"average": "macro"}}``.
    """
    level = _check_level(level)
    n_resamples = _check_resamples(n_resamples)
    predictions = dict(predictions or {})
    probabilities = dict(probabilities or {})
    names = list(dict.fromkeys([*predictions, *probabilities]))
    if len(names) < 2:
        raise InputValidationError("compare() needs at least two models (a dict of name -> predictions).")
    if baseline is not None and baseline not in names:
        raise InputValidationError(f"baseline={baseline!r} is not one of the models: {', '.join(names)}.")
    yt = as_observations(y_true, "y_true")
    categorical = is_categorical(yt)
    if metrics is None:
        if yt.ndim > 2 or (yt.dtype == object and yt.size and not isinstance(yt[0], (str, bytes))):
            is_detection = yt.dtype == object and isinstance(yt[0], Mapping)
            metric_names = ["mean_average_precision"] if is_detection else ["dice", "iou"]
        elif categorical:
            metric_names = list(_DEFAULT_CLF) if predictions else []
            if probabilities:
                metric_names += _DEFAULT_CLF_PROB if np.unique(yt).shape[0] <= 2 and yt.ndim == 1 else ["roc_auc"]
        else:
            metric_names = list(_DEFAULT_REG)
    else:
        metric_names = [resolve_metric(m)[1] for m in metrics]
    kwargs_by_metric = {k: dict(v) for k, v in (metric_kwargs or {}).items()}

    calls: dict[tuple[str, str], MetricCall] = {}
    for metric in metric_names:
        fn, _ = resolve_metric(metric)
        prob = _needs_prob(metric)
        source = probabilities if prob else predictions
        for model in names:
            if model not in source:
                kind = "probabilities" if prob else "predictions"
                raise InputValidationError(f"Metric '{metric}' needs {kind} for model '{model}'.")
            calls[(model, metric)] = MetricCall(
                fn,
                yt,
                None if prob else source[model],
                source[model] if prob else None,
                sample_weight,
                kwargs_by_metric.get(metric),
            )
        # identical label set for every model of this metric
        labs = [calls[(m, metric)].kwargs.get("labels") for m in names]
        if all(lab is not None for lab in labs):
            shared = labs[0]
            for lab in labs[1:]:
                shared = np.union1d(shared, lab)
            for m in names:
                calls[(m, metric)].kwargs["labels"] = shared

    rng = np.random.default_rng(random_state)
    use_strata = categorical if stratify is None else bool(stratify)
    idx = resample_indices(yt.shape[0], n_resamples, rng, yt if use_strata else None)
    a = (1 - level) / 2
    boot: dict[tuple[str, str], np.ndarray] = {}
    estimates: list[dict[str, Any]] = []
    for metric in metric_names:
        for model in names:
            call = calls[(model, metric)]
            dist, failed = bootstrap_distribution(call, idx)
            boot[(model, metric)] = dist
            low, high = np.nanquantile(dist, [a, 1 - a])
            estimates.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": call(),
                    "low": float(low),
                    "high": float(high),
                    "failed_resamples": int(failed),
                }
            )

    pairs = (
        [(m, baseline) for m in names if m != baseline]
        if baseline is not None
        else list(itertools.combinations(names, 2))
    )
    tests: list[dict[str, Any]] = []
    for metric in metric_names:
        block: list[dict[str, Any]] = []
        for ma, mb in pairs:
            diff_dist = boot[(ma, metric)] - boot[(mb, metric)]
            diff_dist = diff_dist[~np.isnan(diff_dist)]
            observed = calls[(ma, metric)]() - calls[(mb, metric)]()
            low, high = np.quantile(diff_dist, [a, 1 - a])
            if metric == "accuracy" and sample_weight is None:
                t = mcnemar_test(yt, predictions[ma], predictions[mb])
                test_name, stat, p = t.test, t.statistic, t.p_value
            elif metric == "roc_auc" and sample_weight is None and yt.ndim == 1 and np.unique(yt).shape[0] == 2:
                t = delong_test(yt, probabilities[ma], probabilities[mb])
                test_name, stat, p = t.test, t.statistic, t.p_value
            else:
                extreme = np.sum(np.abs(diff_dist - observed) >= abs(observed))
                test_name, stat, p = "paired-bootstrap", observed, (1 + extreme) / (diff_dist.shape[0] + 1)
            block.append(
                {
                    "metric": metric,
                    "model_a": ma,
                    "model_b": mb,
                    "difference": float(observed),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "test": test_name,
                    "statistic": float(stat),
                    "p_value": float(p),
                }
            )
        if block:
            adjusted = adjust_pvalues([r["p_value"] for r in block], method=correction)  # type: ignore[arg-type]
            for r, pa in zip(block, adjusted):
                r["p_adjusted"] = float(pa)
                r["significant"] = bool(pa < alpha)
        tests.extend(block)

    settings = {
        "n_samples": int(yt.shape[0]),
        "level": level,
        "alpha": alpha,
        "n_resamples": n_resamples,
        "correction": correction,
        "random_state": random_state,
        "stratified": use_strata,
        "baseline": baseline,
        "ci_method": "bootstrap-percentile (paired resamples)",
    }
    return ComparisonResult(tuple(names), tuple(metric_names), tuple(estimates), tuple(tests), settings)
