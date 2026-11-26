"""Confidence intervals and baseline comparisons for :func:`evalsuite.evaluate`.

``ci="analytic"`` (the default) attaches closed-form intervals where a standard one exists and costs
nothing: Wilson intervals for proportions (accuracy, precision, recall, specificity, NPV), DeLong's interval
for binary ROC AUC, and Student-t intervals for metrics that are means of per-observation losses (Brier score,
log loss, MAE, MSE, mean bias error; RMSE from the MSE interval). ``ci="bootstrap"`` uses the paired percentile
bootstrap of :func:`evalsuite.compare` for every metric. A baseline (``baseline_pred`` / ``baseline_prob``)
adds paired tests against it: McNemar's test for accuracy, DeLong's test for binary ROC AUC and a paired
bootstrap test otherwise, with p-values adjusted within each metric (Holm).
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Optional

import numpy as np

from .core.exceptions import InputValidationError
from .core.result import EvaluationResult, MetricResult

_PROPORTIONS = {"precision", "recall", "specificity", "npv", "sensitivity", "ppv"}
_MEAN_LOSSES = {"brier_score", "log_loss", "mae", "mse", "rmse", "mean_bias_error"}


def _t_quantile(level: float, df: int) -> float:
    from scipy import stats

    return float(stats.t.ppf(1 - (1 - level) / 2, df))


def _mean_ci(terms: np.ndarray, level: float) -> tuple[float, float]:
    n = terms.shape[0]
    if n < 2:
        return math.nan, math.nan
    m = float(np.mean(terms))
    d = terms - m
    # Sample standard deviation via a BLAS dot product (one pass over the deviations instead of np.std's
    # several temporaries); same two-pass formula, so the same accuracy.
    se = math.sqrt(float(np.dot(d, d)) / (n - 1)) / math.sqrt(n)
    q = _t_quantile(level, n - 1)
    return m - q * se, m + q * se


def _analytic(
    result: EvaluationResult,
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
    level: float,
) -> dict[str, tuple[float, float, str]]:
    """Closed-form intervals for the metrics that have a standard one (unweighted data only)."""
    from . import proportion_ci, roc_auc_ci

    out: dict[str, tuple[float, float, str]] = {}

    def wilson(k: int, n: int) -> Optional[tuple[float, float, str]]:
        if n <= 0:
            return None
        ci = proportion_ci(int(k), int(n), level=level)
        return ci.low, ci.high, "wilson"

    if result.task == "classification":
        cm = result.confusion_matrix
        labels = list(result.labels or ())
        for name, m in result.metrics.items():
            if isinstance(m.value, np.ndarray):
                continue
            if name == "accuracy" and cm is not None:
                c = wilson(int(np.trace(cm)), int(cm.sum()))
                if c:
                    out[name] = c
            elif name in _PROPORTIONS and cm is not None and m.params.get("average") == "binary":
                pos = m.params.get("pos_label")
                if pos not in labels:
                    continue
                i = labels.index(pos)
                tp = int(cm[i, i])
                fp = int(cm[:, i].sum()) - tp
                fn = int(cm[i, :].sum()) - tp
                tn = int(cm.sum()) - tp - fp - fn
                k, n = {
                    "precision": (tp, tp + fp),
                    "ppv": (tp, tp + fp),
                    "recall": (tp, tp + fn),
                    "sensitivity": (tp, tp + fn),
                    "specificity": (tn, tn + fp),
                    "npv": (tn, tn + fn),
                }[name]
                c = wilson(k, n)
                if c:
                    out[name] = c
            elif result.target_type == "binary" and y_prob is not None and np.ndim(y_prob) == 1:
                pos = m.params.get("pos_label")
                if name == "roc_auc":
                    ci = roc_auc_ci(y_true, y_prob, level=level, pos_label=pos)
                    out[name] = (ci.low, ci.high, "delong")
                elif name in ("brier_score", "log_loss"):
                    yt = np.asarray(y_true)
                    positive = pos if pos is not None else (labels[-1] if labels else 1)
                    y = (yt == positive).astype(float)
                    p = np.asarray(y_prob, dtype=float)
                    if name == "brier_score":
                        terms = (p - y) ** 2
                    else:
                        eps = np.finfo(np.float64).eps
                        q = np.clip(p, eps, 1 - eps)
                        terms = -(y * np.log(q) + (1 - y) * np.log(1 - q))
                    lo, hi = _mean_ci(terms, level)
                    out[name] = (max(0.0, lo), hi, "t")
    elif result.task == "regression" and np.ndim(y_true) == 1:
        e = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
        n = e.shape[0]
        if n >= 2:
            # One pass of BLAS dot products serves every error metric: Σe², Σe⁴ (and Σe) give the means and
            # sample variances of |e|, e² and e without temporaries for |e| (|e|·|e| = e·e).
            tq = _t_quantile(level, n - 1)
            ee = float(np.dot(e, e))
            sq: Optional[np.ndarray] = None

            def t_ci(mean: float, sum_sq: float) -> tuple[float, float]:
                var = max(0.0, (sum_sq - n * mean * mean) / (n - 1))
                half = tq * math.sqrt(var / n)
                return mean - half, mean + half

            for name in result.metrics:
                if name == "mae":
                    lo, hi = t_ci(float(result.metrics["mae"].value), ee)
                    out[name] = (max(0.0, lo), hi, "t")
                elif name in ("mse", "rmse"):
                    if sq is None:
                        sq = e * e
                    lo, hi = t_ci(ee / n, float(np.dot(sq, sq)))
                    lo = max(0.0, lo)
                    out[name] = (lo, hi, "t") if name == "mse" else (math.sqrt(lo), math.sqrt(hi), "t (from MSE)")
                elif name == "mean_bias_error":
                    lo, hi = t_ci(float(e.mean()), ee)
                    out[name] = (lo, hi, "t")
    return out


def _metric_kwargs(names: list[str], result: EvaluationResult) -> dict[str, dict[str, Any]]:
    """Options that reproduce each metric exactly as evaluate() computed it."""
    from .stats._resolve import resolve_metric

    out: dict[str, dict[str, Any]] = {}
    for name in names:
        fn, _ = resolve_metric(name)
        accepted = set(inspect.signature(fn).parameters)
        params = dict(result.metrics[name].params)
        kw: dict[str, Any] = {
            k: params[k] for k in ("average", "pos_label", "zero_division") if k in accepted and k in params
        }
        if "average" in kw and kw["average"] in ("auto",):
            kw.pop("average")
        out[name] = kw
    return out


def attach(
    result: EvaluationResult,
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
    *,
    ci: Any,
    level: float,
    baseline_pred: Any,
    baseline_prob: Any,
    n_resamples: int,
    random_state: Optional[int],
    sample_weight: Any,
    clusters: Any = None,
) -> EvaluationResult:
    """Return ``result`` with intervals in each metric's params (``ci_low``, ``ci_high``, ``ci_level``,
    ``ci_method``) and, with a baseline, the paired comparison (``baseline_*`` params and ``comparison``)."""
    if ci in (None, False, "none"):
        mode = "none"
    elif ci is True or ci == "analytic":
        mode = "analytic"
    elif ci == "bootstrap":
        mode = "bootstrap"
    else:
        raise InputValidationError("ci must be 'analytic' (default), 'bootstrap', or False.")
    has_baseline = baseline_pred is not None or baseline_prob is not None
    if mode == "none" and not has_baseline:
        return result
    names = [k for k, m in result.metrics.items() if not isinstance(m.value, np.ndarray)]
    intervals: dict[str, tuple[float, float, str]] = {}
    if clusters is not None:
        return _attach_clustered(
            result,
            y_true,
            y_pred,
            y_prob,
            names,
            mode,
            level,
            baseline_pred,
            baseline_prob,
            n_resamples,
            random_state,
            sample_weight,
            clusters,
        )
    if mode == "analytic" and sample_weight is None:
        intervals = _analytic(result, y_true, y_pred, y_prob, level)
    comparison = None
    if mode == "bootstrap" or has_baseline:
        from .stats.compare import compare

        preds: dict[str, Any] = {}
        probs: dict[str, Any] = {}
        if y_pred is not None:
            preds["model"] = y_pred
        if y_prob is not None:
            probs["model"] = y_prob
        if baseline_pred is not None:
            preds["baseline"] = baseline_pred
        if baseline_prob is not None:
            probs["baseline"] = baseline_prob
        from .stats.compare import _needs_prob

        usable = [n for n in names if not has_baseline or ("baseline" in (probs if _needs_prob(n) else preds))]
        if has_baseline and not usable:
            raise InputValidationError(
                "The baseline needs the same kind of input as the metrics: baseline_pred for label/value metrics, "
                "baseline_prob for probability metrics."
            )
        comparison = compare(
            y_true,
            preds,
            probabilities=probs,
            metrics=usable,
            baseline="baseline" if has_baseline else None,
            level=level,
            n_resamples=n_resamples,
            random_state=random_state,
            sample_weight=sample_weight,
            metric_kwargs=_metric_kwargs(usable, result),
            _allow_single=True,
        )
        if mode == "bootstrap" and comparison is not None:
            for e in comparison.estimates:
                if e["model"] == "model":
                    intervals[e["metric"]] = (e["low"], e["high"], "percentile bootstrap")
    tests = {t["metric"]: t for t in comparison.tests} if comparison is not None and has_baseline else {}
    base_values = (
        {e["metric"]: e["estimate"] for e in comparison.estimates if e["model"] == "baseline"}
        if tests and comparison is not None
        else {}
    )
    metrics: dict[str, MetricResult] = {}
    for key, m in result.metrics.items():
        extra: dict[str, Any] = {}
        if key in intervals:
            lo, hi, method = intervals[key]
            extra.update(ci_low=float(lo), ci_high=float(hi), ci_level=level, ci_method=method)
        if key in tests:
            t = tests[key]
            extra.update(
                baseline_value=float(base_values[key]),
                baseline_difference=float(t["difference"]),
                baseline_ci_low=float(t["ci_low"]),
                baseline_ci_high=float(t["ci_high"]),
                baseline_test=t["test"],
                baseline_p_value=float(t["p_value"]),
                baseline_p_adjusted=float(t["p_adjusted"]),
            )
        metrics[key] = MetricResult(m.metric, m.name, m.value, {**m.params, **extra}, m.labels) if extra else m
    return EvaluationResult(
        task=result.task,
        metrics=metrics,
        n_samples=result.n_samples,
        target_type=result.target_type,
        labels=result.labels,
        confusion_matrix=result.confusion_matrix,
        metadata={
            **result.metadata,
            "ci": mode,
            "ci_level": level,
            **({"n_resamples": n_resamples, "random_state": random_state} if comparison is not None else {}),
        },
        comparison=comparison,
    )


def _attach_clustered(
    result: EvaluationResult,
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
    names: list[str],
    mode: str,
    level: float,
    baseline_pred: Any,
    baseline_prob: Any,
    n_resamples: int,
    random_state: Optional[int],
    sample_weight: Any,
    clusters: Any,
) -> EvaluationResult:
    """Intervals and baseline tests that resample whole clusters (``evaluate(..., clusters=...)``): analytic
    intervals assume independent observations, so with clusters every interval is a cluster bootstrap."""
    import warnings

    from . import adjust_pvalues
    from .core.exceptions import EvalSuiteError
    from .stats._resolve import resolve_metric
    from .stats.compare import _needs_prob
    from .stats.intervals import _cluster_members

    yt = np.asarray(y_true)
    members = _cluster_members(clusters, yt.shape[0])
    g = len(members)
    if g < 2:
        raise InputValidationError("Clustered intervals need at least two clusters.")
    kwargs = _metric_kwargs(names, result)
    rng = np.random.default_rng(random_state)
    picks = [np.concatenate([members[k] for k in rng.integers(0, g, g)]) for _ in range(n_resamples)]
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    def value(name: str, second: Any, idx: Optional[np.ndarray]) -> float:
        fn, _ = resolve_metric(name)
        s = np.asarray(second)
        kw = dict(kwargs[name])
        if w is not None:
            kw["sample_weight"] = w if idx is None else w[idx]
        if idx is None:
            return float(fn(yt, s, **kw)) if not _needs_prob(name) else float(fn(yt, y_prob=s, **kw))
        return (
            float(fn(yt[idx], s[idx], **kw)) if not _needs_prob(name) else float(fn(yt[idx], y_prob=s[idx], **kw))
        )

    def dist(name: str, second: Any) -> np.ndarray:
        out = np.empty(n_resamples)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for b, idx in enumerate(picks):
                try:
                    out[b] = value(name, second, idx)
                except EvalSuiteError:
                    out[b] = np.nan
        return out

    a = (1 - level) / 2
    extra: dict[str, dict[str, Any]] = {k: {} for k in names}
    raw_p: dict[str, float] = {}
    for name in names:
        second = y_prob if _needs_prob(name) else y_pred
        if second is None:
            continue
        d_model = dist(name, second)
        if mode != "none" and d_model is not None:
            v = d_model[~np.isnan(d_model)]
            if v.size:
                lo, hi = np.quantile(v, [a, 1 - a])
                extra[name].update(
                    ci_low=float(lo),
                    ci_high=float(hi),
                    ci_level=level,
                    ci_method=f"cluster bootstrap ({g} clusters)",
                )
        base = baseline_prob if _needs_prob(name) else baseline_pred
        if base is not None and d_model is not None:
            d_base = dist(name, base)
            diff = d_model - d_base
            diff = diff[~np.isnan(diff)]
            est = result.metrics[name].value
            base_val = value(name, base, None)
            if diff.size:
                lo, hi = np.quantile(diff, [a, 1 - a])
                p = float(min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))))
                raw_p[name] = p
                extra[name].update(
                    baseline_value=base_val,
                    baseline_difference=float(est) - base_val,
                    baseline_ci_low=float(lo),
                    baseline_ci_high=float(hi),
                    baseline_test="paired cluster bootstrap",
                    baseline_p_value=p,
                )
    if raw_p:
        adj = adjust_pvalues(list(raw_p.values()), method="holm")
        for name, p in zip(raw_p, np.atleast_1d(adj)):
            extra[name]["baseline_p_adjusted"] = float(p)
    metrics: dict[str, MetricResult] = {}
    for key, m in result.metrics.items():
        e = extra.get(key) or {}
        metrics[key] = MetricResult(m.metric, m.name, m.value, {**m.params, **e}, m.labels) if e else m
    return EvaluationResult(
        task=result.task,
        metrics=metrics,
        n_samples=result.n_samples,
        target_type=result.target_type,
        labels=result.labels,
        confusion_matrix=result.confusion_matrix,
        metadata={
            **result.metadata,
            "ci": "cluster bootstrap",
            "ci_level": level,
            "clusters": g,
            "n_resamples": n_resamples,
            "random_state": random_state,
        },
    )
