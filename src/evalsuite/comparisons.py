"""Comparisons beyond "is there a difference": equivalence and non-inferiority, many models over many data sets,
and metrics when some labels are missing.

>>> import evalsuite as es
>>> es.noninferiority_test("accuracy", y, pred_new, pred_old, margin=0.02)   # is the new model no worse?
>>> es.equivalence_test("roc_auc", y, y_prob_a=pa, y_prob_b=pb, margin=0.01)  # practically the same (TOST)?
>>> nem = es.nemenyi_test(scores)          # datasets × models → average ranks, critical difference, pairwise p
>>> es.plot.critical_difference(nem)       # the critical-difference diagram (Demšar 2006)
>>> es.ipw_metric("accuracy", y, pred, observed=has_label, p_observed=propensity)

Non-inferiority and equivalence follow the FDA / EMA two one-sided tests (TOST, Schuirmann 1987) on the paired
bootstrap distribution of metric(A) − metric(B): non-inferior when the lower (1 − 2α) bound exceeds −margin;
equivalent when the whole (1 − 2α) interval lies inside ±margin. The Nemenyi test (Nemenyi 1963; Demšar, JMLR
2006) compares every pair of models by average rank across data sets after a Friedman test. ``ipw_metric`` weights
labelled examples by the inverse probability of having a label (Horvitz–Thompson), which removes the bias of
evaluating only on cases that happened to be labelled when labelling depends on observed covariates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from .core.exceptions import InputValidationError
from .core.result import MetricResult
from .core.validation import numeric
from .stats.results import ConfidenceInterval, TestResult

__all__ = [
    "equivalence_test",
    "noninferiority_test",
    "NemenyiResult",
    "nemenyi_test",
    "ipw_metric",
]


def _margin_test(
    kind: str,
    metric: Any,
    y_true: Any,
    y_pred_a: Any,
    y_pred_b: Any,
    *,
    margin: float,
    alpha: float,
    y_prob_a: Any,
    y_prob_b: Any,
    sample_weight: Any,
    higher_is_better: Optional[bool],
    n_resamples: int,
    random_state: Optional[int],
    stratify: Optional[bool],
    metric_kwargs: dict[str, Any],
) -> TestResult:
    from .stats.compare import _info
    from .stats.intervals import _check_resamples
    from .stats.paired import _paired_differences

    if not margin > 0 or not math.isfinite(margin):
        raise InputValidationError("margin must be a positive number on the metric's scale.")
    if not 0 < alpha < 0.5:
        raise InputValidationError("alpha must be in (0, 0.5).")
    n_resamples = _check_resamples(n_resamples)
    name, observed, diff, call_a, call_b, use_strata, fa, fb = _paired_differences(
        metric,
        y_true,
        y_pred_a,
        y_pred_b,
        y_prob_a,
        y_prob_b,
        sample_weight,
        n_resamples,
        random_state,
        stratify,
        "numpy",
        metric_kwargs,
    )
    diff = diff[~np.isnan(diff)]
    if diff.size < n_resamples // 2:
        raise InputValidationError(f"{name} was undefined on most bootstrap resamples.")
    if higher_is_better is None:
        info = _info(name)
        higher_is_better = True if info is None or info.higher_is_better is None else bool(info.higher_is_better)
    s = 1.0 if higher_is_better else -1.0
    d, dd = s * observed, s * diff  # oriented so that larger means A better
    centred = dd - d
    b = dd.size
    low, high = (float(x) for x in np.quantile(diff, [alpha, 1 - alpha]))
    lo_o, hi_o = (float(x) for x in np.quantile(dd, [alpha, 1 - alpha]))
    if kind == "noninferiority":
        p = (1 + np.sum(centred >= d + margin)) / (b + 1)
        decision = lo_o > -margin
        label = "non-inferior" if decision else "non-inferiority not shown"
    else:
        p_low = (1 + np.sum(centred >= d + margin)) / (b + 1)
        p_high = (1 + np.sum(centred <= d - margin)) / (b + 1)
        p = max(p_low, p_high)
        decision = lo_o > -margin and hi_o < margin
        label = "equivalent" if decision else "equivalence not shown"
    ci = ConfidenceInterval(
        observed,
        low,
        high,
        1 - 2 * alpha,
        "bootstrap-percentile",
        f"{name}_difference",
        {"n_resamples": n_resamples, "random_state": random_state},
    )
    return TestResult(
        kind,
        observed,
        float(p),
        alternative="greater" if kind == "noninferiority" else "two one-sided",
        estimate=observed,
        ci=ci,
        params={
            "metric": name,
            "margin": margin,
            "alpha": alpha,
            "higher_is_better": higher_is_better,
            "decision": label,
            "conclusion": bool(decision),
            "metric_a": call_a(),
            "metric_b": call_b(),
            "n_resamples": n_resamples,
            "random_state": random_state,
            "stratified": use_strata,
            "failed_resamples": max(fa, fb),
        },
    )


def noninferiority_test(
    metric: Any,
    y_true: Any,
    y_pred_a: Any = None,
    y_pred_b: Any = None,
    *,
    margin: float,
    alpha: float = 0.025,
    y_prob_a: Any = None,
    y_prob_b: Any = None,
    sample_weight: Any = None,
    higher_is_better: Optional[bool] = None,
    n_resamples: int = 2000,
    random_state: Optional[int] = None,
    stratify: Optional[bool] = None,
    **metric_kwargs: Any,
) -> TestResult:
    """Is model A (new) no worse than model B (reference) by more than ``margin``? One-sided test at ``alpha``
    (0.025 by default, as regulators expect) on paired bootstrap differences; ``params["conclusion"]`` is True when
    the lower (1 − 2α) confidence bound of A − B (oriented so higher is better) is above −margin."""
    return _margin_test(
        "noninferiority",
        metric,
        y_true,
        y_pred_a,
        y_pred_b,
        margin=margin,
        alpha=alpha,
        y_prob_a=y_prob_a,
        y_prob_b=y_prob_b,
        sample_weight=sample_weight,
        higher_is_better=higher_is_better,
        n_resamples=n_resamples,
        random_state=random_state,
        stratify=stratify,
        metric_kwargs=metric_kwargs,
    )


def equivalence_test(
    metric: Any,
    y_true: Any,
    y_pred_a: Any = None,
    y_pred_b: Any = None,
    *,
    margin: float,
    alpha: float = 0.05,
    y_prob_a: Any = None,
    y_prob_b: Any = None,
    sample_weight: Any = None,
    n_resamples: int = 2000,
    random_state: Optional[int] = None,
    stratify: Optional[bool] = None,
    **metric_kwargs: Any,
) -> TestResult:
    """Are models A and B equivalent within ±``margin``? Two one-sided tests (TOST) at ``alpha`` on paired
    bootstrap differences; ``params["conclusion"]`` is True when the (1 − 2α) interval of A − B lies inside
    ±margin."""
    return _margin_test(
        "equivalence",
        metric,
        y_true,
        y_pred_a,
        y_pred_b,
        margin=margin,
        alpha=alpha,
        y_prob_a=y_prob_a,
        y_prob_b=y_prob_b,
        sample_weight=sample_weight,
        higher_is_better=True,
        n_resamples=n_resamples,
        random_state=random_state,
        stratify=stratify,
        metric_kwargs=metric_kwargs,
    )


# ------------------------------------------------------------------------------------------------- Nemenyi


@dataclass(frozen=True)
class NemenyiResult:
    """Friedman test, average ranks, critical difference and pairwise Nemenyi p-values."""

    models: tuple[str, ...]
    average_ranks: tuple[float, ...]
    critical_difference: float
    p_values: np.ndarray  # models × models
    friedman: TestResult
    n_datasets: int
    alpha: float

    def ranking(self) -> list[tuple[str, float]]:
        return sorted(zip(self.models, self.average_ranks), key=lambda x: x[1])

    def differs(self, a: str, b: str) -> bool:
        i, j = self.models.index(a), self.models.index(b)
        return bool(abs(self.average_ranks[i] - self.average_ranks[j]) > self.critical_difference)

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        f = self.friedman
        lines = [
            f"Nemenyi test: {len(self.models)} models on {self.n_datasets} data sets "
            f"(Friedman χ²={f.statistic:.{d}f}, p={f.p_value:.{d}g}); critical difference "
            f"{self.critical_difference:.{d}f} at α={self.alpha:g}",
            "  Average rank (1 = best):",
        ]
        lines += [f"    {m:<20} {r:.{d}f}" for m, r in self.ranking()]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "models": list(self.models),
                "average_ranks": list(self.average_ranks),
                "critical_difference": self.critical_difference,
                "p_values": self.p_values.tolist(),
                "friedman": {"statistic": self.friedman.statistic, "p_value": self.friedman.p_value},
                "n_datasets": self.n_datasets,
                "alpha": self.alpha,
            }
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        head = "| Model | Avg rank | " + " | ".join(self.models) + " |"
        sep = "| --- | ---: |" + " ---: |" * len(self.models)
        rows = [head, sep]
        for i, m in enumerate(self.models):
            cells = " | ".join("—" if i == j else f"{self.p_values[i, j]:.{d}g}" for j in range(len(self.models)))
            rows.append(f"| {m} | {self.average_ranks[i]:.{d}f} | {cells} |")
        return "\n".join(rows) + f"\n\nCritical difference: {self.critical_difference:.{d}f}\n"

    def to_html(self, *, digits: int = 3) -> str:
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))


def nemenyi_test(
    scores: Any,
    *,
    models: Optional[list[str]] = None,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> NemenyiResult:
    """Compare k models over N data sets (or folds): ``scores`` is N × k (a DataFrame's columns name the models),
    or a mapping model → N scores. Ranks are computed per data set (ties share the average rank; 1 = best)."""
    from scipy import stats

    from .stats.hypothesis import friedman_test

    if isinstance(scores, dict):
        models = list(models or scores)
        x = np.column_stack([numeric(scores[m], m).ravel() for m in models])
    elif hasattr(scores, "columns"):
        models = list(models or [str(c) for c in scores.columns])
        x = numeric(np.asarray(scores), "scores")
    else:
        x = numeric(scores, "scores")
        if x.ndim != 2:
            raise InputValidationError("scores must be data sets × models (2-D).")
        models = list(models or [f"model_{j}" for j in range(x.shape[1])])
    if x.ndim != 2 or x.shape[1] != len(models):
        raise InputValidationError(f"{len(models)} model names for a table with {x.shape[1]} columns.")
    n, k = x.shape
    if k < 2 or n < 2:
        raise InputValidationError("Need at least 2 models and 2 data sets.")
    if not np.all(np.isfinite(x)):
        raise InputValidationError("scores must be finite (drop data sets with missing results).")
    if not 0 < alpha < 1:
        raise InputValidationError("alpha must be in (0, 1).")
    ranks = np.apply_along_axis(stats.rankdata, 1, -x if higher_is_better else x)
    avg = ranks.mean(axis=0)
    se = math.sqrt(k * (k + 1) / (6.0 * n))
    q = float(stats.studentized_range.ppf(1 - alpha, k, np.inf)) / math.sqrt(2)
    cd = q * se
    z = np.abs(avg[:, None] - avg[None, :]) / se
    p = np.asarray(stats.studentized_range.sf(z * math.sqrt(2), k, np.inf))
    np.fill_diagonal(p, 1.0)
    friedman = friedman_test(*[x[:, j] for j in range(k)]) if k >= 3 else None
    if friedman is None:
        w = stats.wilcoxon(x[:, 0], x[:, 1]) if np.any(x[:, 0] != x[:, 1]) else None
        friedman = TestResult("wilcoxon", float(w.statistic) if w else 0.0, float(w.pvalue) if w else 1.0)
    return NemenyiResult(tuple(models), tuple(float(v) for v in avg), float(cd), p, friedman, n, alpha)


# ------------------------------------------------------------------------------------------- missing labels


def ipw_metric(
    metric: Any,
    y_true: Any,
    y_pred: Any = None,
    *,
    observed: Any,
    p_observed: Any,
    y_prob: Any = None,
    clip: Union[float, tuple[float, float]] = 0.01,
    **metric_kwargs: Any,
) -> MetricResult:
    """A metric over the whole population when only some examples are labelled: labelled examples
    (``observed`` True) are weighted by 1 / ``p_observed`` (their estimated probability of being labelled, e.g.
    from a logistic model of labelling on covariates). Propensities are clipped to [clip, 1] to limit extreme
    weights. ``params`` reports the effective sample size and the unweighted (complete-case) value for
    comparison."""
    from .stats._resolve import resolve_metric
    from .stats.compare import _needs_prob

    fn, name = resolve_metric(metric)
    obs = np.asarray(observed).astype(bool).ravel()
    ps = numeric(p_observed, "p_observed").ravel()
    if obs.size != ps.size:
        raise InputValidationError(
            f"observed and p_observed must have the same length; got {obs.size} and {ps.size}."
        )
    if not np.all(np.isfinite(ps)) or np.any((ps <= 0) | (ps > 1)):
        raise InputValidationError("p_observed must be probabilities in (0, 1].")
    lo = clip[0] if isinstance(clip, tuple) else float(clip)
    w = 1.0 / np.clip(ps, lo, 1.0)
    idx = np.flatnonzero(obs)
    if idx.size < 2:
        raise InputValidationError("At least 2 labelled examples are needed.")
    y = np.asarray(y_true)
    if y.shape[0] != obs.size:
        raise InputValidationError(f"y_true has {y.shape[0]} rows for {obs.size} observed flags.")
    prob = y_prob is not None or (isinstance(metric, str) and _needs_prob(name))
    second = y_prob if y_prob is not None else y_pred
    if second is None:
        raise InputValidationError("Give y_pred, or y_prob for probability metrics.")
    s = np.asarray(second)
    yy, ss, ww = y[idx], s[idx], w[idx]
    if prob:
        weighted = fn(yy, y_prob=ss, sample_weight=ww, **metric_kwargs)
        naive = fn(yy, y_prob=ss, **metric_kwargs)
    else:
        weighted = fn(yy, ss, sample_weight=ww, **metric_kwargs)
        naive = fn(yy, ss, **metric_kwargs)
    ess = float(ww.sum() ** 2 / np.sum(ww**2))
    value = float(getattr(weighted, "value", weighted))
    return MetricResult(
        name=f"{name} (inverse probability weighted)",
        value=value,
        metric=f"ipw_{name}",
        params={
            "complete_case": float(getattr(naive, "value", naive)),
            "n_labelled": int(idx.size),
            "n_total": int(obs.size),
            "effective_sample_size": ess,
            "max_weight": float(ww.max()),
            "clip": lo,
        },
    )
