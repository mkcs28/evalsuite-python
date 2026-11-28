"""Bayesian comparison of two models: the probability that one is better, not a p-value.

>>> import evalsuite as es
>>> b = es.bayes_compare(y_true, pred_a, pred_b, metric="accuracy", rope=0.01)
>>> b.p_a_better, b.p_rope, b.p_b_better         # probabilities that sum to 1
>>> b.summary()                                   # "A is better with probability 0.97 ..."
>>> es.bayes_correlated_ttest(cv_scores_a, cv_scores_b, test_fraction=0.2, rope=0.01)

``bayes_compare`` puts a Bayesian bootstrap (Rubin 1981: flat Dirichlet weights over the test examples) on the
paired test set and returns the posterior of the metric difference, its credible interval and the
probabilities of A better, practically equivalent (inside the region of practical equivalence, ROPE) and B
better. ``bayes_correlated_ttest`` is the correlated Bayesian t-test of Benavoli et al. (2017) for scores from
k-fold cross-validation, where training sets overlap.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np

from .core.exceptions import EvalSuiteError, InputValidationError
from .core.validation import numeric

__all__ = ["BayesComparison", "bayes_compare", "bayes_correlated_ttest"]


@dataclass(frozen=True)
class BayesComparison:
    """Posterior of ``metric(A) − metric(B)`` and the decision probabilities."""

    metric: str
    estimate_a: float
    estimate_b: float
    mean: float
    low: float
    high: float
    p_a_better: float
    p_rope: float
    p_b_better: float
    rope: tuple[float, float]
    level: float
    method: str
    higher_is_better: bool
    samples: np.ndarray = field(repr=False)

    @property
    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def p_rope_or_better(self) -> float:
        """Probability that A is not practically worse than B (non-inferiority)."""
        return self.p_a_better + self.p_rope

    def decision(self, *, threshold: float = 0.95) -> str:
        """``"A better"``, ``"B better"``, ``"equivalent"`` or ``"undecided"`` at a posterior probability."""
        if self.p_a_better >= threshold:
            return "A better"
        if self.p_b_better >= threshold:
            return "B better"
        if self.p_rope >= threshold:
            return "equivalent"
        return "undecided"

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        pct = f"{self.level * 100:g}%"
        lo, hi = self.rope
        return "\n".join(
            [
                f"Bayesian comparison of {self.metric} ({self.method})",
                f"  A: {self.estimate_a:.{d}f}   B: {self.estimate_b:.{d}f}",
                f"  Difference A − B: {self.mean:.{d}f}  "
                f"({pct} credible interval {self.low:.{d}f} to {self.high:.{d}f})",
                f"  P(A better) = {self.p_a_better:.{d}f}   P(practically equal, ROPE [{lo:g}, {hi:g}]) = "
                f"{self.p_rope:.{d}f}   P(B better) = {self.p_b_better:.{d}f}",
                f"  Decision at 0.95: {self.decision()}",
            ]
        )

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {k: v for k, v in self.__dict__.items() if k != "samples"} | {"decision": self.decision()}
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self, *, digits: int = 3) -> str:
        """The Markdown tables as an HTML fragment (escaped), e.g. for :func:`evalsuite.html_report`."""
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        rows = ["| Quantity | Value |", "| --- | ---: |"]
        rows += [
            f"| {self.metric} of A | {self.estimate_a:.{d}f} |",
            f"| {self.metric} of B | {self.estimate_b:.{d}f} |",
            f"| A − B (posterior mean) | {self.mean:.{d}f} |",
            f"| {self.level * 100:g}% credible interval | {self.low:.{d}f} to {self.high:.{d}f} |",
            f"| P(A better) | {self.p_a_better:.{d}f} |",
            f"| P(practically equal) | {self.p_rope:.{d}f} |",
            f"| P(B better) | {self.p_b_better:.{d}f} |",
        ]
        return "\n".join(rows) + "\n"


def _rope(rope: Union[float, tuple[float, float]]) -> tuple[float, float]:
    if isinstance(rope, (int, float)):
        r = abs(float(rope))
        return (-r, r)
    lo, hi = (float(x) for x in rope)
    if lo > hi:
        raise InputValidationError("rope must be (low, high) with low ≤ high.")
    return (lo, hi)


def _decide(diff: np.ndarray, rope: tuple[float, float], better_sign: float) -> tuple[float, float, float]:
    d = diff * better_sign  # positive means A better
    lo, hi = rope
    p_rope = float(np.mean((d >= lo) & (d <= hi)))
    p_a = float(np.mean(d > hi))
    return p_a, p_rope, float(1.0 - p_a - p_rope)


def _curve_fast_path(
    fn: Callable[..., Any], y: np.ndarray, a: np.ndarray, b: np.ndarray, kwargs: dict[str, Any]
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """For binary ROC AUC and average precision: a function of the weight matrix (draws × examples) giving the
    posterior differences, from :func:`evalsuite.stats.intervals._weighted_curves`; None for other metrics."""
    from .classification.metrics import average_precision, roc_auc
    from .stats._resolve import MetricCall
    from .stats.intervals import _weighted_curves

    if fn not in (roc_auc, average_precision):
        return None
    try:
        ia = MetricCall(fn, y, None, a, None, dict(kwargs))._curve_inputs()
        ib = MetricCall(fn, y, None, b, None, dict(kwargs))._curve_inputs()
    except EvalSuiteError:
        return None
    if ia is None or ib is None:
        return None
    rows = np.arange(y.shape[0])
    is_auc = fn is roc_auc

    def run(weights: np.ndarray) -> np.ndarray:
        va = _weighted_curves(is_auc, ia[1], ia[2], None, rows, weights)
        vb = _weighted_curves(is_auc, ib[1], ib[2], None, rows, weights)
        return np.asarray(va - vb)

    return run


def bayes_compare(
    y_true: Any,
    pred_a: Any,
    pred_b: Any,
    *,
    metric: Union[str, Callable[..., Any]] = "accuracy",
    rope: Union[float, tuple[float, float]] = 0.01,
    level: float = 0.95,
    n_samples: int = 4000,
    higher_is_better: Optional[bool] = None,
    random_state: Optional[int] = None,
    **metric_kwargs: Any,
) -> BayesComparison:
    """Posterior of the difference in ``metric`` between two models scored on the same examples.

    Each posterior draw weights the test examples by flat Dirichlet weights (Bayesian bootstrap) and evaluates
    the metric on both models with those weights, so the pairing is kept. ``pred_a`` / ``pred_b`` are labels, or
    scores for probability metrics (``metric="roc_auc"``, ``"brier"``...). ``rope`` is the region of practical
    equivalence on the metric's scale (±0.01 by default)."""
    from .stats._resolve import resolve_metric
    from .stats.compare import _needs_prob

    if n_samples < 100:
        raise InputValidationError("n_samples must be at least 100.")
    if not 0 < level < 1:
        raise InputValidationError("level must be in (0, 1).")
    fn, name = resolve_metric(metric)
    label = name if isinstance(metric, str) else getattr(fn, "__name__", "metric")
    prob = _needs_prob(name) if isinstance(metric, str) else False
    y = np.asarray(y_true)
    a, b = np.asarray(pred_a), np.asarray(pred_b)
    if not (y.shape[0] == a.shape[0] == b.shape[0]):
        raise InputValidationError(
            f"y_true, pred_a and pred_b must have one row per example; got {y.shape[0]}, {a.shape[0]} and "
            f"{b.shape[0]}."
        )
    n = y.shape[0]
    if n < 2:
        raise InputValidationError("At least 2 examples are needed.")

    def score(pred: np.ndarray, w: Optional[np.ndarray]) -> float:
        kw = dict(metric_kwargs)
        if w is not None:
            kw["sample_weight"] = w
        res = fn(y, y_prob=pred, **kw) if prob else fn(y, pred, **kw)
        value: Any = getattr(res, "value", res)
        return float(value)

    est_a, est_b = score(a, None), score(b, None)
    if higher_is_better is None:
        from .stats.compare import _info

        info = _info(name)
        higher_is_better = True if info is None or info.higher_is_better is None else bool(info.higher_is_better)
    rng = np.random.default_rng(random_state)
    diff = np.empty(n_samples)
    failed = 0
    fast = _curve_fast_path(fn, y, a, b, metric_kwargs) if prob and "sample_weight" not in metric_kwargs else None
    if fast is not None:  # ROC AUC / average precision: all draws at once from one sort per model
        weights = np.vstack([rng.gamma(1.0, 1.0, n) for _ in range(n_samples)])
        weights *= n / weights.sum(axis=1, keepdims=True)
        diff = fast(weights)
        failed = int(np.sum(~np.isfinite(diff)))
    for s in range(n_samples if fast is None else 0):
        w = rng.gamma(1.0, 1.0, n)
        w *= n / w.sum()
        try:
            diff[s] = score(a, w) - score(b, w)
        except (EvalSuiteError, TypeError) as exc:
            if isinstance(exc, TypeError) and "sample_weight" in str(exc):
                raise InputValidationError(
                    f"{label} does not accept sample_weight, which the Bayesian bootstrap needs."
                ) from exc
            diff[s] = np.nan
            failed += 1
    diff = diff[np.isfinite(diff)]
    if diff.size < n_samples // 2:
        raise InputValidationError(f"{label} was undefined on most posterior draws ({failed} of {n_samples}).")
    rp = _rope(rope)
    sign = 1.0 if higher_is_better else -1.0
    pa, pr, pb = _decide(diff, rp, sign)
    a_ = (1 - level) / 2
    return BayesComparison(
        metric=label,
        estimate_a=est_a,
        estimate_b=est_b,
        mean=float(diff.mean()),
        low=float(np.quantile(diff, a_)),
        high=float(np.quantile(diff, 1 - a_)),
        p_a_better=pa,
        p_rope=pr,
        p_b_better=pb,
        rope=rp,
        level=level,
        method="Bayesian bootstrap (Rubin 1981)",
        higher_is_better=bool(higher_is_better),
        samples=diff,
    )


def bayes_correlated_ttest(
    scores_a: Any,
    scores_b: Any,
    *,
    test_fraction: Optional[float] = None,
    rope: Union[float, tuple[float, float]] = 0.01,
    level: float = 0.95,
    higher_is_better: bool = True,
    n_samples: int = 50_000,
    random_state: Optional[int] = None,
    metric: str = "score",
) -> BayesComparison:
    """Correlated Bayesian t-test (Benavoli, Corani, Demšar & Zaffalon 2017) for per-fold cross-validation scores
    of two models on the same folds. ``test_fraction`` is n_test / n_total of a fold (1/k for k-fold; the default),
    which sets the correlation between folds (Nadeau & Bengio 2003). The posterior of the mean difference is a
    Student t with n − 1 degrees of freedom; probabilities are exact and the draws only feed ``samples``."""
    from scipy import stats

    sa = numeric(scores_a, "scores_a").ravel()
    sb = numeric(scores_b, "scores_b").ravel()
    if sa.size != sb.size:
        raise InputValidationError(f"Both models need a score per fold; got {sa.size} and {sb.size}.")
    if sa.size < 2 or not (np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))):
        raise InputValidationError("At least 2 finite scores per model are needed.")
    if not 0 < level < 1:
        raise InputValidationError("level must be in (0, 1).")
    d = sa - sb
    n = d.size
    rho = 1.0 / n if test_fraction is None else float(test_fraction)
    if not 0 <= rho < 1:
        raise InputValidationError("test_fraction must be in [0, 1).")
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    rp = _rope(rope)
    sign = 1.0 if higher_is_better else -1.0
    if var == 0:
        draws = np.full(1000, mean)
        pa, pr, pb = _decide(draws, rp, sign)
        lo = hi = mean
    else:
        scale = math.sqrt((1.0 / n + rho / (1.0 - rho)) * var)
        post = stats.t(df=n - 1, loc=mean, scale=scale)
        # exact probabilities on the "A better" orientation
        lo_r, hi_r = rp
        if sign > 0:
            pa = float(post.sf(hi_r))
            pb = float(post.cdf(lo_r))
        else:
            pa = float(post.cdf(-hi_r))
            pb = float(post.sf(-lo_r))
        pr = float(1.0 - pa - pb)
        a_ = (1 - level) / 2
        lo, hi = float(post.ppf(a_)), float(post.ppf(1 - a_))
        draws = post.rvs(size=n_samples, random_state=np.random.default_rng(random_state))
    return BayesComparison(
        metric=metric,
        estimate_a=float(sa.mean()),
        estimate_b=float(sb.mean()),
        mean=mean,
        low=lo,
        high=hi,
        p_a_better=pa,
        p_rope=pr,
        p_b_better=pb,
        rope=rp,
        level=level,
        method="correlated Bayesian t-test (Benavoli et al. 2017)",
        higher_is_better=higher_is_better,
        samples=np.asarray(draws),
    )
