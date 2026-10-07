"""Confidence intervals: bootstrap (percentile, basic, BCa), Wilson and Clopper-Pearson, DeLong."""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

from ..core.exceptions import EvalSuiteError, InputValidationError, StatisticalTestError
from ..core.types import ArrayLike
from ..core.validation import to_numpy
from ._resolve import MetricCall, resolve_metric
from .results import ConfidenceInterval

__all__ = ["accuracy_ci", "bootstrap_ci", "proportion_ci", "roc_auc_ci"]

BootstrapMethod = Literal["percentile", "basic", "bca"]
MAX_FAILED_FRACTION = 0.1


def _check_level(level: float) -> float:
    if not (isinstance(level, (int, float)) and 0 < level < 1):
        raise InputValidationError("level must be between 0 and 1, for example 0.95.")
    return float(level)


def _check_resamples(n_resamples: int) -> int:
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 100:
        raise InputValidationError("n_resamples must be an integer of at least 100 (1000 or more recommended).")
    return int(n_resamples)


def resample_indices(
    n: int, n_resamples: int, rng: np.random.Generator, strata: Optional[NDArray[Any]] = None
) -> NDArray[np.int64]:
    """(n_resamples, n) indices drawn with replacement, optionally within strata (keeps class balance)."""
    if strata is None:
        return rng.integers(0, n, size=(n_resamples, n))
    out = np.empty((n_resamples, n), dtype=np.int64)
    pos = 0
    for value in np.unique(strata, axis=0) if strata.ndim > 1 else np.unique(strata):
        members = np.flatnonzero(np.all(strata == value, axis=1) if strata.ndim > 1 else strata == value)
        k = members.shape[0]
        out[:, pos : pos + k] = members[rng.integers(0, k, size=(n_resamples, k))]
        pos += k
    return out


def bootstrap_distribution(call: MetricCall, indices: NDArray[np.int64]) -> tuple[NDArray[np.float64], int]:
    """Metric value on each resample; resamples where the metric is undefined become NaN."""
    values = np.empty(indices.shape[0])
    failed = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zero-division warnings inside resamples are expected noise
        for b, idx in enumerate(indices):
            try:
                values[b] = call(idx)
            except EvalSuiteError:
                values[b] = np.nan
                failed += 1
    return values, failed


def _jackknife(call: MetricCall, n: int, rng: np.random.Generator, groups: int) -> NDArray[np.float64]:
    """Leave-one-out (or leave-one-group-out for large n) estimates for the BCa acceleration."""
    blocks = [np.array([i]) for i in range(n)] if n <= groups else np.array_split(rng.permutation(n), groups)
    everything = np.arange(n)
    out = np.empty(len(blocks))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, block in enumerate(blocks):
            keep = np.setdiff1d(everything, block, assume_unique=True)
            try:
                out[j] = call(keep)
            except EvalSuiteError:
                out[j] = np.nan
    return out


def bootstrap_ci(
    metric: Any,
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    *,
    y_prob: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
    level: float = 0.95,
    method: BootstrapMethod = "bca",
    n_resamples: int = 2000,
    random_state: Optional[int] = None,
    stratify: Optional[bool] = None,
    jackknife_groups: int = 1000,
    **metric_kwargs: Any,
) -> ConfidenceInterval:
    """Bootstrap confidence interval for any metric.

    ``metric`` is a metric function (``es.f1``) or its name (``"f1"``). Pass ``y_pred`` for label/value
    metrics or ``y_prob`` for probability metrics; extra keyword arguments go to the metric (for example
    ``average="macro"``). Resampling is stratified by class for classification targets unless
    ``stratify=False``, so every resample contains every class. For classification metrics that accept
    ``labels``, the full label set is fixed across resamples.

    ``method``: ``"bca"`` (bias-corrected and accelerated; default, Efron 1987), ``"percentile"`` or
    ``"basic"``. Set ``random_state`` for reproducible intervals. Resamples on which the metric is undefined
    are counted in ``params["failed_resamples"]``; more than 10% raises :class:`StatisticalTestError`.

    References: Efron B. Better bootstrap confidence intervals. JASA. 1987;82(397):171-185.
    Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall; 1993.
    """
    level = _check_level(level)
    n_resamples = _check_resamples(n_resamples)
    if method not in ("percentile", "basic", "bca"):
        raise InputValidationError("method must be 'bca', 'percentile' or 'basic'.")
    fn, name = resolve_metric(metric)
    call = MetricCall(fn, y_true, y_pred, y_prob, sample_weight, metric_kwargs)
    estimate = call()
    rng = np.random.default_rng(random_state)
    use_strata = call.categorical if stratify is None else bool(stratify)
    idx = resample_indices(call.n, n_resamples, rng, call.y_true if use_strata else None)
    boot, failed = bootstrap_distribution(call, idx)
    if failed > MAX_FAILED_FRACTION * n_resamples:
        raise StatisticalTestError(
            f"The metric was undefined on {failed} of {n_resamples} bootstrap resamples. The sample is probably "
            "too small or too imbalanced for a reliable bootstrap interval."
        )
    alpha = 1 - level
    valid = boot[~np.isnan(boot)]
    params: dict[str, Any] = {
        "n_resamples": n_resamples,
        "random_state": random_state,
        "stratified": use_strata,
        "failed_resamples": failed,
        **{k: v for k, v in metric_kwargs.items()},
    }
    if method == "percentile":
        low, high = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    elif method == "basic":
        q_low, q_high = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
        low, high = 2 * estimate - q_high, 2 * estimate - q_low
    else:
        low, high, extra = _bca(call, valid, estimate, alpha, rng, jackknife_groups)
        params.update(extra)
    return ConfidenceInterval(estimate, float(low), float(high), level, f"bootstrap-{method}", name, params)


def _bca(
    call: MetricCall,
    boot: NDArray[np.float64],
    estimate: float,
    alpha: float,
    rng: np.random.Generator,
    groups: int,
) -> tuple[float, float, dict[str, Any]]:
    # bias correction: share of the bootstrap distribution below the estimate (ties count half)
    share = (np.sum(boot < estimate) + 0.5 * np.sum(boot == estimate)) / boot.shape[0]
    if share <= 0 or share >= 1:
        warnings.warn(
            "BCa is undefined because the estimate lies outside the bootstrap distribution; "
            "returning the percentile interval.",
            RuntimeWarning,
            stacklevel=3,
        )
        low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        return float(low), float(high), {"bca_fallback": "percentile"}
    z0 = special.ndtri(share)
    jack = _jackknife(call, call.n, rng, groups)
    jack = jack[~np.isnan(jack)]
    diffs = jack.mean() - jack
    denom = 6.0 * np.sum(diffs**2) ** 1.5
    accel = float(np.sum(diffs**3) / denom) if denom > 0 else 0.0
    probs = []
    for z_alpha in (special.ndtri(alpha / 2), special.ndtri(1 - alpha / 2)):
        num = z0 + z_alpha
        probs.append(special.ndtr(z0 + num / (1 - accel * num)))
    low, high = np.quantile(boot, probs)
    return float(low), float(high), {"bias_correction": float(z0), "acceleration": accel}


def proportion_ci(
    successes: int,
    n: int,
    *,
    level: float = 0.95,
    method: Literal["wilson", "clopper-pearson", "normal"] = "wilson",
) -> ConfidenceInterval:
    """Confidence interval for a proportion ``successes / n``.

    ``"wilson"`` (default; good coverage even for small n or extreme proportions), ``"clopper-pearson"``
    (exact, conservative) or ``"normal"`` (Wald; shown for comparison, not recommended).

    References: Wilson EB. JASA. 1927;22(158):209-212. Clopper CJ, Pearson ES. Biometrika. 1934;26(4):404-413.
    Brown LD, Cai TT, DasGupta A. Interval estimation for a binomial proportion. Stat Sci. 2001;16(2):101-133.
    """
    level = _check_level(level)
    if not (isinstance(n, (int, np.integer)) and n > 0):
        raise InputValidationError("n must be a positive integer.")
    if not (0 <= successes <= n):
        raise InputValidationError("successes must be between 0 and n.")
    p = successes / n
    alpha = 1 - level
    z = special.ndtri(1 - alpha / 2)
    if method == "wilson":
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        low, high = centre - half, centre + half
    elif method == "clopper-pearson":
        low = 0.0 if successes == 0 else stats.beta.ppf(alpha / 2, successes, n - successes + 1)
        high = 1.0 if successes == n else stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    elif method == "normal":
        half = z * np.sqrt(p * (1 - p) / n)
        low, high = max(0.0, p - half), min(1.0, p + half)
    else:
        raise InputValidationError("method must be 'wilson', 'clopper-pearson' or 'normal'.")
    return ConfidenceInterval(
        p, float(low), float(high), level, method, "proportion", {"successes": int(successes), "n": int(n)}
    )


def accuracy_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    level: float = 0.95,
    method: Literal["wilson", "clopper-pearson", "normal"] = "wilson",
) -> ConfidenceInterval:
    """Analytic confidence interval for accuracy (a proportion of correct predictions)."""
    yt = to_numpy(y_true, "y_true", allow_2d=True)
    yp = to_numpy(y_pred, "y_pred", allow_2d=True)
    if yt.shape != yp.shape:
        raise InputValidationError(f"y_true has shape {yt.shape} but y_pred has shape {yp.shape}.")
    correct = (yt == yp).all(axis=1) if yt.ndim == 2 else yt == yp
    ci = proportion_ci(int(correct.sum()), int(correct.shape[0]), level=level, method=method)
    return ConfidenceInterval(ci.estimate, ci.low, ci.high, level, method, "accuracy", ci.params)


# ---- DeLong ---------------------------------------------------------------------------------------
def delong_placements(
    y: NDArray[np.float64], scores: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """AUCs and structural components for k score columns (Sun & Xu fast DeLong).

    Returns (auc[k], v10[k, m], v01[k, n]) where m/n are the numbers of positives/negatives.
    """
    pos = scores[:, y == 1]
    neg = scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        from ..core.exceptions import MetricInputError

        raise MetricInputError("DeLong's method needs at least one positive and one negative observation.")
    tx = np.apply_along_axis(stats.rankdata, 1, pos)
    ty = np.apply_along_axis(stats.rankdata, 1, neg)
    tz = np.apply_along_axis(stats.rankdata, 1, np.concatenate([pos, neg], axis=1))
    auc = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    return auc, v10, v01


def delong_covariance(v10: NDArray[np.float64], v01: NDArray[np.float64]) -> NDArray[np.float64]:
    m, n = v10.shape[1], v01.shape[1]
    s10 = np.atleast_2d(np.cov(v10)) if m > 1 else np.zeros((v10.shape[0],) * 2)
    s01 = np.atleast_2d(np.cov(v01)) if n > 1 else np.zeros((v01.shape[0],) * 2)
    cov: NDArray[np.float64] = s10 / m + s01 / n
    return cov


def binary_scores(
    y_true: ArrayLike, y_prob: ArrayLike, pos_label: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    yt = to_numpy(y_true, "y_true")
    s = to_numpy(y_prob, "y_prob").astype(np.float64)
    if s.shape[0] != yt.shape[0]:
        raise InputValidationError(
            "y_true and y_prob must contain the same number of observations. "
            f"Received {yt.shape[0]} and {s.shape[0]}."
        )
    labels = np.unique(yt)
    if labels.shape[0] > 2:
        raise InputValidationError("DeLong's method is for binary targets.")
    if pos_label is None:
        if not set(labels.tolist()) <= {0, 1}:
            raise InputValidationError(f"Labels are {labels.tolist()}; specify pos_label=...")
        pos_label = 1
    return (yt == pos_label).astype(np.float64), s


def roc_auc_ci(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    level: float = 0.95,
    pos_label: Any = None,
) -> ConfidenceInterval:
    """DeLong confidence interval for a binary ROC AUC (normal approximation, clipped to [0, 1]).

    Reference: DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated
    receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44(3):837-845.
    Sun X, Xu W. Fast implementation of DeLong's algorithm. IEEE Signal Process Lett. 2014;21(11):1389-1393.
    """
    level = _check_level(level)
    y, s = binary_scores(y_true, y_prob, pos_label)
    auc, v10, v01 = delong_placements(y, s[None, :])
    var = float(delong_covariance(v10, v01)[0, 0])
    se = np.sqrt(var)
    z = special.ndtri(1 - (1 - level) / 2)
    a = float(auc[0])
    return ConfidenceInterval(
        a,
        max(0.0, a - z * se),
        min(1.0, a + z * se),
        level,
        "delong",
        "roc_auc",
        {"standard_error": float(se), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum())},
    )
