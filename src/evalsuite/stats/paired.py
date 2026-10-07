"""Paired tests for comparing two models evaluated on the same observations."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from scipy import special, stats

from ..core.exceptions import InputValidationError
from ..core.types import ArrayLike
from ..core.validation import to_numpy
from ._resolve import MetricCall, resolve_metric
from .intervals import (
    _check_level,
    _check_resamples,
    binary_scores,
    bootstrap_distribution,
    delong_covariance,
    delong_placements,
    resample_indices,
)
from .results import ConfidenceInterval, TestResult

__all__ = ["delong_test", "mcnemar_test", "paired_bootstrap_test"]


def mcnemar_test(
    y_true: ArrayLike,
    y_pred_a: ArrayLike,
    y_pred_b: ArrayLike,
    *,
    exact: Optional[bool] = None,
    correction: bool = True,
) -> TestResult:
    """McNemar's test: do two classifiers have the same error rate on the same observations?

    Uses only discordant pairs: b = A right & B wrong, c = A wrong & B right. ``exact=None`` (default) uses
    the exact binomial test when b + c < 25 and the chi-squared test otherwise (with Edwards' continuity
    correction unless ``correction=False``). ``estimate`` is the accuracy difference A − B.

    References: McNemar Q. Psychometrika. 1947;12(2):153-157. Edwards AL. Psychometrika. 1948;13:185-187.
    Dietterich TG. Approximate statistical tests for comparing supervised classification learning
    algorithms. Neural Comput. 1998;10(7):1895-1923.
    """
    yt = to_numpy(y_true, "y_true", allow_2d=True)
    a = to_numpy(y_pred_a, "y_pred_a", allow_2d=True)
    b_ = to_numpy(y_pred_b, "y_pred_b", allow_2d=True)
    if not (yt.shape == a.shape == b_.shape):
        raise InputValidationError(
            f"y_true, y_pred_a and y_pred_b must have the same shape; got {yt.shape}, {a.shape}, {b_.shape}."
        )
    ok_a = (yt == a).all(axis=1) if yt.ndim == 2 else yt == a
    ok_b = (yt == b_).all(axis=1) if yt.ndim == 2 else yt == b_
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    n_disc = b + c
    use_exact = n_disc < 25 if exact is None else bool(exact)
    estimate = (b - c) / yt.shape[0]
    params = {"b": b, "c": c, "n": int(yt.shape[0])}
    if n_disc == 0:
        return TestResult(
            "mcnemar-exact" if use_exact else "mcnemar",
            0.0,
            1.0,
            estimate=estimate,
            params={**params, "note": "no discordant pairs: the classifiers agree on every observation"},
        )
    if use_exact:
        p = min(1.0, 2 * stats.binom.cdf(min(b, c), n_disc, 0.5))
        return TestResult("mcnemar-exact", float(min(b, c)), float(p), estimate=estimate, params=params)
    stat = (abs(b - c) - (1 if correction else 0)) ** 2 / n_disc
    p = stats.chi2.sf(stat, 1)
    return TestResult(
        "mcnemar", float(stat), float(p), estimate=estimate, params={**params, "continuity_correction": correction}
    )


def delong_test(
    y_true: ArrayLike,
    y_prob_a: ArrayLike,
    y_prob_b: ArrayLike,
    *,
    pos_label: Any = None,
    level: float = 0.95,
) -> TestResult:
    """DeLong's test for two correlated ROC AUCs (same observations, two models). ``estimate`` is AUC_A − AUC_B
    and ``ci`` its confidence interval.

    Reference: DeLong ER, DeLong DM, Clarke-Pearson DL. Biometrics. 1988;44(3):837-845.
    """
    level = _check_level(level)
    y, sa = binary_scores(y_true, y_prob_a, pos_label)
    _, sb = binary_scores(y_true, y_prob_b, pos_label)
    if sb.shape != sa.shape:
        raise InputValidationError("y_prob_a and y_prob_b must have the same length.")
    auc, v10, v01 = delong_placements(y, np.vstack([sa, sb]))
    cov = delong_covariance(v10, v01)
    diff = float(auc[0] - auc[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        z, p = 0.0, 1.0
        se = 0.0
    else:
        se = float(np.sqrt(var))
        z = diff / se
        p = float(2 * special.ndtr(-abs(z)))
    zc = special.ndtri(1 - (1 - level) / 2)
    ci = ConfidenceInterval(diff, diff - zc * se, diff + zc * se, level, "delong", "roc_auc_difference")
    return TestResult(
        "delong",
        z,
        p,
        estimate=diff,
        ci=ci,
        params={"auc_a": float(auc[0]), "auc_b": float(auc[1]), "standard_error": se},
    )


def paired_bootstrap_test(
    metric: Any,
    y_true: ArrayLike,
    y_pred_a: Optional[ArrayLike] = None,
    y_pred_b: Optional[ArrayLike] = None,
    *,
    y_prob_a: Optional[ArrayLike] = None,
    y_prob_b: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
    n_resamples: int = 2000,
    level: float = 0.95,
    random_state: Optional[int] = None,
    stratify: Optional[bool] = None,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    **metric_kwargs: Any,
) -> TestResult:
    """Paired bootstrap test for any metric: is metric(A) − metric(B) different from 0?

    Both models are evaluated on the same resampled observations. The p-value uses the shifted bootstrap
    distribution (difference re-centred on 0 under the null), with the +1 correction so it is never 0:
    p = (1 + #{|δ* − δ̂| ≥ |δ̂|}) / (B + 1) for a two-sided test. ``ci`` is the percentile interval of δ*.

    Reference: Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall; 1993, ch. 16.
    """
    level = _check_level(level)
    n_resamples = _check_resamples(n_resamples)
    if alternative not in ("two-sided", "greater", "less"):
        raise InputValidationError("alternative must be 'two-sided', 'greater' or 'less'.")
    fn, name = resolve_metric(metric)
    by_prob = y_prob_a is not None or y_prob_b is not None
    if by_prob and (y_prob_a is None or y_prob_b is None or y_pred_a is not None or y_pred_b is not None):
        raise InputValidationError("Give either y_pred_a and y_pred_b, or y_prob_a and y_prob_b.")
    if not by_prob and (y_pred_a is None or y_pred_b is None):
        raise InputValidationError("Give either y_pred_a and y_pred_b, or y_prob_a and y_prob_b.")
    if by_prob:
        call_a = MetricCall(fn, y_true, None, y_prob_a, sample_weight, metric_kwargs)
        call_b = MetricCall(fn, y_true, None, y_prob_b, sample_weight, metric_kwargs)
    else:
        call_a = MetricCall(fn, y_true, y_pred_a, None, sample_weight, metric_kwargs)
        call_b = MetricCall(fn, y_true, y_pred_b, None, sample_weight, metric_kwargs)
    if call_a.kwargs.get("labels") is not None and call_b.kwargs.get("labels") is not None:
        shared = np.union1d(call_a.kwargs["labels"], call_b.kwargs["labels"])
        call_a.kwargs["labels"] = call_b.kwargs["labels"] = shared
    observed = call_a() - call_b()
    rng = np.random.default_rng(random_state)
    use_strata = call_a.categorical if stratify is None else bool(stratify)
    idx = resample_indices(call_a.n, n_resamples, rng, call_a.y_true if use_strata else None)
    da, fa = bootstrap_distribution(call_a, idx)
    db, fb = bootstrap_distribution(call_b, idx)
    diff = da - db
    diff = diff[~np.isnan(diff)]
    centred = diff - observed
    if alternative == "two-sided":
        extreme = np.sum(np.abs(centred) >= abs(observed))
    elif alternative == "greater":
        extreme = np.sum(centred >= observed)
    else:
        extreme = np.sum(centred <= observed)
    p = (1 + extreme) / (diff.shape[0] + 1)
    alpha = 1 - level
    low, high = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    ci = ConfidenceInterval(
        observed,
        float(low),
        float(high),
        level,
        "bootstrap-percentile",
        f"{name}_difference",
        {"n_resamples": n_resamples, "random_state": random_state},
    )
    return TestResult(
        "paired-bootstrap",
        observed,
        float(p),
        alternative=alternative,
        estimate=observed,
        ci=ci,
        params={
            "metric": name,
            "n_resamples": n_resamples,
            "random_state": random_state,
            "stratified": use_strata,
            "failed_resamples": int(max(fa, fb)),
            "metric_a": call_a(),
            "metric_b": call_b(),
            **metric_kwargs,
        },
    )
