"""Effect sizes and multiple-comparison correction."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import special

from ..core.exceptions import InputValidationError, StatisticalTestError
from ..core.types import ArrayLike
from ..core.validation import check_finite, to_numpy

__all__ = ["adjust_pvalues", "cliffs_delta", "cohens_d", "hedges_g"]


def _samples(a: ArrayLike, b: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = to_numpy(a, "a").astype(np.float64)
    y = to_numpy(b, "b").astype(np.float64)
    check_finite(x, "a")
    check_finite(y, "b")
    return x, y


def cohens_d(a: ArrayLike, b: ArrayLike, *, paired: bool = False) -> float:
    """Cohen's d for mean(a) − mean(b).

    Independent samples: difference over the pooled standard deviation. ``paired=True``: mean difference over
    the standard deviation of the differences (d_z), for per-fold or per-subject scores of two models.

    Reference: Cohen J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed. Erlbaum; 1988.
    Lakens D. Calculating and reporting effect sizes. Front Psychol. 2013;4:863.
    """
    x, y = _samples(a, b)
    if paired:
        if x.shape != y.shape:
            raise InputValidationError("Paired samples must have the same length.")
        d = x - y
        if d.shape[0] < 2:
            raise StatisticalTestError("Cohen's d needs at least two pairs.")
        sd = d.std(ddof=1)
        if sd == 0:
            raise StatisticalTestError("Cohen's d is undefined: the paired differences are all identical.")
        return float(d.mean() / sd)
    n1, n2 = x.shape[0], y.shape[0]
    if n1 < 2 or n2 < 2:
        raise StatisticalTestError("Cohen's d needs at least two observations per group.")
    pooled = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    if pooled == 0:
        raise StatisticalTestError("Cohen's d is undefined: both groups have zero variance.")
    return float((x.mean() - y.mean()) / pooled)


def hedges_g(a: ArrayLike, b: ArrayLike, *, paired: bool = False) -> float:
    """Hedges' g: Cohen's d with the exact small-sample bias correction J(df) = Γ(df/2) / (√(df/2) Γ((df−1)/2)).

    Reference: Hedges LV. Distribution theory for Glass's estimator of effect size and related estimators.
    J Educ Stat. 1981;6(2):107-128.
    """
    x, y = _samples(a, b)
    df = (x.shape[0] - 1) if paired else (x.shape[0] + y.shape[0] - 2)
    if df < 2:
        raise StatisticalTestError("Hedges' g needs at least three observations.")
    j = np.exp(special.gammaln(df / 2) - special.gammaln((df - 1) / 2)) / np.sqrt(df / 2)
    return float(j * cohens_d(a, b, paired=paired))


def cliffs_delta(a: ArrayLike, b: ArrayLike) -> float:
    """Cliff's delta: P(a > b) − P(a < b) over all pairs; a non-parametric effect size in [−1, 1].

    Computed in O((n + m) log m). Reference: Cliff N. Dominance statistics: ordinal analyses to answer ordinal
    questions. Psychol Bull. 1993;114(3):494-509.
    """
    x, y = _samples(a, b)
    ys = np.sort(y)
    less = np.searchsorted(ys, x, side="left")  # y values below each x
    greater = y.shape[0] - np.searchsorted(ys, x, side="right")  # y values above each x
    return float((less.sum() - greater.sum()) / (x.shape[0] * y.shape[0]))


def adjust_pvalues(
    p_values: ArrayLike, *, method: Literal["bonferroni", "holm", "bh", "by"] = "holm"
) -> NDArray[np.float64]:
    """Adjust p-values for multiple comparisons (same order as the input).

    ``"holm"`` (default; controls the family-wise error rate, uniformly more powerful than Bonferroni),
    ``"bonferroni"``, ``"bh"`` (Benjamini-Hochberg false discovery rate) or ``"by"``
    (Benjamini-Yekutieli, FDR under arbitrary dependence).

    References: Holm S. Scand J Stat. 1979;6(2):65-70. Benjamini Y, Hochberg Y. J R Stat Soc B.
    1995;57(1):289-300. Benjamini Y, Yekutieli D. Ann Stat. 2001;29(4):1165-1188.
    """
    p = to_numpy(p_values, "p_values").astype(np.float64)
    check_finite(p, "p_values")
    if np.any((p < 0) | (p > 1)):
        raise InputValidationError("p-values must be between 0 and 1.")
    m = p.shape[0]
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    if method == "bonferroni":
        adj = np.minimum(ranked * m, 1)
    elif method == "holm":
        adj = np.minimum(np.maximum.accumulate(ranked * (m - np.arange(m))), 1)
    elif method in ("bh", "by"):
        factor = np.sum(1.0 / np.arange(1, m + 1)) if method == "by" else 1.0
        scaled = ranked * m * factor / np.arange(1, m + 1)
        adj = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1)
    else:
        raise InputValidationError("method must be 'holm', 'bonferroni', 'bh' or 'by'.")
    out = np.empty(m)
    out[order] = adj
    return out
