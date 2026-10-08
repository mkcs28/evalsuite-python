"""Hypothesis tests for comparing scores, groups and categorical outcomes.

Each test returns a :class:`TestResult` with the statistic, p-value, an effect size in ``estimate`` (named in
``params["effect_size"]``) and, where standard, a confidence interval. Computation is delegated to SciPy, whose
implementations are the reference; EvalSuite adds input validation, effect sizes and consistent results.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from ..core.exceptions import InputValidationError, StatisticalTestError
from ..core.types import ArrayLike
from ..core.validation import check_finite, to_numpy
from .intervals import _check_level
from .results import ConfidenceInterval, TestResult

__all__ = [
    "chi_square_test",
    "fisher_exact_test",
    "friedman_test",
    "kruskal_wallis_test",
    "mann_whitney_test",
    "paired_t_test",
    "shapiro_wilk_test",
    "t_test",
    "wilcoxon_test",
]

Alternative = Literal["two-sided", "greater", "less"]
_ALTS = ("two-sided", "greater", "less")


def _vec(x: ArrayLike, name: str, min_n: int = 2) -> NDArray[np.float64]:
    a = to_numpy(x, name).astype(np.float64)
    check_finite(a, name)
    if a.ndim != 1:
        raise InputValidationError(f"{name} must be one-dimensional.")
    if a.shape[0] < min_n:
        raise StatisticalTestError(f"{name} needs at least {min_n} observations; got {a.shape[0]}.")
    return a


def _alt(alternative: str) -> str:
    if alternative not in _ALTS:
        raise InputValidationError("alternative must be 'two-sided', 'greater' or 'less'.")
    return alternative


def _mean_diff_ci(
    diff: float, se: float, df: float, level: float, alternative: str, name: str
) -> ConfidenceInterval:
    if alternative == "two-sided":
        q = stats.t.ppf(1 - (1 - level) / 2, df)
        lo, hi = diff - q * se, diff + q * se
    elif alternative == "greater":
        lo, hi = diff - stats.t.ppf(level, df) * se, math.inf
    else:
        lo, hi = -math.inf, diff + stats.t.ppf(level, df) * se
    return ConfidenceInterval(diff, lo, hi, level, "t", name, {"df": float(df), "se": float(se)})


def t_test(
    a: ArrayLike,
    b: ArrayLike,
    *,
    equal_var: bool = False,
    alternative: Alternative = "two-sided",
    level: float = 0.95,
) -> TestResult:
    """Two-sample t-test for mean(a) − mean(b). Welch's test by default (``equal_var=False``), which does not
    assume equal variances; ``equal_var=True`` gives Student's test. ``estimate`` is the mean difference with a
    confidence interval; ``params["cohens_d"]`` is the standardised effect size.

    References: Welch BL. Biometrika. 1947;34(1-2):28-35. Student. Biometrika. 1908;6(1):1-25.
    """
    level = _check_level(level)
    x, y = _vec(a, "a"), _vec(b, "b")
    if x.var() == 0 and y.var() == 0:
        raise StatisticalTestError("The t-test is undefined: both samples have zero variance.")
    res = stats.ttest_ind(x, y, equal_var=equal_var, alternative=_alt(alternative))
    n1, n2 = x.shape[0], y.shape[0]
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)) if se else math.nan
    diff = float(x.mean() - y.mean())
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return TestResult(
        "welch-t" if not equal_var else "student-t",
        float(res.statistic),
        float(res.pvalue),
        alternative=alternative,
        estimate=diff,
        ci=_mean_diff_ci(diff, se, df, level, alternative, "mean_difference"),
        params={
            "df": float(df),
            "effect_size": "mean difference",
            "cohens_d": diff / pooled if pooled else math.nan,
            "n_a": n1,
            "n_b": n2,
        },
    )


def paired_t_test(
    a: ArrayLike, b: ArrayLike, *, alternative: Alternative = "two-sided", level: float = 0.95
) -> TestResult:
    """Paired t-test for the mean of a − b (for example per-fold scores of two models on the same folds).
    ``estimate`` is the mean difference with a confidence interval; ``params["cohens_dz"]`` is d_z.
    """
    level = _check_level(level)
    x, y = _vec(a, "a"), _vec(b, "b")
    if x.shape != y.shape:
        raise InputValidationError(f"Paired samples must have the same length; got {x.shape[0]} and {y.shape[0]}.")
    d = x - y
    sd = d.std(ddof=1)
    if sd == 0:
        raise StatisticalTestError("The paired t-test is undefined: every difference is identical.")
    res = stats.ttest_rel(x, y, alternative=_alt(alternative))
    n = d.shape[0]
    diff = float(d.mean())
    return TestResult(
        "paired-t",
        float(res.statistic),
        float(res.pvalue),
        alternative=alternative,
        estimate=diff,
        ci=_mean_diff_ci(diff, sd / math.sqrt(n), n - 1, level, alternative, "mean_difference"),
        params={"df": n - 1, "effect_size": "mean difference", "cohens_dz": diff / sd, "n": n},
    )


def mann_whitney_test(a: ArrayLike, b: ArrayLike, *, alternative: Alternative = "two-sided") -> TestResult:
    """Mann–Whitney U (Wilcoxon rank-sum) test. ``statistic`` is U for ``a``; ``estimate`` is the
    rank-biserial correlation r = 2U/(n₁n₂) − 1 (equal to Cliff's delta), and ``params["auc"]`` = U/(n₁n₂)
    = P(a > b) + ½P(a = b).

    References: Mann HB, Whitney DR. Ann Math Stat. 1947;18(1):50-60. Kerby DS. Compr Psychol. 2014;3:11.IT.3.1.
    """
    x, y = _vec(a, "a", 1), _vec(b, "b", 1)
    res = stats.mannwhitneyu(x, y, alternative=_alt(alternative), method="auto")
    u = float(res.statistic)
    n1, n2 = x.shape[0], y.shape[0]
    return TestResult(
        "mann-whitney",
        u,
        float(res.pvalue),
        alternative=alternative,
        estimate=2 * u / (n1 * n2) - 1,
        params={"effect_size": "rank-biserial correlation", "auc": u / (n1 * n2), "n_a": n1, "n_b": n2},
    )


def wilcoxon_test(
    a: ArrayLike,
    b: ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
) -> TestResult:
    """Wilcoxon signed-rank test for paired samples (a − b). ``estimate`` is the matched-pairs rank-biserial
    correlation (T⁺ − T⁻) / (T⁺ + T⁻) over the non-zero differences.

    Reference: Wilcoxon F. Biometrics Bull. 1945;1(6):80-83.
    """
    x, y = _vec(a, "a", 1), _vec(b, "b", 1)
    if x.shape != y.shape:
        raise InputValidationError(f"Paired samples must have the same length; got {x.shape[0]} and {y.shape[0]}.")
    d = x - y
    nz = d[d != 0]
    if nz.shape[0] == 0:
        raise StatisticalTestError("The Wilcoxon test is undefined: every difference is zero.")
    res = stats.wilcoxon(x, y, alternative=_alt(alternative), zero_method=zero_method)
    ranks = stats.rankdata(np.abs(nz))
    t_plus, t_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return TestResult(
        "wilcoxon",
        float(res.statistic),
        float(res.pvalue),
        alternative=alternative,
        estimate=float((t_plus - t_minus) / (t_plus + t_minus)),
        params={
            "effect_size": "matched-pairs rank-biserial correlation",
            "n": int(d.shape[0]),
            "n_nonzero": int(nz.shape[0]),
            "zero_method": zero_method,
        },
    )


def _groups(groups: tuple[ArrayLike, ...], min_groups: int, what: str) -> list[NDArray[np.float64]]:
    if len(groups) < min_groups:
        raise InputValidationError(f"{what} needs at least {min_groups} groups; got {len(groups)}.")
    return [_vec(g, f"group {i + 1}", 1) for i, g in enumerate(groups)]


def kruskal_wallis_test(*groups: ArrayLike) -> TestResult:
    """Kruskal–Wallis H test: do two or more independent groups come from the same distribution?
    ``estimate`` is epsilon-squared, H / ((n² − 1)/(n + 1)).

    References: Kruskal WH, Wallis WA. JASA. 1952;47(260):583-621. Tomczak M, Tomczak E. Trends Sport Sci.
    2014;1(21):19-25.
    """
    gs = _groups(groups, 2, "The Kruskal–Wallis test")
    res = stats.kruskal(*gs)
    n = sum(g.shape[0] for g in gs)
    h = float(res.statistic)
    return TestResult(
        "kruskal-wallis",
        h,
        float(res.pvalue),
        estimate=h / ((n**2 - 1) / (n + 1)),
        params={"df": len(gs) - 1, "effect_size": "epsilon-squared", "n": n, "k": len(gs)},
    )


def friedman_test(*groups: ArrayLike) -> TestResult:
    """Friedman test for three or more related samples (for example several models scored on the same
    datasets or folds; each argument is one model's scores). ``estimate`` is Kendall's W = χ² / (n (k − 1)).

    References: Friedman M. JASA. 1937;32(200):675-701. Demšar J. Statistical comparisons of classifiers over
    multiple data sets. JMLR. 2006;7:1-30.
    """
    gs = _groups(groups, 3, "The Friedman test")
    n = gs[0].shape[0]
    if any(g.shape[0] != n for g in gs):
        raise InputValidationError("Every sample in the Friedman test needs the same number of blocks.")
    if n < 2:
        raise StatisticalTestError("The Friedman test needs at least two blocks.")
    res = stats.friedmanchisquare(*gs)
    k = len(gs)
    chi = float(res.statistic)
    return TestResult(
        "friedman",
        chi,
        float(res.pvalue),
        estimate=chi / (n * (k - 1)),
        params={"df": k - 1, "effect_size": "Kendall's W", "n_blocks": n, "k": k},
    )


def shapiro_wilk_test(x: ArrayLike) -> TestResult:
    """Shapiro–Wilk test of normality (3 ≤ n ≤ 5000 for an accurate p-value).

    Reference: Shapiro SS, Wilk MB. Biometrika. 1965;52(3-4):591-611.
    """
    a = _vec(x, "x", 3)
    if np.ptp(a) == 0:
        raise StatisticalTestError("The Shapiro–Wilk test is undefined for constant data.")
    res = stats.shapiro(a)
    return TestResult("shapiro-wilk", float(res.statistic), float(res.pvalue), params={"n": int(a.shape[0])})


def _table(table: ArrayLike) -> NDArray[np.float64]:
    t = np.asarray(table, dtype=np.float64)
    if t.ndim != 2 or t.shape[0] < 2 or t.shape[1] < 2:
        raise InputValidationError("Give a contingency table with at least 2 rows and 2 columns.")
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise InputValidationError("Contingency table counts must be finite and non-negative.")
    return t


def chi_square_test(table: ArrayLike, *, correction: bool = True) -> TestResult:
    """Pearson's χ² test of independence for an r × c contingency table. Yates' continuity correction is
    applied to 2 × 2 tables unless ``correction=False``. ``estimate`` is Cramér's V (from the uncorrected
    statistic, as in :func:`cramers_v`); ``params["expected"]``
    holds the expected counts (the approximation is doubtful when many are below 5).

    Reference: Pearson K. Philos Mag. 1900;50(302):157-175. Cramér H. Mathematical Methods of Statistics. 1946.
    """
    t = _table(table)
    if np.any(t.sum(axis=0) == 0) or np.any(t.sum(axis=1) == 0):
        raise StatisticalTestError("The χ² test is undefined when a row or column total is zero.")
    chi2, p, dof, expected = stats.chi2_contingency(t, correction=correction)
    n = t.sum()
    uncorrected = stats.chi2_contingency(t, correction=False)[0]
    v = math.sqrt((uncorrected / n) / (min(t.shape) - 1))
    return TestResult(
        "chi-square",
        float(chi2),
        float(p),
        estimate=v,
        params={
            "df": int(dof),
            "effect_size": "Cramér's V",
            "expected": np.asarray(expected).tolist(),
            "yates_correction": bool(correction and t.shape == (2, 2)),
            "min_expected": float(np.min(expected)),
        },
    )


def fisher_exact_test(table: ArrayLike, *, alternative: Alternative = "two-sided") -> TestResult:
    """Fisher's exact test for a 2 × 2 table. ``estimate`` is the sample odds ratio (a·d)/(b·c).

    Reference: Fisher RA. J R Stat Soc. 1922;85(1):87-94.
    """
    t = _table(table)
    if t.shape != (2, 2):
        raise InputValidationError("Fisher's exact test here is for 2 × 2 tables.")
    if not np.all(t == np.round(t)):
        raise InputValidationError("Fisher's exact test needs integer counts.")
    odds, p = stats.fisher_exact(t.astype(np.int64), alternative=_alt(alternative))  # tuple in every SciPy
    return TestResult(
        "fisher-exact",
        float(odds),
        float(p),
        alternative=alternative,
        estimate=float(odds),
        params={"effect_size": "odds ratio"},
    )
