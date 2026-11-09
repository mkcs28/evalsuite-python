"""Hypothesis tests for comparing scores, groups and categorical outcomes.

Each test returns a :class:`TestResult` with the statistic, p-value, an effect size in ``estimate`` (named in
``params["effect_size"]``) and, where standard, a confidence interval. Computation is delegated to SciPy, whose
implementations are the reference; EvalSuite adds input validation, effect sizes and consistent results.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scipy import special, stats
else:
    from .._lazy import lazy_module

    special = lazy_module("scipy.special")
    stats = lazy_module("scipy.stats")

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
    a = to_numpy(x, name).astype(np.float64, copy=False)
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


def _t_pvalue(t: float, df: float, alternative: str) -> float:
    """Student-t p-value for statistic ``t`` (identical to SciPy's t-tests)."""
    if alternative == "two-sided":
        return min(2.0 * float(special.stdtr(df, -abs(t))), 1.0)
    if alternative == "less":
        return float(special.stdtr(df, t))
    return float(special.stdtr(df, -t))


def _z_pvalue(z: float, alternative: str) -> float:
    if alternative == "two-sided":
        return min(2.0 * float(special.ndtr(-abs(z))), 1.0)
    if alternative == "less":
        return float(special.ndtr(z))
    return float(special.ndtr(-z))


def _rank_sums(
    values: NDArray[np.float64], labels: NDArray[np.intp], k: int
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Sum of average ranks (as ``scipy.stats.rankdata``) per label 0..k-1, and the tie-group sizes.

    One unstable sort: tied values share their group's average rank, so their order does not matter, and the
    per-label sums come from ``bincount`` without scattering ranks back to the input order."""
    order = np.argsort(values)
    sv = values[order]
    n = sv.shape[0]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(sv[1:], sv[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    sizes = np.diff(np.r_[starts, n])
    avg = starts + (sizes + 1) / 2.0
    ranks = avg[np.cumsum(new) - 1]
    return np.bincount(labels[order], weights=ranks, minlength=k), sizes


def _signed_rank(nz: NDArray[np.float64]) -> tuple[float, float, NDArray[np.float64]]:
    """Wilcoxon signed-rank sums of the non-zero differences: (T+, T-, tie-group sizes)."""
    sums, sizes = _rank_sums(np.abs(nz), (nz > 0).astype(np.intp), 2)
    return float(sums[1]), float(sums[0]), sizes.astype(np.float64)


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
    _alt(alternative)
    x, y = _vec(a, "a"), _vec(b, "b")
    n1, n2 = x.shape[0], y.shape[0]
    m1, m2 = float(x.mean()), float(y.mean())
    # Two-pass variances (as NumPy/SciPy compute them), each sample read once more.
    v1 = float(np.dot(x - m1, x - m1)) / (n1 - 1)
    v2 = float(np.dot(y - m2, y - m2)) / (n2 - 1)
    if v1 == 0 and v2 == 0:
        raise StatisticalTestError("The t-test is undefined: both samples have zero variance.")
    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)) if se else math.nan
    t = (m1 - m2) / se
    # Student-t tail probabilities directly (what scipy.stats.ttest_ind returns), without re-reading the data.
    p = _t_pvalue(t, df, alternative)
    diff = m1 - m2
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return TestResult(
        "welch-t" if not equal_var else "student-t",
        float(t),
        p,
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
    _alt(alternative)
    d = x - y
    n = d.shape[0]
    diff = float(d.mean())
    c = d - diff
    sd = math.sqrt(float(np.dot(c, c)) / (n - 1))
    if sd == 0:
        raise StatisticalTestError("The paired t-test is undefined: every difference is identical.")
    t = diff / (sd / math.sqrt(n))
    return TestResult(
        "paired-t",
        t,
        _t_pvalue(t, n - 1, alternative),
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
    _alt(alternative)
    n1, n2 = x.shape[0], y.shape[0]
    labels = np.zeros(n1 + n2, dtype=np.intp)
    labels[n1:] = 1
    sums, sizes = _rank_sums(np.concatenate([x, y]), labels, 2)
    u = float(sums[0]) - n1 * (n1 + 1) / 2
    ties = sizes[sizes > 1].astype(np.float64)
    if (n1 <= 8 or n2 <= 8) and ties.size == 0:
        # SciPy's method="auto" uses the exact null distribution here.
        p = float(stats.mannwhitneyu(x, y, alternative=alternative, method="exact").pvalue)
    else:
        # Tie-corrected normal approximation with continuity correction, as scipy.stats.mannwhitneyu.
        n = n1 + n2
        s_ = math.sqrt(n1 * n2 / 12 * ((n + 1) - float(np.sum(ties**3 - ties)) / (n * (n - 1))))
        num = u - n1 * n2 / 2
        sign = 1.0 if alternative == "greater" else -1.0 if alternative == "less" else float(np.sign(num))
        num -= 0.5 * sign
        p = _z_pvalue(num / s_, alternative) if s_ > 0 else math.nan
    return TestResult(
        "mann-whitney",
        u,
        p,
        alternative=alternative,
        estimate=2 * u / (n1 * n2) - 1,
        params={"effect_size": "rank-biserial correlation", "auc": u / (n1 * n2), "n_a": n1, "n_b": n2},
    )


def _wilcoxon_fast(
    d: NDArray[np.float64],
    nz: NDArray[np.float64],
    t_plus: float,
    t_minus: float,
    ties: NDArray[np.float64],
    alternative: str,
    zero_method: str,
) -> tuple[float, Optional[float]]:
    """SciPy's ``wilcoxon`` (method="auto", correction=False) from ranks already computed: the exact null
    distribution without ties or zeros, the tie-corrected normal approximation otherwise. Returns p=None where
    SciPy would run a permutation test (ties or zeros with n <= 13), other zero methods, or old SciPy. As in
    SciPy, samples of more than 50 pairs always use the normal approximation."""
    stat = min(t_plus, t_minus) if alternative == "two-sided" else t_plus
    if zero_method != "wilcox":
        return stat, None
    n_zero = d.shape[0] - nz.shape[0]
    has_ties = bool(np.any(ties > 1))
    count = nz.shape[0]
    if d.shape[0] <= 50 and not (has_ties or n_zero > 0):
        try:
            from scipy.stats._wilcoxon import WilcoxonDistribution
        except ImportError:  # SciPy < 1.12
            return stat, None
        dist = WilcoxonDistribution(np.asarray(count))
        if alternative == "less":
            p = float(dist.cdf(np.ceil(t_plus)))
        elif alternative == "greater":
            p = float(dist.sf(np.floor(t_plus)))
        else:
            p = float(np.clip(2 * min(dist.sf(np.floor(t_plus)), dist.cdf(np.ceil(t_plus))), 0, 1))
        return stat, p
    if d.shape[0] <= 13 and (has_ties or n_zero > 0):
        return stat, None
    mn = count * (count + 1.0) * 0.25
    se = math.sqrt((count * (count + 1.0) * (2.0 * count + 1.0) - float(np.sum(ties**3 - ties)) / 2) / 24)
    return stat, _z_pvalue((t_plus - mn) / se, alternative)


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
    _alt(alternative)
    t_plus, t_minus, ties = _signed_rank(nz)
    stat, p = _wilcoxon_fast(d, nz, t_plus, t_minus, ties, alternative, zero_method)
    if p is None:
        res = stats.wilcoxon(x, y, alternative=alternative, zero_method=zero_method)
        stat, p = float(res.statistic), float(res.pvalue)
    return TestResult(
        "wilcoxon",
        stat,
        p,
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
    sizes_g = np.array([g.shape[0] for g in gs])
    n = int(sizes_g.sum())
    labels = np.repeat(np.arange(len(gs), dtype=np.intp), sizes_g)
    sums, ties = _rank_sums(np.concatenate(gs), labels, len(gs))
    # H with the tie correction 1 − Σ(t³ − t)/(n³ − n), as scipy.stats.kruskal.
    t = ties[ties > 1].astype(np.float64)
    correction = 1.0 - float(np.sum(t**3 - t)) / (n**3 - n)
    if correction == 0:
        raise StatisticalTestError("The Kruskal–Wallis test is undefined: all values are identical.")
    h = (12.0 / (n * (n + 1)) * float(np.sum(sums**2 / sizes_g)) - 3 * (n + 1)) / correction
    return TestResult(
        "kruskal-wallis",
        h,
        float(special.chdtrc(len(gs) - 1, h)),
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
    k = len(gs)
    if k > 32:
        res = stats.friedmanchisquare(*gs)
        chi, p = float(res.statistic), float(res.pvalue)
    else:
        # Within-block average ranks by pairwise comparison (k is small): rank = 1 + #less + (#equal − 1)/2,
        # and each tie group of size t contributes Σ(eq² − 1) = t³ − t; the statistic and tie correction are
        # those of scipy.stats.friedmanchisquare.
        data = np.stack(gs, axis=1)
        less = np.zeros(data.shape)
        eq = np.zeros(data.shape)
        for j in range(k):
            col = data[:, j : j + 1]
            less += col < data
            eq += col == data
        ranks = 1.0 + less + (eq - 1.0) / 2.0
        r = ranks.sum(axis=0)
        c = 1.0 - float(np.sum(eq * eq - 1.0)) / (n * k * (k * k - 1))
        if c == 0:
            raise StatisticalTestError("The Friedman test is undefined: every block has all values tied.")
        chi = (12.0 / (n * k * (k + 1)) * float(np.dot(r, r)) - 3.0 * n * (k + 1)) / c
        p = float(special.chdtrc(k - 1, chi))
    return TestResult(
        "friedman",
        chi,
        p,
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
    # Same computation as scipy.stats.chi2_contingency, but the expected table is built once and serves both
    # the (possibly Yates-corrected) test statistic and the uncorrected statistic behind Cramér's V.
    n = float(t.sum())
    expected = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
    dof = (t.shape[0] - 1) * (t.shape[1] - 1)
    dev = t - expected
    uncorrected = float(np.sum(dev * dev / expected))
    if dof == 0:
        chi2, p = 0.0, 1.0
    else:
        if correction and dof == 1:
            adj = np.abs(dev) - np.minimum(0.5, np.abs(dev))
            chi2 = float(np.sum(adj * adj / expected))
        else:
            chi2 = uncorrected
        p = float(special.chdtrc(dof, chi2))
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


def _fisher_p(a: int, b: int, c: int, d: int, alternative: str) -> float:
    """Fisher's exact test p-value for [[a, b], [c, d]] from the hypergeometric pmf in log space.

    Only tables within 40 standard deviations of the mode are summed: anything farther has a probability below
    e^-800 relative to the mode and underflows to zero anyway. Two-sided: tables no more probable than the
    observed one (relative tolerance 1e-7, as R's fisher.test). Matches scipy.stats.fisher_exact to ~1e-12."""
    n1, n2, n = a + b, c + d, a + c
    total = n1 + n2
    lo, hi = max(0, n - n2), min(n, n1)
    mode = (n + 1) * (n1 + 1) // (total + 2)
    sd = math.sqrt(max(n * (n1 / total) * (n2 / total) * (total - n) / max(total - 1, 1), 1.0))
    wlo, whi = max(lo, int(mode - 40 * sd) - 2), min(hi, int(mode + 40 * sd) + 2)
    x = np.arange(wlo, whi + 1, dtype=np.float64)
    gl = special.gammaln
    lp = -(gl(x + 1) + gl(n1 - x + 1) + gl(n - x + 1) + gl(n2 - n + x + 1))
    w = np.exp(lp - lp.max())
    tot = float(w.sum())
    i = a - wlo
    if alternative == "less":
        p = 0.0 if i < 0 else 1.0 if i >= w.shape[0] else float(w[: i + 1].sum()) / tot
    elif alternative == "greater":
        p = 1.0 if i < 0 else 0.0 if i >= w.shape[0] else float(w[i:].sum()) / tot
    else:
        p = 0.0 if not 0 <= i < w.shape[0] else float(w[w <= w[i] * (1 + 1e-7)].sum()) / tot
    return min(p, 1.0)


def fisher_exact_test(table: ArrayLike, *, alternative: Alternative = "two-sided") -> TestResult:
    """Fisher's exact test for a 2 × 2 table. ``estimate`` is the sample odds ratio (a·d)/(b·c).

    Reference: Fisher RA. J R Stat Soc. 1922;85(1):87-94.
    """
    t = _table(table)
    if t.shape != (2, 2):
        raise InputValidationError("Fisher's exact test here is for 2 × 2 tables.")
    if not np.all(t == np.round(t)):
        raise InputValidationError("Fisher's exact test needs integer counts.")
    _alt(alternative)
    a, b, c, d = (int(v) for v in t.ravel())
    if 0 in (a + b, c + d, a + c, b + d):
        odds, p = math.nan, 1.0  # a zero margin: as scipy.stats.fisher_exact
    else:
        odds = a * d / (b * c) if b > 0 and c > 0 else math.inf
        p = _fisher_p(a, b, c, d, alternative)
    return TestResult(
        "fisher-exact",
        float(odds),
        float(p),
        alternative=alternative,
        estimate=float(odds),
        params={"effect_size": "odds ratio"},
    )
