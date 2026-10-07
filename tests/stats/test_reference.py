"""Statistical methods checked against statsmodels and SciPy, and against brute-force definitions."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

import evalsuite as es


# ---- McNemar ------------------------------------------------------------------------------------
def _pair(seed: int, n: int, flip: float):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    a = np.where(rng.random(n) < 0.8, y, 1 - y)
    b = np.where(rng.random(n) < flip, y, 1 - y)
    return y, a, b


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("n,flip", [(40, 0.75), (300, 0.7)])
@pytest.mark.parametrize("exact,correction", [(True, True), (False, True), (False, False)])
def test_mcnemar_matches_statsmodels(seed: int, n: int, flip: float, exact: bool, correction: bool) -> None:
    y, a, b = _pair(seed, n, flip)
    ok_a, ok_b = a == y, b == y
    table = [[np.sum(ok_a & ok_b), np.sum(ok_a & ~ok_b)], [np.sum(~ok_a & ok_b), np.sum(~ok_a & ~ok_b)]]
    ref = sm_mcnemar(table, exact=exact, correction=correction)
    ours = es.mcnemar_test(y, a, b, exact=exact, correction=correction)
    if table[0][1] + table[1][0] == 0:
        assert ours.p_value == 1.0
        return
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-10, abs=1e-12)
    assert ours.statistic == pytest.approx(ref.statistic, rel=1e-10)


def test_mcnemar_auto_switches_to_exact_and_reports_counts() -> None:
    y = np.array([1] * 20 + [0] * 20)
    a = y.copy()
    b = y.copy()
    b[:5] = 1 - b[:5]  # A right, B wrong on 5
    r = es.mcnemar_test(y, a, b)
    assert r.test == "mcnemar-exact" and r.params["b"] == 5 and r.params["c"] == 0
    assert r.estimate == pytest.approx(5 / 40)
    assert es.mcnemar_test(y, a, a).p_value == 1.0


# ---- p-value adjustment -------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize(
    "ours,theirs", [("holm", "holm"), ("bonferroni", "bonferroni"), ("bh", "fdr_bh"), ("by", "fdr_by")]
)
def test_adjust_pvalues_matches_statsmodels(seed: int, ours: str, theirs: str) -> None:
    rng = np.random.default_rng(seed)
    p = rng.random(rng.integers(1, 25)) ** 2
    np.testing.assert_allclose(es.adjust_pvalues(p, method=ours), multipletests(p, method=theirs)[1], rtol=1e-12)


# ---- proportion intervals -----------------------------------------------------------------------
@pytest.mark.parametrize("k,n", [(0, 10), (1, 10), (5, 10), (10, 10), (37, 50), (480, 500), (3, 1000)])
@pytest.mark.parametrize("ours,theirs", [("wilson", "wilson"), ("clopper-pearson", "beta"), ("normal", "normal")])
@pytest.mark.parametrize("level", [0.9, 0.95, 0.99])
def test_proportion_ci_matches_statsmodels(k: int, n: int, ours: str, theirs: str, level: float) -> None:
    ci = es.proportion_ci(k, n, level=level, method=ours)
    lo, hi = proportion_confint(k, n, alpha=1 - level, method=theirs)
    assert ci.low == pytest.approx(max(lo, 0.0), abs=1e-10)
    assert ci.high == pytest.approx(min(hi, 1.0), abs=1e-10)


def test_accuracy_ci_is_wilson_of_correct_predictions() -> None:
    y = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
    p = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
    ci = es.accuracy_ci(y, p)
    lo, hi = proportion_confint(8, 10, alpha=0.05, method="wilson")
    assert (ci.estimate, ci.low, ci.high) == pytest.approx((0.8, lo, hi))
    assert ci.format() == f"0.800 (95% CI {lo:.3f}–{hi:.3f})"


# ---- bootstrap ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("method", ["percentile", "basic", "bca"])
def test_bootstrap_matches_scipy_for_the_mean(seed: int, method: str) -> None:
    """mean_bias_error(0, x) is the mean of x, so both libraries bootstrap the same statistic."""
    rng = np.random.default_rng(seed)
    x = rng.exponential(2.0, 120)  # skewed, where BCa differs from percentile
    ours = es.bootstrap_ci(
        "mean_bias_error", np.zeros_like(x), x, method=method, n_resamples=20000, random_state=seed, stratify=False
    )
    ref = sps.bootstrap(
        (x,), np.mean, method=method, n_resamples=20000, random_state=seed + 100, vectorized=True
    ).confidence_interval
    # independent random resamples: agreement to Monte Carlo error
    assert ours.low == pytest.approx(ref.low, abs=0.04)
    assert ours.high == pytest.approx(ref.high, abs=0.04)
    assert ours.estimate == pytest.approx(x.mean())


def test_bootstrap_is_reproducible_and_records_settings() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    p = np.where(rng.random(200) < 0.7, y, rng.integers(0, 3, 200))
    a = es.bootstrap_ci(es.f1, y, p, average="macro", random_state=42, n_resamples=500)
    b = es.bootstrap_ci("f1", y, p, average="macro", random_state=42, n_resamples=500)
    assert (a.low, a.high) == (b.low, b.high)
    assert a.params["average"] == "macro" and a.params["stratified"] is True
    assert a.low <= a.estimate <= a.high
    assert a.estimate == float(es.f1(y, p, average="macro"))


def test_bootstrap_coverage_of_a_known_proportion() -> None:
    """Accuracy with true value 0.8: the 95% percentile interval should cover it about 95% of the time."""
    rng = np.random.default_rng(7)
    hits = 0
    reps = 150
    for _ in range(reps):
        y = np.ones(200, dtype=int)
        p = np.where(rng.random(200) < 0.8, 1, 0)
        ci = es.bootstrap_ci(
            "accuracy",
            y,
            p,
            method="percentile",
            n_resamples=400,
            random_state=int(rng.integers(1e9)),
            stratify=False,
        )
        hits += ci.contains(0.8)
    assert 0.88 <= hits / reps <= 0.99


def test_bootstrap_probability_metric_and_failed_resamples() -> None:
    y = np.array([0] * 30 + [1] * 30)
    s = np.r_[np.linspace(0, 0.6, 30), np.linspace(0.4, 1, 30)]
    ci = es.bootstrap_ci("roc_auc", y, y_prob=s, random_state=1, n_resamples=300)
    assert ci.params["failed_resamples"] == 0  # stratified: every resample keeps both classes
    with pytest.raises(es.StatisticalTestError, match="undefined"):
        es.bootstrap_ci(
            "roc_auc",
            np.r_[[0], np.ones(30, int)],
            y_prob=np.linspace(0, 1, 31),
            stratify=False,
            random_state=1,
            n_resamples=300,
        )


# ---- DeLong -------------------------------------------------------------------------------------
def _brute_delong(y: np.ndarray, scores: list[np.ndarray]):
    """Direct O(m·n) implementation of DeLong et al. (1988) for comparison."""
    pos, neg = y == 1, y == 0
    v10s, v01s, aucs = [], [], []
    for s in scores:
        x, z = s[pos], s[neg]
        psi = (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])
        aucs.append(psi.mean())
        v10s.append(psi.mean(axis=1))
        v01s.append(psi.mean(axis=0))
    m, n = pos.sum(), neg.sum()
    s10 = np.atleast_2d(np.cov(np.vstack(v10s)))
    s01 = np.atleast_2d(np.cov(np.vstack(v01s)))
    return np.array(aucs), s10 / m + s01 / n


@pytest.mark.parametrize("seed", range(20))
def test_delong_matches_brute_force_and_roc_auc(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 80)
    y[:2] = [0, 1]
    sa = np.round(np.clip(y * 0.4 + rng.random(80) * 0.6, 0, 1), 2)  # ties included
    sb = np.round(np.clip(y * 0.2 + rng.random(80) * 0.8, 0, 1), 2)
    aucs, cov = _brute_delong(y, [sa, sb])
    ci = es.roc_auc_ci(y, sa)
    assert ci.estimate == pytest.approx(aucs[0]) == pytest.approx(float(es.roc_auc(y, sa)))
    assert ci.params["standard_error"] == pytest.approx(np.sqrt(cov[0, 0]))
    t = es.delong_test(y, sa, sb)
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    z = (aucs[0] - aucs[1]) / se
    assert t.statistic == pytest.approx(z)
    assert t.p_value == pytest.approx(2 * sps.norm.sf(abs(z)))
    assert t.estimate == pytest.approx(aucs[0] - aucs[1])


def test_delong_ci_coverage() -> None:
    """Binormal scores with known AUC Φ(1/√2) ≈ 0.7602: 95% DeLong intervals should cover it ~95%."""
    true_auc = sps.norm.cdf(1 / np.sqrt(2))
    rng = np.random.default_rng(3)
    hits, reps = 0, 400
    for _ in range(reps):
        y = np.r_[np.zeros(100, int), np.ones(100, int)]
        s = sps.norm.cdf(np.r_[rng.normal(0, 1, 100), rng.normal(1, 1, 100)])
        hits += es.roc_auc_ci(y, s).contains(true_auc)
    assert 0.92 <= hits / reps <= 0.98


def test_delong_identical_models() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    t = es.delong_test(y, s, s)
    assert t.p_value == 1.0 and t.estimate == 0.0


# ---- effect sizes -------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(15))
def test_effect_sizes(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(1.0, 2.0, 30)
    b = rng.normal(0.0, 1.5, 25)
    pooled = np.sqrt(((29) * a.var(ddof=1) + 24 * b.var(ddof=1)) / 53)
    d = (a.mean() - b.mean()) / pooled
    assert es.cohens_d(a, b) == pytest.approx(d)
    assert es.hedges_g(a, b) == pytest.approx(d * (1 - 3 / (4 * 53 - 1)), rel=1e-3)  # Hedges' approximation
    diff = a[:25] - b
    assert es.cohens_d(a[:25], b, paired=True) == pytest.approx(diff.mean() / diff.std(ddof=1))
    brute = np.mean(np.sign(a[:, None] - b[None, :]))
    assert es.cliffs_delta(a, b) == pytest.approx(brute)


def test_cliffs_delta_with_ties_and_bounds() -> None:
    assert es.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    assert es.cliffs_delta([5, 6], [1, 2]) == 1.0
    assert es.cliffs_delta([1, 1], [1, 2]) == pytest.approx(-0.5)


# ---- paired bootstrap ---------------------------------------------------------------------------
def test_paired_bootstrap_test_behaviour() -> None:
    y, a, b = _pair(5, 400, 0.6)
    t = es.paired_bootstrap_test("f1", y, a, b, random_state=0, n_resamples=1000)
    assert t.estimate == pytest.approx(float(es.f1(y, a)) - float(es.f1(y, b)))
    assert t.p_value < 0.01 and t.ci.low > 0
    same = es.paired_bootstrap_test("f1", y, a, a, random_state=0, n_resamples=500)
    assert same.estimate == 0 and same.p_value == 1.0
    again = es.paired_bootstrap_test("f1", y, a, b, random_state=0, n_resamples=1000)
    assert again.p_value == t.p_value
    one_sided = es.paired_bootstrap_test("f1", y, a, b, random_state=0, alternative="greater")
    assert one_sided.p_value <= t.p_value


def test_paired_bootstrap_null_calibration() -> None:
    """Two equally good, independent models: p-values should be roughly uniform (few false positives)."""
    rng = np.random.default_rng(11)
    ps = []
    for _ in range(60):
        y = rng.integers(0, 2, 150)
        a = np.where(rng.random(150) < 0.75, y, 1 - y)
        b = np.where(rng.random(150) < 0.75, y, 1 - y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps.append(
                es.paired_bootstrap_test(
                    "accuracy", y, a, b, n_resamples=300, random_state=int(rng.integers(1e9))
                ).p_value
            )
    assert np.mean(np.array(ps) < 0.05) <= 0.15
