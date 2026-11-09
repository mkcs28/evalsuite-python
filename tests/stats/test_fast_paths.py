"""The one-pass t-test and single-partition percentiles must equal SciPy / NumPy exactly."""

import numpy as np
import pytest
from scipy import stats

import evalsuite as es


@pytest.mark.parametrize("equal_var", [False, True])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize(("n1", "n2", "shift"), [(2, 2, 0.0), (3, 30, 0.4), (500, 400, -0.1), (50, 60, 30.0)])
def test_t_test_matches_scipy(equal_var, alternative, n1, n2, shift):
    rng = np.random.default_rng(n1 + n2)
    a, b = rng.normal(0, 1, n1), rng.normal(shift, 2.0, n2)
    got = es.t_test(a, b, equal_var=equal_var, alternative=alternative)
    ref = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
    assert got.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert got.p_value == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("n", [1, 2, 3, 10, 1001])
@pytest.mark.parametrize("q", [0, 12.5, 50, 95, 99.9, 100])
def test_latency_percentiles_match_numpy(n, q):
    x = np.random.default_rng(n).lognormal(0, 0.5, n)
    r = es.latency_percentiles(x, percentile=q)
    assert float(r) == pytest.approx(np.percentile(x, q), rel=1e-12)
    for key, p in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99)):
        assert r.params[key] == pytest.approx(np.percentile(x, p), rel=1e-12)
    assert r.params["min"] == x.min() and r.params["max"] == x.max()


def _pairs(rng, n, kind):
    a = rng.normal(0, 1, n)
    b = a + rng.normal(0.1, 1, n)
    if kind == "ties":
        a, b = np.round(a, 1), np.round(b, 1)
    if kind == "zeros":
        b[: max(1, n // 5)] = a[: max(1, n // 5)]
    return a, b


@pytest.mark.parametrize("n", [5, 13, 14, 50, 51, 400])
@pytest.mark.parametrize("kind", ["cont", "ties", "zeros"])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("zero_method", ["wilcox", "pratt", "zsplit"])
def test_wilcoxon_matches_scipy(n, kind, alternative, zero_method):
    a, b = _pairs(np.random.default_rng(n), n, kind)
    got = es.wilcoxon_test(a, b, alternative=alternative, zero_method=zero_method)
    ref = stats.wilcoxon(a, b, alternative=alternative, zero_method=zero_method)
    assert got.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert got.p_value == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_paired_t_matches_scipy(alternative):
    a, b = _pairs(np.random.default_rng(7), 300, "cont")
    got, ref = es.paired_t_test(a, b, alternative=alternative), stats.ttest_rel(a, b, alternative=alternative)
    assert got.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert got.p_value == pytest.approx(ref.pvalue, rel=1e-9)


@pytest.mark.parametrize(("n1", "n2"), [(3, 5), (8, 30), (9, 9), (300, 200)])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_mann_whitney_matches_scipy(n1, n2, ties, alternative):
    rng = np.random.default_rng(n1 * n2)
    a, b = rng.normal(0, 1, n1), rng.normal(0.3, 1, n2)
    if ties:
        a, b = np.round(a), np.round(b)
    got, ref = (
        es.mann_whitney_test(a, b, alternative=alternative),
        stats.mannwhitneyu(a, b, alternative=alternative),
    )
    assert got.statistic == ref.statistic
    assert got.p_value == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("ties", [False, True])
def test_kruskal_and_friedman_match_scipy(ties):
    rng = np.random.default_rng(11)
    gs = [rng.normal(i * 0.1, 1, 60) for i in range(4)]
    if ties:
        gs = [np.round(g) for g in gs]
    k, kr = es.kruskal_wallis_test(*gs), stats.kruskal(*gs)
    f, fr = es.friedman_test(*gs), stats.friedmanchisquare(*gs)
    assert k.statistic == pytest.approx(kr.statistic, rel=1e-12) and k.p_value == pytest.approx(
        kr.pvalue, rel=1e-9
    )
    assert f.statistic == pytest.approx(fr.statistic, rel=1e-12) and f.p_value == pytest.approx(
        fr.pvalue, rel=1e-9
    )


def test_kruskal_all_identical_is_an_error():
    with pytest.raises(es.StatisticalTestError):
        es.kruskal_wallis_test([1.0, 1.0], [1.0, 1.0])


@pytest.mark.parametrize(
    "table", [[[3, 1], [1, 3]], [[40, 10], [11, 39]], [[4000, 1000], [1100, 3900]], [[0, 5], [5, 0]]]
)
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_fisher_matches_scipy(table, alternative):
    got, ref = (
        es.fisher_exact_test(table, alternative=alternative),
        stats.fisher_exact(table, alternative=alternative),
    )
    assert got.p_value == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-12)
    assert got.statistic == ref.statistic


def test_fisher_zero_margin():
    assert es.fisher_exact_test([[0, 0], [3, 4]]).p_value == 1.0


@pytest.mark.parametrize("shape", [(2, 2), (2, 4), (5, 5)])
@pytest.mark.parametrize("correction", [True, False])
def test_chi_square_and_cramers_v_match_scipy(shape, correction):
    t = np.random.default_rng(shape[1]).integers(1, 40, shape)
    got = es.chi_square_test(t, correction=correction)
    chi2, p, _, expected = stats.chi2_contingency(t, correction=correction)
    assert got.statistic == pytest.approx(chi2, rel=1e-12) and got.p_value == pytest.approx(p, rel=1e-9)
    assert np.allclose(got.params["expected"], expected)
    assert es.cramers_v(t) == pytest.approx(stats.contingency.association(t, method="cramer"), rel=1e-12)


def test_proportion_ci_clopper_pearson_matches_beta_quantiles():
    ci = es.proportion_ci(7, 40, method="clopper-pearson")
    assert ci.low == pytest.approx(stats.beta.ppf(0.025, 7, 34), rel=1e-12)
    assert ci.high == pytest.approx(stats.beta.ppf(0.975, 8, 33), rel=1e-12)
