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
