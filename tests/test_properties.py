"""Property-based tests (Hypothesis): invariants that must hold for any input, not just hand-picked examples."""

import contextlib
import math

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra.numpy import arrays  # noqa: E402

import evalsuite as es  # noqa: E402

FAST = settings(max_examples=60, deadline=None)
labels = st.integers(min_value=5, max_value=60).flatmap(
    lambda n: st.tuples(
        arrays(np.int64, n, elements=st.integers(0, 1)), arrays(np.int64, n, elements=st.integers(0, 1))
    )
)
# probabilities on a 1/1000 grid: 1 − s is then exact, so tie structure is preserved under flipping
probs = st.integers(min_value=5, max_value=60).flatmap(
    lambda n: st.tuples(
        arrays(np.int64, n, elements=st.integers(0, 1)),
        arrays(np.int64, n, elements=st.integers(0, 1000)).map(lambda a: a / 1000.0),
    )
)


@FAST
@given(labels)
def test_classification_metrics_lie_in_their_range(yp):
    y, p = yp
    for fn in (es.accuracy, es.balanced_accuracy, es.f1, es.precision, es.recall, es.jaccard):
        try:
            v = fn(y, p).value
        except es.EvalSuiteError:
            continue
        assert 0.0 <= v <= 1.0 or math.isnan(v)
    with contextlib.suppress(es.EvalSuiteError):
        assert -1.0 <= es.mcc(y, p).value <= 1.0


@FAST
@given(labels)
def test_accuracy_is_symmetric_and_perfect_on_itself(yp):
    y, p = yp
    assert es.accuracy(y, p).value == es.accuracy(p, y).value
    assert es.accuracy(y, y).value == 1.0


@FAST
@given(probs)
def test_roc_auc_flips_with_the_scores(yp):
    y, s = yp
    if y.min() == y.max():
        return
    a = es.roc_auc(y, y_prob=s).value
    b = es.roc_auc(y, y_prob=1 - s).value
    assert a + b == pytest.approx(1.0, abs=1e-12)


@FAST
@given(probs)
def test_brier_and_log_loss_are_non_negative(yp):
    y, s = yp
    assert es.brier_score(y, y_prob=s).value >= 0
    with contextlib.suppress(es.EvalSuiteError):
        assert es.log_loss(y, y_prob=s, labels=[0, 1]).value >= 0


@FAST
@given(
    st.integers(5, 60).flatmap(
        lambda n: st.tuples(
            arrays(np.float64, n, elements=st.floats(-1e6, 1e6, allow_nan=False)),
            arrays(np.float64, n, elements=st.floats(-1e6, 1e6, allow_nan=False)),
        )
    )
)
def test_regression_error_identities(yp):
    y, p = yp
    mse = es.mse(y, p).value
    assert mse >= 0 and es.mae(y, p).value >= 0
    assert es.rmse(y, p).value == pytest.approx(math.sqrt(mse), rel=1e-9, abs=1e-12)
    assert es.mae(y, y).value == 0


@FAST
@given(st.integers(20, 200), st.floats(0.05, 0.3), st.integers(0, 10_000))
def test_conformal_threshold_is_an_order_statistic(n, alpha, seed):
    s = np.random.default_rng(seed).random(n)
    q = es.conformal_quantile(s, alpha)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        assert q == math.inf
    else:
        assert q == np.sort(s)[k - 1]


@FAST
@given(st.integers(2, 8), st.integers(3, 25), st.integers(0, 10_000))
def test_nemenyi_ranks_average_to_the_middle(k, n, seed):
    x = np.random.default_rng(seed).random((n, k))
    r = es.nemenyi_test(x)
    assert np.mean(r.average_ranks) == pytest.approx((k + 1) / 2)
    assert np.allclose(r.p_values, r.p_values.T)
