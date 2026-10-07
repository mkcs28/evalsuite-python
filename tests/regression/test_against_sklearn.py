from __future__ import annotations

import warnings

import numpy as np
import pytest
import sklearn.metrics as skm
from hypothesis import given, settings
from hypothesis import strategies as st

import evalsuite as es

from ..conftest import assert_close


def data(seed: int, n: int = 80, outputs: int = 1, positive: bool = False):
    rng = np.random.default_rng(seed)
    shape = (n,) if outputs == 1 else (n, outputs)
    y = rng.normal(10, 3, shape)
    if positive:
        y = np.abs(y) + 0.5
    p = y + rng.normal(0, 1.5, shape)
    if positive:
        p = np.abs(p)
    return y, p, rng.random(n) * 2


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("weighted", [False, True])
def test_single_output(seed: int, weighted: bool) -> None:
    y, p, w = data(seed, positive=True)
    w = w if weighted else None
    kw = {"sample_weight": w}
    assert_close(es.mae(y, p, **kw), skm.mean_absolute_error(y, p, **kw))
    assert_close(es.mse(y, p, **kw), skm.mean_squared_error(y, p, **kw))
    assert_close(es.rmse(y, p, **kw), skm.root_mean_squared_error(y, p, **kw))
    assert_close(es.r2(y, p, **kw), skm.r2_score(y, p, **kw))
    assert_close(es.explained_variance(y, p, **kw), skm.explained_variance_score(y, p, **kw))
    assert_close(es.mape(y, p, **kw), skm.mean_absolute_percentage_error(y, p, **kw))
    assert_close(es.msle(y, p, **kw), skm.mean_squared_log_error(y, p, **kw))
    assert_close(es.rmsle(y, p, **kw), skm.root_mean_squared_log_error(y, p, **kw))
    for alpha in (0.1, 0.5, 0.9):
        assert_close(es.quantile_loss(y, p, alpha=alpha, **kw), skm.mean_pinball_loss(y, p, alpha=alpha, **kw))
    if not weighted:
        assert_close(es.median_absolute_error(y, p), skm.median_absolute_error(y, p))
        assert_close(es.max_error(y, p), skm.max_error(y, p))


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values", [0.2, 0.3, 0.5]])
def test_multi_output(seed: int, multioutput) -> None:
    y, p, w = data(seed, outputs=3)
    for es_fn, sk_fn in (
        (es.mae, skm.mean_absolute_error),
        (es.mse, skm.mean_squared_error),
        (es.rmse, skm.root_mean_squared_error),
        (es.r2, skm.r2_score),
        (es.explained_variance, skm.explained_variance_score),
    ):
        assert_close(
            es_fn(y, p, multioutput=multioutput, sample_weight=w),
            sk_fn(y, p, multioutput=multioutput, sample_weight=w),
        )
    assert_close(
        es.median_absolute_error(y, p, multioutput=multioutput),
        skm.median_absolute_error(y, p, multioutput=multioutput),
    )


def test_own_definitions() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.5, 1.5, 3.5, 5.0])
    e = p - y
    assert_close(es.mean_bias_error(y, p), e.mean())
    assert_close(es.smape(y, p), np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p))))
    assert_close(es.rae(y, p), np.abs(e).sum() / np.abs(y - y.mean()).sum())
    assert_close(es.rse(y, p), (e**2).sum() / ((y - y.mean()) ** 2).sum())
    assert_close(es.rse(y, p), 1 - float(es.r2(y, p)))
    a = np.abs(e)
    assert_close(es.huber_loss(y, p, delta=0.6), np.mean(np.where(a <= 0.6, 0.5 * a**2, 0.6 * (a - 0.3))))
    n, k = 4, 1
    assert_close(es.adjusted_r2(y, p, n_features=k), 1 - (1 - float(es.r2(y, p))) * (n - 1) / (n - k - 1))
    # smape: a pair that is both zero contributes 0
    assert_close(es.smape([0.0, 2.0], [0.0, 2.0]), 0.0)


def test_weighted_median_definition() -> None:
    err_true = np.array([0.0, 0.0, 0.0, 0.0])
    pred = np.array([1.0, 2.0, 3.0, 100.0])
    # weight concentrated on the third observation: weighted median error is 3
    assert_close(es.median_absolute_error(err_true, pred, sample_weight=[1, 1, 5, 1]), 3.0)
    # equal weights: smallest value reaching half the weight
    assert_close(es.median_absolute_error(err_true, pred, sample_weight=[1, 1, 1, 1]), 2.0)


def test_domain_errors_are_explicit() -> None:
    with pytest.raises(es.MetricInputError, match="zero value"):
        es.mape([0.0, 1.0], [0.5, 1.0])
    with pytest.raises(es.MetricInputError, match="non-negative"):
        es.msle([-1.0, 1.0], [0.5, 1.0])
    with pytest.raises(es.MetricInputError, match="more observations"):
        es.adjusted_r2([1.0, 2.0, 3.0], [1.0, 2.0, 2.5], n_features=2)
    with pytest.raises(es.MetricInputError, match="constant"):
        es.rae([2.0, 2.0], [1.0, 3.0])


def test_constant_target_r2_warns_and_matches_sklearn() -> None:
    with pytest.warns(es.UndefinedMetricWarning):
        v = es.r2([3.0, 3.0, 3.0], [3.0, 3.0, 3.0])
    assert float(v) == 1.0
    with pytest.warns(es.UndefinedMetricWarning):
        v = es.r2([3.0, 3.0, 3.0], [2.0, 3.0, 4.0])
    assert float(v) == 0.0


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.tuples(st.floats(-1e3, 1e3), st.floats(-1e3, 1e3), st.floats(0.01, 10)), min_size=2, max_size=50)
)
def test_property_core_regression(rows) -> None:
    y = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows])
    w = np.array([r[2] for r in rows])
    assert_close(es.mae(y, p, sample_weight=w), skm.mean_absolute_error(y, p, sample_weight=w), tol=1e-8)
    assert_close(es.mse(y, p, sample_weight=w), skm.mean_squared_error(y, p, sample_weight=w), tol=1e-7)
    assert float(es.rmse(y, p)) >= float(es.mae(y, p)) - 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if np.ptp(y) > 1e-6:
            assert_close(es.r2(y, p, sample_weight=w), skm.r2_score(y, p, sample_weight=w), tol=1e-6)
