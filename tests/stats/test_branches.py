from __future__ import annotations

import warnings

import numpy as np
import pytest

import evalsuite as es
from evalsuite.stats._resolve import MetricCall, is_categorical


def test_bca_degenerate_and_fallback() -> None:
    # perfect predictions: every resample equals the estimate (ties count half), so BCa is defined: [1, 1]
    y = np.array([0, 1] * 30)
    ci = es.bootstrap_ci("accuracy", y, y, random_state=0, n_resamples=200)
    assert ci.low == ci.high == 1.0 and "bca_fallback" not in ci.params

    # a statistic that is strictly smaller on (almost) every resample: the number of distinct values
    def distinct(y_true, y_pred):
        return float(np.unique(y_pred).shape[0])

    x = np.arange(50, dtype=float)
    with pytest.warns(RuntimeWarning, match="percentile"):
        fallback = es.bootstrap_ci(distinct, np.zeros(50), x, random_state=0, n_resamples=200)
    assert fallback.params["bca_fallback"] == "percentile" and fallback.high < fallback.estimate


def test_bca_with_large_sample_uses_grouped_jackknife() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=1500)
    ci = es.bootstrap_ci(
        "mae", y, y + rng.normal(0, 1, 1500), random_state=0, n_resamples=200, jackknife_groups=100
    )
    assert ci.low < ci.estimate < ci.high and "acceleration" in ci.params


def test_weighted_and_multilabel_resampling() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 120)
    p = np.where(rng.random(120) < 0.8, y, 1 - y)
    w = rng.random(120)
    ci = es.bootstrap_ci("f1", y, p, sample_weight=w, random_state=0, n_resamples=200)
    assert ci.estimate == float(es.f1(y, p, sample_weight=w))
    Y = rng.integers(0, 2, (80, 3))
    P = np.where(rng.random((80, 3)) < 0.8, Y, 1 - Y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml = es.bootstrap_ci("f1", Y, P, average="micro", random_state=0, n_resamples=200)
    assert ml.params["stratified"] is True and 0 <= ml.low <= ml.high <= 1


def test_paired_tests_more_branches() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    a = np.where(rng.random(200) < 0.9, y, 1 - y)
    b = np.where(rng.random(200) < 0.6, y, 1 - y)
    less = es.paired_bootstrap_test("accuracy", y, a, b, alternative="less", random_state=0, n_resamples=300)
    assert less.p_value > 0.5
    qa = np.clip(y * 0.5 + rng.random(200) * 0.5, 0, 1)
    qb = rng.random(200)
    prob = es.paired_bootstrap_test("roc_auc", y, y_prob_a=qa, y_prob_b=qb, random_state=0, n_resamples=300)
    assert prob.estimate > 0
    with pytest.raises(es.InputValidationError, match="either"):
        es.paired_bootstrap_test("roc_auc", y, a, y_prob_b=qb)
    with pytest.raises(es.InputValidationError, match="same number"):
        es.delong_test(y, qa, qb[:-1])
    zero = es.delong_test(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.2, 0.8]), np.array([0.1, 0.9, 0.1, 0.9]))
    assert zero.p_value == 1.0  # both perfect: no variance in the difference


def test_resolve_helpers() -> None:
    assert is_categorical(np.array([0, 1, 1]))
    assert not is_categorical(np.array([0.5, 1.5]))
    assert not is_categorical(np.array([[0, 2], [1, 3]]))  # not a valid label array
    with pytest.raises(es.InputValidationError, match="same number"):
        MetricCall(es.f1, [0, 1, 1], [0, 1])
    mixed = MetricCall(es.f1, np.array([1, "a"], dtype=object), np.array(["a", 1], dtype=object))
    assert "labels" not in mixed.kwargs  # incomparable labels: left for the metric to report


def test_compare_with_weights_uses_bootstrap_for_every_metric() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 150)
    preds = {"a": np.where(rng.random(150) < 0.85, y, 1 - y), "b": np.where(rng.random(150) < 0.7, y, 1 - y)}
    r = es.compare(y, preds, metrics=["accuracy"], sample_weight=rng.random(150), random_state=0, n_resamples=200)
    assert r.tests[0]["test"] == "paired-bootstrap"
