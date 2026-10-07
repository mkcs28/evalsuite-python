"""Reference tests: EvalSuite must agree with scikit-learn wherever the definitions coincide."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import sklearn.metrics as skm
from hypothesis import given, settings
from hypothesis import strategies as st

import evalsuite as es

from ..conftest import assert_close

AVERAGES_MC = ["micro", "macro", "weighted", None]


def labels_data(seed: int, n: int, k: int, weighted: bool):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    p = np.where(rng.random(n) < 0.6, y, rng.integers(0, k, n))
    w = rng.random(n) * 3 if weighted else None
    return y, p, w


def sk(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def es_(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("weighted", [False, True])
def test_binary_label_metrics(seed: int, weighted: bool) -> None:
    y, p, w = labels_data(seed, 60, 2, weighted)
    kw = {"sample_weight": w}
    assert_close(es.accuracy(y, p, **kw), skm.accuracy_score(y, p, **kw))
    assert_close(es.balanced_accuracy(y, p, **kw), skm.balanced_accuracy_score(y, p, **kw))
    assert_close(
        es.balanced_accuracy(y, p, adjusted=True, **kw),
        skm.balanced_accuracy_score(y, p, adjusted=True, **kw),
    )
    assert_close(es.precision(y, p, zero_division=0, **kw), sk(skm.precision_score, y, p, zero_division=0, **kw))
    assert_close(es.recall(y, p, zero_division=0, **kw), sk(skm.recall_score, y, p, zero_division=0, **kw))
    assert_close(es.f1(y, p, zero_division=0, **kw), sk(skm.f1_score, y, p, zero_division=0, **kw))
    assert_close(
        es.fbeta(y, p, beta=2, zero_division=0, **kw),
        sk(skm.fbeta_score, y, p, beta=2, zero_division=0, **kw),
    )
    assert_close(es.jaccard(y, p, zero_division=0, **kw), sk(skm.jaccard_score, y, p, zero_division=0, **kw))
    assert_close(es_(es.mcc, y, p, **kw), sk(skm.matthews_corrcoef, y, p, **kw))
    assert_close(es_(es.cohen_kappa, y, p, **kw), sk(skm.cohen_kappa_score, y, p, **kw))
    assert_close(es.hamming_loss(y, p, **kw), skm.hamming_loss(y, p, **kw))
    assert_close(es.confusion_matrix(y, p, **kw), skm.confusion_matrix(y, p, **kw))
    # specificity = recall of the negative class
    assert_close(
        es.specificity(y, p, zero_division=0, **kw),
        sk(skm.recall_score, y, p, pos_label=0, zero_division=0, **kw),
    )
    # NPV = precision of the negative class
    assert_close(
        es.npv(y, p, zero_division=0, **kw), sk(skm.precision_score, y, p, pos_label=0, zero_division=0, **kw)
    )


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("average", AVERAGES_MC)
def test_multiclass_averaging(seed: int, k: int, weighted: bool, average: str | None) -> None:
    y, p, w = labels_data(seed, 80, k, weighted)
    kw = {"sample_weight": w, "zero_division": 0, "average": average}
    assert_close(es.precision(y, p, **kw), sk(skm.precision_score, y, p, **kw))
    assert_close(es.recall(y, p, **kw), sk(skm.recall_score, y, p, **kw))
    assert_close(es.f1(y, p, **kw), sk(skm.f1_score, y, p, **kw))
    assert_close(es.fbeta(y, p, beta=0.5, **kw), sk(skm.fbeta_score, y, p, beta=0.5, **kw))
    assert_close(es.jaccard(y, p, **kw), sk(skm.jaccard_score, y, p, **kw))


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("weighted", [False, True])
def test_multiclass_other(seed: int, k: int, weighted: bool) -> None:
    y, p, w = labels_data(seed, 80, k, weighted)
    kw = {"sample_weight": w}
    assert_close(es.accuracy(y, p, **kw), skm.accuracy_score(y, p, **kw))
    assert_close(es.balanced_accuracy(y, p, **kw), sk(skm.balanced_accuracy_score, y, p, **kw))
    assert_close(es_(es.mcc, y, p, **kw), sk(skm.matthews_corrcoef, y, p, **kw))
    for wt in (None, "linear", "quadratic"):
        assert_close(
            es_(es.cohen_kappa, y, p, weights=wt, **kw), sk(skm.cohen_kappa_score, y, p, weights=wt, **kw)
        )
    assert_close(es.confusion_matrix(y, p, **kw), skm.confusion_matrix(y, p, **kw))
    for norm in ("true", "pred", "all"):
        assert_close(
            es.confusion_matrix(y, p, normalize=norm, **kw),
            sk(skm.confusion_matrix, y, p, normalize=norm, **kw),
        )


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", "samples", None])
@pytest.mark.parametrize("weighted", [False, True])
def test_multilabel(seed: int, average: str | None, weighted: bool) -> None:
    rng = np.random.default_rng(seed)
    Y = rng.integers(0, 2, (50, 4))
    P = np.where(rng.random((50, 4)) < 0.7, Y, 1 - Y)
    w = rng.random(50) if weighted else None
    kw = {"sample_weight": w, "zero_division": 0, "average": average}
    assert_close(es.precision(Y, P, **kw), sk(skm.precision_score, Y, P, **kw))
    assert_close(es.recall(Y, P, **kw), sk(skm.recall_score, Y, P, **kw))
    assert_close(es.f1(Y, P, **kw), sk(skm.f1_score, Y, P, **kw))
    assert_close(es.jaccard(Y, P, **kw), sk(skm.jaccard_score, Y, P, **kw))
    assert_close(es.accuracy(Y, P, sample_weight=w), skm.accuracy_score(Y, P, sample_weight=w))
    assert_close(es.hamming_loss(Y, P, sample_weight=w), skm.hamming_loss(Y, P, sample_weight=w))


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("weighted", [False, True])
def test_binary_probability_metrics(seed: int, weighted: bool) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 70)
    y[:2] = [0, 1]
    score = np.clip(y * 0.4 + rng.random(70) * 0.6, 0, 1)
    score = np.round(score, 1)  # create ties on purpose
    w = rng.random(70) * 2 if weighted else None
    kw = {"sample_weight": w}
    assert_close(es.roc_auc(y, score, **kw), skm.roc_auc_score(y, score, **kw))
    assert_close(es.average_precision(y, score, **kw), skm.average_precision_score(y, score, **kw))
    assert_close(es.log_loss(y, score, **kw), skm.log_loss(y, score, **kw), tol=1e-9)
    assert_close(es.brier_score(y, score, **kw), skm.brier_score_loss(y, score, **kw))
    fpr, tpr, thr = es.roc_curve(y, score, **kw)
    sfpr, stpr, sthr = skm.roc_curve(y, score, drop_intermediate=False, **kw)
    assert_close(fpr, sfpr)
    assert_close(tpr, stpr)
    assert_close(thr[1:], sthr[1:])
    pr, rc, th = es.pr_curve(y, score, **kw)
    spr, src, sth = skm.precision_recall_curve(y, score, **kw)
    assert_close(pr, spr)
    assert_close(rc, src)
    assert_close(th, sth)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("weighted", [False, True])
def test_multiclass_probability_metrics(seed: int, k: int, weighted: bool) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, 90)
    y[:k] = np.arange(k)
    logits = rng.normal(size=(90, k)) + 1.5 * np.eye(k)[y]
    prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    w = rng.random(90) if weighted else None
    kw = {"sample_weight": w}
    for avg in ("macro", "weighted"):
        assert_close(
            es.roc_auc(y, prob, average=avg, **kw),
            skm.roc_auc_score(y, prob, multi_class="ovr", average=avg, **kw),
        )
    if not weighted:  # scikit-learn's OvO does not take sample weights
        assert_close(es.roc_auc(y, prob, multi_class="ovo"), skm.roc_auc_score(y, prob, multi_class="ovo"))
    assert_close(es.log_loss(y, prob, **kw), skm.log_loss(y, prob, **kw), tol=1e-9)
    for kk in (1, 2):
        assert_close(es.top_k_accuracy(y, prob, k=kk, **kw), skm.top_k_accuracy_score(y, prob, k=kk, **kw))
    onehot = np.eye(k)[y]
    assert_close(
        es.average_precision(y, prob, average="macro", **kw),
        skm.average_precision_score(onehot, prob, average="macro", **kw),
    )


@pytest.mark.parametrize("seed", range(10))
def test_multilabel_probability_metrics(seed: int) -> None:
    rng = np.random.default_rng(seed)
    Y = rng.integers(0, 2, (60, 3))
    Y[:2] = [[0, 0, 0], [1, 1, 1]]
    S = np.clip(Y * 0.3 + rng.random((60, 3)) * 0.7, 0, 1)
    for avg in ("micro", "macro", "weighted", None):
        assert_close(es.roc_auc(Y, S, average=avg), skm.roc_auc_score(Y, S, average=avg))
        assert_close(es.average_precision(Y, S, average=avg), skm.average_precision_score(Y, S, average=avg))


def test_string_labels_and_explicit_pos_label() -> None:
    y = np.array(["cat", "dog", "dog", "cat", "dog", "cat"])
    p = np.array(["cat", "dog", "cat", "cat", "dog", "dog"])
    assert_close(es.f1(y, p, pos_label="dog"), skm.f1_score(y, p, pos_label="dog"))
    assert_close(es.f1(y, p, average="macro"), skm.f1_score(y, p, average="macro"))
    with pytest.raises(es.InputValidationError, match="pos_label"):
        es.f1(y, p)


def test_explicit_label_order_controls_per_class_output() -> None:
    y = [2, 0, 1, 2, 1, 0]
    p = [2, 0, 2, 2, 1, 1]
    r = es.recall(y, p, average=None, labels=[2, 1, 0])
    assert r.labels == (2, 1, 0)
    assert_close(r, skm.recall_score(y, p, average=None, labels=[2, 1, 0]))


@settings(max_examples=150, deadline=None)
@given(
    data=st.lists(st.tuples(st.integers(0, 3), st.integers(0, 3), st.floats(0.01, 5)), min_size=1, max_size=60),
    average=st.sampled_from(["micro", "macro", "weighted"]),
)
def test_property_f1_precision_recall(data: list[tuple[int, int, float]], average: str) -> None:
    y = np.array([d[0] for d in data])
    p = np.array([d[1] for d in data])
    w = np.array([d[2] for d in data])
    for es_fn, sk_fn in (
        (es.f1, skm.f1_score),
        (es.precision, skm.precision_score),
        (es.recall, skm.recall_score),
    ):
        assert_close(
            es_fn(y, p, average=average, sample_weight=w, zero_division=0),
            sk(sk_fn, y, p, average=average, sample_weight=w, zero_division=0, labels=np.unique(np.r_[y, p])),
        )


@settings(max_examples=150, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 1), st.floats(0, 1)), min_size=2, max_size=60))
def test_property_roc_auc_bounds_and_agreement(data: list[tuple[int, float]]) -> None:
    y = np.array([d[0] for d in data])
    s = np.array([d[1] for d in data])
    if len(set(y)) < 2:
        with pytest.raises(es.MetricInputError):
            es.roc_auc(y, s)
        return
    v = float(es.roc_auc(y, s))
    assert 0 <= v <= 1
    assert_close(v, skm.roc_auc_score(y, s), tol=1e-9)
