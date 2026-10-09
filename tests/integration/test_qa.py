"""Release QA: edge cases against reference libraries, documented conventions, exports, reproducibility.

Each test names the QA item it covers. Regression tests for defects fixed in 0.3.1 are marked "0.3.1 fix".
"""

from __future__ import annotations

import html
import json
import math
import warnings

import numpy as np
import pytest

import evalsuite as es

sk = pytest.importorskip("sklearn.metrics")


# ---------------------------------------------------------------- classification vs scikit-learn
def _binary(seed: int, n: int = 300, prevalence: float = 0.5):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    p = np.clip(0.35 * y + rng.random(n) * 0.65, 0, 1)
    return y, (p >= 0.5).astype(int), p


@pytest.mark.parametrize("prevalence", [0.5, 0.05])  # balanced and imbalanced
@pytest.mark.parametrize("seed", [0, 1])
def test_binary_metrics_match_sklearn(seed: int, prevalence: float) -> None:
    y, pred, prob = _binary(seed, prevalence=prevalence)
    pairs = {
        es.accuracy: sk.accuracy_score,
        es.precision: lambda a, b: sk.precision_score(a, b, zero_division=0),
        es.recall: sk.recall_score,
        es.f1: lambda a, b: sk.f1_score(a, b, zero_division=0),
        es.mcc: sk.matthews_corrcoef,
        es.balanced_accuracy: sk.balanced_accuracy_score,
        es.sensitivity: sk.recall_score,
        es.ppv: lambda a, b: sk.precision_score(a, b, zero_division=0),
    }
    for ours, ref in pairs.items():
        assert float(ours(y, pred)) == pytest.approx(ref(y, pred), abs=1e-12), ours.__name__
    assert float(es.specificity(y, pred)) == pytest.approx(sk.recall_score(y, pred, pos_label=0))
    assert float(es.npv(y, pred)) == pytest.approx(sk.precision_score(y, pred, pos_label=0, zero_division=0))
    assert float(es.roc_auc(y, prob)) == pytest.approx(sk.roc_auc_score(y, prob), abs=1e-12)
    np.testing.assert_array_equal(es.confusion_matrix(y, pred), sk.confusion_matrix(y, pred))


@pytest.mark.parametrize("average", ["macro", "micro", "weighted"])
def test_multiclass_with_missing_predicted_class_matches_sklearn(average: str) -> None:
    y = np.array([0, 1, 2, 2, 1, 0, 2, 2])
    pred = np.array([0, 1, 1, 1, 1, 0, 1, 0])  # class 2 is never predicted
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ours, ref in [(es.precision, sk.precision_score), (es.recall, sk.recall_score), (es.f1, sk.f1_score)]:
            expected = ref(y, pred, average=average, zero_division=0)
            assert float(ours(y, pred, average=average)) == pytest.approx(expected, abs=1e-12)


def test_multiclass_auc_follows_sorted_label_order_for_string_labels() -> None:
    y = np.array(["cat", "dog", "bird", "dog", "cat", "bird"])
    prob = np.array(
        [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.7, 0.2, 0.1], [0.2, 0.2, 0.6], [0.2, 0.6, 0.2], [0.5, 0.3, 0.2]]
    )  # columns: bird, cat, dog
    assert float(es.roc_auc(y, prob)) == pytest.approx(sk.roc_auc_score(y, prob, multi_class="ovr"))


def test_zero_division_policy_is_documented_and_configurable() -> None:
    y, pred = [0, 1, 1], [0, 0, 0]  # no predicted positives
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert float(es.precision(y, pred)) == 0.0
        assert float(es.precision(y, pred, zero_division=1.0)) == 1.0
    assert float(es.recall([0, 1, 1], [1, 0, 0])) == 0.0  # zero true positives


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: es.accuracy([], []), "empty"),
        (lambda: es.accuracy([0, np.nan], [0, 1]), "NaN"),
        (lambda: es.roc_auc([0, 1], [0.2, np.inf]), "infinite"),
        (lambda: es.roc_auc([0, 1], [0.2, 1.4]), r"\[0, 1\]"),
        (lambda: es.accuracy([0, 1], [0]), "same number"),
        (lambda: es.roc_auc([0, 1, 2], np.full((3, 2), 0.5)), "columns"),
        (lambda: es.roc_auc([0, 1], np.full((2, 2, 2), 0.5)), "1-D or 2-D"),
        (lambda: es.roc_auc([0, 1, 2], np.full((3, 3), 0.5)), "sum to 1"),
    ],
)
def test_invalid_classification_inputs_raise_clear_errors(call, match: str) -> None:
    with pytest.raises(es.InputValidationError, match=match):
        call()


def test_single_class_target_auc_is_an_error_and_mcc_follows_sklearn() -> None:
    with pytest.raises(es.MetricInputError, match="no negative"):
        es.roc_auc([1, 1, 1], [0.2, 0.5, 0.9])
    assert float(es.mcc([1, 1, 1], [1, 1, 1])) == sk.matthews_corrcoef([1, 1, 1], [1, 1, 1])


def test_mixed_number_and_string_labels_are_rejected() -> None:  # 0.3.1 fix
    with pytest.raises(es.InputValidationError, match="mixes numbers and strings"):
        es.accuracy([0, "a"], [0, 1])
    with pytest.raises(es.InputValidationError, match="string labels but y_pred has number"):
        es.accuracy(np.array(["a", "b"]), np.array([0, 1]))
    assert float(es.accuracy(["a", "b"], ["a", "a"])) == 0.5  # all strings stay fine


# ---------------------------------------------------------------- regression vs scikit-learn
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_regression_matches_sklearn_for_int_and_float_inputs(dtype) -> None:
    rng = np.random.default_rng(3)
    y = (rng.normal(50, 10, 200)).astype(dtype)
    p = (y + rng.normal(0, 3, 200)).astype(dtype)
    yf, pf = y.astype(np.float64), p.astype(np.float64)
    assert float(es.mae(y, p)) == pytest.approx(sk.mean_absolute_error(yf, pf), rel=1e-12)
    assert float(es.mse(y, p)) == pytest.approx(sk.mean_squared_error(yf, pf), rel=1e-12)
    assert float(es.rmse(y, p)) == pytest.approx(math.sqrt(sk.mean_squared_error(yf, pf)), rel=1e-12)
    assert float(es.r2(y, p)) == pytest.approx(sk.r2_score(yf, pf), rel=1e-12)
    assert isinstance(es.mse(y, p).value, float)


def test_multioutput_regression_matches_sklearn() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(size=(100, 3))
    p = y + rng.normal(0, 0.5, (100, 3))
    np.testing.assert_allclose(
        es.mae(y, p, multioutput="raw_values").value, sk.mean_absolute_error(y, p, multioutput="raw_values")
    )
    assert float(es.r2(y, p)) == pytest.approx(sk.r2_score(y, p))


def test_degenerate_regression_inputs() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert float(es.r2([2, 2, 2], [2, 2, 3])) == sk.r2_score([2, 2, 2], [2, 2, 3])  # constant target
        assert float(es.r2([2, 2, 2], [2, 2, 2])) == 1.0
    assert float(es.mae([1.5], [1.0])) == 0.5  # single sample
    for bad, match in [(([], []), "empty"), (([1, np.nan], [1, 2]), "NaN"), (([1, np.inf], [1, 2]), "infinite")]:
        with pytest.raises(es.InputValidationError, match=match):
            es.mse(*bad)


def test_rmse_does_not_overflow_for_large_values() -> None:  # 0.3.1 fix
    value = float(es.rmse([1e154, 2e154], [0.0, 0.0]))
    assert math.isfinite(value)
    assert value == pytest.approx(math.sqrt((1 + 4) / 2) * 1e154, rel=1e-12)
    assert float(es.rmse([1.0, 3.0], [0.0, 0.0])) == pytest.approx(math.sqrt(5))  # unchanged path


# ---------------------------------------------------------------- statistics
def test_resampling_needs_two_observations() -> None:  # 0.3.1 fix
    with pytest.raises(es.StatisticalTestError, match="at least 2 observations"):
        es.bootstrap_ci("accuracy", [1], [1])
    with pytest.raises(es.StatisticalTestError, match="at least 2 observations"):
        es.paired_bootstrap_test("accuracy", [1], [1], [0])


def test_randomised_procedures_are_reproducible_with_a_seed() -> None:
    y, pred, _ = _binary(5)
    a = es.bootstrap_ci("f1", y, pred, n_resamples=300, random_state=7)
    b = es.bootstrap_ci("f1", y, pred, n_resamples=300, random_state=7)
    assert (a.low, a.high) == (b.low, b.high)
    c1 = es.compare(y, {"A": pred, "B": 1 - pred}, n_resamples=200, random_state=1)
    c2 = es.compare(y, {"A": pred, "B": 1 - pred}, n_resamples=200, random_state=1)
    assert c1.to_json() == c2.to_json()


def test_multiple_testing_matches_statsmodels() -> None:
    mt = pytest.importorskip("statsmodels.stats.multitest")
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.5])
    for ours, theirs in [
        ("bonferroni", "bonferroni"),
        ("holm", "holm"),
        ("hochberg", "simes-hochberg"),
        ("bh", "fdr_bh"),
        ("by", "fdr_by"),
    ]:
        np.testing.assert_allclose(es.adjust_pvalues(p, method=ours), mt.multipletests(p, method=theirs)[1])


# ---------------------------------------------------------------- clinical
def test_ppv_and_npv_follow_bayes_with_prevalence() -> None:
    y, pred, _ = _binary(6, n=2000, prevalence=0.2)
    se, sp = float(es.sensitivity(y, pred)), float(es.specificity(y, pred))
    prev = y.mean()
    assert float(es.ppv(y, pred)) == pytest.approx(se * prev / (se * prev + (1 - sp) * (1 - prev)))
    assert float(es.npv(y, pred)) == pytest.approx(sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev))
    assert float(es.lr_positive(y, pred)) == pytest.approx(se / (1 - sp))
    assert float(es.lr_negative(y, pred)) == pytest.approx((1 - se) / sp)


def test_decision_curve_net_benefit_and_reference_strategies() -> None:
    y, _, prob = _binary(8, n=500, prevalence=0.3)
    dca = es.decision_curve(y, {"m": prob}, thresholds=[0.1, 0.25, 0.5])
    prev = y.mean()
    for i, t in enumerate(dca.thresholds):
        pos = prob >= t
        tp, fp = np.sum(pos & (y == 1)), np.sum(pos & (y == 0))
        expected = tp / y.size - fp / y.size * t / (1 - t)
        assert dca.net_benefit["m"][i] == pytest.approx(expected)
        assert dca.treat_all[i] == pytest.approx(prev - (1 - prev) * t / (1 - t))
        assert dca.treat_none[i] == 0.0


# ---------------------------------------------------------------- segmentation conventions
def test_segmentation_empty_masks_absent_classes_and_ignore_index() -> None:
    empty = np.zeros((4, 4), int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_class = es.dice(empty, empty, num_classes=3, average=None).value
    assert per_class[0] == 1.0 and np.isnan(per_class[1:]).all()  # absent classes are NaN, not 0
    assert float(es.dice(empty, empty, num_classes=3, empty=1.0)) == 1.0  # explicit policy
    t = np.array([[1, 1, 255], [0, 0, 255]])
    p = np.array([[1, 0, 1], [0, 0, 0]])
    assert float(es.pixel_accuracy(t, p, ignore_index=255)) == pytest.approx(3 / 4)


def test_per_image_and_dataset_aggregation_differ_as_documented() -> None:
    a = np.zeros((2, 10, 10), int)
    b = a.copy()
    a[0, :5] = 1
    b[0, :5] = 1  # image 0 perfect
    a[1, 0, 0] = 1  # image 1: one pixel, missed
    dataset = float(es.dice(a, b, average=None, aggregate="dataset").value[1])
    image = float(es.dice(a, b, average=None, aggregate="image").value[1])
    assert dataset == pytest.approx(2 * 50 / (51 + 50))
    assert image == pytest.approx((1.0 + 0.0) / 2)


# ---------------------------------------------------------------- exports and API
UNTRUSTED = '<script>alert("x")</script> & café'


def _reports():
    rng = np.random.default_rng(0)
    yc = rng.integers(0, 2, 60)
    masks = rng.integers(0, 2, (2, 8, 8))
    det = [{"boxes": [[0, 0, 10, 10]], "labels": [1]}]
    pred = [{"boxes": [[0, 0, 10, 10]], "labels": [1], "scores": [0.9]}]
    return {
        "classification": es.classification_report(*[np.where(v == 0, UNTRUSTED, "β-class") for v in (yc, yc)]),
        "segmentation": es.segmentation_report(masks, masks, class_names={0: UNTRUSTED, 1: "β-class"}),
        "detection": es.detection_report(det, pred),
        "evaluation": es.evaluate(yc, yc),
        "diagnostic": es.diagnostic_report(yc, yc),
    }


@pytest.mark.parametrize("kind", ["classification", "segmentation", "detection", "evaluation", "diagnostic"])
def test_every_export_format_is_valid_and_safe(kind: str, tmp_path) -> None:
    report = _reports()[kind]
    data = json.loads(report.to_json())  # valid JSON (NaN is written as null)
    assert isinstance(data, dict)
    page = report.to_html()
    assert "<script>" not in page
    if UNTRUSTED in str(getattr(report, "rows", "")) or kind == "classification":
        assert html.escape(UNTRUSTED) in page or "&lt;script&gt;" in page
    for ext in ("csv", "md", "tex", "json", "html"):
        out = tmp_path / f"r.{ext}"
        report.save(out)
        assert out.read_text(encoding="utf-8").strip()


def test_json_export_preserves_values_and_non_finite() -> None:
    masks = np.zeros((1, 4, 4), int)
    masks[0, 0, 0] = 1
    pred = np.zeros_like(masks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = es.segmentation_report(masks, pred)
    data = json.loads(rep.to_json())
    assert "NaN" not in rep.to_json() and "Infinity" not in rep.to_json()
    assert data["values"]["miou"] == pytest.approx(rep["miou"])


def test_public_api_names_are_stable() -> None:
    required = [
        "evaluate",
        "compare",
        "bootstrap_ci",
        "accuracy",
        "f1",
        "roc_auc",
        "mae",
        "rmse",
        "r2",
        "diagnostic_report",
        "decision_curve",
        "calibration_report",
        "t_test",
        "adjust_pvalues",
        "dice",
        "iou",
        "miou",
        "hausdorff_distance",
        "segmentation_report",
        "detection_report",
        "mean_average_precision",
        "box_iou",
        "from_coco",
        "plot",
        "list_metrics",
        "metric_info",
    ]
    missing = [name for name in required if not hasattr(es, name)]
    assert not missing
    assert isinstance(es.accuracy([0, 1], [0, 1]), es.MetricResult)
    assert isinstance(es.t_test([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]), es.TestResult)
