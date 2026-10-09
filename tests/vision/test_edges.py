"""Edge cases and explicit errors for the vision metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

import evalsuite as es

T = np.array([[0, 1, 1], [0, 2, 2]])
P = np.array([[0, 1, 0], [2, 2, 2]])


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: es.dice(np.array([1, 2]), np.array([1, 2])), "2-D mask"),
        (lambda: es.dice([], []), "no images"),
        (lambda: es.dice([T, T], [T]), "2 images"),
        (lambda: es.dice([np.array([1, 2])], [np.array([1, 2])]), "at least 2 dimensions"),
        (lambda: es.dice(T, P, labels=[1, 1]), "distinct"),
        (lambda: es.dice(T, P, num_classes=0), "at least 1"),
        (lambda: es.dice(T, P, labels=[0], include_background=False), "no classes"),
        (lambda: es.dice(T, P, num_classes=2), "num_classes"),
        (lambda: es.dice(T, P, aggregate="pixels"), "aggregate"),  # type: ignore[arg-type]
        (lambda: es.dice(T, P, average="samples"), "average"),  # type: ignore[arg-type]
        (lambda: es.dice(T, P, empty=2.0), "empty"),
        (lambda: es.pixel_accuracy(np.full((2, 2), 9), np.full((2, 2), 9), ignore_index=9), "ignored"),
        (lambda: es.hausdorff_distance(T, P, percentile=0), "percentile"),
        (lambda: es.hausdorff_distance(T, P, spacing=(1.0,)), "spacing"),
        (lambda: es.hausdorff_distance(T, P, average="micro"), "average"),  # type: ignore[arg-type]
        (lambda: es.boundary_iou(T, P, dilation_ratio=0), "dilation_ratio"),
    ],
)
def test_segmentation_errors(call, match) -> None:
    with pytest.raises(es.InputValidationError, match=match):
        call()


def test_segmentation_variants() -> None:
    assert float(es.dice(T.astype(float), P.astype(float))) == float(es.dice(T, P))
    assert float(es.dice(T, P, average="weighted")) > 0
    assert float(es.dice([T, T], [P, P], average="micro", aggregate="image")) == pytest.approx(
        float(es.dice(T, P, average="micro"))
    )
    r = es.dice(
        [T, np.zeros_like(T)], [P, np.zeros_like(P)], aggregate="image", empty=1.0, include_background=False
    )
    assert 0 < float(r) <= 1
    with pytest.warns(es.UndefinedMetricWarning):
        assert math.isnan(
            float(es.dice(np.zeros((2, 2), int), np.zeros((2, 2), int), include_background=False, num_classes=2))
        )
    assert float(es.boundary_iou(T, P, average="micro", include_background=True)) > 0
    assert len(es.boundary_iou(T, P, average=None).value) == 2
    assert float(es.boundary_iou(T, P, empty=1.0, labels=[1, 2, 3])) > 0
    with pytest.warns(es.UndefinedMetricWarning):
        assert math.isnan(
            float(es.hausdorff_distance(np.zeros((3, 3), int), np.zeros((3, 3), int), num_classes=2))
        )
    assert len(es.average_surface_distance(T, P, average=None).value) == 2
    assert float(es.hausdorff_distance(T, P, ignore_index=0, include_background=False)) >= 0
    assert es.segmentation_confusion(T, P).sum() == T.size
    assert es.iou(T, P, ignore_index=0).params["ignore_index"] == 0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        ({"boxes": []}, [], "lists"),
        ([], [], "No images"),
        ([{"boxes": [[0, 0, 1, 1]], "labels": [1]}], [{"boxes": [], "labels": []}, {}], "images"),
        (["x"], [{}], "dicts"),
        ([{"boxes": [[0, 0, 1]], "labels": [1]}], [{}], "shape"),
        ([{"boxes": [[0, 0, np.nan, 1]], "labels": [1]}], [{}], "NaN"),
        ([{"boxes": [[0, 0, 1, 1]], "labels": [1, 2]}], [{}], "one label per box"),
        (
            [{"boxes": [[0, 0, 1, 1]], "labels": [1]}],
            [{"boxes": [[0, 0, 1, 1]], "labels": [1], "scores": []}],
            "score",
        ),
        ([{"boxes": [[0, 0, 1, 1]], "labels": [1], "iscrowd": [0, 0]}], [{}], "iscrowd"),
    ],
)
def test_detection_errors(y_true, y_pred, match) -> None:
    with pytest.raises(es.InputValidationError, match=match):
        es.detection_report(y_true, y_pred)


def test_detection_options_and_errors() -> None:
    gt = [{"boxes": [[0, 0, 10, 10]], "labels": [1]}]
    dt = [{"boxes": [[0, 0, 10, 10]], "labels": [1], "scores": [0.9]}]
    with pytest.raises(es.InputValidationError, match="box_format"):
        es.box_iou([[0, 0, 1, 1]], [[0, 0, 1, 1]], box_format="yolo")  # type: ignore[arg-type]
    with pytest.raises(es.InputValidationError, match="max_detections"):
        es.detection_report(gt, dt, max_detections=0)
    with pytest.raises(es.InputValidationError, match="iou_threshold"):
        es.average_precision_detection(gt, dt, iou_threshold=1.5)
    with pytest.raises(es.InputValidationError, match="interpolation"):
        es.average_precision_detection(gt, dt, interpolation="pascal")  # type: ignore[arg-type]
    with pytest.raises(es.InputValidationError, match="average"):
        es.average_precision_detection(gt, dt, average="micro")  # type: ignore[arg-type]
    assert float(es.average_precision_detection(gt, dt, average="macro")) == pytest.approx(1.0)
    assert float(es.average_precision_detection(gt, dt, interpolation="voc", average="macro")) == pytest.approx(
        1.0
    )
    assert es.box_iou(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)
    # a class with detections but no ground truth is undefined, not 0
    only_dt = [{"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "labels": [1, 2], "scores": [0.9, 0.5]}]
    r = es.average_precision_detection(gt, only_dt)
    assert r.per_class()[1] == pytest.approx(1.0) and math.isnan(r.per_class()[2])
    with pytest.raises(es.InputValidationError, match="COCO dataset"):
        es.from_coco({"images": []})
    rep = es.detection_report(gt, [{"boxes": [], "labels": []}])
    assert rep["map"] == 0.0 and math.isnan(rep["map_small"]) is False
    with pytest.raises(KeyError):
        rep["nope"]


def test_evaluate_points_vision_inputs_to_the_right_function() -> None:
    with pytest.raises(es.UnsupportedTaskError, match="detection_report"):
        es.evaluate([{"boxes": [[0, 0, 1, 1]], "labels": [1]}], [{"boxes": [], "labels": []}])
    with pytest.raises(es.UnsupportedTaskError, match="segmentation_report"):
        es.evaluate(np.zeros((2, 3, 3), int), np.zeros((2, 3, 3), int))


def test_nested_list_is_one_2d_mask() -> None:  # 0.3.1 fix
    as_list = es.dice([[1, 1], [0, 0]], [[1, 0], [0, 0]], average=None).value
    as_array = es.dice(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [0, 0]]), average=None).value
    np.testing.assert_array_equal(as_list, as_array)
    stacked = es.dice([[[1, 1], [0, 0]], [[1, 0], [0, 0]]], [[[1, 1], [0, 0]], [[1, 0], [0, 0]]])
    assert float(stacked) == 1.0  # a list of 2-D masks is still a list of images
    with pytest.raises(es.InputValidationError):
        es.dice([1, 0, 1], [1, 0, 1])
