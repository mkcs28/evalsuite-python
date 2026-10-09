"""Segmentation: overlap metrics against scikit-learn on flattened pixels, surface distances against SciPy's
exact point-set distances, Boundary IoU against an independent distance-transform construction."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial.distance import cdist, directed_hausdorff

import evalsuite as es

skm = pytest.importorskip("sklearn.metrics")


def _masks(seed: int, n: int = 4, shape=(48, 40), k: int = 4):
    rng = np.random.default_rng(seed)
    t = np.zeros((n, *shape), dtype=np.int64)
    for i in range(n):
        for c in range(1, k):
            cy, cx = rng.integers(5, shape[0] - 5), rng.integers(5, shape[1] - 5)
            r = rng.integers(3, 12)
            yy, xx = np.ogrid[: shape[0], : shape[1]]
            t[i][(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = c
    p = t.copy()
    for i in range(n):
        p[i] = ndimage.shift(t[i], (rng.integers(-2, 3), rng.integers(-2, 3)), order=0, mode="nearest")
        noise = rng.random(shape) < 0.03
        p[i][noise] = rng.integers(0, k, noise.sum())
    return t, p


def test_overlap_metrics_match_sklearn_on_pixels() -> None:
    t, p = _masks(0)
    ft, fp = t.ravel(), p.ravel()
    labels = list(range(4))
    ref_iou = skm.jaccard_score(ft, fp, labels=labels, average=None)
    ref_dice = skm.f1_score(ft, fp, labels=labels, average=None)
    np.testing.assert_allclose(es.iou(t, p, average=None).value, ref_iou)
    np.testing.assert_allclose(es.dice(t, p, average=None).value, ref_dice)
    assert float(es.miou(t, p)) == pytest.approx(ref_iou.mean())
    assert float(es.iou(t, p, average="weighted")) == pytest.approx(skm.jaccard_score(ft, fp, average="weighted"))
    assert float(es.dice(t, p, average="micro")) == pytest.approx(skm.f1_score(ft, fp, average="micro"))
    assert float(es.pixel_accuracy(t, p)) == pytest.approx(skm.accuracy_score(ft, fp))
    assert float(es.mean_pixel_accuracy(t, p)) == pytest.approx(skm.balanced_accuracy_score(ft, fp))
    np.testing.assert_array_equal(es.segmentation_confusion(t, p), skm.confusion_matrix(ft, fp, labels=labels))


def test_image_aggregate_is_mean_of_per_image_scores() -> None:
    t, p = _masks(1)
    per = [skm.f1_score(a.ravel(), b.ravel(), labels=[1, 2, 3], average=None) for a, b in zip(t, p)]
    ref = np.mean(per, axis=0).mean()
    r = es.dice(t, p, aggregate="image", include_background=False)
    assert float(r) == pytest.approx(ref)
    scores = es.per_image_scores(t, p, "dice", include_background=False)
    np.testing.assert_allclose(scores, [np.mean(x) for x in per])


def test_ignore_index_and_absent_classes() -> None:
    t = np.array([[0, 1, 255], [1, 1, 255]])
    p = np.array([[0, 1, 1], [0, 255, 0]])
    # valid pixels: (0,0) 0->0, (0,1) 1->1, (1,0) 1->0, (1,1) 1->255 (miss)
    r = es.iou(t, p, ignore_index=255, average=None, num_classes=3)
    assert r.value[0] == pytest.approx(1 / 2)  # tp 1, fp 1 (pixel (1,0))
    assert r.value[1] == pytest.approx(1 / 3)  # tp 1, fn 2
    assert math.isnan(r.value[2])  # class 2 absent from both: undefined, not 0
    assert float(es.iou(t, p, ignore_index=255, num_classes=3)) == pytest.approx((1 / 2 + 1 / 3) / 2)
    assert float(es.iou(t, p, ignore_index=255, num_classes=3, empty=1.0)) == pytest.approx(
        (1 / 2 + 1 / 3 + 1) / 3
    )
    assert float(es.pixel_accuracy(t, p, ignore_index=255)) == pytest.approx(2 / 4)


def test_inputs_lists_booleans_and_errors() -> None:
    a = np.zeros((10, 12), bool)
    a[2:6, 3:8] = True
    b = np.zeros((7, 7), bool)
    b[1:4, 1:4] = True
    r = es.dice([a, b], [a, b], include_background=False)
    assert float(r) == 1.0 and r.params["n_images"] == 2
    with pytest.raises(es.InputValidationError, match="shape"):
        es.dice(a, b)
    with pytest.raises(es.InputValidationError, match="integer class labels"):
        es.dice(a.astype(float) * 0.5, a)
    with pytest.raises(es.InputValidationError, match="negative"):
        es.dice(-a.astype(int), a)


def _surface_points(mask):
    er = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1), border_value=0)
    return np.argwhere(mask & ~er).astype(float)


def test_surface_distances_match_exact_point_sets() -> None:
    t, p = _masks(2, n=1, k=2)
    a, b = t[0] == 1, p[0] == 1
    pa, pb = _surface_points(a), _surface_points(b)
    hd = max(directed_hausdorff(pa, pb)[0], directed_hausdorff(pb, pa)[0])
    assert float(es.hausdorff_distance(t, p)) == pytest.approx(hd)
    d = cdist(pa, pb)
    d_ab, d_ba = d.min(1), d.min(0)
    hd95 = max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))
    assert float(es.hausdorff_distance(t, p, percentile=95)) == pytest.approx(hd95)
    assd = (d_ab.sum() + d_ba.sum()) / (d_ab.size + d_ba.size)
    assert float(es.average_surface_distance(t, p)) == pytest.approx(assd)
    # anisotropic spacing scales the coordinates
    sp = (2.0, 0.5)
    hd_sp = max(directed_hausdorff(pa * sp, pb * sp)[0], directed_hausdorff(pb * sp, pa * sp)[0])
    assert float(es.hausdorff_distance(t, p, spacing=sp)) == pytest.approx(hd_sp)


def test_surface_distance_with_missing_class_is_inf_with_warning() -> None:
    t = np.zeros((1, 8, 8), int)
    t[0, 2:5, 2:5] = 1
    p = np.zeros_like(t)
    with pytest.warns(es.UndefinedMetricWarning):
        assert math.isinf(float(es.hausdorff_distance(t, p)))
    assert float(es.hausdorff_distance(t, t)) == 0.0


def _boundary_reference(mask, d):
    padded = np.pad(mask, 1)
    inside = ndimage.distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1]
    return mask & (inside <= d)


def test_boundary_iou_matches_distance_transform_definition() -> None:
    t, p = _masks(3, n=2)
    inter = np.zeros(3)
    union = np.zeros(3)
    for a, b in zip(t, p):
        d = max(1, round(0.02 * math.hypot(*a.shape)))
        for j, c in enumerate([1, 2, 3]):
            ga, pb = _boundary_reference(a == c, d), _boundary_reference(b == c, d)
            inter[j] += (ga & pb).sum()
            union[j] += (ga | pb).sum()
    ref = inter / union
    np.testing.assert_allclose(es.boundary_iou(t, p, average=None).value, ref)
    assert float(es.boundary_iou(t, p)) == pytest.approx(ref.mean())
    assert float(es.boundary_iou(t, t)) == 1.0


def test_registry_documents_vision_metrics() -> None:
    for m in (
        "segmentation.dice",
        "segmentation.iou",
        "segmentation.miou",
        "segmentation.boundary_iou",
        "segmentation.hausdorff_distance",
        "detection.mean_average_precision",
        "detection.box_iou",
    ):
        info = es.metric_info(m)
        assert info.formula and info.references
