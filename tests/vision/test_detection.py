"""Object detection: every number checked against pycocotools on random data with crowds and all sizes."""

from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pytest

import evalsuite as es


def _scene(seed: int, n_images: int = 25, n_classes: int = 4, crowd: bool = True):
    rng = np.random.default_rng(seed)
    y_true, y_pred = [], []
    for _ in range(n_images):
        n = int(rng.integers(0, 7))
        xy = rng.uniform(0, 400, (n, 2))
        wh = rng.choice([8, 20, 40, 70, 150], n)[:, None] * rng.uniform(0.6, 1.4, (n, 2))
        boxes = np.column_stack([xy, xy + wh])
        labels = rng.integers(1, n_classes + 1, n)
        iscrowd = (rng.random(n) < 0.08).astype(int) if crowd else np.zeros(n, int)
        y_true.append({"boxes": boxes, "labels": labels, "iscrowd": iscrowd})
        dets, dl, ds = [], [], []
        for b, lab in zip(boxes, labels):
            if rng.random() < 0.85:
                jitter = rng.normal(0, 0.12, 4) * np.r_[b[2] - b[0], b[3] - b[1], b[2] - b[0], b[3] - b[1]]
                nb = b + jitter
                nb[2:] = np.maximum(nb[2:], nb[:2] + 1)
                dets.append(nb)
                dl.append(lab if rng.random() < 0.9 else rng.integers(1, n_classes + 1))
                ds.append(rng.uniform(0.3, 1.0))
        for _ in range(int(rng.integers(0, 4))):  # false positives
            p = rng.uniform(0, 400, 2)
            dets.append(np.r_[p, p + rng.uniform(5, 120, 2)])
            dl.append(rng.integers(1, n_classes + 1))
            ds.append(rng.uniform(0, 0.8))
        y_pred.append({"boxes": np.array(dets).reshape(-1, 4), "labels": np.array(dl), "scores": np.array(ds)})
    return y_true, y_pred


def _coco_stats(y_true, y_pred, max_det=100, iou_thr=None):
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, dets = [], [], []
    cats = sorted({int(c) for t, p in zip(y_true, y_pred) for c in (*t["labels"], *p["labels"])})
    aid = 1
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        images.append({"id": i + 1, "width": 600, "height": 600})
        for b, lab, c in zip(t["boxes"], t["labels"], t["iscrowd"]):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append(
                {
                    "id": aid,
                    "image_id": i + 1,
                    "category_id": int(lab),
                    "bbox": [b[0], b[1], w, h],
                    "area": w * h,
                    "iscrowd": int(c),
                }
            )
            aid += 1
        for b, lab, s in zip(p["boxes"], p["labels"], p["scores"]):
            dets.append(
                {
                    "image_id": i + 1,
                    "category_id": int(lab),
                    "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                    "score": float(s),
                }
            )
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in cats]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else COCO()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = sorted({1, 10, max_det})
        if iou_thr is not None:
            ev.params.iouThrs = np.array([iou_thr])
        ev.evaluate()
        ev.accumulate()
        if iou_thr is None:
            ev.summarize()
    return ev


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_detection_report_matches_pycocotools(seed: int) -> None:
    y_true, y_pred = _scene(seed)
    ev = _coco_stats(y_true, y_pred)
    rep = es.detection_report(y_true, y_pred)
    keys = [
        "map",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar_1",
        "mar_10",
        "mar_100",
        "mar_small",
        "mar_medium",
        "mar_large",
    ]
    for key, ref in zip(keys, ev.stats):
        ours = rep[key]
        if ref == -1:
            assert math.isnan(ours), key
        else:
            assert ours == pytest.approx(ref, abs=1e-12), key
    assert float(es.mean_average_precision(y_true, y_pred)) == pytest.approx(ev.stats[0], abs=1e-12)


def test_per_class_and_single_threshold_match_pycocotools() -> None:
    y_true, y_pred = _scene(7)
    ev = _coco_stats(y_true, y_pred, iou_thr=0.5)
    prec = ev.eval["precision"][0, :, :, 0, -1]
    ref = [
        float(np.mean(prec[:, k][prec[:, k] > -1])) if np.any(prec[:, k] > -1) else math.nan
        for k in range(prec.shape[1])
    ]
    ours = es.average_precision_detection(y_true, y_pred, iou_threshold=0.5)
    np.testing.assert_allclose(ours.value, ref, atol=1e-12)
    assert float(es.mean_average_precision(y_true, y_pred, iou_threshold=0.5)) == pytest.approx(np.nanmean(ref))


def test_max_detections_matches_pycocotools() -> None:
    y_true, y_pred = _scene(3)
    ev = _coco_stats(y_true, y_pred, max_det=2)  # summarize() hard-codes maxDets=100, so read eval directly
    m = list(ev.params.maxDets).index(2)
    prec = ev.eval["precision"][:, :, :, 0, m]
    ref = float(np.mean(prec[prec > -1]))
    assert es.detection_report(y_true, y_pred, max_detections=2)["map"] == pytest.approx(ref, abs=1e-12)
    assert float(es.mean_average_precision(y_true, y_pred, max_detections=2)) == pytest.approx(ref, abs=1e-12)


def test_perfect_and_empty_predictions() -> None:
    y_true, _ = _scene(5, crowd=False)
    perfect = [{"boxes": t["boxes"], "labels": t["labels"], "scores": np.ones(len(t["labels"]))} for t in y_true]
    assert es.detection_report(y_true, perfect)["map"] == pytest.approx(1.0)
    empty = [{"boxes": np.zeros((0, 4)), "labels": [], "scores": []} for _ in y_true]
    assert es.detection_report(y_true, empty)["map"] == 0.0


def test_voc_interpolation_hand_example() -> None:
    # one class, 2 ground truths in 1 image; detections: TP (0.9), FP (0.8), TP (0.7)
    y_true = [{"boxes": [[0, 0, 10, 10], [20, 20, 30, 30]], "labels": [1, 1]}]
    y_pred = [
        {
            "boxes": [[0, 0, 10, 10], [50, 50, 60, 60], [20, 20, 30, 30]],
            "labels": [1, 1, 1],
            "scores": [0.9, 0.8, 0.7],
        }
    ]
    # precision/recall points: (1, .5), (.5, .5), (2/3, 1) -> envelope 1 on [0,.5], 2/3 on (.5,1]
    ap = es.average_precision_detection(y_true, y_pred, interpolation="voc", average="macro")
    assert float(ap) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_box_iou_and_formats() -> None:
    a = [[0, 0, 10, 10]]
    b = [[5, 5, 15, 15], [20, 20, 30, 30]]
    np.testing.assert_allclose(es.box_iou(a, b), [[25 / 175, 0.0]])
    np.testing.assert_allclose(es.box_iou([[0, 0, 10, 10]], [[5, 5, 10, 10]], box_format="xywh"), [[25 / 175]])
    np.testing.assert_allclose(es.box_iou([[5, 5, 10, 10]], [[10, 10, 10, 10]], box_format="cxcywh"), [[25 / 175]])
    with pytest.raises(es.InputValidationError, match="negative width"):
        es.box_iou([[10, 0, 0, 10]], a)
    with pytest.raises(es.InputValidationError):
        es.detection_report([{"boxes": a, "labels": [1]}], [])


def test_from_coco_roundtrip() -> None:
    gt = {
        "images": [{"id": 7}, {"id": 9}],
        "annotations": [{"image_id": 9, "category_id": 2, "bbox": [0, 0, 10, 10], "iscrowd": 0}],
        "categories": [{"id": 2}],
    }
    res = [{"image_id": 9, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.9}]
    y_true, y_pred = es.from_coco(gt, res)
    assert len(y_true) == 2 and y_true[0]["boxes"] == [] and y_true[1]["area"] == [100]
    assert es.detection_report(y_true, y_pred, box_format="xywh")["map"] == pytest.approx(1.0)
    with pytest.raises(es.InputValidationError):
        es.from_coco(gt, [{"image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1], "score": 1}])


def test_report_exports(tmp_path) -> None:
    y_true, y_pred = _scene(0)
    rep = es.detection_report(y_true, y_pred)
    assert "mAP@[.50:.95]" in rep.summary() and float(rep) == rep["map"]
    assert rep.to_markdown().startswith("| Measure") and "\\toprule" in rep.to_latex()
    assert rep.to_csv().startswith("measure,class,value") and rep.to_html().startswith("<!doctype")
    for ext in ("json", "csv", "md", "tex", "html", "txt"):
        rep.save(tmp_path / f"d.{ext}")
    assert len(rep.to_dataframe()) == 12
