"""v0.3.0 outputs: segmentation report, vision plots, CLI commands, comparison over images, benchmarks."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

import evalsuite as es

from .test_detection import _scene
from .test_segmentation import _masks


def test_segmentation_report_values_and_exports(tmp_path) -> None:
    t, p = _masks(0)
    rep = es.segmentation_report(t, p, class_names={0: "bg", 1: "a", 2: "b", 3: "c"})
    assert rep["miou"] == pytest.approx(float(es.miou(t, p)))
    assert rep["dice"] == pytest.approx(float(es.dice(t, p)))
    assert rep[1]["iou"] == pytest.approx(es.iou(t, p, average=None).value[1])
    assert rep[1]["hd95"] == pytest.approx(float(es.hausdorff_distance(t, p, percentile=95, labels=[1])))
    text = rep.summary()
    assert "mIoU" in text and "HD95 (pixels)" in text and "bg" in text
    assert json.loads(rep.to_json())["values"]["pixel_accuracy"] == pytest.approx(float(es.pixel_accuracy(t, p)))
    assert rep.to_markdown().startswith("| Class | Dice") and "\\toprule" in rep.to_latex()
    assert rep.to_csv().startswith("class,name,dice") and rep.to_html().startswith("<!doctype")
    for ext in ("json", "csv", "md", "tex", "html", "txt"):
        rep.save(tmp_path / f"s.{ext}")
    assert list(rep.to_dataframe().columns)[:2] == ["name", "dice"]
    with pytest.raises(KeyError):
        rep["nope"]


def test_compare_and_bootstrap_resample_images() -> None:
    t, p = _masks(4, n=10)
    worse = p.copy()
    worse[np.random.default_rng(0).random(p.shape) < 0.2] = 0
    r = es.compare(t, {"A": p, "B": worse}, n_resamples=200, random_state=0)
    assert set(r.settings["metrics"]) == {"dice", "iou"} if "metrics" in r.settings else True
    assert "dice" in r.summary() and "best: A" in r.summary()
    ci = es.bootstrap_ci("dice", t, p, n_resamples=200, random_state=0)
    assert ci.low <= float(es.dice(t, p)) <= ci.high
    yt, yp = _scene(1, n_images=15)
    det = es.compare(yt, {"m1": yp, "m2": yp}, n_resamples=100, random_state=0)
    assert "mean_average_precision" in det.summary()
    sa, sb = es.per_image_scores(t, p), es.per_image_scores(t, worse)
    assert es.wilcoxon_test(sa, sb).p_value < 0.05


def test_vision_plots() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    t, p = _masks(0, n=1)
    ax = es.plot.segmentation(None, t[0], p[0], class_names={1: "a"})
    assert "Dice" in ax.get_title()
    ax = es.plot.segmentation(np.random.default_rng(0).random(t[0].shape), t[0], p[0])
    ax = es.plot.per_class(es.iou(t, p, average=None), class_names={0: "bg"}, sort=True)
    assert "IoU per class" in ax.get_title()
    with pytest.raises(es.InputValidationError):
        es.plot.per_class(es.iou(t, p))
    yt, yp = _scene(0)
    ax = es.plot.detection_pr(yt, yp, class_names={1: "cat"})
    assert ax.get_legend().get_texts()[0].get_text().count("AP =") == 1
    curves = es.detection_pr_curve(yt, yp)
    assert all(rc.shape == pr.shape == (101,) for rc, pr in curves.values())


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "evalsuite", *args], capture_output=True, encoding="utf-8", timeout=300
    )


def test_cli_segmentation_and_detection(tmp_path) -> None:
    t, p = _masks(0)
    np.save(tmp_path / "t.npy", t)
    np.savez(tmp_path / "p.npz", masks=p)
    out = _cli("segmentation", str(tmp_path / "t.npy"), str(tmp_path / "p.npz"), "-f", "json")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["values"]["miou"] == pytest.approx(float(es.miou(t, p)))
    out = _cli("segmentation", str(tmp_path / "t.npy"), str(tmp_path / "missing.npy"))
    assert out.returncode == 2 and "no such file" in out.stderr
    yt, yp = _scene(0, crowd=False)
    images = [{"id": i + 1} for i in range(len(yt))]
    anns, res, aid = [], [], 1
    for i, (g, d) in enumerate(zip(yt, yp)):
        for b, c in zip(g["boxes"], g["labels"]):
            anns.append(
                {
                    "id": aid,
                    "image_id": i + 1,
                    "category_id": int(c),
                    "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                    "iscrowd": 0,
                }
            )
            aid += 1
        for b, c, s in zip(d["boxes"], d["labels"], d["scores"]):
            res.append(
                {
                    "image_id": i + 1,
                    "category_id": int(c),
                    "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                    "score": float(s),
                }
            )
    (tmp_path / "gt.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": []}))
    (tmp_path / "res.json").write_text(json.dumps(res))
    out = _cli("detection", str(tmp_path / "gt.json"), str(tmp_path / "res.json"), "-f", "json")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["values"]["map"] == pytest.approx(es.detection_report(yt, yp)["map"])
    pytest.importorskip("matplotlib")
    out = _cli(
        "detection", str(tmp_path / "gt.json"), str(tmp_path / "res.json"), "--plot", str(tmp_path / "pr.png")
    )
    assert out.returncode == 0 and (tmp_path / "pr.png").stat().st_size > 1000
    out = _cli(
        "segmentation", str(tmp_path / "t.npy"), str(tmp_path / "p.npz"), "--plot", str(tmp_path / "pc.png")
    )
    assert out.returncode == 0 and (tmp_path / "pc.png").stat().st_size > 1000


def test_vision_benchmarks_agree_with_references() -> None:
    from evalsuite.benchmarks import run_benchmarks

    pytest.importorskip("sklearn")
    b = run_benchmarks(sizes=(20_000,), repeat=1, suite="vision")
    assert len(b.rows) == 3
    for row in b.rows:
        if row["reference_ms"] is not None:
            assert row["max_abs_diff"] < 1e-9, row["case"]
