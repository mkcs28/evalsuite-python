"""Object detection metrics: box IoU, average precision per class and COCO-style mean average precision.

The COCO protocol (Lin et al. 2014) is implemented exactly as in the reference ``pycocotools`` evaluator:
greedy matching of score-sorted detections to the best-overlapping unmatched ground truth, crowd regions
that may absorb several detections, area ranges, a maximum number of detections per image, and
101-point interpolated precision. The test suite checks every number against ``pycocotools``.

Input format (one entry per image, in the same order for truth and predictions)::

    y_true = [{"boxes": [[x1, y1, x2, y2], ...], "labels": [3, ...]}, ...]          # optional "iscrowd", "area"
    y_pred = [{"boxes": [[x1, y1, x2, y2], ...], "labels": [3, ...], "scores": [0.9, ...]}, ...]

``box_format`` is ``"xyxy"`` (default), ``"xywh"`` (COCO) or ``"cxcywh"`` (YOLO-style centres).
:func:`from_coco` converts COCO-format ground truth and result files to this format.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

from ..core.exceptions import InputValidationError
from ..core.export import PathLike, csv_text, html_document, html_table, save_as
from ..core.registry import register
from ..core.result import MetricResult, _fmt, _json_safe, _latex_escape, _latex_table

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "DetectionReport",
    "average_precision_detection",
    "box_iou",
    "detection_pr_curve",
    "detection_report",
    "from_coco",
    "mean_average_precision",
]

_C = "detection"
BoxFormat = Literal["xyxy", "xywh", "cxcywh"]
_REF_COCO = "Lin TY, Maire M, Belongie S, et al. Microsoft COCO: common objects in context. ECCV 2014:740-755."
_REF_VOC = (
    "Everingham M, Van Gool L, Williams CKI, Winn J, Zisserman A. The PASCAL Visual Object Classes (VOC) "
    "challenge. Int J Comput Vis. 2010;88(2):303-338."
)
_REF_PADILLA = (
    "Padilla R, Netto SL, da Silva EAB. A survey on performance metrics for object-detection algorithms. "
    "IWSSIP 2020:237-242."
)
_AREAS = {"all": (0.0, 1e10), "small": (0.0, 32.0**2), "medium": (32.0**2, 96.0**2), "large": (96.0**2, 1e10)}
_REC_THRS = np.linspace(0.0, 1.00, 101)
_COCO_IOUS = np.linspace(0.5, 0.95, 10)


# ---- boxes -----------------------------------------------------------------------------------------
def _to_xyxy(boxes: Any, fmt: BoxFormat, name: str) -> NDArray[np.float64]:
    b = np.asarray(boxes, dtype=np.float64)
    if b.size == 0:
        return np.zeros((0, 4))
    if b.ndim != 2 or b.shape[1] != 4:
        raise InputValidationError(f"{name} must have shape (n, 4); received {b.shape}.")
    if not np.all(np.isfinite(b)):
        raise InputValidationError(f"{name} contains NaN or infinite coordinates.")
    out: NDArray[np.float64]
    if fmt == "xyxy":
        out = b.copy()
    elif fmt == "xywh":
        out = np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])
    elif fmt == "cxcywh":
        out = np.column_stack(
            [b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]
        )
    else:
        raise InputValidationError("box_format must be 'xyxy', 'xywh' or 'cxcywh'.")
    if np.any(out[:, 2] < out[:, 0]) or np.any(out[:, 3] < out[:, 1]):
        raise InputValidationError(f"{name} has boxes with negative width or height (check box_format).")
    return np.asarray(out, dtype=np.float64)


def _count(n: int, noun: str) -> str:
    plural = noun + ("es" if noun.endswith("s") else "s")
    return f"{n} {noun if n == 1 else plural}"


def _area(b: NDArray[np.float64]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return out


def _iou_matrix(
    dt: NDArray[np.float64], gt: NDArray[np.float64], crowd: Optional[NDArray[np.bool_]] = None
) -> NDArray[np.float64]:
    """IoU of every detection with every ground truth; for crowd ground truth, intersection over the
    detection's area (as in COCO)."""
    if dt.shape[0] == 0 or gt.shape[0] == 0:
        return np.zeros((dt.shape[0], gt.shape[0]))
    lt = np.maximum(dt[:, None, :2], gt[None, :, :2])
    rb = np.minimum(dt[:, None, 2:], gt[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    a_d, a_g = _area(dt)[:, None], _area(gt)[None, :]
    union = a_d + a_g - inter
    if crowd is not None and crowd.any():
        union = np.where(crowd[None, :], a_d, union)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


@register(
    category=_C,
    task="detection",
    name="Box IoU",
    definition="Overlap of two bounding boxes: area of intersection over area of union.",
    formula="IoU(A, B) = area(A ∩ B) / area(A ∪ B)",
    range="[0, 1]",
    input_requirements=("boxes_a (n, 4)", "boxes_b (m, 4)"),
    references=(_REF_VOC, _REF_COCO),
)
def box_iou(boxes_a: Any, boxes_b: Any, *, box_format: BoxFormat = "xyxy") -> NDArray[np.float64]:
    """IoU matrix of shape (n, m) between two sets of boxes."""
    return _iou_matrix(_to_xyxy(boxes_a, box_format, "boxes_a"), _to_xyxy(boxes_b, box_format, "boxes_b"))


# ---- input parsing ---------------------------------------------------------------------------------
@dataclass
class _Image:
    gt_boxes: NDArray[np.float64]
    gt_labels: NDArray[np.int64]
    gt_crowd: NDArray[np.bool_]
    gt_area: NDArray[np.float64]
    dt_boxes: NDArray[np.float64]
    dt_labels: NDArray[np.int64]
    dt_scores: NDArray[np.float64]


def _labels(x: Any, n: int, name: str) -> NDArray[np.int64]:
    arr = np.asarray(x if x is not None else [], dtype=np.int64).ravel()
    if arr.shape[0] != n:
        raise InputValidationError(f"{name} must have one label per box ({n}); received {arr.shape[0]}.")
    return arr


def _parse(
    y_true: Sequence[Mapping[str, Any]], y_pred: Sequence[Mapping[str, Any]], fmt: BoxFormat
) -> list[_Image]:
    if isinstance(y_true, Mapping) or isinstance(y_pred, Mapping):
        raise InputValidationError("y_true and y_pred must be lists with one dict per image.")
    t, p = list(y_true), list(y_pred)
    if len(t) != len(p):
        raise InputValidationError(f"y_true has {len(t)} images but y_pred has {len(p)}.")
    if not t:
        raise InputValidationError("No images to evaluate.")
    images = []
    for i, (g, d) in enumerate(zip(t, p)):
        if not isinstance(g, Mapping) or not isinstance(d, Mapping):
            raise InputValidationError(f"Image {i}: expected dicts with 'boxes' and 'labels'.")
        gb = _to_xyxy(g.get("boxes", []), fmt, f"y_true[{i}]['boxes']")
        db = _to_xyxy(d.get("boxes", []), fmt, f"y_pred[{i}]['boxes']")
        gl = _labels(g.get("labels"), gb.shape[0], f"y_true[{i}]['labels']")
        dl = _labels(d.get("labels"), db.shape[0], f"y_pred[{i}]['labels']")
        scores = np.asarray(d.get("scores", np.ones(db.shape[0])), dtype=np.float64).ravel()
        if scores.shape[0] != db.shape[0] or not np.all(np.isfinite(scores)):
            raise InputValidationError(f"y_pred[{i}]['scores'] must have one finite score per box.")
        crowd = np.asarray(g.get("iscrowd", np.zeros(gb.shape[0])), dtype=bool).ravel()
        area = np.asarray(g["area"], dtype=np.float64).ravel() if "area" in g else _area(gb)
        if crowd.shape[0] != gb.shape[0] or area.shape[0] != gb.shape[0]:
            raise InputValidationError(f"y_true[{i}]: 'iscrowd' and 'area' need one value per box.")
        images.append(_Image(gb, gl, crowd, area, db, dl, scores))
    return images


# ---- COCO evaluation ---------------------------------------------------------------------------------
@dataclass
class _Eval:
    dt_scores: NDArray[np.float64]
    dt_matched: NDArray[np.bool_]  # (T, D)
    dt_ignore: NDArray[np.bool_]  # (T, D)
    gt_ignore: NDArray[np.bool_]  # (G,)


@dataclass
class _Pair:
    """One image and one class: ground truth, score-sorted (truncated) detections and their IoUs."""

    gt_b: NDArray[np.float64]
    gt_crowd: NDArray[np.bool_]
    gt_area: NDArray[np.float64]
    dt_b: NDArray[np.float64]
    dt_s: NDArray[np.float64]
    dt_area: NDArray[np.float64]
    ious: NDArray[np.float64]


def _prepare(images: list[_Image], max_det: int) -> tuple[list[int], dict[int, list[_Pair]]]:
    """Group boxes by class once and compute every IoU once (as COCOeval.computeIoU), so area ranges and
    detection limits reuse them."""
    cats = sorted({int(c) for im in images for c in (*im.gt_labels.tolist(), *im.dt_labels.tolist())})
    pairs: dict[int, list[_Pair]] = {c: [] for c in cats}
    for im in images:
        for cat in np.union1d(im.gt_labels, im.dt_labels).astype(int).tolist():
            gmask = im.gt_labels == cat
            dmask = im.dt_labels == cat
            dt_b, dt_s = im.dt_boxes[dmask], im.dt_scores[dmask]
            order = np.argsort(-dt_s, kind="mergesort")[:max_det]  # stable, as COCOeval
            dt_b, dt_s = dt_b[order], dt_s[order]
            gt_b, crowd = im.gt_boxes[gmask], im.gt_crowd[gmask]
            pairs[cat].append(
                _Pair(gt_b, crowd, im.gt_area[gmask], dt_b, dt_s, _area(dt_b), _iou_matrix(dt_b, gt_b, crowd))
            )
    return cats, pairs


def _evaluate_image(pair: _Pair, area_rng: tuple[float, float], iou_thrs: NDArray[np.float64]) -> _Eval:
    dt_b, dt_s = pair.dt_b, pair.dt_s
    g_ignore = pair.gt_crowd | (pair.gt_area < area_rng[0]) | (pair.gt_area > area_rng[1])
    gorder = np.argsort(g_ignore, kind="mergesort")
    gt_b, gt_crowd, g_ignore = pair.gt_b[gorder], pair.gt_crowd[gorder], g_ignore[gorder]
    ious = pair.ious[:, gorder]
    t_n, d_n, g_n = len(iou_thrs), dt_b.shape[0], gt_b.shape[0]
    gtm = np.zeros((t_n, g_n), dtype=bool)
    dtm = np.zeros((t_n, d_n), dtype=bool)
    dt_ig = np.zeros((t_n, d_n), dtype=bool)
    # COCOeval's greedy matching, vectorised over thresholds and ground truths: for each detection (in score
    # order) take the last best-IoU available ground truth at or above the threshold, preferring ground truths
    # that are not ignored (COCO stops at the first ignored one once a regular match exists); crowd regions
    # stay available after a match.
    thr = np.minimum(np.asarray(iou_thrs, dtype=np.float64), 1 - 1e-10)
    if g_n:
        rows = np.arange(t_n)
        for di in range(d_n):
            row = ious[di]
            ok = (~gtm | gt_crowd[None, :]) & (row[None, :] >= thr[:, None])
            regular = ok & ~g_ignore[None, :]
            cand = np.where(regular.any(1)[:, None], regular, ok)
            found = cand.any(1)
            if not found.any():
                continue
            vals = np.where(cand, row[None, :], -1.0)
            m = g_n - 1 - np.argmax(vals[:, ::-1], axis=1)
            hit = rows[found]
            dtm[hit, di] = True
            dt_ig[hit, di] = g_ignore[m[found]]
            gtm[hit, m[found]] = True
    dt_area = pair.dt_area
    outside = (dt_area < area_rng[0]) | (dt_area > area_rng[1])
    dt_ig = dt_ig | (~dtm & outside[None, :])
    return _Eval(dt_s, dtm, dt_ig, g_ignore)


def _coco_eval(
    images: list[_Image],
    iou_thrs: NDArray[np.float64],
    max_dets: Sequence[int],
    areas: Sequence[str],
) -> tuple[list[int], NDArray[np.float64], NDArray[np.float64]]:
    """precision (T, R, K, A, M) and recall (T, K, A, M); -1 where undefined (no ground truth)."""
    top = max(max_dets)
    cats, pairs = _prepare(images, top)
    t_n, r_n, k_n, a_n, m_n = len(iou_thrs), _REC_THRS.size, len(cats), len(areas), len(max_dets)
    precision = -np.ones((t_n, r_n, k_n, a_n, m_n))
    recall = -np.ones((t_n, k_n, a_n, m_n))
    for k, cat in enumerate(cats):
        for a, area_name in enumerate(areas):
            evals = [_evaluate_image(pair, _AREAS[area_name], iou_thrs) for pair in pairs[cat]]
            if not evals:
                continue
            gt_ignore = np.concatenate([e.gt_ignore for e in evals])
            npig = int(np.count_nonzero(~gt_ignore))
            if npig == 0:
                continue
            for m, max_det in enumerate(max_dets):
                scores = np.concatenate([e.dt_scores[:max_det] for e in evals])
                order = np.argsort(-scores, kind="mergesort")
                dtm = np.concatenate([e.dt_matched[:, :max_det] for e in evals], axis=1)[:, order]
                dtig = np.concatenate([e.dt_ignore[:, :max_det] for e in evals], axis=1)[:, order]
                tps = dtm & ~dtig
                fps = ~dtm & ~dtig
                tp_sum = np.cumsum(tps, axis=1).astype(np.float64)
                fp_sum = np.cumsum(fps, axis=1).astype(np.float64)
                for ti in range(t_n):
                    tp, fp = tp_sum[ti], fp_sum[ti]
                    nd = tp.shape[0]
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    recall[ti, k, a, m] = rc[-1] if nd else 0.0
                    pr = np.maximum.accumulate(pr[::-1])[::-1] if nd else pr
                    idx = np.searchsorted(rc, _REC_THRS, side="left")
                    q = np.zeros(r_n)
                    valid = idx < nd
                    q[valid] = pr[idx[valid]]
                    precision[ti, :, k, a, m] = q
    return cats, precision, recall


def _mean_valid(x: NDArray[np.float64]) -> float:
    v = x[x > -1]
    return float(np.mean(v)) if v.size else math.nan


# ---- results -----------------------------------------------------------------------------------------
_SUMMARY = (
    ("map", "mAP@[.50:.95]"),
    ("map_50", "mAP@.50"),
    ("map_75", "mAP@.75"),
    ("map_small", "mAP small"),
    ("map_medium", "mAP medium"),
    ("map_large", "mAP large"),
    ("mar_1", "mAR@1"),
    ("mar_10", "mAR@10"),
    ("mar_100", "mAR@100"),
    ("mar_small", "mAR small"),
    ("mar_medium", "mAR medium"),
    ("mar_large", "mAR large"),
)


@dataclass(frozen=True, eq=False)
class DetectionReport:
    """COCO summary (mAP@[.50:.95], mAP@.50, mAP@.75, by object size, and mean average recall) plus AP per
    class. ``report["map"]`` is the headline number; NaN marks a value with no ground truth to score."""

    values: Any
    per_class: Any
    params: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "per_class", MappingProxyType(dict(self.per_class)))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __getitem__(self, name: str) -> float:
        try:
            return float(self.values[name])
        except KeyError:
            raise KeyError(f"No value {name!r}. Available: {', '.join(self.values)}.") from None

    def __float__(self) -> float:
        return self["map"]

    def summary(self, *, digits: int = 3) -> str:
        p = self.params
        lines = [
            f"EvalSuite detection evaluation (COCO protocol, {_count(p['n_images'], 'image')}, "
            f"{_count(p['n_classes'], 'class')}, "
            f"max {p['max_detections']} detections per image)"
        ]
        width = max(len(label) for _, label in _SUMMARY)
        lines += [f"  {label:<{width}}  {_fmt(self.values[k], digits)}" for k, label in _SUMMARY]
        lines.append(
            "  AP@[.50:.95] per class: " + ", ".join(f"{c}: {_fmt(v, digits)}" for c, v in self.per_class.items())
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            _json_safe(
                {
                    "values": dict(self.values),
                    "per_class_ap": {str(k): v for k, v in self.per_class.items()},
                    "params": dict(self.params),
                }
            ),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        frame: pd.DataFrame = pd.DataFrame({"value": dict(self.values)})
        return frame

    def _rows(self, digits: int) -> list[list[str]]:
        return [[label, _fmt(self.values[k], digits)] for k, label in _SUMMARY]

    def to_markdown(self, *, digits: int = 3) -> str:
        lines = ["| Measure | Value |", "| --- | ---: |"] + [
            "| " + " | ".join(r) + " |" for r in self._rows(digits)
        ]
        lines += ["", "| Class | AP@[.50:.95] |", "| --- | ---: |"]
        lines += [f"| {c} | {_fmt(v, digits)} |" for c, v in self.per_class.items()]
        return "\n".join(lines)

    def to_latex(self, *, digits: int = 3, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        rows = [[_latex_escape(c) for c in r] for r in self._rows(digits)]
        return _latex_table(["Measure", "Value"], rows, caption or "Object detection (COCO protocol).", label)

    def to_csv(self, path: Optional[PathLike] = None) -> str:
        rows: list[list[Any]] = [[k, "", float(v)] for k, v in self.values.items()]
        rows += [["ap", c, float(v)] for c, v in self.per_class.items()]
        text = csv_text(["measure", "class", "value"], rows)
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def to_html(self, *, digits: int = 3, full: bool = True) -> str:
        body = html_table(["Measure", "Value"], self._rows(digits), caption="COCO summary")
        body += "\n" + html_table(
            ["Class", "AP@[.50:.95]"],
            [[str(c), _fmt(v, digits)] for c, v in self.per_class.items()],
            caption="Average precision per class",
        )
        return html_document("EvalSuite detection evaluation", body) if full else body

    def save(self, path: PathLike) -> str:
        return save_as(
            path,
            {
                "json": self.to_json,
                "csv": self.to_csv,
                "markdown": self.to_markdown,
                "latex": self.to_latex,
                "html": self.to_html,
                "text": self.summary,
            },
        )


Detections = Sequence[Mapping[str, Any]]


def detection_report(
    y_true: Detections,
    y_pred: Detections,
    *,
    box_format: BoxFormat = "xyxy",
    max_detections: int = 100,
) -> DetectionReport:
    """Full COCO evaluation (the 12 numbers pycocotools prints, plus AP per class)."""
    if not (isinstance(max_detections, (int, np.integer)) and max_detections >= 1):
        raise InputValidationError("max_detections must be a positive integer.")
    images = _parse(y_true, y_pred, box_format)
    max_dets = sorted({1, 10, int(max_detections)})
    areas = ("all", "small", "medium", "large")
    cats, prec, rec = _coco_eval(images, _COCO_IOUS, max_dets, areas)
    m = max_dets.index(int(max_detections))
    t50, t75 = 0, 5

    def ap(t: Optional[int], a: int) -> float:
        p = prec[:, :, :, a, m] if t is None else prec[t, :, :, a, m]
        return _mean_valid(p)

    def ar(a: int, mi: int) -> float:
        return _mean_valid(rec[:, :, a, mi])

    values = {
        "map": ap(None, 0),
        "map_50": ap(t50, 0),
        "map_75": ap(t75, 0),
        "map_small": ap(None, 1),
        "map_medium": ap(None, 2),
        "map_large": ap(None, 3),
        "mar_1": ar(0, max_dets.index(1)),
        "mar_10": ar(0, max_dets.index(10)),
        "mar_100": ar(0, m),
        "mar_small": ar(1, m),
        "mar_medium": ar(2, m),
        "mar_large": ar(3, m),
    }
    per_class = {c: _mean_valid(prec[:, :, k, 0, m]) for k, c in enumerate(cats)}
    return DetectionReport(
        values,
        per_class,
        {
            "n_images": len(images),
            "n_classes": len(cats),
            "max_detections": int(max_detections),
            "iou_thresholds": _COCO_IOUS.round(2).tolist(),
            "box_format": box_format,
        },
    )


@register(
    category=_C,
    task="detection",
    name="Mean average precision (COCO mAP)",
    definition="Average precision (area under the 101-point interpolated precision-recall curve) averaged over "
    "classes and over IoU thresholds 0.50:0.05:0.95, as in the COCO benchmark.",
    formula="mAP = (1/|T|·|K|) Σ_t Σ_k AP_k(t)",
    range="[0, 1]",
    input_requirements=("y_true boxes and labels per image", "y_pred boxes, labels and scores per image"),
    references=(_REF_COCO, _REF_PADILLA),
)
def mean_average_precision(
    y_true: Detections,
    y_pred: Detections,
    *,
    iou_threshold: Optional[float] = None,
    box_format: BoxFormat = "xyxy",
    max_detections: int = 100,
) -> MetricResult:
    """COCO mAP@[.50:.95] by default; ``iou_threshold=0.5`` gives mAP@.50 (the PASCAL VOC-style headline)."""
    thrs = _COCO_IOUS if iou_threshold is None else np.array([_check_iou(iou_threshold)])
    images = _parse(y_true, y_pred, box_format)
    _cats, prec, _rec = _coco_eval(images, thrs, [int(max_detections)], ("all",))
    name = "mAP@[.50:.95]" if iou_threshold is None else f"mAP@{iou_threshold:g}"
    return MetricResult(
        "mean_average_precision",
        name,
        _mean_valid(prec[:, :, :, 0, 0]),
        {"iou_threshold": iou_threshold, "max_detections": int(max_detections), "interpolation": "coco-101"},
    )


def _check_iou(t: float) -> float:
    if not (0 < float(t) < 1):
        raise InputValidationError("iou_threshold must be between 0 and 1.")
    return float(t)


def _voc_ap(rc: NDArray[np.float64], pr: NDArray[np.float64]) -> float:
    """Area under the precision envelope at every recall change (PASCAL VOC 2010+, 'all points')."""
    mrec = np.concatenate([[0.0], rc, [1.0]])
    mpre = np.concatenate([[0.0], pr, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


@register(
    category=_C,
    task="detection",
    name="Average precision (per class)",
    definition="Area under the precision-recall curve of one class's detections at a fixed IoU threshold.",
    formula="AP = Σ_i (r_{i+1} − r_i) · p_interp(r_{i+1})",
    range="[0, 1]",
    input_requirements=("y_true boxes and labels per image", "y_pred boxes, labels and scores per image"),
    references=(_REF_VOC, _REF_COCO),
)
def average_precision_detection(
    y_true: Detections,
    y_pred: Detections,
    *,
    iou_threshold: float = 0.5,
    interpolation: Literal["coco", "voc"] = "coco",
    average: Optional[Literal["macro"]] = None,
    box_format: BoxFormat = "xyxy",
    max_detections: int = 100,
) -> MetricResult:
    """AP for each class at one IoU threshold (per class by default; ``average="macro"`` gives mAP).

    ``interpolation="coco"`` uses 101 recall points (as pycocotools); ``"voc"`` uses every recall change
    (PASCAL VOC 2010 and later). Crowd boxes and areas follow the COCO rules in both cases.
    """
    t = _check_iou(iou_threshold)
    images = _parse(y_true, y_pred, box_format)
    if interpolation == "coco":
        cats, prec, _ = _coco_eval(images, np.array([t]), [int(max_detections)], ("all",))
        per = np.array([_mean_valid(prec[0, :, k, 0, 0]) for k in range(len(cats))])
    elif interpolation == "voc":
        cats, pairs = _prepare(images, int(max_detections))
        per = np.full(len(cats), math.nan)
        for k, cat in enumerate(cats):
            evals = [_evaluate_image(pair, _AREAS["all"], np.array([t])) for pair in pairs[cat]]
            npig = int(sum(np.count_nonzero(~e.gt_ignore) for e in evals))
            if npig == 0:
                continue
            scores = np.concatenate([e.dt_scores for e in evals])
            order = np.argsort(-scores, kind="mergesort")
            dtm = np.concatenate([e.dt_matched[0] for e in evals])[order]
            dtig = np.concatenate([e.dt_ignore[0] for e in evals])[order]
            tp = np.cumsum(dtm & ~dtig).astype(np.float64)
            fp = np.cumsum(~dtm & ~dtig).astype(np.float64)
            keep = ~dtig
            tp, fp = tp[keep], fp[keep]
            per[k] = _voc_ap(tp / npig, tp / np.maximum(tp + fp, np.spacing(1))) if tp.size else 0.0
    else:
        raise InputValidationError("interpolation must be 'coco' or 'voc'.")
    params = {"iou_threshold": t, "interpolation": interpolation, "max_detections": int(max_detections)}
    if average is None:
        return MetricResult("average_precision_detection", f"AP@{t:g}", per, params, labels=tuple(cats))
    if average != "macro":
        raise InputValidationError("average must be 'macro' or None.")
    valid = per[~np.isnan(per)]
    return MetricResult(
        "average_precision_detection", f"mAP@{t:g}", float(valid.mean()) if valid.size else math.nan, params
    )


def detection_pr_curve(
    y_true: Detections,
    y_pred: Detections,
    *,
    iou_threshold: float = 0.5,
    box_format: BoxFormat = "xyxy",
    max_detections: int = 100,
) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """COCO 101-point interpolated precision-recall curve per class at one IoU threshold:
    ``{class: (recall_points, precision)}``. Classes without ground truth are left out."""
    t = _check_iou(iou_threshold)
    images = _parse(y_true, y_pred, box_format)
    cats, prec, _ = _coco_eval(images, np.array([t]), [int(max_detections)], ("all",))
    return {
        c: (_REC_THRS.copy(), prec[0, :, k, 0, 0].copy()) for k, c in enumerate(cats) if prec[0, 0, k, 0, 0] > -1
    }


def from_coco(
    ground_truth: Union[PathLike, Mapping[str, Any]],
    results: Union[PathLike, Sequence[Mapping[str, Any]], None] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert a COCO ground-truth file (or dict) and an optional COCO results list/file into
    ``(y_true, y_pred)`` with boxes in ``xywh`` format (pass ``box_format="xywh"``). Images keep the
    ground-truth file's order; images without annotations or detections get empty entries."""

    def load(x: Any) -> Any:
        if isinstance(x, (str, bytes)) or hasattr(x, "__fspath__"):
            with open(x, encoding="utf-8") as fh:
                return json.load(fh)
        return x

    gt = load(ground_truth)
    if not isinstance(gt, Mapping) or "images" not in gt or "annotations" not in gt:
        raise InputValidationError("ground_truth must be a COCO dataset with 'images' and 'annotations'.")
    ids = [img["id"] for img in gt["images"]]
    pos = {i: k for k, i in enumerate(ids)}
    y_true: list[dict[str, Any]] = [{"boxes": [], "labels": [], "iscrowd": [], "area": []} for _ in ids]
    for ann in gt["annotations"]:
        t = y_true[pos[ann["image_id"]]]
        t["boxes"].append(ann["bbox"])
        t["labels"].append(ann["category_id"])
        t["iscrowd"].append(ann.get("iscrowd", 0))
        w, h = ann["bbox"][2], ann["bbox"][3]
        t["area"].append(ann.get("area", w * h))
    y_pred: list[dict[str, Any]] = [{"boxes": [], "labels": [], "scores": []} for _ in ids]
    for det in load(results) or []:
        if det["image_id"] not in pos:
            raise InputValidationError(f"Detection for image_id {det['image_id']} is not in the ground truth.")
        d = y_pred[pos[det["image_id"]]]
        d["boxes"].append(det["bbox"])
        d["labels"].append(det["category_id"])
        d["scores"].append(det["score"])
    return y_true, y_pred
