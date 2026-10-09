"""Semantic segmentation metrics: overlap (Dice, IoU, pixel accuracy) and boundary/surface metrics
(Boundary IoU, Hausdorff distance, HD95, average symmetric surface distance).

Input conventions (stated once, applied everywhere):

* ``y_true`` and ``y_pred`` are integer label masks with the same shape. A 2-D array is one image. For
  three or more dimensions the **first axis indexes images** (``(N, H, W)`` or ``(N, D, H, W)``); pass a
  single 3-D volume as ``volume[None]``. A list of arrays allows images of different sizes.
* Boolean masks are treated as labels {0, 1}.
* ``ignore_index`` pixels are excluded from every count (for example 255 for "void").
* ``aggregate="dataset"`` (default) sums pixel counts over all images before computing each class's score
  (the standard for semantic segmentation benchmarks); ``"image"`` computes the score per image and averages
  the images (the usual convention in medical imaging).
* A class absent from both the truth and the prediction has an undefined score. It is NaN per class and is
  left out of the macro average (``empty="ignore"``); ``empty=1.0`` counts it as perfect instead.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..core.exceptions import InputValidationError, UndefinedMetricWarning
from ..core.export import PathLike, csv_text, html_document, html_table, save_as
from ..core.registry import register
from ..core.result import MetricResult, _fmt, _json_safe, _latex_escape, _latex_table
from ..core.types import ArrayLike

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "SegmentationReport",
    "average_surface_distance",
    "boundary_iou",
    "dice",
    "hausdorff_distance",
    "iou",
    "mean_pixel_accuracy",
    "miou",
    "pixel_accuracy",
    "per_image_scores",
    "segmentation_confusion",
    "segmentation_report",
]

_C = "segmentation"
Masks = Union[ArrayLike, Sequence[ArrayLike]]
Average = Optional[Literal["macro", "micro", "weighted"]]
Aggregate = Literal["dataset", "image"]
Empty = Union[Literal["ignore"], float]

_REF_DICE = "Dice LR. Measures of the amount of ecologic association between species. Ecology. 1945;26(3):297-302."
_REF_JACCARD = "Jaccard P. The distribution of the flora in the alpine zone. New Phytol. 1912;11(2):37-50."
_REF_VOC = (
    "Everingham M, Van Gool L, Williams CKI, Winn J, Zisserman A. The PASCAL Visual Object Classes (VOC) "
    "challenge. Int J Comput Vis. 2010;88(2):303-338."
)
_REF_LONG = (
    "Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation. CVPR 2015:3431-3440."
)
_REF_BIOU = (
    "Cheng B, Girshick R, Dollár P, Berg AC, Kirillov A. Boundary IoU: improving object-centric image "
    "segmentation evaluation. CVPR 2021:15334-15342."
)
_REF_MAIER = (
    "Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation. "
    "Nat Methods. 2024;21(2):195-212."
)
_REF_HD = (
    "Huttenlocher DP, Klanderman GA, Rucklidge WJ. Comparing images using the Hausdorff distance. IEEE Trans "
    "Pattern Anal Mach Intell. 1993;15(9):850-863."
)


# ---- input handling ------------------------------------------------------------------------------
def _as_images(masks: Masks, name: str) -> list[NDArray[Any]]:
    if isinstance(masks, (list, tuple)) or (isinstance(masks, np.ndarray) and masks.dtype == object):
        images = [np.asarray(m) for m in masks]
    else:
        arr = np.asarray(masks)
        if arr.ndim < 2:
            raise InputValidationError(
                f"{name} must be a 2-D mask, an array of masks with images on the first axis, or a list of masks; "
                f"received shape {arr.shape}."
            )
        images = [arr] if arr.ndim == 2 else list(arr)
    if not images:
        raise InputValidationError(f"{name} contains no images.")
    out = []
    for i, m in enumerate(images):
        if m.ndim < 2:
            raise InputValidationError(f"{name}[{i}] must have at least 2 dimensions; received shape {m.shape}.")
        if m.dtype == bool:
            m = m.astype(np.int64)
        if not np.issubdtype(m.dtype, np.integer):
            if np.issubdtype(m.dtype, np.floating) and np.all(np.isfinite(m)) and np.all(m == np.round(m)):
                m = m.astype(np.int64)
            else:
                raise InputValidationError(
                    f"{name} must contain integer class labels (or booleans); found dtype {m.dtype}. "
                    "Threshold or argmax probability maps first."
                )
        out.append(m)
    return out


def _pairs(y_true: Masks, y_pred: Masks) -> list[tuple[NDArray[Any], NDArray[Any]]]:
    t = _as_images(y_true, "y_true")
    p = _as_images(y_pred, "y_pred")
    if len(t) != len(p):
        raise InputValidationError(f"y_true has {len(t)} images but y_pred has {len(p)}.")
    for i, (a, b) in enumerate(zip(t, p)):
        if a.shape != b.shape:
            raise InputValidationError(f"Image {i}: y_true has shape {a.shape} but y_pred has shape {b.shape}.")
    return list(zip(t, p))


def _classes(
    pairs: list[tuple[NDArray[Any], NDArray[Any]]],
    num_classes: Optional[int],
    labels: Optional[Sequence[int]],
    ignore_index: Optional[int],
    include_background: bool,
) -> NDArray[np.int64]:
    if labels is not None:
        cls = np.asarray(labels, dtype=np.int64)
        if cls.ndim != 1 or cls.size == 0 or np.unique(cls).size != cls.size:
            raise InputValidationError("labels must be a non-empty list of distinct class indices.")
    else:
        if num_classes is None:
            top = -1
            for a, b in pairs:
                for m in (a, b):
                    valid = m[m != ignore_index] if ignore_index is not None else m
                    if valid.size:
                        top = max(top, int(valid.max()))
            num_classes = top + 1
        if num_classes < 1:
            raise InputValidationError("num_classes must be at least 1.")
        cls = np.arange(num_classes, dtype=np.int64)
    if not include_background:
        cls = cls[cls != 0]
        if cls.size == 0:
            raise InputValidationError("include_background=False leaves no classes to evaluate.")
    for a, b in pairs:
        for m, name in ((a, "y_true"), (b, "y_pred")):
            valid = m if ignore_index is None else m[m != ignore_index]
            if valid.size and int(valid.min()) < 0:
                raise InputValidationError(f"{name} contains negative labels; use ignore_index for void pixels.")
            if valid.size and num_classes is not None and labels is None and int(valid.max()) >= num_classes:
                raise InputValidationError(
                    f"{name} contains label {int(valid.max())}, but num_classes={num_classes} allows 0 to "
                    f"{num_classes - 1}."
                )
    return cls


def segmentation_confusion(
    y_true: Masks,
    y_pred: Masks,
    *,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> NDArray[np.int64]:
    """Pixel confusion matrix summed over all images: rows are true classes, columns predicted classes."""
    pairs = _pairs(y_true, y_pred)
    cls = _classes(pairs, num_classes, None, ignore_index, True)
    k = cls.size
    total = np.zeros((k, k), dtype=np.int64)
    for a, b in pairs:
        total += _confusion(a, b, k, ignore_index)
    return total


def _confusion(a: NDArray[Any], b: NDArray[Any], k: int, ignore_index: Optional[int]) -> NDArray[np.int64]:
    t = a.ravel().astype(np.int64)
    p = b.ravel().astype(np.int64)
    keep = np.ones(t.shape, dtype=bool) if ignore_index is None else (t != ignore_index)
    if ignore_index is not None:
        # a prediction of the ignore label on a valid pixel is a wrong prediction of "no class"
        p = np.where(p == ignore_index, k, p)
    if np.any(t[keep] >= k) or np.any(p[keep] > k):
        raise InputValidationError(f"Masks contain labels >= num_classes={k}.")
    flat = t[keep] * (k + 1) + p[keep]
    cm = np.bincount(flat, minlength=k * (k + 1)).reshape(k, k + 1)
    return cm[:, :k].astype(np.int64)


def _per_image_counts(
    pairs: list[tuple[NDArray[Any], NDArray[Any]]], cls: NDArray[np.int64], ignore_index: Optional[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """tp, fp, fn per image and class, shape (n_images, n_classes)."""
    k = max(int(np.max(cls)) + 1, 1)
    for a, b in pairs:
        for m in (a, b):
            valid = m if ignore_index is None else m[m != ignore_index]
            if valid.size:
                k = max(k, int(valid.max()) + 1)
    tp = np.zeros((len(pairs), cls.size))
    fp = np.zeros_like(tp)
    fn = np.zeros_like(tp)
    for i, (a, b) in enumerate(pairs):
        cm = _confusion(a, b, k, ignore_index)
        diag = np.diag(cm)
        # every valid true pixel of the class that is not predicted as the class is a miss, including pixels
        # predicted as the ignore label (dropped from the confusion matrix columns)
        t_valid = a.ravel() if ignore_index is None else a.ravel()[a.ravel() != ignore_index]
        true_count = np.bincount(t_valid.astype(np.int64), minlength=k)[:k]
        tp[i] = diag[cls]
        fp[i] = cm.sum(0)[cls] - diag[cls]
        fn[i] = true_count[cls] - diag[cls]
    return tp, fp, fn


def _score(
    tp: NDArray[np.float64],
    fp: NDArray[np.float64],
    fn: NDArray[np.float64],
    kind: Literal["dice", "iou"],
) -> NDArray[np.float64]:
    den = (2 * tp + fp + fn) if kind == "dice" else (tp + fp + fn)
    num = 2 * tp if kind == "dice" else tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _reduce(
    per_class: NDArray[np.float64],
    support: NDArray[np.float64],
    average: Average,
    empty: Empty,
    name: str,
) -> Union[float, NDArray[np.float64]]:
    if empty != "ignore":
        if not (isinstance(empty, (int, float)) and 0 <= float(empty) <= 1):
            raise InputValidationError('empty must be "ignore" or a value in [0, 1] (usually 1.0).')
        per_class = np.where(np.isnan(per_class), float(empty), per_class)
    if average is None:
        return per_class
    if average == "macro":
        if np.all(np.isnan(per_class)):
            warnings.warn(
                f"{name} is undefined: no class occurs in either mask.", UndefinedMetricWarning, stacklevel=4
            )
            return math.nan
        return float(np.nanmean(per_class))
    if average == "weighted":
        ok = ~np.isnan(per_class)
        if support[ok].sum() == 0:
            return math.nan
        return float(np.average(per_class[ok], weights=support[ok]))
    raise InputValidationError("average must be 'macro', 'micro', 'weighted' or None.")


def _overlap(
    kind: Literal["dice", "iou"],
    name: str,
    y_true: Masks,
    y_pred: Masks,
    *,
    num_classes: Optional[int],
    labels: Optional[Sequence[int]],
    ignore_index: Optional[int],
    average: Average,
    aggregate: Aggregate,
    include_background: bool,
    empty: Empty,
) -> MetricResult:
    pairs = _pairs(y_true, y_pred)
    cls = _classes(pairs, num_classes, labels, ignore_index, include_background)
    tp, fp, fn = _per_image_counts(pairs, cls, ignore_index)
    params: dict[str, Any] = {
        "average": average,
        "aggregate": aggregate,
        "ignore_index": ignore_index,
        "include_background": include_background,
        "empty": empty,
        "n_images": len(pairs),
    }
    if aggregate not in ("dataset", "image"):
        raise InputValidationError("aggregate must be 'dataset' or 'image'.")
    if average == "micro":
        if aggregate == "dataset":
            value = _score(np.array([tp.sum()]), np.array([fp.sum()]), np.array([fn.sum()]), kind)[0]
        else:
            value = float(np.nanmean(_score(tp.sum(1), fp.sum(1), fn.sum(1), kind)))
        return MetricResult(kind, name, float(value), params)
    support = (tp + fn).sum(0)
    if aggregate == "dataset":
        per_class = _score(tp.sum(0), fp.sum(0), fn.sum(0), kind)
    else:
        scores = _score(tp, fp, fn, kind)
        if empty != "ignore":
            scores = np.where(np.isnan(scores), float(empty), scores)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per_class = np.nanmean(scores, axis=0)
    value = _reduce(per_class, support, average, empty, name)
    if isinstance(value, np.ndarray):
        return MetricResult(kind, name, value, params, labels=tuple(int(c) for c in cls))
    return MetricResult(kind, name, value, params)


@register(
    category=_C,
    task="segmentation",
    name="Dice coefficient",
    definition="Overlap between predicted and true region of a class: twice the intersection over the sum of "
    "the two areas (identical to the F1 score on pixels).",
    formula="Dice = 2|A ∩ B| / (|A| + |B|) = 2TP / (2TP + FP + FN)",
    range="[0, 1]",
    input_requirements=("y_true mask", "y_pred mask"),
    references=(_REF_DICE, _REF_MAIER),
)
def dice(
    y_true: Masks,
    y_pred: Masks,
    *,
    num_classes: Optional[int] = None,
    labels: Optional[Sequence[int]] = None,
    ignore_index: Optional[int] = None,
    average: Average = "macro",
    aggregate: Aggregate = "dataset",
    include_background: bool = True,
    empty: Empty = "ignore",
) -> MetricResult:
    """Dice coefficient (Sørensen–Dice) per class.

    ``average``: ``"macro"`` (mean over classes, default), ``"micro"`` (pooled pixels), ``"weighted"`` (by
    true pixel count) or ``None`` (per class). See :mod:`evalsuite.vision.segmentation` for ``aggregate``,
    ``empty`` and the input conventions.
    """
    return _overlap(
        "dice",
        "Dice",
        y_true,
        y_pred,
        num_classes=num_classes,
        labels=labels,
        ignore_index=ignore_index,
        average=average,
        aggregate=aggregate,
        include_background=include_background,
        empty=empty,
    )


@register(
    category=_C,
    task="segmentation",
    name="Intersection over union (Jaccard)",
    definition="Overlap between predicted and true region of a class: intersection over union.",
    formula="IoU = |A ∩ B| / |A ∪ B| = TP / (TP + FP + FN)",
    range="[0, 1]",
    input_requirements=("y_true mask", "y_pred mask"),
    references=(_REF_JACCARD, _REF_VOC),
)
def iou(
    y_true: Masks,
    y_pred: Masks,
    *,
    num_classes: Optional[int] = None,
    labels: Optional[Sequence[int]] = None,
    ignore_index: Optional[int] = None,
    average: Average = "macro",
    aggregate: Aggregate = "dataset",
    include_background: bool = True,
    empty: Empty = "ignore",
) -> MetricResult:
    """Intersection over union (Jaccard index) per class; same options as :func:`dice`."""
    return _overlap(
        "iou",
        "IoU",
        y_true,
        y_pred,
        num_classes=num_classes,
        labels=labels,
        ignore_index=ignore_index,
        average=average,
        aggregate=aggregate,
        include_background=include_background,
        empty=empty,
    )


@register(
    category=_C,
    task="segmentation",
    name="Mean intersection over union (mIoU)",
    definition="IoU of each class from pixel counts summed over the dataset, averaged over the classes that "
    "occur: the standard semantic segmentation benchmark score.",
    formula="mIoU = (1/K) Σ_k TP_k / (TP_k + FP_k + FN_k)",
    range="[0, 1]",
    input_requirements=("y_true mask", "y_pred mask"),
    references=(_REF_VOC, _REF_LONG),
)
def miou(
    y_true: Masks,
    y_pred: Masks,
    *,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    include_background: bool = True,
) -> MetricResult:
    """Mean IoU: :func:`iou` with ``average="macro"`` and ``aggregate="dataset"``."""
    r = iou(
        y_true,
        y_pred,
        num_classes=num_classes,
        ignore_index=ignore_index,
        include_background=include_background,
    )
    return MetricResult("miou", "mIoU", float(r), r.params)


@register(
    category=_C,
    task="segmentation",
    name="Pixel accuracy",
    definition="Proportion of (non-ignored) pixels whose class is predicted correctly.",
    formula="Σ_k TP_k / N_pixels",
    range="[0, 1]",
    input_requirements=("y_true mask", "y_pred mask"),
    references=(_REF_LONG,),
)
def pixel_accuracy(y_true: Masks, y_pred: Masks, *, ignore_index: Optional[int] = None) -> MetricResult:
    """Overall pixel accuracy, excluding ``ignore_index`` pixels."""
    correct = total = 0
    for a, b in _pairs(y_true, y_pred):
        keep = np.ones(a.shape, bool) if ignore_index is None else a != ignore_index
        correct += int(np.sum((a == b) & keep))
        total += int(keep.sum())
    if total == 0:
        raise InputValidationError("Every pixel is ignored; nothing to evaluate.")
    return MetricResult("pixel_accuracy", "Pixel accuracy", correct / total, {"ignore_index": ignore_index})


@register(
    category=_C,
    task="segmentation",
    name="Mean pixel accuracy",
    definition="Per-class pixel recall (correct pixels of the class over its true pixels), averaged over the "
    "classes that occur in the truth.",
    formula="(1/K) Σ_k TP_k / (TP_k + FN_k)",
    range="[0, 1]",
    input_requirements=("y_true mask", "y_pred mask"),
    references=(_REF_LONG,),
)
def mean_pixel_accuracy(
    y_true: Masks,
    y_pred: Masks,
    *,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> MetricResult:
    """Mean of per-class pixel accuracy over classes present in ``y_true``."""
    pairs = _pairs(y_true, y_pred)
    cls = _classes(pairs, num_classes, None, ignore_index, True)
    tp, _fp, fn = _per_image_counts(pairs, cls, ignore_index)
    t, f = tp.sum(0), fn.sum(0)
    present = (t + f) > 0
    return MetricResult(
        "mean_pixel_accuracy",
        "Mean pixel accuracy",
        float(np.mean(t[present] / (t[present] + f[present]))),
        {"ignore_index": ignore_index},
    )


# ---- boundary and surface metrics ------------------------------------------------------------------
def _structure(ndim: int) -> NDArray[np.bool_]:
    out: NDArray[np.bool_] = ndimage.generate_binary_structure(ndim, 1)
    return out


def _surface(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Foreground pixels with at least one background neighbour (4/6-connectivity)."""
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, structure=_structure(mask.ndim), border_value=0)
    out: NDArray[np.bool_] = mask & ~eroded
    return out


def _directed_distances(
    a_surface: NDArray[np.bool_], b_surface: NDArray[np.bool_], spacing: Optional[Sequence[float]]
) -> NDArray[np.float64]:
    """Distance from each surface point of A to the nearest surface point of B."""
    dist = ndimage.distance_transform_edt(~b_surface, sampling=spacing)
    return np.asarray(dist[a_surface], dtype=np.float64)


def _check_spacing(spacing: Optional[Sequence[float]], ndim: int) -> Optional[tuple[float, ...]]:
    if spacing is None:
        return None
    sp = tuple(float(s) for s in spacing)
    if len(sp) != ndim or any(not (s > 0 and math.isfinite(s)) for s in sp):
        raise InputValidationError(f"spacing must have {ndim} positive values (one per image axis).")
    return sp


def _surface_metric(
    which: Literal["hd", "assd"],
    y_true: Masks,
    y_pred: Masks,
    *,
    num_classes: Optional[int],
    labels: Optional[Sequence[int]],
    ignore_index: Optional[int],
    include_background: bool,
    spacing: Optional[Sequence[float]],
    percentile: float,
    average: Optional[Literal["macro"]],
    metric: str,
    name: str,
) -> MetricResult:
    if not (0 < percentile <= 100):
        raise InputValidationError("percentile must be in (0, 100].")
    pairs = _pairs(y_true, y_pred)
    cls = _classes(pairs, num_classes, labels, ignore_index, include_background)
    values = np.full((len(pairs), cls.size), np.nan)
    one_empty = 0
    for i, (a, b) in enumerate(pairs):
        sp = _check_spacing(spacing, a.ndim)
        valid = np.ones(a.shape, bool) if ignore_index is None else a != ignore_index
        for j, c in enumerate(cls):
            ta, pb = (a == c) & valid, (b == c) & valid
            if not ta.any() and not pb.any():
                continue
            if not ta.any() or not pb.any():
                values[i, j] = math.inf
                one_empty += 1
                continue
            sa, sb = _surface(ta), _surface(pb)
            d_ab = _directed_distances(sa, sb, sp)
            d_ba = _directed_distances(sb, sa, sp)
            if which == "hd":
                values[i, j] = max(np.percentile(d_ab, percentile), np.percentile(d_ba, percentile))
            else:
                values[i, j] = (d_ab.sum() + d_ba.sum()) / (d_ab.size + d_ba.size)
    if one_empty:
        warnings.warn(
            f"{name} is infinite for {one_empty} image/class pair(s) where the class occurs in only one of the "
            "two masks.",
            UndefinedMetricWarning,
            stacklevel=3,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_class = np.nanmean(values, axis=0)
    params: dict[str, Any] = {
        "spacing": None if spacing is None else tuple(float(s) for s in spacing),
        "ignore_index": ignore_index,
        "include_background": include_background,
        "n_images": len(pairs),
        "aggregate": "image",
    }
    if which == "hd":
        params["percentile"] = percentile
    if average is None:
        return MetricResult(metric, name, per_class, params, labels=tuple(int(c) for c in cls))
    if average != "macro":
        raise InputValidationError("average must be 'macro' or None for surface distances.")
    if np.all(np.isnan(per_class)):
        warnings.warn(f"{name} is undefined: no class occurs in any mask.", UndefinedMetricWarning, stacklevel=3)
        return MetricResult(metric, name, math.nan, params)
    return MetricResult(metric, name, float(np.nanmean(per_class)), params)


@register(
    category=_C,
    task="segmentation",
    name="Hausdorff distance",
    definition="Largest distance from a point on one boundary to the nearest point on the other (in pixels, "
    "or physical units with spacing); percentile=95 gives the robust HD95.",
    formula="HD = max(max_{a∈∂A} d(a, ∂B), max_{b∈∂B} d(b, ∂A))",
    range="[0, ∞)",
    input_requirements=("y_true mask", "y_pred mask"),
    references=(_REF_HD, _REF_MAIER),
    higher_is_better=False,
)
def hausdorff_distance(
    y_true: Masks,
    y_pred: Masks,
    *,
    percentile: float = 100.0,
    spacing: Optional[Sequence[float]] = None,
    num_classes: Optional[int] = None,
    labels: Optional[Sequence[int]] = None,
    ignore_index: Optional[int] = None,
    include_background: bool = False,
    average: Optional[Literal["macro"]] = "macro",
) -> MetricResult:
    """Symmetric Hausdorff distance between class boundaries, per image, averaged over images then classes.

    ``percentile=95`` returns HD95 (the 95th percentile of each directed distance set, then the larger of
    the two, as in MONAI). Boundaries are mask pixels with a background neighbour (face connectivity).
    ``spacing`` gives the physical size of a pixel/voxel along each axis. The background class is excluded
    by default. A class present in only one mask gives ``inf`` (with a warning).
    """
    return _surface_metric(
        "hd",
        y_true,
        y_pred,
        num_classes=num_classes,
        labels=labels,
        ignore_index=ignore_index,
        include_background=include_background,
        spacing=spacing,
        percentile=percentile,
        average=average,
        metric="hausdorff_distance",
        name="HD95" if percentile == 95 else ("Hausdorff distance" if percentile == 100 else f"HD{percentile:g}"),
    )


@register(
    category=_C,
    task="segmentation",
    name="Average symmetric surface distance",
    definition="Mean distance from every boundary point of each mask to the nearest boundary point of the "
    "other, pooled over both directions.",
    formula="ASSD = (Σ_{a∈∂A} d(a, ∂B) + Σ_{b∈∂B} d(b, ∂A)) / (|∂A| + |∂B|)",
    range="[0, ∞)",
    input_requirements=("y_true mask", "y_pred mask"),
    references=(_REF_MAIER,),
    higher_is_better=False,
)
def average_surface_distance(
    y_true: Masks,
    y_pred: Masks,
    *,
    spacing: Optional[Sequence[float]] = None,
    num_classes: Optional[int] = None,
    labels: Optional[Sequence[int]] = None,
    ignore_index: Optional[int] = None,
    include_background: bool = False,
    average: Optional[Literal["macro"]] = "macro",
) -> MetricResult:
    """Average symmetric surface distance (ASSD), per image, averaged over images then classes."""
    return _surface_metric(
        "assd",
        y_true,
        y_pred,
        num_classes=num_classes,
        labels=labels,
        ignore_index=ignore_index,
        include_background=include_background,
        spacing=spacing,
        percentile=100.0,
        average=average,
        metric="average_surface_distance",
        name="ASSD",
    )


def _boundary_region(mask: NDArray[np.bool_], d: int) -> NDArray[np.bool_]:
    """Pixels of the mask within distance d of its contour (mask minus its d-step erosion), as in Cheng et al."""
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(
        mask, structure=ndimage.generate_binary_structure(mask.ndim, mask.ndim), iterations=d, border_value=0
    )
    out: NDArray[np.bool_] = mask & ~eroded
    return out


@register(
    category=_C,
    task="segmentation",
    name="Boundary IoU",
    definition="IoU computed only on the pixels within distance d of each mask's contour, so it is sensitive "
    "to boundary quality and fair across object sizes; d is a fraction of the image diagonal.",
    formula="BIoU = |(G_d ∩ G) ∩ (P_d ∩ P)| / |(G_d ∩ G) ∪ (P_d ∩ P)|",
    range="[0, 1]",
    input_requirements=("y_true mask", "y_pred mask"),
    references=(_REF_BIOU,),
)
def boundary_iou(
    y_true: Masks,
    y_pred: Masks,
    *,
    dilation_ratio: float = 0.02,
    num_classes: Optional[int] = None,
    labels: Optional[Sequence[int]] = None,
    ignore_index: Optional[int] = None,
    include_background: bool = False,
    average: Average = "macro",
    empty: Empty = "ignore",
) -> MetricResult:
    """Boundary IoU (Cheng et al. 2021) per class, with the boundary band width
    d = max(1, round(dilation_ratio × image diagonal)). Counts are summed over images per class."""
    if not (0 < dilation_ratio < 1):
        raise InputValidationError("dilation_ratio must be between 0 and 1 (0.02 is the paper's default).")
    pairs = _pairs(y_true, y_pred)
    cls = _classes(pairs, num_classes, labels, ignore_index, include_background)
    inter = np.zeros(cls.size)
    union = np.zeros(cls.size)
    for a, b in pairs:
        d = max(1, round(dilation_ratio * math.sqrt(sum(s * s for s in a.shape))))
        valid = np.ones(a.shape, bool) if ignore_index is None else a != ignore_index
        for j, c in enumerate(cls):
            gb = _boundary_region((a == c) & valid, d)
            pb = _boundary_region((b == c) & valid, d)
            inter[j] += np.sum(gb & pb)
            union[j] += np.sum(gb | pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(union > 0, inter / np.where(union > 0, union, 1), np.nan)
    params: dict[str, Any] = {
        "dilation_ratio": dilation_ratio,
        "average": average,
        "ignore_index": ignore_index,
        "include_background": include_background,
        "n_images": len(pairs),
    }
    if average == "micro":
        return MetricResult("boundary_iou", "Boundary IoU", float(inter.sum() / union.sum()), params)
    value = _reduce(per_class, union, average, empty, "Boundary IoU")
    if isinstance(value, np.ndarray):
        return MetricResult("boundary_iou", "Boundary IoU", value, params, labels=tuple(int(c) for c in cls))
    return MetricResult("boundary_iou", "Boundary IoU", value, params)


def per_image_scores(
    y_true: Masks,
    y_pred: Masks,
    metric: Literal["dice", "iou"] = "dice",
    *,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    include_background: bool = True,
) -> NDArray[np.float64]:
    """Macro Dice or IoU of each image separately (for paired tests between models over the same images)."""
    pairs = _pairs(y_true, y_pred)
    cls = _classes(pairs, num_classes, None, ignore_index, include_background)
    tp, fp, fn = _per_image_counts(pairs, cls, ignore_index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanmean(_score(tp, fp, fn, metric), axis=1), dtype=np.float64)


# ---- report --------------------------------------------------------------------------------------------
_REPORT_COLUMNS = ("dice", "iou", "boundary_iou", "hd95", "assd")


@dataclass(frozen=True, eq=False)
class SegmentationReport:
    """Every segmentation measure in one object: dataset-level summary values plus a per-class table
    (Dice, IoU, Boundary IoU, HD95, ASSD and pixel support)."""

    values: Any
    rows: tuple[Any, ...]
    params: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "rows", tuple(MappingProxyType(dict(r)) for r in self.rows))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __getitem__(self, name: Any) -> Any:
        if name in self.values:
            return float(self.values[name])
        for row in self.rows:
            if row["class"] == name:
                return row
        raise KeyError(f"No value or class {name!r}. Values: {', '.join(self.values)}.")

    def _table(self, digits: int) -> tuple[list[str], list[list[str]]]:
        header = ["Class", "Dice", "IoU", "Boundary IoU", "HD95", "ASSD", "Pixels"]
        body = [
            [str(r["name"]), *(_fmt(r[c], digits) for c in _REPORT_COLUMNS), f"{int(r['support']):,}"]
            for r in self.rows
        ]
        return header, body

    def summary(self, *, digits: int = 4) -> str:
        p = self.params
        unit = " (physical units)" if p.get("spacing") else " (pixels)"
        lines = [
            f"EvalSuite segmentation evaluation ({p['n_images']} images, {len(self.rows)} classes, "
            f"aggregate={p['aggregate']})",
            f"  mIoU                        {_fmt(self.values['miou'], digits)}",
            f"  Mean Dice                   {_fmt(self.values['dice'], digits)}",
            f"  Pixel accuracy              {_fmt(self.values['pixel_accuracy'], digits)}",
            f"  Mean pixel accuracy         {_fmt(self.values['mean_pixel_accuracy'], digits)}",
            f"  Boundary IoU                {_fmt(self.values['boundary_iou'], digits)}",
            f"  {'HD95' + unit:<27} {_fmt(self.values['hd95'], digits)}",
            f"  {'ASSD' + unit:<27} {_fmt(self.values['assd'], digits)}",
            "",
        ]
        header, body = self._table(digits)
        widths = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(header)]
        lines.append("  ".join(h.ljust(widths[i]) if i == 0 else h.rjust(widths[i]) for i, h in enumerate(header)))
        lines += [
            "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(r)) for r in body
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            _json_safe(
                {
                    "values": dict(self.values),
                    "per_class": [dict(r) for r in self.rows],
                    "params": dict(self.params),
                }
            ),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        frame: pd.DataFrame = pd.DataFrame([dict(r) for r in self.rows]).set_index("class")
        return frame

    def to_markdown(self, *, digits: int = 4) -> str:
        header, body = self._table(digits)
        lines = ["| " + " | ".join(header) + " |", "| --- |" + " ---: |" * (len(header) - 1)]
        return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in body])

    def to_latex(self, *, digits: int = 4, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        header, body = self._table(digits)
        return _latex_table(
            [_latex_escape(h) for h in header],
            [[_latex_escape(c) for c in r] for r in body],
            caption or "Segmentation results per class.",
            label,
        )

    def to_csv(self, path: Optional[PathLike] = None) -> str:
        cols = ["class", "name", *_REPORT_COLUMNS, "support"]
        text = csv_text(cols, [[r[c] for c in cols] for r in self.rows])
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def to_html(self, *, digits: int = 4, full: bool = True) -> str:
        summary = [
            ["mIoU", _fmt(self.values["miou"], digits)],
            ["Mean Dice", _fmt(self.values["dice"], digits)],
            ["Pixel accuracy", _fmt(self.values["pixel_accuracy"], digits)],
            ["Boundary IoU", _fmt(self.values["boundary_iou"], digits)],
            ["HD95", _fmt(self.values["hd95"], digits)],
            ["ASSD", _fmt(self.values["assd"], digits)],
        ]
        header, body = self._table(digits)
        html = html_table(["Measure", "Value"], summary, caption="Summary")
        html += "\n" + html_table(header, body, caption="Per class")
        return html_document("EvalSuite segmentation evaluation", html) if full else html

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


def segmentation_report(
    y_true: Masks,
    y_pred: Masks,
    *,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    class_names: Optional[Mapping[int, str]] = None,
    spacing: Optional[Sequence[float]] = None,
    aggregate: Aggregate = "dataset",
    include_background: bool = True,
    dilation_ratio: float = 0.02,
) -> SegmentationReport:
    """Dice, IoU, Boundary IoU, HD95 and ASSD per class plus mIoU, mean Dice, pixel accuracy and mean pixel
    accuracy, in one call. Surface distances are per image (averaged), overlap scores use ``aggregate``."""
    pairs = _pairs(y_true, y_pred)
    cls = _classes(pairs, num_classes, None, ignore_index, include_background)
    common: dict[str, Any] = {"labels": cls.tolist(), "ignore_index": ignore_index, "include_background": True}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        d = np.asarray(dice(y_true, y_pred, average=None, aggregate=aggregate, **common).value, dtype=np.float64)
        j = np.asarray(iou(y_true, y_pred, average=None, aggregate=aggregate, **common).value, dtype=np.float64)
        b = np.asarray(
            boundary_iou(y_true, y_pred, average=None, dilation_ratio=dilation_ratio, **common).value,
            dtype=np.float64,
        )
        hd = np.asarray(
            hausdorff_distance(y_true, y_pred, percentile=95, spacing=spacing, average=None, **common).value,
            dtype=np.float64,
        )
        sd = np.asarray(
            average_surface_distance(y_true, y_pred, spacing=spacing, average=None, **common).value,
            dtype=np.float64,
        )
    tp, _fp, fn = _per_image_counts(pairs, cls, ignore_index)
    support = (tp + fn).sum(0)

    def mean(x: Any) -> float:
        arr = np.asarray(x, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        return float(arr.mean()) if arr.size else math.nan

    rows = [
        {
            "class": int(c),
            "name": class_names.get(int(c), str(int(c))) if class_names else str(int(c)),
            "dice": float(d[i]),
            "iou": float(j[i]),
            "boundary_iou": float(b[i]),
            "hd95": float(hd[i]),
            "assd": float(sd[i]),
            "support": float(support[i]),
        }
        for i, c in enumerate(cls)
    ]
    values = {
        "miou": mean(j),
        "dice": mean(d),
        "pixel_accuracy": float(pixel_accuracy(y_true, y_pred, ignore_index=ignore_index)),
        "mean_pixel_accuracy": float(
            mean_pixel_accuracy(y_true, y_pred, num_classes=num_classes, ignore_index=ignore_index)
        ),
        "boundary_iou": mean(b),
        "hd95": mean(hd),
        "assd": mean(sd),
    }
    return SegmentationReport(
        values,
        tuple(rows),
        {
            "n_images": len(pairs),
            "aggregate": aggregate,
            "ignore_index": ignore_index,
            "spacing": None if spacing is None else tuple(float(s) for s in spacing),
            "dilation_ratio": dilation_ratio,
            "include_background": include_background,
        },
    )
