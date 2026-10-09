"""Computer vision evaluation: semantic segmentation and object detection."""

from .detection import (
    DetectionReport,
    average_precision_detection,
    box_iou,
    detection_pr_curve,
    detection_report,
    from_coco,
    mean_average_precision,
)
from .segmentation import (
    SegmentationReport,
    average_surface_distance,
    boundary_iou,
    dice,
    hausdorff_distance,
    iou,
    mean_pixel_accuracy,
    miou,
    per_image_scores,
    pixel_accuracy,
    segmentation_confusion,
    segmentation_report,
)

__all__ = [
    "SegmentationReport",
    "segmentation_report",
    "DetectionReport",
    "average_precision_detection",
    "average_surface_distance",
    "box_iou",
    "boundary_iou",
    "detection_pr_curve",
    "detection_report",
    "dice",
    "from_coco",
    "hausdorff_distance",
    "iou",
    "mean_average_precision",
    "mean_pixel_accuracy",
    "miou",
    "per_image_scores",
    "pixel_accuracy",
    "segmentation_confusion",
]
