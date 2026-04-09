"""
IoU-based cost matrix for data association.

All bounding boxes are expected in [x1, y1, x2, y2] (tlbr) format.
"""

from __future__ import annotations

import numpy as np


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection over Union between two bounding boxes.

    Args:
        box_a, box_b: arrays of shape (4,) — [x1, y1, x2, y2]

    Returns:
        IoU in [0, 1].
    """
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU between every pair of boxes from two sets.

    Args:
        boxes_a: (M, 4) — [x1, y1, x2, y2]
        boxes_b: (N, 4) — [x1, y1, x2, y2]

    Returns:
        iou_matrix: (M, N) float32
    """
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    # Broadcast to (M, N, 4)
    a = boxes_a[:, None, :]   # (M, 1, 4)
    b = boxes_b[None, :, :]   # (1, N, 4)

    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)  # (M, N)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # (M,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # (N,)

    union = area_a[:, None] + area_b[None, :] - inter  # (M, N)
    iou_mat = np.where(union > 0, inter / union, 0.0)
    return iou_mat.astype(np.float32)


def iou_cost_matrix(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
) -> np.ndarray:
    """
    Cost matrix for Hungarian assignment: cost = 1 - IoU.

    Args:
        track_boxes: (M, 4) predicted track positions [x1, y1, x2, y2]
        det_boxes:   (N, 4) detected boxes          [x1, y1, x2, y2]

    Returns:
        cost: (M, N) float32, values in [0, 1]
    """
    return 1.0 - iou_batch(track_boxes, det_boxes)
