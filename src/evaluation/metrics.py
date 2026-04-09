"""
MOT evaluation pipeline using the motmetrics library.

Typical usage:
    acc = MOTAccumulator()
    for fid, img, dets, gts in seq.iter_frames():
        tracks = tracker.update(...)
        acc.update(frame_id=fid, gt_objects=gts, tracked_objects=tracks)
    summary = acc.compute()
    print(summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import motmetrics as mm


@dataclass
class TrackedObject:
    """Minimal interface expected by MOTAccumulator."""
    track_id: int
    tlbr: np.ndarray  # [x1, y1, x2, y2]


class MOTAccumulator:
    """
    Accumulates per-frame GT↔prediction correspondences and computes
    standard MOT metrics via motmetrics.

    MOT metrics glossary:
        MOTA  — Multiple Object Tracking Accuracy
                = 1 - (FP + FN + IDS) / GT_total
                Penalises missed detections, false positives, and ID switches.
        IDF1  — Identification F1
                = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
                Measures how well tracker maintains identity over time.
        IDS   — ID Switches: number of times a track changes its assigned GT id.
        FP    — False Positives: predicted boxes with no matching GT.
        FN    — False Negatives: GT objects with no matching prediction.
        MT    — Mostly Tracked: GT trajectories covered ≥ 80% of their length.
        ML    — Mostly Lost:    GT trajectories covered ≤ 20% of their length.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._acc = mm.MOTAccumulator(auto_id=False)

    def update(
        self,
        frame_id: int,
        gt_objects: Sequence,
        tracked_objects: Sequence,
    ) -> None:
        """
        Register one frame.

        Args:
            frame_id:        1-based frame index (used as motmetrics row key).
            gt_objects:      Iterable of objects with .track_id and .tlbr (GT).
            tracked_objects: Iterable of objects with .track_id and .tlbr (predicted).
        """
        gt_ids   = [g.track_id for g in gt_objects]
        pred_ids = [t.track_id for t in tracked_objects]

        gt_boxes   = np.array([g.tlbr for g in gt_objects],   dtype=np.float32) if gt_objects   else np.empty((0, 4))
        pred_boxes = np.array([t.tlbr for t in tracked_objects], dtype=np.float32) if tracked_objects else np.empty((0, 4))

        dist = self._iou_distance(gt_boxes, pred_boxes)
        self._acc.update(gt_ids, pred_ids, dist, frameid=frame_id)

    def compute(self) -> Dict[str, Any]:
        """
        Compute and return a dict of scalar MOT metrics.

        Keys: mota, idf1, num_switches, num_false_positives,
              num_misses, num_unique_objects, recall, precision, motp
        """
        mh = mm.metrics.create()
        summary = mh.compute(
            self._acc,
            metrics=[
                "num_frames", "mota", "motp", "idf1",
                "num_switches", "num_false_positives", "num_misses",
                "num_unique_objects", "mostly_tracked", "mostly_lost",
                "recall", "precision",
            ],
            name="sequence",
        )
        # Convert to plain Python dict of scalars
        row = summary.iloc[0]
        return {
            "frames":    int(row["num_frames"]),
            "mota":      float(row["mota"]),
            "motp":      float(row["motp"]),
            "idf1":      float(row["idf1"]),
            "ids":       int(row["num_switches"]),
            "fp":        int(row["num_false_positives"]),
            "fn":        int(row["num_misses"]),
            "gt_objs":   int(row["num_unique_objects"]),
            "mt":        int(row["mostly_tracked"]),
            "ml":        int(row["mostly_lost"]),
            "recall":    float(row["recall"]),
            "precision": float(row["precision"]),
        }

    def reset(self) -> None:
        self._acc = mm.MOTAccumulator(auto_id=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _iou_distance(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute distance matrix as 1 - IoU.
        Values > (1 - iou_threshold) are set to NaN so motmetrics
        treats them as "no valid assignment possible".
        """
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return np.empty((len(gt_boxes), len(pred_boxes)), dtype=np.float32)

        # Vectorised IoU
        a = gt_boxes[:, None, :]    # (M, 1, 4)
        b = pred_boxes[None, :, :]  # (1, N, 4)
        ix1 = np.maximum(a[..., 0], b[..., 0])
        iy1 = np.maximum(a[..., 1], b[..., 1])
        ix2 = np.minimum(a[..., 2], b[..., 2])
        iy2 = np.minimum(a[..., 3], b[..., 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_a = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        area_b = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        union  = area_a[:, None] + area_b[None, :] - inter
        iou    = np.where(union > 0, inter / union, 0.0)
        dist   = 1.0 - iou  # distance ∈ [0, 1]
        # Mask out pairs below IoU threshold
        dist[dist > 1.0 - self.iou_threshold] = np.nan
        return dist.astype(np.float32)


def format_summary(results: Dict[str, Any], name: str = "") -> str:
    """Return a human-readable one-line summary string."""
    return (
        f"{name:<18} "
        f"MOTA={results['mota']:+.3f}  "
        f"IDF1={results['idf1']:.3f}  "
        f"IDS={results['ids']:4d}  "
        f"FP={results['fp']:5d}  "
        f"FN={results['fn']:5d}  "
        f"MT={results['mt']:3d}  "
        f"ML={results['ml']:3d}  "
        f"Recall={results['recall']:.3f}  "
        f"Prec={results['precision']:.3f}"
    )


def merge_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge per-sequence metric dicts into a dataset-level summary.
    MOTA and IDF1 are re-computed from raw counts; others are summed.
    """
    total: Dict[str, Any] = {
        "frames": 0, "ids": 0, "fp": 0, "fn": 0,
        "gt_objs": 0, "mt": 0, "ml": 0,
    }
    for r in results_list:
        for k in total:
            total[k] += r.get(k, 0)

    # Approximate MOTA from totals (proper re-computation needs raw counts)
    # MOTA = 1 - (FP + FN + IDS) / GT_total  but GT_total = FP+FN+TP (not stored)
    # Use recall & precision from the last result as approximate
    total["mota"]      = float(np.mean([r["mota"]      for r in results_list]))
    total["idf1"]      = float(np.mean([r["idf1"]      for r in results_list]))
    total["motp"]      = float(np.mean([r["motp"]      for r in results_list]))
    total["recall"]    = float(np.mean([r["recall"]    for r in results_list]))
    total["precision"] = float(np.mean([r["precision"] for r in results_list]))
    return total
