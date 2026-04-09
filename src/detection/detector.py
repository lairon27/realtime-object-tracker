"""
YOLOv8 detection wrapper.

Input:  BGR numpy array (H, W, 3)
Output: numpy array of shape (N, 6) — [x1, y1, x2, y2, confidence, class_id]

Only detections matching `target_classes` are returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np


class YOLOv8Detector:
    """
    Thin wrapper around ultralytics YOLOv8.

    Args:
        model_path:        Path to .pt weights, or a model name like 'yolov8n.pt'
                           (auto-downloaded by ultralytics on first run).
        confidence:        Minimum detection confidence to keep.
        iou_threshold:     NMS IoU threshold.
        target_classes:    COCO class ids to keep. None = keep all.
                           COCO class 0 = person.
        device:            'cpu', 'cuda', 'mps', or '' (auto).
    """

    PERSON_CLASS_ID = 0  # COCO

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO  # lazy import — heavy

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes if target_classes is not None else [self.PERSON_CLASS_ID]
        self.device = device

    # Public API

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run inference on a single BGR frame.

        Returns:
            detections: np.ndarray of shape (N, 6)
                        columns: [x1, y1, x2, y2, confidence, class_id]
                        Empty array of shape (0, 6) when nothing is detected.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def detect_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run inference on a list of frames.

        Returns list of (N_i, 6) arrays, one per frame.
        """
        results = self.model.predict(
            source=frames,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            verbose=False,
        )
        return [self._parse_single(r) for r in results]

    # Internal helpers

    @staticmethod
    def _parse_results(results) -> np.ndarray:
        """Parse ultralytics Results list (single frame) → (N, 6)."""
        return YOLOv8Detector._parse_single(results[0])

    @staticmethod
    def _parse_single(result) -> np.ndarray:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy()       # (N, 4)  x1y1x2y2
        conf = boxes.conf.cpu().numpy()        # (N,)
        cls  = boxes.cls.cpu().numpy()         # (N,)

        detections = np.column_stack([xyxy, conf, cls]).astype(np.float32)
        return detections


def load_detector_from_config(cfg: dict) -> YOLOv8Detector:
    """Convenience factory: build detector from a config dict."""
    det_cfg = cfg.get("detector", {})
    return YOLOv8Detector(
        model_path=det_cfg.get("model", "yolov8n.pt"),
        confidence=det_cfg.get("confidence_threshold", 0.25),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        target_classes=[det_cfg.get("target_class", 0)],
        device=det_cfg.get("device", "cpu"),
    )
