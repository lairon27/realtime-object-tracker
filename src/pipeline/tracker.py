"""
End-to-end tracking pipeline.

Connects:  video / MOT17 sequence  →  detector  →  tracker  →  annotated output

Usage:
    pipeline = TrackingPipeline.from_config("configs/default.yaml")
    for frame_id, annotated_frame, tracks in pipeline.run_video("video.mp4"):
        cv2.imshow("tracking", annotated_frame)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from src.detection.detector import YOLOv8Detector
from src.tracking.sort import SORTTracker, Track
from src.tracking.bytetrack import ByteTracker
from src.visualization.draw import draw_tracks, draw_detections


class TrackingPipeline:
    """
    End-to-end object tracking pipeline.

    Args:
        detector:  YOLOv8Detector instance.
        tracker:   SORTTracker or ByteTracker instance.
        show_dets: Also draw raw detections (transparent layer).
    """

    def __init__(
        self,
        detector: YOLOv8Detector,
        tracker,
        show_dets: bool = False,
    ) -> None:
        self.detector  = detector
        self.tracker   = tracker
        self.show_dets = show_dets
        self._frame_times: List[float] = []

    # Factory

    @classmethod
    def from_config(cls, config_path: str, tracker_override: Optional[str] = None) -> "TrackingPipeline":
        """Build pipeline from a YAML config file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        det_cfg = cfg.get("detector", {})
        detector = YOLOv8Detector(
            model_path=det_cfg.get("model", "yolov8n.pt"),
            confidence=det_cfg.get("confidence_threshold", 0.25),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            target_classes=[det_cfg.get("target_class", 0)],
            device=det_cfg.get("device", "cpu"),
        )

        algo = tracker_override or cfg.get("tracker", {}).get("algorithm", "bytetrack")
        tracker = cls._build_tracker(algo, cfg)

        return cls(detector=detector, tracker=tracker)

    @staticmethod
    def _build_tracker(name: str, cfg: dict):
        trk = cfg.get("tracker", {})
        max_age  = trk.get("max_age", 30)
        min_hits = trk.get("min_hits", 3)

        if name == "sort":
            return SORTTracker(
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=trk.get("iou_threshold", 0.3),
            )
        if name == "bytetrack":
            bt = cfg.get("bytetrack", {})
            return ByteTracker(
                high_thresh=bt.get("high_thresh", 0.6),
                low_thresh=bt.get("low_thresh", 0.1),
                match_thresh=bt.get("match_thresh", 0.8),
                max_age=max_age,
                min_hits=min_hits,
            )
        raise ValueError(f"Unknown tracker algorithm: {name!r}. Choose 'sort' or 'bytetrack'.")

    # Main interfaces

    def run_video(
        self,
        source: str,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[int, np.ndarray, List[Track]], None, None]:
        """
        Process a video file frame by frame.

        Yields:
            (frame_id, annotated_frame, confirmed_tracks)
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")

        frame_id = 0
        self._frame_times = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
                if max_frames and frame_id > max_frames:
                    break

                annotated, tracks = self._process_frame(frame)
                yield frame_id, annotated, tracks
        finally:
            cap.release()

    def run_mot17(
        self,
        seq_path: str,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[int, np.ndarray, List[Track]], None, None]:
        """
        Process a MOT17 sequence directory frame by frame.

        Yields:
            (frame_id, annotated_frame, confirmed_tracks)
        """
        from src.data_loader import MOT17Sequence
        seq = MOT17Sequence(seq_path)
        self._frame_times = []

        for fid, img, _, _ in seq.iter_frames(end=max_frames):
            annotated, tracks = self._process_frame(img)
            yield fid, annotated, tracks

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Track]]:
        """Process a single BGR frame. Returns (annotated_frame, tracks)."""
        return self._process_frame(frame)

    # Stats

    @property
    def mean_fps(self) -> float:
        if not self._frame_times:
            return 0.0
        return 1.0 / (sum(self._frame_times) / len(self._frame_times))

    # Internal

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Track]]:
        t0 = time.perf_counter()

        dets   = self.detector.detect(frame)
        inp    = dets[:, :5] if len(dets) else np.empty((0, 5), dtype=np.float32)
        tracks = self.tracker.update(inp)

        self._frame_times.append(time.perf_counter() - t0)

        annotated = frame.copy()
        if self.show_dets and len(dets):
            annotated = draw_detections(annotated, dets, color=(180, 180, 180),
                                        show_conf=False, thickness=1)
        annotated = draw_tracks(annotated, tracks, show_id=True)

        # FPS overlay
        if len(self._frame_times) >= 5:
            fps = 1.0 / (sum(self._frame_times[-10:]) / min(10, len(self._frame_times)))
            cv2.putText(annotated, f"FPS: {fps:.1f}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(annotated, f"Tracks: {len(tracks)}", (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        return annotated, tracks
