"""
SORT — Simple Online and Realtime Tracking.

Algorithm:
  1. Predict all active tracks via Kalman filter.
  2. Build IoU cost matrix between predicted tracks and new detections.
  3. Solve assignment with the Hungarian algorithm.
  4. matched       → update track with detection
     unmatched det → create new tentative track
     unmatched trk → increment age; delete if too old
  5. Return confirmed tracks (hits >= min_hits).

Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from src.tracking.kalman_filter import KalmanFilter
from src.association.iou_matching import iou_cost_matrix
from src.association.hungarian import associate


class TrackState(Enum):
    Tentative = auto()   # newly created, waiting for min_hits confirmations
    Confirmed = auto()   # actively tracked
    Lost      = auto()   # missed for 1+ frames but within max_age
    Deleted   = auto()   # removed from tracker


_next_id = 1


def _new_id() -> int:
    global _next_id
    tid = _next_id
    _next_id += 1
    return tid


class Track:
    """
    Single object track with Kalman state and lifecycle management.

    Attributes:
        track_id:   Unique integer ID.
        state:      TrackState enum.
        hits:       Total number of updates received.
        age:        Total frames since creation.
        time_since_update: Frames since last detection match.
        score:      Confidence of the latest matched detection.
    """

    def __init__(self, tlwh: np.ndarray, score: float, kf: KalmanFilter) -> None:
        self.track_id = _new_id()
        self.state    = TrackState.Tentative
        self.hits     = 1
        self.age      = 1
        self.time_since_update = 0
        self.score    = score
        self._kf      = kf
        self.mean, self.covariance = kf.initiate(tlwh)

     
    # Lifecycle
     

    def predict(self) -> None:
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, tlwh: np.ndarray, score: float) -> None:
        self.mean, self.covariance = self._kf.update(self.mean, self.covariance, tlwh)
        self.hits += 1
        self.time_since_update = 0
        self.score = score

    def mark_missed(self) -> None:
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > 0:
            self.state = TrackState.Lost

    # Coordinate helpers

    @property
    def tlbr(self) -> np.ndarray:
        return KalmanFilter.to_tlbr(self.mean)

    @property
    def tlwh(self) -> np.ndarray:
        return KalmanFilter.to_tlwh(self.mean)

    def __repr__(self) -> str:
        return (f"Track(id={self.track_id}, state={self.state.name}, "
                f"hits={self.hits}, tsu={self.time_since_update})")


class SORTTracker:
    """
    SORT multi-object tracker.

    Args:
        max_age:       Maximum frames a track can persist without a detection match.
        min_hits:      Minimum detections before a track is confirmed.
        iou_threshold: Minimum IoU to accept a match (cost threshold = 1 - iou_threshold).
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self._cost_thresh  = 1.0 - iou_threshold
        self._kf           = KalmanFilter()
        self.tracks: List[Track] = []
        self.frame_count = 0

    # Main update step

    def update(self, detections: np.ndarray) -> List[Track]:
        """
        Process one frame of detections and update all tracks.

        Args:
            detections: (N, 5) or (N, 6) array — [x1, y1, x2, y2, score, ...]
                        or (N, 4) — [x1, y1, x2, y2] (score defaults to 1.0)

        Returns:
            List of confirmed Track objects with current state.
        """
        self.frame_count += 1
        det_boxes, det_scores = self._parse_detections(detections)

        # 1. Predict all existing tracks
        for t in self.tracks:
            t.predict()

        # 2. Associate predictions with detections
        matched, unmatched_tracks, unmatched_dets = self._associate(det_boxes)

        # 3a. Update matched tracks
        for t_idx, d_idx in matched:
            tlwh = self._tlbr_to_tlwh(det_boxes[d_idx])
            self.tracks[t_idx].update(tlwh, det_scores[d_idx])
            self._update_state(self.tracks[t_idx])

        # 3b. Mark unmatched tracks as missed
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # 3c. Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            tlwh = self._tlbr_to_tlwh(det_boxes[d_idx])
            self.tracks.append(Track(tlwh, det_scores[d_idx], self._kf))

        # 4. Remove deleted tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Deleted]

        # 5. Delete tracks that exceeded max_age
        for t in self.tracks:
            if t.time_since_update > self.max_age:
                t.state = TrackState.Deleted
        self.tracks = [t for t in self.tracks if t.state != TrackState.Deleted]

        return self.confirmed_tracks

    # Properties

    @property
    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.state == TrackState.Confirmed]

    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.state != TrackState.Deleted]

    def reset(self) -> None:
        global _next_id
        _next_id = 1
        self.tracks = []
        self.frame_count = 0

    # Internal helpers

    def _associate(
        self, det_boxes: np.ndarray
    ) -> Tuple[list, list, list]:
        """Run IoU + Hungarian association between active tracks and detections."""
        active = self.active_tracks
        if not active:
            return [], [], list(range(len(det_boxes)))

        track_boxes = np.array([t.tlbr for t in active], dtype=np.float32)
        matched_raw, unmatched_t_raw, unmatched_d = associate(
            track_boxes, det_boxes, iou_cost_matrix, self._cost_thresh
        )

        # Map back from active_tracks indices to self.tracks indices
        active_to_global = {i: self.tracks.index(active[i]) for i in range(len(active))}
        matched          = [(active_to_global[r], c) for r, c in matched_raw]
        unmatched_tracks = [active_to_global[i] for i in unmatched_t_raw]

        return matched, unmatched_tracks, unmatched_d

    def _update_state(self, track: Track) -> None:
        """Promote tentative → confirmed if enough hits."""
        if track.state == TrackState.Tentative and track.hits >= self.min_hits:
            track.state = TrackState.Confirmed
        elif track.state == TrackState.Lost:
            track.state = TrackState.Confirmed

    @staticmethod
    def _parse_detections(
        detections: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split detections into boxes (N,4) and scores (N,)."""
        if detections is None or len(detections) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
        dets = np.asarray(detections, dtype=np.float32)
        boxes  = dets[:, :4]
        scores = dets[:, 4] if dets.shape[1] > 4 else np.ones(len(dets), dtype=np.float32)
        return boxes, scores

    @staticmethod
    def _tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = tlbr
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
