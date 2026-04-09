"""
ByteTrack — BYTE: Multi-Object Tracking by Associating Every Detection Box.

Key idea over SORT:
  - Every detection box is used, not just high-confidence ones.
  - Two-stage matching per frame:
      Stage 1: high-confidence detections  ↔  confirmed + lost tracks  (IoU)
      Stage 2: low-confidence  detections  ↔  unmatched tracks          (IoU)
  - Low-confidence detections rescue partially occluded tracks that SORT would lose.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
           Every Detection Box", ECCV 2022.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.tracking.kalman_filter import KalmanFilter
from src.tracking.sort import Track, TrackState, _new_id
from src.association.iou_matching import iou_cost_matrix
from src.association.hungarian import hungarian_match


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Args:
        high_thresh:   Confidence threshold to be a "high" detection (stage 1).
        low_thresh:    Minimum confidence to be a "low" detection (stage 2).
                       Detections below low_thresh are discarded entirely.
        match_thresh:  Max IoU cost (1 - IoU) to accept a match.
        max_age:       Frames a lost track is kept before deletion.
        min_hits:      Detections before a track is confirmed.
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float  = 0.1,
        match_thresh: float = 0.8,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        self.high_thresh  = high_thresh
        self.low_thresh   = low_thresh
        self.match_thresh = match_thresh
        self.max_age      = max_age
        self.min_hits     = min_hits

        self._kf: KalmanFilter = KalmanFilter()
        self.tracks: List[Track] = []
        self.frame_count: int = 0
   
    # Main update

    def update(self, detections: np.ndarray) -> List[Track]:
        """
        Process one frame and return confirmed tracks.

        Args:
            detections: (N, 5) or (N, 6) — [x1, y1, x2, y2, score, ...]
                        Rows with score < low_thresh are silently dropped.

        Returns:
            List of confirmed Track objects.
        """
        self.frame_count += 1
        det_boxes, det_scores = self._parse(detections)

        # Split into high / low confidence pools
        high_mask = det_scores >= self.high_thresh
        low_mask  = (det_scores >= self.low_thresh) & ~high_mask

        high_boxes  = det_boxes[high_mask]
        high_scores = det_scores[high_mask]
        low_boxes   = det_boxes[low_mask]
        low_scores  = det_scores[low_mask]

        # 1. Predict all tracks
        for t in self.tracks:
            t.predict()

        # Partition tracks
        confirmed = [t for t in self.tracks if t.state == TrackState.Confirmed]
        lost      = [t for t in self.tracks if t.state == TrackState.Lost]
        tentative = [t for t in self.tracks if t.state == TrackState.Tentative]

        # Stage 1: high-confidence detections ↔ confirmed + lost tracks
        stage1_tracks = confirmed + lost
        matched1, unmatched_t1, unmatched_d_high = self._match(
            stage1_tracks, high_boxes, self.match_thresh
        )

        for t_idx, d_idx in matched1:
            track = stage1_tracks[t_idx]
            tlwh  = self._tlbr_to_tlwh(high_boxes[d_idx])
            track.update(tlwh, high_scores[d_idx])
            self._promote(track)

        # Tracks not matched in stage 1 (from confirmed + lost)
        unmatched_stage1_tracks = [stage1_tracks[i] for i in unmatched_t1]

        # Stage 2: low-confidence detections ↔ unmatched stage-1 tracks
        matched2, unmatched_t2, _ = self._match(
            unmatched_stage1_tracks, low_boxes, self.match_thresh
        )

        for t_idx, d_idx in matched2:
            track = unmatched_stage1_tracks[t_idx]
            tlwh  = self._tlbr_to_tlwh(low_boxes[d_idx])
            track.update(tlwh, low_scores[d_idx])
            self._promote(track)

        # Tracks still unmatched after both stages → mark missed
        for t_idx in unmatched_t2:
            unmatched_stage1_tracks[t_idx].mark_missed()

        # Stage 3: tentative tracks ↔ remaining high-confidence detections
        matched3, unmatched_t3, unmatched_d_new = self._match(
            tentative, high_boxes[unmatched_d_high], self.match_thresh
        )
        # Remap detection indices back to original high_boxes indices
        high_unmatched_idx = np.where(unmatched_d_high)[0] if isinstance(unmatched_d_high, np.ndarray) \
                             else list(unmatched_d_high)

        for t_idx, d_idx in matched3:
            track  = tentative[t_idx]
            # d_idx refers into high_boxes[unmatched_d_high], map back
            orig_d = _remap(unmatched_d_high, d_idx)
            tlwh   = self._tlbr_to_tlwh(high_boxes[orig_d])
            track.update(tlwh, high_scores[orig_d])
            self._promote(track)

        for t_idx in unmatched_t3:
            tentative[t_idx].mark_missed()

        # New tracks for high detections unmatched by all three stages
        # unmatched_d_new is indices into high_boxes[unmatched_d_high]
        for d_idx in unmatched_d_new:
            orig_d = _remap(unmatched_d_high, d_idx)
            tlwh   = self._tlbr_to_tlwh(high_boxes[orig_d])
            self.tracks.append(Track(tlwh, high_scores[orig_d], self._kf))

        # Cleanup: remove deleted / too-old tracks
        self.tracks = [
            t for t in self.tracks
            if t.state != TrackState.Deleted and t.time_since_update <= self.max_age
        ]

        return self.confirmed_tracks

    # Properties

    @property
    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.state == TrackState.Confirmed]

    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.state != TrackState.Deleted]

    def reset(self) -> None:
        self.tracks = []
        self.frame_count = 0

    # Internal helpers

    def _match(
        self,
        tracks: List[Track],
        det_boxes: np.ndarray,
        threshold: float,
    ) -> Tuple[list, list, list]:
        if not tracks or len(det_boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(det_boxes)))

        track_boxes = np.array([t.tlbr for t in tracks], dtype=np.float32)
        cost = iou_cost_matrix(track_boxes, det_boxes)
        return hungarian_match(cost, threshold=threshold)

    def _promote(self, track: Track) -> None:
        if track.state == TrackState.Tentative and track.hits >= self.min_hits:
            track.state = TrackState.Confirmed
        elif track.state == TrackState.Lost:
            track.state = TrackState.Confirmed

    @staticmethod
    def _parse(detections: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if detections is None or len(detections) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
        dets   = np.asarray(detections, dtype=np.float32)
        boxes  = dets[:, :4]
        scores = dets[:, 4] if dets.shape[1] > 4 else np.ones(len(dets), dtype=np.float32)
        return boxes, scores

    @staticmethod
    def _tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = tlbr
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def _remap(unmatched_mask_or_indices, local_idx: int) -> int:
    """Map a local index (into a filtered sub-array) back to original array index."""
    if isinstance(unmatched_mask_or_indices, np.ndarray) and unmatched_mask_or_indices.dtype == bool:
        return int(np.where(unmatched_mask_or_indices)[0][local_idx])
    # It's a list of original indices
    return int(unmatched_mask_or_indices[local_idx])
