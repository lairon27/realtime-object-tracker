"""
DeepSORT — Deep Simple Online and Realtime Tracking.

Improvements over SORT:
  1. Appearance embedding (Re-ID) per detection → handles re-identification
     after occlusions better than IoU alone.
  2. Cascade matching ordered by track age — recently seen tracks get
     priority, reducing identity switches for occluded objects.
  3. Combined cost: lambda * IoU_cost + (1-lambda) * cosine_distance,
     gated by Mahalanobis distance to reject impossible matches early.

Reference: Wojke et al., "Simple Online and Realtime Tracking with a Deep
           Association Metric", ICASSP 2018.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from src.tracking.kalman_filter import KalmanFilter
from src.tracking.sort import Track, TrackState, _new_id
from src.association.iou_matching import iou_cost_matrix
from src.association.appearance import (
    AppearanceExtractor,
    appearance_cost_matrix,
    cosine_distance,
)
from src.association.hungarian import hungarian_match

# Chi-squared 95% confidence gate for 4 DoF (Mahalanobis)
CHI2_95_4DOF = 9.4877


class DeepTrack(Track):
    """Track extended with an appearance embedding gallery (ring buffer)."""

    def __init__(
        self,
        tlwh: np.ndarray,
        score: float,
        embedding: np.ndarray,
        kf: KalmanFilter,
        nn_budget: int = 100,
    ) -> None:
        super().__init__(tlwh, score, kf)
        self.nn_budget: int = nn_budget
        self.gallery: Deque[np.ndarray] = deque(maxlen=nn_budget)
        self.gallery.append(embedding)

    def update_appearance(self, embedding: np.ndarray) -> None:
        self.gallery.append(embedding)

    @property
    def gallery_array(self) -> np.ndarray:
        """Return gallery as (K, D) array."""
        return np.array(self.gallery, dtype=np.float32)


class DeepSORTTracker:
    """
    DeepSORT multi-object tracker.

    Args:
        extractor:            AppearanceExtractor instance.
        max_cosine_distance:  Maximum cosine distance to accept appearance match.
        nn_budget:            Max appearance embeddings stored per track.
        iou_threshold:        Min IoU to accept IoU-only fallback match.
        lambda_iou:           Weight of IoU in combined cost (0 = appearance only,
                              1 = IoU only).
        max_age:              Frames to keep a lost track.
        min_hits:             Min hits to confirm a track.
    """

    def __init__(
        self,
        extractor: AppearanceExtractor,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 100,
        iou_threshold: float = 0.7,
        lambda_iou: float = 0.5,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        self.extractor           = extractor
        self.max_cos_dist        = max_cosine_distance
        self.nn_budget           = nn_budget
        self.iou_cost_thresh     = 1.0 - iou_threshold
        self.lambda_iou          = lambda_iou
        self.max_age             = max_age
        self.min_hits            = min_hits

        self._kf     = KalmanFilter()
        self.tracks: List[DeepTrack] = []
        self.frame_count = 0

    # Main update

    def update(self, frame: np.ndarray, detections: np.ndarray) -> List[DeepTrack]:
        """
        Process one frame.

        Args:
            frame:      BGR image — needed for appearance extraction.
            detections: (N, 5+) — [x1, y1, x2, y2, score, ...].

        Returns:
            Confirmed DeepTrack objects.
        """
        self.frame_count += 1
        det_boxes, det_scores = self._parse(detections)

        # Extract appearance embeddings for all detections
        embeddings = self.extractor.extract(frame, det_boxes)  # (N, D)

        # Predict all tracks
        for t in self.tracks:
            t.predict()

        # Cascade matching: confirmed tracks, ordered by time_since_update
        confirmed  = [t for t in self.tracks if t.state == TrackState.Confirmed]
        unconfirmed = [t for t in self.tracks
                       if t.state in (TrackState.Tentative, TrackState.Lost)]

        matched_c, unmatched_tc, unmatched_d = self._cascade_match(
            confirmed, det_boxes, embeddings
        )
         
        # IoU-only fallback: unmatched confirmed + all unconfirmed tracks
        iou_candidates = [confirmed[i] for i in unmatched_tc] + unconfirmed
        remaining_dets = list(unmatched_d)

        matched_iou, unmatched_ti, final_unmatched_d = self._iou_match(
            iou_candidates, det_boxes, remaining_dets
        )

        # Apply updates
        for t_idx, d_idx in matched_c:
            confirmed[t_idx].update(self._tlbr_to_tlwh(det_boxes[d_idx]), det_scores[d_idx])
            confirmed[t_idx].update_appearance(embeddings[d_idx])
            self._promote(confirmed[t_idx])

        for t_idx, d_idx in matched_iou:
            iou_candidates[t_idx].update(self._tlbr_to_tlwh(det_boxes[d_idx]), det_scores[d_idx])
            iou_candidates[t_idx].update_appearance(embeddings[d_idx])
            self._promote(iou_candidates[t_idx])

        # Mark missed
        for i in unmatched_ti:
            iou_candidates[i].mark_missed()

        # New tracks for fully unmatched detections
        for d_idx in final_unmatched_d:
            tlwh = self._tlbr_to_tlwh(det_boxes[d_idx])
            self.tracks.append(
                DeepTrack(tlwh, det_scores[d_idx], embeddings[d_idx], self._kf, self.nn_budget)
            )

        # Cleanup
        self.tracks = [
            t for t in self.tracks
            if t.state != TrackState.Deleted and t.time_since_update <= self.max_age
        ]
        return self.confirmed_tracks

    # Properties

    @property
    def confirmed_tracks(self) -> List[DeepTrack]:
        return [t for t in self.tracks if t.state == TrackState.Confirmed]

    @property
    def active_tracks(self) -> List[DeepTrack]:
        return [t for t in self.tracks if t.state != TrackState.Deleted]

    def reset(self) -> None:
        self.tracks = []
        self.frame_count = 0

    # Matching helpers

    def _cascade_match(
        self,
        confirmed: List[DeepTrack],
        det_boxes: np.ndarray,
        embeddings: np.ndarray,
    ) -> Tuple[list, list, list]:
        """
        Cascade matching: iterate over age groups (1, 2, ..., max_age).
        Tracks seen more recently get priority.
        """
        if not confirmed or len(det_boxes) == 0:
            return [], list(range(len(confirmed))), list(range(len(det_boxes)))

        unmatched_dets = list(range(len(det_boxes)))
        all_matched: list = []
        matched_track_set: set = set()

        for age in range(1, self.max_age + 1):
            age_tracks = [(i, t) for i, t in enumerate(confirmed)
                          if t.time_since_update == age and i not in matched_track_set]
            if not age_tracks or not unmatched_dets:
                break

            t_indices, tracks_age = zip(*age_tracks)
            t_indices = list(t_indices)
            tracks_age = list(tracks_age)

            det_sub = det_boxes[unmatched_dets]
            emb_sub = embeddings[unmatched_dets]

            cost = self._combined_cost(tracks_age, det_sub, emb_sub)

            # Gate by Mahalanobis distance
            cost = self._gate_cost(cost, tracks_age, det_sub)

            matched_sub, unmatched_t_sub, unmatched_d_sub = hungarian_match(
                cost, threshold=self.max_cos_dist
            )

            for local_ti, local_di in matched_sub:
                global_ti = t_indices[local_ti]
                global_di = unmatched_dets[local_di]
                all_matched.append((global_ti, global_di))
                matched_track_set.add(global_ti)

            unmatched_dets = [unmatched_dets[j] for j in unmatched_d_sub]

        unmatched_tracks = [i for i in range(len(confirmed)) if i not in matched_track_set]
        return all_matched, unmatched_tracks, unmatched_dets

    def _iou_match(
        self,
        candidates: List[DeepTrack],
        det_boxes: np.ndarray,
        det_indices: List[int],
    ) -> Tuple[list, list, list]:
        if not candidates or not det_indices:
            return [], list(range(len(candidates))), det_indices

        track_boxes = np.array([t.tlbr for t in candidates], dtype=np.float32)
        det_sub     = det_boxes[det_indices]
        cost        = iou_cost_matrix(track_boxes, det_sub)

        matched_sub, unmatched_t, unmatched_d_sub = hungarian_match(
            cost, threshold=self.iou_cost_thresh
        )

        matched = [(ti, det_indices[di]) for ti, di in matched_sub]
        unmatched_dets = [det_indices[j] for j in unmatched_d_sub]
        return matched, unmatched_t, unmatched_dets

    def _combined_cost(
        self,
        tracks: List[DeepTrack],
        det_boxes: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """lambda * IoU_cost + (1-lambda) * cosine_distance."""
        track_boxes  = np.array([t.tlbr for t in tracks], dtype=np.float32)
        iou_cost     = iou_cost_matrix(track_boxes, det_boxes)

        galleries    = [t.gallery_array for t in tracks]
        appear_cost  = appearance_cost_matrix(galleries, embeddings)

        return self.lambda_iou * iou_cost + (1.0 - self.lambda_iou) * appear_cost

    def _gate_cost(
        self,
        cost: np.ndarray,
        tracks: List[DeepTrack],
        det_boxes: np.ndarray,
    ) -> np.ndarray:
        """Set cost to infinity where Mahalanobis distance exceeds 95% gate."""
        cost = cost.copy()
        # Convert tlbr to xyah for Mahalanobis
        meas = np.array(
            [self._tlbr_to_xyah(b) for b in det_boxes], dtype=np.float64
        )
        for i, track in enumerate(tracks):
            dists = self._kf.gating_distance(track.mean, track.covariance, meas)
            cost[i, dists > CHI2_95_4DOF] = 1e5  # large finite penalty — inf breaks linear_sum_assignment
        return cost

    # Internal

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

    @staticmethod
    def _tlbr_to_xyah(tlbr: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = tlbr
        w = x2 - x1
        h = max(y2 - y1, 1.0)
        return np.array([x1 + w / 2, y1 + h / 2, w / h, h], dtype=np.float64)
