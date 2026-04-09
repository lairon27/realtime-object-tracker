"""Tests for the Hungarian assignment wrapper."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.association.hungarian import hungarian_match, associate
from src.association.iou_matching import iou_cost_matrix


class TestHungarianMatch:
    def test_perfect_match_2x2(self):
        # Identity cost matrix → each track matches its detection
        cost = np.array([[0.0, 1.0],
                         [1.0, 0.0]], dtype=np.float32)
        matched, unmatched_t, unmatched_d = hungarian_match(cost, threshold=0.7)
        assert set(matched) == {(0, 0), (1, 1)}
        assert unmatched_t == []
        assert unmatched_d == []

    def test_threshold_rejects_bad_match(self):
        # Both pairs have cost > threshold
        cost = np.array([[0.9, 0.95],
                         [0.92, 0.91]], dtype=np.float32)
        matched, unmatched_t, unmatched_d = hungarian_match(cost, threshold=0.7)
        assert matched == []
        assert sorted(unmatched_t) == [0, 1]
        assert sorted(unmatched_d) == [0, 1]

    def test_partial_match(self):
        # Track 0 matches det 0 (cost=0.2), track 1 has no good match
        cost = np.array([[0.2, 0.9],
                         [0.8, 0.85]], dtype=np.float32)
        matched, unmatched_t, unmatched_d = hungarian_match(cost, threshold=0.7)
        assert (0, 0) in matched
        assert 1 in unmatched_t

    def test_more_dets_than_tracks(self):
        # 1 track, 3 detections → 1 matched, 2 unmatched dets
        cost = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
        matched, unmatched_t, unmatched_d = hungarian_match(cost, threshold=0.7)
        assert len(matched) == 1
        assert matched[0] == (0, 0)
        assert unmatched_t == []
        assert sorted(unmatched_d) == [1, 2]

    def test_more_tracks_than_dets(self):
        # 3 tracks, 1 detection
        cost = np.array([[0.1],
                         [0.5],
                         [0.9]], dtype=np.float32)
        matched, unmatched_t, unmatched_d = hungarian_match(cost, threshold=0.7)
        assert len(matched) == 1
        assert matched[0] == (0, 0)
        assert sorted(unmatched_t) == [1, 2]
        assert unmatched_d == []

    def test_empty_cost_matrix_no_tracks(self):
        cost = np.empty((0, 3), dtype=np.float32)
        matched, unmatched_t, unmatched_d = hungarian_match(cost, threshold=0.7)
        assert matched == []
        assert unmatched_t == []
        assert sorted(unmatched_d) == [0, 1, 2]

    def test_empty_cost_matrix_no_dets(self):
        cost = np.empty((2, 0), dtype=np.float32)
        matched, unmatched_t, unmatched_d = hungarian_match(cost, threshold=0.7)
        assert matched == []
        assert sorted(unmatched_t) == [0, 1]
        assert unmatched_d == []

    def test_optimal_assignment(self):
        # Cost favours (0,1) and (1,0) over diagonal
        cost = np.array([[0.8, 0.1],
                         [0.2, 0.9]], dtype=np.float32)
        matched, _, _ = hungarian_match(cost, threshold=0.7)
        # Hungarian should pick (0,1) and (1,0) — total cost 0.3 vs 1.7
        assert set(matched) == {(0, 1), (1, 0)}

    def test_all_indices_accounted_for(self):
        """Every index must appear exactly once across matched + unmatched."""
        cost = np.random.rand(4, 5).astype(np.float32)
        matched, unmatched_t, unmatched_d = hungarian_match(cost, threshold=0.5)

        all_tracks = [r for r, _ in matched] + unmatched_t
        all_dets   = [c for _, c in matched] + unmatched_d
        assert sorted(all_tracks) == list(range(4))
        assert sorted(all_dets)   == list(range(5))


class TestAssociateWithIoU:
    def _make_boxes(self, coords):
        return np.array(coords, dtype=np.float32)

    def test_identical_boxes_matched(self):
        boxes = self._make_boxes([[0, 0, 10, 10], [20, 20, 30, 30]])
        matched, unmatched_t, unmatched_d = associate(
            boxes, boxes, iou_cost_matrix, threshold=0.7
        )
        assert len(matched) == 2
        assert unmatched_t == []
        assert unmatched_d == []

    def test_no_overlap_all_unmatched(self):
        tracks = self._make_boxes([[0, 0, 5, 5]])
        dets   = self._make_boxes([[100, 100, 200, 200]])
        matched, unmatched_t, unmatched_d = associate(
            tracks, dets, iou_cost_matrix, threshold=0.7
        )
        assert matched == []
        assert unmatched_t == [0]
        assert unmatched_d == [0]

    def test_empty_tracks(self):
        tracks = self._make_boxes([]).reshape(0, 4)
        dets   = self._make_boxes([[0, 0, 10, 10]])
        matched, unmatched_t, unmatched_d = associate(
            tracks, dets, iou_cost_matrix, threshold=0.7
        )
        assert matched == []
        assert unmatched_t == []
        assert unmatched_d == [0]

    def test_empty_dets(self):
        tracks = self._make_boxes([[0, 0, 10, 10]])
        dets   = self._make_boxes([]).reshape(0, 4)
        matched, unmatched_t, unmatched_d = associate(
            tracks, dets, iou_cost_matrix, threshold=0.7
        )
        assert matched == []
        assert unmatched_t == [0]
        assert unmatched_d == []
