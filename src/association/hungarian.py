"""
Hungarian algorithm for optimal assignment of detections to tracks.

Wraps scipy.optimize.linear_sum_assignment and applies a cost threshold
to reject poor matches.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_match(
    cost_matrix: np.ndarray,
    threshold: float = 0.7,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve the linear assignment problem and filter by cost threshold.

    Args:
        cost_matrix: (M, N) float array — rows=tracks, cols=detections.
                     Entry (i, j) is the cost of assigning detection j to track i.
                     For IoU-based matching, cost = 1 - IoU, so threshold = 1 - min_iou.
        threshold:   Maximum cost to accept a match (inclusive).
                     Pairs with cost > threshold are treated as unmatched.
                     Default 0.7 corresponds to min IoU = 0.3.

    Returns:
        matched:            list of (track_idx, det_idx) pairs
        unmatched_tracks:   list of track indices with no accepted match
        unmatched_dets:     list of detection indices with no accepted match
    """
    n_tracks, n_dets = cost_matrix.shape

    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched: List[Tuple[int, int]] = []
    matched_track_set: set[int] = set()
    matched_det_set:   set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= threshold:
            matched.append((int(r), int(c)))
            matched_track_set.add(r)
            matched_det_set.add(c)

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_track_set]
    unmatched_dets   = [j for j in range(n_dets)   if j not in matched_det_set]

    return matched, unmatched_tracks, unmatched_dets


def associate(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    cost_fn,
    threshold: float = 0.7,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Convenience wrapper: build cost matrix with cost_fn, then run Hungarian.

    Args:
        track_boxes: (M, 4) track bboxes [x1, y1, x2, y2]
        det_boxes:   (N, 4) detection bboxes [x1, y1, x2, y2]
        cost_fn:     callable(track_boxes, det_boxes) → (M, N) cost matrix
        threshold:   max cost for a valid match

    Returns:
        matched, unmatched_tracks, unmatched_dets
    """
    if len(track_boxes) == 0:
        return [], [], list(range(len(det_boxes)))
    if len(det_boxes) == 0:
        return [], list(range(len(track_boxes))), []

    cost = cost_fn(track_boxes, det_boxes)
    return hungarian_match(cost, threshold)
