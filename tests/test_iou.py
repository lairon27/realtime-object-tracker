"""Tests for IoU computation and cost matrix."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.association.iou_matching import iou, iou_batch, iou_cost_matrix


class TestIoU:
    def test_identical_boxes(self):
        box = np.array([0., 0., 10., 10.])
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.array([0., 0., 5., 5.])
        b = np.array([10., 10., 20., 20.])
        assert iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # a = [0,0,10,10]  area=100
        # b = [5,5,15,15]  area=100
        # inter = 5x5 = 25, union = 175
        a = np.array([0., 0., 10., 10.])
        b = np.array([5., 5., 15., 15.])
        expected = 25.0 / 175.0
        assert iou(a, b) == pytest.approx(expected, abs=1e-6)

    def test_containment(self):
        outer = np.array([0., 0., 10., 10.])
        inner = np.array([2., 2.,  8.,  8.])
        # inter = 6*6=36, union = 100
        assert iou(outer, inner) == pytest.approx(36.0 / 100.0, abs=1e-6)

    def test_symmetry(self):
        a = np.array([0., 0., 6., 8.])
        b = np.array([3., 2., 9., 10.])
        assert iou(a, b) == pytest.approx(iou(b, a))

    def test_touching_edges(self):
        # boxes share an edge but no area overlap
        a = np.array([0., 0., 5., 5.])
        b = np.array([5., 0., 10., 5.])
        assert iou(a, b) == pytest.approx(0.0)


class TestIoUBatch:
    def test_shape(self):
        a = np.random.rand(4, 4)
        b = np.random.rand(7, 4)
        result = iou_batch(a, b)
        assert result.shape == (4, 7)

    def test_empty_a(self):
        a = np.empty((0, 4), dtype=np.float32)
        b = np.array([[0., 0., 5., 5.]])
        result = iou_batch(a, b)
        assert result.shape == (0, 1)

    def test_empty_b(self):
        a = np.array([[0., 0., 5., 5.]])
        b = np.empty((0, 4), dtype=np.float32)
        result = iou_batch(a, b)
        assert result.shape == (1, 0)

    def test_diagonal_is_one_for_identical(self):
        boxes = np.array([[0., 0., 10., 10.],
                          [5., 5., 15., 15.],
                          [20., 20., 30., 30.]], dtype=np.float32)
        mat = iou_batch(boxes, boxes)
        assert np.allclose(np.diag(mat), 1.0, atol=1e-5)

    def test_values_in_range(self):
        a = np.random.rand(5, 4)
        a[:, 2:] += a[:, :2] + 0.1   # ensure x2>x1, y2>y1
        b = np.random.rand(8, 4)
        b[:, 2:] += b[:, :2] + 0.1
        mat = iou_batch(a, b)
        assert np.all(mat >= 0) and np.all(mat <= 1)

    def test_consistent_with_scalar_iou(self):
        a = np.array([[0., 0., 10., 10.]], dtype=np.float32)
        b = np.array([[5., 5., 15., 15.]], dtype=np.float32)
        mat = iou_batch(a, b)
        assert mat[0, 0] == pytest.approx(iou(a[0], b[0]), abs=1e-5)


class TestIoUCostMatrix:
    def test_cost_is_one_minus_iou(self):
        a = np.array([[0., 0., 10., 10.]], dtype=np.float32)
        b = np.array([[0., 0., 10., 10.]], dtype=np.float32)
        cost = iou_cost_matrix(a, b)
        assert cost[0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_no_overlap_cost_is_one(self):
        a = np.array([[0., 0., 5., 5.]], dtype=np.float32)
        b = np.array([[10., 10., 20., 20.]], dtype=np.float32)
        cost = iou_cost_matrix(a, b)
        assert cost[0, 0] == pytest.approx(1.0, abs=1e-5)
