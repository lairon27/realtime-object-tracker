"""
Tests for KalmanFilter.

Scenarios:
1. Round-trip: initiate → to_tlwh must recover the original bbox.
2. Predict-only drift: without updates, prediction should diverge slowly
   (constant-velocity model with zero initial velocity → stays put).
3. Noisy observations: filter must converge and reduce noise vs raw measurements.
4. Coordinate conversions: to_tlwh / to_tlbr consistency.
5. Gating distance: distance to own projection must be near zero.
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracking.kalman_filter import KalmanFilter


# Helpers

def make_tlwh(x=100., y=200., w=60., h=150.):
    return np.array([x, y, w, h], dtype=np.float64)


# Tests

class TestKalmanInitiate:
    def test_mean_shape(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(make_tlwh())
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)

    def test_mean_position_correct(self):
        kf = KalmanFilter()
        x, y, w, h = 100., 200., 60., 150.
        mean, _ = kf.initiate(make_tlwh(x, y, w, h))
        cx_expected = x + w / 2
        cy_expected = y + h / 2
        ar_expected = w / h
        assert abs(mean[0] - cx_expected) < 1e-9
        assert abs(mean[1] - cy_expected) < 1e-9
        assert abs(mean[2] - ar_expected) < 1e-9
        assert abs(mean[3] - h) < 1e-9

    def test_initial_velocity_zero(self):
        kf = KalmanFilter()
        mean, _ = kf.initiate(make_tlwh())
        assert np.allclose(mean[4:], 0.0)

    def test_covariance_positive_definite(self):
        kf = KalmanFilter()
        _, cov = kf.initiate(make_tlwh())
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)


class TestKalmanRoundTrip:
    def test_to_tlwh_recovers_bbox(self):
        """initiate then to_tlwh should return original bbox."""
        kf = KalmanFilter()
        tlwh = make_tlwh(100., 200., 60., 150.)
        mean, _ = kf.initiate(tlwh)
        recovered = kf.to_tlwh(mean)
        assert np.allclose(recovered, tlwh, atol=1e-6)

    def test_to_tlbr_consistent_with_tlwh(self):
        kf = KalmanFilter()
        tlwh = make_tlwh(100., 200., 60., 150.)
        mean, _ = kf.initiate(tlwh)
        tlbr = kf.to_tlbr(mean)
        expected = np.array([100., 200., 160., 350.], dtype=np.float32)
        assert np.allclose(tlbr, expected, atol=1e-4)


class TestKalmanPredict:
    def test_predict_shape(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(make_tlwh())
        mean_p, cov_p = kf.predict(mean, cov)
        assert mean_p.shape == (8,)
        assert cov_p.shape == (8, 8)

    def test_predict_covariance_grows(self):
        """Uncertainty must increase after prediction (no update)."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(make_tlwh())
        _, cov1 = kf.predict(mean, cov)
        _, cov2 = kf.predict(mean, cov1)
        # Trace (total variance) must grow
        assert np.trace(cov1) > np.trace(cov)
        assert np.trace(cov2) > np.trace(cov1)

    def test_predict_zero_velocity_stays_put(self):
        """With zero initial velocity, predicted position == current position."""
        kf = KalmanFilter()
        tlwh = make_tlwh(100., 200., 60., 150.)
        mean, cov = kf.initiate(tlwh)  # velocities are 0
        mean_p, _ = kf.predict(mean, cov)
        # Position components should be unchanged
        assert np.allclose(mean_p[:4], mean[:4], atol=1e-9)


class TestKalmanUpdate:
    def test_update_shape(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(make_tlwh())
        mean_p, cov_p = kf.predict(mean, cov)
        mean_u, cov_u = kf.update(mean_p, cov_p, make_tlwh())
        assert mean_u.shape == (8,)
        assert cov_u.shape == (8, 8)

    def test_update_reduces_uncertainty(self):
        """Covariance trace must decrease after update."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(make_tlwh())
        mean_p, cov_p = kf.predict(mean, cov)
        _, cov_u = kf.update(mean_p, cov_p, make_tlwh())
        assert np.trace(cov_u) < np.trace(cov_p)

    def test_update_moves_toward_measurement(self):
        """Updated position must be between prediction and measurement."""
        kf = KalmanFilter()
        tlwh_init = make_tlwh(100., 200., 60., 150.)
        mean, cov = kf.initiate(tlwh_init)
        mean_p, cov_p = kf.predict(mean, cov)

        # Provide a measurement shifted far away
        tlwh_meas = make_tlwh(200., 300., 60., 150.)
        mean_u, _ = kf.update(mean_p, cov_p, tlwh_meas)

        cx_pred = mean_p[0]
        cx_meas = 200. + 30.   # cx of tlwh_meas
        cx_upd  = mean_u[0]

        # Updated cx should be between predicted and measured
        assert min(cx_pred, cx_meas) <= cx_upd <= max(cx_pred, cx_meas)


class TestKalmanNoisyTrajectory:
    """
    Synthetic test: linear trajectory + Gaussian noise.
    The filtered estimate must have lower RMSE than the raw noisy observations.
    """

    def test_filter_reduces_position_error(self):
        rng = np.random.default_rng(0)
        kf = KalmanFilter()

        n_steps = 60
        true_cx = np.linspace(100., 400., n_steps)
        true_cy = np.full(n_steps, 250.)
        true_h  = 150.
        true_ar = 0.4

        noise_std = 8.0  # pixels

        # Noisy measurements
        noisy_cx = true_cx + rng.normal(0, noise_std, n_steps)
        noisy_cy = true_cy + rng.normal(0, noise_std, n_steps)

        # First observation
        def make_obs(i):
            w_obs = true_ar * true_h
            x_obs = noisy_cx[i] - w_obs / 2
            y_obs = noisy_cy[i] - true_h / 2
            return np.array([x_obs, y_obs, w_obs, true_h])

        mean, cov = kf.initiate(make_obs(0))

        filtered_cx, filtered_cy = [], []
        for i in range(1, n_steps):
            mean, cov = kf.predict(mean, cov)
            mean, cov = kf.update(mean, cov, make_obs(i))
            filtered_cx.append(mean[0])
            filtered_cy.append(mean[1])

        filtered_cx = np.array(filtered_cx)
        filtered_cy = np.array(filtered_cy)

        rmse_noisy    = np.sqrt(np.mean((noisy_cx[1:] - true_cx[1:]) ** 2 +
                                        (noisy_cy[1:] - true_cy[1:]) ** 2))
        rmse_filtered = np.sqrt(np.mean((filtered_cx - true_cx[1:]) ** 2 +
                                        (filtered_cy - true_cy[1:]) ** 2))

        print(f"\n  RMSE noisy   : {rmse_noisy:.2f} px")
        print(f"  RMSE filtered: {rmse_filtered:.2f} px")
        assert rmse_filtered < rmse_noisy, (
            f"Filter should reduce error: filtered={rmse_filtered:.2f} >= noisy={rmse_noisy:.2f}"
        )


class TestGatingDistance:
    def test_self_distance_near_zero(self):
        """Distance from a state to its own projected measurement should be ~0."""
        kf = KalmanFilter()
        tlwh = make_tlwh(100., 200., 60., 150.)
        mean, cov = kf.initiate(tlwh)
        # Project mean into measurement space
        z_mean, _ = kf.project(mean, cov)
        # Reshape to (1, 4)
        dist = kf.gating_distance(mean, cov, z_mean.reshape(1, 4))
        assert dist[0] < 1e-6

    def test_far_measurement_large_distance(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(make_tlwh(100., 200., 60., 150.))
        far_meas = np.array([[1000., 1000., 0.4, 150.]])
        dist = kf.gating_distance(mean, cov, far_meas)
        assert dist[0] > 1.0
