"""
Kalman Filter for bounding box state estimation.

State vector (8-dim):
    [cx, cy, ar, h,  vx, vy, va, vh]
     0   1   2   3   4   5   6   7

    cx, cy  — bbox center (pixels)
    ar      — aspect ratio w/h  (dimensionless)
    h       — bbox height (pixels)
    vx, vy  — velocity of center (pixels/frame)
    va, vh  — velocity of ar and h (per frame)

Measurement vector (4-dim):
    [cx, cy, ar, h]

Motion model: constant velocity (linear).
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """
    Standard discrete-time Kalman Filter with constant-velocity motion model.

    Usage:
        kf = KalmanFilter()
        mean, cov = kf.initiate(tlwh)          # first observation
        mean, cov = kf.predict(mean, cov)       # between frames
        mean, cov = kf.update(mean, cov, tlwh)  # when detection arrives
        tlwh = kf.to_tlwh(mean)                 # recover bbox
    """

    def __init__(
        self,
        process_noise_scale: float = 1.0,
        measurement_noise_scale: float = 1.0,
    ) -> None:
        ndim = 4  # measurement dimension
        dt = 1.0  # time step (frames)

        # State transition matrix F  (8×8)
        # x_k+1 = F @ x_k
        self._F = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._F[i, ndim + i] = dt  # pos += vel * dt

        # Measurement matrix H  (4×8)
        # z_k = H @ x_k
        self._H = np.eye(ndim, 2 * ndim, dtype=np.float64)

        # Process noise covariance Q — how much we trust the motion model
        # Scaled per dimension: position dims get small noise, velocity dims get larger
        self._std_process = process_noise_scale * np.array(
            [1e-2, 1e-2, 1e-5, 1e-2,   # position: cx, cy, ar, h
             1e-2, 1e-2, 1e-5, 1e-2],   # velocity
            dtype=np.float64,
        )

        # Measurement noise covariance R — sensor uncertainty
        self._std_meas = measurement_noise_scale * np.array(
            [1e-1, 1e-1, 1e-2, 1e-1],  # cx, cy, ar, h
            dtype=np.float64,
        )
     
    # Public API

    def initiate(self, tlwh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create initial state from a single [x, y, w, h] bounding box.

        Returns:
            mean:       (8,)  initial state vector
            covariance: (8,8) initial covariance matrix
        """
        cx, cy, ar, h = self._tlwh_to_xyah(tlwh)
        mean = np.array([cx, cy, ar, h, 0., 0., 0., 0.], dtype=np.float64)

        # Initial position uncertainty: proportional to bbox size
        std_pos = [
            2 * self._std_meas[0] * h,  # cx
            2 * self._std_meas[1] * h,  # cy
            1e-2,                        # ar
            2 * self._std_meas[3] * h,  # h
        ]
        # Initial velocity uncertainty: larger (we have no velocity info yet)
        std_vel = [
            10 * self._std_meas[0] * h,
            10 * self._std_meas[1] * h,
            1e-5,
            10 * self._std_meas[3] * h,
        ]
        cov_diag = np.array(std_pos + std_vel, dtype=np.float64) ** 2
        covariance = np.diag(cov_diag)
        return mean, covariance

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate state forward by one time step.

        x_k|k-1 = F @ x_k-1|k-1
        P_k|k-1 = F @ P_k-1|k-1 @ F^T + Q
        """
        h = mean[3]
        std = self._std_process.copy()
        # Scale position/velocity noise by current height estimate
        std[[0, 1, 3]] *= h   # position noise
        std[[4, 5, 7]] *= h   # velocity noise

        Q = np.diag(std ** 2)
        mean_pred = self._F @ mean
        cov_pred  = self._F @ covariance @ self._F.T + Q
        return mean_pred, cov_pred

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        tlwh: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Correct predicted state with a new measurement.

        x_k|k = x_k|k-1 + K @ (z - H @ x_k|k-1)
        P_k|k = (I - K @ H) @ P_k|k-1

        Args:
            mean, covariance: output of predict()
            tlwh: new detection in [x, y, w, h] format

        Returns:
            Updated mean, covariance.
        """
        z = np.array(self._tlwh_to_xyah(tlwh), dtype=np.float64)

        h = mean[3]
        std_meas = self._std_meas.copy()
        std_meas[[0, 1, 3]] *= h  # scale by height

        R = np.diag(std_meas ** 2)
        S = self._H @ covariance @ self._H.T + R        # innovation covariance
        K = covariance @ self._H.T @ np.linalg.inv(S)  # Kalman gain

        innovation = z - self._H @ mean
        mean_upd = mean + K @ innovation
        cov_upd  = (np.eye(len(mean)) - K @ self._H) @ covariance
        return mean_upd, cov_upd

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution into measurement space.

        Returns (projected_mean [4], projected_cov [4×4]).
        Useful for computing Mahalanobis distance during matching.
        """
        h = mean[3]
        std_meas = self._std_meas.copy()
        std_meas[[0, 1, 3]] *= h
        R = np.diag(std_meas ** 2)
        z_mean = self._H @ mean
        z_cov  = self._H @ covariance @ self._H.T + R
        return z_mean, z_cov

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
    ) -> np.ndarray:
        """
        Compute squared Mahalanobis distance from predicted state to each measurement.

        Args:
            measurements: (N, 4) array of [cx, cy, ar, h]

        Returns:
            distances: (N,) array of squared Mahalanobis distances
        """
        z_mean, z_cov = self.project(mean, covariance)
        diff = measurements - z_mean  # (N, 4)
        z_cov_inv = np.linalg.inv(z_cov)
        # d² = diff @ Σ⁻¹ @ diff^T  (row-wise)
        distances = np.sum(diff @ z_cov_inv * diff, axis=1)
        return distances

    # Coordinate conversions

    @staticmethod
    def to_tlwh(mean: np.ndarray) -> np.ndarray:
        """Convert state mean → [x, y, w, h]."""
        cx, cy, ar, h = mean[0], mean[1], mean[2], mean[3]
        w = ar * h
        x = cx - w / 2
        y = cy - h / 2
        return np.array([x, y, w, h], dtype=np.float32)

    @staticmethod
    def to_tlbr(mean: np.ndarray) -> np.ndarray:
        """Convert state mean → [x1, y1, x2, y2]."""
        tlwh = KalmanFilter.to_tlwh(mean)
        return np.array(
            [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]],
            dtype=np.float32,
        )

    # Internal

    @staticmethod
    def _tlwh_to_xyah(tlwh: np.ndarray) -> tuple[float, float, float, float]:
        """[x, y, w, h] → (cx, cy, aspect_ratio, height)."""
        x, y, w, h = float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3])
        h = max(h, 1.0)  # avoid division by zero
        cx = x + w / 2
        cy = y + h / 2
        ar = w / h
        return cx, cy, ar, h
