import numpy as np


class AttitudeEKF:
    """Extended Kalman Filter for roll and pitch estimation.

    State:
        x = [phi, theta]^T

    Input:
        u = [p, q, r, Va]^T

    Measurement:
        z = [ax, ay, az]^T
    """

    def __init__(
        self,
        dt: float,
        g: float = 9.80665,
        q_diag: np.ndarray | None = None,
        r_diag: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        p0: np.ndarray | None = None,
    ) -> None:
        self.dt = dt
        self.g = g
        self.Q = np.diag(q_diag if q_diag is not None else np.array([2.5e-3, 2.5e-3]))
        self.R = np.diag(
            r_diag if r_diag is not None else np.array([0.35**2, 0.35**2, 0.35**2])
        )
        self.x = np.array(x0 if x0 is not None else [0.0, 0.0], dtype=float).reshape(2, 1)
        self.P = np.array(
            p0 if p0 is not None else np.diag([np.deg2rad(10.0) ** 2, np.deg2rad(10.0) ** 2]),
            dtype=float,
        )

    def state_function(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        phi, theta = self._flatten_state(x)
        p, q, r, _ = self._flatten_input(u)

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        tan_theta = np.tan(theta)

        return np.array(
            [
                [p + q * sin_phi * tan_theta + r * cos_phi * tan_theta],
                [q * cos_phi - r * sin_phi],
            ],
            dtype=float,
        )

    def measurement_function(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        phi, theta = self._flatten_state(x)
        p, q, r, va = self._flatten_input(u)

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        return np.array(
            [
                [q * va * sin_theta + self.g * sin_theta],
                [r * va * cos_theta - p * va * sin_theta - self.g * cos_theta * sin_phi],
                [-q * va * cos_theta - self.g * cos_theta * cos_phi],
            ],
            dtype=float,
        )

    def process_jacobian(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        phi, theta = self._flatten_state(x)
        _, q, r, _ = self._flatten_input(u)

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        tan_theta = np.tan(theta)
        cos_theta = np.cos(theta)
        sec_theta_sq = 1.0 / max(cos_theta * cos_theta, 1.0e-8)

        return np.array(
            [
                [q * cos_phi * tan_theta - r * sin_phi * tan_theta, (q * sin_phi + r * cos_phi) * sec_theta_sq],
                [-q * sin_phi - r * cos_phi, 0.0],
            ],
            dtype=float,
        )

    def measurement_jacobian(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        phi, theta = self._flatten_state(x)
        p, q, r, va = self._flatten_input(u)

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        return np.array(
            [
                [0.0, q * va * cos_theta + self.g * cos_theta],
                [-self.g * cos_theta * cos_phi, -r * va * sin_theta - p * va * cos_theta + self.g * sin_theta * sin_phi],
                [self.g * cos_theta * sin_phi, (q * va + self.g * cos_phi) * sin_theta],
            ],
            dtype=float,
        )

    def predict(self, u: np.ndarray) -> np.ndarray:
        u_col = np.array(u, dtype=float).reshape(4, 1)

        self.x = self.x + self.state_function(self.x, u_col) * self.dt
        self._normalize_state()

        f_jac = self.process_jacobian(self.x, u_col)
        f_discrete = np.eye(2) + f_jac * self.dt
        self.P = f_discrete @ self.P @ f_discrete.T + self.Q * self.dt
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    def update(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        z_col = np.array(z, dtype=float).reshape(3, 1)
        u_col = np.array(u, dtype=float).reshape(4, 1)

        h_jac = self.measurement_jacobian(self.x, u_col)
        innovation = z_col - self.measurement_function(self.x, u_col)
        innovation_cov = h_jac @ self.P @ h_jac.T + self.R
        innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)

        try:
            innovation_cov_inv = np.linalg.inv(innovation_cov)
        except np.linalg.LinAlgError:
            innovation_cov_inv = np.linalg.pinv(innovation_cov)

        kalman_gain = self.P @ h_jac.T @ innovation_cov_inv
        self.x = self.x + kalman_gain @ innovation
        self._normalize_state()

        identity = np.eye(2)
        kh = kalman_gain @ h_jac
        self.P = (identity - kh) @ self.P @ (identity - kh).T + kalman_gain @ self.R @ kalman_gain.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    def _normalize_state(self) -> None:
        self.x[0, 0] = self._wrap_angle(self.x[0, 0])
        self.x[1, 0] = np.clip(self.x[1, 0], -np.deg2rad(89.0), np.deg2rad(89.0))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _flatten_state(x: np.ndarray) -> tuple[float, float]:
        phi = float(np.asarray(x).reshape(-1)[0])
        theta = float(np.asarray(x).reshape(-1)[1])
        return phi, theta

    @staticmethod
    def _flatten_input(u: np.ndarray) -> tuple[float, float, float, float]:
        p, q, r, va = np.asarray(u).reshape(-1)
        return float(p), float(q), float(r), float(va)
