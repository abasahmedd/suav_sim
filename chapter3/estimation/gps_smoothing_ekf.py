import numpy as np

from DataDisplay.constants import EARTH_GRAVITY_CONSTANT


class GPSSmoothingEKF:
    """EKF for navigation-state estimation and GPS smoothing.

    State:
        x = [pn, pe, Vg, chi, wn, we, psi]^T

    Input:
        u = [Va, q, r, phi, theta]^T
    """

    def __init__(
        self,
        dt: float,
        q_diag: np.ndarray | None = None,
        r_gps_diag: np.ndarray | None = None,
        r_pseudo_diag: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        p0: np.ndarray | None = None,
    ) -> None:
        self.dt = dt
        self.g = EARTH_GRAVITY_CONSTANT
        self.Q = np.diag(
            q_diag
            if q_diag is not None
            else np.array([2.0e-2, 2.0e-2, 8.0e-2, 3.0e-3, 2.0e-3, 2.0e-3, 3.0e-3])
        )
        self.R_gps = np.diag(
            r_gps_diag
            if r_gps_diag is not None
            else np.array([2.5**2, 2.5**2, 0.35**2, np.deg2rad(2.0) ** 2])
        )
        self.R_pseudo = np.diag(
            r_pseudo_diag if r_pseudo_diag is not None else np.array([0.45**2, 0.45**2])
        )
        self.x = np.array(
            x0 if x0 is not None else [0.0, 0.0, 24.0, 0.0, 0.0, 0.0, 0.0],
            dtype=float,
        ).reshape(7, 1)
        self.P = np.array(
            p0
            if p0 is not None
            else np.diag(
                [
                    10.0**2,
                    10.0**2,
                    2.0**2,
                    np.deg2rad(10.0) ** 2,
                    3.0**2,
                    3.0**2,
                    np.deg2rad(15.0) ** 2,
                ]
            ),
            dtype=float,
        )

    def predict(self, u: np.ndarray) -> np.ndarray:
        u_col = np.array(u, dtype=float).reshape(5, 1)
        self.x = self.x + self.process_model(self.x, u_col) * self.dt
        self._wrap_state_angles()

        f_jac = self.process_jacobian(self.x, u_col)
        f_discrete = np.eye(7) + f_jac * self.dt
        self.P = f_discrete @ self.P @ f_discrete.T + self.Q * self.dt
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    def update_gps(self, z_gps: np.ndarray, u: np.ndarray) -> np.ndarray:
        z_col = np.array(z_gps, dtype=float).reshape(4, 1)
        u_col = np.array(u, dtype=float).reshape(5, 1)

        h_jac = self.gps_measurement_jacobian(self.x)
        predicted = self.gps_measurement_model(self.x)
        innovation = z_col - predicted
        innovation[3, 0] = self.wrap_angle(innovation[3, 0])

        innovation_cov = h_jac @ self.P @ h_jac.T + self.R_gps
        innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)
        kalman_gain = self.P @ h_jac.T @ self._safe_inverse(innovation_cov)

        self.x = self.x + kalman_gain @ innovation
        self._wrap_state_angles()
        self._joseph_update(kalman_gain, h_jac, self.R_gps)
        return self.x.copy()

    def update_pseudo(self, z_pseudo: np.ndarray, u: np.ndarray) -> np.ndarray:
        z_col = np.array(z_pseudo, dtype=float).reshape(2, 1)
        u_col = np.array(u, dtype=float).reshape(5, 1)

        h_jac = self.pseudo_measurement_jacobian(self.x, u_col)
        predicted = self.pseudo_measurement_model(self.x, u_col)
        innovation = z_col - predicted

        innovation_cov = h_jac @ self.P @ h_jac.T + self.R_pseudo
        innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)
        kalman_gain = self.P @ h_jac.T @ self._safe_inverse(innovation_cov)

        self.x = self.x + kalman_gain @ innovation
        self._wrap_state_angles()
        self._joseph_update(kalman_gain, h_jac, self.R_pseudo)
        return self.x.copy()

    def process_model(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pn, pe, vg, chi, wn, we, psi = self._flatten_state(x)
        va, q, r, phi, theta = self._flatten_input(u)

        vg_safe = self._safe_speed(vg)
        psi_dot = self._psi_dot(q, r, phi, theta)

        pn_dot = vg * np.cos(chi)
        pe_dot = vg * np.sin(chi)
        vg_dot = (
            (va * np.cos(psi) + wn) * (-va * psi_dot * np.sin(psi))
            + (va * np.sin(psi) + we) * (va * psi_dot * np.cos(psi))
        ) / vg_safe
        chi_dot = (self.g / vg_safe) * np.tan(phi) * np.cos(chi - psi)

        return np.array(
            [[pn_dot], [pe_dot], [vg_dot], [chi_dot], [0.0], [0.0], [psi_dot]],
            dtype=float,
        )

    def process_jacobian(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        _, _, vg, chi, wn, we, psi = self._flatten_state(x)
        va, q, r, phi, theta = self._flatten_input(u)

        vg_safe = self._safe_speed(vg)
        cos_chi = np.cos(chi)
        sin_chi = np.sin(chi)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        tan_phi = np.tan(phi)

        psi_dot = self._psi_dot(q, r, phi, theta)

        term1 = (va * cos_psi + wn) * (-va * psi_dot * sin_psi)
        term2 = (va * sin_psi + we) * (va * psi_dot * cos_psi)
        vg_dot_num = term1 + term2

        dvg_dot_dvg = -vg_dot_num / (vg_safe**2)
        dvg_dot_dwn = -psi_dot * va * sin_psi / vg_safe
        dvg_dot_dwe = psi_dot * va * cos_psi / vg_safe
        dvg_dot_dpsi = (
            -va * psi_dot * (va + wn * cos_psi + we * sin_psi)
        ) / vg_safe

        dchi_dot_dvg = -(self.g / (vg_safe**2)) * tan_phi * np.cos(chi - psi)
        dchi_dot_dchi = -(self.g / vg_safe) * tan_phi * np.sin(chi - psi)
        dchi_dot_dpsi = (self.g / vg_safe) * tan_phi * np.sin(chi - psi)

        return np.array(
            [
                [0.0, 0.0, cos_chi, -vg * sin_chi, 0.0, 0.0, 0.0],
                [0.0, 0.0, sin_chi, vg * cos_chi, 0.0, 0.0, 0.0],
                [0.0, 0.0, dvg_dot_dvg, 0.0, dvg_dot_dwn, dvg_dot_dwe, dvg_dot_dpsi],
                [0.0, 0.0, dchi_dot_dvg, dchi_dot_dchi, 0.0, 0.0, dchi_dot_dpsi],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def gps_measurement_model(self, x: np.ndarray) -> np.ndarray:
        pn, pe, vg, chi, _, _, _ = self._flatten_state(x)
        return np.array([[pn], [pe], [vg], [chi]], dtype=float)

    def gps_measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        _ = x
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def pseudo_measurement_model(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        _, _, vg, chi, wn, we, psi = self._flatten_state(x)
        va, _, _, _, _ = self._flatten_input(u)

        return np.array(
            [
                [va * np.cos(psi) + wn - vg * np.cos(chi)],
                [va * np.sin(psi) + we - vg * np.sin(chi)],
            ],
            dtype=float,
        )

    def pseudo_measurement_jacobian(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        _, _, vg, chi, _, _, psi = self._flatten_state(x)
        va, _, _, _, _ = self._flatten_input(u)

        sin_chi = np.sin(chi)
        cos_chi = np.cos(chi)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)

        return np.array(
            [
                [0.0, 0.0, -cos_chi, vg * sin_chi, 1.0, 0.0, -va * sin_psi],
                [0.0, 0.0, -sin_chi, -vg * cos_chi, 0.0, 1.0, va * cos_psi],
            ],
            dtype=float,
        )

    def combined_measurement_model(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.vstack((self.gps_measurement_model(x), self.pseudo_measurement_model(x, u)))

    def combined_measurement_jacobian(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.vstack((self.gps_measurement_jacobian(x), self.pseudo_measurement_jacobian(x, u)))

    def _psi_dot(self, q: float, r: float, phi: float, theta: float) -> float:
        cos_theta_safe = self._safe_cos(theta)
        return q * np.sin(phi) / cos_theta_safe + r * np.cos(phi) / cos_theta_safe

    def _joseph_update(self, kalman_gain: np.ndarray, h_jac: np.ndarray, measurement_cov: np.ndarray) -> None:
        identity = np.eye(self.P.shape[0])
        kh = kalman_gain @ h_jac
        self.P = (identity - kh) @ self.P @ (identity - kh).T + kalman_gain @ measurement_cov @ kalman_gain.T
        self.P = 0.5 * (self.P + self.P.T)

    def _wrap_state_angles(self) -> None:
        self.x[3, 0] = self.wrap_angle(self.x[3, 0])
        self.x[6, 0] = self.wrap_angle(self.x[6, 0])

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _safe_speed(value: float) -> float:
        return float(np.sign(value) * max(abs(value), 1.0e-3)) if value != 0.0 else 1.0e-3

    @staticmethod
    def _safe_cos(angle: float) -> float:
        cos_theta = np.cos(angle)
        if abs(cos_theta) < 1.0e-4:
            return 1.0e-4 if cos_theta >= 0.0 else -1.0e-4
        return float(cos_theta)

    @staticmethod
    def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    @staticmethod
    def _flatten_state(x: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
        pn, pe, vg, chi, wn, we, psi = np.asarray(x).reshape(-1)
        return (
            float(pn),
            float(pe),
            float(vg),
            float(chi),
            float(wn),
            float(we),
            float(psi),
        )

    @staticmethod
    def _flatten_input(u: np.ndarray) -> tuple[float, float, float, float, float]:
        va, q, r, phi, theta = np.asarray(u).reshape(-1)
        return float(va), float(q), float(r), float(phi), float(theta)
