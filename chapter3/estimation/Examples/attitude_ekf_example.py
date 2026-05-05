from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chapter3.estimation.attitude_ekf import AttitudeEKF


def attitude_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    phi = x[0]
    theta = x[1]
    p, q, r, _ = u

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_theta = np.tan(theta)

    phi_dot = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
    theta_dot = q * cos_phi - r * sin_phi
    return np.array([phi_dot, theta_dot], dtype=float)


def accelerometer_measurement(x: np.ndarray, u: np.ndarray, g: float = 9.80665) -> np.ndarray:
    phi = x[0]
    theta = x[1]
    p, q, r, va = u

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array(
        [
            q * va * sin_theta + g * sin_theta,
            r * va * cos_theta - p * va * sin_theta - g * cos_theta * sin_phi,
            -q * va * cos_theta - g * cos_theta * cos_phi,
        ],
        dtype=float,
    )


def build_test_maneuver(time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p_true = np.deg2rad(8.0) * np.sin(2.0 * np.pi * 0.22 * time)
    q_true = np.deg2rad(6.0) * np.sin(2.0 * np.pi * 0.17 * time + 0.4)
    r_true = np.deg2rad(4.0) * np.sin(2.0 * np.pi * 0.11 * time + 0.8)

    p_true += np.deg2rad(3.0) * ((time > 6.0) & (time < 11.0))
    q_true += np.deg2rad(-2.5) * ((time > 12.0) & (time < 16.0))
    r_true += np.deg2rad(2.0) * ((time > 18.0) & (time < 22.0))

    va_true = 25.0 + 1.2 * np.sin(2.0 * np.pi * 0.05 * time) + 0.6 * np.sin(2.0 * np.pi * 0.13 * time)

    rate_stack = np.column_stack((p_true, q_true, r_true))
    return p_true, q_true, r_true, va_true, rate_stack


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times New Roman PS", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def main() -> None:
    configure_matplotlib()
    np.random.seed(7)

    dt = 0.01
    t_end = 25.0
    time = np.arange(0.0, t_end, dt)
    n_steps = time.size
    g = 9.80665

    p_true, q_true, r_true, va_true, rates_true = build_test_maneuver(time)

    true_state = np.zeros((n_steps, 2), dtype=float)
    true_state[0] = np.deg2rad([4.0, -2.5])

    for k in range(1, n_steps):
        u_prev = np.array([p_true[k - 1], q_true[k - 1], r_true[k - 1], va_true[k - 1]], dtype=float)
        true_state[k] = true_state[k - 1] + attitude_dynamics(true_state[k - 1], u_prev) * dt
        true_state[k, 1] = np.clip(true_state[k, 1], -np.deg2rad(89.0), np.deg2rad(89.0))

    gyro_noise_std = np.deg2rad(0.35)
    va_noise_std = 0.15
    accel_noise_std = 0.28

    gyro_meas = rates_true + np.random.normal(0.0, gyro_noise_std, size=rates_true.shape)
    va_meas = va_true + np.random.normal(0.0, va_noise_std, size=n_steps)

    accel_true = np.zeros((n_steps, 3), dtype=float)
    accel_meas = np.zeros((n_steps, 3), dtype=float)
    for k in range(n_steps):
        u_true = np.array([p_true[k], q_true[k], r_true[k], va_true[k]], dtype=float)
        accel_true[k] = accelerometer_measurement(true_state[k], u_true, g=g)
        accel_meas[k] = accel_true[k] + np.random.normal(0.0, accel_noise_std, size=3)

    ekf = AttitudeEKF(
        dt=dt,
        g=g,
        q_diag=np.array([3.5e-3, 3.5e-3]),
        r_diag=np.array([accel_noise_std**2, accel_noise_std**2, accel_noise_std**2]),
        x0=np.deg2rad([0.0, 0.0]),
        p0=np.diag([np.deg2rad(12.0) ** 2, np.deg2rad(12.0) ** 2]),
    )

    est_state = np.zeros((n_steps, 2), dtype=float)
    est_state[0] = ekf.x.reshape(-1)

    for k in range(1, n_steps):
        u_meas = np.array([gyro_meas[k, 0], gyro_meas[k, 1], gyro_meas[k, 2], va_meas[k]], dtype=float)
        ekf.predict(u_meas)
        ekf.update(accel_meas[k], u_meas)
        est_state[k] = ekf.x.reshape(-1)

    output_dir = PROJECT_ROOT / "chapter3" / "estimation" / "Examples" / "exported_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "attitude_ekf_roll_pitch.png"
    pdf_path = output_dir / "attitude_ekf_roll_pitch.pdf"

    phi_true_deg = np.rad2deg(true_state[:, 0])
    theta_true_deg = np.rad2deg(true_state[:, 1])
    phi_est_deg = np.rad2deg(est_state[:, 0])
    theta_est_deg = np.rad2deg(est_state[:, 1])

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)

    axes[0].plot(time, phi_true_deg, color="black", linestyle="-", linewidth=0.9, label="True roll")
    axes[0].plot(time, phi_est_deg, color="red", linestyle="--", linewidth=0.9, label="EKF estimate")
    axes[0].set_ylabel(r"Roll, $\phi$ (deg)")
    axes[0].set_title("Attitude Estimation Using EKF")
    axes[0].grid(True, linestyle="--", linewidth=0.45, alpha=0.55)
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].plot(time, theta_true_deg, color="black", linestyle="-", linewidth=0.9, label="True pitch")
    axes[1].plot(time, theta_est_deg, color="red", linestyle="--", linewidth=0.9, label="EKF estimate")
    axes[1].set_xlabel("Time, t (s)")
    axes[1].set_ylabel(r"Pitch, $\theta$ (deg)")
    axes[1].grid(True, linestyle="--", linewidth=0.45, alpha=0.55)
    axes[1].legend(loc="upper right", frameon=True)

    for axis in axes:
        axis.tick_params(direction="in", length=4, width=0.8)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    
    plt.close(fig)

    roll_rmse = float(np.sqrt(np.mean((phi_est_deg - phi_true_deg) ** 2)))
    pitch_rmse = float(np.sqrt(np.mean((theta_est_deg - theta_true_deg) ** 2)))

    print(f"PNG figure saved to: {png_path}")
    
    print(f"Roll RMSE (deg): {roll_rmse:.3f}")
    print(f"Pitch RMSE (deg): {pitch_rmse:.3f}")


if __name__ == "__main__":
    main()
