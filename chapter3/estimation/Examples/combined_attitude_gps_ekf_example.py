from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chapter3.estimation.attitude_ekf import AttitudeEKF
from chapter3.estimation.gps_smoothing_ekf import GPSSmoothingEKF


GRAVITY = 9.80665


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


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


def attitude_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    phi, theta = x
    p, q, r, _ = u
    phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    theta_dot = q * np.cos(phi) - r * np.sin(phi)
    return np.array([phi_dot, theta_dot], dtype=float)


def accel_measurement(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    phi, theta = x
    p, q, r, va = u
    return np.array(
        [
            q * va * np.sin(theta) + GRAVITY * np.sin(theta),
            r * va * np.cos(theta) - p * va * np.sin(theta) - GRAVITY * np.cos(theta) * np.sin(phi),
            -q * va * np.cos(theta) - GRAVITY * np.cos(theta) * np.cos(phi),
        ],
        dtype=float,
    )


def psi_dot(q: float, r: float, phi: float, theta: float) -> float:
    cos_theta = np.cos(theta)
    if abs(cos_theta) < 1.0e-4:
        cos_theta = 1.0e-4 if cos_theta >= 0.0 else -1.0e-4
    return q * np.sin(phi) / cos_theta + r * np.cos(phi) / cos_theta


def nav_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    pn, pe, vg, chi, wn, we, psi = x
    va, q, r, phi, theta = u
    vg_safe = max(abs(vg), 1.0e-3)
    psi_rate = psi_dot(q, r, phi, theta)

    pn_dot = vg * np.cos(chi)
    pe_dot = vg * np.sin(chi)
    vg_dot = (
        (va * np.cos(psi) + wn) * (-va * psi_rate * np.sin(psi))
        + (va * np.sin(psi) + we) * (va * psi_rate * np.cos(psi))
    ) / vg_safe
    chi_dot = (GRAVITY / vg_safe) * np.tan(phi) * np.cos(chi - psi)

    return np.array([pn_dot, pe_dot, vg_dot, chi_dot, 0.0, 0.0, psi_rate], dtype=float)


def gps_measurement(x: np.ndarray) -> np.ndarray:
    pn, pe, vg, chi, _, _, _ = x
    return np.array([pn, pe, vg, chi], dtype=float)


def build_maneuver(time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.deg2rad(6.5) * np.sin(2.0 * np.pi * 0.09 * time)
    q = np.deg2rad(4.0) * np.sin(2.0 * np.pi * 0.07 * time + 0.35)
    r = np.deg2rad(5.5) * np.sin(2.0 * np.pi * 0.06 * time + 0.85)

    p += np.deg2rad(2.5) * ((time > 8.0) & (time < 14.0))
    q += np.deg2rad(-1.8) * ((time > 18.0) & (time < 24.0))
    r += np.deg2rad(1.5) * ((time > 28.0) & (time < 34.0))

    va = 24.0 + 1.0 * np.sin(2.0 * np.pi * 0.035 * time) + 0.5 * np.sin(2.0 * np.pi * 0.11 * time)
    return p, q, r, va


def style_axes(axes: np.ndarray) -> None:
    for axis in axes.reshape(-1):
        axis.grid(True, linestyle="--", linewidth=0.45, alpha=0.55)
        axis.tick_params(direction="in", length=4, width=0.8)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)


def main() -> None:
    configure_matplotlib()
    np.random.seed(21)

    dt = 0.02
    t_end = 40.0
    time = np.arange(0.0, t_end + dt, dt)
    n_steps = time.size

    p_true, q_true, r_true, va_true = build_maneuver(time)

    true_att = np.zeros((n_steps, 2), dtype=float)
    true_nav = np.zeros((n_steps, 7), dtype=float)

    true_att[0] = np.deg2rad([3.0, -1.5])
    psi0 = np.deg2rad(12.0)
    wn0 = 2.2
    we0 = -1.4
    vg_n0 = va_true[0] * np.cos(psi0) + wn0
    vg_e0 = va_true[0] * np.sin(psi0) + we0
    true_nav[0] = np.array(
        [0.0, 0.0, np.hypot(vg_n0, vg_e0), np.arctan2(vg_e0, vg_n0), wn0, we0, psi0],
        dtype=float,
    )

    for k in range(1, n_steps):
        u_prev_att = np.array([p_true[k - 1], q_true[k - 1], r_true[k - 1], va_true[k - 1]], dtype=float)
        true_att[k] = true_att[k - 1] + attitude_dynamics(true_att[k - 1], u_prev_att) * dt
        true_att[k, 1] = np.clip(true_att[k, 1], -np.deg2rad(89.0), np.deg2rad(89.0))

        u_prev_nav = np.array(
            [va_true[k - 1], q_true[k - 1], r_true[k - 1], true_att[k - 1, 0], true_att[k - 1, 1]],
            dtype=float,
        )
        true_nav[k] = true_nav[k - 1] + nav_dynamics(true_nav[k - 1], u_prev_nav) * dt
        true_nav[k, 3] = wrap_angle(true_nav[k, 3])
        true_nav[k, 6] = wrap_angle(true_nav[k, 6])

    gyro_noise_std = np.deg2rad(0.30)
    va_noise_std = 0.12
    accel_noise_std = 0.24
    gps_pos_noise_std = 1.8
    gps_vg_noise_std = 0.22
    gps_chi_noise_std = np.deg2rad(1.4)

    gyro_meas = np.column_stack((p_true, q_true, r_true)) + np.random.normal(0.0, gyro_noise_std, size=(n_steps, 3))
    va_meas = va_true + np.random.normal(0.0, va_noise_std, size=n_steps)

    accel_meas = np.zeros((n_steps, 3), dtype=float)
    gps_meas = np.zeros((n_steps, 4), dtype=float)

    for k in range(n_steps):
        u_true_att = np.array([p_true[k], q_true[k], r_true[k], va_true[k]], dtype=float)
        accel_meas[k] = accel_measurement(true_att[k], u_true_att) + np.random.normal(0.0, accel_noise_std, size=3)

        gps_meas[k, 0] = true_nav[k, 0] + np.random.normal(0.0, gps_pos_noise_std)
        gps_meas[k, 1] = true_nav[k, 1] + np.random.normal(0.0, gps_pos_noise_std)
        gps_meas[k, 2] = true_nav[k, 2] + np.random.normal(0.0, gps_vg_noise_std)
        gps_meas[k, 3] = wrap_angle(true_nav[k, 3] + np.random.normal(0.0, gps_chi_noise_std))

    attitude_ekf = AttitudeEKF(
        dt=dt,
        g=GRAVITY,
        q_diag=np.array([3.0e-3, 3.0e-3]),
        r_diag=np.array([accel_noise_std**2, accel_noise_std**2, accel_noise_std**2]),
        x0=np.deg2rad([0.0, 0.0]),
        p0=np.diag([np.deg2rad(10.0) ** 2, np.deg2rad(10.0) ** 2]),
    )

    gps_ekf = GPSSmoothingEKF(
        dt=dt,
        q_diag=np.array([1.5e-2, 1.5e-2, 6.0e-2, 2.0e-3, 8.0e-4, 8.0e-4, 2.0e-3]),
        r_gps_diag=np.array([gps_pos_noise_std**2, gps_pos_noise_std**2, gps_vg_noise_std**2, gps_chi_noise_std**2]),
        r_pseudo_diag=np.array([0.35**2, 0.35**2]),
        x0=np.array([0.0, 0.0, true_nav[0, 2] - 0.8, true_nav[0, 3] + np.deg2rad(2.0), 0.4, -0.3, psi0], dtype=float),
    )

    est_att = np.zeros((n_steps, 2), dtype=float)
    est_nav = np.zeros((n_steps, 7), dtype=float)
    est_att[0] = attitude_ekf.x.reshape(-1)
    est_nav[0] = gps_ekf.x.reshape(-1)

    zero_pseudo = np.zeros(2, dtype=float)

    for k in range(1, n_steps):
        att_input = np.array([gyro_meas[k, 0], gyro_meas[k, 1], gyro_meas[k, 2], va_meas[k]], dtype=float)
        attitude_ekf.predict(att_input)
        attitude_ekf.update(accel_meas[k], att_input)
        est_att[k] = attitude_ekf.x.reshape(-1)

        nav_input = np.array(
            [va_meas[k], gyro_meas[k, 1], gyro_meas[k, 2], est_att[k, 0], est_att[k, 1]],
            dtype=float,
        )
        gps_ekf.predict(nav_input)
        gps_ekf.update_gps(gps_meas[k], nav_input)
        gps_ekf.update_pseudo(zero_pseudo, nav_input)
        est_nav[k] = gps_ekf.x.reshape(-1)

    output_dir = PROJECT_ROOT / "chapter3" / "estimation" / "Examples" / "exported_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "combined_attitude_gps_ekf_results.png"
    pdf_path = output_dir / "combined_attitude_gps_ekf_results.pdf"

    true_att_deg = np.rad2deg(true_att)
    est_att_deg = np.rad2deg(est_att)
    true_nav_deg = true_nav.copy()
    est_nav_deg = est_nav.copy()
    true_nav_deg[:, 3] = np.rad2deg(true_nav[:, 3])
    est_nav_deg[:, 3] = np.rad2deg(est_nav[:, 3])
    true_nav_deg[:, 6] = np.rad2deg(true_nav[:, 6])
    est_nav_deg[:, 6] = np.rad2deg(est_nav[:, 6])

    fig, axes = plt.subplots(3, 2, figsize=(10.0, 8.2), sharex=True)

    axes[0, 0].plot(time, true_att_deg[:, 0], color="black", linewidth=0.9, label="True")
    axes[0, 0].plot(time, est_att_deg[:, 0], color="red", linestyle="--", linewidth=0.9, label="EKF")
    axes[0, 0].set_ylabel(r"Roll, $\phi$ (deg)")
    axes[0, 0].set_title("Combined Attitude and GPS EKF Example")
    axes[0, 0].legend(loc="upper right", frameon=True)

    axes[0, 1].plot(time, true_att_deg[:, 1], color="black", linewidth=0.9, label="True")
    axes[0, 1].plot(time, est_att_deg[:, 1], color="red", linestyle="--", linewidth=0.9, label="EKF")
    axes[0, 1].set_ylabel(r"Pitch, $\theta$ (deg)")
    axes[0, 1].legend(loc="upper right", frameon=True)

    axes[1, 0].plot(time, true_nav[:, 2], color="black", linewidth=0.9, label="True")
    axes[1, 0].plot(time, est_nav[:, 2], color="red", linestyle="--", linewidth=0.9, label="EKF")
    axes[1, 0].set_ylabel(r"$V_g$ (m/s)")
    axes[1, 0].legend(loc="upper right", frameon=True)

    axes[1, 1].plot(time, true_nav_deg[:, 3], color="black", linewidth=0.9, label="True")
    axes[1, 1].plot(time, est_nav_deg[:, 3], color="red", linestyle="--", linewidth=0.9, label="EKF")
    axes[1, 1].set_ylabel(r"Course, $\chi$ (deg)")
    axes[1, 1].legend(loc="upper right", frameon=True)

    axes[2, 0].plot(time, true_nav[:, 4], color="black", linewidth=0.9, label="True")
    axes[2, 0].plot(time, est_nav[:, 4], color="red", linestyle="--", linewidth=0.9, label="EKF")
    axes[2, 0].plot(time, true_nav[:, 5], color="black", linewidth=0.9, alpha=0.45)
    axes[2, 0].plot(time, est_nav[:, 5], color="red", linestyle="--", linewidth=0.9, alpha=0.45)
    axes[2, 0].set_xlabel("Time, t (s)")
    axes[2, 0].set_ylabel(r"Wind, $w_n/w_e$ (m/s)")
    axes[2, 0].legend(["True $w_n$", "EKF $w_n$", "True $w_e$", "EKF $w_e$"], loc="upper right", frameon=True)

    axes[2, 1].plot(time, true_nav_deg[:, 6], color="black", linewidth=0.9, label="True")
    axes[2, 1].plot(time, est_nav_deg[:, 6], color="red", linestyle="--", linewidth=0.9, label="EKF")
    axes[2, 1].set_xlabel("Time, t (s)")
    axes[2, 1].set_ylabel(r"Heading, $\psi$ (deg)")
    axes[2, 1].legend(loc="upper right", frameon=True)

    style_axes(axes)
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    roll_rmse = float(np.sqrt(np.mean((est_att_deg[:, 0] - true_att_deg[:, 0]) ** 2)))
    pitch_rmse = float(np.sqrt(np.mean((est_att_deg[:, 1] - true_att_deg[:, 1]) ** 2)))
    vg_rmse = float(np.sqrt(np.mean((est_nav[:, 2] - true_nav[:, 2]) ** 2)))
    chi_error = np.rad2deg(wrap_angle(est_nav[:, 3] - true_nav[:, 3]))
    psi_error = np.rad2deg(wrap_angle(est_nav[:, 6] - true_nav[:, 6]))
    chi_rmse = float(np.sqrt(np.mean(chi_error**2)))
    psi_rmse = float(np.sqrt(np.mean(psi_error**2)))

    print(f"PNG figure saved to: {png_path}")
    print(f"PDF figure saved to: {pdf_path}")
    print(f"Roll RMSE (deg): {roll_rmse:.3f}")
    print(f"Pitch RMSE (deg): {pitch_rmse:.3f}")
    print(f"Vg RMSE (m/s): {vg_rmse:.3f}")
    print(f"Chi RMSE (deg): {chi_rmse:.3f}")
    print(f"Psi RMSE (deg): {psi_rmse:.3f}")


if __name__ == "__main__":
    main()
