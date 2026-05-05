import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import yaml
import os

# ============================================================
# Global font settings
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13


# ============================================================
# Helper functions
# ============================================================

def clean_poly(poly):
    poly = np.asarray(poly, dtype=float)
    poly = np.trim_zeros(poly, trim="f")
    if len(poly) == 0:
        return np.array([0.0])
    return poly


def get_zeros(num):
    num = clean_poly(num)
    if len(num) <= 1:
        return np.array([])
    return np.roots(num)


def get_poles(den):
    den = clean_poly(den)
    if len(den) <= 1:
        return np.array([])
    return np.roots(den)


def make_tf(num, den):
    num = clean_poly(num)
    den = clean_poly(den)
    return signal.TransferFunction(num, den)


def plot_step_response(num, den, t_final, title, ylabel, filename):
    system = make_tf(num, den)
    t = np.linspace(0, t_final, 2500)
    tout, y = signal.step(system, T=t)

    reference = np.ones_like(tout)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.plot(tout, reference, color="red", linestyle="--", linewidth=1.5, label="Reference Value")
    ax.plot(tout, y, color="black", linestyle="-", linewidth=2.5, label="Actual Value")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.65)
    ax.legend(loc="best")

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pole_zero(num, den, title, filename, xlim=None, ylim=None):
    zeros = get_zeros(num)
    poles = get_poles(den)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    if len(poles) > 0:
        ax.scatter(np.real(poles), np.imag(poles), marker="x", color="red", s=120, linewidths=2.5, label="System Poles")

    if len(zeros) > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker="o", facecolors="none", edgecolors="blue", s=120, linewidths=2.5, label="System Zeros")

    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(0, color="black", linewidth=1.2)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"Real Axis (Seconds$^{-1}$)")
    ax.set_ylabel(r"Imaginary Axis (Seconds$^{-1}$)")
    ax.grid(True, linestyle="--", alpha=0.65)
    ax.legend(loc="best")

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def add_polynomials(p1, p2):
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    max_len = max(len(p1), len(p2))
    p1, p2 = np.pad(p1, (max_len - len(p1), 0)), np.pad(p2, (max_len - len(p2), 0))
    return clean_poly(p1 + p2)


def propeller_thrust(params, delta_t, Va):
    rho, D, KV, KQ, Rmotor, i0, Vmax = params["rho"], params["Dprop"], params["KV"], params["KQ"], params["Rmotor"], params["i0"], params["Vmax"]
    CQ2, CQ1, CQ0 = params["CQ2"], params["CQ1"], params["CQ0"]
    CT2, CT1, CT0 = params["CT2"], params["CT1"], params["CT0"]
    Vin = Vmax * delta_t
    a = rho * D**5 * CQ0 / (2.0 * np.pi) ** 2
    b = rho * D**4 * CQ1 * Va / (2.0 * np.pi) + (KQ**2) / Rmotor
    c = rho * D**3 * CQ2 * Va**2 - (KQ / Rmotor) * Vin + KQ * i0
    disc = max(b**2 - 4.0 * a * c, 0.0)
    Omega = max((-b + np.sqrt(disc)) / (2.0 * a), 1e-6)
    n = Omega / (2.0 * np.pi)
    J = 2.0 * np.pi * Va / (Omega * D)
    CT = CT2 * J**2 + CT1 * J + CT0
    return rho * n**2 * D**4 * CT


def compute_airspeed_derivatives(params, Va_star, alpha_star, theta_star, delta_e_star, delta_t_star):
    m, rho, S, g = params["m"], params["rho"], params["S"], 9.81
    CD_0, CD_alpha, CD_delta_e = params["CD_0"], params["CD_alpha"], params["CD_delta_e"]
    eps_V, eps_dt = 0.1, 1e-4
    dT_dVa = (propeller_thrust(params, delta_t_star, Va_star + eps_V) - propeller_thrust(params, delta_t_star, Va_star - eps_V)) / (2.0 * eps_V)
    dT_ddelta_t = (propeller_thrust(params, delta_t_star + eps_dt, Va_star) - propeller_thrust(params, delta_t_star - eps_dt, Va_star)) / (2.0 * eps_dt)
    a_v1 = (rho * Va_star * S / m) * (CD_0 + CD_alpha * alpha_star + CD_delta_e * delta_e_star) - (1.0 / m) * dT_dVa
    a_v2 = (1.0 / m) * dT_ddelta_t
    a_v3 = g * np.cos(theta_star - alpha_star)
    return a_v1, a_v2, a_v3


def main():
    res_dir = os.path.join(os.path.dirname(__file__), "result")
    if not os.path.exists(res_dir): os.makedirs(res_dir)
    yaml_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../user/uav_parameters.yaml"))
    with open(yaml_file, "r") as f: params = yaml.safe_load(f)

    Va_trim, alpha_trim, theta_trim, delta_e_trim, delta_t_trim = 25.0, 0.0, 0.0, np.deg2rad(-7.69), 0.66
    rho, S, c, Jy = params["rho"], params["S"], params["c"], params["Jy"]
    qbar = 0.5 * rho * Va_trim**2
    Cm_q, Cm_alpha, Cm_delta_e = params["Cm_q"], params["Cm_alpha"], params["Cm_delta_e"]
    a_theta1, a_theta2, a_theta3 = -(qbar * S * c / Jy) * Cm_q * (c / (2.0 * Va_trim)), -(qbar * S * c / Jy) * Cm_alpha, (qbar * S * c / Jy) * Cm_delta_e

    # USER REQUIREMENTS
    zeta_pitch, wn_pitch = 0.707, 20.0
    zeta_altitude, wn_altitude = 0.707, 1.0
    zeta_airspeed, wn_airspeed = 0.707, 2.0

    kp_theta = (wn_pitch**2 - a_theta2) / a_theta3
    kd_theta = (2.0 * zeta_pitch * wn_pitch - a_theta1) / a_theta3
    num_pitch, den_pitch = [kp_theta * a_theta3], [1.0, a_theta1 + kd_theta * a_theta3, a_theta2 + kp_theta * a_theta3]
    K_theta_DC = (kp_theta * a_theta3) / (a_theta2 + kp_theta * a_theta3)

    kp_h = (2.0 * zeta_altitude * wn_altitude) / (K_theta_DC * Va_trim)
    ki_h = (wn_altitude**2) / (K_theta_DC * Va_trim)
    num_altitude = np.polymul([Va_trim * num_pitch[0]], [kp_h, ki_h])
    den_altitude = add_polynomials(np.polymul([1.0, 0.0, 0.0], den_pitch), np.polymul([Va_trim * num_pitch[0]], [kp_h, ki_h]))

    a_v1, a_v2, a_v3 = compute_airspeed_derivatives(params, Va_trim, alpha_trim, theta_trim, delta_e_trim, delta_t_trim)
    kp_v, ki_v = (2.0 * zeta_airspeed * wn_airspeed - a_v1) / a_v2, (wn_airspeed**2) / a_v2
    num_airspeed, den_airspeed = [a_v2 * kp_v, a_v2 * ki_v], [1.0, a_v1 + a_v2 * kp_v, a_v2 * ki_v]

    plot_step_response(num_pitch, den_pitch, 1.0, f"Pitch Step Response (wn={wn_pitch}, zeta={zeta_pitch})", r"Pitch $\theta$ (rad)", os.path.join(res_dir, "pitch_step.png"))
    plot_pole_zero(num_pitch, den_pitch, "Pitch Pole-Zero Map", os.path.join(res_dir, "pitch_pz.png"))
    plot_step_response(num_altitude, den_altitude, 15.0, f"Altitude Cascade Step (wn={wn_altitude})", r"Altitude $h$ (m)", os.path.join(res_dir, "alt_step.png"))
    plot_pole_zero(num_altitude, den_altitude, "Altitude Pole-Zero Map", os.path.join(res_dir, "alt_pz.png"))
    plot_step_response(num_airspeed, den_airspeed, 10.0, f"Airspeed Step (wn={wn_airspeed})", r"Airspeed $V_a$ (m/s)", os.path.join(res_dir, "airspeed_step.png"))
    plot_pole_zero(num_airspeed, den_airspeed, "Airspeed Pole-Zero Map", os.path.join(res_dir, "airspeed_pz.png"))

if __name__ == "__main__":
    main()