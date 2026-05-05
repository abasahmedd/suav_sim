import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy import signal

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from chapter1.airframe_parameters import load_airframe_parameters_from_yaml
from chapter2.autopilot_design_parameters import AutopilotDesignParameters
from chapter1.aircraft_states import AircraftStates
from chapter1.control_deflections import ControlDeflections
from sim_math.transfer_function import TransferFunction


# ============================================================
# Font settings
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10


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


def plot_pole_zero_standalone(num, den, title, filename):
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    zeros = get_zeros(num)
    poles = get_poles(den)
    if len(poles) > 0:
        ax.scatter(np.real(poles), np.imag(poles), marker="x", color="red", s=90, linewidths=2, label="System Poles")
    if len(zeros) > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker="o", facecolors="none", edgecolors="blue", s=90, linewidths=2, label="System Zeros")
    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"Real Axis (Seconds$^{-1}$)")
    ax.set_ylabel(r"Imaginary Axis (Seconds$^{-1}$)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_step_response_standalone(tf, t_final, title, ylabel, filename):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    t = np.linspace(0, t_final, 1000)
    t, y = tf.step(t=t)
    reference = np.ones_like(t)
    ax.plot(t, reference, color="red", linestyle="--", linewidth=1.0, label="Reference Value")
    ax.plot(t, y, color="black", linestyle="-", linewidth=2.0, label="Actual Value")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def impulse_response(num, den, t):
    system = signal.TransferFunction(num, den)
    tout, y = signal.impulse(system, T=t)
    return tout, y


def closed_loop_disturbance_tf(num_p, den_p, num_h, den_h, feedback_sign=-1.0):
    num_cl = np.polymul(num_p, den_h)
    den_1 = np.polymul(den_p, den_h)
    den_2 = np.polymul(num_p, num_h)
    max_len = max(len(den_1), len(den_2))
    den_1 = np.pad(den_1, (max_len - len(den_1), 0))
    den_2 = np.pad(den_2, (max_len - len(den_2), 0))
    den_cl = den_1 + feedback_sign * den_2
    return clean_poly(num_cl), clean_poly(den_cl)


def compute_lateral_yaw_derivatives(params, Va):
    Gamma = params.Jx * params.Jz - params.Jxz ** 2
    Gamma3, Gamma4, Gamma8 = params.Jz / Gamma, params.Jxz / Gamma, params.Jx / Gamma
    C_p_p = Gamma3 * params.Cl_p + Gamma4 * params.Cn_p
    C_p_delta_a = Gamma3 * params.Cl_delta_a + Gamma4 * params.Cn_delta_a
    C_r_beta = Gamma4 * params.Cl_beta + Gamma8 * params.Cn_beta
    C_r_r = Gamma4 * params.Cl_r + Gamma8 * params.Cn_r
    C_r_delta_r = Gamma4 * params.Cl_delta_r + Gamma8 * params.Cn_delta_r
    qbar = 0.5 * params.rho * Va ** 2
    Y_v = qbar * params.S / (params.m * Va) * params.CY_beta
    Y_r = qbar * params.S / (params.m * Va) * params.CY_r * params.b / (2.0 * Va) - 1.0
    Y_delta_r = qbar * params.S / (params.m * Va) * params.CY_delta_r
    N_v, N_r = qbar * params.S * params.b * C_r_beta, qbar * params.S * params.b * C_r_r * params.b / (2.0 * Va)
    N_delta_r = qbar * params.S * params.b * C_r_delta_r
    return {"Y_v": Y_v, "Y_r": Y_r, "Y_delta_r": Y_delta_r, "N_v": N_v, "N_r": N_r, "N_delta_r": N_delta_r, "C_p_p": C_p_p, "C_p_delta_a": C_p_delta_a}


def main():
    # Setup results directory
    res_dir = os.path.join(os.path.dirname(__file__), "result")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    params_file = "user/uav_parameters.yaml"
    params = load_airframe_parameters_from_yaml(params_file)
    Va_trim = 25.0
    delta_e_trim = np.deg2rad(-7.69)
    state_trim = AircraftStates()
    state_trim.update(x=np.array([0, 0, 0, Va_trim, 0, 0, 0, 0, 0, 0, 0, 0]))
    deltas_trim = ControlDeflections(np.array([0.0, delta_e_trim, 0.0, 0.66]))

    control_params = AutopilotDesignParameters()
    control_params.wn_roll, control_params.zeta_roll = 5.0, 0.707
    control_params.BW_course, control_params.zeta_course = 10.0, 0.707
    control_params.calculate_control_gains(params, state_trim, deltas_trim)

    lat = compute_lateral_yaw_derivatives(params, Va_trim)
    qbar = 0.5 * params.rho * Va_trim ** 2
    a_phi1 = -qbar * params.S * params.b * lat["C_p_p"] * params.b / (2.0 * Va_trim)
    a_phi2 = qbar * params.S * params.b * lat["C_p_delta_a"]
    num_roll = [control_params.kp_roll_aileron * a_phi2]
    den_roll = [1.0, a_phi1 + a_phi2 * control_params.kd_roll_aileron, control_params.kp_roll_aileron * a_phi2]
    tf_roll = TransferFunction(num_roll, den_roll)

    g, Vg = 9.81, state_trim.groundspeed
    if Vg <= 0.0: Vg = Va_trim
    k_chi = g / Vg
    kp_chi, ki_chi = control_params.kp_course_roll, control_params.ki_course_roll
    C, d1, d0 = num_roll[0], den_roll[1], den_roll[2]
    num_course = [kp_chi * k_chi * C, ki_chi * k_chi * C]
    den_course = [1.0, d1, d0, kp_chi * k_chi * C, ki_chi * k_chi * C]
    tf_course = TransferFunction(num_course, den_course)

    num_yaw_open = [lat["N_delta_r"], lat["Y_delta_r"] * lat["N_v"] - lat["N_delta_r"] * lat["Y_v"]]
    den_yaw_open = [1.0, -(lat["Y_v"] + lat["N_r"]), lat["Y_v"] * lat["N_r"] - lat["Y_r"] * lat["N_v"]]
    kr, pwo = abs(float(control_params.kr_damper)), abs(float(control_params.pwo_damper))
    num_yaw_closed, den_yaw_closed = closed_loop_disturbance_tf(num_yaw_open, den_yaw_open, [kr, 0.0], [1.0, pwo])

    # EXPORT INDIVIDUAL IMAGES
    plot_step_response_standalone(tf_roll, 5, "Roll Angle Step Response", r"Roll Angle $\phi$ (rad)", os.path.join(res_dir, "roll_step_response.png"))
    plot_pole_zero_standalone(num_roll, den_roll, "Roll Angle Pole-Zero Map", os.path.join(res_dir, "roll_pole_zero.png"))
    plot_step_response_standalone(tf_course, 15, "Course Cascade Step Response", r"Course Angle $\chi$ (rad)", os.path.join(res_dir, "course_cascade_response.png"))
    plot_pole_zero_standalone(num_course, den_course, "Course Cascade Pole-Zero Map", os.path.join(res_dir, "course_pole_zero.png"))

    # Yaw Rate Impulse Export
    plt.figure(figsize=(8, 6))
    t = np.linspace(0, 50, 3000)
    t_open, y_open = impulse_response(num_yaw_open, den_yaw_open, t)
    t_closed, y_closed = impulse_response(num_yaw_closed, den_yaw_closed, t)
    plt.plot(t, np.zeros_like(t), color="red", linestyle="--", linewidth=1.0, label="Reference (Steady)")
    plt.plot(t_open, y_open, color="gray", linestyle="--", alpha=0.7, linewidth=1.5, label="Open Loop")
    plt.plot(t_closed, y_closed, color="black", linestyle="-", linewidth=2.0, label="Closed Loop (Yaw Damper)")
    plt.title("Yaw Rate Impulse Response", fontweight="bold")
    plt.xlabel("Time (s)")
    plt.ylabel(r"Yaw Rate $r$ (rad/s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(res_dir, "yaw_rate_impulse.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plot_pole_zero_standalone(num_yaw_closed, den_yaw_closed, "Closed-Loop Yaw Rate Pole-Zero Map", os.path.join(res_dir, "yaw_rate_pole_zero.png"))

    print(f"\nAll plots have been exported to: {res_dir}")

if __name__ == "__main__":
    main()