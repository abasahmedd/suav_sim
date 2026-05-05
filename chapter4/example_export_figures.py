import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from chapter1.airframe_parameters import load_airframe_parameters_from_yaml
from chapter1.six_dof_equations_of_motion import EquationsOfMotion
from chapter2.complete_autopilot_system import CompleteAutopilot
from chapter3.estimation.attitude_ekf import AttitudeEKF
from chapter3.estimation.gps_smoothing_ekf import GPSSmoothingEKF
from chapter3.path.mission_management import MissionManagement
from chapter3.path.waypoints import load_waypoints_from_txt
from chapter3.sensors.integrated_sensor_system import IntegratedSensorSystem
from DataDisplay.constants import EARTH_GRAVITY_CONSTANT
from DataDisplay.sim_console_live import SimConsoleLive
from sim_math.simulation_data import SimulationData


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def estimate_altitude_from_baro(baro_kpa: float) -> float:
    rho0 = 1.225
    g = 9.81
    return (101325.0 - baro_kpa * 1e3) / (rho0 * g)


def estimate_airspeed_from_pressure(diff_pressure_kpa: float, altitude_m: float) -> float:
    rho = 1.225 * np.exp(-altitude_m / 8500.0)
    return float(np.sqrt(max(2.0 * diff_pressure_kpa * 1e3 / max(rho, 1.0e-6), 0.0)))


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times New Roman PS", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def create_simulation():
    params_file = "user/uav_parameters.yaml"
    uav_params = load_airframe_parameters_from_yaml(params_file)

    dt = 0.01
    uav = EquationsOfMotion(dt, uav_params, use_quat=True)
    uav.trim(Va=25.0)

    autopilot = CompleteAutopilot(dt, uav_params, uav.state)

    waypoints_file = "user/discussion_simple_square.wp"
    waypoints_list = load_waypoints_from_txt(waypoints_file)
    mission = MissionManagement(dt, autopilot.config, nav_type="fillets")
    mission.initialize(waypoints_list, Va=25.0, h=30.0, chi=0.0)

    return dt, uav, autopilot, mission, waypoints_list


def create_route_coords(waypoints_list) -> np.ndarray:
    waypoint_coords = waypoints_list.get_all_waypoint_coords()
    if len(waypoint_coords) >= 5:
        return np.array(
            [
                waypoint_coords[0],
                waypoint_coords[1],
                waypoint_coords[2],
                waypoint_coords[3],
                waypoint_coords[4],
                waypoint_coords[1],
            ]
        )
    return waypoint_coords


def get_target_ned_reference(mission: MissionManagement, state_pos: np.ndarray) -> np.ndarray:
    if mission.target_waypoint is not None:
        return mission.target_waypoint.ned_coords.copy()
    return state_pos.copy()


def run_simulation(
    dt: float,
    uav: EquationsOfMotion,
    autopilot: CompleteAutopilot,
    mission: MissionManagement,
    sim_time: float,
):
    cli = SimConsoleLive()
    sensors = IntegratedSensorSystem(uav.state)
    sensors.initialize(0.0)

    attitude_ekf = AttitudeEKF(
        dt=dt,
        g=EARTH_GRAVITY_CONSTANT,
        q_diag=np.array([3.0e-3, 3.0e-3]),
        r_diag=np.array([0.30**2, 0.30**2, 0.30**2]),
        x0=np.array([uav.state.roll, uav.state.pitch]),
    )
    gps_ekf = GPSSmoothingEKF(
        dt=dt,
        x0=np.array(
            [
                uav.state.pn,
                uav.state.pe,
                uav.state.groundspeed,
                uav.state.course_angle,
                0.0,
                0.0,
                uav.state.yaw,
            ]
        ),
    )

    logs = {
        "t": [],
        "pn": [],
        "pe": [],
        "pd": [],
        "pn_ref": [],
        "pe_ref": [],
        "pd_ref": [],
        "roll": [],
        "pitch": [],
        "yaw": [],
        "roll_ref": [],
        "pitch_ref": [],
        "yaw_ref": [],
        "roll_ekf": [],
        "pitch_ekf": [],
        "yaw_ekf": [],
        "pn_ekf": [],
        "pe_ekf": [],
        "pd_ekf": [],
        "course": [],
        "course_ref": [],
        "course_ekf": [],
        "alt": [],
        "alt_ref": [],
        "alt_ekf": [],
        "va": [],
        "va_ref": [],
        "va_ekf": [],
        "delta_a": [],
        "delta_e": [],
        "delta_r": [],
        "delta_t": [],
        "dist_wp": [],
    }

    t_sim = 0.0
    k_sim = 0
    t0 = time.time()
    baro_altitude_bias = None
    estimated_altitude = float(uav.state.altitude)
    estimated_airspeed = float(uav.state.airspeed)

    while t_sim < sim_time:
        t_sim += dt
        k_sim += 1

        uav.update(autopilot.control_deflections)
        sensors.update(t_sim)
        readings = sensors.read(t_sim)

        baro_altitude = estimate_altitude_from_baro(readings.baro.data.item())
        if baro_altitude_bias is None:
            baro_altitude_bias = baro_altitude - uav.state.altitude
        baro_altitude_corrected = baro_altitude - baro_altitude_bias
        if readings.gps.is_new:
            estimated_altitude = 0.85 * baro_altitude_corrected + 0.15 * float(readings.gps.data[2])
        else:
            estimated_altitude = 0.98 * estimated_altitude + 0.02 * baro_altitude_corrected
        estimated_airspeed = estimate_airspeed_from_pressure(
            readings.airspeed.data.item(),
            estimated_altitude,
        )

        gyro_rad = np.deg2rad(readings.gyro.data.astype(float))
        accel_si = readings.accel.data.astype(float) * EARTH_GRAVITY_CONSTANT
        attitude_input = np.array([gyro_rad[0], gyro_rad[1], gyro_rad[2], estimated_airspeed], dtype=float)
        attitude_ekf.predict(attitude_input)
        if readings.accel.is_new:
            attitude_ekf.update(accel_si, attitude_input)

        estimated_roll = float(attitude_ekf.x[0, 0])
        estimated_pitch = float(attitude_ekf.x[1, 0])
        nav_input = np.array(
            [estimated_airspeed, gyro_rad[1], gyro_rad[2], estimated_roll, estimated_pitch],
            dtype=float,
        )
        gps_ekf.predict(nav_input)
        if readings.gps.is_new:
            gps_data = readings.gps.data.astype(float)
            gps_measurement = np.array(
                [gps_data[0], gps_data[1], gps_data[3], np.deg2rad(gps_data[4])],
                dtype=float,
            )
            gps_ekf.update_gps(gps_measurement, nav_input)
        gps_ekf.update_pseudo(np.zeros(2, dtype=float), nav_input)

        flight_cmd = mission.update(uav.state.ned_position, uav.state.course_angle)
        autopilot.status.update_aircraft_state(uav.state)
        autopilot.control_course_altitude_airspeed(
            flight_cmd.course, flight_cmd.altitude, flight_cmd.airspeed
        )

        target_ned = get_target_ned_reference(mission, uav.state.ned_position)
        dist_wp = mission.route_manager.get_distance_to_waypoint(
            uav.state.ned_position, is_3d=False
        )

        logs["t"].append(t_sim)
        logs["pn"].append(uav.state.pn)
        logs["pe"].append(uav.state.pe)
        logs["pd"].append(uav.state.pd)
        logs["pn_ref"].append(target_ned[0])
        logs["pe_ref"].append(target_ned[1])
        logs["pd_ref"].append(target_ned[2])
        logs["roll"].append(uav.state.roll)
        logs["pitch"].append(uav.state.pitch)
        logs["yaw"].append(uav.state.yaw)
        logs["roll_ref"].append(autopilot.status.target_roll)
        logs["pitch_ref"].append(autopilot.status.target_pitch)
        logs["yaw_ref"].append(autopilot.status.target_course)
        logs["roll_ekf"].append(estimated_roll)
        logs["pitch_ekf"].append(estimated_pitch)
        logs["yaw_ekf"].append(gps_ekf.x[6, 0])
        logs["pn_ekf"].append(gps_ekf.x[0, 0])
        logs["pe_ekf"].append(gps_ekf.x[1, 0])
        logs["pd_ekf"].append(-estimated_altitude)
        logs["course"].append(uav.state.course_angle)
        logs["course_ref"].append(autopilot.status.target_course)
        logs["course_ekf"].append(gps_ekf.x[3, 0])
        logs["alt"].append(uav.state.altitude)
        logs["alt_ref"].append(autopilot.status.target_altitude)
        logs["alt_ekf"].append(estimated_altitude)
        logs["va"].append(uav.state.airspeed)
        logs["va_ref"].append(autopilot.status.target_airspeed)
        logs["va_ekf"].append(estimated_airspeed)
        logs["delta_a"].append(autopilot.control_deflections.delta_a)
        logs["delta_e"].append(autopilot.control_deflections.delta_e)
        logs["delta_r"].append(autopilot.control_deflections.delta_r)
        logs["delta_t"].append(autopilot.control_deflections.delta_t)
        logs["dist_wp"].append(dist_wp)

        t_real = time.time() - t0
        if t_sim > t_real:
            cli.update(
                SimulationData(
                    dt_sim=dt,
                    t_sim=t_sim,
                    k_sim=k_sim,
                    uav_state=uav.state,
                    control_deflections=autopilot.control_deflections,
                    autopilot_status=autopilot.status,
                    mission_control=mission,
                )
            )

    return {key: np.array(value) for key, value in logs.items()}


def save_figures(logs: dict, waypoints_list, output_dir: Path) -> None:
    configure_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    t = logs["t"]
    pn = logs["pn"]
    pe = logs["pe"]
    pd = logs["pd"]
    pn_ekf = logs["pn_ekf"]
    pe_ekf = logs["pe_ekf"]
    pd_ekf = logs["pd_ekf"]

    roll = np.rad2deg(logs["roll"])
    pitch = np.rad2deg(logs["pitch"])
    yaw = np.rad2deg(np.unwrap(logs["yaw"]))
    roll_ref = np.rad2deg(logs["roll_ref"])
    pitch_ref = np.rad2deg(logs["pitch_ref"])
    yaw_ref = np.rad2deg(np.unwrap(logs["yaw_ref"]))
    roll_ekf = np.rad2deg(logs["roll_ekf"])
    pitch_ekf = np.rad2deg(logs["pitch_ekf"])
    yaw_ekf = np.rad2deg(np.unwrap(logs["yaw_ekf"]))
    course = np.rad2deg(np.unwrap(logs["course"]))
    course_ref = np.rad2deg(np.unwrap(logs["course_ref"]))
    course_ekf = np.rad2deg(np.unwrap(logs["course_ekf"]))
    alt = logs["alt"]
    alt_ref = logs["alt_ref"]
    alt_ekf = logs["alt_ekf"]
    va = logs["va"]
    va_ref = logs["va_ref"]
    va_ekf = logs["va_ekf"]

    e_roll_ref = np.rad2deg(wrap_angle(logs["roll_ref"] - logs["roll"]))
    e_pitch_ref = np.rad2deg(wrap_angle(logs["pitch_ref"] - logs["pitch"]))
    e_yaw_ref = np.rad2deg(wrap_angle(logs["yaw_ref"] - logs["yaw"]))
    e_roll_ekf = np.rad2deg(wrap_angle(logs["roll_ekf"] - logs["roll"]))
    e_pitch_ekf = np.rad2deg(wrap_angle(logs["pitch_ekf"] - logs["pitch"]))
    e_yaw_ekf = np.rad2deg(wrap_angle(logs["yaw_ekf"] - logs["yaw"]))

    wp_coords = waypoints_list.get_all_waypoint_coords()
    wp_pn = wp_coords[:, 0]
    wp_pe = wp_coords[:, 1]
    wp_alt = -wp_coords[:, 2]
    route_coords = create_route_coords(waypoints_list)
    route_pn = route_coords[:, 0]
    route_pe = route_coords[:, 1]
    route_alt = -route_coords[:, 2]

    fig, axs = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
    attitude_data = [
        (roll, roll_ref, roll_ekf, "Roll (deg)"),
        (pitch, pitch_ref, pitch_ekf, "Pitch (deg)"),
        (yaw, yaw_ref, yaw_ekf, "Yaw / Heading (deg)"),
    ]
    for axis, (actual, ref, ekf, label) in zip(axs, attitude_data):
        axis.plot(t, actual, "k-", linewidth=1.0, label="Actual")
        axis.plot(t, ref, "b-", linewidth=0.9, label="Reference")
        axis.plot(t, ekf, "r--", linewidth=0.9, label="EKF")
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    axs[0].legend(loc="best")
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Attitude Response: Actual, Reference and EKF", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "1_attitude_actual_reference_ekf.png", bbox_inches="tight")
    fig.savefig(output_dir / "1_attitude_actual_reference_ekf.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(4, 1, figsize=(8.0, 7.2), sharex=True)
    control_data = [
        (np.rad2deg(logs["delta_a"]), "Aileron (deg)"),
        (np.rad2deg(logs["delta_e"]), "Elevator (deg)"),
        (np.rad2deg(logs["delta_r"]), "Rudder (deg)"),
        (100.0 * logs["delta_t"], "Throttle (%)"),
    ]
    for axis, (signal, label) in zip(axs, control_data):
        axis.plot(t, signal, "k-", linewidth=0.95, label="Control input")
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    axs[0].legend(loc="best")
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Control Input Responses", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "2_control_input_responses.png", bbox_inches="tight")
    fig.savefig(output_dir / "2_control_input_responses.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
    nav_data = [
        (va, va_ref, va_ekf, "Airspeed (m/s)"),
        (course, course_ref, course_ekf, "Course Angle (deg)"),
        (alt, alt_ref, alt_ekf, "Altitude (m)"),
    ]
    for axis, (actual, ref, ekf, label) in zip(axs, nav_data):
        axis.plot(t, actual, "k-", linewidth=1.0, label="Actual")
        axis.plot(t, ref, "b-", linewidth=0.9, label="Reference")
        axis.plot(t, ekf, "r--", linewidth=0.9, label="EKF")
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    axs[0].legend(loc="best")
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Airspeed, Course Angle and Altitude", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "3_airspeed_course_altitude_actual_reference_ekf.png", bbox_inches="tight")
    fig.savefig(output_dir / "3_airspeed_course_altitude_actual_reference_ekf.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 6.0))
    plt.plot(pe, pn, "k-", linewidth=1.1, label="Actual")
    plt.plot(pe_ekf, pn_ekf, "r--", linewidth=0.9, label="EKF")
    plt.plot(route_pe, route_pn, "b:", linewidth=1.0, label="Reference route")
    plt.plot(wp_pe, wp_pn, "bo", markersize=4, label="Waypoints")
    plt.xlabel("East, pe (m)")
    plt.ylabel("North, pn (m)")
    plt.title("2D Top View Flight Trajectory with Waypoints")
    plt.grid(True, linestyle="--", linewidth=0.45, alpha=0.6)
    plt.axis("equal")
    plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_dir / "4_2d_waypoint_top_view_flight_trajectory.png", bbox_inches="tight")
    fig.savefig(output_dir / "4_2d_waypoint_top_view_flight_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pe, pn, -pd, "k-", linewidth=1.1, label="Actual")
    ax.plot(pe_ekf, pn_ekf, -pd_ekf, "r--", linewidth=0.9, label="EKF")
    ax.plot(route_pe, route_pn, route_alt, "b:", linewidth=1.0, label="Reference route")
    ax.plot(wp_pe, wp_pn, wp_alt, "bo", markersize=4, label="Waypoints")
    ax.set_xlabel("East, pe (m)")
    ax.set_ylabel("North, pn (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("3D Waypoint Flight Trajectory")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_dir / "5_3d_waypoint_flight_trajectory.png", bbox_inches="tight")
    fig.savefig(output_dir / "5_3d_waypoint_flight_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
    error_data = [
        (e_roll_ref, e_roll_ekf, "Roll Error (deg)"),
        (e_pitch_ref, e_pitch_ekf, "Pitch Error (deg)"),
        (e_yaw_ref, e_yaw_ekf, "Yaw / Heading Error (deg)"),
    ]
    for axis, (ref_error, ekf_error, label) in zip(axs, error_data):
        axis.plot(t, ref_error, "b-", linewidth=0.9, label="Reference - Actual")
        axis.plot(t, ekf_error, "r--", linewidth=0.9, label="EKF - Actual")
        axis.axhline(0.0, color="k", linewidth=0.6, alpha=0.7)
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    axs[0].legend(loc="best")
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Attitude Tracking Error", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "6_attitude_tracking_error.png", bbox_inches="tight")
    fig.savefig(output_dir / "6_attitude_tracking_error.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dt, uav, autopilot, mission, waypoints_list = create_simulation()
    sim_time = float(os.environ.get("SIM_MAX_TIME", "120.0"))
    logs = run_simulation(
        dt=dt,
        uav=uav,
        autopilot=autopilot,
        mission=mission,
        sim_time=sim_time,
    )
    output_dir = PROJECT_ROOT / "chapter4" / "exported_figures"
    save_figures(logs, waypoints_list, output_dir)
    print(f"Figures exported to: {output_dir}")


if __name__ == "__main__":
    main()
