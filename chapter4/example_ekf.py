import os
import sys
import time
import atexit
from pathlib import Path

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
from chapter4.example_export_figures import save_figures
from DataDisplay.constants import EARTH_GRAVITY_CONSTANT
from DataDisplay.ekf_attitude_position_panel import EkfAttitudePositionPanel
from DataDisplay.sim_console_live import SimConsoleLive
from sim_math.simulation_data import SimulationData


def estimate_altitude_from_baro(baro_kpa: float) -> float:
    rho0 = 1.225
    g = 9.81
    return (101325.0 - baro_kpa * 1e3) / (rho0 * g)


def estimate_airspeed_from_pressure(diff_pressure_kpa: float, altitude_m: float) -> float:
    rho = 1.225 * np.exp(-altitude_m / 8500.0)
    return float(np.sqrt(max(2.0 * diff_pressure_kpa * 1e3 / max(rho, 1.0e-6), 0.0)))


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times New Roman PS", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    }
)


params_file = "user/uav_parameters.yaml"
uav_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = EquationsOfMotion(dt, uav_params, use_quat=True)
x_trim, delta_trim = uav.trim(Va=25.0)

autopilot = CompleteAutopilot(dt, uav_params, uav.state)

waypoints_file = "user/discussion_simple_square.wp"
waypoints_list = load_waypoints_from_txt(waypoints_file)
mission = MissionManagement(dt, autopilot.config, nav_type="fillets")
mission.initialize(waypoints_list, Va=25.0, h=0.0, chi=0.0)

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

cli = SimConsoleLive()
enable_gui = os.environ.get("SIM_ENABLE_GUI", "1") != "0"
if enable_gui:
    try:
        waypoint_coords = waypoints_list.get_all_waypoint_coords()
        route_coords = np.array(
            [
                waypoint_coords[0],
                waypoint_coords[1],
                waypoint_coords[2],
                waypoint_coords[3],
                waypoint_coords[4],
                waypoint_coords[1],
            ]
        )
        gui = EkfAttitudePositionPanel(
            use_blit=False,
            pos_3d=True,
            waypoint_coords=waypoint_coords,
            route_coords=route_coords,
        )
    except Exception as exc:
        gui = None
        print(f"GUI disabled: {exc}")
else:
    gui = None

t_sim = 0.0
k_sim = 0
t0 = time.time()
max_sim_time = os.environ.get("SIM_MAX_TIME")
max_sim_time = float(max_sim_time) if max_sim_time else None

estimated_roll = float(uav.state.roll)
estimated_pitch = float(uav.state.pitch)
estimated_position_ned = np.array([uav.state.pn, uav.state.pe, uav.state.pd], dtype=float)
last_airspeed = float(uav.state.airspeed)
baro_altitude_bias = None
estimated_altitude = float(uav.state.altitude)
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
export_done = False


def export_results() -> None:
    global export_done
    if export_done or not logs["t"]:
        return

    export_done = True
    plt.close("all")
    plt.switch_backend("Agg")
    output_dir = PROJECT_ROOT / "chapter4" / "exported_figures"
    save_figures(
        {key: np.array(value) for key, value in logs.items()},
        waypoints_list,
        output_dir,
    )
    print(f"Figures exported to: {output_dir}")


atexit.register(export_results)

while True:
    if max_sim_time is not None and t_sim >= max_sim_time:
        break

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
        gps_altitude = float(readings.gps.data[2])
        estimated_altitude = 0.85 * baro_altitude_corrected + 0.15 * gps_altitude
    else:
        estimated_altitude = 0.98 * estimated_altitude + 0.02 * baro_altitude_corrected
    last_airspeed = estimate_airspeed_from_pressure(readings.airspeed.data.item(), estimated_altitude)

    gyro_rad = np.deg2rad(readings.gyro.data.astype(float))
    accel_si = readings.accel.data.astype(float) * EARTH_GRAVITY_CONSTANT

    attitude_input = np.array([gyro_rad[0], gyro_rad[1], gyro_rad[2], last_airspeed], dtype=float)
    attitude_ekf.predict(attitude_input)
    if readings.accel.is_new:
        attitude_ekf.update(accel_si, attitude_input)

    estimated_roll = float(attitude_ekf.x[0, 0])
    estimated_pitch = float(attitude_ekf.x[1, 0])

    nav_input = np.array(
        [last_airspeed, gyro_rad[1], gyro_rad[2], estimated_roll, estimated_pitch],
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

    estimated_position_ned = np.array(
        [gps_ekf.x[0, 0], gps_ekf.x[1, 0], -estimated_altitude],
        dtype=float,
    )

    flight_cmd = mission.update(uav.state.ned_position, uav.state.course_angle)
    autopilot.status.update_aircraft_state(uav.state)
    autopilot.control_course_altitude_airspeed(
        flight_cmd.course, flight_cmd.altitude, flight_cmd.airspeed
    )
    target_ned = (
        mission.target_waypoint.ned_coords.copy()
        if mission.target_waypoint is not None
        else uav.state.ned_position.copy()
    )
    dist_wp = mission.route_manager.get_distance_to_waypoint(
        uav.state.ned_position,
        is_3d=False,
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
    logs["pd_ekf"].append(estimated_position_ned[2])
    logs["course"].append(uav.state.course_angle)
    logs["course_ref"].append(autopilot.status.target_course)
    logs["course_ekf"].append(gps_ekf.x[3, 0])
    logs["alt"].append(uav.state.altitude)
    logs["alt_ref"].append(autopilot.status.target_altitude)
    logs["alt_ekf"].append(estimated_altitude)
    logs["va"].append(uav.state.airspeed)
    logs["va_ref"].append(autopilot.status.target_airspeed)
    logs["va_ekf"].append(last_airspeed)
    logs["delta_a"].append(autopilot.control_deflections.delta_a)
    logs["delta_e"].append(autopilot.control_deflections.delta_e)
    logs["delta_r"].append(autopilot.control_deflections.delta_r)
    logs["delta_t"].append(autopilot.control_deflections.delta_t)
    logs["dist_wp"].append(dist_wp)

    if gui is not None:
        gui.add_data(
            time=t_sim,
            state=uav.state,
            ap_status=autopilot.status,
            estimated_position_ned=estimated_position_ned,
            estimated_roll=estimated_roll,
            estimated_pitch=estimated_pitch,
        )

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
        if gui is not None:
            gui.update(state=uav.state, pause=0.01)
            if not plt.fignum_exists(gui.fig.number):
                print("GUI window closed. Exporting figures...")
                break

export_results()
