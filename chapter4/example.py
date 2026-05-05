import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from chapter1.six_dof_equations_of_motion import EquationsOfMotion
from chapter1.airframe_parameters import load_airframe_parameters_from_yaml
from chapter2.complete_autopilot_system import CompleteAutopilot
from chapter3.path.mission_management import MissionManagement
from chapter3.path.waypoints import load_waypoints_from_txt
from DataDisplay.sim_console_live import SimConsoleLive
from DataDisplay.attitude_position_panel import AttitudePositionPanel
from sim_math.simulation_data import SimulationData


#====================Load parameters============================#
params_file = "user/uav_parameters.yaml"
uav_params = load_airframe_parameters_from_yaml(params_file)


#====================Simulation parameters======================#
dt = 0.01
uav = EquationsOfMotion(dt, uav_params, use_quat=True)


#====================Initial trim===============================#
x_trim, delta_trim = uav.trim(Va=25.0)


#====================Initialize CompleteAutopilot=======================#
autopilot = CompleteAutopilot(dt, uav_params, uav.state)


#====================Load waypoints=============================#
waypoints_file = "user/discussion_simple_square.wp"
waypoints_list = load_waypoints_from_txt(waypoints_file)
mission = MissionManagement(dt, autopilot.config, nav_type="fillets")
mission.initialize(waypoints_list, Va=25.0, h=0.0, chi=0.0)


#====================Initialize CLI and GUI=======================#
cli = SimConsoleLive()
try:
    gui = AttitudePositionPanel(use_blit=False, pos_3d=True)
except Exception as exc:
    gui = None
    print(f"GUI disabled: {exc}")


#====================Simulation parameters======================#
t_sim = 0.0  # simulation time
k_sim = 0  # simulation steps
t0 = time.time()
max_sim_time = os.environ.get("SIM_MAX_TIME")
max_sim_time = float(max_sim_time) if max_sim_time else None


while True:
    if max_sim_time is not None and t_sim >= max_sim_time:
        break

    t_sim += dt
    k_sim += 1

    uav.update(autopilot.control_deflections)  # update simulation states

    flight_cmd = mission.update(uav.state.ned_position, uav.state.course_angle)
    autopilot.status.update_aircraft_state(uav.state)
    autopilot.control_course_altitude_airspeed(
        flight_cmd.course, flight_cmd.altitude, flight_cmd.airspeed
    )

    if gui is not None:
        gui.add_data(state=uav.state)

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
