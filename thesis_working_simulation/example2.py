
import math
import time

from simulator.aircraft import AircraftDynamics, load_airframe_parameters_from_yaml
from simulator.autopilot import Autopilot
from simulator.autopilot.mission_control import MissionControl
from simulator.autopilot.waypoints import load_waypoints_from_txt
from simulator.cli import SimConsole
from simulator.gui import AttitudePositionPanel
from simulator.utils.simulation_data import SimulationData


#====================Load parameters============================#
params_file = r"config/uav_parameters.yaml"
uav_params = load_airframe_parameters_from_yaml(params_file)  


#====================Simulation parameters======================#
dt = 0.01
uav = AircraftDynamics(dt, uav_params, use_quat=True)

#====================Initial trim===============================#
x_trim, delta_trim = uav.trim(Va=25.0, verbose=False)


#====================Initialize Autopilot=======================#
autopilot = Autopilot(dt, uav_params, uav.state)



#====================Load waypoints=============================#
waypoints_file = r"config/waypoint.wp"
waypoints_list = load_waypoints_from_txt(waypoints_file)
mission = MissionControl(dt, autopilot.config)
mission.initialize(waypoints_list, Va=25.0, h=0.0, chi=0.0)


#====================Initialize CLI and GUI=======================#
# cli = SimConsole()  # Disabled
gui = AttitudePositionPanel(use_blit=False, pos_3d=False) # Changed to 2D Top View

# Add waypoints to the plot
gui.add_waypoints(waypoints_list)

try:
    gui.fig.canvas.manager.window.state('zoomed')
except Exception:
    try:
        gui.fig.canvas.manager.window.showMaximized()
    except Exception:
        gui.fig.canvas.manager.full_screen_toggle()


#====================Simulation parameters======================#
t_sim = 0.0  # simulation time
k_sim = 0  # simulation steps
t0 = time.time()



while True:
    t_sim += dt
    k_sim += 1

    uav.update(autopilot.control_deltas)  # update simulation states

    flight_cmd = mission.update(uav.state.ned_position, uav.state.course_angle)
    autopilot.status.update_aircraft_state(uav.state)
    autopilot.control_course_altitude_airspeed(
        flight_cmd.course, flight_cmd.altitude, flight_cmd.airspeed

    )

    sim_data = SimulationData(
        dt_sim=dt,
        t_sim=t_sim,
        k_sim=k_sim,
        uav_state=uav.state,
        control_deltas=autopilot.control_deltas,
        autopilot_status=autopilot.status,
        mission_control=mission,
    )
    
    gui.add_data(data=sim_data)

    t_real = time.time() - t0
    if t_sim > t_real:
        # Limit GUI/CLI updates to 10 FPS (every 0.1 seconds) to prevent crashes
        if not hasattr(mission, "last_gui_time"):
            mission.last_gui_time = 0.0
            
        if t_sim - mission.last_gui_time >= 0.1:
            gui.update(data=sim_data, pause=0.001)
            mission.last_gui_time = t_sim
        else:
            time.sleep(0.01)  # sleep to maintain real-time without overloading UI

