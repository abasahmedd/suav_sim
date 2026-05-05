import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chapter1.six_dof_equations_of_motion import EquationsOfMotion
from chapter1.airframe_parameters import load_airframe_parameters_from_yaml
from chapter1.control_deflections import ControlDeflections
from chapter2.complete_autopilot_system import CompleteAutopilot
from chapter3.path.mission_management import MissionManagement
from chapter3.path.waypoints import load_waypoints_from_txt
from DataDisplay.attitude_position_panel import AttitudePositionPanel
from DataDisplay.sim_console_live import SimConsoleLive
from sim_math.simulation_data import SimulationData

def main():
    # 1. Load Aircraft Parameters
    params_file = "user/uav_parameters.yaml"
    uav_params = load_airframe_parameters_from_yaml(params_file)

    # 2. Initialize Equations of Motion
    dt = 0.01
    uav = EquationsOfMotion(dt, uav_params, use_quat=True)

    # 3. Perform Trim Calculation at 25 m/s
    Va_trim = 25.0
    print(f"\nCalculating Trim at {Va_trim} m/s...")
    x_trim, delta_trim_array = uav.trim(Va=Va_trim, gamma=0.0, R_orb=np.inf, update=True, verbose=True)
    
    # This is our fixed trim configuration that we will use throughout the flight
    fixed_trim_deflections = ControlDeflections(delta_trim_array)

    # 4. Initialize Monitoring Systems (Autopilot and Mission)
    # We initialize them but we WILL NOT use their control outputs.
    # We only use them to provide data for the Console/GUI.
    autopilot = CompleteAutopilot(dt, uav_params, uav.state)
    
    waypoints_file = "user/go_waypoint.wp"
    waypoints_list = load_waypoints_from_txt(waypoints_file)
    
    mission = MissionManagement(dt, autopilot.config)
    mission.initialize(waypoints_list, Va=Va_trim, h=uav.state.altitude, chi=uav.state.yaw)

    # 5. Initialize GUI and Console
    cli = SimConsoleLive()
    gui = AttitudePositionPanel(figsize=(12, 6), use_blit=False, pos_3d=True)

    # 6. Simulation Loop (WITHOUT ACTIVE CONTROL)
    print("\nStarting simulation WITHOUT active control...")
    print("The Console will show targets, but the aircraft will only use fixed trim deflections.")
    
    t_sim = 0.0
    t0 = time.time()
    
    while t_sim < 60.0:
        t_sim += dt
        
        # We update the monitoring systems so they calculate errors (for display only)
        mission.update(uav.state.ned_position, uav.state.course_angle)
        autopilot.status.update_aircraft_state(uav.state)
        
        # !!! IMPORTANT: We DO NOT call autopilot.control_... methods !!!
        # Instead, we use the fixed trim values calculated at the start.
        uav.update(fixed_trim_deflections)
        
        # Visualization
        t_real = time.time() - t0
        if t_sim > t_real:
            gui.add_data(state=uav.state)
            gui.update(state=uav.state, pause=0.001)
            
            # Now SimulationData has valid objects, so the CLI won't crash
            cli.update(
                SimulationData(
                    dt_sim=dt,
                    t_sim=t_sim,
                    k_sim=int(t_sim/dt),
                    uav_state=uav.state,
                    control_deflections=fixed_trim_deflections,
                    autopilot_status=autopilot.status,
                    mission_control=mission,
                )
            )
            
        if not plt.fignum_exists(gui.fig.number):
            break

    print("\nSimulation finished.")

if __name__ == "__main__":
    main()
