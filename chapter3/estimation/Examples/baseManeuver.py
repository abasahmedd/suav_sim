import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from chapter1.six_dof_equations_of_motion import EquationsOfMotion
from chapter1.airframe_parameters import load_airframe_parameters_from_yaml
from chapter2.complete_autopilot_system import CompleteAutopilot
from chapter3.path.mission_management import MissionManagement
from chapter3.path.waypoints import load_waypoints_from_txt
from DataDisplay.sim_console_live import SimConsoleLive
from DataDisplay.attitude_position_panel import AttitudePositionPanel
from sim_math.simulation_data import SimulationData

def run_benchmark():
    
    #== ==================Load parameters============================#    
    params_file = r"C:\Users\ASUS\Desktop\sim\user\uav_parameters.yaml"
    uav_params = load_airframe_parameters_from_yaml(params_file)
    
    #====================Simulation parameters======================#    
    dt = 0.01
    uav = EquationsOfMotion(dt, uav_params, use_quat=True)
    
    
    #====================Initial trim===============================#    
    Va_trim = 17.0
    print("Trimming aircraft at Va =", Va_trim, "m/s...")
    x_trim, delta_trim = uav.trim(Va=Va_trim)   


    #====================Initialize Autopilot=======================#    
    autopilot = CompleteAutopilot(dt, uav_params, uav.state)
    
    
    #====================Simulation parameters======================#    
    t_end = 40.0
    time_array = np.arange(0, t_end, dt)
    n_steps = len(time_array)
    
    
    #====================Logging arrays=============================#    
    log_h = np.zeros(n_steps)
    log_h_c = np.zeros(n_steps)
    log_chi = np.zeros(n_steps)
    log_chi_c = np.zeros(n_steps)
    
    
    #====================Initial Setup: h = 100m====================#    
    uav.state.ned_position[2] = -100.0
    Va_c = 17.0
    
    
    #====================Simulation loop============================#    
    print("Starting Benchmark Maneuver Simulation...")
    for i, t in enumerate(time_array):
        # Update references based on time for benchmark maneuver



        #===================def the fig overview =====================#
        # Commanded altitude based on 
        if t < 5.0:
            h_c = 100.0
        elif t < 15.0:
            h_c = 105.0
        elif t < 25.0:
            h_c = 95.0
        else:
            h_c = 100.0
            
        # Commanded heading (course angle) based on 
        if t < 10.0:
            chi_c = np.deg2rad(30.0)
        elif t < 20.0:
            chi_c = np.deg2rad(-30.0)
        else:
            chi_c = np.deg2rad(0.0)
        #===================def the fig overview =====================#

        # 1. Update UAV dynamics with current control surfaces
        uav.update(autopilot.control_deflections)
        
        # 2. Update Autopilot with new state
        autopilot.status.update_aircraft_state(uav.state)
        
        # 3. Compute control outputs
        autopilot.control_course_altitude_airspeed(chi_c, h_c, Va_c)
        
        # Add trim feedforward to prevent initial dive
        autopilot.control_deflections.delta_e += delta_trim[1]
        autopilot.control_deflections.delta_t += delta_trim[3]
        
        # 4. Log data for plotting
        log_h[i] = -uav.state.ned_position[2]  # Altitude is -pd
        log_h_c[i] = h_c
        log_chi[i] = np.rad2deg(uav.state.course_angle)
        log_chi_c[i] = np.rad2deg(chi_c)
        
    print("Simulation Complete. Plotting results...")
    
    # Plotting to match the book layout
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Altitude Plot
    axes[0].plot(time_array, log_h_c, 'k-', lw=1.0, label=r'Commanded altitude ($h_c$)')
    axes[0].plot(time_array, log_h, 'b-', lw=1.5, label=r'Actual altitude ($h$)')
    axes[0].set_ylabel('$h$ (m)')
    axes[0].set_ylim([90, 110])
    axes[0].set_yticks(np.arange(90, 111, 5))
    axes[0].set_title('Commanded altitude', fontsize=10)
    axes[0].grid(True, linestyle='-', alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Course Angle Plot
    axes[1].plot(time_array, log_chi_c, 'k-', lw=1.0, label=r'Commanded heading ($\psi_c$)')
    axes[1].plot(time_array, log_chi, 'b-', lw=1.5, label=r'Actual heading ($\psi$)')
    axes[1].set_xlabel(r'$t$ (s)')
    
    # Let's fix y-label to match the book strictly
    axes[1].set_ylabel(r'$\psi_c$ (deg)')
    axes[1].set_ylim([-45, 45])
    axes[1].set_yticks(np.arange(-40, 41, 10))
    axes[1].set_title('Commanded heading', fontsize=10)
    axes[1].grid(True, linestyle='-', alpha=0.3)
    axes[1].legend(loc='upper right')
    
    for ax in axes:
        ax.set_xlim([0, 40])
        ax.set_xticks(np.arange(0, 41, 5))
        
    plt.tight_layout()
    # Save the plot in the script's directory
    save_path = os.path.join(os.path.dirname(__file__), 'benchmark_maneuver.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    run_benchmark()
