
import os
import sys
from pathlib import Path

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from chapter1.six_dof_equations_of_motion import EquationsOfMotion
from chapter1.airframe_parameters import load_airframe_parameters_from_yaml
from chapter3.sensors.integrated_sensor_system import IntegratedSensorSystem
from chapter3.estimation.model_inversion import ModelInversionFilter
from sim_math.rotation import multi_rotation, quat2euler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
params_file = PROJECT_ROOT / "user" / "uav_parameters.yaml"
uav_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = EquationsOfMotion(dt, uav_params, use_quat=True)
x_trim, delta_trim = uav.trim(Va=25.0, R_orb=100.0, gamma=np.deg2rad(10.0))
uav.set_state(x_trim)
uav.set_control_deltas(delta_trim)

sensors = IntegratedSensorSystem(uav.state)
sensors.initialize(t=0.0)

estimator = ModelInversionFilter(dt)

# Create arrays to store simulation data
sim_steps = int(100 / dt)
uav_states_size = len(uav.state.x)  # pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r
uav_states = np.zeros((sim_steps, uav_states_size))
estimated_states_size = 13  # roll, pitch, yaw, p, q, r, Va, h, pn, pe, pd, Vg, heading
estimated_states = np.zeros((sim_steps, estimated_states_size))

t = 0.0
for k in tqdm.tqdm(range(sim_steps)):
    t += dt
    uav.update()

    sensors.update(t)
    readings = sensors.read(t)
    estimated_state = estimator.update(readings)

    uav_states[k, :13] = uav.state.x
    estimated_states[k, :] = estimated_state.as_array()

# Setup academic plot style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--"
})

# Create directory for saving figures
output_dir = Path(__file__).resolve().parent / "inversemodelresults"
output_dir.mkdir(exist_ok=True)

time = np.arange(0, sim_steps * dt, dt)

# Plot true vs estimated roll, pitch, yaw
true_euler = np.rad2deg(quat2euler(uav_states[:, 6:10]))
estimated_euler = np.rad2deg(estimated_states[:, 0:3])
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, true_euler[:, 0], label="True Roll")
plt.plot(time, estimated_euler[:, 0], label="Estimated Roll", linestyle="--")
plt.ylabel("Roll (deg)")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time, true_euler[:, 1], label="True Pitch")
plt.plot(time, estimated_euler[:, 1], label="Estimated Pitch", linestyle="--")
plt.ylabel("Pitch (deg)")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(time, true_euler[:, 2] % 360, label="True Yaw")
plt.plot(time, estimated_euler[:, 2], label="Estimated Yaw", linestyle="--")
plt.ylabel("Yaw (deg)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "01_euler_angles.png", dpi=300, bbox_inches='tight')

# Plot true vs estimated angular rates p, q, r
true_rates = np.rad2deg(uav_states[:, 10:13])
estimated_rates = np.rad2deg(estimated_states[:, 3:6])
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, true_rates[:, 0], label="True p")
plt.plot(time, estimated_rates[:, 0], label="Estimated p", linestyle="--")
plt.ylabel("p (deg/s)")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time, true_rates[:, 1], label="True q")
plt.plot(time, estimated_rates[:, 1], label="Estimated q", linestyle="--")
plt.ylabel("q (deg/s)")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(time, true_rates[:, 2], label="True r")
plt.plot(time, estimated_rates[:, 2], label="Estimated r", linestyle="--")
plt.ylabel("r (deg/s)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "02_angular_rates.png", dpi=300, bbox_inches='tight')

# Plot true vs estimated airspeed and altitude
true_Va = np.linalg.norm(uav_states[:, 3:6], axis=1)
estimated_Va = estimated_states[:, 6]
true_h = -uav_states[:, 2]
estimated_h = estimated_states[:, 7]
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, true_Va, label="True Airspeed")
plt.plot(time, estimated_Va, label="Estimated Airspeed", linestyle="--")
plt.ylabel("Airspeed (m/s)")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time, true_h, label="True Altitude")
plt.plot(time, estimated_h, label="Estimated Altitude", linestyle="--")
plt.ylabel("Altitude (m)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "03_airspeed_altitude.png", dpi=300, bbox_inches='tight')

# Plot true vs estimated position (pn, pe, pd)
true_pn = uav_states[:, 0]
true_pe = uav_states[:, 1]
true_pd = uav_states[:, 2]
estimated_pn = estimated_states[:, 8]
estimated_pe = estimated_states[:, 9]
estimated_pd = estimated_states[:, 10]
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, true_pn, label="True pn")
plt.plot(time, estimated_pn, label="Estimated pn", linestyle="--")
plt.ylabel("pn (m)")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time, true_pe, label="True pe")
plt.plot(time, estimated_pe, label="Estimated pe", linestyle="--")
plt.ylabel("pe (m)")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(time, true_pd, label="True pd")
plt.plot(time, estimated_pd, label="Estimated pd", linestyle="--")
plt.ylabel("pd (m)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "04_position.png", dpi=300, bbox_inches='tight')

# Plot true vs estimated ground speed and course
ned_velocities = multi_rotation(angles=np.deg2rad(true_euler), values=uav_states[:, 3:6], reverse=True)
true_Vg = np.linalg.norm(ned_velocities, axis=1)
true_course = np.arctan2(ned_velocities[:, 1], ned_velocities[:, 0])
estimated_Vg = estimated_states[:, 11]
estimated_course = estimated_states[:, 12]
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, true_Vg, label="True Ground Speed")
plt.plot(time, estimated_Vg, label="Estimated Ground Speed", linestyle="--")
plt.ylabel("Ground Speed (m/s)")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time, np.rad2deg(true_course) % 360, label="True Course")
plt.plot(
    time,
    np.rad2deg(estimated_course) % 360,
    label="Estimated Course",
    linestyle="--",
)
plt.ylabel("Course (deg)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "05_ground_speed_course.png", dpi=300, bbox_inches='tight')

# Plot true vs estimated 3D trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(true_pe, true_pn, -true_pd, label="True Trajectory")
ax.plot(estimated_pe, estimated_pn, -estimated_pd, label="Estimated Trajectory", linestyle="--")
ax.set_xlabel("East (m)")
ax.set_ylabel("North (m)")
ax.set_zlabel("Altitude (m)")
ax.set_title("UAV 3D Flight Trajectory")
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / "06_3d_trajectory.png", dpi=300, bbox_inches='tight')

plt.close("all")
print(f"Saved figures to {output_dir}")
