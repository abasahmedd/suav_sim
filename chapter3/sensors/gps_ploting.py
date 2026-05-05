import numpy as np
import matplotlib.pyplot as plt

# =========================
# Academic plotting settings
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.dpi"] = 300

# Parameters
sample_rate = 1.0          # Hz
Ts = 1.0 / sample_rate     # sampling time
time_constant = 1100.0     # s
position_std = np.array([0.21, 0.21, 0.40])  # [north, east, down]

# Total simulation time: 12 hours
t_final = 12 * 3600  # seconds
N = int(t_final / Ts)

# Storage
time = np.arange(0, t_final, Ts)
position_error = np.zeros((N, 3))   # north, east, down

# Initial error
err = np.zeros(3)

# Gauss-Markov update
phi = np.exp(-Ts / time_constant)

for k in range(N):
    err = phi * err + np.random.normal(0.0, position_std, size=3)
    position_error[k, :] = err

# Altitude error = -down error
altitude_error = -position_error[:, 2]

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# =========================
# Upper plot: full 12 hours
# =========================
axes[0].plot(time / 3600, altitude_error, color='red', linewidth=0.8)
axes[0].set_xlabel("time (hours)")
axes[0].set_ylabel("altitude error (m)")
axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# =========================
# Lower plot: zoom from 100 s to 200 s
# =========================
mask = (time >= 100) & (time <= 200)
axes[1].plot(time[mask], altitude_error[mask], color='red', linewidth=0.8)
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("altitude error (m)")
axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Layout
plt.tight_layout()

# Optional: save figure for thesis
plt.savefig("gps_gauss_markov_altitude_error.png", dpi=300, bbox_inches="tight")

# Show
plt.show()