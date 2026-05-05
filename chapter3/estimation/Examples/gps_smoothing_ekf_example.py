import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Global plotting style
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["lines.linewidth"] = 2.0


# ============================================================
# Utility functions
# ============================================================

def wrap_pi(angle):
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def numerical_jacobian(f, x, u, eps=1e-5):
    """
    Numerical Jacobian df/dx.
    """
    n = len(x)
    A = np.zeros((n, n))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps

        fp = f(x + dx, u)
        fm = f(x - dx, u)

        A[:, i] = (fp - fm) / (2.0 * eps)

    return A


# ============================================================
# State vector:
#
# x = [pn, pe, Vg, chi, wn, we, psi]^T
#
# pn  : north position
# pe  : east position
# Vg  : ground speed
# chi : course angle
# wn  : north wind component
# we  : east wind component
# psi : heading angle
#
# Input vector:
#
# u = [Va, q, r, phi, theta]^T
#
# Va    : airspeed
# q     : pitch rate
# r     : yaw rate
# phi   : roll angle
# theta : pitch angle
# ============================================================


def process_model(x, u):
    """
    Navigation process model used in the EKF prediction step.

    The model is consistent with the horizontal wind-triangle relation:

        Vg_n = Va cos(psi) + wn
        Vg_e = Va sin(psi) + we
    """

    pn, pe, Vg, chi, wn, we, psi = x
    Va, q, r, phi, theta = u

    cos_theta = np.cos(theta)
    if abs(cos_theta) < 0.1:
        cos_theta = 0.1 * np.sign(cos_theta) if cos_theta != 0 else 0.1

    # Heading-rate dynamics
    psi_dot = q * np.sin(phi) / cos_theta + r * np.cos(phi) / cos_theta

    # Wind-triangle components
    Vgn = Va * np.cos(psi) + wn
    Vge = Va * np.sin(psi) + we

    Vg_calc = np.sqrt(Vgn**2 + Vge**2)
    Vg_safe = max(Vg_calc, 0.1)

    # Derivatives of Vgn and Vge
    Vgn_dot = -Va * psi_dot * np.sin(psi)
    Vge_dot = Va * psi_dot * np.cos(psi)

    # Ground-speed derivative
    Vg_dot = (Vgn * Vgn_dot + Vge * Vge_dot) / Vg_safe

    # Course-angle derivative
    chi_dot = (Vgn * Vge_dot - Vge * Vgn_dot) / (Vg_safe**2)

    # Position dynamics
    pn_dot = Vg * np.cos(chi)
    pe_dot = Vg * np.sin(chi)

    # Wind components are slowly varying
    wn_dot = 0.0
    we_dot = 0.0

    return np.array([
        pn_dot,
        pe_dot,
        Vg_dot,
        chi_dot,
        wn_dot,
        we_dot,
        psi_dot
    ])


def h_gps(x):
    """
    GPS measurement model.

    z_GPS = [pn_GPS, pe_GPS, Vg_GPS, chi_GPS]^T
    """

    return np.array([
        x[0],   # pn
        x[1],   # pe
        x[2],   # Vg
        x[3]    # chi
    ])


def h_wind_triangle(x, u):
    """
    Wind-triangle pseudo-measurement function.

    Desired pseudo-measurement:

        z_w = [0, 0]^T

    h_w(x,u) =
    [
        Va cos(psi) + wn - Vg cos(chi)
        Va sin(psi) + we - Vg sin(chi)
    ]
    """

    pn, pe, Vg, chi, wn, we, psi = x
    Va, q, r, phi, theta = u

    h1 = Va * np.cos(psi) + wn - Vg * np.cos(chi)
    h2 = Va * np.sin(psi) + we - Vg * np.sin(chi)

    return np.array([h1, h2])


def H_wind_triangle(x, u):
    """
    Wind-triangle pseudo-measurement Jacobian.

    H_w = dh_w/dx
    """

    pn, pe, Vg, chi, wn, we, psi = x
    Va, q, r, phi, theta = u

    H = np.array([
        [
            0.0,
            0.0,
            -np.cos(chi),
            Vg * np.sin(chi),
            1.0,
            0.0,
            -Va * np.sin(psi)
        ],
        [
            0.0,
            0.0,
            -np.sin(chi),
            -Vg * np.cos(chi),
            0.0,
            1.0,
            Va * np.cos(psi)
        ]
    ])

    return H


# ============================================================
# Simulation settings
# ============================================================

np.random.seed(10)

dt = 0.02
t_final = 40.0
t = np.arange(0.0, t_final + dt, dt)
N = len(t)

gps_dt = 0.5
gps_step = int(gps_dt / dt)


# ============================================================
# Input signals
# ============================================================

Va_true = 17.0 + 0.3 * np.sin(0.10 * t)
theta_true = np.deg2rad(2.0) * np.ones(N)

phi_true = np.deg2rad(20.0) * np.sin(0.32 * t)

q_true = np.zeros(N)

r_true = (
    np.deg2rad(7.5) * np.sin(0.28 * t)
    + np.deg2rad(3.0) * np.sin(0.11 * t)
)

u_true = np.zeros((N, 5))
u_true[:, 0] = Va_true
u_true[:, 1] = q_true
u_true[:, 2] = r_true
u_true[:, 3] = phi_true
u_true[:, 4] = theta_true


# ============================================================
# Generate true navigation trajectory
#
# Important:
# Vg and chi are generated from the wind triangle.
# Therefore, the true trajectory is physically consistent with:
#
# Vg cos(chi) = Va cos(psi) + wn
# Vg sin(chi) = Va sin(psi) + we
# ============================================================

x_true = np.zeros((N, 7))

pn = 0.0
pe = 0.0
psi = np.deg2rad(0.0)

wn = 3.0
we = -1.5

Vgn = Va_true[0] * np.cos(psi) + wn
Vge = Va_true[0] * np.sin(psi) + we

Vg = np.sqrt(Vgn**2 + Vge**2)
chi = np.arctan2(Vge, Vgn)

x_true[0, :] = np.array([
    pn,
    pe,
    Vg,
    chi,
    wn,
    we,
    psi
])

for k in range(1, N):

    Va = Va_true[k - 1]
    q = q_true[k - 1]
    r = r_true[k - 1]
    phi = phi_true[k - 1]
    theta = theta_true[k - 1]

    cos_theta = np.cos(theta)
    if abs(cos_theta) < 0.1:
        cos_theta = 0.1 * np.sign(cos_theta) if cos_theta != 0 else 0.1

    # Heading propagation
    psi_dot = q * np.sin(phi) / cos_theta + r * np.cos(phi) / cos_theta
    psi = wrap_pi(psi + dt * psi_dot)

    # Slowly varying true wind components
    wn = 3.0 + 0.20 * np.sin(0.05 * t[k])
    we = -1.5 + 0.15 * np.cos(0.04 * t[k])

    # Wind triangle
    Vgn = Va_true[k] * np.cos(psi) + wn
    Vge = Va_true[k] * np.sin(psi) + we

    Vg = np.sqrt(Vgn**2 + Vge**2)
    chi = np.arctan2(Vge, Vgn)

    # Position propagation
    pn = pn + dt * Vgn
    pe = pe + dt * Vge

    x_true[k, :] = np.array([
        pn,
        pe,
        Vg,
        chi,
        wn,
        we,
        psi
    ])


# ============================================================
# Generate noisy GPS measurements
#
# GPS measurement:
#
# z_GPS = [pn_GPS, pe_GPS, Vg_GPS, chi_GPS]^T
# ============================================================

gps_available = np.zeros(N, dtype=bool)
gps_available[::gps_step] = True

z_gps = np.full((N, 4), np.nan)

sigma_pn = 1.0
sigma_pe = 1.0
sigma_Vg = 0.25
sigma_chi = np.deg2rad(1.5)

for k in range(N):
    if gps_available[k]:

        z_gps[k, 0] = x_true[k, 0] + np.random.randn() * sigma_pn
        z_gps[k, 1] = x_true[k, 1] + np.random.randn() * sigma_pe
        z_gps[k, 2] = x_true[k, 2] + np.random.randn() * sigma_Vg
        z_gps[k, 3] = wrap_pi(x_true[k, 3] + np.random.randn() * sigma_chi)


# ============================================================
# EKF initialization
# ============================================================

x_est = np.zeros((N, 7))

# Initial estimate from first GPS sample
x_est[0, :] = np.array([
    z_gps[0, 0],     # pn
    z_gps[0, 1],     # pe
    z_gps[0, 2],     # Vg
    z_gps[0, 3],     # chi
    0.0,             # wn initial guess
    0.0,             # we initial guess
    z_gps[0, 3]      # psi initial guess approximately chi
])

# Initial covariance
P = np.diag([
    4.0**2,                  # pn
    4.0**2,                  # pe
    1.5**2,                  # Vg
    np.deg2rad(10.0)**2,     # chi
    8.0**2,                  # wn
    8.0**2,                  # we
    np.deg2rad(25.0)**2      # psi
])

# Process noise covariance
Q = np.diag([
    0.03**2,                 # pn
    0.03**2,                 # pe
    0.08**2,                 # Vg
    np.deg2rad(0.3)**2,      # chi
    0.025**2,                # wn
    0.025**2,                # we
    np.deg2rad(0.3)**2       # psi
])

# GPS measurement noise covariance
R_gps = np.diag([
    sigma_pn**2,
    sigma_pe**2,
    sigma_Vg**2,
    sigma_chi**2
])

# Wind-triangle pseudo-measurement covariance
R_w = np.diag([
    0.25**2,
    0.25**2
])

# GPS measurement Jacobian
H_gps = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
])

I7 = np.eye(7)


# ============================================================
# EKF loop
#
# Step 1: Prediction
# Step 2: GPS measurement correction
# Step 3: Wind-triangle pseudo-measurement correction
# ============================================================

for k in range(1, N):

    x_prev = x_est[k - 1, :].copy()
    u_prev = u_true[k - 1, :].copy()

    # --------------------------------------------------------
    # Step 1: Prediction
    # --------------------------------------------------------

    x_dot = process_model(x_prev, u_prev)

    x_minus = x_prev + dt * x_dot
    x_minus[3] = wrap_pi(x_minus[3])
    x_minus[6] = wrap_pi(x_minus[6])

    A_cont = numerical_jacobian(process_model, x_prev, u_prev)
    A_discrete = I7 + dt * A_cont

    P_minus = A_discrete @ P @ A_discrete.T + Q

    # --------------------------------------------------------
    # Step 2: GPS measurement correction
    # --------------------------------------------------------

    x_plus = x_minus.copy()
    P_plus = P_minus.copy()

    if gps_available[k]:

        z = z_gps[k, :].copy()
        z_hat = h_gps(x_minus)

        innovation_gps = z - z_hat
        innovation_gps[3] = wrap_pi(innovation_gps[3])

        S_gps = H_gps @ P_minus @ H_gps.T + R_gps
        K_gps = P_minus @ H_gps.T @ np.linalg.inv(S_gps)

        x_plus = x_minus + K_gps @ innovation_gps
        x_plus[3] = wrap_pi(x_plus[3])
        x_plus[6] = wrap_pi(x_plus[6])

        # Joseph covariance update
        P_plus = (
            (I7 - K_gps @ H_gps)
            @ P_minus
            @ (I7 - K_gps @ H_gps).T
            + K_gps @ R_gps @ K_gps.T
        )

    # --------------------------------------------------------
    # Step 3: Wind-triangle pseudo-measurement correction
    # --------------------------------------------------------

    u_now = u_true[k, :].copy()

    z_w = np.array([0.0, 0.0])
    h_w = h_wind_triangle(x_plus, u_now)

    innovation_w = z_w - h_w

    H_w = H_wind_triangle(x_plus, u_now)

    S_w = H_w @ P_plus @ H_w.T + R_w
    K_w = P_plus @ H_w.T @ np.linalg.inv(S_w)

    x_final = x_plus + K_w @ innovation_w
    x_final[3] = wrap_pi(x_final[3])
    x_final[6] = wrap_pi(x_final[6])

    # Joseph covariance update
    P = (
        (I7 - K_w @ H_w)
        @ P_plus
        @ (I7 - K_w @ H_w).T
        + K_w @ R_w @ K_w.T
    )

    # Symmetrize covariance
    P = 0.5 * (P + P.T)

    x_est[k, :] = x_final


# ============================================================
# Compute plotting variables
# ============================================================

pn_error = x_est[:, 0] - x_true[:, 0]
pe_error = x_est[:, 1] - x_true[:, 1]

Vg_true = x_true[:, 2]
Vg_est = x_est[:, 2]

chi_true_deg = np.rad2deg(x_true[:, 3])
chi_est_deg = np.rad2deg(x_est[:, 3])

wn_true = x_true[:, 4]
wn_est = x_est[:, 4]

we_true = x_true[:, 5]
we_est = x_est[:, 5]

psi_true_deg = np.rad2deg(x_true[:, 6])
psi_est_deg = np.rad2deg(x_est[:, 6])


# ============================================================
# RMSE values
# ============================================================

rmse_pn = np.sqrt(np.mean(pn_error**2))
rmse_pe = np.sqrt(np.mean(pe_error**2))
rmse_Vg = np.sqrt(np.mean((Vg_est - Vg_true)**2))
rmse_chi = np.rad2deg(np.sqrt(np.mean(wrap_pi(x_est[:, 3] - x_true[:, 3])**2)))
rmse_wn = np.sqrt(np.mean((wn_est - wn_true)**2))
rmse_we = np.sqrt(np.mean((we_est - we_true)**2))
rmse_psi = np.rad2deg(np.sqrt(np.mean(wrap_pi(x_est[:, 6] - x_true[:, 6])**2)))

print("\n================ RMSE Results ================")
print(f"pn error RMSE  = {rmse_pn:.3f} m")
print(f"pe error RMSE  = {rmse_pe:.3f} m")
print(f"Vg RMSE        = {rmse_Vg:.3f} m/s")
print(f"chi RMSE       = {rmse_chi:.3f} deg")
print(f"wn RMSE        = {rmse_wn:.3f} m/s")
print(f"we RMSE        = {rmse_we:.3f} m/s")
print(f"psi RMSE       = {rmse_psi:.3f} deg")
print("==============================================\n")


# ============================================================
# Plot Figure
#
# True      : black solid
# Estimated : red dashed
# Errors    : red dashed, with black zero reference
# ============================================================

fig, axs = plt.subplots(4, 2, figsize=(12, 11))
axs = axs.flatten()

# ---------- (a) North position error ----------
axs[0].plot(t, np.zeros_like(t), color="black", linestyle="-", label="Zero Reference")
axs[0].plot(t, pn_error, color="red", linestyle="--", label="Estimated Error")
axs[0].set_ylabel("North Error (m)")
axs[0].set_xlabel("Time (s)")
axs[0].set_title("(a) North Position Error")
axs[0].grid(True, linestyle="--", alpha=0.6)

# ---------- (b) East position error ----------
axs[1].plot(t, np.zeros_like(t), color="black", linestyle="-", label="Zero Reference")
axs[1].plot(t, pe_error, color="red", linestyle="--", label="Estimated Error")
axs[1].set_ylabel("East Error (m)")
axs[1].set_xlabel("Time (s)")
axs[1].set_title("(b) East Position Error")
axs[1].grid(True, linestyle="--", alpha=0.6)

# ---------- (c) Ground speed ----------
axs[2].plot(t, Vg_true, color="black", linestyle="-", label="True")
axs[2].plot(t, Vg_est, color="red", linestyle="--", label="Estimated")
axs[2].set_ylabel(r"$V_g$ (m/s)")
axs[2].set_xlabel("Time (s)")
axs[2].set_title("(c) Ground Speed")
axs[2].grid(True, linestyle="--", alpha=0.6)

# ---------- (d) Course angle ----------
axs[3].plot(t, chi_true_deg, color="black", linestyle="-", label="True")
axs[3].plot(t, chi_est_deg, color="red", linestyle="--", label="Estimated")
axs[3].set_ylabel(r"$\chi$ (deg)")
axs[3].set_xlabel("Time (s)")
axs[3].set_title("(d) Course Angle")
axs[3].grid(True, linestyle="--", alpha=0.6)

# ---------- (e) North wind component ----------
axs[4].plot(t, wn_true, color="black", linestyle="-", label="True")
axs[4].plot(t, wn_est, color="red", linestyle="--", label="Estimated")
axs[4].set_ylabel(r"$w_n$ (m/s)")
axs[4].set_xlabel("Time (s)")
axs[4].set_title("(e) North Wind Component")
axs[4].grid(True, linestyle="--", alpha=0.6)

# ---------- (f) East wind component ----------
axs[5].plot(t, we_true, color="black", linestyle="-", label="True")
axs[5].plot(t, we_est, color="red", linestyle="--", label="Estimated")
axs[5].set_ylabel(r"$w_e$ (m/s)")
axs[5].set_xlabel("Time (s)")
axs[5].set_title("(f) East Wind Component")
axs[5].grid(True, linestyle="--", alpha=0.6)

# ---------- (g) Heading angle ----------
axs[6].plot(t, psi_true_deg, color="black", linestyle="-", label="True")
axs[6].plot(t, psi_est_deg, color="red", linestyle="--", label="Estimated")
axs[6].set_ylabel(r"$\psi$ (deg)")
axs[6].set_xlabel("Time (s)")
axs[6].set_title("(g) Heading Angle")
axs[6].grid(True, linestyle="--", alpha=0.6)

# ---------- Legend panel ----------
axs[7].axis("off")
axs[7].plot([], [], color="black", linestyle="-", label="True / Zero Reference")
axs[7].plot([], [], color="red", linestyle="--", label="Estimated")
axs[7].legend(loc="center", frameon=True)

fig.suptitle(
    "Navigation-State Estimation Results Obtained from the GPS-Smoothing EKF",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(
    "figure_63_navigation_ekf_times_new_roman.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()