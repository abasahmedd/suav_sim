import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # ==========================================================
    # Academic Plot Settings
    # ==========================================================
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    # ==========================================================
    # 1. Simulation Parameters
    # ==========================================================
    dt = 0.01          # Sampling time, Ts = 0.01 s
    t_end = 10.0       # Simulation duration
    time = np.arange(0.0, t_end, dt)
    n_steps = len(time)

    # ==========================================================
    # 2. True Aircraft State: Pitch Rate q
    # ==========================================================
    q_true = 0.5 * np.sin(2.0 * np.pi * 0.2 * time)

    # Add a small maneuver change after 5 seconds
    q_true[time > 5.0] += 0.3

    # ==========================================================
    # 3. Simulated Noisy Sensor Measurement
    # ==========================================================
    noise_std = 0.15
    np.random.seed(42)

    noise = np.random.normal(0.0, noise_std, n_steps)
    q_measured = q_true + noise

    # ==========================================================
    # 4. Exponential Low-Pass Filter
    # y[k+1] = exp(-a*Ts)y[k] + (1 - exp(-a*Ts))u[k]
    # ==========================================================
    a = 5.0                      # Filter bandwidth parameter, rad/s
    alpha_exp = np.exp(-a * dt)  # Exponential smoothing coefficient

    q_filtered = np.zeros(n_steps)
    q_filtered[0] = q_measured[0]

    for k in range(n_steps - 1):
        u_k = q_measured[k]
        y_k = q_filtered[k]

        q_filtered[k + 1] = alpha_exp * y_k + (1.0 - alpha_exp) * u_k

    # ==========================================================
    # 5. Plotting
    # ==========================================================
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.plot(
        time,
        q_measured,
        color="lightgray",
        linewidth=0.7,
        alpha=0.8,
        label=r"Noisy measurement, $u_k$"
    )

    ax.plot(
        time,
        q_true,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=r"True pitch rate, $q_{\mathrm{true}}$"
    )

    ax.plot(
        time,
        q_filtered,
        color="red",
        linestyle="-",
        linewidth=0.9,
        label=rf"Filtered signal, $y_k$"
    )

    ax.set_xlabel(r"Time, $t$ (s)")
    ax.set_ylabel(r"Pitch rate, $q$ (rad/s)")

    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    ax.tick_params(direction="in", length=4, width=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()

    # ==========================================================
    # 6. Save Figure to Required Folder
    # ==========================================================
    output_dir = Path(r"C:\Users\ASUS\Desktop\sim\chapter3\estimation\Examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_png = output_dir / "lpf_pitch_rate_academic.png"
   

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
  

    print(f"PNG figure saved to: {output_png}")



if __name__ == "__main__":
    main()