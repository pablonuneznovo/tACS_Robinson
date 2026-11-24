import os
import numpy as np
import matplotlib
matplotlib.use('Agg' if os.environ.get('SHOW_FIGURE', '0') == '0' else 'TkAgg')
import matplotlib.pyplot as plt
import scienceplots
from scipy.integrate import odeint
from reproduction_config import FIGURES_DIR

# =============================================================================
# 1. FORMATTING & STYLE
# =============================================================================
plt.style.use(['science', 'no-latex'])

# --- BUMPED UP FONT SIZES TO MATCH PREVIOUS FIGURES ---
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'mathtext.default': 'regular'  # Forces math text to use the standard font
})

# CHANGED: Adjusted width from 11 to 8 to make the plot narrower
fig, ax = plt.subplots(figsize=(8, 6))
t_hours = np.linspace(0, 14 * 24, 5000)

# =============================================================================
# 2. PURE BCM SYSTEM (4 Coupled ODEs)
# =============================================================================
def pure_bcm(state, t, stim_signal, t_array, params):
    Ca, w_hat, W, V = state
    tau_ca, w_max, eta_0, z, C_max, theta_d, theta_p, slope = params

    stim = np.interp(t, t_array, stim_signal)

    # ODE 1: Calcium Envelope
    dCa = ((C_max * stim) - Ca) / tau_ca

    # Dual-Sigmoid BCM Functions
    sig_d = 1.0 / (1.0 + np.exp(-(Ca - theta_d) / slope))
    sig_p = 1.0 / (1.0 + np.exp(-(Ca - theta_p) / slope))

    Omega = 0.5 - 0.5 * sig_d + 1.0 * sig_p
    eta = eta_0 * sig_d

    # ODE 2: Biological Target Weight
    dw_hat = eta * (w_max * Omega - w_hat)

    # ODE 3 & 4: Second-Order Transduction Delay
    dW = V
    dV = (w_hat - W - 2 * z * V) / (z ** 2)

    return [dCa, dw_hat, dW, dV]


# =============================================================================
# 3. SIMULATION WRAPPER
# =============================================================================
def simulate_calcium_group(n_subjects, is_active):
    all_Y = []

    # Timeline: 5 sessions (Days 1 to 5)
    tACS_stimulus = np.zeros_like(t_hours)
    for day in range(5):
        t_start = (day + 1) * 24
        tACS_stimulus[(t_hours >= t_start) & (t_hours < t_start + 2)] = 1.0

    for _ in range(n_subjects):
        # CLINICAL BOUNDS: Start with deeply pathological negative baselines
        Y_0 = np.random.uniform(-0.15, -0.05)

        tau_ca = 0.5
        # Max capacity bounded to realistic clinical recovery magnitudes (~0.15 to 0.25 gain)
        w_max = np.random.uniform(0.15, 0.25)
        eta_0 = np.random.uniform(0.5, 0.8)
        z_del = np.random.uniform(36.0, 60.0)  # 1.5 to 2.5 days delay

        theta_d = 0.15
        theta_p = 0.50
        slope = 0.01

        C_max = np.random.uniform(0.7, 0.9) if is_active else np.random.uniform(0.01, 0.05)

        params = (tau_ca, w_max, eta_0, z_del, C_max, theta_d, theta_p, slope)

        sim = odeint(pure_bcm, [0.0, 0.0, 0.0, 0.0], t_hours, args=(tACS_stimulus, t_hours, params), hmax=1.0)

        W_sim = sim[:, 2]  # The consolidated structural weight

        # MACROSCOPIC Loop Gain = Baseline + Structural Consolidation
        all_Y.append(Y_0 + W_sim)

    return all_Y


# =============================================================================
# 4. PLOTTING
# =============================================================================
np.random.seed(42)
Y_act = simulate_calcium_group(8, True)
Y_shm = simulate_calcium_group(8, False)

for y in Y_act: ax.plot(t_hours / 24, y, color='#1973D9', alpha=0.15, lw=1.5)
for y in Y_shm: ax.plot(t_hours / 24, y, color='#E53333', alpha=0.15, lw=1.5)

ax.plot(t_hours / 24, np.mean(Y_act, axis=0), color='#1973D9', lw=4, label='Active group mean')
ax.plot(t_hours / 24, np.mean(Y_shm, axis=0), color='#E53333', lw=4, label='Sham group mean')

# Threshold line at Y = 0 (Commented out, but updated variable name for consistency if used)
# ax.axhline(0.0, color='black', linestyle='--', lw=2.0, alpha=0.8, label='Zero-Crossing Threshold ($S_{CT}=0$)')

for day in range(5):
    ax.axvspan(day + 1, day + 1 + (2 / 24), color='gray', alpha=0.2, lw=0)

ax.axvline(5.1, color='black', linestyle=':', lw=1.5, alpha=0.6)
ax.text(5.2, 0.15, 'End of protocol', fontsize=14, style='italic', transform=ax.get_xaxis_transform())
ax.axvline(12, color='black', linestyle=':', lw=1.5, alpha=0.6)
ax.text(9, 0.15, '1-week follow-up', fontsize=14, style='italic', transform=ax.get_xaxis_transform())

# ax.set_title('Macroscopic Loop Gain ($S_{CT}$): Consolidation of Pathological Baselines', pad=15, fontweight='bold')
ax.set_xlabel('Time (Days)', fontweight='bold')
ax.set_ylabel(r'Thalamocortical loop gain ($S_{CT}$)', fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# LEGEND: Upper right corner inside the plot
ax.legend(frameon=True, facecolor='white', edgecolor='black', loc='upper right')

# BOUNDED Y-AXIS to match clinical empirical limits (-0.2 to 0.2)
ax.set_xlim(0, 13)
ax.set_ylim(-0.20, 0.10)

plt.tight_layout()
print("Simulation generated. Saving as SM_Figure5.png...")
plt.savefig(FIGURES_DIR / 'SM_Figure5.png', dpi=300)
if os.environ.get('SHOW_FIGURE', '0') != '0':
    plt.show()
plt.close(fig)
