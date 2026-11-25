import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg' if os.environ.get('SHOW_FIGURE', '0') == '0' else 'TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe  # For trajectory outlines
import scienceplots
from scipy.io import loadmat
import warnings
from reproduction_config import CRSR_XLSX, FITTED_DIR, FIGURES_DIR

# Suppress mean of empty slice warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. SCIENCEPLOTS FORMATTING (DEJAVU SANS, ORIGINAL COLORS)
# =============================================================================
plt.style.use(['science', 'no-latex'])

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 20,  # Increased from 16
    'xtick.labelsize': 16,  # Increased from 14
    'ytick.labelsize': 16,  # Increased from 14
    'legend.fontsize': 14,
    'mathtext.default': 'regular'  # Forces math text to use the standard font
})

plt.rcParams['axes.grid'] = False
plt.rcParams['pdf.fonttype'] = 42

# --- ORIGINAL MATLAB COLORS (Converted to Bold Hex) ---
# DOC State
C_UWS = '#D92633'  # Original: [0.85 0.15 0.20]
C_MCSM = '#66B2F2'  # Original: [0.40 0.70 0.95]
C_MCSP = '#1A59BF'  # Original: [0.10 0.35 0.75]
C_EMCS = '#26A64D'  # Original: [0.15 0.65 0.30]
C_UNK = '#262626'  # Original: [0.15 0.15 0.15]

# =============================================================================
# 2. SETTINGS & DATA LOADING
# =============================================================================
data_dir = str(FITTED_DIR)
clin_file = str(CRSR_XLSX)

# ---> FILTER SPECIFIC PATIENTS HERE <---
# Add the exact subject IDs (as strings) you want to see.
# Leave the list empty [] to show ALL patients.
# Example: target_subjects = ['1', '4', '8']
target_subjects = []

mat_files = sorted(glob.glob(os.path.join(data_dir, '*_model_fits.mat')))
nSub = len(mat_files)

if nSub == 0:
    raise FileNotFoundError(f"No .mat files found in {data_dir}")

active_idx = [0, 1, 2, 3, 4, 5]
sham_idx = list(range(6, nSub))
run_idx = 0  # 0 = Pre

show_post24 = True
color_mode = 'doc_state'  # Options: 'default', 'doc_state'

# Use the clinical assessments reported in Supplementary Table T1.
# Sessions 1–5 use pre-stimulation assessments. Sessions 6 and 7 are the
# 24-hour and 1-week follow-ups, identified by session number rather than
# the spreadsheet timepoint label (which is inconsistent for case 13).
clin_data = pd.read_excel(clin_file, header=1)
clin_data[['Cases', 'Session']] = clin_data[['Cases', 'Session']].ffill()
clinical_rows = clin_data.loc[
    (clin_data['Session'].between(1, 5) & clin_data['Timepoint'].eq('pre'))
    | clin_data['Session'].isin([6, 7])
]
state_colors = {'UWS': C_UWS, 'MCS-': C_MCSM, 'MCS+': C_MCSP, 'EMCS': C_EMCS}
clinical_states = {}
for _, row in clinical_rows.iterrows():
    key = (str(int(row['Cases'])), int(row['Session']))
    if key in clinical_states:
        raise ValueError(f'Duplicate clinical assessment for case {key[0]}, session {key[1]}')
    state = row['Category']
    if pd.isna(state) or not str(state).strip():
        clinical_states[key] = None
    else:
        state = str(state).strip().upper()
        if state not in state_colors:
            raise ValueError(f'Unexpected diagnostic category {state!r} for {key}')
        clinical_states[key] = state

for file_path in mat_files:
    sub_id = str(int(re.search(r'sub-(\d+)', os.path.basename(file_path)).group(1)))
    for session in range(1, 8):
        if (sub_id, session) not in clinical_states:
            raise ValueError(f'No clinical assessment row for case {sub_id}, session {session}')
        if clinical_states[(sub_id, session)] is None:
            print(f'Unavailable clinical category: case {sub_id}, session {session}')

# Initialize Matrices
X_all = np.full((nSub, 7), np.nan)
Y_all = np.full((nSub, 7), np.nan)
eps_d = np.finfo(float).eps

# Parse Parameters
for s, file_path in enumerate(mat_files):
    mat = loadmat(file_path)
    raw_names = mat['param_names'].flatten()
    pnames = [str(item[0]) if isinstance(item, (np.ndarray, list)) and len(item) > 0 else str(item) for item in
              raw_names]
    pnames_lower = [n.lower() for n in pnames]

    dat = mat['params_sub']

    if 'x' in pnames_lower and 'y' in pnames_lower:
        X_all[s, :dat.shape[0]] = dat[:, run_idx, pnames_lower.index('x')]
        Y_all[s, :dat.shape[0]] = dat[:, run_idx, pnames_lower.index('y')]
    else:
        try:
            Gee = dat[:, run_idx, pnames_lower.index('gee')]
            Gei = dat[:, run_idx, pnames_lower.index('gei')]
            Gese = dat[:, run_idx, pnames_lower.index('gese')]
            Gesre = dat[:, run_idx, pnames_lower.index('gesre')]
            Gsrs = dat[:, run_idx, pnames_lower.index('gsrs')]

            denX = (1 - Gei)
            denY = (1 - Gsrs) * (1 - Gei)

            X_all[s, :dat.shape[0]] = (Gee / np.maximum(np.abs(denX), eps_d)) * np.sign(denX)
            Y_all[s, :dat.shape[0]] = ((Gese + Gesre) / np.maximum(np.abs(denY), eps_d)) * np.sign(denY)
        except ValueError:
            raise ValueError(f"Subject {s}: Could not find parameters to compute X/Y.")

# Derived Metrics
X_minus_Y = X_all - Y_all
X_plus_Y = X_all + Y_all


# =============================================================================
# 3. HELPER FUNCTIONS FOR COLORS & PLOTTING
# =============================================================================
def get_pt_color(subID, t, default_c):
    if color_mode == 'default':
        return default_c
    return state_colors.get(clinical_states[(subID, t)], C_UNK)


def plot_trajectories(ax, idx_group, x_data, y_data, default_colors, subFiles, target_subjects=None):
    # Setup path effect for the main trajectory so it stands out clearly
    outline = [pe.Stroke(linewidth=4, foreground='white'), pe.Normal()]

    for i, s in enumerate(idx_group):
        x_seq, y_seq = x_data[s, :], y_data[s, :]
        default_c = default_colors[i]

        match = re.search(r'(sub-\d+)', subFiles[s])
        subID = match.group(1).replace('sub-', '') if match else ''
        subID = str(int(subID)) if subID.isdigit() else subID

        # ---> THE FILTER <---
        # Skip this iteration if we have a target list and this subject isn't on it
        if target_subjects and subID not in target_subjects:
            continue

        line_c = '#9E9E9E' if color_mode == 'doc_state' else get_pt_color(subID, 1, default_c)

        # 1. Very Faint Trajectory Line (The "wandering" path)
        ax.plot(x_seq, y_seq, '-', color=line_c, alpha=0.3, linewidth=1.0, zorder=1)

        # 2. Small daily markers (Days 2-5), using the same colors as the legend
        for t in range(1, 5):
            if not np.isnan(x_seq[t]) and not np.isnan(y_seq[t]):
                c_pt = get_pt_color(subID, t + 1, default_c)
                ax.scatter(x_seq[t], y_seq[t], s=25, color=c_pt, edgecolors='none', zorder=2)

        # 3. Bold Main Vectors (Start -> Post24 -> End) with White Outline
        if not np.isnan(x_seq[0]) and not np.isnan(x_seq[6]):
            c_start = get_pt_color(subID, 1, default_c)
            c_end = get_pt_color(subID, 7, default_c)

            if show_post24 and not np.isnan(x_seq[5]):
                c_p24 = get_pt_color(subID, 6, default_c)
                # Thick line with white stroke
                ax.plot([x_seq[0], x_seq[5], x_seq[6]], [y_seq[0], y_seq[5], y_seq[6]],
                        '-', color=line_c, linewidth=2.0, zorder=3, path_effects=outline)

                # Markers with clear borders
                ax.scatter(x_seq[0], y_seq[0], s=80, color=c_start, edgecolors='black', lw=1, zorder=4)
                ax.scatter(x_seq[5], y_seq[5], s=100, color=c_p24, marker='d', edgecolors='black', lw=1, zorder=4)
                ax.scatter(x_seq[6], y_seq[6], s=110, color=c_end, marker='s', edgecolors='black', lw=1, zorder=4)
            else:
                ax.plot([x_seq[0], x_seq[6]], [y_seq[0], y_seq[6]],
                        '-', color=line_c, linewidth=2.0, zorder=3, path_effects=outline)
                ax.scatter(x_seq[0], y_seq[0], s=80, color=c_start, edgecolors='black', lw=1, zorder=4)
                ax.scatter(x_seq[6], y_seq[6], s=110, color=c_end, marker='s', edgecolors='black', lw=1, zorder=4)


# =============================================================================
# 4. PLOTTING THE PANELS (TIGHT LAYOUT WITH CORRECTED WSPACE)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

import matplotlib.cm as cm

cmap_act = [cm.Blues(x) for x in np.linspace(0.5, 0.9, len(active_idx))]
cmap_shm = [cm.Reds(x) for x in np.linspace(0.5, 0.9, len(sham_idx))]

lim_X = [0.65, 1.05]
lim_Y = [-0.15, 0.15]
lim_XmY = [0.55, 1.15]
lim_XpY = [0.80, 1.05]

# Panel 1: Active (X vs Y)
ax = axes[0, 0]
ax.plot([0.5, 1.2], [0.5, -0.2], 'k-', lw=1.5, zorder=0)  # S_CC + S_CT = 1
ax.plot([0.5, 1.1], [-0.23, 0.37], 'k--', lw=1.5, zorder=0)  # S_CC - S_CT = 0.73
plot_trajectories(ax, active_idx, X_all, Y_all, cmap_act, mat_files, target_subjects)
ax.set_title('Active Group', fontweight='bold', pad=8)
ax.set_ylabel(r'$S_{CT}$', fontweight='bold')
ax.set_xlabel(r'$S_{CC}$', fontweight='bold')
ax.set_xlim(lim_X)
ax.set_ylim(lim_Y)

# Panel 2: Sham (X vs Y)
ax = axes[0, 1]
ax.plot([0.5, 1.2], [0.5, -0.2], 'k-', lw=1.5, zorder=0)
ax.plot([0.5, 1.1], [-0.23, 0.37], 'k--', lw=1.5, zorder=0)
plot_trajectories(ax, sham_idx, X_all, Y_all, cmap_shm, mat_files, target_subjects)
ax.set_title('Sham Group', fontweight='bold', pad=8)
ax.set_ylabel(r'$S_{CT}$', fontweight='bold')
ax.set_xlabel(r'$S_{CC}$', fontweight='bold')
ax.set_xlim(lim_X)
ax.set_ylim(lim_Y)

# Panel 3: Active (X-Y vs X+Y)
ax = axes[1, 0]
ax.plot([0.3, 1.2], [1, 1], 'k-', lw=1.5, zorder=0)
ax.plot([0.73, 0.73], [0.4, 1.2], 'k--', lw=1.5, zorder=0)
plot_trajectories(ax, active_idx, X_minus_Y, X_plus_Y, cmap_act, mat_files, target_subjects)
ax.set_xlabel(r'$S_{CC} - S_{CT}$', fontweight='bold')
ax.set_ylabel(r'$S_{CC} + S_{CT}$', fontweight='bold')
ax.set_xlim(lim_XmY)
ax.set_ylim(lim_XpY)

# Panel 4: Sham (X-Y vs X+Y)
ax = axes[1, 1]
ax.plot([0.3, 1.2], [1, 1], 'k-', lw=1.5, zorder=0)
ax.plot([0.73, 0.73], [0.4, 1.2], 'k--', lw=1.5, zorder=0)
plot_trajectories(ax, sham_idx, X_minus_Y, X_plus_Y, cmap_shm, mat_files, target_subjects)
ax.set_xlabel(r'$S_{CC} - S_{CT}$', fontweight='bold')
ax.set_ylabel(r'$S_{CC} + S_{CT}$', fontweight='bold')
ax.set_xlim(lim_XmY)
ax.set_ylim(lim_XpY)

# =============================================================================
# 5. ELEGANT GLOBAL LEGEND & SPACING
# =============================================================================
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Session 1 (Start)', markerfacecolor='k', markeredgecolor='k',
           markersize=9),
]
if show_post24:
    legend_elements.append(
        Line2D([0], [0], marker='d', color='w', label='24h Post', markerfacecolor='k', markeredgecolor='k',
               markersize=10))
legend_elements.append(
    Line2D([0], [0], marker='s', color='w', label='1 Week Post (End)', markerfacecolor='k', markeredgecolor='k',
           markersize=9))

if color_mode == 'doc_state':
    legend_elements += [
        Line2D([0], [0], marker='o', color='w', label='UWS', markerfacecolor=C_UWS, markeredgecolor='k', markersize=9),
        Line2D([0], [0], marker='o', color='w', label='MCS-', markerfacecolor=C_MCSM, markeredgecolor='k',
               markersize=9),
        Line2D([0], [0], marker='o', color='w', label='MCS+', markerfacecolor=C_MCSP, markeredgecolor='k',
               markersize=9),
        Line2D([0], [0], marker='o', color='w', label='eMCS', markerfacecolor=C_EMCS, markeredgecolor='k',
               markersize=9),
        Line2D([0], [0], marker='o', color='w', label='Unavailable', markerfacecolor=C_UNK, markeredgecolor='k',
               markersize=9)
    ]
# Increased wspace from 0.15 to 0.30 to separate columns and avoid Y-axis clipping
plt.subplots_adjust(bottom=0.15, top=0.90, hspace=0.25, wspace=0.4)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))

print("State-space figure successfully generated. Saving high-res version...")
plt.savefig(FIGURES_DIR / 'Figure3.png', dpi=200)

if os.environ.get('SHOW_FIGURE', '0') != '0':
    plt.show()
plt.close(fig)
