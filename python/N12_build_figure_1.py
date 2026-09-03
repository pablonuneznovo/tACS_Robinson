import os
import glob
import numpy as np
import pandas as pd
import matplotlib

if os.environ.get('SHOW_FIGURE', '1') == '0':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Imported for the rectangle legend
import scienceplots  # Imports the science plotting styles
from scipy.io import loadmat
from scipy.stats import mannwhitneyu
import warnings
from reproduction_config import FITTED_DIR, FIGURES_DIR

# Suppress mean of empty slice warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. SCIENCEPLOTS FORMATTING (CHANGE FONT SIZES HERE)
# =============================================================================
plt.style.use(['science', 'no-latex'])

# --- CHANGE THESE VALUES TO SCALE ALL FONTS IN THE FIGURE INSTANTLY ---
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,  # Global font size
    'axes.titlesize': 21,  # Title size (A, B, C)
    'axes.labelsize': 14,  # Axis label size
    'xtick.labelsize': 12,  # X-axis tick labels
    'ytick.labelsize': 12,  # Y-axis tick labels
    'legend.fontsize': 16,  # Global legend font size
    'mathtext.default': 'regular'  # Forces math text ($...$) to use the standard font
})
# ----------------------------------------------------------------------

plt.rcParams['axes.grid'] = False
plt.rcParams['pdf.fonttype'] = 42

# Soft, pastel colors restored
COLOR_ACT = '#A1C9F4'  # Pastel Blue
COLOR_SHM = '#FF9F9B'  # Pastel Red
COLORS = {"Active": COLOR_ACT, "Sham": COLOR_SHM}

# Unicode mapping
GREEK_MAP = {
    'Alpha': 'α',
    'Beta': 'β',
    'Gamma': 'γ',
    'Delta': 'δ',
    'Theta': 'θ',
    'Rho': 'ρ',
    'EMG': 'EMG',
    'PEMG': 'EMG',
    'p_emg': 'EMG'
}

# =============================================================================
# 2. DATA LOADING & PREPARATION
# =============================================================================
data_dir = str(FITTED_DIR)
mat_files = sorted(glob.glob(os.path.join(data_dir, '*_model_fits.mat')))
if not mat_files:
    raise FileNotFoundError(f"No fitted result files found in {data_dir}")
S = len(mat_files)
T = 7
R = 4

active_idx = [0, 1, 2, 3, 4, 5]
sham_idx = [6, 7, 8, 9, 10]

timeline_labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', '24h', '1 week']
run_for_evo = 0  # 0 corresponds to 'Pre'

sample_data = loadmat(mat_files[0])
raw_names = sample_data['param_names'].flatten()
pnames = [str(item[0]) if isinstance(item, (np.ndarray, list)) and len(item) > 0 else str(item) for item in raw_names]
P = len(pnames)

params_full = np.full((S, T, R, P), np.nan)
sub_ids = []

for s, file_path in enumerate(mat_files):
    mat = loadmat(file_path)
    dat = mat['params_sub']
    params_full[s, :dat.shape[0], :dat.shape[1], :] = dat
    sub_ids.append(os.path.basename(file_path).replace('_model_fits.mat', ''))

print(f"Loaded {S} subjects with {P} parameters.")


# =============================================================================
# 3. HELPER FUNCTION: SCIENCE-STYLE BOXPLOTS
# =============================================================================
def draw_science_boxplots(ax, df, x_col, y_col, hue_col, x_order):
    box_width = 0.35
    line_weight = 1.2

    if hue_col:
        hue_order = ["Active", "Sham"]
        n_hues = len(hue_order)
        for i, x_val in enumerate(x_order):
            for j, hue_val in enumerate(hue_order):
                offset = (j - (n_hues - 1) / 2) * (box_width + 0.05)
                pos = i + offset
                subset = df[(df[x_col] == x_val) & (df[hue_col] == hue_val)][y_col].dropna().values

                if len(subset) > 0:
                    bp = ax.boxplot([subset], positions=[pos], widths=box_width,
                                    patch_artist=True, showfliers=False, manage_ticks=False)
                    for patch in bp['boxes']:
                        patch.set_facecolor(COLORS[hue_val])
                        patch.set_alpha(0.85)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(line_weight)
                    for median in bp['medians']:
                        median.set_color('black')
                        median.set_linewidth(line_weight)
                    for whisker, cap in zip(bp['whiskers'], bp['caps']):
                        whisker.set_color('black')
                        whisker.set_linewidth(line_weight)
                        cap.set_color('black')
                        cap.set_linewidth(line_weight)

                    jitter = np.random.uniform(-box_width / 4, box_width / 4, size=len(subset))
                    ax.scatter(np.full_like(subset, pos) + jitter, subset,
                               facecolors=COLORS[hue_val], edgecolors='black',
                               alpha=0.8, s=35, zorder=3, linewidth=0.8)
    else:
        for i, x_val in enumerate(x_order):
            pos = i
            subset = df[df[x_col] == x_val][y_col].dropna().values
            color = COLORS.get(x_val, '#cccccc')

            if len(subset) > 0:
                bp = ax.boxplot([subset], positions=[pos], widths=box_width,
                                patch_artist=True, showfliers=False, manage_ticks=False)
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.85)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(line_weight)
                for median in bp['medians']:
                    median.set_color('black')
                    median.set_linewidth(line_weight)
                for whisker, cap in zip(bp['whiskers'], bp['caps']):
                    whisker.set_color('black')
                    whisker.set_linewidth(line_weight)
                    cap.set_color('black')
                    cap.set_linewidth(line_weight)

                jitter = np.random.uniform(-box_width / 4, box_width / 4, size=len(subset))
                ax.scatter(np.full_like(subset, pos) + jitter, subset,
                           facecolors=color, edgecolors='black',
                           alpha=0.8, s=35, zorder=3, linewidth=0.8)

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order)
    ax.set_xlim(-0.5, len(x_order) - 0.5)


# =============================================================================
# 4. BUILD THE MASTER FIGURE 1 (6 Rows x 3 Columns)
# =============================================================================
print("\n=== GENERATING MASTER FIGURE 1 ===")

target_params = ['X', 'Y', 'Z', 'Alpha', 'Beta', 'EMG']
param_indices = []

for tp in target_params:
    for idx, name in enumerate(pnames):
        if tp.lower() in name.lower() or name.lower() in ['pemg', 'p_emg']:
            param_indices.append((idx, name))
            break

# Create subplots with custom width ratios: Col A is wider, Col C is narrower
fig, axs = plt.subplots(len(param_indices), 3, figsize=(14.7, 17.3),
                        gridspec_kw={'width_ratios': [1.5, 1.0, 0.6]})

for row_idx, (p, raw_name) in enumerate(param_indices):
    param_disp = GREEK_MAP.get(raw_name, raw_name)

    # Rename X, Y, and Z for plotting purposes
    if raw_name.strip().upper() == 'X':
        param_disp = r'$S_{CC}$'
    elif raw_name.strip().upper() == 'Y':
        param_disp = r'$S_{CT}$'
    elif raw_name.strip().upper() == 'Z':
        param_disp = r'$S_{IT}$'
    elif 'emg' in raw_name.lower():
        param_disp = 'EMG'

    records_evo, records_long = [], []
    valid_t = [t for t in range(T) if not np.all(np.isnan(params_full[:, t, run_for_evo, p]))]
    valid_timeline = [timeline_labels[t] for t in valid_t]

    for s in range(S):
        grp = "Active" if s in active_idx else "Sham"

        # Col A Data
        for t in valid_t:
            val = params_full[s, t, run_for_evo, p]
            if not np.isnan(val):
                records_evo.append({'Timepoint': timeline_labels[t], 'Value': val, 'Group': grp})

        # Col C Data
        v_day1 = params_full[s, 0, run_for_evo, p]
        v_1wk = params_full[s, 6, run_for_evo, p]
        if not np.isnan(v_day1) and not np.isnan(v_1wk):
            records_long.append({'Value': v_1wk - v_day1, 'Group': grp})

    df_evo = pd.DataFrame(records_evo)
    df_long = pd.DataFrame(records_long)

    # -------------------------------------------------------------------------
    # COLUMN A: Temporal Evolution (Boxplots)
    # -------------------------------------------------------------------------
    ax_a = axs[row_idx, 0]
    if not df_evo.empty:
        draw_science_boxplots(ax_a, df_evo, 'Timepoint', 'Value', 'Group', valid_timeline)

    ax_a.set_ylabel(f'{param_disp}', fontweight='bold', fontsize=22)

    if row_idx == 0:
        ax_a.set_title('A', loc='left', fontweight='bold')

    # -------------------------------------------------------------------------
    # COLUMN B: Acute Shift Line Plot (Post - Pre)
    # -------------------------------------------------------------------------
    ax_b = axs[row_idx, 1]
    daily_delta = params_full[:, 0:5, 1, p] - params_full[:, 0:5, 0, p]
    mean_act = np.nanmean(daily_delta[active_idx, :], axis=0)
    mean_shm = np.nanmean(daily_delta[sham_idx, :], axis=0)
    x_days = np.arange(1, 6)

    ax_b.plot(x_days, mean_act, '-o', color=COLOR_ACT, markeredgecolor='black',
              linewidth=2.5, markersize=8, label='Active')
    ax_b.plot(x_days, mean_shm, '-s', color=COLOR_SHM, markeredgecolor='black',
              linewidth=2.5, markersize=8, label='Sham')
    ax_b.axhline(0, color='black', linestyle='--', linewidth=1.2)

    ax_b.set_xticks(x_days)
    ax_b.set_xticklabels(['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'])
    ax_b.set_ylabel(f'Δ (Post - Pre)')

    if row_idx == 0:
        ax_b.set_title('B', loc='left', fontweight='bold')

    # -------------------------------------------------------------------------
    # COLUMN C: Long-term Absolute Change (Boxplots)
    # -------------------------------------------------------------------------
    ax_c = axs[row_idx, 2]
    if not df_long.empty:
        draw_science_boxplots(ax_c, df_long, 'Group', 'Value', None, ['Active', 'Sham'])
        ax_c.axhline(0, color='black', linestyle='--', linewidth=1.2)

        # Calculate Stats
        active_vals = df_long[df_long['Group'] == 'Active']['Value'].dropna()
        sham_vals = df_long[df_long['Group'] == 'Sham']['Value'].dropna()
        if len(active_vals) > 0 and len(sham_vals) > 0:
            stat, p_val = mannwhitneyu(active_vals, sham_vals, alternative='two-sided')
            u_max = len(active_vals) * len(sham_vals)
            r_rb = abs(1 - (2 * stat) / u_max)

            # Add Stat Box matching paper format
            bbox_props = dict(boxstyle="square,pad=0.4", fc="white", ec="black", lw=1.0)
            stat_text = f"p-value={p_val:.4f}\n|r|={r_rb:.3f}"

            # --- CONDITIONAL TEXT BOX PLACEMENT ---
            if row_idx == 0:  # First row (X) -> Bottom Right
                x_pos, y_pos = 0.95, 0.05
                h_align, v_align = 'right', 'bottom'
            elif row_idx == 2:  # Third row (Z) -> Top Left
                x_pos, y_pos = 0.05, 0.95
                h_align, v_align = 'left', 'top'
            else:  # All other rows -> Top Right
                x_pos, y_pos = 0.95, 0.95
                h_align, v_align = 'right', 'top'

            # Reduced fontsize to 10 so it fits in the narrower column
            ax_c.text(x_pos, y_pos, stat_text, transform=ax_c.transAxes, fontsize=10,
                      verticalalignment=v_align, horizontalalignment=h_align,
                      bbox=bbox_props, fontweight='bold')

    ax_c.set_ylabel(f'Δ vs Day 1')

    if row_idx == 0:
        ax_c.set_title('C', loc='left', fontweight='bold')

# =============================================================================
# 5. GLOBAL FORMATTING: DESPINE (Un-box) ALL PLOTS
# =============================================================================
for ax in axs.flat:
    # Turn off top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Ensure ticks only appear on bottom and left
    ax.tick_params(top=False, right=False, which='both')

# =============================================================================
# 6. GLOBAL LEGEND & FINAL LAYOUT ADJUSTMENTS
# =============================================================================
legend_elements = [
    mpatches.Patch(facecolor=COLOR_ACT, edgecolor='black', linewidth=1.5, label='Active'),
    mpatches.Patch(facecolor=COLOR_SHM, edgecolor='black', linewidth=1.5, label='Sham')
]

plt.tight_layout()
fig.subplots_adjust(wspace=0.35, hspace=0.35, top=0.94)

fig.legend(handles=legend_elements, loc='lower center', ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 0.96), borderaxespad=0.)

print("Figure 1 successfully generated. Saving high-res version...")
plt.savefig(os.path.join(FIGURES_DIR, 'Figure_1_HighRes.png'), dpi=200)

print("Save complete. Displaying on screen...")
plt.show()
