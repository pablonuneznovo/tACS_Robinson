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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import scienceplots
from scipy.io import loadmat
from scipy.stats import spearmanr
import warnings
from reproduction_config import CRSR_XLSX, FITTED_DIR, FIGURES_DIR

warnings.filterwarnings("ignore")

# =============================================================================
# 1. SCIENCEPLOTS FORMATTING (BIGGER FONTS)
# =============================================================================
plt.style.use(['science', 'no-latex'])

# --- BUMPED UP FONT SIZES ---
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 15,  # Global font size
    'axes.titlesize': 18,  # Subplot titles
    'axes.labelsize': 16,  # X and Y axis labels
    'xtick.labelsize': 14,  # Tick labels
    'ytick.labelsize': 14,  # Tick labels
    'legend.fontsize': 14,
    'mathtext.default': 'regular'  # Forces math text ($...$) to use the standard font
})
plt.rcParams['axes.grid'] = False
plt.rcParams['pdf.fonttype'] = 42

# Greek symbol and LaTeX mapping for labels
GREEK_MAP = {
    'Alpha': 'α',
    'Beta': 'β',
    'Gamma': 'γ',
    'Delta': 'δ',
    'Theta': 'θ',
    'Rho': 'ρ',
    'X': r'$S_{CC}$',
    'Y': r'$S_{CT}$',
    'Z': r'$S_{IT}$'
}

# =============================================================================
# 2. LOAD DATA
# =============================================================================
data_dir = str(FITTED_DIR)
crsr_file = str(CRSR_XLSX)

mat_files = sorted(glob.glob(os.path.join(data_dir, '*_model_fits.mat')))
if not mat_files:
    raise FileNotFoundError(f"No fitted result files found in {data_dir}")
S = len(mat_files)
T, R = 7, 4

# Extract parameter names
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
# 3. LOAD & CLEAN CRS-R DATA
# =============================================================================
crsr_total = np.full((S, 7, 2), np.nan)
crsr_subscales = np.full((S, 7, 2, 6), np.nan)
subscale_names = ['Auditory', 'Visual', 'Motor', 'Oromotor', 'Communication', 'Arousal']

print("Parsing Clinical Excel Data...")
df_crsr = pd.read_excel(crsr_file, header=None)

last_valid_case = np.nan
last_valid_sess = np.nan

for idx, row in df_crsr.iterrows():
    try:
        num_case = float(row[0])
        if not np.isnan(num_case): last_valid_case = num_case
    except:
        pass

    try:
        num_sess = float(row[1])
        if not np.isnan(num_sess): last_valid_sess = num_sess
    except:
        pass

    if np.isnan(last_valid_case): continue

    s_idx = [i for i, sid in enumerate(sub_ids) if str(int(last_valid_case)) in sid]
    if not s_idx: continue
    s = s_idx[0]
    sess = last_valid_sess
    tp = str(row[3]).strip().lower()

    try:
        score = float(row[10])
    except:
        score = np.nan

    subs = np.full(6, np.nan)
    for c in range(6):
        try:
            subs[c] = float(row[4 + c])
        except:
            pass

    if 'pre' in tp:
        r_idx, sess_idx = 0, int(sess) - 1
    elif 'post' in tp and 'week' not in tp and 'protocol' not in tp:
        r_idx, sess_idx = 1, int(sess) - 1
    elif 'protocol' in tp:
        r_idx, sess_idx = 0, 5
    elif 'week' in tp:
        r_idx, sess_idx = 0, 6
    else:
        continue

    if 0 <= sess_idx < 7:
        crsr_total[s, sess_idx, r_idx] = score
        crsr_subscales[s, sess_idx, r_idx, :] = subs

# =============================================================================
# 4. FIGURE 3 SETUP: GRIDSPEC LAYOUT
# =============================================================================
print("\n=== GENERATING MASTER FIGURE 3 ===")

# Create a large figure to hold everything comfortably
fig = plt.figure(figsize=(16, 12))

# Create a 3x3 Grid
# Row 0: X, Y, Z
# Row 1: Alpha, (Heatmap span), (Heatmap span)
# Row 2: Beta,  (Heatmap span), (Heatmap span)
gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 1.2, 1.2])

# Assign Axes according to the grid
ax_x = fig.add_subplot(gs[0, 0])
ax_y = fig.add_subplot(gs[0, 1])
ax_z = fig.add_subplot(gs[0, 2])
ax_alpha = fig.add_subplot(gs[1, 0])
ax_beta = fig.add_subplot(gs[2, 0])
ax_heat = fig.add_subplot(gs[1:3, 1:3])  # Heatmap spans rows 1-2 and cols 1-2

axes_scat = [ax_x, ax_y, ax_z, ax_alpha, ax_beta]
target_params = ['X', 'Y', 'Z', 'Alpha', 'Beta']

# Map target names to their actual indices in the data
param_indices = []
for tp in target_params:
    for idx, name in enumerate(pnames):
        if tp.lower() == name.lower():
            param_indices.append(idx)
            break

# =============================================================================
# 5. PLOT PANEL A: SCATTER PLOTS
# =============================================================================
c_flat = crsr_total[:, :, 0].flatten()

for i, ax in enumerate(axes_scat):
    p_idx = param_indices[i]
    raw_name = pnames[p_idx]
    p_disp = GREEK_MAP.get(raw_name, raw_name)

    p_flat = params_full[:, :, 0, p_idx].flatten()
    mask = ~np.isnan(p_flat) & ~np.isnan(c_flat)
    p_v, c_v = p_flat[mask], c_flat[mask]

    if len(p_v) > 3:
        # Scatter Points
        ax.scatter(p_v, c_v, s=90, facecolor='#d3d3d3', edgecolor='black', alpha=0.8, zorder=2)
        rho, pval = spearmanr(p_v, c_v)

        # Regression Line
        pf = np.polyfit(p_v, c_v, 1)
        x_fit = np.linspace(np.min(p_v), np.max(p_v), 100)
        ax.plot(x_fit, np.polyval(pf, x_fit), 'r-', linewidth=2.5, zorder=3)

        # Formatted p-value text
        pval_str = "< 0.001" if pval < 0.001 else f"= {pval:.3f}"

        # Stat Box
        bbox_props = dict(boxstyle="square,pad=0.4", fc="white", ec="black", lw=1.0)
        stat_text = f"Spearman ρ = {rho:.3f}\np-value {pval_str}"
        ax.text(0.05, 0.05, stat_text, transform=ax.transAxes, fontsize=13,
                verticalalignment='bottom', bbox=bbox_props, fontweight='bold')

    # Formatting (Grid removed, label shortened)
    ax.grid(False)
    ax.set_xlabel(f'Baseline {p_disp}', fontweight='bold')
    ax.set_ylabel('Baseline CRS-R score', fontweight='bold')

    # Despine top and right for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add "A" Label higher up
ax_x.text(-0.25, 1.20, 'A', transform=ax_x.transAxes, fontsize=26, fontweight='bold', va='top')

# =============================================================================
# 6. PLOT PANEL B: HEATMAP
# =============================================================================
corr_matrix = np.full((len(target_params), 6), np.nan)
pval_matrix = np.full((len(target_params), 6), np.nan)

# Collect data strictly for the 5 target parameters
for i, p_idx in enumerate(param_indices):
    p_flat = params_full[:, :, 0, p_idx].flatten()
    for sub in range(6):
        s_flat = crsr_subscales[:, :, 0, sub].flatten()
        mask = ~np.isnan(p_flat) & ~np.isnan(s_flat)
        if np.sum(mask) > 5:
            r, pv = spearmanr(p_flat[mask], s_flat[mask])
            corr_matrix[i, sub] = r
            pval_matrix[i, sub] = pv

# Custom Blue-White-Red Colormap
colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
bwr_cmap = LinearSegmentedColormap.from_list('bwr_custom', colors, N=256)

cax = ax_heat.imshow(corr_matrix, cmap=bwr_cmap, vmin=-0.6, vmax=0.6, aspect='auto')
cbar = fig.colorbar(cax, ax=ax_heat, fraction=0.046, pad=0.04)
cbar.set_label("Spearman correlation (ρ)", fontweight='bold', fontsize=16)

# Axes formatting
ax_heat.set_xticks(np.arange(6))
ax_heat.set_yticks(np.arange(len(target_params)))
ax_heat.set_xticklabels(subscale_names, rotation=45, ha='right', fontweight='bold')

# Map y-axis labels to Greek/Display names
heat_labels = [GREEK_MAP.get(tp, tp) for tp in target_params]
ax_heat.set_yticklabels(heat_labels, fontweight='bold')

# Overlay significance text
for i in range(len(target_params)):
    for j in range(6):
        val, pval = corr_matrix[i, j], pval_matrix[i, j]
        if not np.isnan(val):
            stars = ""
            if pval < 0.001:
                stars = "***"
            elif pval < 0.01:
                stars = "**"
            elif pval < 0.05:
                stars = "*"

            txt = f"{val:.2f}{stars}"
            # Turn text white if background is too dark for contrast
            text_col = 'white' if abs(val) > 0.4 else 'black'
            ax_heat.text(j, i, txt, ha='center', va='center', color=text_col,
                         fontweight='bold', fontsize=14)

# Add title and "B" Label closer to the heatmap
ax_heat.text(-0.05, 1.08, 'B', transform=ax_heat.transAxes, fontsize=26, fontweight='bold', va='top')

# =============================================================================
# 7. FINAL LAYOUT & SAVE
# =============================================================================
plt.tight_layout()
fig.subplots_adjust(wspace=0.35, hspace=0.45)

print("Figure 3 successfully generated. Saving high-res version...")
plt.savefig(os.path.join(FIGURES_DIR, 'Figure_3_HighRes.png'), dpi=200)

print("Save complete. Displaying on screen...")
plt.show()
