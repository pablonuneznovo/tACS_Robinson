import os
import warnings

import matplotlib

if os.environ.get("SHOW_FIGURE", "1") == "0":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")

import matplotlib.cm as cm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.io import loadmat
from reproduction_config import CRSR_CSV, FIGURES_DIR as REPRO_FIGURES_DIR, FITTED_DIR, PSD_DIR as PSD_DATA_DIR

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    plt.style.use(["science", "no-latex"])
except OSError:
    print("SciencePlots style not found; using Matplotlib defaults.")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = str(REPRO_FIGURES_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

PARAM_DIR = str(FITTED_DIR)
PSD_DIR = str(PSD_DATA_DIR)
CLIN_FILE = str(CRSR_CSV)
OUT_FILE = os.path.join(OUT_DIR, "Figure_5_direct_HighRes.png")

SUBJECTS = [
    ("A", "sub-007", "Active subject S07"),
    ("B", "sub-009", "Sham subject S09"),
    ("C", "sub-010", "Sham subject S10"),
]

SESSION_LABELS = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "24h", "7 days"]
SESSION_COLORS = [
    np.array([0.0, 0.4470, 0.7410]) + (np.array([0.8500, 0.3250, 0.0980]) - np.array([0.0, 0.4470, 0.7410])) * (i / 6.0)
    for i in range(7)
]

C_UWS = "#D92633"
C_MCSM = "#66B2F2"
C_MCSP = "#1A59BF"
C_EMCS = "#26A64D"
C_UNK = "#262626"
DOC_COLORS = {
    "UWS": C_UWS,
    "MCS-": C_MCSM,
    "MCS+": C_MCSP,
    "EMCS": C_EMCS,
    "UNKNOWN": C_UNK,
    "": C_UNK,
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 25,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
    }
)


def decode_param_names(raw_names):
    return [str(item[0]) if isinstance(item, (np.ndarray, list)) and len(item) > 0 else str(item) for item in raw_names]


def load_clinical_table():
    if not os.path.exists(CLIN_FILE):
        print(f"Clinical file not found: {CLIN_FILE}. DOC-state colors will be unknown.")
        return None
    return pd.read_csv(CLIN_FILE)


def doc_state_color(clin_data, subject_id, session_number):
    if clin_data is None:
        return C_UNK

    if session_number <= 5:
        event_str = f"ses-{session_number:02d}"
        time_str = "pre"
    elif session_number == 6:
        event_str = "ses-06"
        time_str = "post24"
    else:
        event_str = "ses-07"
        time_str = "post7"

    subject_rows = clin_data[clin_data["Record ID"].astype(str).str.fullmatch(subject_id, na=False)]
    row = subject_rows[
        subject_rows["Event Name"].astype(str).str.contains(event_str, na=False)
        & subject_rows["timepoint"].astype(str).str.contains(time_str, case=False, na=False)
    ]

    if row.empty:
        return C_UNK

    state = str(row["doc_state"].iloc[0]).strip().upper()
    return DOC_COLORS.get(state, C_UNK)


def load_xy_trajectory(subject_id):
    path = os.path.join(PARAM_DIR, f"{subject_id}_model_fits.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mat = loadmat(path)
    names = decode_param_names(mat["param_names"].flatten())
    lower = [name.lower() for name in names]
    dat = mat["params_sub"]

    if "x" not in lower or "y" not in lower:
        raise ValueError(f"{path} does not contain X/Y parameters. Available: {names}")

    run_idx = 0
    x = dat[:, run_idx, lower.index("x")]
    y = dat[:, run_idx, lower.index("y")]
    return x - y, x + y


def load_psd(subject_id):
    path = os.path.join(PSD_DIR, f"{subject_id}_model_fits.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mat = loadmat(path)
    freq_ref = mat["freq_ref"].flatten()
    psd_sub = mat["psd_sub"]
    freq_mask = (freq_ref >= 1) & (freq_ref <= 30)
    f_plot = freq_ref[freq_mask]

    psd_matrix_db = np.full((7, len(f_plot)), np.nan)
    valid_sessions = []
    for sess in range(7):
        psd_current = psd_sub[freq_mask, sess, 0]
        if not np.all(np.isnan(psd_current)):
            psd_matrix_db[sess, :] = 10 * np.log10(psd_current)
            valid_sessions.append(sess)

    return f_plot, psd_matrix_db, valid_sessions


def plot_single_trajectory(ax, subject_id, clin_data):
    x_seq, y_seq = load_xy_trajectory(subject_id)
    outline = [pe.Stroke(linewidth=4, foreground="white"), pe.Normal()]

    ax.plot([0.3, 1.2], [1, 1], "k--", lw=1.6, zorder=0)
    ax.plot([0.73, 0.73], [0.4, 1.2], "k-", lw=1.6, zorder=0)

    ax.plot(x_seq, y_seq, "-", color="#b0b0b0", alpha=0.35, linewidth=1.5, zorder=1)

    for idx in range(1, 5):
        if np.isfinite(x_seq[idx]) and np.isfinite(y_seq[idx]):
            ax.scatter(
                x_seq[idx],
                y_seq[idx],
                s=34,
                color=doc_state_color(clin_data, subject_id, idx + 1),
                alpha=0.55,
                edgecolors="none",
                zorder=2,
            )

    if np.isfinite(x_seq[0]) and np.isfinite(x_seq[6]):
        if np.isfinite(x_seq[5]):
            ax.plot(
                [x_seq[0], x_seq[5], x_seq[6]],
                [y_seq[0], y_seq[5], y_seq[6]],
                "-",
                color="#8f8f8f",
                linewidth=2.4,
                zorder=3,
                path_effects=outline,
            )
            main_points = [(0, "o", 120), (5, "d", 140), (6, "s", 140)]
        else:
            ax.plot(
                [x_seq[0], x_seq[6]],
                [y_seq[0], y_seq[6]],
                "-",
                color="#8f8f8f",
                linewidth=2.4,
                zorder=3,
                path_effects=outline,
            )
            main_points = [(0, "o", 120), (6, "s", 140)]

        for idx, marker, size in main_points:
            ax.scatter(
                x_seq[idx],
                y_seq[idx],
                s=size,
                color=doc_state_color(clin_data, subject_id, idx + 1),
                marker=marker,
                edgecolors="black",
                linewidth=1.5,
                zorder=4,
            )

    ax.set_xlim(0.55, 1.15)
    ax.set_ylim(0.80, 1.05)
    ax.set_xlabel(r"$S_{CC} - S_{CT}$", fontweight="bold", fontsize=20)
    ax.set_ylabel(r"$S_{CC} + S_{CT}$", fontweight="bold", fontsize=20)
    ax.tick_params(top=True, right=True, direction="in", which="both")


def add_panel_label(ax, label):
    ax.text(
        -0.22,
        1.19,
        label,
        transform=ax.transAxes,
        fontsize=28,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def plot_psd_2d(ax, f_plot, psd_matrix_db, valid_sessions):
    for sess in valid_sessions:
        ax.plot(f_plot, psd_matrix_db[sess, :], color=SESSION_COLORS[sess], linewidth=2.0, label=SESSION_LABELS[sess])

    finite_values = psd_matrix_db[np.isfinite(psd_matrix_db)]
    if finite_values.size:
        ymin, ymax = np.nanmin(finite_values), np.nanmax(finite_values)
        margin = 0.06 * (ymax - ymin)
        ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax.set_ylabel("Power (dB)", fontweight="bold")
    ax.set_xlim(1, 30)
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(loc="upper right", frameon=False)


def plot_psd_3d(ax, f_plot, psd_matrix_db, valid_sessions):
    for sess in valid_sessions:
        y_val = np.full_like(f_plot, sess)
        ax.plot(f_plot, y_val, psd_matrix_db[sess, :], color=SESSION_COLORS[sess], linewidth=2.0, zorder=sess)

    finite_values = psd_matrix_db[np.isfinite(psd_matrix_db)]
    if finite_values.size:
        ymin, ymax = np.nanmin(finite_values), np.nanmax(finite_values)
        margin = 0.06 * (ymax - ymin)
        ax.set_zlim(ymin - margin, ymax + margin)

    ax.set_xlabel("Frequency (Hz)", fontweight="bold", labelpad=10)
    ax.set_zlabel("Power (dB)", fontweight="bold", labelpad=8)
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 6)
    ax.set_yticks(range(7))
    ax.set_yticklabels(SESSION_LABELS, rotation=-15, ha="left")
    ax.view_init(elev=35, azim=-45)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    ax.grid(True, alpha=0.15)


def trajectory_legend(fig):
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Session 1 (Start)", markerfacecolor="black", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="d", color="w", label="24h Post", markerfacecolor="black", markeredgecolor="black", markersize=11),
        Line2D([0], [0], marker="s", color="w", label="1 Week Post (End)", markerfacecolor="black", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="UWS", markerfacecolor=C_UWS, markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="MCS-", markerfacecolor=C_MCSM, markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="MCS+", markerfacecolor=C_MCSP, markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="eMCS", markerfacecolor=C_EMCS, markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Unknown", markerfacecolor=C_UNK, markeredgecolor="black", markersize=10),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=16,
        columnspacing=2.0,
        handletextpad=0.7,
    )


def main():
    clin_data = load_clinical_table()

    fig = plt.figure(figsize=(17.0, 18.4))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=3,
        left=0.06,
        right=0.985,
        top=0.95,
        bottom=0.12,
        wspace=0.25,
        hspace=0.36,
        width_ratios=[1.05, 1.15, 1.25],
    )

    for row, (panel_label, subject_id, row_title) in enumerate(SUBJECTS):
        ax_traj = fig.add_subplot(gs[row, 0])
        ax_2d = fig.add_subplot(gs[row, 1])
        ax_3d = fig.add_subplot(gs[row, 2], projection="3d")

        plot_single_trajectory(ax_traj, subject_id, clin_data)
        f_plot, psd_matrix_db, valid_sessions = load_psd(subject_id)
        plot_psd_2d(ax_2d, f_plot, psd_matrix_db, valid_sessions)
        plot_psd_3d(ax_3d, f_plot, psd_matrix_db, valid_sessions)

        ax_2d.set_title(row_title, fontweight="bold", fontsize=25, pad=18)
        add_panel_label(ax_traj, panel_label)

    trajectory_legend(fig)
    plt.savefig(OUT_FILE, dpi=200)
    print(f"Saved figure: {OUT_FILE}")

    if os.environ.get("SHOW_FIGURE", "1") != "0":
        plt.show()


if __name__ == "__main__":
    main()
