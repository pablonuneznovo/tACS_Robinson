"""SD-only stability analysis of 100 repeat fits of the saved EEG spectra."""

from pathlib import Path
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scienceplots
from scipy.io import loadmat
from reproduction_config import FIGURES_DIR, REPEAT_FITS_DIR

HERE = FIGURES_DIR
INPUT = REPEAT_FITS_DIR
SUBJECTS = ["sub-001", "sub-004", "sub-005", "sub-006", "sub-007", "sub-008",
            "sub-009", "sub-010", "sub-011", "sub-012", "sub-013"]
NAMES = ["S_CC", "S_CT", "S_IT", "alpha", "beta", "t0", "EMG"]
DISPLAY = [r"$S_{CC}$", r"$S_{CT}$", r"$S_{IT}$", "α", "β", r"$t_0$", "EMG"]
PATIENT_LABELS = [s.removeprefix("sub-") for s in SUBJECTS]
ACTIVE = np.arange(6)
SHAM = np.arange(6, 11)
N_REPEATS = 100
COLOR_ACT = "#A1C9F4"
COLOR_SHM = "#FF9F9B"
COLORS = {label: COLOR_ACT if i in ACTIVE else COLOR_SHM
          for i, label in enumerate(PATIENT_LABELS)}
COLORS.update({"Active": COLOR_ACT, "Sham": COLOR_SHM})
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 21, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 16, "mathtext.default": "regular", "axes.grid": False,
})
np.random.seed(20260925)


# Copied without styling changes from N19_build_figure_2_with_t0.py, as in the
# earlier Louis revision figure. Patient labels replace the original x labels.
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


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


files = list(INPUT.glob("*.mat"))
assert len(files) == 11 * 2 * N_REPEATS, (INPUT, len(files))
fits = np.full((11, 2, N_REPEATS, 7), np.nan)
chisq = np.full((11, 2, N_REPEATS), np.nan)
original = np.full((11, 2, 7), np.nan)
original_chisq = np.full((11, 2), np.nan)
seen = set()
for path in files:
    match = re.fullmatch(r"(sub-\d+)_ses-(01|07)_repeat-(\d{3})\.mat", path.name)
    assert match, path
    subject, session, rep = match.groups()
    i, t, k = SUBJECTS.index(subject), ["01", "07"].index(session), int(rep) - 1
    assert 0 <= k < N_REPEATS and (i, t, k) not in seen, path
    seen.add((i, t, k))
    result = loadmat(path, simplify_cells=True)["result"]
    assert result["success"], (path, result.get("error_message"))
    fits[i, t, k] = result["parameters"]
    chisq[i, t, k] = result["chisq"]
    if k == 0:
        original[i, t] = result["original_parameters"]
        original_chisq[i, t] = result["original_chisq"]
    else:
        assert np.allclose(original[i, t], result["original_parameters"])
        assert np.isclose(original_chisq[i, t], result["original_chisq"])
assert len(seen) == 2200 and np.isfinite(fits).all() and np.isfinite(chisq).all()
assert np.isfinite(original).all() and np.isfinite(original_chisq).all()

# Identical repeat indices identify the same batch number, not paired biological
# observations. Pairing here gives one 100-draw set of numerical deltas.
deltas = fits[:, 1, :, :] - fits[:, 0, :, :]
original_deltas = original[:, 1, :] - original[:, 0, :]
patient_rows = []
for i, subject in enumerate(SUBJECTS):
    for p, name in enumerate(NAMES):
        values = deltas[i, :, p]
        original_delta = original_deltas[i, p]
        sd = np.std(values, ddof=1)
        patient_rows.append(dict(
            subject=subject, group="Active" if i in ACTIVE else "Sham",
            parameter=name, repeat_count=N_REPEATS,
            original_delta=float(original_delta),
            repeat_delta_mean=float(np.mean(values)),
            repeat_delta_sd=float(sd),
            absolute_original_delta_over_sd=float(abs(original_delta) / sd) if sd else np.nan,
            absolute_original_delta_exceeds_sd=bool(abs(original_delta) > sd),
        ))
write_rows(HERE / "Repeat_fit_100_patient_SD.csv", patient_rows)

group_rows = []
for p, name in enumerate(NAMES):
    for group, indices in [("Active", ACTIVE), ("Sham", SHAM)]:
        values = np.mean(deltas[indices, :, p], axis=0)
        original_mean = np.mean(original_deltas[indices, p])
        sd = np.std(values, ddof=1)
        group_rows.append(dict(
            parameter=name, group=group, patient_count=len(indices),
            repeat_count=N_REPEATS, original_mean_delta=float(original_mean),
            repeat_mean_delta_mean=float(np.mean(values)),
            repeat_mean_delta_sd=float(sd),
            absolute_original_mean_delta_over_sd=float(abs(original_mean) / sd) if sd else np.nan,
            absolute_original_mean_delta_exceeds_sd=bool(abs(original_mean) > sd),
        ))
write_rows(HERE / "Repeat_fit_100_group_mean_SD.csv", group_rows)


def style_panel(ax):
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")


fig, axes = plt.subplots(7, 2, figsize=(14.7, 20.0), sharey="row",
                         gridspec_kw={"width_ratios": [1.5, .6]})
for p, name in enumerate(NAMES):
    ax = axes[p, 0]
    records = [{"Patient": PATIENT_LABELS[i], "Value": value}
               for i in range(11) for value in deltas[i, :, p]]
    draw_science_boxplots(ax, pd.DataFrame(records), "Patient", "Value", None,
                          PATIENT_LABELS)
    ax.scatter(range(11), original_deltas[:, p], marker="D", color="black",
               s=18, zorder=5)
    ax.axvline(5.5, color="#777777", ls="--", lw=.8)
    ax.set_ylabel(DISPLAY[p], fontweight="bold", fontsize=22)
    style_panel(ax)

    ax = axes[p, 1]
    records = [{"Group": group, "Value": value}
               for group, indices in [("Active", ACTIVE), ("Sham", SHAM)]
               for value in np.mean(deltas[indices, :, p], axis=0)]
    draw_science_boxplots(ax, pd.DataFrame(records), "Group", "Value", None,
                          ["Active", "Sham"])
    ax.scatter([0, 1], [np.mean(original_deltas[indices, p])
                        for indices in (ACTIVE, SHAM)], marker="D",
               color="black", s=18, zorder=5)
    ax.set_ylabel("Group mean Δ", fontsize=14)
    style_panel(ax)
    ax.tick_params(labelleft=True)

axes[0, 0].set_title("A", loc="left", fontweight="bold")
axes[0, 1].set_title("B", loc="left", fontweight="bold")
axes[-1, 0].set_xlabel("Patient")
axes[-1, 1].set_xlabel("Group")
fig.subplots_adjust(left=.08, right=.97, bottom=.04, top=.94, hspace=.42, wspace=.35)
fig.legend(handles=[
    mpatches.Patch(facecolor=COLOR_ACT, edgecolor="black", label="Active"),
    mpatches.Patch(facecolor=COLOR_SHM, edgecolor="black", label="Sham"),
    plt.Line2D([0], [0], marker="D", color="none", markerfacecolor="black",
               markersize=5, label="Original fit"),
], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .99))
fig.savefig(HERE / "Repeat_fit_100_delta_vs_variability.png", dpi=220)
plt.close(fig)

print(f"Validated {len(files)} successful fits from {INPUT}")
for p, name in enumerate(NAMES):
    relevant = [r for r in patient_rows if r["parameter"] == name]
    print(name, sum(r["absolute_original_delta_exceeds_sd"] for r in relevant), "/ 11")
for row in group_rows:
    print(row["parameter"], row["group"], "original mean delta", round(row["original_mean_delta"], 5),
          "refit SD", round(row["repeat_mean_delta_sd"], 5))
