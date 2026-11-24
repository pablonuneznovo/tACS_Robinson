import glob
import os
import warnings

import matplotlib

if os.environ.get("SHOW_FIGURE", "1") == "0":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import mannwhitneyu
from reproduction_config import FITTED_DIR, FIGURES_DIR as CONFIG_FIGURES_DIR


warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = str(CONFIG_FIGURES_DIR)
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_DIR = str(FITTED_DIR)
OUTPUT_FIG = os.path.join(FIGURES_DIR, "SM_Figure4.png")
OUTPUT_CSV = os.path.join(FIGURES_DIR, "Supp_Figure_Response_Duration_stats.csv")
OUTPUT_CAPTION = os.path.join(FIGURES_DIR, "Supp_Figure_Response_Duration_caption.txt")

TIME_LABELS = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "24h", "1 week"]
COLOR_ACT = "#A1C9F4"
COLOR_SHM = "#FF9F9B"
COLORS = {"Active": COLOR_ACT, "Sham": COLOR_SHM}

try:
    plt.style.use(["science", "no-latex"])
except OSError:
    print("SciencePlots style not found; using Matplotlib defaults.")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
    }
)


def decode_matlab_name(item):
    if isinstance(item, (np.ndarray, list)) and len(item) > 0:
        return str(item[0])
    return str(item)


def find_param_index(pnames, target):
    lower = [name.lower() for name in pnames]
    aliases = {
        "alpha": ["alpha", "a"],
        "beta": ["beta", "b"],
    }[target]
    for alias in aliases:
        if alias in lower:
            return lower.index(alias)
    raise ValueError(f"Could not find {target}. Available parameters: {pnames}")


def load_response_duration():
    mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_model_fits.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No *_model_fits.mat files found in {DATA_DIR}")

    records = []
    for subject_idx, file_path in enumerate(mat_files):
        mat = loadmat(file_path)
        pnames = [decode_matlab_name(item) for item in mat["param_names"].flatten()]
        alpha_idx = find_param_index(pnames, "alpha")
        beta_idx = find_param_index(pnames, "beta")
        params = mat["params_sub"]

        subject = os.path.basename(file_path).replace("_model_fits.mat", "")
        group = "Active" if subject_idx < 6 else "Sham"

        for session_idx, label in enumerate(TIME_LABELS):
            if session_idx >= params.shape[0]:
                continue
            alpha = params[session_idx, 0, alpha_idx]
            beta = params[session_idx, 0, beta_idx]
            if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0 or beta <= 0:
                continue

            response_duration = (1.0 / alpha) + (1.0 / beta)
            records.append(
                {
                    "subject": subject,
                    "group": group,
                    "session_index": session_idx,
                    "timepoint": label,
                    "alpha": alpha,
                    "beta": beta,
                    "response_duration_s": response_duration,
                    "response_duration_ms": response_duration * 1000.0,
                }
            )

    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError("No valid alpha/beta values were found.")

    baseline = data[data["session_index"] == 0][["subject", "response_duration_s", "response_duration_ms"]]
    baseline = baseline.rename(
        columns={
            "response_duration_s": "baseline_response_duration_s",
            "response_duration_ms": "baseline_response_duration_ms",
        }
    )
    data = data.merge(baseline, on="subject", how="left")
    data["delta_response_duration_s"] = data["response_duration_s"] - data["baseline_response_duration_s"]
    data["delta_response_duration_ms"] = data["response_duration_ms"] - data["baseline_response_duration_ms"]

    return data


def sem(values):
    values = pd.Series(values).dropna()
    if len(values) <= 1:
        return np.nan
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def descriptive_stats(frame, value_col):
    rows = []
    for group, group_df in frame.groupby("group"):
        values = group_df[value_col].dropna()
        rows.append(
            {
                "analysis": value_col,
                "group": group,
                "n": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "sem": sem(values),
                "median": float(values.median()) if len(values) else np.nan,
                "min": float(values.min()) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
            }
        )
    return rows


def add_panel_label(ax, label):
    ax.text(
        -0.16,
        1.12,
        label,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def plot_longitudinal(ax, data):
    x = np.arange(len(TIME_LABELS))
    for group in ["Active", "Sham"]:
        group_df = data[data["group"] == group]
        means = group_df.groupby("session_index")["response_duration_ms"].mean().reindex(x)
        errors = group_df.groupby("session_index")["response_duration_ms"].apply(sem).reindex(x)
        ax.errorbar(
            x,
            means,
            yerr=errors,
            marker="o" if group == "Active" else "s",
            color=COLORS[group],
            markeredgecolor="black",
            linewidth=2.2,
            markersize=7,
            capsize=3,
            label=group,
            zorder=3,
        )

        for _, sub_df in group_df.groupby("subject"):
            sub_df = sub_df.sort_values("session_index")
            ax.plot(
                sub_df["session_index"],
                sub_df["response_duration_ms"],
                color=COLORS[group],
                alpha=0.18,
                linewidth=1.0,
                zorder=1,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(TIME_LABELS, rotation=35, ha="right")
    ax.set_ylabel(r"$1/\alpha + 1/\beta$ (ms)", fontweight="bold")
    ax.set_xlabel("Timepoint", fontweight="bold")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_delta(ax, data, u_stat, p_value):
    week = data[data["session_index"] == 6].copy()
    positions = {"Active": 0, "Sham": 1}

    rng = np.random.default_rng(7)
    box_values = [week[week["group"] == group]["delta_response_duration_ms"].dropna() for group in ["Active", "Sham"]]
    bp = ax.boxplot(
        box_values,
        positions=[0, 1],
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"linewidth": 1.3},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    for patch, group in zip(bp["boxes"], ["Active", "Sham"]):
        patch.set_facecolor(COLORS[group])
        patch.set_edgecolor("black")
        patch.set_alpha(0.75)

    for group in ["Active", "Sham"]:
        values = week[week["group"] == group]["delta_response_duration_ms"].dropna().to_numpy()
        jitter = rng.normal(0, 0.04, size=len(values))
        ax.scatter(
            np.full(len(values), positions[group]) + jitter,
            values,
            s=75,
            facecolor=COLORS[group],
            edgecolor="black",
            linewidth=1.1,
            zorder=4,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Active", "Sham"])
    ax.set_ylabel(r"$\Delta(1/\alpha + 1/\beta)$ at 1 week (ms)", fontweight="bold")
    ax.set_xlabel("Group", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    ax.text(
        0.05,
        0.96,
        f"Mann-Whitney U = {u_stat:.1f}\np = {p_value:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor="black"),
        va="top",
    )
    ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.08 * y_span)


def write_caption():
    caption = (
        "Figure S: Long-term changes in the fitted macroscopic response duration. "
        "A: Group mean trajectories of the response-duration proxy "
        "$(1/\\alpha + 1/\\beta)$ across the protocol, computed from the fitted "
        "$\\alpha$ and $\\beta$ parameters before each session and at follow-up. "
        "Thin lines show individual participants and markers show group mean "
        "$\\pm$ SEM. B: One-week change in $(1/\\alpha + 1/\\beta)$ relative to "
        "Day 1 baseline for active and sham stimulation. Negative values indicate "
        "a shorter overall response duration. Group differences were assessed with "
        "a two-sided Mann-Whitney U test."
    )
    with open(OUTPUT_CAPTION, "w", encoding="utf-8") as file:
        file.write(caption + "\n")


def main():
    data = load_response_duration()
    week = data[data["session_index"] == 6].copy()
    active_delta = week[week["group"] == "Active"]["delta_response_duration_ms"].dropna()
    sham_delta = week[week["group"] == "Sham"]["delta_response_duration_ms"].dropna()

    u_stat, p_value = mannwhitneyu(active_delta, sham_delta, alternative="two-sided", method="auto")
    u_max = len(active_delta) * len(sham_delta)
    rank_biserial = (2 * u_stat / u_max) - 1

    stats_rows = []
    stats_rows.extend(descriptive_stats(data[data["session_index"] == 0], "response_duration_ms"))
    stats_rows.extend(descriptive_stats(week, "response_duration_ms"))
    stats_rows.extend(descriptive_stats(week, "delta_response_duration_ms"))
    stats_rows.append(
        {
            "analysis": "mann_whitney_delta_response_duration_ms",
            "group": "Active_vs_Sham",
            "n": f"{len(active_delta)} vs {len(sham_delta)}",
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "u_stat": float(u_stat),
            "p_value": float(p_value),
            "rank_biserial_r": float(rank_biserial),
        }
    )

    data.to_csv(OUTPUT_CSV.replace("_stats.csv", "_per_subject.csv"), index=False)
    pd.DataFrame(stats_rows).to_csv(OUTPUT_CSV, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    plot_longitudinal(axes[0], data)
    plot_delta(axes[1], data, u_stat, p_value)
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
    write_caption()

    print("\nResponse-duration proxy: 1/alpha + 1/beta")
    print("Units in the figure and tables are milliseconds.")
    print("\nOne-week delta versus Day 1 baseline")
    print(f"Active: mean={active_delta.mean():.3f} ms, SD={active_delta.std(ddof=1):.3f} ms, n={len(active_delta)}")
    print(f"Sham:   mean={sham_delta.mean():.3f} ms, SD={sham_delta.std(ddof=1):.3f} ms, n={len(sham_delta)}")
    print(f"Mann-Whitney U={u_stat:.3f}, p={p_value:.4f}, rank-biserial r={rank_biserial:.3f}")
    print(f"\nSaved figure: {OUTPUT_FIG}")
    print(f"Saved statistics table: {OUTPUT_CSV}")
    print(f"Saved per-subject table: {OUTPUT_CSV.replace('_stats.csv', '_per_subject.csv')}")
    print(f"Saved caption: {OUTPUT_CAPTION}")

    if os.environ.get("SHOW_FIGURE", "1") != "0":
        plt.show()


if __name__ == "__main__":
    main()
