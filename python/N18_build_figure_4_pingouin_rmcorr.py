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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
from scipy.stats import spearmanr
from reproduction_config import CRSR_XLSX, FIGURES_DIR as REPRO_FIGURES_DIR, FITTED_DIR

try:
    import pingouin as pg
except ImportError as exc:
    raise ImportError(
        "This script uses the Python repeated-measures correlation toolbox "
        "`pingouin.rm_corr`, which is tested against the R `rmcorr` package.\n"
        "Install it with:\n"
        "    pip install pingouin\n"
        "or, in conda:\n"
        "    conda install -c conda-forge pingouin"
    ) from exc

try:
    import statsmodels.formula.api as smf
except ImportError:
    smf = None

warnings.filterwarnings("ignore")


BASEDIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = str(REPRO_FIGURES_DIR)
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_DIR = str(FITTED_DIR)
CRSR_FILE = str(CRSR_XLSX)

OUTPUT_FIG = os.path.join(FIGURES_DIR, "Figure_4_RMcorr_HighRes.png")
OUTPUT_CSV = os.path.join(FIGURES_DIR, "Figure_4_RMcorr_stats.csv")

TARGET_PARAMS = ["X", "Y", "Z", "Alpha", "Beta"]
SUBSCALE_NAMES = ["Auditory", "Visual", "Motor", "Oromotor", "Communication", "Arousal"]

DISPLAY_NAME = {
    "X": r"$S_{CC}$",
    "Y": r"$S_{CT}$",
    "Z": r"$S_{IT}$",
    "Alpha": r"$\alpha$",
    "Beta": r"$\beta$",
}

try:
    plt.style.use(["science", "no-latex"])
except OSError:
    print("SciencePlots style not found; using Matplotlib defaults.")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "mathtext.default": "regular",
        "axes.grid": False,
        "pdf.fonttype": 42,
    }
)


def decode_matlab_name(item):
    if isinstance(item, (np.ndarray, list)) and len(item) > 0:
        return str(item[0])
    return str(item)


def load_model_parameters():
    mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_model_fits.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No *_model_fits.mat files found in {DATA_DIR}")

    sample = loadmat(mat_files[0])
    pnames = [decode_matlab_name(item) for item in sample["param_names"].flatten()]

    n_subjects = len(mat_files)
    params = np.full((n_subjects, 7, 4, len(pnames)), np.nan)
    subject_ids = []

    for subject_idx, file_path in enumerate(mat_files):
        mat = loadmat(file_path)
        dat = mat["params_sub"]
        params[subject_idx, : dat.shape[0], : dat.shape[1], :] = dat
        subject_ids.append(os.path.basename(file_path).replace("_model_fits.mat", ""))

    return params, pnames, subject_ids


def subject_case_id(subject_id):
    # sub-001 -> 1, sub-010 -> 10. Keeps matching exact and avoids 1 matching 10.
    digits = "".join(ch for ch in subject_id if ch.isdigit())
    return int(digits) if digits else None


def load_crsr(subject_ids):
    crsr_total = np.full((len(subject_ids), 7, 2), np.nan)
    crsr_subscales = np.full((len(subject_ids), 7, 2, len(SUBSCALE_NAMES)), np.nan)
    case_to_subject = {subject_case_id(sid): i for i, sid in enumerate(subject_ids)}

    df_crsr = pd.read_excel(CRSR_FILE, header=None)
    last_valid_case = np.nan
    last_valid_session = np.nan

    for _, row in df_crsr.iterrows():
        try:
            case = float(row[0])
            if not np.isnan(case):
                last_valid_case = case
        except Exception:
            pass

        try:
            session = float(row[1])
            if not np.isnan(session):
                last_valid_session = session
        except Exception:
            pass

        if np.isnan(last_valid_case) or np.isnan(last_valid_session):
            continue

        subject_idx = case_to_subject.get(int(last_valid_case))
        if subject_idx is None:
            continue

        timepoint = str(row[3]).strip().lower()

        try:
            total_score = float(row[10])
        except Exception:
            total_score = np.nan

        subscales = np.full(len(SUBSCALE_NAMES), np.nan)
        for sub_idx in range(len(SUBSCALE_NAMES)):
            try:
                subscales[sub_idx] = float(row[4 + sub_idx])
            except Exception:
                pass

        if "pre" in timepoint:
            run_idx, session_idx = 0, int(last_valid_session) - 1
        elif "post" in timepoint and "week" not in timepoint and "protocol" not in timepoint:
            run_idx, session_idx = 1, int(last_valid_session) - 1
        elif "protocol" in timepoint:
            run_idx, session_idx = 0, 5
        elif "week" in timepoint:
            run_idx, session_idx = 0, 6
        else:
            continue

        if 0 <= session_idx < 7:
            crsr_total[subject_idx, session_idx, run_idx] = total_score
            crsr_subscales[subject_idx, session_idx, run_idx, :] = subscales

    return crsr_total, crsr_subscales


def build_long_dataframe(params, pnames, subject_ids, crsr_total, crsr_subscales):
    pnames_lower = [name.lower() for name in pnames]
    rows = []

    for subject_idx, subject_id in enumerate(subject_ids):
        group = "Active" if subject_idx < 6 else "Sham"
        for session_idx in range(7):
            row = {
                "subject": subject_id,
                "group": group,
                "session": session_idx + 1,
                "crsr_total": crsr_total[subject_idx, session_idx, 0],
            }

            for param in TARGET_PARAMS:
                try:
                    p_idx = pnames_lower.index(param.lower())
                except ValueError as exc:
                    raise ValueError(f"Parameter {param} was not found in the model-fit files.") from exc
                row[param] = params[subject_idx, session_idx, 0, p_idx]

            for sub_idx, sub_name in enumerate(SUBSCALE_NAMES):
                row[sub_name] = crsr_subscales[subject_idx, session_idx, 0, sub_idx]

            rows.append(row)

    return pd.DataFrame(rows)


def filtered_pair_data(data, x_col, y_col):
    df = data[["subject", "session", x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    counts = df.groupby("subject").size()
    keep_subjects = counts[counts >= 2].index
    return df[df["subject"].isin(keep_subjects)].copy()


def run_pingouin_rmcorr(data, x_col, y_col):
    df = filtered_pair_data(data, x_col, y_col)
    if len(df) < 4 or df["subject"].nunique() < 2:
        return {
            "data": df,
            "r": np.nan,
            "p": np.nan,
            "dof": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "power": np.nan,
        }

    stats = pg.rm_corr(data=df, x=x_col, y=y_col, subject="subject")
    row = stats.iloc[0]
    ci = row.get("CI95", [np.nan, np.nan])
    return {
        "data": df,
        "r": float(row["r"]),
        "p": float(row["pval"]),
        "dof": float(row["dof"]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "power": float(row["power"]),
    }


def flat_spearman(data, x_col, y_col):
    df = data[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 4:
        return np.nan, np.nan, len(df)
    r, p = spearmanr(df[x_col], df[y_col])
    return float(r), float(p), len(df)


def fixed_effect_time_check(data, x_col, y_col):
    if smf is None:
        return np.nan, np.nan, np.nan, 0

    df = filtered_pair_data(data, x_col, y_col)
    if len(df) < 6:
        return np.nan, np.nan, np.nan, len(df)

    model_df = df.rename(columns={x_col: "x", y_col: "y"})
    fit = smf.ols("y ~ x + session + C(subject)", data=model_df).fit()
    return float(fit.params["x"]), float(fit.pvalues["x"]), float(fit.df_resid), len(model_df)


def p_text(p):
    if not np.isfinite(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_rmcorr_panel(ax, stat, x_col, y_col):
    df = stat["data"]
    ax.scatter(df[x_col], df[y_col], s=90, facecolor="#d3d3d3", edgecolor="black", alpha=0.8, zorder=3)

    # Draw subject-specific parallel lines from a subject fixed-effects model for visualization.
    # The reported statistic still comes from pingouin.rm_corr.
    if smf is not None and len(df) >= 6 and df["subject"].nunique() >= 2:
        model_df = df.rename(columns={x_col: "x", y_col: "y"})
        fit = smf.ols("y ~ x + C(subject)", data=model_df).fit()
        slope = fit.params["x"]

        for subject_id, sdf in model_df.groupby("subject"):
            if len(sdf) < 2:
                continue
            xs = np.linspace(sdf["x"].min(), sdf["x"].max(), 25)
            pred = pd.DataFrame({"x": xs, "subject": subject_id})
            ax.plot(xs, fit.predict(pred), color="#9e9e9e", linewidth=1.0, alpha=0.5, zorder=1)

        x_all = np.linspace(model_df["x"].min(), model_df["x"].max(), 100)
        intercept = model_df["y"].mean() - slope * model_df["x"].mean()
        ax.plot(x_all, intercept + slope * x_all, color="#d62728", linewidth=2.5, zorder=2)

    stat_text = f"rmcorr r = {stat['r']:.3f}\np-value {p_text(stat['p'])}"
    ax.text(
        0.05,
        0.05,
        stat_text,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="bottom",
        bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="black", lw=1.0),
        fontweight="bold",
    )

    ax.set_xlabel(f"Baseline {DISPLAY_NAME.get(x_col, x_col)}", fontweight="bold")
    ax.set_ylabel("Baseline CRS-R score", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    params, pnames, subject_ids = load_model_parameters()
    crsr_total, crsr_subscales = load_crsr(subject_ids)
    data = build_long_dataframe(params, pnames, subject_ids, crsr_total, crsr_subscales)

    stats_rows = []
    total_stats = {}

    print("\nRepeated-measures correlation for CRS-R total")
    print("Computed with pingouin.rm_corr, the Python implementation tested against R rmcorr.")
    print(f"{'Parameter':<10} {'r_rm':>9} {'p_rm':>9} {'dof':>5} {'CI95':>17} {'flat Spearman p':>17} {'FE+time p':>12}")
    print("-" * 92)

    for param in TARGET_PARAMS:
        rm = run_pingouin_rmcorr(data, param, "crsr_total")
        total_stats[param] = rm
        flat_r, flat_p, flat_n = flat_spearman(data, param, "crsr_total")
        fe_beta, fe_p, fe_df, fe_n = fixed_effect_time_check(data, param, "crsr_total")

        stats_rows.append(
            {
                "outcome": "crsr_total",
                "parameter": param,
                "rmcorr_toolbox": "pingouin.rm_corr",
                "rmcorr_r": rm["r"],
                "rmcorr_p": rm["p"],
                "rmcorr_dof": rm["dof"],
                "rmcorr_ci_low": rm["ci_low"],
                "rmcorr_ci_high": rm["ci_high"],
                "rmcorr_power": rm["power"],
                "rmcorr_n": len(rm["data"]),
                "rmcorr_subjects": rm["data"]["subject"].nunique(),
                "flat_spearman_r": flat_r,
                "flat_spearman_p": flat_p,
                "flat_spearman_n": flat_n,
                "fixed_effect_time_beta": fe_beta,
                "fixed_effect_time_p": fe_p,
                "fixed_effect_time_df": fe_df,
                "fixed_effect_time_n": fe_n,
            }
        )

        ci_text = f"[{rm['ci_low']:.2f}, {rm['ci_high']:.2f}]"
        print(f"{param:<10} {rm['r']:>9.3f} {rm['p']:>9.4f} {rm['dof']:>5.0f} {ci_text:>17} {flat_p:>17.4f} {fe_p:>12.4f}")

    corr_matrix = np.full((len(TARGET_PARAMS), len(SUBSCALE_NAMES)), np.nan)
    pval_matrix = np.full_like(corr_matrix, np.nan)

    print("\nRepeated-measures correlation for CRS-R subscales")
    for i, param in enumerate(TARGET_PARAMS):
        for j, subscale in enumerate(SUBSCALE_NAMES):
            rm = run_pingouin_rmcorr(data, param, subscale)
            flat_r, flat_p, flat_n = flat_spearman(data, param, subscale)
            fe_beta, fe_p, fe_df, fe_n = fixed_effect_time_check(data, param, subscale)

            corr_matrix[i, j] = rm["r"]
            pval_matrix[i, j] = rm["p"]

            stats_rows.append(
                {
                    "outcome": subscale,
                    "parameter": param,
                    "rmcorr_toolbox": "pingouin.rm_corr",
                    "rmcorr_r": rm["r"],
                    "rmcorr_p": rm["p"],
                    "rmcorr_dof": rm["dof"],
                    "rmcorr_ci_low": rm["ci_low"],
                    "rmcorr_ci_high": rm["ci_high"],
                    "rmcorr_power": rm["power"],
                    "rmcorr_n": len(rm["data"]),
                    "rmcorr_subjects": rm["data"]["subject"].nunique(),
                    "flat_spearman_r": flat_r,
                    "flat_spearman_p": flat_p,
                    "flat_spearman_n": flat_n,
                    "fixed_effect_time_beta": fe_beta,
                    "fixed_effect_time_p": fe_p,
                    "fixed_effect_time_df": fe_df,
                    "fixed_effect_time_n": fe_n,
                }
            )

    pd.DataFrame(stats_rows).to_csv(OUTPUT_CSV, index=False)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 1.2, 1.2])
    axes_scat = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[2, 0]),
    ]
    ax_heat = fig.add_subplot(gs[1:3, 1:3])

    for ax, param in zip(axes_scat, TARGET_PARAMS):
        plot_rmcorr_panel(ax, total_stats[param], param, "crsr_total")

    axes_scat[0].text(-0.25, 1.20, "A", transform=axes_scat[0].transAxes, fontsize=26, fontweight="bold")

    cmap = LinearSegmentedColormap.from_list("bwr_custom", [(0, 0, 1), (1, 1, 1), (1, 0, 0)], N=256)
    cax = ax_heat.imshow(corr_matrix, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
    cbar = fig.colorbar(cax, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Repeated-measures correlation (r)", fontweight="bold", fontsize=16)

    ax_heat.set_xticks(np.arange(len(SUBSCALE_NAMES)))
    ax_heat.set_yticks(np.arange(len(TARGET_PARAMS)))
    ax_heat.set_xticklabels(SUBSCALE_NAMES, rotation=45, ha="right", fontweight="bold")
    ax_heat.set_yticklabels([DISPLAY_NAME[param] for param in TARGET_PARAMS], fontweight="bold")

    for i in range(len(TARGET_PARAMS)):
        for j in range(len(SUBSCALE_NAMES)):
            val, pval = corr_matrix[i, j], pval_matrix[i, j]
            if not np.isnan(val):
                txt = f"{val:.2f}{stars(pval)}"
                text_col = "white" if abs(val) > 0.4 else "black"
                ax_heat.text(j, i, txt, ha="center", va="center", color=text_col, fontweight="bold", fontsize=14)

    ax_heat.text(-0.05, 1.08, "B", transform=ax_heat.transAxes, fontsize=26, fontweight="bold", va="top")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.35, hspace=0.45)
    plt.savefig(OUTPUT_FIG, dpi=200)

    print(f"\nSaved figure: {OUTPUT_FIG}")
    print(f"Saved statistics table: {OUTPUT_CSV}")
    print("\nConsole robustness note:")
    print("  Main rmcorr statistics are from pingouin.rm_corr.")
    print("  flat Spearman is the old independent-samples analysis, printed only for comparison.")
    if smf is None:
        print("  statsmodels is not installed, so FE+time robustness checks were skipped.")
        print("  Install it with: pip install statsmodels")
    else:
        print("  FE+time is CRS-R ~ parameter + session + C(subject), used as a linear time-drift robustness check.")

    plt.show()


if __name__ == "__main__":
    main()
