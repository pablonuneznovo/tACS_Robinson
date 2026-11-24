import glob
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # Registers the original paper's science/no-latex style.
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
from scipy.stats import spearmanr
from clinical_crsr import load_crsr_table
from reproduction_config import CRSR_XLSX, FITTED_DIR, FIGURES_DIR as CONFIG_FIGURES_DIR

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
FIGURES_DIR = str(CONFIG_FIGURES_DIR)
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_DIR = str(FITTED_DIR)
CRSR_FILE = str(CRSR_XLSX)

OUTPUT_FIG = os.path.join(FIGURES_DIR, "SM_Figure3.png")
OUTPUT_CSV = os.path.join(FIGURES_DIR, "Figure_4_Time_Adjusted_stats.csv")

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


def load_crsr(subject_ids):
    return load_crsr_table(CRSR_FILE, subject_ids, len(SUBSCALE_NAMES))


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


def time_adjusted_association(data, x_col, y_col):
    """Original rmcorr ANCOVA with session as the sole added covariate.

    Pingouin's rmcorr corresponds to y ~ x + C(subject). We retain its
    conventional test and add C(session), which uses six degrees of freedom.
    """
    df = filtered_pair_data(data, x_col, y_col)
    model_df = df.rename(columns={x_col: "x", y_col: "y"}).copy()
    fit = smf.ols("y ~ x + C(subject) + C(session)", data=model_df).fit()
    if np.linalg.matrix_rank(fit.model.exog) != fit.model.exog.shape[1]:
        raise ValueError(f"Rank-deficient model for {x_col} and {y_col}")
    x_resid = smf.ols("x ~ C(subject) + C(session)", data=model_df).fit().resid
    y_resid = smf.ols("y ~ C(subject) + C(session)", data=model_df).fit().resid
    partial_r = float(np.corrcoef(x_resid, y_resid)[0, 1])
    model_df["x_resid"] = x_resid
    model_df["y_resid"] = y_resid
    ci = fit.conf_int().loc["x"]
    return {
        "data": model_df,
        "r": partial_r,
        "beta": float(fit.params["x"]),
        "p": float(fit.pvalues["x"]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "dof": float(fit.df_resid),
        "n": len(model_df),
        "subjects": model_df["subject"].nunique(),
    }


def verify_original_rmcorr_equivalence(data, x_col, y_col, rm):
    """Confirm the no-time ANCOVA matches the original Pingouin result."""
    df = filtered_pair_data(data, x_col, y_col)
    model_df = df.rename(columns={x_col: "x", y_col: "y"})
    fit = smf.ols("y ~ x + C(subject)", data=model_df).fit()
    t_value = fit.tvalues["x"]
    r_value = t_value / np.sqrt(t_value ** 2 + fit.df_resid)
    if not (np.isclose(r_value, rm["r"], atol=1e-8)
            and np.isclose(fit.pvalues["x"], rm["p"], atol=1e-8)):
        raise ValueError(f"No-time model differs from Pingouin for {x_col}/{y_col}")


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
    ax.scatter(df["x_resid"], df["y_resid"], s=90, facecolor="#d3d3d3",
               edgecolor="black", alpha=0.8, zorder=3)
    xs = np.linspace(df["x_resid"].min(), df["x_resid"].max(), 100)
    ax.plot(xs, stat["beta"] * xs, color="#d62728", linewidth=2.5, zorder=2)
    ax.axhline(0, color="0.75", linewidth=0.8, zorder=1)
    ax.axvline(0, color="0.75", linewidth=0.8, zorder=1)

    stat_text = f"Partial r = {stat['r']:.3f}\np-value {p_text(stat['p'])}"
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

    ax.set_xlabel(f"Adjusted {DISPLAY_NAME.get(x_col, x_col)}", fontweight="bold")
    ax.set_ylabel("Adjusted CRS-R score", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    params, pnames, subject_ids = load_model_parameters()
    crsr_total, crsr_subscales = load_crsr(subject_ids)
    data = build_long_dataframe(params, pnames, subject_ids, crsr_total, crsr_subscales)

    stats_rows = []
    total_stats = {}

    print("\nTime-adjusted correlations for CRS-R total")
    print("Same rmcorr ANCOVA, with one added categorical session term.")
    print(f"{'Parameter':<10} {'partial r':>10} {'p-value':>11} {'n':>5} {'raw rm r':>10}")
    print("-" * 53)

    for param in TARGET_PARAMS:
        adj = time_adjusted_association(data, param, "crsr_total")
        rm = run_pingouin_rmcorr(data, param, "crsr_total")
        verify_original_rmcorr_equivalence(data, param, "crsr_total", rm)
        total_stats[param] = adj

        stats_rows.append(
            {
                "outcome": "crsr_total",
                "parameter": param,
                "partial_r_patient_session_adjusted": adj["r"],
                "beta": adj["beta"],
                "p_value": adj["p"],
                "beta_ci_low": adj["ci_low"],
                "beta_ci_high": adj["ci_high"],
                "degrees_of_freedom": adj["dof"],
                "n_observations": adj["n"],
                "n_patients": adj["subjects"],
                "original_rmcorr_r": rm["r"],
                "original_rmcorr_p": rm["p"],
            }
        )

        print(f"{param:<10} {adj['r']:>10.3f} {adj['p']:>11.4f} {adj['n']:>5} {rm['r']:>10.3f}")

    corr_matrix = np.full((len(TARGET_PARAMS), len(SUBSCALE_NAMES)), np.nan)
    pval_matrix = np.full_like(corr_matrix, np.nan)

    print("\nTime-adjusted correlations for CRS-R subscales")
    for i, param in enumerate(TARGET_PARAMS):
        for j, subscale in enumerate(SUBSCALE_NAMES):
            adj = time_adjusted_association(data, param, subscale)
            rm = run_pingouin_rmcorr(data, param, subscale)
            verify_original_rmcorr_equivalence(data, param, subscale, rm)

            corr_matrix[i, j] = adj["r"]
            pval_matrix[i, j] = adj["p"]

            stats_rows.append(
                {
                    "outcome": subscale,
                    "parameter": param,
                    "partial_r_patient_session_adjusted": adj["r"],
                    "beta": adj["beta"],
                    "p_value": adj["p"],
                    "beta_ci_low": adj["ci_low"],
                    "beta_ci_high": adj["ci_high"],
                    "degrees_of_freedom": adj["dof"],
                    "n_observations": adj["n"],
                    "n_patients": adj["subjects"],
                    "original_rmcorr_r": rm["r"],
                    "original_rmcorr_p": rm["p"],
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
    fig.colorbar(cax, ax=ax_heat, fraction=0.046, pad=0.04)

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
    print("  Adjusted model: outcome ~ parameter + C(session) + C(subject).")
    print("  Patient effects are the same as in Pingouin rmcorr; no cluster correction was added.")
    print("  Original Pingouin rmcorr values are retained in the CSV for comparison.")
    plt.close(fig)


if __name__ == "__main__":
    main()
