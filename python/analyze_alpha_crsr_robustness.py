"""Sensitivity of Figure 4 alpha–CRS-R correlations to 100 repeat fits.

Only sessions 1 and 7 were refitted. Sessions 2–6 retain their published fits.
Each draw selects one of 100 fits independently per refitted recording. The
patient/session design and clinical scores remain unchanged.
"""

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from scipy.io import loadmat
from scipy.stats import t as student_t
from reproduction_config import FIGURES_DIR, REPEAT_FITS_DIR
import generate_sm_figure_3 as paper

REPEATS_DIR = REPEAT_FITS_DIR
N_DRAWS = 10000
RNG = np.random.default_rng(20260928)
OUTCOMES = ["crsr_total", "Auditory", "Visual", "Motor", "Oromotor",
            "Communication", "Arousal"]

def residual_maker(frame, adjust_time):
    subject = pd.get_dummies(frame["subject"], drop_first=True, dtype=float)
    controls = [np.ones((len(frame), 1)), subject.to_numpy()]
    if adjust_time:
        session = pd.get_dummies(frame["session"].astype(str), drop_first=True,
                                 dtype=float)
        controls.append(session.to_numpy())
    matrix = np.column_stack(controls)
    assert np.linalg.matrix_rank(matrix) == matrix.shape[1]
    return np.eye(len(frame)) - matrix @ np.linalg.pinv(matrix), matrix.shape[1]


def correlations(x_draws, y, residual, n_controls):
    """OLS partial r and its conventional t-test for every draw."""
    x_res = residual @ x_draws.T
    y_res = residual @ y
    denominator = np.sqrt(np.sum(x_res ** 2, axis=0) * np.sum(y_res ** 2))
    r = np.sum(x_res * y_res[:, None], axis=0) / denominator
    df = len(y) - n_controls - 1
    t_stat = r * np.sqrt(df / (1 - r ** 2))
    p = 2 * student_t.sf(np.abs(t_stat), df)
    return r, p, df


def main():
    params, pnames, subjects = paper.load_model_parameters()
    crsr_total, crsr_subscales = paper.load_crsr(subjects)
    data = paper.build_long_dataframe(params, pnames, subjects, crsr_total,
                                      crsr_subscales)
    assert len(subjects) == 11 and set(data["session"]) == set(range(1, 8))
    assert pnames[[name.lower() for name in pnames].index("alpha")].lower() == "alpha"

    files = list(REPEATS_DIR.glob("*.mat"))
    assert len(files) == 2200, len(files)
    alpha_repeats = {}
    originals = {}
    for path in files:
        match = re.fullmatch(r"(sub-\d+)_ses-(01|07)_repeat-(\d{3})\.mat", path.name)
        assert match, path
        subject, session_text, repeat_text = match.groups()
        session, k = int(session_text), int(repeat_text) - 1
        key = (subject, session)
        result = loadmat(path, simplify_cells=True)["result"]
        assert result["success"], path
        if key not in alpha_repeats:
            alpha_repeats[key] = np.full(100, np.nan)
            originals[key] = float(result["original_parameters"][3])
        alpha_repeats[key][k] = float(result["parameters"][3])
        assert np.isclose(originals[key], result["original_parameters"][3])
    assert len(alpha_repeats) == 22
    assert all(np.isfinite(values).all() for values in alpha_repeats.values())

    # Verify the saved refit baseline is precisely the published Figure 4 value.
    for (subject, session), original in originals.items():
        row = data[(data["subject"] == subject) & (data["session"] == session)]
        assert len(row) == 1, (subject, session)
        assert np.isclose(float(row["Alpha"].iloc[0]), original,
                          atol=1e-7, rtol=1e-7), (subject, session,
                                                    float(row["Alpha"].iloc[0]), original)

    x = np.broadcast_to(data["Alpha"].to_numpy(), (N_DRAWS, len(data))).copy()
    for column, row in data.reset_index(drop=True).iterrows():
        key = (row["subject"], int(row["session"]))
        if key in alpha_repeats:
            x[:, column] = RNG.choice(alpha_repeats[key], size=N_DRAWS)

    published = pd.read_csv(FIGURES_DIR / "Figure_4_Time_Adjusted_stats.csv")
    rows = []
    total_draws = {}
    for outcome in OUTCOMES:
        frame = paper.filtered_pair_data(data, "Alpha", outcome)
        indices = frame.index.to_numpy()
        y = frame[outcome].to_numpy(dtype=float)
        x_original = frame["Alpha"].to_numpy(dtype=float)
        x_values = x[:, indices]
        reported = published[(published["outcome"] == outcome) &
                             (published["parameter"] == "Alpha")].iloc[0]
        for label, adjust_time in [("repeated-measures", False),
                                   ("time-adjusted", True)]:
            residual, n_controls = residual_maker(frame, adjust_time)
            r_draw, p_draw, df = correlations(x_values, y, residual, n_controls)
            r_original, p_original, _ = correlations(x_original[None, :], y,
                                                       residual, n_controls)
            target_r = (float(reported["partial_r_patient_session_adjusted"])
                        if adjust_time else float(reported["original_rmcorr_r"]))
            target_p = (float(reported["p_value"]) if adjust_time
                        else float(reported["original_rmcorr_p"]))
            assert np.isclose(r_original[0], target_r, atol=1e-8), (outcome, label)
            assert np.isclose(p_original[0], target_p, atol=1e-8), (outcome, label)
            rows.append(dict(
                outcome=outcome, analysis=label, observations=len(frame),
                patients=frame["subject"].nunique(), degrees_of_freedom=df,
                refitted_sessions="1 and 7 only", draws=N_DRAWS,
                original_r=float(r_original[0]), original_p=float(p_original[0]),
                refit_r_median=float(np.median(r_draw)),
                refit_r_q025=float(np.quantile(r_draw, .025)),
                refit_r_q975=float(np.quantile(r_draw, .975)),
                refit_r_min=float(np.min(r_draw)),
                refit_r_max=float(np.max(r_draw)),
                fraction_positive=float(np.mean(r_draw > 0)),
                fraction_p_below_005=float(np.mean(p_draw < .05)),
                refit_p_median=float(np.median(p_draw)),
            ))
            if outcome == "crsr_total":
                total_draws[label] = r_draw

    output = FIGURES_DIR / "Alpha_CRSR_repeat_fit_sensitivity_100.csv"
    pd.DataFrame(rows).to_csv(output, index=False)

    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.titlesize": 16, "axes.labelsize": 14,
                         "axes.grid": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, (label, color) in zip(axes, [("repeated-measures", "#A1C9F4"),
                                         ("time-adjusted", "#FF9F9B")]):
        values = total_draws[label]
        row = next(item for item in rows if item["outcome"] == "crsr_total"
                   and item["analysis"] == label)
        ax.hist(values, bins=45, density=True, color=color, edgecolor="black",
                linewidth=.5, alpha=.85)
        ax.axvline(row["original_r"], color="black", linewidth=1.8,
                   label="Original fit")
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_title("Repeated-measures" if label == "repeated-measures"
                     else "Time-adjusted", fontweight="bold")
        ax.set_xlabel("Alpha–CRS-R correlation (r)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Density")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    figure = FIGURES_DIR / "Alpha_CRSR_repeat_fit_sensitivity_100.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)
    print(f"Validated {len(files)} repeat fits and 22 exact original-alpha matches")
    print("Saved", output)
    print("Saved", figure)
    for row in rows:
        print(row["outcome"], row["analysis"], "original r", round(row["original_r"], 3),
              "refit median", round(row["refit_r_median"], 3),
              "range", (round(row["refit_r_q025"], 3), round(row["refit_r_q975"], 3)),
              "positive", round(row["fraction_positive"], 3),
              "p<.05", round(row["fraction_p_below_005"], 3))


if __name__ == "__main__":
    main()
