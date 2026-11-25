"""Patient-level bootstrap CIs for the manuscript's follow-up contrasts.

Original 40-Hz fits, fixed treatment allocation, signed within-patient changes
from Day 1 pre-stimulation to the 24-h and one-week follow-ups. The full
nonparametric bootstrap distribution is evaluated by enumerating each unique
participant resample and weighting it by its multinomial probability.
"""

import csv
import itertools
import math

import numpy as np
from scipy.io import loadmat
from scipy.stats import mannwhitneyu
from reproduction_config import FITTED_DIR, FIGURES_DIR


OUTPUT_DIR = FIGURES_DIR
ACTIVE = ["sub-001", "sub-004", "sub-005", "sub-006", "sub-007", "sub-008"]
SHAM = ["sub-009", "sub-010", "sub-011", "sub-012", "sub-013"]
PARAMETERS = [("X", "S_CC", r"$S_{CC}$"), ("Y", "S_CT", r"$S_{CT}$"),
              ("Z", "S_IT", r"$S_{IT}$"), ("Alpha", "alpha", r"$\alpha$"),
              ("Beta", "beta", r"$\beta$"), ("t0", "t0", r"$t_0$"),
              ("EMGa", "EMG", "EMG")]
SESSIONS = [(5, "24 hours"), (6, "1 week")]


def load_changes():
    expected = set(ACTIVE + SHAM)
    files = {p.stem.removesuffix("_model_fits"): p
             for p in FITTED_DIR.glob("*_model_fits.mat")}
    if set(files) != expected:
        raise ValueError(f"Unexpected cohort: {sorted(files)}")
    changes = []
    for subject in ACTIVE + SHAM:
        source = loadmat(files[subject], simplify_cells=True)
        names = list(source["param_names"])
        cols = [names.index(name) for name, _, _ in PARAMETERS]
        values = np.asarray(source["params_sub"])[:, 0, :][:, cols]
        changes.append(values[[s for s, _ in SESSIONS], :] - values[0, :])
    changes = np.stack(changes)
    if not np.isfinite(changes).all():
        raise ValueError("Missing original baseline or follow-up estimates")
    return changes


def rb(x, y):
    # P(active > sham) - P(active < sham); ties contribute zero.
    return np.mean(np.sign(x[..., :, None] - y[..., None, :]), axis=(-2, -1))


def bootstrap_configurations(n):
    """Counts of participants in all distinct size-n resamples, with weights."""
    counts, probabilities = [], []
    for draw in itertools.combinations_with_replacement(range(n), n):
        multiplicities = np.bincount(draw, minlength=n)
        prob = (math.factorial(n) /
                math.prod(math.factorial(int(k)) for k in multiplicities) / n**n)
        counts.append(multiplicities)
        probabilities.append(prob)
    weights = np.asarray(probabilities)
    assert np.isclose(weights.sum(), 1)
    return np.asarray(counts), weights


def calculate():
    deltas = load_changes()
    a, s = deltas[:len(ACTIVE)], deltas[len(ACTIVE):]
    ca, wa = bootstrap_configurations(len(a))
    cs, ws = bootstrap_configurations(len(s))
    weights = (wa[:, None] * ws[None, :]).ravel()
    n_pairs = len(a) * len(s)
    results = []
    for p, (_, name, _) in enumerate(PARAMETERS):
        for t, (_, timepoint) in enumerate(SESSIONS):
            av, sv = a[:, t, p], s[:, t, p]
            observed = mannwhitneyu(av, sv, alternative="two-sided", method="exact")
            # Check that the original samples have no ties. SciPy's exact MW
            # p-value is appropriate here; bootstrap samples can include ties.
            assert len(np.unique(np.r_[av, sv])) == len(av) + len(sv)
            effect = float(rb(av, sv))
            assert np.isclose(effect, 2 * observed.statistic / (len(av) * len(sv)) - 1)
            # Each configuration's integer score is wins minus losses;
            # aggregate its multinomial probability in the discrete CDF.
            scores = (ca @ np.sign(av[:, None] - sv[None, :]).astype(int) @ cs.T).ravel()
            mass = np.bincount(scores + n_pairs, weights=weights, minlength=2*n_pairs+1)
            cdf = np.cumsum(mass / mass.sum())
            ci_low, ci_high = (np.searchsorted(cdf, [0.025, 0.975]) - n_pairs) / n_pairs
            results.append(dict(parameter=name, followup=timepoint,
                                n_active=len(av), n_sham=len(sv),
                                U_active=float(observed.statistic),
                                rank_biserial=effect,
                                ci95_lower=float(ci_low), ci95_upper=float(ci_high),
                                p_uncorrected=float(observed.pvalue),
                                bootstrap_configurations=len(weights)))
    return results


def write_table(rows):
    lookup = {(row["parameter"], row["followup"]): row for row in rows}
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Between-group comparisons of signed changes in the model parameters from Day 1 (pre-stimulation) to the 24-hour and one-week follow-ups. Signed rank-biserial correlations ($r_{rb}$) are reported with percentile 95\% confidence intervals (CI) from a participant-level bootstrap within each group (6 active, 5 sham), evaluated over all distinct resamples and weighted by their probabilities. Positive values indicate more positive changes in the active group. Uncorrected \textit{p}-values were obtained using two-sided exact Mann--Whitney \textit{U} tests.}",
        r"\label{tab:supp-followup-effect-sizes}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{24 hours} & \multicolumn{3}{c}{1 week} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Parameter & $r_{rb}$ & 95\% CI & $p$ & $r_{rb}$ & 95\% CI & $p$ \\",
        r"\midrule",
    ]
    for _, name, label in PARAMETERS:
        line = [label]
        for _, followup in SESSIONS:
            row = lookup[(name, followup)]
            line.extend([f"${row['rank_biserial']:.3f}$",
                         f"$[{row['ci95_lower']:.3f},\ {row['ci95_upper']:.3f}]$",
                         f"{row['p_uncorrected']:.4f}"])
        lines.append(" & ".join(line) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT_DIR / "Supplementary_Table_T2_rank_biserial_CIs.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = calculate()
    output = OUTPUT_DIR / "Followup_rank_biserial_95CI.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_table(rows)
    print(f"{'Parameter':<8} {'Follow-up':<9} {'r':>7} {'95% CI':>20} {'p':>9}")
    for row in rows:
        print(f"{row['parameter']:<8} {row['followup']:<9} {row['rank_biserial']:+7.3f} "
              f"[{row['ci95_lower']:+6.3f}, {row['ci95_upper']:+6.3f}] {row['p_uncorrected']:9.4f}")
    print("Saved", output)
    print("Saved", OUTPUT_DIR / "Supplementary_Table_T2_rank_biserial_CIs.tex")


if __name__ == "__main__":
    main()
