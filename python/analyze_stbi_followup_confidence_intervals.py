"""Same participant-resampling percentile intervals as Table T2, for sTBI only."""
from reproduction_config import FIGURES_DIR
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from analyze_followup_rank_biserial_cis import ACTIVE, SHAM, PARAMETERS, SESSIONS, load_changes, bootstrap_configurations, rb


def main():
    changes = load_changes()
    active = [i for i, sid in enumerate(ACTIVE) if sid not in {'sub-006', 'sub-008'}]
    sham = list(range(len(ACTIVE), len(ACTIVE) + len(SHAM)))
    n_a, n_s = len(active), len(sham)
    ca, wa = bootstrap_configurations(n_a)
    cs, ws = bootstrap_configurations(n_s)
    weights = (wa[:, None] * ws[None, :]).ravel()
    n_pairs = n_a * n_s
    rows = []
    for p, (_, name, _) in enumerate(PARAMETERS):
        for t, (_, followup) in enumerate(SESSIONS):
            a, s = changes[active, t, p], changes[sham, t, p]
            assert len(np.unique(np.r_[a,s])) == len(a) + len(s)
            test = mannwhitneyu(a, s, alternative='two-sided', method='exact')
            r = float(rb(a, s))
            assert np.isclose(r, 2*test.statistic/n_pairs-1)
            scores = (ca @ np.sign(a[:, None] - s[None, :]).astype(int) @ cs.T).ravel()
            mass = np.bincount(scores+n_pairs, weights=weights, minlength=2*n_pairs+1)
            cdf = np.cumsum(mass/mass.sum())
            low, high = (np.searchsorted(cdf, [0.025, 0.975])-n_pairs)/n_pairs
            rows.append(dict(parameter=name, followup=followup, n_active=n_a, n_sham=n_s,
                             U_active=float(test.statistic), rank_biserial=r, ci95_lower=low,
                             ci95_upper=high, p_uncorrected=float(test.pvalue),
                             bootstrap_configurations=len(weights)))
    frame=pd.DataFrame(rows)
    output=FIGURES_DIR/'Followup_sTBI_rank_biserial_95CI.csv'
    frame.to_csv(output,index=False)
    print(frame.to_string(index=False, float_format=lambda v:f'{v:.4f}'))


if __name__ == '__main__':
    main()
