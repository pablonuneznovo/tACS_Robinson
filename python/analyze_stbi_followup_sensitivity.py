"""Reproduce the exact follow-up contrasts restricted to the nine sTBI cases."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from reproduction_config import FIGURES_DIR
from analyze_followup_rank_biserial_cis import ACTIVE, SHAM, PARAMETERS, SESSIONS, load_changes


def main():
    changes = load_changes()
    # Two active cases had anoxic (HIE) injury, as recorded in participants.txt.
    included = [i for i, sid in enumerate(ACTIVE) if sid not in {'sub-006', 'sub-008'}]
    sham = list(range(len(ACTIVE), len(ACTIVE)+len(SHAM)))
    rows = []
    for subset, active in [('all', list(range(len(ACTIVE)))), ('sTBI', included)]:
        for t, (_, followup) in enumerate(SESSIONS):
            for p, (param, _, _) in enumerate(PARAMETERS):
                a, b = changes[active, t, p], changes[sham, t, p]
                assert len(np.unique(np.r_[a,b])) == len(a) + len(b), 'Unexpected tied changes'
                mw = mannwhitneyu(a, b, alternative='two-sided', method='exact')
                rows.append({'cohort': subset, 'followup': followup, 'parameter': param,
                             'n_active': len(a), 'n_sham': len(b),
                             'median_active_change': np.median(a), 'median_sham_change': np.median(b),
                             'U_active': mw.statistic, 'rank_biserial': 2*mw.statistic/(len(a)*len(b))-1,
                             'p_uncorrected': mw.pvalue})
    output = FIGURES_DIR / 'Followup_MW_all_and_sTBI.csv'
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
