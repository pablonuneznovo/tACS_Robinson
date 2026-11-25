"""Compute the Table T2-style follow-up intervals for the 25 Hz model refits."""
import pandas as pd
import analyze_followup_rank_biserial_cis as analysis
from reproduction_config import FITTED_25HZ_DIR, FIGURES_DIR


def main():
    analysis.FITTED_DIR = FITTED_25HZ_DIR
    results = pd.DataFrame(analysis.calculate())
    path = FIGURES_DIR / 'Followup_25Hz_rank_biserial_95CI.csv'
    results.to_csv(path, index=False)
    print(results.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
