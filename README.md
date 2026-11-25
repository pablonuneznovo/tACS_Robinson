# Reproducing the original paper results

This folder contains the minimal analysis scripts for the original paper
pipeline and its Supplementary Information.

## Contents

- `matlab/N1_fit_data_XYZab.m` — FieldTrip preprocessing, Welch PSD estimation,
  and original Robinson/BrainTrak fitting to approximately 1.1–40 Hz.
- `matlab/N1_fit_data_XYZab_cutoff25.m` — fitting to approximately 1.1–25 Hz for Supplementary Figure S6.
- `matlab/N10_repeat_parameter_fits.m` — 100 repeat fits for Day 1 and the one-week follow-up.
- `python/generate_figure_2.py` — Figure 2 (`Figure2.png`), with individual Day 1, 24-hour and 1-week trajectories.
- `python/generate_figure_3.py` — Figure 3 (`Figure3.png`).
- `python/generate_figure_4.py` — Figure 4 (`Figure4.png`), session-adjusted CRS-R correlations and their statistics table.
- `python/generate_sm_figure_1.py` — Supplementary Figure S1 (`SM_Figure1.png`).
- `python/generate_sm_figure_2.py` — Supplementary Figure S2 (`SM_Figure2.png`).
- `python/generate_sm_figure_3.py` — Supplementary Figure S3 (`SM_Figure3.png`), repeated-measures CRS-R correlations without session adjustment and their statistics table.
- `python/generate_sm_figure_4.py` — Supplementary Figure S4 (`SM_Figure4.png`).
- `python/generate_sm_figure_5.py` — Supplementary Figure S5 (`SM_Figure5.png`).
- `python/generate_sm_figure_6.py` — Supplementary Figure S6 (`SM_Figure6.png`).
- `python/generate_sm_figure_7.py` — Supplementary Figure S7 (`SM_Figure7.png`).
- `python/generate_sm_figure_8.py` — Supplementary Figure S8 (`SM_Figure8.png`), the sTBI-only sensitivity figure.
- `python/generate_sm_figure_9.py` — Supplementary Figure S9 (`SM_Figure9.png`), acute within-session changes.
- `python/analyze_alpha_crsr_robustness.py` — alpha–CRS-R repeat-fit sensitivity analysis reported in the manuscript.
- `python/analyze_followup_rank_biserial_cis.py` — signed effect sizes, confidence intervals and uncorrected p-values at both follow-ups (Supplementary Table T2).
- `python/analyze_stbi_followup_sensitivity.py` — full-cohort and sTBI-only exact follow-up comparisons.
- `python/analyze_stbi_followup_confidence_intervals.py` — participant-bootstrap 95% CIs for the sTBI-only follow-up contrasts.
- `python/analyze_25hz_followup_confidence_intervals.py` — participant-bootstrap 95% CIs for follow-up contrasts from the 25-Hz refits.
- `python/clinical_crsr.py` — shared CRS-R table reader.
- `python/reproduction_config.py` — portable data and output path configuration.

No participant data, BrainTrak source, corticothalamic-model source, or FieldTrip
source is included. These must be obtained separately and are not redistributed
by this repository. The participant data is available from the authors upon reasonable request.

## External MATLAB dependencies

Install or clone:

- [BrainTrak](https://github.com/BrainDynamicsUSYD/braintrak)
- [corticothalamic-model](https://github.com/BrainDynamicsUSYD/corticothalamic-model)
- [FieldTrip](https://www.fieldtriptoolbox.org/download/)

Before running the MATLAB scripts, set these MATLAB environment variables or
replace the corresponding `getenv` calls in the scripts:

```matlab
setenv('VANGUARD_ROOT', 'D:\path\to\Vanguard data');
setenv('FIELDTRIP_DIR', 'D:\path\to\fieldtrip');
setenv('BRAINTRAK_DIR', 'D:\path\to\braintrak');
setenv('CORTICOTHALAMIC_MODEL_DIR', 'D:\path\to\corticothalamic-model');
```

`N1_fit_data_XYZab.m` writes fitted files to:

```text
<VANGUARD_ROOT>/Fitted_Parameters Previous Session Start XYZab Alpha emphasized/
```

`N1_fit_data_XYZab_cutoff25.m` writes the separate 25-Hz fitted files to:

```text
<VANGUARD_ROOT>/Fitted_Parameters Previous Session Start XYZab Alpha emphasized cutoff25Hz/
```

After running the original 40-Hz fitting script, run `N10_repeat_parameter_fits.m`
to generate the 100 repeat fits required for Supplementary Figure S7 and the
alpha–CRS-R sensitivity analysis. By default, the results are saved in:

```text
<VANGUARD_ROOT>/repeat_fit_results 100/
```

## Python setup and execution

From this folder:

```bash
python -m pip install -r requirements.txt
```

Set the data paths before running the figure scripts. The fitted parameter and
PSD folders default to the output folder created by the MATLAB script. Override
paths when the clinical tables, PSD files, or repeat fits are stored elsewhere:

```bash
export VANGUARD_ROOT=/path/to/Vanguard\ data
export VANGUARD_CRSR_XLSX=/path/to/Supp.\ Table\ 1.xlsx
export VANGUARD_CRSR_CSV=/path/to/updated_crsr_categories_pilot.csv
export VANGUARD_FITTED_25HZ_DIR=/path/to/25Hz_fitted_results
export VANGUARD_REPEAT_FITS_DIR=/path/to/repeat_fit_results\ 100
```

On Windows PowerShell, use `$env:VANGUARD_ROOT = 'D:\path\to\Vanguard data'`.
Set `SHOW_FIGURE=0` to save figures without opening interactive plotting windows.

Run the main figures with:

```bash
python python/generate_figure_2.py
python python/generate_figure_3.py
python python/generate_figure_4.py
```

Run the supplementary figures with:

```bash
python python/generate_sm_figure_1.py
python python/generate_sm_figure_2.py
python python/generate_sm_figure_3.py
python python/generate_sm_figure_4.py
python python/generate_sm_figure_5.py
python python/generate_sm_figure_6.py
python python/generate_sm_figure_7.py
python python/generate_sm_figure_8.py
python python/generate_sm_figure_9.py
```

Supplementary Figure S6 requires the 25-Hz fitted results, and S7 requires the
100 repeat fits.

After generating Figure 4, run the alpha–CRS-R repeat-fit analysis:

```bash
python python/analyze_alpha_crsr_robustness.py
```

Generate Supplementary Table T2 with:

```bash
python python/analyze_followup_rank_biserial_cis.py
python python/analyze_stbi_followup_sensitivity.py
python python/analyze_stbi_followup_confidence_intervals.py
python python/analyze_25hz_followup_confidence_intervals.py
```

Figures and associated statistics are written to `figures/`. The scripts retain
the subject ordering and group indices used in the original analysis, so check
those indices if the dataset is reorganized.
