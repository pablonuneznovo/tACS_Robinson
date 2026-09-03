# Reproducing the original paper results

This folder contains the minimal analysis scripts for the original paper
pipeline. It uses the original 40-Hz fitting range; the exploratory 25-Hz
cutoff scripts are intentionally excluded.

## Contents

- `matlab/N1_fit_data_XYZab.m` — FieldTrip preprocessing, Welch PSD estimation,
  and original Robinson/BrainTrak fitting to approximately 1.1–40 Hz.
- `python/N12_build_figure_1.py` — Figure 1.
- `python/N19_build_figure_2_with_t0.py` — Figure 2.
- `python/N13_build_figure_3.py` — Figure 3.
- `python/N18_build_figure_4_pingouin_rmcorr.py` — Figure 4 and its statistics table.
- `python/N20_build_figure_5_direct.py` — Figure 5.
- `python/reproduction_config.py` — portable data and output path configuration.

No participant data, BrainTrak source, corticothalamic-model source, or FieldTrip
source is included. These must be obtained separately and are not redistributed
by this repository.

## External MATLAB dependencies

Install or clone:

- [BrainTrak](https://github.com/BrainDynamicsUSYD/braintrak)
- [corticothalamic-model](https://github.com/BrainDynamicsUSYD/corticothalamic-model)
- [FieldTrip](https://www.fieldtriptoolbox.org/download/)

Before running the MATLAB script, set these MATLAB environment variables or
replace the corresponding `getenv` calls in the script:

```matlab
setenv('VANGUARD_ROOT', 'D:\\path\\to\\Vanguard data');
setenv('FIELDTRIP_DIR', 'D:\\path\\to\\fieldtrip');
setenv('BRAINTRAK_DIR', 'D:\\path\\to\\braintrak');
setenv('CORTICOTHALAMIC_MODEL_DIR', 'D:\\path\\to\\corticothalamic-model');
```

The script writes fitted files to:

```text
<VANGUARD_ROOT>/Fitted_Parameters Previous Session Start XYZab Alpha emphasized/
```

## Python setup and execution

From this folder:

```bash
python -m pip install -r requirements.txt
```

Set the data paths before running the figure scripts. The fitted parameter and
PSD folders default to the output folder created by the MATLAB script. Override
paths when the clinical tables or PSD files are stored elsewhere:

```bash
export VANGUARD_ROOT=/path/to/Vanguard\ data
export VANGUARD_CRSR_XLSX=/path/to/Supp.\ Table\ 1.xlsx
export VANGUARD_CRSR_CSV=/path/to/updated_crsr_categories_pilot.csv
```

On Windows PowerShell, use `$env:VANGUARD_ROOT = 'D:\path\to\Vanguard data'`.

Run the main figures with:

```bash
python python/N12_build_figure_1.py
python python/N19_build_figure_2_with_t0.py
python python/N13_build_figure_3.py
python python/N18_build_figure_4_pingouin_rmcorr.py
python python/N20_build_figure_5_direct.py
```

Figures and Figure 4 statistics are written to `figures/`. The scripts retain
the subject ordering and group indices used in the original analysis, so check
those indices if the dataset is reorganized.
