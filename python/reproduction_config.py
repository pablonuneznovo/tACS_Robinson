"""Portable paths for reproducing the original paper analyses.

Set VANGUARD_ROOT to the directory containing the raw Vanguard subject folders.
The remaining variables can be overridden when the clinical files or fitted
results are stored elsewhere.
"""

import os
from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


VANGUARD_ROOT = _path_from_env("VANGUARD_ROOT", REPRO_ROOT / "data" / "Vanguard data")
FITTED_DIR = _path_from_env(
    "VANGUARD_FITTED_DIR",
    VANGUARD_ROOT / "Fitted_Parameters Previous Session Start XYZab Alpha emphasized",
)
PSD_DIR = _path_from_env("VANGUARD_PSD_DIR", FITTED_DIR)
CRSR_XLSX = _path_from_env("VANGUARD_CRSR_XLSX", VANGUARD_ROOT / "Supp. Table 1.xlsx")
CRSR_CSV = _path_from_env("VANGUARD_CRSR_CSV", VANGUARD_ROOT / "updated_crsr_categories_pilot.csv")
FIGURES_DIR = _path_from_env("PAPER_FIGURES_DIR", REPRO_ROOT / "figures")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
