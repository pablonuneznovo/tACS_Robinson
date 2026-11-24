"""Read the clinical assessment table using the numeric case/session identifiers."""

import numpy as np
import pandas as pd


def _integer(value):
    try:
        number = float(value)
        return int(number) if np.isfinite(number) and number.is_integer() else None
    except (TypeError, ValueError, OverflowError):
        return None


def load_crsr_table(path, subject_ids, n_subscales=6):
    """Return CRS-R totals and subscales as subject x session x pre/post arrays.

    Sessions 1–5 contain pre/post stimulation assessments. Sessions 6 and 7
    have one assessment each, at 24 h and 1 week post-protocol, respectively.
    The numeric session is authoritative for these follow-ups. A conflicting
    timepoint label is reported, rather than allowing it to overwrite another
    session. Duplicate assessments are rejected.
    """
    totals = np.full((len(subject_ids), 7, 2), np.nan)
    subscales = np.full((len(subject_ids), 7, 2, n_subscales), np.nan)
    case_indices = {int(''.join(c for c in sid if c.isdigit())): i
                    for i, sid in enumerate(subject_ids)}
    if len(case_indices) != len(subject_ids):
        raise ValueError("Duplicate numeric case identifiers in fitted subject files")

    frame = pd.read_excel(path, header=None)
    case = session = None
    assigned = set()
    for row_index, row in frame.iterrows():
        new_case = _integer(row.iloc[0])
        if new_case is not None:
            if new_case != case:
                session = None
            case = new_case
        new_session = _integer(row.iloc[1])
        if new_session is not None:
            session = new_session
        if case not in case_indices or session not in range(1, 8):
            continue
        timepoint = str(row.iloc[3]).strip().lower()
        if not ("pre" in timepoint or "post" in timepoint or
                "protocol" in timepoint or "week" in timepoint):
            continue

        if session in (6, 7):
            run = 0
            expected = "protocol" if session == 6 else "week"
            if expected not in timepoint or (session == 6 and "week" in timepoint):
                print(f"CRS-R label mismatch at spreadsheet row {row_index + 1}: "
                      f"case {case}, session {session}, label {timepoint!r}; "
                      "using the numeric session identifier.")
        elif "pre" in timepoint:
            run = 0
        elif "post" in timepoint and "week" not in timepoint and "protocol" not in timepoint:
            run = 1
        else:
            raise ValueError(f"Unexpected assessment label for case {case}, session {session}: {timepoint!r}")

        key = (case_indices[case], session - 1, run)
        if key in assigned:
            raise ValueError(f"Duplicate CRS-R assessment: case {case}, session {session}, run {run}")
        assigned.add(key)
        totals[key] = pd.to_numeric(row.iloc[10], errors="coerce")
        subscales[key] = pd.to_numeric(row.iloc[4:4+n_subscales], errors="coerce").to_numpy(dtype=float)

    return totals, subscales
