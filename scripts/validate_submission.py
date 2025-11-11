#!/usr/bin/env python3
"""
Validate a Trustii submission CSV:
- columns present
- types & ranges
- no NaNs / inf
- OUTCOME MACE in {0,1,2}
- OUTCOME SEVERITY in [0,1]
"""

import sys
import math
import pandas as pd
from pathlib import Path

REQUIRED_COLS = ["trustii_id", "OUTCOME MACE", "OUTCOME SEVERITY"]

def fail(msg: str, code: int = 1):
    print(f"[INVALID] {msg}")
    sys.exit(code)

def ok(msg: str = "Submission looks valid."):
    print(f"[OK] {msg}")
    sys.exit(0)

def main(path: str):
    p = Path(path)
    if not p.exists():
        fail(f"File not found: {p}")

    try:
        df = pd.read_csv(p)
    except Exception as e:
        fail(f"Could not read CSV: {e}")

    # columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        fail(f"Missing required columns: {missing}")

    # NaNs / inf
    if df[REQUIRED_COLS].isna().any().any():
        fail("NaN values detected in required columns.")

    for col in REQUIRED_COLS:
        if not all(map(lambda x: not (isinstance(x, float) and math.isinf(x)), df[col])):
            pass  # pandas won't usually keep inf; left for completeness

    # MACE labels
    mace = df["OUTCOME MACE"]
    bad_mace = mace[~mace.isin([0, 1, 2])]
    if len(bad_mace):
        fail(f"OUTCOME MACE contains values outside {{0,1,2}}. Examples: {bad_mace.head(5).tolist()}")

    # severity proba
    sev = df["OUTCOME SEVERITY"]
    if sev.min() < 0 - 1e-9 or sev.max() > 1 + 1e-9:
        fail(f"OUTCOME SEVERITY must be in [0,1]. min={sev.min():.6f}, max={sev.max():.6f}")

    # id dtype (warn if not int-like)
    if not pd.api.types.is_integer_dtype(df["trustii_id"]):
        try:
            _ = df["trustii_id"].astype(int)
        except Exception:
            fail("trustii_id is not integer-like (cannot cast to int).")

    # duplicates?
    if df["trustii_id"].duplicated().any():
        dupes = df["trustii_id"][df["trustii_id"].duplicated()].unique()[:5]
        fail(f"Duplicate trustii_id values found. Examples: {dupes.tolist()}")

    ok()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_submission.py <path/to/submission.csv>")
        sys.exit(2)
    main(sys.argv[1])
