# src/data_io.py
from typing import List, Tuple
import os
import pandas as pd

# ---- Clinical columns (12) + pathogenic flag ----
CLINICAL_FEATURES: List[str] = [
    "Age_Baseline",
    "Age_Diag",
    "BMI",
    "BSA",
    "Genre",
    "Epaiss_max",
    "Gradient",
    "TVNS",
    "FEVG",
    "ATCD_MS",
    "SYNCOPE",
    "Diam_OG",
]
PATHOGENIC_COL = "Variant.Pathogene"

def snp_cols_in_range(df: pd.DataFrame, start: int = 1, end: int = 75) -> List[str]:
    """Return SNP columns present in df within SNP{start}..SNP{end}."""
    cols = []
    for i in range(start, end + 1):
        name = f"SNP{i}"
        if name in df.columns:
            cols.append(name)
    return cols

def load_train_test(
    data_dir: str = "data/raw",
    train_name: str = "train.csv",
    test_name: str = "test.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSVs from data/raw (or a given directory)."""
    train_path = os.path.join(data_dir, train_name)
    test_path  = os.path.join(data_dir, test_name)
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    return df_train, df_test

def severity_features() -> List[str]:
    """Features used for OUTCOME SEVERITY (clinical + Variant.Pathogene)."""
    return CLINICAL_FEATURES + [PATHOGENIC_COL]

def mace_features(prs_col: str | None = None) -> List[str]:
    """Features used for OUTCOME MACE; optionally include a PRS column."""
    feats = CLINICAL_FEATURES + [PATHOGENIC_COL]
    if prs_col:
        feats.append(prs_col)
    return feats
