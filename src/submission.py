# src/submission.py
import pandas as pd
from pathlib import Path

ID_COL = "trustii_id"
MACE_COL = "OUTCOME MACE"
SEV_COL = "OUTCOME SEVERITY"

def build_and_save_submission(
    df_test: pd.DataFrame,
    mace_labels: pd.Series,
    sev_proba: pd.Series,
    out_path: str | Path,
) -> Path:
    """
    Create the submission dataframe with required columns and save to CSV.

    - df_test must contain 'trustii_id'
    - mace_labels: int {0,1,2}
    - sev_proba: float in [0,1]
    """
    if ID_COL not in df_test.columns:
        raise ValueError(f"'{ID_COL}' not found in test dataframe")

    sub = pd.DataFrame({
        ID_COL: df_test[ID_COL].values,
        MACE_COL: mace_labels.astype(int).values,
        SEV_COL: sev_proba.astype(float).values,
    })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    return out_path
