# scripts/profile_inference.py
"""
Profiles inference time (not training) for the current best pipeline.
- SEVERITY: clinical-only LightGBM + isotonic calibration
- MACE: clinical + PRS_MACE_K50 (per-dataset z-score), LightGBM multiclass
Saves results to docs/execution_profile.md
"""

from time import perf_counter
from pathlib import Path
import numpy as np
import pandas as pd

from src import (
    load_train_test, snp_cols_in_range,
    rank_snps_on_train, build_prs_per_dataset,
    severity_features, mace_features,
    train_calibrated_severity, predict_severity_proba,
    train_mace_model, predict_mace_labels,
)

def time_once(fn, *args, **kwargs):
    t0 = perf_counter()
    out = fn(*args, **kwargs)
    dt = perf_counter() - t0
    return out, dt

def main():
    # 1) Load
    df_train, df_test = load_train_test("data/raw", "train.csv", "test.csv")

    # 2) Build PRS_MACE_K50 (train-only SNP ranking; per-dataset z-score)
    snps = snp_cols_in_range(df_train, 1, 75)
    mace_rank = rank_snps_on_train(df_train, snps, target="OUTCOME MACE", method="spearman")
    df_train["PRS_MACE_K50"] = build_prs_per_dataset(df_train, mace_rank, K=50)
    df_test["PRS_MACE_K50"]  = build_prs_per_dataset(df_test,  mace_rank, K=50)

    feats_sev  = severity_features()
    feats_mace = mace_features("PRS_MACE_K50")

    # 3) Train models (fast) — this is just to obtain inference objects
    sev_model = train_calibrated_severity(
        df_train, features=feats_sev,
        target_col="OUTCOME SEVERITY",
        n_estimators=300, learning_rate=0.05, seed=42
    )
    mace_model = train_mace_model(
        df_train, features=feats_mace,
        target_col="OUTCOME MACE",
        n_estimators=300, learning_rate=0.05,
        class_weight={0:1,1:3,2:4}, seed=42
    )

    # 4) Warm-up predictions (not counted)
    _ = sev_model.predict_proba(df_test[feats_sev])[:, 1]
    _ = mace_model.predict(df_test[feats_mace])

    # 5) Timed inference (repeat a few times to smooth)
    R = 5
    sev_times, mace_times = [], []
    n_test = len(df_test)

    for _ in range(R):
        _, dt = time_once(sev_model.predict_proba, df_test[feats_sev])
        sev_times.append(dt)

        _, dt = time_once(mace_model.predict, df_test[feats_mace])
        mace_times.append(dt)

    sev_ms = np.array(sev_times) * 1000.0
    mace_ms = np.array(mace_times) * 1000.0

    # 6) Aggregate
    result = {
        "test_rows": n_test,
        "severity_mean_ms": float(sev_ms.mean()),
        "severity_p95_ms": float(np.percentile(sev_ms, 95)),
        "severity_per_row_ms": float(sev_ms.mean() / max(n_test, 1)),
        "mace_mean_ms": float(mace_ms.mean()),
        "mace_p95_ms": float(np.percentile(mace_ms, 95)),
        "mace_per_row_ms": float(mace_ms.mean() / max(n_test, 1)),
        "repeats": R,
    }

    # 7) Save simple markdown
    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "execution_profile.md"
    md.write_text(
f"""# Execution Profile (inference only)

Test rows: **{result['test_rows']}**

## Severity (calibrated)
- Mean latency (batch): **{result['severity_mean_ms']:.2f} ms**
- P95 latency (batch): **{result['severity_p95_ms']:.2f} ms**
- Mean per-row: **{result['severity_per_row_ms']:.4f} ms/row**

## MACE (multiclass)
- Mean latency (batch): **{result['mace_mean_ms']:.2f} ms**
- P95 latency (batch): **{result['mace_p95_ms']:.2f} ms**
- Mean per-row: **{result['mace_per_row_ms']:.4f} ms/row**

Notes:
- Measured on your current machine with Python, LightGBM, scikit-learn.
- Numbers are approximate; repeat counts = {result['repeats']}.
- Pipeline: clinical-only severity (isotonic-calibrated), clinical+PRS_MACE_K50 for MACE.
"""
    )
    print(f"Wrote {md}")

if __name__ == "__main__":
    main()

