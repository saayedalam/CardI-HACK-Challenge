# make_submission.py
# Reproduce best pipeline (LB ~0.1102):
# - SEVERITY: clinical-only LightGBM + isotonic calibration
# - MACE: clinical + PRS_MACE_K50 (per-dataset z-score), LightGBM multiclass
# - PRS SNP ranks computed on TRAIN only (SNP1..SNP75), Spearman

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src import (
    load_train_test, snp_cols_in_range,
    rank_snps_on_train, build_prs_per_dataset,
    severity_features, mace_features,
    train_calibrated_severity, predict_severity_proba,
    train_mace_model, predict_mace_labels,
)
from src.submission import build_and_save_submission

def main():
    # 1) Load data
    df_train, df_test = load_train_test("data/raw", "train.csv", "test.csv")

    # 2) PRS_MACE_K50
    snps = snp_cols_in_range(df_train, start=1, end=75)
    mace_rank = rank_snps_on_train(df_train, snps, target="OUTCOME MACE", method="spearman")

    # Per-dataset z-score PRS (train by train stats, test by test stats)
    df_train["PRS_MACE_K50"] = build_prs_per_dataset(df_train, mace_rank, K=50)
    df_test["PRS_MACE_K50"]  = build_prs_per_dataset(df_test,  mace_rank, K=50)

    # 3) Features
    feats_sev  = severity_features()                    # clinical + Variant.Pathogene
    feats_mace = mace_features(prs_col="PRS_MACE_K50")  # clinical + Variant.Pathogene + PRS

    # 4) Train SEVERITY with isotonic calibration (fixed CV seed)
    sev_model = train_calibrated_severity(
        df_train, features=feats_sev,
        target_col="OUTCOME SEVERITY",
        n_estimators=300, learning_rate=0.05, seed=42
    )
    sev_proba_test = predict_severity_proba(sev_model, df_test, feats_sev)

    # 5) Train MACE multiclass LGBM (class weights fixed)
    mace_model = train_mace_model(
        df_train, features=feats_mace, target_col="OUTCOME MACE",
        n_estimators=300, learning_rate=0.05,
        class_weight={0:1, 1:3, 2:4}, seed=42
    )
    mace_labels_test = predict_mace_labels(mace_model, df_test, feats_mace)

    # 6) Save submission
    out_path = build_and_save_submission(
        df_test, mace_labels_test, sev_proba_test,
        out_path="submissions/submission_best_pipeline.csv"
    )
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
