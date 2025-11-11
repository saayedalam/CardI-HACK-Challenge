#!/usr/bin/env python3
"""
5-fold leak-free CV for the current best pipeline:
- SEVERITY: clinical-only LightGBM (uncalibrated here; we report rLL with base model to avoid nested CV)
- MACE: clinical + PRS_MACE_K50
Writes docs/cv_report.md with QWK/rLL/combined stats (mean ± std).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from lightgbm import LGBMClassifier

# --- project helpers ---
from src import (
    load_train_test, snp_cols_in_range, rank_snps_on_train, build_prs_per_dataset,
    severity_features, mace_features,
)

# challenge’s rescaled weighted log loss components (same weights we’ve used)
def weighted_log_loss(y_true, y_prob, pos_w=1.0, neg_w=1.5, eps=1e-12):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob), eps, 1 - eps)
    w = np.where(y_true == 1, pos_w, neg_w)
    return -np.mean(w * (y_true*np.log(y_prob) + (1-y_true)*np.log(1-y_prob)))

def rescaled_rll(y_true, y_prob, pos_w=1.0, neg_w=1.5):
    # dummy: predict base rate
    p = np.mean(y_true)
    dummy = weighted_log_loss(y_true, np.full_like(y_prob, p), pos_w, neg_w)
    worst = weighted_log_loss(y_true, 1 - y_true, pos_w, neg_w)
    wll = weighted_log_loss(y_true, y_prob, pos_w, neg_w)
    if wll <= dummy:
        return 1.0 - (wll / dummy)
    return (wll - dummy) / (worst - dummy)

def main():
    df_train, _ = load_train_test("data/raw", "train.csv", "test.csv")

    snps = snp_cols_in_range(df_train, 1, 75)
    feats_sev  = severity_features()
    feats_mace = mace_features("PRS_MACE_K50")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    qwks, rlls, scores = [], [], []

    for tr, va in skf.split(df_train[feats_sev], df_train["OUTCOME MACE"].values):
        dtr, dva = df_train.iloc[tr].copy(), df_train.iloc[va].copy()

        # PRS ranks from TRAIN fold only; per-dataset z-score on each split
        rank_mace = rank_snps_on_train(dtr, snps, target="OUTCOME MACE", method="spearman")
        dtr["PRS_MACE_K50"] = build_prs_per_dataset(dtr, rank_mace, 50)
        dva["PRS_MACE_K50"] = build_prs_per_dataset(dva, rank_mace, 50)

        # SEVERITY (uncalibrated in CV report to avoid nested CV)
        sev = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
        sev.fit(dtr[feats_sev], dtr["OUTCOME SEVERITY"])
        sev_proba = sev.predict_proba(dva[feats_sev])[:, 1]
        rll = rescaled_rll(dva["OUTCOME SEVERITY"].values, sev_proba)

        # MACE (multiclass, with class weights)
        mace = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, random_state=42,
            class_weight={0:1,1:3,2:4}
        )
        mace.fit(dtr[feats_mace], dtr["OUTCOME MACE"])
        mace_pred = mace.predict(dva[feats_mace])
        qwk = cohen_kappa_score(dva["OUTCOME MACE"].values, mace_pred, weights="quadratic")

        qwks.append(qwk); rlls.append(rll); scores.append(0.7*qwk + 0.3*rll)

    mean = lambda arr: float(np.mean(arr))
    std  = lambda arr: float(np.std(arr))

    report = {
        "qwk_mean": mean(qwks), "qwk_std": std(qwks),
        "rll_mean": mean(rlls), "rll_std": std(rlls),
        "score_mean": mean(scores), "score_std": std(scores),
    }

    md = f"""# Cross-Validation Report (5-fold, leak-free)

**Pipeline**: SEVERITY (clinical-only LGBM, uncalibrated), MACE (clinical + PRS_MACE_K50)

- QWK (MACE): **{report['qwk_mean']:.3f} ± {report['qwk_std']:.3f}**
- Rescaled weighted log loss (SEVERITY): **{report['rll_mean']:.3f} ± {report['rll_std']:.3f}**
- Combined (0.7·QWK + 0.3·rLL): **{report['score_mean']:.3f} ± {report['score_std']:.3f}**

Notes:
- PRS ranks computed on train-fold only; per-dataset z-score on each split.
- SEVERITY is reported **uncalibrated** in CV to avoid nested CV randomness; final submission uses isotonic calibration with fixed CV seed.
"""
    Path("docs").mkdir(exist_ok=True, parents=True)
    Path("docs/cv_report.md").write_text(md)
    print("Wrote docs/cv_report.md")

if __name__ == "__main__":
    main()
