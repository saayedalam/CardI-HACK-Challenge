# Model Card — CardI-HACK (HCM Prognosis)

## Intended use
- Predict baseline severity (**OUTCOME SEVERITY**, binary) and future major events (**OUTCOME MACE**, ordinal 0/1/2) for HCM patients in the **CardI-HACK** challenge.
- For research/competition only (synthetic cohort). **Not for clinical decisions.**

## Data
- Train: ~347 rows, Test: ~149 rows (synthetic).
- Features: 12 clinical + `Variant.Pathogene` + 288 SNPs.
- PRS uses only **SNP1–SNP75** per challenge guidance.

## Model summary
- **SEVERITY:** LightGBM (300 trees, lr 0.05, seed 42) + **isotonic calibration** (StratifiedKFold 5, shuffle, seed 42).  
  Features: clinical + `Variant.Pathogene`.
- **MACE:** LightGBM multiclass (300 trees, lr 0.05, class_weight {0:1,1:3,2:4}, seed 42).  
  Features: clinical + `Variant.Pathogene` + **PRS_MACE_K50**.

## PRS construction
- Rank SNPs by **Spearman correlation** with MACE on **train only**, within SNP1–SNP75.
- Build **PRS_MACE_K50** = weighted sum of **per-dataset z-scored** SNPs (train z-scored by train stats; test by test stats) with signed correlation weights.

## Evaluation (local CV, leak-free)
- 5-fold stratified by MACE.
- Baseline (clinical-only): combined score ≈ 0.021 (mean).
- Mixed with PRS_MACE_K50 (PRS for MACE only): combined score ≈ 0.082 (mean).
- Leaderboard best: **~0.1102** (as of latest submission).

## Interpretability
- Clinical drivers: age (baseline/diagnosis), LA diameter (Diam_OG), LV wall thickness (Epaiss_max), BMI/BSA, EF.
- Genetic signal enters via compact **PRS_MACE_K50** instead of raw SNPs (reduces overfitting).

## Limitations
- Small N; synthetic distribution may differ from real-world patients.
- Ordinal MACE modeled as multiclass; thresholds may not reflect clinical cutoffs.
- PRS is correlation-weighted (simple); no LD modeling or external GWAS.

## Ethical/fairness notes
- Do not deploy clinically. Unknown subgroup performance; fairness not established.
- Outputs are for ranking/experimentation only.

## Reproducibility
- Environment locked in `docs/requirements.txt` and `docs/ENVIRONMENT.md`.
- Seeds/config in `config.yaml`.
- End-to-end script: `make_submission.py`.
