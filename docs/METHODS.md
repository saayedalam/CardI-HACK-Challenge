# Methods Summary — CardI-HACK (HCM Prognosis)

## Goal
Predict (1) **OUTCOME SEVERITY** (binary, baseline severity) as calibrated probability and (2) **OUTCOME MACE** (ordinal 0/1/2) as class label for synthetic HCM patients.

## Data & Features
- Train: 347 rows; Test: 149 rows (synthetic).
- Inputs: 12 clinical features + `Variant.Pathogene` + SNPs (`SNP1..SNP288`; use subset present).
- Targets in train: `OUTCOME SEVERITY` (0/1), `OUTCOME MACE` (0/1/2).

### Feature sets
- **Severity model features:** 12 clinical + `Variant.Pathogene`.
- **MACE model features:** same clinical set + `Variant.Pathogene` + **PRS_MACE_K50**.

## Polygenic Risk Score (PRS)
- **Ranking:** For MACE only, compute **Spearman correlation** of each SNP with `OUTCOME MACE` on **train only** within `SNP1..SNP75` (per challenge guidance). Sort by absolute correlation.
- **Construction (K=50):** For a dataset (train or test), z-score the top-K SNPs **within that dataset** (per-dataset mean/std), then compute a signed, correlation-weighted sum → `PRS_MACE_K50`.
- Rationale: a compact PRS reduces overfitting vs hundreds of raw SNPs and delivered consistent lift for MACE.

## Models
- **Severity:** LightGBM classifier (300 trees, lr 0.05, seed 42) → **isotonic calibration** using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` to output well-calibrated probabilities.
- **MACE:** LightGBM multiclass (300 trees, lr 0.05, `class_weight={0:1,1:3,2:4}`, seed 42) on clinical + `PRS_MACE_K50`.
- (Optional) **Bagging:** 5 seeds with majority vote; neutral vs. single model in our tests.

## Validation & Leakage Control
- **Leak-free CV (5-fold stratified on MACE):**
  - In each fold, rank SNPs and build PRS **using only the training fold**.
  - Train models on fold-train; evaluate on fold-valid.
- **Calibration** evaluated via out-of-fold curves (see `docs/figures/calibration_severity.png`).

## Metrics & Leaderboard Mapping
- Local score approximates challenge metric:
  - QWK on MACE (quadratic).
  - Rescaled weighted log loss on SEVERITY (weights {0:1.5, 1:1}) and composite `0.7*QWK + 0.3*rLL`.
- **Observed:** clinical-only baseline ≈ 0.021 (CV mean); mixed with PRS_MACE_K50 ≈ 0.082 (CV mean).
- **Public leaderboard best:** ~0.1102 with current pipeline.

## Handling Class Imbalance & Ordinality
- For MACE, class weights emphasize higher severities (`{0:1, 1:3, 2:4}`) aligning with QWK’s penalty structure.
- We tested an ordinal decomposition (y≥1 / y≥2), but it underperformed multiclass in CV.

## Reproducibility
- Versions locked in `docs/requirements.txt` and `docs/ENVIRONMENT.md`.
- Seeds and choices in `config.yaml`.
- End-to-end script: `make_submission.py` (produces `submissions/submission_best_pipeline.csv`).

## Interpretability
- Clinical drivers (from importance/analysis): age at baseline/diagnosis, left atrial diameter, LV wall thickness, BMI/BSA, EF.
- Genetic signal enters as **PRS_MACE_K50**; individual SNP effects are aggregated for stability.

## Limitations
- Small N and synthetic cohort; generalization to real-world HCM unknown.
- PRS is simple (correlation-weighted); no LD modeling or external GWAS priors.
- Ordinal thresholds for MACE are learned from data, not clinical cutoffs.

## Ethical note
- Research-only; not for clinical decision-making. Fairness and subgroup performance have not been established.
