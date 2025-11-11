# Cross-Validation Report (5-fold, leak-free)

**Pipeline**: SEVERITY (clinical-only LGBM, uncalibrated), MACE (clinical + PRS_MACE_K50)

- QWK (MACE): **0.110 ± 0.083**
- Rescaled weighted log loss (SEVERITY): **0.009 ± 0.002**
- Combined (0.7·QWK + 0.3·rLL): **0.079 ± 0.058**

Notes:
- PRS ranks computed on train-fold only; per-dataset z-score on each split.
- SEVERITY is reported **uncalibrated** in CV to avoid nested CV randomness; final submission uses isotonic calibration with fixed CV seed.
