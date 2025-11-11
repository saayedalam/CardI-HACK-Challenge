# CardI-HACK - HCM Prognosis (Reproducible Baseline)

Public leaderboard (as of 2025-11-10): ~0.1102  
End-to-end ML pipeline with clinical features + PRS for MACE, calibrated probabilities for SEVERITY, and full reproducibility (Docker, scripts, docs).

## Highlights
- Models
  - SEVERITY: LightGBM (clinical + Variant.Pathogene) + isotonic calibration (fixed CV seed).
  - MACE: LightGBM (clinical + Variant.Pathogene + PRS_MACE_K50), class_weight {0:1,1:3,2:4}.
- PRS construction
  - Rank SNP1..SNP75 by Spearman correlation on train only.
  - PRS_MACE_K50 = signed, correlation-weighted sum of per-dataset z-scored SNPs (train and test each z-scored with their own stats).
- Reproducibility
  - docs/requirements.txt, docs/ENVIRONMENT.md, config.yaml
  - Dockerfile + .dockerignore
  - One-command runner: scripts/run_end_to_end.sh
- Interpretability
  - Severity calibration curve: docs/figures/calibration_severity.png
  - MACE OOF confusion: docs/figures/confusion_mace_oof.png
  - Feature importances: docs/figures/importance_{severity,mace}.png

## Quick start
~~~bash
# 1) Put data
#   data/raw/train.csv
#   data/raw/test.csv

# 2) Create env and install
python -m venv .venv && source .venv/bin/activate
pip install -r docs/requirements.txt

# 3) Build submission
PYTHONPATH=. python make_submission.py

# 4) (Optional) Profile + validate
PYTHONPATH=. python scripts/profile_inference.py
python scripts/validate_submission.py submissions/submission_best_pipeline.csv
~~~

## Docker
~~~bash
docker build -t cardi-hack:latest .
docker run --rm \
  -e PYTHONPATH=/app \
  -v "$PWD/data/raw:/app/data/raw" \
  -v "$PWD/submissions:/app/submissions" \
  cardi-hack:latest
~~~

## Docs
- Methods: docs/METHODS.md
- Model card: docs/MODEL_CARD.md
- CV report: docs/cv_report.md
- Execution profile: docs/execution_profile.md

Note: This repo uses synthetic data provided by the challenge and is for research/competition only.
