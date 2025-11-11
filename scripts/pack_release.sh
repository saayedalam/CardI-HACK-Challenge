#!/usr/bin/env bash
set -euo pipefail

# run from project root
cd "$(dirname "$0")/.."

REL="release/cardihack_v1"
ZIP="${REL}.zip"

# fresh release dir
rm -rf "$REL" "$ZIP"
mkdir -p "$REL"

# build submission (so the ZIP includes a ready CSV if you want)
PYTHONPATH=. python make_submission.py
python scripts/validate_submission.py submissions/submission_best_pipeline.csv

# copy essentials
mkdir -p "$REL/docs" "$REL/src" "$REL/scripts" "$REL/notebooks" "$REL/submissions"
cp -v README.md config.yaml Dockerfile .dockerignore "$REL/"
cp -v docs/requirements.txt docs/ENVIRONMENT.md docs/MODEL_CARD.md docs/METHODS.md "$REL/docs/" || true
cp -v docs/execution_profile.md docs/cv_report.md "$REL/docs/" || true
cp -v docs/figures/"calibration_severity.png" "$REL/docs/" 2>/dev/null || true
cp -v docs/figures/"confusion_mace_oof.png" "$REL/docs/" 2>/dev/null || true
cp -v docs/figures/"importance_"*.png "$REL/docs/" 2>/dev/null || true
cp -rv src/* "$REL/src/"
cp -v make_submission.py "$REL/"
cp -v scripts/profile_inference.py scripts/validate_submission.py "$REL/scripts/"
# optional: include notebook as reference
cp -v notebooks/01_eda_baseline.ipynb "$REL/notebooks/" 2>/dev/null || true
# include the built submission (optional)
cp -v submissions/submission_best_pipeline.csv "$REL/submissions/"

# zip
(cd release && zip -r "$(basename "$ZIP")" "$(basename "$REL")" >/dev/null)
echo "Packed -> $ZIP"
