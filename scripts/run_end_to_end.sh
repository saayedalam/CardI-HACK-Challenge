#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from project root
cd "$(dirname "$0")/.."

echo ">>> Building best submission..."
PYTHONPATH=. python make_submission.py

echo ">>> Profiling inference..."
PYTHONPATH=. python scripts/profile_inference.py

echo ">>> Done."
echo "Submission: submissions/submission_best_pipeline.csv"
echo "Profile:    docs/execution_profile.md"
