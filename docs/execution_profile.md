# Execution Profile (inference only)

Test rows: **149**

## Severity (calibrated)
- Mean latency (batch): **9.50 ms**
- P95 latency (batch): **10.48 ms**
- Mean per-row: **0.0637 ms/row**

## MACE (multiclass)
- Mean latency (batch): **3.10 ms**
- P95 latency (batch): **3.35 ms**
- Mean per-row: **0.0208 ms/row**

Notes:
- Measured on your current machine with Python, LightGBM, scikit-learn.
- Numbers are approximate; repeat counts = 5.
- Pipeline: clinical-only severity (isotonic-calibrated), clinical+PRS_MACE_K50 for MACE.
