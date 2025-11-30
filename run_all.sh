#!/usr/bin/env bash
# run_all.sh — Train + Calibrate RF, threshold on valid (F1-opt), evaluate on test, and run inference.
# Usage: bash run_all.sh
set -euo pipefail

IN_CSV="data/dataset.csv"
MODEL="models/rf_calibrated.joblib"
META="models/model_meta.json"

echo "[run_all] Using dataset: $IN_CSV"
[[ -f "$IN_CSV" ]] || { echo "[ERROR] Missing $IN_CSV"; exit 1; }

echo "[run_all] Training + calibration + evaluation..."
python scripts/train_rf.py --data "$IN_CSV"

THR=$(python - <<'PY'
import json
with open("models/model_meta.json") as f:
    m=json.load(f)
print(m["decision"]["threshold"])
PY
)

echo "[run_all] Threshold (valid F1-opt) = $THR"
echo "[run_all] Inference on test split (for completeness)"
python scripts/infer_rf.py --in_csv "$IN_CSV" --out_csv "results/pred_all_rf.csv" --model "$MODEL" --threshold "$THR"

echo "[run_all] Done. See results/ and models/"
