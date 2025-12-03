#!/usr/bin/env bash
set -euo pipefail

# Root of WASB-SBDT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

OUT_BASE="../../outputs/hrnet_vs_hrcnet"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"

# Common settings
DATASET="tennis"
LOSS="hm_bce"
OPTIMIZER="adam_multistep"
DETECTOR="tracknetv2"
TRANSFORM="default"
TRACKER="online"

########################################
# 1) Train + eval HRNet (model=wasb)
########################################

HRNET_OUT="$OUT_BASE/hrnet"

echo "===== Training HRNet (model=wasb -> name: hrnet) ====="
python main.py \
  --config-name=train \
  dataset="$DATASET" \
  model=wasb \
  loss="$LOSS" \
  optimizer="$OPTIMIZER" \
  detector="$DETECTOR" \
  transform="$TRANSFORM" \
  tracker="$TRACKER" \
  output_dir="$HRNET_OUT"

echo "===== Evaluating HRNet (model=wasb) ====="
python main.py \
  --config-name=eval \
  dataset="$DATASET" \
  model=wasb \
  detector.model_path="$HRNET_OUT/best_model.pth.tar" \
  | tee "$LOG_DIR/eval_hrnet.log"

########################################
# 2) Train + eval HRCNet (model=hrcnet)
########################################

HRCNET_OUT="$OUT_BASE/hrcnet"

echo "===== Training HRCNet (model=hrcnet -> name: hrcnet) ====="
python main.py \
  --config-name=train \
  dataset="$DATASET" \
  model=hrcnet \
  loss="$LOSS" \
  optimizer="$OPTIMIZER" \
  detector="$DETECTOR" \
  transform="$TRANSFORM" \
  tracker="$TRACKER" \
  output_dir="$HRCNET_OUT"

echo "===== Evaluating HRCNet (model=hrcnet) ====="
python main.py \
  --config-name=eval \
  dataset="$DATASET" \
  model=hrcnet \
  detector.model_path="$HRCNET_OUT/best_model.pth.tar" \
  | tee "$LOG_DIR/eval_hrcnet.log"

########################################
# 3) Summarize metrics
########################################

echo
echo "================ Metric Summary ================"
echo "HRNet (model=wasb / backbone=HRNet):"
grep "^|" "$LOG_DIR/eval_hrnet.log" | tail -n 1 || echo "  (metrics not found)"

echo
echo "HRCNet (model=hrcnet / backbone=HRCNet):"
grep "^|" "$LOG_DIR/eval_hrcnet.log" | tail -n 1 || echo "  (metrics not found)"

echo "================================================"
