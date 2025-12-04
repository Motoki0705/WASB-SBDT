#!/usr/bin/env bash

# Run HRCNet training for multiple temporal_type settings and
# store each run under outputs/{model_name}.

set -euo pipefail

# Move to project root (the directory that contains main.py and configs/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
cd "${PROJECT_ROOT}"

TEMPORAL_TYPES=("none" "tsm" "conv1d" "convgru" "transformer")

for TEMP in "${TEMPORAL_TYPES[@]}"; do
  MODEL_NAME="hrcnet_${TEMP}"
  echo "===== Running training: temporal_type=${TEMP}, model_name=${MODEL_NAME} ====="

  python main.py \
    model.temporal_type="${TEMP}" \
    output_dir="outputs/${MODEL_NAME}"

  echo "===== Finished: ${MODEL_NAME} ====="
  echo
done
