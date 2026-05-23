#!/usr/bin/env bash
set -euo pipefail

IMAGE_ROOT="${1:-COCO}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/detection_task/.mplconfig}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$MPLCONFIGDIR"

python3 detection_task/src/run_detection.py \
  --image-root "$IMAGE_ROOT" \
  --num-images 6 \
  --score-threshold 0.45 \
  --max-detections 6
