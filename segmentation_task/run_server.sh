#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/export/data/dataset/FoodSeg103}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/segmentation_task/.mplconfig}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$MPLCONFIGDIR"

python3 segmentation_task/src/train_unet_foodseg.py \
  --data-root "$DATA_ROOT" \
  --epochs 8 \
  --batch-size 16 \
  --image-size 128 \
  --train-limit 800 \
  --val-limit 200
