#!/usr/bin/env bash
set -euo pipefail

COCO_ROOT="${1:-/export/data/dataset/COCO}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/caption_task/.mplconfig}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$MPLCONFIGDIR"

python3 caption_task/src/train_caption_cnn_lstm.py \
  --coco-root "$COCO_ROOT" \
  --epochs 5 \
  --batch-size 64 \
  --train-limit 5000 \
  --sample-limit 8 \
  --vocab-size 3000
