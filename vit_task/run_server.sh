#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/export/space0/pan-p/data}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/vit_task/.mplconfig}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$MPLCONFIGDIR"

python3 vit_task/src/train_vit_cifar.py \
  --data-root "$DATA_ROOT" \
  --epochs 10 \
  --batch-size 128 \
  --train-limit 20000 \
  --val-limit 2000 \
  --patch-size 4 \
  --embed-dim 128 \
  --depth 4 \
  --num-heads 4
