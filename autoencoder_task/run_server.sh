#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/export/data/dataset}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/autoencoder_task/.mplconfig}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$MPLCONFIGDIR"

python3 autoencoder_task/src/train_autoencoder.py \
  --data-root "$DATA_ROOT" \
  --epochs 10 \
  --batch-size 128 \
  --latent-dim 64 \
  --train-limit 20000 \
  --eval-limit 1000
