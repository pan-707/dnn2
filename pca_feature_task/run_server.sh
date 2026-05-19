#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/export/space0/pan-p/data}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/pca_feature_task/.mplconfig}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$MPLCONFIGDIR"

python3 pca_feature_task/src/pca_dcnn_features.py \
  --data-root "$DATA_ROOT" \
  --feature-limit 500 \
  --cluster-limit 100 \
  --batch-size 64
