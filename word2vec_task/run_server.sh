#!/usr/bin/env bash
set -euo pipefail

export GENSIM_DATA_DIR="${GENSIM_DATA_DIR:-$PWD/word2vec_task/models}"
mkdir -p "$GENSIM_DATA_DIR"

python3 word2vec_task/src/word2vec_analogies.py \
  --model-name glove-wiki-gigaword-50 \
  --topn 5
