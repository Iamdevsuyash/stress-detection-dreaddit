#!/usr/bin/env bash
# LSTM baselines, 3 seeds. Owner: Suyash.  (~1-2 min per run on GPU)
# GloVe: put glove.6B.100d.txt in data/glove/ or pass GLOVE=/path/to/file
set -euo pipefail
GLOVE=${GLOVE:-data/glove/glove.6B.100d.txt}
for SEED in 13 42 87; do
  python -m src.train_lstm --glove "$GLOVE" --seed "$SEED"          # tag: lstm-glove
  python -m src.train_lstm --no_glove --seed "$SEED"                 # tag: lstm-random (ablation)
done
python -m src.eval summarize
