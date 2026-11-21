#!/usr/bin/env bash
# =============================================================================
# LSTM baselines: BiLSTM + GloVe and the random-embedding ablation, 3 seeds each.
# Area lead: Suyash. Runtime: ~1-2 min per run on GPU (CPU also works).
#
# What it produces (6 runs):
#   results/preds/lstm-glove__seed{13,42,87}.csv   + results/metrics/*.json
#   results/preds/lstm-random__seed{13,42,87}.csv  + results/metrics/*.json
#   results/summary.csv (updated)
# Comparing lstm-glove with lstm-random shows what pretrained word vectors add (+4.2 macro-F1).
#
# GloVe: put glove.6B.100d.txt in data/glove/ or pass GLOVE=/path/to/file;
# if missing it is downloaded automatically by src/train_lstm.py (~860 MB, once).
# =============================================================================
set -euo pipefail   # stop at the first error, on unset variables, and on failures inside pipes
GLOVE=${GLOVE:-data/glove/glove.6B.100d.txt}
for SEED in 13 42 87; do                                             # the three project-wide seeds
  python -m src.train_lstm --glove "$GLOVE" --seed "$SEED"          # tag: lstm-glove
  python -m src.train_lstm --no_glove --seed "$SEED"                 # tag: lstm-random (ablation)
done
python -m src.eval summarize                                         # rebuild results/summary.csv
