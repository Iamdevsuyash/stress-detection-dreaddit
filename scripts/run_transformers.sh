#!/usr/bin/env bash
# =============================================================================
# General vs domain-adapted encoders (RQ1): 4 models x 3 seeds = 12 fine-tuning runs.
# Area lead: Saurabh. Runtime: ~2 min per run on a Kaggle T4 (about 25-30 min in total).
#
# Models: bert-base, mentalbert, roberta-base, mentalroberta (all with identical hyper-parameters).
# What it produces:
#   results/preds/<model>__seed<k>.csv + results/metrics/<model>__seed<k>.json for every run
#   models/<model>__seed13/   fine-tuned weights of the seed-13 runs (needed by run_explain.sh; git-ignored)
#   results/summary.csv        mean +- std over seeds
#   significance tests printed at the end (saved by hand to results/significance.txt)
#
# MentalBERT/MentalRoBERTa are gated: accept their terms on HuggingFace and set HF_TOKEN first.
# Usage: bash scripts/run_transformers.sh                 # all models
#        MODELS="bert-base mentalbert" bash scripts/run_transformers.sh
# =============================================================================
set -euo pipefail   # stop at the first error
MODELS=${MODELS:-"bert-base mentalbert roberta-base mentalroberta"}
for MODEL in $MODELS; do
  for SEED in 13 42 87; do
    if [ "$SEED" = "13" ]; then SAVE="--save"; else SAVE=""; fi   # keep seed-13 weights for explain.py
    python -m src.train_transformer --model "$MODEL" --seed "$SEED" $SAVE
  done
done

# Stretch goal (not run for the report): domain-adaptive pretraining of BERT on Dreaddit train text,
# then fine-tune it like the other encoders. Enable with DAPT=1.
if [ "${DAPT:-0}" = "1" ]; then
  python -m src.dapt --base bert-base-uncased --out models/bert-dapt --epochs 3
  for SEED in 13 42 87; do
    python -m src.train_transformer --model models/bert-dapt --tag bert-dapt --seed "$SEED"
  done
fi

python -m src.eval summarize                                   # rebuild results/summary.csv
# Paired significance tests per seed (McNemar + bootstrap): general model (--a) vs domain model (--b).
python -m src.eval compare --a bert-base --b mentalbert
python -m src.eval compare --a roberta-base --b mentalroberta
