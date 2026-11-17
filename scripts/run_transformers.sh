#!/usr/bin/env bash
# General vs domain-adapted encoders, 3 seeds each. Owner: Saurabh.  (~3-5 min per run on a T4)
# MentalBERT/MentalRoBERTa are gated: accept terms on HuggingFace and set HF_TOKEN first.
# Usage: bash scripts/run_transformers.sh                 # all models
#        MODELS="bert-base mentalbert" bash scripts/run_transformers.sh
set -euo pipefail
MODELS=${MODELS:-"bert-base mentalbert roberta-base mentalroberta"}
for MODEL in $MODELS; do
  for SEED in 13 42 87; do
    if [ "$SEED" = "13" ]; then SAVE="--save"; else SAVE=""; fi   # keep seed-13 weights for explain.py
    python -m src.train_transformer --model "$MODEL" --seed "$SEED" $SAVE
  done
done

# Stretch goal: domain-adaptive pretraining of BERT on Dreaddit train text, then fine-tune
if [ "${DAPT:-0}" = "1" ]; then
  python -m src.dapt --base bert-base-uncased --out models/bert-dapt --epochs 3
  for SEED in 13 42 87; do
    python -m src.train_transformer --model models/bert-dapt --tag bert-dapt --seed "$SEED"
  done
fi

python -m src.eval summarize
python -m src.eval compare --a bert-base --b mentalbert
python -m src.eval compare --a roberta-base --b mentalroberta
