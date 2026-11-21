#!/usr/bin/env bash
# =============================================================================
# Explainability + error analysis (RQ3). Area lead: Sarthak. Runtime: ~15-20 min on a T4.
# Needs the seed-13 checkpoints saved by run_transformers.sh (models/<tag>__seed13).
#
# Step 1 (GPU): LIME on 100 test segments + attention heatmaps for 4 examples, BERT vs MentalBERT
#   -> results/explain/*_lime_global.csv, figures/lime_top_words.png,
#      results/explain/*.html and figures/attention_*.png (contain post text: git-ignored)
# Step 2 (CPU): per-subreddit F1, BERT/MentalBERT disagreements, error lists, confusion matrices
#   -> results/analysis/per_subreddit.csv, figures/confusion_*.png,
#      results/analysis/{disagreements,errors_*}.csv (contain post text: git-ignored)
# =============================================================================
set -euo pipefail   # stop at the first error
python -m src.explain --ckpts models/bert-base__seed13 models/mentalbert__seed13 --n 100 --num_samples 500
python -m src.analysis --a bert-base --b mentalbert --seed 13   # CPU only, reads results/preds
