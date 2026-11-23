#!/usr/bin/env bash
# =============================================================================
# Explainability + error analysis (RQ3). Area lead: Sarthak. Runtime: ~15-20 min on a T4.
# Needs the seed-13 checkpoints saved by run_transformers.sh (models/<tag>__seed13).
#
# Why BERT vs MentalBERT only: they are the pair with the clearest RQ1 gain, so they are the interesting pair to
# explain (does domain pretraining change WHICH words the model uses?). Report answer: both rely on first-person
# pronouns, negation and emotion words rather than topic words; MentalBERT leans less on "I" and more on emotion
# vocabulary. Why seed 13 and 100 posts: LIME needs hundreds of forward passes per post, so we keep it small;
# the report therefore calls the difference one of degree, not of kind.
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
