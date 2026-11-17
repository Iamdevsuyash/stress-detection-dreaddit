#!/usr/bin/env bash
# Explainability + error analysis. Owner: Sarthak.
# Needs the seed-13 checkpoints saved by run_transformers.sh (models/<tag>__seed13).
set -euo pipefail
python -m src.explain --ckpts models/bert-base__seed13 models/mentalbert__seed13 --n 100 --num_samples 500
python -m src.analysis --a bert-base --b mentalbert --seed 13   # CPU only, reads results/preds
