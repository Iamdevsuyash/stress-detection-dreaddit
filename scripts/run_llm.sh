#!/usr/bin/env bash
# =============================================================================
# LLM zero-/few-shot classification (RQ2): Qwen2.5-7B-Instruct, no training.
# Area lead: Suyash. Needs a GPU + `pip install bitsandbytes`.
# A 7B model in 4-bit fits a single 16 GB T4. Rough time on a T4: 0-shot ~5 min, each 4-shot seed ~15 min.
#
# Runs: 0-shot once (deterministic, seed 42) and 4-shot with seeds 13/42/87 (the seed picks which four
# training examples are shown, so the 3 seeds measure how much the result depends on that choice).
# --tune_threshold: every run is saved twice, with the default threshold P(Yes) >= 0.5 and with a
# threshold chosen on the validation split (tag suffix "-tuned"); validation scores go to results/val_probs/.
#
# Smoke test first (then delete its files so they stay out of the summary):
#   python -m src.llm_zeroshot --load_4bit --shots 4 --limit 16 --tag llm-smoke && rm -f results/*/llm-smoke*
# =============================================================================
set -euo pipefail   # stop at the first error
LLM=${LLM:-Qwen/Qwen2.5-7B-Instruct}
SHOTS=${SHOTS:-4}
python -m src.llm_zeroshot --model "$LLM" --load_4bit --shots 0 --seed 42 --tune_threshold     # 0-shot
for SEED in 13 42 87; do
  python -m src.llm_zeroshot --model "$LLM" --load_4bit --shots "$SHOTS" --seed "$SEED" --tune_threshold   # 4-shot
done
python -m src.eval summarize                                   # rebuild results/summary.csv
