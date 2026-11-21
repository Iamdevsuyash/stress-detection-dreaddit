#!/usr/bin/env bash
# LLM zero-/few-shot. Owner: Suyash.  Needs a GPU + `pip install bitsandbytes`.
# A 7B model in 4-bit fits a single 16 GB T4. Rough time on a T4: 0-shot ~5 min, each 4-shot seed ~15 min.
# 0-shot is deterministic -> one run. 4-shot: the seed picks the in-context examples, so 3 seeds
# measure how much the result depends on which examples were shown.
# Smoke test first (then delete its files so they stay out of the summary):
#   python -m src.llm_zeroshot --load_4bit --shots 4 --limit 16 --tag llm-smoke && rm -f results/*/llm-smoke*
set -euo pipefail
LLM=${LLM:-Qwen/Qwen2.5-7B-Instruct}
SHOTS=${SHOTS:-4}
python -m src.llm_zeroshot --model "$LLM" --load_4bit --shots 0 --seed 42 --tune_threshold
for SEED in 13 42 87; do
  python -m src.llm_zeroshot --model "$LLM" --load_4bit --shots "$SHOTS" --seed "$SEED" --tune_threshold
done
python -m src.eval summarize
