#!/usr/bin/env bash
# LLM zero-/few-shot. Owner: Suyash.  Needs a GPU + `pip install bitsandbytes`.
# A 7B model in 4-bit fits a single 16 GB T4; ~10-20 min for the test set.
set -euo pipefail
LLM=${LLM:-Qwen/Qwen2.5-7B-Instruct}
python -m src.llm_zeroshot --model "$LLM" --load_4bit --shots 0 --tune_threshold
python -m src.llm_zeroshot --model "$LLM" --load_4bit --shots 4 --tune_threshold
python -m src.eval summarize
