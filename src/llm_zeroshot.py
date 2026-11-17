"""Zero-/few-shot stress classification with an instruction-tuned LLM.

Instead of parsing generated text, we read the next-token probabilities of
"Yes" vs "No" after the prompt: one forward pass per post, no parsing failures,
and we get a probability (so ROC-AUC and threshold tuning work).

Usage (Kaggle/Colab GPU):
    python -m src.llm_zeroshot --model Qwen/Qwen2.5-7B-Instruct --load_4bit
    python -m src.llm_zeroshot --model Qwen/Qwen2.5-7B-Instruct --load_4bit --shots 4
    python -m src.llm_zeroshot --model meta-llama/Llama-3.1-8B-Instruct --load_4bit   # gated
    # add --tune_threshold to also report a run whose threshold is picked on the val split

4-bit loading needs `pip install bitsandbytes` (CUDA only).
"""
import argparse

import numpy as np
import pandas as pd
import torch
import transformers
from packaging.version import Version
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_splits
from src.eval import save_run
from src.utils import set_seed

INSTRUCTION = (
    "You will read a short post from Reddit. Decide whether the author of the post is expressing "
    "stress, i.e. they feel overwhelmed, anxious, under pressure, or unable to cope with a situation. "
    "Answer with a single word: Yes or No."
)


def user_turn(text: str) -> str:
    return f'{INSTRUCTION}\n\nPost: """{text}"""\n\nIs the author stressed?'


def build_messages(text, shots):
    msgs = []
    for ex_text, ex_label in shots:
        msgs.append({"role": "user", "content": user_turn(ex_text)})
        msgs.append({"role": "assistant", "content": "Yes" if ex_label == 1 else "No"})
    msgs.append({"role": "user", "content": user_turn(text)})
    return msgs


def answer_token_ids(tok, word):
    ids = set()
    for variant in (word, " " + word, word.lower(), " " + word.lower()):
        enc = tok.encode(variant, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


@torch.no_grad()
def score(model, tok, texts, shots, batch_size, max_len):
    yes_ids, no_ids = answer_token_ids(tok, "Yes"), answer_token_ids(tok, "No")
    prompts = [tok.apply_chat_template(build_messages(t, shots), tokenize=False, add_generation_prompt=True)
               for t in texts]
    probs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        logp = torch.log_softmax(logits, -1)
        yes = torch.logsumexp(logp[:, yes_ids], -1)
        no = torch.logsumexp(logp[:, no_ids], -1)
        probs.append(torch.sigmoid(yes - no).cpu())  # P(Yes | Yes or No)
        print(f"  scored {min(i + batch_size, len(prompts))}/{len(prompts)}", end="\r")
    print()
    return torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--shots", type=int, default=0, help="number of in-context examples (balanced, from train)")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="debug: score only N test posts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    train, val, test = load_splits()
    if args.limit:
        test, val = test.head(args.limit), val.head(args.limit)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    kwargs = {"device_map": "auto", dtype_key: dtype}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()

    shots = []
    if args.shots:
        per_class = args.shots // 2
        pos = train[train.label == 1].sample(n=per_class, random_state=args.seed)
        neg = train[train.label == 0].sample(n=args.shots - per_class, random_state=args.seed)
        ex = pd.concat([pos, neg]).sample(frac=1, random_state=args.seed)
        shots = list(zip(ex["text"], ex["label"]))

    short = args.model.split("/")[-1].lower()
    tag = args.tag or f"llm-{short}-{args.shots}shot"
    prob = score(model, tok, list(test["text"]), shots, args.batch_size, args.max_len)
    save_run(tag, args.seed, test, (prob >= 0.5).astype(int), prob, extra={"hf_model": args.model, "threshold": 0.5,
                                                                         "shots": args.shots})

    if args.tune_threshold:
        vprob = score(model, tok, list(val["text"]), shots, args.batch_size, args.max_len)
        grid = np.linspace(0.05, 0.95, 91)
        best = max(grid, key=lambda t: f1_score(val["label"], (vprob >= t).astype(int), average="macro"))
        save_run(f"{tag}-tuned", args.seed, test, (prob >= best).astype(int), prob,
                 extra={"hf_model": args.model, "threshold": float(best), "shots": args.shots})


if __name__ == "__main__":
    main()
