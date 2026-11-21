"""Zero-/few-shot stress classification with an instruction-tuned LLM (RQ2), without any training.

What this file does
    Prompts Qwen2.5-7B-Instruct (loaded in 4-bit so it fits on one 16 GB T4 GPU) with an instruction and
    one Dreaddit segment, and asks "Is the author stressed?". In the 4-shot setting, four labelled
    training segments (2 stress, 2 not) are shown first as earlier chat turns. Every test segment gets
    a stress score; the run is saved with src.eval.save_run(). With --tune_threshold a second run
    ("<tag>-tuned") uses a decision threshold chosen on the validation split.

How the score is computed
    Instead of parsing generated text, we read the next-token probabilities of
    "Yes" vs "No" after the prompt: one forward pass per post, no parsing failures,
    and we get a probability (so ROC-AUC and threshold tuning work).
    Known limitation (discussed in the report): P(Yes) rounds to exactly 1.0 for many posts, so scores tie
    and the tuned threshold can land at 1.0. Log-odds (log P(Yes) - log P(No)) would avoid the ties; we kept
    P(Yes) because the reported results were produced with it.

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
from src.utils import RESULTS, set_seed

# The task instruction shown to the LLM (quoted verbatim in the Methods section of the report).
INSTRUCTION = (
    "You will read a short post from Reddit. Decide whether the author of the post is expressing "
    "stress, i.e. they feel overwhelmed, anxious, under pressure, or unable to cope with a situation. "
    "Answer with a single word: Yes or No."
)


def user_turn(text: str) -> str:
    """The user message for one segment: instruction + the segment in triple quotes + the question."""
    return f'{INSTRUCTION}\n\nPost: """{text}"""\n\nIs the author stressed?'


def build_messages(text, shots):
    """Build the chat for one segment.

    Each few-shot example becomes a user turn (its segment) followed by an assistant turn with the
    correct answer ("Yes"/"No"), so the model sees solved examples before the real question.
    With shots=[] (0-shot) the chat is only the final user turn.
    """
    msgs = []
    for ex_text, ex_label in shots:
        msgs.append({"role": "user", "content": user_turn(ex_text)})
        msgs.append({"role": "assistant", "content": "Yes" if ex_label == 1 else "No"})
    msgs.append({"role": "user", "content": user_turn(text)})
    return msgs


def answer_token_ids(tok, word):
    """Token ids that can start the answer `word`: with/without a leading space, capitalised or not.

    E.g. "Yes", " Yes", "yes", " yes" may be different tokens; we add up the probability of all of them.
    """
    ids = set()
    for variant in (word, " " + word, word.lower(), " " + word.lower()):
        enc = tok.encode(variant, add_special_tokens=False)
        if enc:
            ids.add(enc[0])  # first token of the variant = the token the model would generate first
    return sorted(ids)


@torch.no_grad()
def score(model, tok, texts, shots, batch_size, max_len):
    """Return P(Yes | Yes or No) for every text: the LLM's stress score.

    For each segment we run ONE forward pass over the full chat prompt (ending where the assistant's
    answer would start) and look at the probability distribution over the next token.
    """
    yes_ids, no_ids = answer_token_ids(tok, "Yes"), answer_token_ids(tok, "No")
    # apply_chat_template formats the messages in the model's own chat format; add_generation_prompt
    # appends the start of the assistant turn, so the next token is the model's answer.
    prompts = [tok.apply_chat_template(build_messages(t, shots), tokenize=False, add_generation_prompt=True)
               for t in texts]
    lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    print(f"  prompt tokens: mean {np.mean(lens):.0f}, max {max(lens)}")
    if max(lens) > max_len:
        # Never truncate: cutting the end would drop the assistant turn we read the answer from.
        raise SystemExit(f"Longest prompt has {max(lens)} tokens > --max_len {max_len}; raise --max_len.")
    order = np.argsort(lens)  # batch similar lengths together -> less padding, faster
    # Only the last position's logits are needed; skip the full [batch, seq, vocab] tensor.
    keep_key = "logits_to_keep" if Version(transformers.__version__) >= Version("4.50") else "num_logits_to_keep"
    probs = np.empty(len(prompts), dtype=np.float64)
    for i in range(0, len(prompts), batch_size):
        idx = order[i : i + batch_size]
        # Left padding (set in main) keeps the last real token at the last position of every row.
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        logits = model(**enc, **{keep_key: 1}).logits[:, -1, :].float()  # next-token scores [batch, vocab]
        logp = torch.log_softmax(logits, -1)
        yes = torch.logsumexp(logp[:, yes_ids], -1)  # log P(any "Yes" token)
        no = torch.logsumexp(logp[:, no_ids], -1)    # log P(any "No" token)
        p = torch.sigmoid(yes - no).cpu().numpy()  # P(Yes | Yes or No)
        if not np.isfinite(p).all():
            raise SystemExit("NaN/inf probabilities (fp16 overflow?). Re-run with --dtype bf16 or fp32 compute.")
        probs[idx] = p  # write back in the original order
        print(f"  scored {min(i + batch_size, len(prompts))}/{len(prompts)}", end="\r")
    print()
    return probs


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
    ap.add_argument("--seed", type=int, default=42, help="picks the few-shot examples (0-shot is deterministic)")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                    help="compute dtype; T4 has no fast bf16, use it only if fp16 gives NaNs")
    args = ap.parse_args()

    set_seed(args.seed)
    train, val, test = load_splits()
    if args.limit:
        test, val = test.head(args.limit), val.head(args.limit)

    # ---- tokenizer and model ----
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"  # pad on the left so every prompt ends at the last position
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if not torch.cuda.is_available():
        dtype = torch.float32
    dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    kwargs = {"device_map": "auto", dtype_key: dtype}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise SystemExit("--load_4bit needs a CUDA GPU, but torch sees none. On Kaggle: Settings -> "
                             "Accelerator -> GPU T4, then restart the session (also check GPU quota).")
        for d in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(d)
            print(f"GPU {d}: {torch.cuda.get_device_name(d)}, {free / 2**30:.1f} / {total / 2**30:.1f} GiB free")
        # A 7B model in 4-bit (~5.5 GB) fits one T4. Pin it to GPU 0: device_map="auto" keeps a memory margin
        # and may plan CPU offload, which bitsandbytes rejects ("Some modules are dispatched on the CPU").
        kwargs["device_map"] = {"": 0}
        # NF4 = 4-bit "NormalFloat" quantisation (QLoRA paper); matrix products are computed in `dtype`.
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                                           bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()

    # ---- few-shot examples: balanced (half stress, half not), drawn from TRAIN with the run's seed ----
    shots = []
    if args.shots:
        per_class = args.shots // 2
        pos = train[train.label == 1].sample(n=per_class, random_state=args.seed)
        neg = train[train.label == 0].sample(n=args.shots - per_class, random_state=args.seed)
        ex = pd.concat([pos, neg]).sample(frac=1, random_state=args.seed)  # shuffle so labels don't alternate
        shots = list(zip(ex["text"], ex["label"]))

    # ---- run 1: default decision rule, stress if P(Yes) >= 0.5 ----
    short = args.model.split("/")[-1].lower()
    tag = args.tag or f"llm-{short}-{args.shots}shot"
    prob = score(model, tok, list(test["text"]), shots, args.batch_size, args.max_len)
    save_run(tag, args.seed, test, (prob >= 0.5).astype(int), prob, extra={"hf_model": args.model, "threshold": 0.5,
                                                                         "shots": args.shots})

    # ---- run 2 (optional): threshold chosen on the VALIDATION split, then applied to the same test scores ----
    if args.tune_threshold:
        vprob = score(model, tok, list(val["text"]), shots, args.batch_size, args.max_len)
        val_dir = RESULTS / "val_probs"
        val_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": val["id"].values, "label": val["label"].values, "prob_stress": vprob}).to_csv(
            val_dir / f"{tag}__seed{args.seed}.csv", index=False)
        # P(Yes) is often saturated near 1 (most posts > 0.95), so a fixed 0.05-0.95 grid cannot reach the
        # best cut. Search the observed val probabilities instead: every distinct decision boundary is tried.
        grid = np.unique(vprob)
        best = max(grid, key=lambda t: f1_score(val["label"], (vprob >= t).astype(int), average="macro"))
        save_run(f"{tag}-tuned", args.seed, test, (prob >= best).astype(int), prob,
                 extra={"hf_model": args.model, "threshold": float(best), "shots": args.shots})


if __name__ == "__main__":
    main()
