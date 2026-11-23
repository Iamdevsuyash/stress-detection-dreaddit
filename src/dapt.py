"""(Stretch goal, NOT used in the report) Continued masked-LM pretraining on in-domain text.

What this file does
    Takes a general model (BERT-base by default) and continues its original pretraining task, masked
    language modelling (hide 15% of the tokens and train the model to predict them), on Reddit text
    from the Dreaddit training split. The adapted checkpoint can then be fine-tuned for classification
    with train_transformer.py, like MentalBERT. We did not run this in the end (no time), so the report
    uses MentalBERT/MentalRoBERTa as the domain-adapted models; the script is kept for future work.

Usage:
    python -m src.dapt --base bert-base-uncased --out models/bert-dapt --epochs 3
    python -m src.dapt --extra_text data/raw/extra_reddit.txt ...   # optional: one post per line
    python -m src.train_transformer --model models/bert-dapt --tag bert-dapt --seed 13

Only TRAIN-split text is used by default, so no test text leaks into pretraining.

Terminology (Gururangan et al., 2020): continued pretraining on a large in-domain corpus is DAPT
(MentalBERT is essentially DAPT on mental-health Reddit); continued pretraining on the task's own
unlabeled text, which is what this script does unless --extra_text is given, is TAPT.
"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup)

from src.data import load_splits
from src.utils import get_device, set_seed


def main():
    """Continue masked-LM pretraining of --base on the training texts and save it to --out."""
    # Defaults (none tuned, since this stretch goal was never run for the report):
    #   mlm_prob 0.15  the masking rate used by the original BERT pretraining.
    #   lr 5e-5        a larger rate than fine-tuning (2e-5), as the whole model is adapted on the MLM objective.
    #   max_len 256    same cut-off as train_transformer.py, so DAPT sees the same text lengths as fine-tuning.
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="bert-base-uncased")
    ap.add_argument("--out", default="models/bert-dapt")
    ap.add_argument("--extra_text", default=None, help="optional extra unlabeled text, one doc per line")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    # ---- unlabeled corpus: training-split texts (labels are not used), plus optional extra text ----
    set_seed(args.seed)
    device = get_device()
    train, _, _ = load_splits()
    texts = list(train["text"])
    if args.extra_text:
        with open(args.extra_text, encoding="utf8") as f:
            texts += [line.strip() for line in f if line.strip()]
    print(f"DAPT corpus: {len(texts)} documents")

    # ---- model with its masked-LM head; the collator masks 15% of tokens at random in every batch ----
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForMaskedLM.from_pretrained(args.base).to(device)
    enc = tok(texts, truncation=True, max_length=args.max_len)
    items = [{"input_ids": ids} for ids in enc["input_ids"]]
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=args.mlm_prob)
    loader = DataLoader(items, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # ---- AdamW, linear schedule with 6% warm-up, fp16 mixed precision on GPU ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss  # cross-entropy on the masked positions only
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        print(f"epoch {epoch} mlm_loss={total / len(loader):.4f}")

    # ---- save the adapted encoder; fine-tune it afterwards with train_transformer.py ----
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved DAPT checkpoint to {args.out}")


if __name__ == "__main__":
    main()
