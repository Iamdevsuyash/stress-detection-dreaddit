"""(Stretch goal) Domain-adaptive pretraining: continue masked-LM training of a
general model on in-domain Reddit text, then fine-tune it with train_transformer.py.

Usage:
    python -m src.dapt --base bert-base-uncased --out models/bert-dapt --epochs 3
    python -m src.dapt --extra_text data/raw/extra_reddit.txt ...   # optional: one post per line
    python -m src.train_transformer --model models/bert-dapt --tag bert-dapt --seed 13

Only TRAIN-split text is used by default, so no test text leaks into pretraining.
"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup)

from src.data import load_splits
from src.utils import get_device, set_seed


def main():
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

    set_seed(args.seed)
    device = get_device()
    train, _, _ = load_splits()
    texts = list(train["text"])
    if args.extra_text:
        with open(args.extra_text, encoding="utf8") as f:
            texts += [line.strip() for line in f if line.strip()]
    print(f"DAPT corpus: {len(texts)} documents")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForMaskedLM.from_pretrained(args.base).to(device)
    enc = tok(texts, truncation=True, max_length=args.max_len)
    items = [{"input_ids": ids} for ids in enc["input_ids"]]
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=args.mlm_prob)
    loader = DataLoader(items, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

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
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        print(f"epoch {epoch} mlm_loss={total / len(loader):.4f}")

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved DAPT checkpoint to {args.out}")


if __name__ == "__main__":
    main()
