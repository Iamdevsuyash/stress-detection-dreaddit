"""Fine-tune any HuggingFace encoder (BERT, RoBERTa, MentalBERT, MentalRoBERTa, DAPT checkpoint).

Usage:
    python -m src.train_transformer --model bert-base --seed 13
    python -m src.train_transformer --model mentalbert --seed 13 --save     # keep weights for explain.py
    python -m src.train_transformer --model models/bert-dapt --tag bert-dapt --seed 13

MentalBERT / MentalRoBERTa are GATED on HuggingFace: accept the terms on the model
page, then `huggingface-cli login` (or set HF_TOKEN; on Kaggle add it as a secret).

Model selection uses the shared validation split (best epoch by val macro-F1).
"""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import load_splits
from src.eval import save_run
from src.utils import MODELS, get_device, set_seed

MODEL_ALIASES = {
    "bert-base": "bert-base-uncased",
    "roberta-base": "roberta-base",
    "mentalbert": "mental/mental-bert-base-uncased",
    "mentalroberta": "mental/mental-roberta-base",
}


def make_loader(tokenizer, texts, labels, max_len, batch_size, shuffle):
    enc = tokenizer(list(texts), truncation=True, max_length=max_len)
    items = [
        {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": int(labels[i])}
        for i in range(len(texts))
    ]

    def collate(batch):
        out = tokenizer.pad(
            [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
            return_tensors="pt",
        )
        out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out

    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
    p = torch.cat(probs).numpy()
    return (p >= 0.5).astype(int), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=f"alias {list(MODEL_ALIASES)} or HF id / local path")
    ap.add_argument("--tag", default=None, help="name used in results files (default: the alias)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--save", action="store_true", help="save best checkpoint to models/<tag>__seed<k>")
    ap.add_argument("--no_record", action="store_true",
                    help="do not write results/ files (e.g. when only re-creating a checkpoint for explain.py)")
    ap.add_argument("--limit", type=int, default=None, help="debug: use only N training examples")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    name = MODEL_ALIASES.get(args.model, args.model)
    tag = args.tag or args.model.replace("/", "_")
    print(f"model={name} tag={tag} device={device}")

    train, val, test = load_splits()
    if args.limit:
        train = train.sample(n=min(args.limit, len(train)), random_state=args.seed)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)

    tr = make_loader(tok, train["text"], train["label"].values, args.max_len, args.batch_size, True)
    va = make_loader(tok, val["text"], val["label"].values, args.max_len, 64, False)
    te = make_loader(tok, test["text"], test["label"].values, args.max_len, 64, False)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    steps = len(tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_state, best_epoch = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in tr:
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
        val_pred, _ = predict(model, va, device)
        val_f1 = f1_score(val["label"], val_pred, average="macro")
        print(f"epoch {epoch} loss={total / len(tr):.4f} val_f1_macro={val_f1:.4f} ({time.time() - t0:.0f}s)")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    pred, prob = predict(model, te, device)
    extra = {"hf_model": name, "best_epoch": best_epoch, "val_f1_macro": best_f1,
             "train_seconds": round(time.time() - t0, 1), **{k: v for k, v in vars(args).items() if k != "model"}}
    if args.no_record:
        print(f"[no_record] test macro-F1 = {f1_score(test['label'], pred, average='macro'):.4f} (not saved)")
    else:
        save_run(tag, args.seed, test, pred, prob, extra=extra)

    if args.save:
        out = MODELS / f"{tag}__seed{args.seed}"
        model.save_pretrained(out)
        tok.save_pretrained(out)
        print(f"saved checkpoint to {out}")


if __name__ == "__main__":
    main()
