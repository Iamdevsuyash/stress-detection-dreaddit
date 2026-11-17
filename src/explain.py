"""Explainability: LIME (global cue words + per-post HTML) and [CLS] attention heatmaps.

Needs checkpoints saved with `train_transformer.py --save`.

Usage:
    python -m src.explain --ckpts models/bert-base__seed13 models/mentalbert__seed13 --n 100
Outputs:
    results/explain/<ckpt>_lime_global.csv      word-level cue weights aggregated over N test posts
    results/explain/<ckpt>_lime_<id>.html       LIME view for a few example posts
    figures/lime_top_words.png                  side-by-side top stress cues per model
    figures/attention_<ckpt>_<id>.png           last-layer [CLS] attention for example posts
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import load_splits
from src.utils import FIGURES, RESULTS, get_device, set_seed

OUT = RESULTS / "explain"


class Classifier:
    def __init__(self, path, device, max_len=256):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, attn_implementation="eager"  # SDPA kernels do not return attention weights
        ).to(device).eval()
        self.device, self.max_len = device, max_len

    @torch.no_grad()
    def predict_proba(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(list(texts[i : i + bs]), truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def cls_attention(self, text):
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        att = self.model(**enc, output_attentions=True).attentions[-1][0]  # (heads, seq, seq)
        weights = att.mean(0)[0].cpu().numpy()  # attention from [CLS]/<s> to every token
        tokens = self.tok.convert_ids_to_tokens(enc["input_ids"][0])
        return tokens, weights


def lime_global(clf, texts, ids, num_features, num_samples, name, html_ids):
    explainer = LimeTextExplainer(class_names=["no_stress", "stress"], random_state=0)
    agg = defaultdict(list)
    for k, (pid, text) in enumerate(zip(ids, texts)):
        exp = explainer.explain_instance(text, clf.predict_proba, num_features=num_features,
                                         num_samples=num_samples, labels=(1,))
        for word, w in exp.as_list(label=1):
            agg[word.lower()].append(w)
        if pid in html_ids:
            exp.save_to_file(str(OUT / f"{name}_lime_{pid}.html"))
        print(f"  LIME {k + 1}/{len(texts)}", end="\r")
    print()
    rows = [{"word": w, "mean_weight": np.mean(v), "sum_weight": np.sum(v), "n_posts": len(v)} for w, v in agg.items()]
    df = pd.DataFrame(rows).sort_values("sum_weight", ascending=False)
    df.to_csv(OUT / f"{name}_lime_global.csv", index=False)
    return df


def plot_top_words(tables: dict, k=15):
    fig, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 0.35 * k + 1.5), squeeze=False)
    for ax, (name, df) in zip(axes[0], tables.items()):
        frequent = df[df.n_posts >= 3]  # ignore words seen in only 1-2 posts
        top = (frequent if len(frequent) >= k else df).head(k).iloc[::-1]
        ax.barh(top["word"], top["sum_weight"], color="#c0504d")
        ax.set_title(f"{name}\ntop stress cues (LIME)", fontsize=10)
        ax.set_xlabel("summed LIME weight toward 'stress'")
    fig.tight_layout()
    fig.savefig(FIGURES / "lime_top_words.png", dpi=200)
    plt.close(fig)


def plot_attention(clf, text, path, max_tokens=60):
    tokens, w = clf.cls_attention(text)
    tokens, w = tokens[1:-1][:max_tokens], w[1:-1][:max_tokens]  # drop special tokens
    fig, ax = plt.subplots(figsize=(min(0.32 * len(tokens) + 1, 20), 1.8))
    ax.imshow(w[None, :], cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100, help="test posts for global LIME (stratified sample)")
    ap.add_argument("--num_features", type=int, default=10)
    ap.add_argument("--num_samples", type=int, default=500, help="LIME perturbations per post")
    ap.add_argument("--n_examples", type=int, default=4, help="posts to render as HTML + attention plots")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    device = get_device()
    _, _, test = load_splits()
    frac = min(1.0, args.n / len(test))
    sample = test.groupby("label", group_keys=False).sample(frac=frac, random_state=args.seed)
    example_ids = set(sample["id"].head(args.n_examples))

    tables = {}
    for ckpt in args.ckpts:
        name = Path(ckpt).name
        print(f"== {name}")
        clf = Classifier(ckpt, device)
        tables[name] = lime_global(clf, list(sample["text"]), list(sample["id"]), args.num_features,
                                   args.num_samples, name, example_ids)
        print(tables[name].head(15).to_string(index=False))
        for pid, text in zip(sample["id"], sample["text"]):
            if pid in example_ids:
                plot_attention(clf, text, FIGURES / f"attention_{name}_{pid}.png")
        del clf
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    plot_top_words(tables)
    print(f"wrote {OUT} and {FIGURES / 'lime_top_words.png'}")


if __name__ == "__main__":
    main()
