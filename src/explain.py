"""Explainability (RQ3): which words make a fine-tuned model predict "stress"?

What this file does
    For each saved checkpoint (we use the seed-13 BERT and MentalBERT models):
    1. LIME: explains the model's prediction on 100 test segments. For each segment LIME creates 500
       perturbed copies with random words removed, asks the model for P(stress) on each copy, and fits
       a small weighted linear model; its 10 largest weights say which words pushed the prediction
       towards (positive weight) or away from (negative weight) "stress". The weights are then summed
       per word over all 100 segments to get a global list of stress cues.
    2. Attention: for 4 example segments, plots how much the [CLS] token attends to each token in the
       last layer (averaged over the 12 heads). This is only an illustration: attention weights are
       not reliable explanations (Jain and Wallace, 2019).

Needs checkpoints saved with `train_transformer.py --save`.

Usage:
    python -m src.explain --ckpts models/bert-base__seed13 models/mentalbert__seed13 --n 100
Outputs:
    results/explain/<ckpt>_lime_global.csv      word-level cue weights aggregated over N test posts
    results/explain/<ckpt>_lime_<id>.html       LIME view for a few example posts
    figures/lime_top_words.png                  side-by-side top stress cues per model
    figures/attention_<ckpt>_<id>.png           last-layer [CLS] attention for example posts
The HTML files and attention plots show post text verbatim, so they are git-ignored (privacy).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw figures to files, no screen needed (Kaggle/servers)
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
    """Wraps a saved fine-tuned checkpoint so LIME can call it like a black-box classifier."""

    def __init__(self, path, device, max_len=256):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, attn_implementation="eager"  # SDPA kernels do not return attention weights
        ).to(device).eval()
        self.device, self.max_len = device, max_len

    @torch.no_grad()
    def predict_proba(self, texts, bs=64):
        """Return class probabilities [n, 2] (columns: not stress, stress) for a list of texts.

        This is the function LIME calls on the perturbed copies of a segment.
        """
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(list(texts[i : i + bs]), truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def cls_attention(self, text):
        """Return (tokens, weights): last-layer attention from [CLS] to each token, averaged over heads."""
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        att = self.model(**enc, output_attentions=True).attentions[-1][0]  # (heads, seq, seq)
        weights = att.mean(0)[0].cpu().numpy()  # attention from [CLS]/<s> to every token
        tokens = self.tok.convert_ids_to_tokens(enc["input_ids"][0])
        return tokens, weights


def lime_global(clf, texts, ids, num_features, num_samples, name, html_ids):
    """Run LIME on every text and aggregate the word weights into one table.

    Returns a DataFrame with one row per word: mean and summed LIME weight towards "stress" and the
    number of segments in which the word was among the top features. Also saves it as CSV and writes
    LIME's interactive HTML view for the example segments in html_ids.
    """
    # random_state=0: LIME removes words at random, so we fix its seed to get the same explanations every run.
    explainer = LimeTextExplainer(class_names=["no_stress", "stress"], random_state=0)
    agg = defaultdict(list)  # word -> list of its weights across segments
    for k, (pid, text) in enumerate(zip(ids, texts)):
        exp = explainer.explain_instance(text, clf.predict_proba, num_features=num_features,
                                         # Explain only the "stress" class: with two classes the probabilities sum
                                         # to 1, so the other class's weights would just be the opposite.
                                         num_samples=num_samples, labels=(1,))
        for word, w in exp.as_list(label=1):
            agg[word.lower()].append(w)
        if pid in html_ids:
            exp.save_to_file(str(OUT / f"{name}_lime_{pid}.html"))
        print(f"  LIME {k + 1}/{len(texts)}", end="\r")
    print()
    rows = [{"word": w, "mean_weight": np.mean(v), "sum_weight": np.sum(v), "n_posts": len(v)} for w, v in agg.items()]
    df = pd.DataFrame(rows).sort_values("sum_weight", ascending=False)  # strongest stress cues first
    df.to_csv(OUT / f"{name}_lime_global.csv", index=False)
    return df


def plot_top_words(tables: dict, k=15):
    """Draw figures/lime_top_words.png: the k words with the largest summed weight, one panel per model."""
    fig, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 0.35 * k + 1.5), squeeze=False)
    for ax, (name, df) in zip(axes[0], tables.items()):
        frequent = df[df.n_posts >= 3]  # ignore words seen in only 1-2 posts
        top = (frequent if len(frequent) >= k else df).head(k).iloc[::-1]  # reverse so the largest is on top
        ax.barh(top["word"], top["sum_weight"], color="#c0504d")
        ax.set_title(f"{name}\ntop stress cues (LIME)", fontsize=10)
        ax.set_xlabel("summed LIME weight toward 'stress'")
    fig.tight_layout()
    fig.savefig(FIGURES / "lime_top_words.png", dpi=200)
    plt.close(fig)


def plot_attention(clf, text, path, max_tokens=60):
    """Save a one-row heatmap of [CLS] attention over the first max_tokens tokens of one segment."""
    tokens, w = clf.cls_attention(text)
    tokens, w = tokens[1:-1][:max_tokens], w[1:-1][:max_tokens]  # drop special tokens
    fig, ax = plt.subplots(figsize=(min(0.32 * len(tokens) + 1, 20), 1.8))
    ax.imshow(w[None, :], cmap="Reds", aspect="auto")  # darker red = more attention
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    """For each checkpoint: run LIME on a stratified test sample, draw attention plots, then compare cues."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100, help="test posts for global LIME (stratified sample)")
    ap.add_argument("--num_features", type=int, default=10)
    # 500 perturbations per post (LIME's own default is 5,000): every one is a full model forward pass, and we
    # explain 100 posts x 2 models, so this keeps the run short enough for a Kaggle session.
    ap.add_argument("--num_samples", type=int, default=500, help="LIME perturbations per post")
    ap.add_argument("--n_examples", type=int, default=4, help="posts to render as HTML + attention plots")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    device = get_device()
    _, _, test = load_splits()
    # Sample ~n test segments with the same stress / not-stress balance as the test set. The same sample
    # is used for every model, so their LIME tables are directly comparable.
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
        del clf  # free GPU memory before loading the next model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    plot_top_words(tables)
    print(f"wrote {OUT} and {FIGURES / 'lime_top_words.png'}")


if __name__ == "__main__":
    main()
