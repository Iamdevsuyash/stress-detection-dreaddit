"""Error analysis from saved predictions (no GPU needed).

Usage:
    python -m src.analysis --a bert-base --b mentalbert --seed 13
Outputs:
    results/analysis/per_subreddit.csv     macro-F1 per subreddit for every model (seed given)
    results/analysis/disagreements.csv     test posts where A and B disagree, with text
    results/analysis/errors_<model>.csv    false positives / negatives of model B, with text
    figures/confusion_<model>.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

from src.data import load_splits
from src.eval import PRED_DIR
from src.utils import FIGURES, RESULTS

OUT = RESULTS / "analysis"


def load(model, seed, test):
    p = pd.read_csv(PRED_DIR / f"{model}__seed{seed}.csv")
    return test.merge(p[["id", "pred", "prob_stress"]], on="id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="general model tag, e.g. bert-base")
    ap.add_argument("--b", required=True, help="domain model tag, e.g. mentalbert")
    ap.add_argument("--seed", default="13")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    _, _, test = load_splits()

    # per-subreddit macro-F1 for every model that has this seed
    rows = []
    for f in sorted(PRED_DIR.glob(f"*__seed{args.seed}.csv")):
        model = f.stem.split("__seed")[0]
        df = load(model, args.seed, test)
        for sub, g in df.groupby("subreddit"):
            rows.append({"model": model, "subreddit": sub, "n": len(g),
                         "f1_macro": f1_score(g["label"], g["pred"], average="macro")})
    per_sub = pd.DataFrame(rows).pivot(index="subreddit", columns="model", values="f1_macro").round(3)
    per_sub.to_csv(OUT / "per_subreddit.csv")
    print(per_sub.to_string())

    a, b = load(args.a, args.seed, test), load(args.b, args.seed, test)
    m = a.merge(b[["id", "pred", "prob_stress"]], on="id", suffixes=(f"_{args.a}", f"_{args.b}"))
    dis = m[m[f"pred_{args.a}"] != m[f"pred_{args.b}"]]
    dis.to_csv(OUT / "disagreements.csv", index=False)
    print(f"\n{len(dis)} disagreements; {args.b} right on {(dis[f'pred_{args.b}'] == dis['label']).sum()}")

    for model, df in [(args.a, a), (args.b, b)]:
        err = df[df["pred"] != df["label"]].assign(
            error=lambda d: d["label"].map({1: "false_negative", 0: "false_positive"}))
        err.sort_values("prob_stress").to_csv(OUT / f"errors_{model}.csv", index=False)
        ConfusionMatrixDisplay.from_predictions(df["label"], df["pred"], display_labels=["no stress", "stress"],
                                                cmap="Blues", colorbar=False)
        plt.title(model)
        plt.tight_layout()
        plt.savefig(FIGURES / f"confusion_{model}.png", dpi=200)
        plt.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
