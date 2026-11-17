"""Shared evaluation: every model saves predictions through `save_run`, and the
report tables + significance tests are produced from those files only.

Usage:
    python -m src.eval summarize                       # results/summary.csv + markdown table
    python -m src.eval compare --a bert-base --b mentalbert   # McNemar + paired bootstrap

Run naming convention: <model_tag>__seed<k>, e.g. "mentalbert__seed13".
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.utils import RESULTS

PRED_DIR = RESULTS / "preds"
METRIC_DIR = RESULTS / "metrics"


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_stress": f1_score(y_true, y_pred, pos_label=1),
        "precision_stress": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_stress": recall_score(y_true, y_pred, pos_label=1),
    }
    if y_prob is not None and not np.any(pd.isna(y_prob)):
        m["roc_auc"] = roc_auc_score(y_true, y_prob)
    return {k: float(v) for k, v in m.items()}


def save_run(model_tag: str, seed, test_df: pd.DataFrame, y_pred, y_prob=None, extra: dict | None = None) -> dict:
    """Write results/preds/<run>.csv and results/metrics/<run>.json; return metrics."""
    run = f"{model_tag}__seed{seed}"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame(
        {
            "id": test_df["id"].values,
            "label": test_df["label"].values,
            "pred": np.asarray(y_pred).astype(int),
            "prob_stress": np.asarray(y_prob, dtype=float) if y_prob is not None else np.nan,
        }
    )
    preds.to_csv(PRED_DIR / f"{run}.csv", index=False)
    metrics = compute_metrics(preds["label"], preds["pred"], None if y_prob is None else preds["prob_stress"])
    record = {"run": run, "model": model_tag, "seed": seed, **metrics, **(extra or {})}
    (METRIC_DIR / f"{run}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return record


def summarize() -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(METRIC_DIR.glob("*.json"))]
    if not rows:
        raise SystemExit("No runs found in results/metrics/")
    df = pd.DataFrame(rows)
    cols = [c for c in ["accuracy", "f1_macro", "f1_stress", "precision_stress", "recall_stress", "roc_auc"] if c in df]
    agg = df.groupby("model")[cols].agg(["mean", "std"])
    n = df.groupby("model").size().rename("n_seeds")

    table = pd.DataFrame(index=agg.index)
    for c in cols:
        mean, std = agg[(c, "mean")] * 100, agg[(c, "std")].fillna(0) * 100
        table[c] = [f"{m:.2f} ± {s:.2f}" if not np.isnan(m) else "–" for m, s in zip(mean, std)]
    table["n_seeds"] = n
    table = table.sort_values("f1_macro")
    table.to_csv(RESULTS / "summary.csv")
    try:
        print(table.to_markdown())  # needs `tabulate`
    except ImportError:
        print(table.to_string())
    return table


def _load_preds(model_tag: str) -> list[pd.DataFrame]:
    files = sorted(PRED_DIR.glob(f"{model_tag}__seed*.csv"))
    if not files:
        raise SystemExit(f"No prediction files for model '{model_tag}'")
    return [pd.read_csv(f).sort_values("id").reset_index(drop=True) for f in files]


def mcnemar(y, pa, pb) -> dict:
    """Exact (binomial) McNemar test on the discordant pairs."""
    from scipy.stats import binomtest

    a_right, b_right = pa == y, pb == y
    b01 = int(np.sum(a_right & ~b_right))  # A right, B wrong
    b10 = int(np.sum(~a_right & b_right))  # A wrong, B right
    p = binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) > 0 else 1.0
    return {"a_only_correct": b01, "b_only_correct": b10, "p_value": float(p)}


def paired_bootstrap(y, pa, pb, n_boot=10000, seed=0) -> dict:
    """Bootstrap the test set; report P(F1_macro(B) <= F1_macro(A)) and 95% CI of the gain."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = f1_score(y[idx], pb[idx], average="macro") - f1_score(y[idx], pa[idx], average="macro")
    return {
        "delta_f1_macro": float(f1_score(y, pb, average="macro") - f1_score(y, pa, average="macro")),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_b_not_better": float(np.mean(deltas <= 0)),
    }


def compare(model_a: str, model_b: str) -> None:
    """Compare two models seed by seed (seeds paired by sorted order)."""
    runs_a, runs_b = _load_preds(model_a), _load_preds(model_b)
    for i, (ra, rb) in enumerate(zip(runs_a, runs_b)):
        assert (ra["id"].values == rb["id"].values).all(), "test ids differ"
        y, pa, pb = ra["label"].values, ra["pred"].values, rb["pred"].values
        print(f"--- pair {i}: {model_a} vs {model_b}")
        print("McNemar:  ", mcnemar(y, pa, pb))
        print("Bootstrap:", paired_bootstrap(y, pa, pb))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summarize")
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True, help="baseline model tag, e.g. bert-base")
    c.add_argument("--b", required=True, help="challenger model tag, e.g. mentalbert")
    args = ap.parse_args()
    summarize() if args.cmd == "summarize" else compare(args.a, args.b)
