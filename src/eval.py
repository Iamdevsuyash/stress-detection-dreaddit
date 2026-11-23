"""Shared evaluation: metrics, saving every run, the summary table and significance tests.

Every model script ends by calling save_run(), which writes the test-set predictions and metrics of
that run. The report's tables and significance tests are then produced ONLY from those saved files,
so every number in the paper is computed by the same code (no hand-typed results).

Files written per run (run name = "<model_tag>__seed<k>", e.g. "mentalbert__seed13"):
    results/preds/<run>.csv     one row per test segment: id, label, pred, prob_stress
    results/metrics/<run>.json  the metrics below plus the run's settings

Usage:
    python -m src.eval summarize                              # results/summary.csv (mean +- std over seeds)
    python -m src.eval compare --a bert-base --b mentalbert   # McNemar + paired bootstrap, seed by seed
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
    """Compute the metrics reported in the paper.

    accuracy          share of segments classified correctly
    f1_macro          mean of the F1 of both classes (our primary metric; robust to class imbalance)
    f1_stress         F1 of the stress class only (the metric used in the original Dreaddit paper)
    precision_stress  of the segments predicted "stress", the share that really are stress
    recall_stress     of the real stress segments, the share the model found
    roc_auc           threshold-free ranking quality of prob_stress (only if scores are given)

    Report (Experimental Setup, "Metrics"): the classes are almost balanced (51.6% stress in test), so accuracy
    is reported too, but macro-F1 is the primary metric because it also penalises a model that favours one
    class. That matters here: the zero-shot LLM has 64.3 accuracy but only 58.2 macro-F1 because it calls nearly
    everything stress. ROC-AUC is included so models can be compared independently of their decision threshold
    (the LLM thresholds differ from the encoders'). f1_stress is kept for comparison with Turcan and McKeown.
    """
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
    """Write results/preds/<run>.csv and results/metrics/<run>.json for one test-set run; return metrics.

    model_tag: short model name used in all tables (e.g. "bert-base", "lstm-glove").
    seed:      the random seed of the run (part of the file name).
    test_df:   the test split (needed for segment ids and gold labels).
    y_pred:    predicted labels (0/1). y_prob: score for the stress class (for ROC-AUC).
    extra:     run settings (hyper-parameters, chosen threshold, ...) stored in the JSON for reference.
    """
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
    """Aggregate all saved runs into results/summary.csv: mean +- std (in %) over seeds, per model.

    This is the table the report's main results table is copied from.
    """
    rows = [json.loads(p.read_text()) for p in sorted(METRIC_DIR.glob("*.json"))]
    if not rows:
        raise SystemExit("No runs found in results/metrics/")
    df = pd.DataFrame(rows)
    cols = [c for c in ["accuracy", "f1_macro", "f1_stress", "precision_stress", "recall_stress", "roc_auc"] if c in df]
    # one row per model, across its seeds. We report mean +- std over seeds (not a single run) because fine-tuning
    # on a small dataset varies noticeably with the seed. std is pandas' sample std (n-1), based on only 3 seeds.
    agg = df.groupby("model")[cols].agg(["mean", "std"])
    n = df.groupby("model").size().rename("n_seeds")

    table = pd.DataFrame(index=agg.index)
    for c in cols:
        # std is NaN when a model has a single run (e.g. deterministic 0-shot LLM): show it as 0.
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
    """Load all seeds' prediction files of one model, sorted by file name (= by seed)."""
    files = sorted(PRED_DIR.glob(f"{model_tag}__seed*.csv"))
    if not files:
        raise SystemExit(f"No prediction files for model '{model_tag}'")
    # Sorting by segment id guarantees row i is the same test segment in every file, which the paired tests
    # below rely on (compare() also asserts that the ids match).
    return [pd.read_csv(f).sort_values("id").reset_index(drop=True) for f in files]


def mcnemar(y, pa, pb) -> dict:
    """Exact (binomial) McNemar test on the discordant pairs.

    Only segments where exactly one model is right are informative. If A and B were equally good,
    "only A right" and "only B right" would be equally likely, so their counts follow a
    Binomial(n, 0.5). A small p-value means one model is right significantly more often.

    Report (Table "Significance of domain-adaptive pretraining"): e.g. BERT vs MentalBERT, seed 42: only BERT
    right 23, only MentalBERT right 41 -> p = 0.033 (significant); seed 13: 39 / 59 -> p = 0.054; seed 87:
    43 / 51 -> p = 0.47. RoBERTa vs MentalRoBERTa never gets below p = 0.26. Our reading: the MentalBERT gain
    is consistent in direction but only sometimes significant, with 715 test segments.
    """
    from scipy.stats import binomtest

    # WHY exact binomial instead of the chi-square approximation: it stays valid when the number of
    # disagreements is small, which is common when two models differ by only a point or two.
    a_right, b_right = pa == y, pb == y
    b01 = int(np.sum(a_right & ~b_right))  # A right, B wrong
    b10 = int(np.sum(~a_right & b_right))  # A wrong, B right
    p = binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) > 0 else 1.0
    return {"a_only_correct": b01, "b_only_correct": b10, "p_value": float(p)}


def paired_bootstrap(y, pa, pb, n_boot=10000, seed=0) -> dict:
    """Paired bootstrap of the macro-F1 difference (B minus A).

    Resample the test set with replacement 10,000 times, recompute both models' macro-F1 on the same
    resample, and record the difference. Returns the observed difference, its 95% confidence interval
    (2.5th and 97.5th percentiles) and the share of resamples in which B is not better than A.

    Note on "p_b_not_better": it is ONE-sided (is B better than A?), unlike McNemar's two-sided p. That is why
    the report says the BERT -> MentalBERT gain is significant for one seed pair under McNemar but for two
    under the bootstrap (0.028 and 0.0095 for seeds 13 and 42). Report intervals, e.g. seed 13: delta +2.6
    macro-F1 points, 95% CI [-0.1, 5.3].
    """
    # Fixed seed so that re-running gives the same confidence interval (the numbers in the report).
    # WHY a bootstrap as well as McNemar: it gives a confidence INTERVAL for the macro-F1 gap, not just a p-value.
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)  # same resampled segments for both models -> "paired"
        deltas[i] = f1_score(y[idx], pb[idx], average="macro") - f1_score(y[idx], pa[idx], average="macro")
    return {
        "delta_f1_macro": float(f1_score(y, pb, average="macro") - f1_score(y, pa, average="macro")),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_b_not_better": float(np.mean(deltas <= 0)),
    }


def compare(model_a: str, model_b: str) -> None:
    """Compare two models seed by seed (pair 0 = seed 13, pair 1 = seed 42, pair 2 = seed 87).

    Used for RQ1: bert-base vs mentalbert and roberta-base vs mentalroberta. The output is saved in
    results/significance.txt and summarised in the report's significance table.

    WHY seed-matched pairs (seed 13 with seed 13, ...) and not pooled predictions: it keeps each test a
    comparison of two runs that share the same training randomness, and shows whether a gain holds up
    across seeds rather than hiding the seed-to-seed variation. WHY test at all: differences of one or two
    points on 715 segments can be chance (Dror et al., 2018). Takeaway in the report: MentalBERT +2.1 macro-F1
    over BERT in every seed, MentalRoBERTa only +0.6 over RoBERTa and never significant.
    """
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
