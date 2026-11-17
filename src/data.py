"""Dreaddit loading and the ONE fixed train/val/test split everybody uses.

Usage:
    python -m src.data              # download (if needed) + write data/splits/*.csv
    python -m src.data --stats      # also print split statistics

Every model must read data via `load_splits()` so all results are comparable.
"""
import argparse
import io
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils import DATA_RAW, DATA_SPLITS

DREADDIT_URL = "http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip"
SPLIT_SEED = 42  # never change this: it defines the shared validation split
VAL_FRACTION = 0.10
KEEP_COLS = ["id", "post_id", "subreddit", "text", "label", "confidence"]
VAL_IDS_FILE = DATA_SPLITS / "val_post_ids.txt"


def download_dreaddit(raw_dir=DATA_RAW) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (raw_dir / "dreaddit-train.csv").exists() and (raw_dir / "dreaddit-test.csv").exists():
        return
    print(f"Downloading Dreaddit from {DREADDIT_URL} ...")
    with urllib.request.urlopen(DREADDIT_URL) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                (raw_dir / name.split("/")[-1]).write_bytes(zf.read(name))
    print(f"Saved CSVs to {raw_dir}")


def _read(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[KEEP_COLS].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df


def make_splits(raw_dir=DATA_RAW, out_dir=DATA_SPLITS) -> None:
    download_dreaddit(raw_dir)
    train_full = _read(raw_dir / "dreaddit-train.csv")
    test = _read(raw_dir / "dreaddit-test.csv")
    # Several Dreaddit segments come from the same Reddit post. The official test set shares
    # no posts with train, so we also keep whole posts on one side of the train/val split.
    # The chosen val posts are committed in VAL_IDS_FILE so every machine gets the identical split
    # (and we never commit Reddit text itself). Only regenerate it if the file is missing.
    if VAL_IDS_FILE.exists():
        val_posts = set(VAL_IDS_FILE.read_text().split())
    else:
        sgkf = StratifiedGroupKFold(n_splits=round(1 / VAL_FRACTION), shuffle=True, random_state=SPLIT_SEED)
        _, va_idx = next(sgkf.split(train_full, train_full["label"], groups=train_full["post_id"]))
        val_posts = set(train_full.iloc[va_idx]["post_id"])
        VAL_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        VAL_IDS_FILE.write_text("\n".join(sorted(val_posts)) + "\n")
    is_val = train_full["post_id"].isin(val_posts)
    train, val = train_full[~is_val], train_full[is_val]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.reset_index(drop=True).to_csv(out_dir / f"{name}.csv", index=False)
    print(f"Wrote splits to {out_dir}: train={len(train)} val={len(val)} test={len(test)}")


def load_splits(split_dir=DATA_SPLITS):
    """Return (train, val, test) DataFrames with columns id, post_id, subreddit, text, label, confidence."""
    if not (split_dir / "train.csv").exists():
        make_splits(out_dir=split_dir)
    return tuple(pd.read_csv(split_dir / f"{s}.csv") for s in ("train", "val", "test"))


def print_stats() -> None:
    train, val, test = load_splits()
    for name, df in [("train", train), ("val", val), ("test", test)]:
        n_words = df["text"].str.split().str.len()
        print(
            f"{name:5s} n={len(df):5d}  stress%={100 * df['label'].mean():5.1f}  "
            f"words mean={n_words.mean():.1f} max={n_words.max()}"
        )
    print("\nPosts per subreddit (train):")
    print(train.groupby("subreddit")["label"].agg(["count", "mean"]).round(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    make_splits()
    if args.stats:
        print_stats()
