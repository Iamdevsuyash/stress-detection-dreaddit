"""Dreaddit loading and the ONE fixed train/val/test split that every model uses.

What this file does
    1. Downloads Dreaddit (Turcan and McKeown, 2019) from the authors' website into data/raw/.
    2. Holds out 10% of the official training set as a validation set, grouped by Reddit post
       (all segments of one post go to the same side), and writes data/splits/{train,val,test}.csv.
    3. Provides load_splits(), the only function any model should use to read data. Because every
       script reads the same split, all results in the report are directly comparable.

Dataset facts (see the Dataset section of the report)
    3,553 labelled segments (5 sentences each) from 10 subreddits; label 1 = stress, 0 = not stress;
    about 52% stress. Split sizes: train 2,555 / val 283 / test 715 (official test set, untouched).

Usage:
    python -m src.data              # download (if needed) + write data/splits/*.csv
    python -m src.data --stats      # also print split statistics
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
VAL_FRACTION = 0.10  # 10% of the official training set becomes the validation set
# Columns we keep from the raw CSVs: segment id, the Reddit post it came from, its community,
# the text, the binary label, and "confidence" = fraction of annotators who agreed with the label.
KEEP_COLS = ["id", "post_id", "subreddit", "text", "label", "confidence"]
# The validation post ids are committed to git so every machine builds the identical split
# without us ever committing Reddit text itself (privacy).
VAL_IDS_FILE = DATA_SPLITS / "val_post_ids.txt"


def download_dreaddit(raw_dir=DATA_RAW) -> None:
    """Download the Dreaddit zip and extract its train/test CSVs into raw_dir (skipped if present)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (raw_dir / "dreaddit-train.csv").exists() and (raw_dir / "dreaddit-test.csv").exists():
        return  # already downloaded
    print(f"Downloading Dreaddit from {DREADDIT_URL} ...")
    with urllib.request.urlopen(DREADDIT_URL) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                (raw_dir / name.split("/")[-1]).write_bytes(zf.read(name))
    print(f"Saved CSVs to {raw_dir}")


def _read(path) -> pd.DataFrame:
    """Read one raw Dreaddit CSV, keep only the columns we use and normalise types."""
    df = pd.read_csv(path)
    df = df[KEEP_COLS].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df


def make_splits(raw_dir=DATA_RAW, out_dir=DATA_SPLITS) -> None:
    """Build data/splits/{train,val,test}.csv from the raw files.

    The official test set is used as-is. The validation set is carved out of the official training
    set with a post-grouped, label-stratified split, so validation scores are not inflated by
    near-duplicate segments of the same post appearing on both sides.
    """
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
        # StratifiedGroupKFold with 10 folds: groups = posts (never split a post), stratified by label
        # (keeps ~52% stress on both sides). We take the first fold as the validation set.
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
    """Return (train, val, test) DataFrames with columns id, post_id, subreddit, text, label, confidence.

    This is the single entry point for data in the whole project: every training, inference and
    analysis script calls it, so all models see exactly the same examples. The split files are
    created automatically on first use.
    """
    if not (split_dir / "train.csv").exists():
        make_splits(out_dir=split_dir)
    return tuple(pd.read_csv(split_dir / f"{s}.csv") for s in ("train", "val", "test"))


def print_stats() -> None:
    """Print size, % stress and length of each split, and segments/% stress per subreddit (train)."""
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
