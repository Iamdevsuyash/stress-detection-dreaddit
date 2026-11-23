"""Small shared helpers used by every script: seeding, device selection and project paths.

Keeping these in one place guarantees that all experiments write to the same folders and are
seeded the same way, which is what makes the results reproducible and comparable.
"""
import os
import random
from pathlib import Path

import numpy as np

# Project folders, resolved relative to this file so scripts work from any working directory.
ROOT = Path(__file__).resolve().parents[1]  # repository root (the folder that contains src/)
DATA_RAW = ROOT / "data" / "raw"            # original Dreaddit CSVs (downloaded, git-ignored)
DATA_SPLITS = ROOT / "data" / "splits"      # our fixed train/val/test CSVs (git-ignored, regenerated)
RESULTS = ROOT / "results"                  # predictions, metrics, summaries (committed)
FIGURES = ROOT / "figures"                  # plots used in the report
MODELS = ROOT / "models"                    # saved fine-tuned checkpoints (git-ignored, too large)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch so that a run can be repeated.

    The seed controls weight initialisation of the classifier head, data shuffling and dropout.
    We run every trained model with seeds 13, 42 and 87 and report mean +- std, because results on
    a small dataset like Dreaddit change noticeably from seed to seed.

    Report (Experimental Setup, "Seeds"; Results): the spread is real, not a formality. BERT's macro-F1
    barely moves (78.74 +- 0.13) but its stress RECALL swings 79.5 +- 6.8 between seeds, and the BiLSTM without
    GloVe has a macro-F1 std of 1.9. One seed would hide this, which is why every table shows mean +- std.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # makes Python's hash-based ordering repeatable
    try:
        import torch

        torch.manual_seed(seed)            # CPU random number generator
        torch.cuda.manual_seed_all(seed)   # every GPU's random number generator
    except ImportError:
        pass  # torch is optional for CPU-only scripts such as src/analysis.py


def get_device():
    """Return the fastest available PyTorch device: CUDA GPU, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
