"""Small shared helpers: seeding, device selection, paths."""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
