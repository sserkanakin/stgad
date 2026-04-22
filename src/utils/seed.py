# Reproducibility helpers: single entry point to seed all RNG sources.
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch (CPU + GPU/MPS) for reproducibility.

    Call this at the top of every training and evaluation script before any
    stochastic operation.  Seeds are drawn from the fixed list [0,1,2,3,4]
    defined in §6 of the project spec.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # MPS does not expose a separate seed API; torch.manual_seed covers it.
        pass
