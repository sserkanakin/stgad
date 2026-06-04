# one place to seed every rng source.
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """seed python, numpy and torch (cpu + gpu/mps).

    call once at the top of a script, before anything stochastic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # mps has no separate seed api; torch.manual_seed already covers it.
