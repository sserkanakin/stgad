# PyTorch Dataset wrapping a preprocessed .npz file.
import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    """Wraps a preprocessed ``{dataset}_{split}.npz`` file.

    Each item is a dict with tensor keys:
        x        float32  (12, N, 1)   normalised input
        y        float32  (12, N, 1)   normalised target
        x_mask   bool     (12, N, 1)   False = missing
        y_mask   bool     (12, N, 1)
        x_tod    int32    (12,)        time-of-day index
        x_dow    int32    (12,)        day-of-week index
        y_tod    int32    (12,)
        y_dow    int32    (12,)
    """

    def __init__(self, npz_path: str) -> None:
        """Args:
            npz_path: Path to the processed .npz file.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError
