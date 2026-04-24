# PyTorch Dataset wrapping a preprocessed .npz file.
# Normalisation is applied at load time (not save time) per DCRNN convention.
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.config import load_config


class TrafficDataset(Dataset):
    """Wraps a raw (un-normalised) ``{dataset}_{split}.npz`` file.

    At instantiation, z-score normalisation is applied to x's speed channel
    (channel 0) only.  y stays in raw mph units throughout — never normalised.
    The time_of_day channel (x[..., 1]) is also left as-is (already in [0, 1)).

    Each item is a dict:
        x        float32  (12, N, 2)   normalised-speed + time_of_day
        y        float32  (12, N, 1)   raw speed in mph
        x_mask   bool     (12, N, 1)   True = valid reading
        y_mask   bool     (12, N, 1)

    Attributes:
        mean (float): Training-set speed mean used for normalisation.
        std  (float): Training-set speed std used for normalisation.
    """

    def __init__(self, npz_path: str, scaler_path: str) -> None:
        """Load raw npz and apply normalisation to x speed channel.

        Args:
            npz_path: Path to the raw processed .npz file.
            scaler_path: Path to ``{dataset}_scaler.yaml`` with mean/std.
        """
        raw = np.load(npz_path)
        scaler = load_config(scaler_path)
        self.mean = float(scaler["mean"])
        self.std = float(scaler["std"])

        # Copy so we can modify without altering the mmap'd file
        x = raw["x"].copy()                   # (N_windows, 12, N, 2)
        x[..., 0] = (x[..., 0] - self.mean) / self.std  # normalise speed only

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(raw["y"].copy())          # raw mph, never normalised
        self.x_mask = torch.from_numpy(raw["x_mask"].copy())
        self.y_mask = torch.from_numpy(raw["y_mask"].copy())

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "x_mask": self.x_mask[idx],
            "y_mask": self.y_mask[idx],
        }
