# pytorch dataset over a preprocessed split .npz.
# normalisation happens at load time (not save time), per the dcrnn convention.
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.config import load_config


class TrafficDataset(Dataset):
    """wraps a raw (un-normalised) split .npz.

    on load we z-score x's speed channel (channel 0) with the saved scaler; the
    time-of-day channel and y (raw units) are left untouched. each item is a dict
    with x (12, N, 2), y (12, N, 1), x_mask, y_mask. exposes .mean and .std.
    """

    def __init__(self, npz_path: str, scaler_path: str) -> None:
        """load the raw npz and z-score x's speed channel using scaler_path."""
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
