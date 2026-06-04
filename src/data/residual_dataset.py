"""dataset over the cached residual files for diffusion training.

each split file (see src.utils.paths.residual_npz) holds residuals, y_pred,
y_true, mask, mean, std from extract_residuals.py. for training we use:
    R0     residuals  (N, H)
    Y_hat  y_pred     (N, H), the denoiser conditioning
    mask   bool       (N, H), true where the observation is valid
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import paths


class ResidualDataset(Dataset):
    """In-memory wrapper around a residuals .npz."""

    def __init__(self, npz_path: str | Path) -> None:
        d = np.load(npz_path)
        self.R0 = torch.from_numpy(d["residuals"].astype(np.float32))     # (B, N, H)
        self.Y_hat = torch.from_numpy(d["y_pred"].astype(np.float32))     # (B, N, H)
        self.mask = torch.from_numpy(d["mask"].astype(bool))              # (B, N, H)

    def __len__(self) -> int:
        return self.R0.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "R0": self.R0[idx],
            "Y_hat": self.Y_hat[idx],
            "mask": self.mask[idx],
        }


def residual_path(dataset: str, split: str, seed: int = 0) -> Path:
    """Canonical path for the residual npz file."""
    return paths.residual_npz(dataset, split, seed)
