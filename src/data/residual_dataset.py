"""dataset over the cached residual files for diffusion training.

each split file (see src.utils.paths.residual_npz) holds residuals, y_pred,
mask, mean, std from extract_residuals.py. for training we use:
    R0     residuals  (N, H)
    Y_hat  y_pred     (N, H), the denoiser conditioning
    mask   bool       (N, H), true where the observation is valid

we model in unit-variance space: R0 is divided by sigma_r (a single global
scalar, the std of the TRAIN residuals, frozen and reused for val/test) and
Y_hat is z-scored by the input scaler (mean/std in the npz, the signal scale
GWN trained on). these are two DIFFERENT scalers. the raw residuals are
recovered downstream as r_raw = r_unit * sigma_r (de-normalize at the output).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import paths


@lru_cache(maxsize=8)
def train_residual_sigma(dataset: str, seed: int = 0, backbone: str = "gwn") -> float:
    """single global std of the TRAIN residuals (masked). one scalar per dataset,
    computed on train only and reused verbatim for val/test, so the modeling
    space is unit-variance without leaking the val distribution (which differs
    from train by design on pems-bay - that drift is a result, not noise)."""
    d = np.load(paths.residual_npz(dataset, "train", seed, backbone))
    R = d["residuals"].astype(np.float64)
    M = d["mask"].astype(bool)
    return float(R[M].std())


class ResidualDataset(Dataset):
    """In-memory wrapper around a residuals .npz, normalized to unit space.

    sigma_r is the train-split residual std (pass the same value for every split).
    """

    def __init__(self, npz_path: str | Path, sigma_r: float) -> None:
        d = np.load(npz_path)
        self.sigma_r = float(sigma_r)
        mean = float(d["mean"])
        std = float(d["std"])
        R0 = d["residuals"].astype(np.float32)
        Y_hat = d["y_pred"].astype(np.float32)
        # model residuals in unit-variance space (divide by the residual scaler)
        self.R0 = torch.from_numpy(R0 / np.float32(self.sigma_r))         # (B, N, H)
        # condition on the z-scored backbone forecast (the signal scaler, distinct)
        self.Y_hat = torch.from_numpy((Y_hat - mean) / std)              # (B, N, H)
        self.mask = torch.from_numpy(d["mask"].astype(bool))             # (B, N, H)
        # raw backbone forecast + ground truth (raw units), for MAPE and the PEMS
        # y==0 eval mask. y_true = y_pred + residual (both raw); residual = R0·sigma_r.
        self.y_pred_raw = torch.from_numpy(Y_hat)                         # (B, N, H) raw
        self.y_true_raw = torch.from_numpy(Y_hat + R0)                    # (B, N, H) raw

    def __len__(self) -> int:
        return self.R0.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "R0": self.R0[idx],
            "Y_hat": self.Y_hat[idx],
            "mask": self.mask[idx],
        }


class SignalDataset(Dataset):
    """pure-signal arm (backbone='raw'): diffuse the FUTURE signal conditioned on the
    HISTORY window (DiffSTG-style), no trained backbone. the future signal is stored
    under the "residuals" key (so the prior + sigma_r reuse the residual path), and the
    conditioning is the z-scored history (NOT a forecast). y_pred is zeros, so the
    reconstructed forecast = denoiser output and y_true = the signal.

    sigma_r is the train-signal std (same role as in ResidualDataset). R0 = signal/σ;
    Y_hat = z-scored history (B, N, T_h); y_true_raw = signal (raw)."""

    def __init__(self, npz_path: str | Path, sigma_r: float) -> None:
        d = np.load(npz_path)
        self.sigma_r = float(sigma_r)
        mean = float(d["mean"])
        std = float(d["std"])
        signal = d["residuals"].astype(np.float32)                       # future signal, raw
        history = d["history"].astype(np.float32)                        # history, raw
        self.R0 = torch.from_numpy(signal / np.float32(self.sigma_r))    # (B, N, H) unit space
        self.Y_hat = torch.from_numpy((history - mean) / std)            # (B, N, T_h) z-scored cond
        self.mask = torch.from_numpy(d["mask"].astype(bool))             # (B, N, H)
        self.y_pred_raw = torch.zeros_like(self.R0)                      # no backbone
        self.y_true_raw = torch.from_numpy(signal)                       # (B, N, H) raw GT

    def __len__(self) -> int:
        return self.R0.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"R0": self.R0[idx], "Y_hat": self.Y_hat[idx], "mask": self.mask[idx]}


def make_dataset(dataset: str, split: str, seed: int, backbone: str, sigma_r: float):
    """ResidualDataset for a trained backbone, or SignalDataset for backbone='raw'."""
    path = residual_path(dataset, split, seed, backbone)
    return SignalDataset(path, sigma_r) if backbone == "raw" else ResidualDataset(path, sigma_r)


def residual_path(dataset: str, split: str, seed: int = 0, backbone: str = "gwn") -> Path:
    """Canonical path for the residual npz file."""
    return paths.residual_npz(dataset, split, seed, backbone)
