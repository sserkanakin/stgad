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
def train_residual_sigma(dataset: str, seed: int = 0, backbone: str = "gwn",
                         norm: str = "global"):
    """residual scale from the TRAIN split (masked), reused verbatim for val/test.

    norm="global":  one scalar std over all masked entries (the historical default).
    norm="pernode": per-sensor std, shape (N,) — sensors differ 3-4x in residual
        scale (heteroscedastic), so per-node unit variance removes the static part
        of that spread before the prior/denoiser see the data (HySRD-Net's node-wise
        normalization ablation: global costs 5-7% CRPS). floored at 10% of the
        global std so a nearly-always-masked sensor cannot blow up the division."""
    d = np.load(paths.residual_npz(dataset, "train", seed, backbone))
    R = d["residuals"].astype(np.float64)
    M = d["mask"].astype(bool)
    g = float(R[M].std())
    if norm == "global":
        return g
    Rm = np.where(M, R, 0.0)
    cnt = M.sum(axis=(0, 2)).clip(min=1)                       # valid entries per node
    var = (Rm ** 2).sum(axis=(0, 2)) / cnt                      # per-node second moment
    return np.sqrt(var).clip(min=0.1 * g)                       # (N,)


def sigma_to_torch(sigma_r, device=None) -> torch.Tensor:
    """broadcastable residual scale: scalar -> (1,1), per-node (N,) -> (N,1).
    multiplies/divides tensors shaped (..., N, H) either way."""
    t = torch.as_tensor(np.asarray(sigma_r), dtype=torch.float32, device=device)
    return t.reshape(1, 1) if t.ndim == 0 else t.reshape(-1, 1)


class ResidualDataset(Dataset):
    """In-memory wrapper around a residuals .npz, normalized to unit space.

    sigma_r is the train-split residual scale (scalar or per-node (N,); pass the
    same value for every split).

    zero_masked: masked coordinates (y_true==0, sensor offline) hold residuals of
        -y_pred — up to ~11 sigma on metrla. the loss never scores them, but the
        graph convs smear them into scored neighbors, so we zero them at the INPUT
        (loss mask + raw y tensors untouched).
    cond_past: prepend the PREVIOUS window's residual field (window i-H, whose
        forecast horizons end exactly at this window's origin — fully observed at
        forecast time, no leakage) to the conditioning: cond = [R_past | Y_hat_z]
        along time, (B, N, H+H). carries recent backbone-error volatility, the
        heteroscedastic structure the forecast alone cannot signal."""

    def __init__(self, npz_path: str | Path, sigma_r, *,
                 cond_past: bool = False, zero_masked: bool = True) -> None:
        d = np.load(npz_path)
        self.sigma_r = sigma_r
        self.cond_past = bool(cond_past)
        sig = sigma_to_torch(sigma_r).numpy()                             # (1,1) or (N,1)
        mean = float(d["mean"])
        std = float(d["std"])
        R0 = d["residuals"].astype(np.float32)
        Y_hat = d["y_pred"].astype(np.float32)
        mask = d["mask"].astype(bool)
        # model residuals in unit-variance space (divide by the residual scaler)
        R0u = R0 / sig[None, :, :]                                        # (B, N, H)
        if zero_masked:
            R0u = np.where(mask, R0u, 0.0).astype(np.float32)
        self.R0 = torch.from_numpy(R0u)
        # condition on the z-scored backbone forecast (the signal scaler, distinct)
        Yz = (Y_hat - mean) / std                                         # (B, N, H)
        if cond_past:
            H = R0u.shape[-1]
            past = np.zeros_like(R0u)                                     # first H windows: none
            src = R0u if zero_masked else np.where(mask, R0u, 0.0).astype(np.float32)
            past[H:] = src[:-H]                                           # window i-H
            self.Y_hat = torch.from_numpy(
                np.concatenate([past, Yz], axis=-1).astype(np.float32))   # (B, N, 2H)
        else:
            self.Y_hat = torch.from_numpy(Yz.astype(np.float32))          # (B, N, H)
        self.mask = torch.from_numpy(mask)                                # (B, N, H)
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
        centered = d["residuals"].astype(np.float32)                     # signal - mean (zero-mean target)
        history = d["history"].astype(np.float32)                        # history, raw
        y_pred = d["y_pred"].astype(np.float32)                          # climatology mean (trivial backbone)
        self.R0 = torch.from_numpy(centered / np.float32(self.sigma_r))  # (B, N, H) z-scored, zero-mean
        self.Y_hat = torch.from_numpy((history - mean) / std)            # (B, N, T_h) z-scored cond
        self.mask = torch.from_numpy(d["mask"].astype(bool))             # (B, N, H)
        self.y_pred_raw = torch.from_numpy(y_pred)                       # mean field (y_pred + σ·sample recon)
        self.y_true_raw = torch.from_numpy(centered + y_pred)            # (B, N, H) raw signal GT

    def __len__(self) -> int:
        return self.R0.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"R0": self.R0[idx], "Y_hat": self.Y_hat[idx], "mask": self.mask[idx]}


def make_dataset(dataset: str, split: str, seed: int, backbone: str, sigma_r,
                 *, cond_past: bool = False, zero_masked: bool = True):
    """ResidualDataset for a trained backbone, or SignalDataset for backbone='raw'
    (the signal arm has no backbone residual to condition on or zero)."""
    path = residual_path(dataset, split, seed, backbone)
    if backbone == "raw":
        return SignalDataset(path, sigma_r)
    return ResidualDataset(path, sigma_r, cond_past=cond_past, zero_masked=zero_masked)


def residual_path(dataset: str, split: str, seed: int = 0, backbone: str = "gwn") -> Path:
    """Canonical path for the residual npz file."""
    return paths.residual_npz(dataset, split, seed, backbone)
