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
        global std so a nearly-always-masked sensor cannot blow up the division.
    norm="perhorizon": per-horizon std, shape (1, H) — forecast-error residuals grow
        1.7-2.9x across the horizon; a STATIONARY graph prior (GAD-ST) is flat by
        construction, so we standardize the growth away before diffusion and multiply
        it back at the output (12 scalars; the horizon-growing uncertainty re-enters
        every sample at de-normalization)."""
    d = np.load(paths.residual_npz(dataset, "train", seed, backbone))
    R = d["residuals"].astype(np.float64)
    M = d["mask"].astype(bool)
    g = float(R[M].std())
    if norm == "global":
        return g
    Rm = np.where(M, R, 0.0)
    if norm == "perhorizon":
        cnt = M.sum(axis=(0, 1)).clip(min=1)                   # valid entries per horizon
        var = (Rm ** 2).sum(axis=(0, 1)) / cnt                  # per-horizon second moment
        return np.sqrt(var).clip(min=0.1 * g).reshape(1, -1)    # (1, H)
    cnt = M.sum(axis=(0, 2)).clip(min=1)                       # valid entries per node
    var = (Rm ** 2).sum(axis=(0, 2)) / cnt                      # per-node second moment
    return np.sqrt(var).clip(min=0.1 * g)                       # (N,)


def sigma_to_torch(sigma_r, device=None) -> torch.Tensor:
    """broadcastable residual scale: scalar -> (1,1), per-node (N,) -> (N,1),
    per-horizon (1,H) -> unchanged. multiplies/divides tensors shaped (..., N, H)."""
    t = torch.as_tensor(np.asarray(sigma_r), dtype=torch.float32, device=device)
    if t.ndim == 2:
        return t                                                # (1, H) per-horizon
    return t.reshape(1, 1) if t.ndim == 0 else t.reshape(-1, 1)


def _load_history(d, npz_path, dataset, split, n_rows) -> np.ndarray:
    """observed history window (B, N, P) in RAW units for the cond_past arm.

    prefers the 'history' key inside the residual npz; otherwise reads the
    processed split it was extracted from. raises when neither is available —
    never silently substitutes a different conditioning field."""
    if "history" in d:
        return d["history"].astype(np.float32)
    if dataset is None or split is None:
        raise ValueError(
            f"cond_past needs the observed history: {npz_path} has no 'history' key "
            "and no dataset/split was passed to locate the processed split. "
            "call make_dataset(...), or re-run scripts/extract_residuals.py.")
    xp = paths.split_npz(dataset, split)
    if not Path(xp).exists():
        raise FileNotFoundError(
            f"cond_past needs the observed history: {npz_path} has no 'history' key "
            f"and the processed split {xp} is missing.")
    x = np.load(xp)["x"]                                       # (B, T, N, 1) raw
    hist = np.ascontiguousarray(x[..., 0].transpose(0, 2, 1))  # (B, N, P)
    if hist.shape[0] != n_rows:
        raise ValueError(
            f"history/residual row mismatch for {dataset}/{split}: processed split has "
            f"{hist.shape[0]} windows, residual cache has {n_rows}. the caches are stale "
            "— re-run scripts/extract_residuals.py for this cell.")
    return hist.astype(np.float32)


class ResidualDataset(Dataset):
    """In-memory wrapper around a residuals .npz, normalized to unit space.

    sigma_r is the train-split residual scale (scalar or per-node (N,); pass the
    same value for every split).

    zero_masked: masked coordinates (y_true==0, sensor offline) hold residuals of
        -y_pred — up to ~11 sigma on metrla. the loss never scores them, but the
        graph convs smear them into scored neighbors, so we zero them at the INPUT
        (loss mask + raw y tensors untouched).
    cond_past: prepend the observed HISTORY window to the conditioning:
        cond = [X_z | Y_hat_z] along time, (B, N, P+H). the history is the window
        the backbone consumed to make this forecast, so it is available at forecast
        time by construction and carries the recent traffic state the point forecast
        summarises away.

        the history comes from 'history' in the residual npz when present
        (extract_residuals.py writes it); otherwise it is read straight from the
        processed split (data/processed/<ds>/<split>.npz['x']), which is the SAME
        array — extract_residuals.py consumes that split in order, no shuffle, so
        row i lines up by construction. reading it there means the ~10G of residual
        caches never have to be regenerated to gain the channel. pass dataset/split
        (make_dataset does) to enable that path.

        there is deliberately NO fallback to the previous window's residual field.
        that used to be the silent default and it is a DIFFERENT experiment
        (cond = [R_prev | Y_hat]); a run that cannot find the history must fail
        loudly rather than quietly train the wrong arm."""

    def __init__(self, npz_path: str | Path, sigma_r, *,
                 cond_past: bool = False, zero_masked: bool = True,
                 dataset: str | None = None, split: str | None = None) -> None:
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
            hist = _load_history(d, npz_path, dataset, split, R0u.shape[0])
            past = (hist - mean) / std                                    # (B, N, P) z-scored
            self.Y_hat = torch.from_numpy(
                np.concatenate([past, Yz], axis=-1).astype(np.float32))   # (B, N, P+H)
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


def make_dataset(dataset: str, split: str, seed: int, backbone: str, sigma_r,
                 *, cond_past: bool = False, zero_masked: bool = True):
    """the ResidualDataset for one (dataset, split, backbone) cell."""
    path = residual_path(dataset, split, seed, backbone)
    return ResidualDataset(path, sigma_r, cond_past=cond_past, zero_masked=zero_masked,
                           dataset=dataset, split=split)


def residual_path(dataset: str, split: str, seed: int = 0, backbone: str = "gwn") -> Path:
    """Canonical path for the residual npz file."""
    return paths.residual_npz(dataset, split, seed, backbone)
