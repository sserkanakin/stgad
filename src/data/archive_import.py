"""one-off bootstrap: import the pre-rewrite gwn residuals into the current layout.

the current pipeline reads residuals from results/residuals/<dataset>/seed<seed>/
as (B, N, H) arrays with keys residuals/y_pred/mask/mean/std. the archived run
(~/Projects/research-tud-archive-2026-05-19) saved them as (B, H, N, 1) with a
y_true field instead of mean/std. the windows are identical (same preprocessing,
verified by matching B counts per split), so we just reshape and attach the
signal scaler from scaler.yaml.

this lets stage 0 (prior validation) run on real gwn residuals without retraining
the backbone: the residual *statistics* the prior is tested against (per-horizon
variance growth, spatial correlation, tails) are robust to the exact checkpoint.
regenerate from the current pipeline (train_backbone -> extract_residuals) when
locking the results chapter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils import paths

ARCHIVE = Path.home() / "Projects" / "research-tud-archive-2026-05-19" / "results" / "residuals"


def _to_BNH(arr: np.ndarray) -> np.ndarray:
    """archive stores (B, H, N, 1); drop the channel and reorder to (B, N, H), so
    vec(R) is column-major over N and Cov(vec(R)) = Σ_T ⊗ Σ_S (matches extract_residuals)."""
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]                 # (B, H, N)
    return np.transpose(arr, (0, 2, 1))   # (B, N, H)


def ensure_residuals(dataset: str, seed: int = 0, verbose: bool = True) -> Path:
    """make sure results/residuals/<dataset>/seed<seed>/{train,val,test}.npz exist.

    if missing, import them from the archive (reshaped to the current layout, with
    mean/std taken from scaler.yaml). idempotent: returns the residual directory
    and does nothing if every split is already there.
    """
    out_dir = paths.residual_dir(dataset, seed)
    splits = ("train", "val", "test")
    if all((out_dir / f"{s}.npz").exists() for s in splits):
        return out_dir

    scaler = json.loads(paths.scaler_json(dataset).read_text())
    mean, std = float(scaler["mean"]), float(scaler["std"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in splits:
        src = ARCHIVE / f"gwn_{dataset}_{s}_seed{seed}.npz"
        if not src.exists():
            raise FileNotFoundError(
                f"archive residual not found: {src}\n"
                "expected the pre-rewrite gwn residuals; pass the right seed or "
                "regenerate via scripts/extract_residuals.py."
            )
        d = np.load(src)
        np.savez_compressed(
            out_dir / f"{s}.npz",
            residuals=_to_BNH(d["residuals"]).astype(np.float32),
            y_pred=_to_BNH(d["y_pred"]).astype(np.float32),
            mask=_to_BNH(d["mask"]).astype(bool),
            mean=np.float32(mean),
            std=np.float32(std),
        )
        if verbose:
            print(f"[bootstrap] {src.name}  ->  {out_dir / (s + '.npz')}")
    return out_dir
