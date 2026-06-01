"""Loaders that reconstruct an eval-ready model + data from disk.

The point of ``load_eval_checkpoint`` is that evaluation never hard-codes
hyperparameters: it reads the ``config.json`` written next to the checkpoint by
``scripts/train_diffusion.py`` and rebuilds the denoiser, prior, and FCPS
schedule exactly as trained. Pass it a checkpoint directory such as
``results/checkpoints/diffusion/diff_gwn_metrla_seed0_<hash>``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.adjacency import load_adjacency
from src.data.residual_dataset import ResidualDataset, residual_path
from src.diffusion.denoiser import R0Denoiser
from src.diffusion.forward_fcps import FCPSSchedule
from src.diffusion.prior import KroneckerPrior, load_prior
from src.utils.checkpoint import load_checkpoint


@dataclass
class EvalBundle:
    """Everything needed to sample from a trained diffusion model."""

    denoiser: R0Denoiser
    prior: KroneckerPrior
    schedule: FCPSSchedule
    cfg: dict
    device: torch.device


def pick_device() -> torch.device:
    """Same precedence as training: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_eval_checkpoint(
    ckpt_dir: str | Path,
    *,
    which: str = "best",
    device: torch.device | None = None,
) -> EvalBundle:
    """Rebuild (denoiser, prior, schedule, cfg) from a training checkpoint dir.

    Parameters
    ----------
    ckpt_dir : directory containing ``config.json`` and ``best.pt`` / ``last.pt``.
    which : "best" or "last".
    device : override; defaults to :func:`pick_device`.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    device = device or pick_device()

    dataset = cfg["dataset"]
    prior = load_prior(dataset)
    schedule = FCPSSchedule(
        c_min=cfg["c_min"], c_0=cfg["c_0"], alpha=cfg["alpha"], T=1.0
    )

    A = load_adjacency(dataset)
    denoiser = R0Denoiser(
        N=prior.N,
        H=prior.H,
        adjacency=A,
        d_h=cfg["d_h"],
        channel_multipliers=tuple(cfg["channel_multipliers"]),
        n_blocks=cfg["n_blocks"],
        device=str(device),
    ).to(device)

    ckpt_path = ckpt_dir / f"{which}.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "last.pt"
    state = load_checkpoint(ckpt_path)
    denoiser.load_state_dict(state["model_state_dict"])
    denoiser.eval()

    return EvalBundle(
        denoiser=denoiser, prior=prior, schedule=schedule, cfg=cfg, device=device
    )


def load_split(dataset: str, split: str, seed: int = 0) -> ResidualDataset:
    """Load a residual split (train/val/test) as a ``ResidualDataset``."""
    return ResidualDataset(residual_path(dataset, split, seed))


def load_split_arrays(dataset: str, split: str, seed: int = 0) -> dict[str, np.ndarray]:
    """Raw arrays from a residual split: residuals, y_pred, y_true, mask, mean, std."""
    d = np.load(residual_path(dataset, split, seed))
    return {k: d[k] for k in d.files}
