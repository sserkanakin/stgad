# checkpoint save/load helpers, used by both training scripts for resumability.
import json
from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    sched,
    epoch: int,
    best_val_mae: float,
    rng_states: dict,
) -> None:
    """save everything needed to resume: model, optimiser, scheduler (may be
    None), epoch, best val mae and the rng states."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "best_val_mae": best_val_mae,
        "rng_states": rng_states,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path) -> dict:
    """load a checkpoint written by save_checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def find_latest_checkpoint(ckpt_dir: Path) -> "Path | None":
    """return last.pt in ckpt_dir, or None if it isn't there."""
    candidate = Path(ckpt_dir) / "last.pt"
    return candidate if candidate.exists() else None


def get_rng_states() -> dict:
    """capture the current python / numpy / torch rng states."""
    import random
    import numpy as np

    states: dict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict) -> None:
    """restore rng states captured by get_rng_states."""
    import random
    import numpy as np

    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
