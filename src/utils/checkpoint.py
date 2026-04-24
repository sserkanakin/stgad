# Checkpoint save/load utilities for backbone and diffusion training.
# Every training script uses these helpers to guarantee resumability.
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
    """Save a full training checkpoint to ``path``.

    The checkpoint contains all state needed to resume training bit-exactly
    (modulo non-deterministic GPU ops): model weights, optimiser state,
    scheduler state, epoch index, best validation MAE, and all RNG states.

    Args:
        path: Destination file path (e.g. ``results/checkpoints/.../last.pt``).
        model: The model whose state_dict to save.
        optim: The optimiser whose state_dict to save.
        sched: LR scheduler (may be None).
        epoch: The epoch that just finished (0-indexed).
        best_val_mae: Best val MAE seen so far (used to decide when to save best.pt).
        rng_states: Dict with keys ``python``, ``numpy``, ``torch_cpu``,
            and optionally ``torch_cuda`` / ``torch_mps``.
    """
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
    """Load a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Path to the ``.pt`` checkpoint file.

    Returns:
        Dict with keys ``model_state_dict``, ``optimizer_state_dict``,
        ``scheduler_state_dict``, ``epoch``, ``best_val_mae``, ``rng_states``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def find_latest_checkpoint(ckpt_dir: Path) -> "Path | None":
    """Return the path to ``last.pt`` in ``ckpt_dir``, or None if absent.

    Args:
        ckpt_dir: Directory that may contain ``last.pt``.

    Returns:
        Path to ``last.pt`` if it exists, else None.
    """
    candidate = Path(ckpt_dir) / "last.pt"
    return candidate if candidate.exists() else None


def get_rng_states() -> dict:
    """Capture current RNG states for all available backends.

    Returns:
        Dict suitable for passing to :func:`save_checkpoint` as ``rng_states``.
    """
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
    """Restore RNG states previously captured by :func:`get_rng_states`.

    Args:
        states: Dict as returned by :func:`get_rng_states`.
    """
    import random
    import numpy as np

    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
