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
    model_state: dict | None = None,
    extra: dict | None = None,
) -> None:
    """save everything needed to resume: model, optimiser, scheduler (may be
    None), epoch, best val mae and the rng states. model_state overrides the
    module's own weights (used to checkpoint EMA weights so downstream eval
    loads them transparently); extra keys are merged into the payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model_state if model_state is not None else model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "best_val_mae": best_val_mae,
        "rng_states": rng_states,
    }
    if extra:
        ckpt.update(extra)
    # write atomically (temp + rename): if the disk fills mid-save, the partial
    # lands on a .tmp and the real best.pt/last.pt is never a truncated half-file
    # that a later resume/eval would fail to read (PytorchStreamReader error).
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


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
