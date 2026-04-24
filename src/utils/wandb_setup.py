# wandb initialisation wrapper with deterministic run IDs and offline fallback.
# Every training script calls init_wandb() instead of wandb.init() directly.
# Loads WANDB_API_KEY from .env at the repo root if not already set.
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env from the repo root into os.environ if WANDB_API_KEY is not set.

    Reads only KEY=VALUE lines; ignores comments and blank lines.
    Does not override variables that are already set in the environment.
    """
    if os.environ.get("WANDB_API_KEY"):
        return  # already set — nothing to do

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def init_wandb(cfg: dict, run_id: str, resume: str = "allow"):
    """Initialise a wandb run with resumability and offline fallback.

    The run uses a deterministic ``run_id`` so that re-running the same
    command after a crash resumes the existing wandb run (same URL, same
    charts) rather than starting a new one.

    Loads ``WANDB_API_KEY`` from ``.env`` at the repo root if not already
    set in the environment.  Falls back to ``mode="offline"`` if no key
    is found.  Offline runs can be synced later with::

        wandb sync ./wandb/offline-run-<timestamp>-<id>

    Args:
        cfg: Full config dict to log as wandb hyperparameters.
        run_id: Deterministic run identifier string, typically
            ``"{backbone}_{dataset}_seed{seed}_{cfg_hash}"``.
        resume: wandb resume mode (default ``"allow"`` — resumes if the
            run exists, creates a new one if not).

    Returns:
        The active ``wandb.Run`` object.
    """
    import wandb

    _load_dotenv()

    mode = os.environ.get("WANDB_MODE", "online")
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        mode = "offline"

    run = wandb.init(
        project="thesis-diffusion",
        entity="tud-thesis",
        id=run_id,
        resume=resume,
        config=cfg,
        mode=mode,
    )
    return run
