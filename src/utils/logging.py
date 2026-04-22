# wandb wrapper: falls back to offline mode when WANDB_API_KEY is not set.
import os
import wandb


def init_wandb(config: dict, name: str, tags: list[str] | None = None) -> None:
    """Initialise a wandb run, falling back to offline mode if not logged in.

    Args:
        config: Full experiment config dict (logged as wandb config).
        name: Run name following the convention ``{phase}_{dataset}_{prior}_seed{seed}``.
        tags: Optional list of tag strings (dataset, prior, seed, phase).
    """
    mode = "offline" if not os.environ.get("WANDB_API_KEY") else "online"
    wandb.init(
        project="struct-residual-diffusion-traffic",
        name=name,
        config=config,
        tags=tags or [],
        mode=mode,
    )
