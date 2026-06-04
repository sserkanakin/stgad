# wandb setup. every training script calls init_wandb() instead of wandb.init()
# directly, so runs are named and grouped the same way everywhere.
import os
from pathlib import Path

from src.utils import paths


def _has_wandb_credentials() -> bool:
    # an api key in the env or a wandb entry in ~/.netrc (written by `wandb login`)
    # both count as valid credentials.
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    if netrc.exists():
        try:
            content = netrc.read_text()
            return "api.wandb.ai" in content or "wandb.ai" in content
        except OSError:
            return False
    return False


def _load_dotenv() -> None:
    # load WANDB_API_KEY from .env at the repo root if it isn't already set.
    if os.environ.get("WANDB_API_KEY"):
        return
    env_path = paths.ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def init_wandb(
    cfg: dict,
    *,
    stage: str,
    dataset: str,
    seed: int,
    prior: str | None = None,
    backbone: str = "gwn",
    resume: str = "allow",
):
    """start a wandb run with consistent naming and grouping.

    grouping in the dashboard:
      project   thesis-diffusion
      job_type  the stage (backbone / diffusion / eval)
      group     "{stage}-{dataset}" (prior appended for diffusion/eval), so
                multi-seed runs of the same config collapse together
      name      run_name(stage, dataset, seed, prior) — matches the results
                folder and the slurm log file
      tags      dataset, backbone, prior, seed — to filter on

    the run id is the name plus the slurm job id, so each manual sbatch gets a
    fresh dashboard entry while slurm's own preempt/requeue (same job id)
    resumes the existing one. falls back to offline mode (loudly) if there is
    no wandb credential.
    """
    import wandb

    _load_dotenv()

    name = paths.run_name(stage, dataset, seed, prior)
    group = f"{stage}-{dataset}" + (f"-{prior}" if prior else "")

    job_suffix = os.environ.get("SLURM_JOB_ID") or f"local{os.getpid()}"
    unique_id = f"{name}__{job_suffix}"

    mode = os.environ.get("WANDB_MODE", "online")
    if mode == "online" and not _has_wandb_credentials():
        print(
            "[init_wandb] WARNING: no wandb credential found, falling back to "
            "offline. run `wandb login` to go online, or push later with "
            "`wandb sync wandb/offline-run-*`.",
            flush=True,
        )
        mode = "offline"

    tags = [dataset, backbone, f"seed{seed}"]
    if prior:
        tags.append(prior)

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "thesis-diffusion"),
        entity=os.environ.get("WANDB_ENTITY", "tud-thesis"),
        id=unique_id,
        name=name,
        group=group,
        job_type=stage,
        resume=resume,
        config={**cfg, "_slurm_job_id": job_suffix},
        mode=mode,
        tags=tags,
    )
    print(
        f"[init_wandb] mode={mode}  group={group}  name={name}  "
        f"url={getattr(run, 'url', '(offline)')}",
        flush=True,
    )
    return run
