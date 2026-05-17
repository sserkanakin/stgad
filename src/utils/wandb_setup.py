# wandb initialisation wrapper with deterministic run IDs and offline fallback.
# Every training script calls init_wandb() instead of wandb.init() directly.
# Loads WANDB_API_KEY from .env at the repo root if not already set.
import os
from pathlib import Path


def _has_wandb_credentials() -> bool:
    """Return True if any credential source (env var or ~/.netrc) is set up.

    `wandb login` writes the API key to ``~/.netrc``; that's a perfectly
    valid credential source. We previously fell back to offline whenever
    ``WANDB_API_KEY`` was unset, which incorrectly demoted runs to offline
    on machines where the user had run ``wandb login`` (e.g. DAIC).
    """
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            content = netrc_path.read_text()
            return "api.wandb.ai" in content or "wandb.ai" in content
        except OSError:
            return False
    return False


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
    """Initialise a wandb run with per-submission uniqueness + crash resumability.

    The wandb ``id`` is keyed off the Slurm job ID (or process ID when running
    locally), so each manual ``sbatch`` submission creates a *fresh* wandb run
    in the dashboard rather than silently appending to an earlier run that
    happened to share the same config hash. Slurm's own preempt-and-requeue
    keeps the same JOB_ID, so the "resume after preemption" behaviour still
    works — only deliberate human resubmissions get fresh runs.

    The wandb ``name`` (display label) stays the deterministic
    ``"{backbone}_{dataset}_seed{seed}_{cfg_hash}"`` form so you can still
    search/filter by config in the dashboard. Tags are added for the dataset,
    prior, and seed so you can group by those without touching the name.

    Loads ``WANDB_API_KEY`` from ``.env`` at the repo root if not already
    set. Falls back to ``mode="offline"`` if no credential is found, with
    a LOUD warning so the silent-offline mode can't be missed::

        wandb sync ./wandb/offline-run-<timestamp>-<id>

    Args:
        cfg: Full config dict to log as wandb hyperparameters.
        run_id: Deterministic run identifier string, typically
            ``"{backbone}_{dataset}_seed{seed}_{cfg_hash}"``. Used as the
            wandb ``name`` (display label, *not* the unique id).
        resume: wandb resume mode (default ``"allow"``). With per-submission
            unique ids this rarely matters in practice — pass ``"never"``
            if you want an explicit error on (extremely unlikely) collision.

    Returns:
        The active ``wandb.Run`` object.
    """
    import wandb

    _load_dotenv()

    # Choose a unique-per-submission id. SLURM_JOB_ID is stable across
    # preempt/restart of the same job (good — resumes correctly), but
    # different across separate sbatch submissions (good — fresh dashboard
    # entry). Off-cluster we fall back to PID, which is unique enough for
    # local dev / single-shot runs.
    job_suffix = os.environ.get("SLURM_JOB_ID") or f"local{os.getpid()}"
    wandb_unique_id = f"{run_id}__{job_suffix}"

    mode = os.environ.get("WANDB_MODE", "online")
    if mode == "online" and not _has_wandb_credentials():
        # LOUD: silent offline fallback was a foot-gun. Print so the user
        # immediately sees that the dashboard won't update for this run.
        print(
            "[init_wandb] WARNING: WANDB_API_KEY missing and no ~/.netrc "
            "wandb credential. Falling back to mode='offline'. "
            "Run `wandb login` on this machine to enable online runs, "
            "or `wandb sync wandb/offline-run-*` later to push history.",
            flush=True,
        )
        mode = "offline"

    # Tags help group runs in the dashboard without parsing the name string.
    tags = [
        f"dataset:{cfg.get('dataset', 'unknown')}",
        f"prior:{cfg.get('prior', 'unknown')}",
        f"seed:{cfg.get('seed', 'unknown')}",
        f"backbone:{cfg.get('backbone', 'unknown')}",
    ]

    run = wandb.init(
        project="thesis-diffusion",
        entity="tud-thesis",
        id=wandb_unique_id,        # unique per sbatch invocation
        name=run_id,                # display label = deterministic from cfg
        resume=resume,
        config={**cfg, "_slurm_job_id": job_suffix},
        mode=mode,
        tags=tags,
    )
    print(
        f"[init_wandb] mode={mode}  id={wandb_unique_id}  name={run_id}  "
        f"url={getattr(run, 'url', '(offline)')}",
        flush=True,
    )
    return run
