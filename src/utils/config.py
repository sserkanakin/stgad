# YAML config loader used by all training and evaluation scripts.
import hashlib
import json
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    """Read a YAML file and return its contents as a plain Python dict.

    Args:
        path: Path to the YAML file (absolute or relative to cwd).

    Returns:
        Dictionary of config values.

    Raises:
        FileNotFoundError: if the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def hash_config(cfg: dict) -> str:
    """Return a deterministic 8-character MD5 hex digest of a config dict.

    Used to generate stable ``run_id`` strings for wandb resumability.
    Keys are sorted before hashing so insertion order does not matter.

    Args:
        cfg: Plain dict of config values (must be JSON-serialisable).

    Returns:
        8-character lowercase hex string.
    """
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
