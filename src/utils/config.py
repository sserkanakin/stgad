# yaml config loader used by the training/eval scripts.
import hashlib
import json
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    """read a yaml file into a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def hash_config(cfg: dict) -> str:
    """short deterministic hash of a config dict (keys sorted first).

    used for provenance; the dict must be json-serialisable.
    """
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
