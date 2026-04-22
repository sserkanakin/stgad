# YAML config loader used by all training and evaluation scripts.
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
