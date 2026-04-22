# Thin wrapper around the official STID repo (third_party/STID).
# Do not modify STID source. All downstream code goes through load_stid().
import sys
from pathlib import Path
import torch
import torch.nn as nn

# STID config per dataset (§2c of CLAUDE_CODE_TASK.md / §9 of PROJECT_SPEC.md)
_STID_CONFIGS = {
    "metrla": {
        "num_nodes": 207,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 1,
        "embed_dim": 32,
        "output_dim": 1,
        "num_layer": 3,
        "if_spatial": True,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "time_of_day_size": 288,
        "day_of_week_size": 7,
    },
    "pemsbay": {
        "num_nodes": 325,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 1,
        "embed_dim": 32,
        "output_dim": 1,
        "num_layer": 3,
        "if_spatial": True,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "time_of_day_size": 288,
        "day_of_week_size": 7,
    },
}


def load_stid(dataset: str, device: torch.device) -> nn.Module:
    """Load the STID model from third_party/STID and return it on ``device``.

    Adds ``third_party/STID`` to sys.path, imports the STID architecture
    module, instantiates with the correct per-dataset config, and returns
    the model placed on ``device``.

    Args:
        dataset: One of ``"metrla"`` or ``"pemsbay"``.
        device: Target torch device.

    Returns:
        Instantiated STID nn.Module on ``device``.

    Raises:
        ValueError: if ``dataset`` is not recognised.
        FileNotFoundError: if ``third_party/STID`` has not been cloned.
    """
    raise NotImplementedError
