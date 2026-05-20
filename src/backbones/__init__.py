# Thin wrapper around the official Graph WaveNet repo (third_party/Graph-WaveNet).
# Do not modify backbone source. All downstream code uses load_backbone().
#
# Wraps:
#   Graph-WaveNet — github.com/nnzhan/Graph-WaveNet  at commit 6b162e8
# License: MIT
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"

# Per-dataset node counts
_NUM_NODES = {"metrla": 207, "pemsbay": 325, "pems08f": 170}

# Graph WaveNet model kwargs
_GWN_BASE = {
    "dropout": 0.3,
    "in_dim": 2,
    "out_dim": 12,
    "residual_channels": 32,
    "dilation_channels": 32,
    "skip_channels": 256,
    "end_channels": 512,
    "kernel_size": 2,
    "blocks": 4,
    "layers": 2,
    "gcn_bool": True,
    "addaptadj": True,
    # aptinit is set per-call from supports[0] (forward random-walk transition),
    # matching the original GWN train.py default (randomadj=False).
}


def _asym_adj(adj: np.ndarray) -> np.ndarray:
    """Row-normalised (asymmetric) transition matrix: D^-1 @ A.

    Matches util.asym_adj in third_party/Graph-WaveNet — copied here so we
    don't import scipy/sparse at module load and so we keep one numpy path
    for both METR-LA/PEMS-BAY (directed) and PEMS08F (symmetric binary)
    adjacencies.
    """
    rowsum = adj.sum(axis=1)
    d_inv = np.where(rowsum > 0, 1.0 / rowsum, 0.0)
    return (np.diag(d_inv) @ adj).astype(np.float32)


# ---------------------------------------------------------------------------
# Graph WaveNet wrapper
# ---------------------------------------------------------------------------

def _patch_gwn_conv1d(model: nn.Module) -> nn.Module:
    """Replace nn.Conv1d layers in GWN's gate/skip/residual lists with nn.Conv2d.

    The original GWN source uses ``nn.Conv1d(kernel_size=(1, k))`` which stores
    a 4D weight tensor ``(out, in, 1, k)`` — identical to ``nn.Conv2d(kernel_size=(1,k))``.
    In PyTorch >= 2.x, ``nn.Conv1d.forward`` strictly requires 3D input, so calling
    it with the 4D feature map ``(B, C, N, T)`` raises an error.  We swap each
    affected layer for an equivalent ``nn.Conv2d``.  Weights are copied by shape
    (the data layouts are the same).
    """
    for list_attr in ("gate_convs", "residual_convs", "skip_convs"):
        conv_list = getattr(model, list_attr)
        for i, conv in enumerate(conv_list):
            if not isinstance(conv, nn.Conv1d):
                continue

            ks = conv.kernel_size                              # e.g. (1, 2)
            dl = conv.dilation                                 # e.g. (1,)  — 1-tuple
            pd = conv.padding if isinstance(conv.padding, tuple) else (conv.padding,)

            ks2 = tuple(ks) if len(ks) == 2 else (1, ks[0])
            dl2 = (1, dl[0]) if len(dl) == 1 else tuple(dl)
            pd2 = (0, pd[0]) if len(pd) == 1 else tuple(pd)

            new_conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=ks2,
                dilation=dl2,
                padding=pd2,
                bias=conv.bias is not None,
            ).to(conv.weight.device)

            with torch.no_grad():
                new_conv.weight.data.copy_(conv.weight.data.view(new_conv.weight.shape))
                if conv.bias is not None:
                    new_conv.bias.data.copy_(conv.bias.data)

            conv_list[i] = new_conv   # nn.ModuleList.__setitem__ re-registers the module

    return model


def _load_gwn(dataset: str, adj_mx: np.ndarray, device: torch.device) -> nn.Module:
    """Instantiate Graph WaveNet from third_party/Graph-WaveNet.

    GWN's forward signature is already (B, C, N, T) → (B, 12, N, 1), but
    our convention is (B, T, N, C) → (B, T, N, 1).  The reshape is done
    inside _GWNWrapper so the training script sees a uniform interface.
    """
    gwn_path = _THIRD_PARTY / "Graph-WaveNet"
    if not gwn_path.exists():
        raise FileNotFoundError(
            f"Graph-WaveNet repo not found at {gwn_path}.\n"
            "Run: bash scripts/download_data.sh"
        )

    gwn_str = str(gwn_path)
    if gwn_str not in sys.path:
        sys.path.insert(0, gwn_str)

    from model import gwnet  # noqa: PLC0415

    num_nodes = _NUM_NODES[dataset]
    kwargs = dict(_GWN_BASE)

    # Double-transition supports: forward + backward random walk on A.
    # Matches the original GWN training recipe (adjtype="doubletransition").
    supports = [
        torch.from_numpy(_asym_adj(adj_mx)).to(device),
        torch.from_numpy(_asym_adj(adj_mx.T)).to(device),
    ]
    aptinit = supports[0]   # initialise adaptive adj from forward transition

    model = gwnet(
        device=device,
        num_nodes=num_nodes,
        supports=supports,
        aptinit=aptinit,
        **kwargs,
    )
    model = model.to(device)
    model = _patch_gwn_conv1d(model)   # fix Conv1d → Conv2d for PyTorch >= 2.x

    wrapper = _GWNWrapper(model)
    wrapper.effective_device = device
    return wrapper


class _GWNWrapper(nn.Module):
    """Reshape adapter: (B, T, N, C) → GWN → (B, T, N, 1).

    GWN forward expects (B, C, N, T) and returns (B, horizon, N, 1).
    """

    def __init__(self, gwn_model: nn.Module) -> None:
        super().__init__()
        self.model = gwn_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 12, N, 2)  normalised speed + time_of_day

        Returns:
            (B, 12, N, 1)  predicted normalised speed per horizon step
        """
        # (B, T, N, C) → (B, C, N, T)
        x_in = x.permute(0, 3, 2, 1)   # (B, 2, N, 12)
        out = self.model(x_in)           # (B, 12, N, 1)
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_backbone(
    name: str,
    dataset: str,
    adj_mx: np.ndarray,
    device: torch.device,
) -> nn.Module:
    """Instantiate the GWN backbone from third_party/ and return it on device.

    The wrapper exposes a uniform forward signature::

        pred = model(x)   # x: (B, 12, N, 2) → pred: (B, 12, N, 1)

    Args:
        name: Must be ``"gwn"``. Kept as a parameter for call-site clarity.
        dataset: ``"metrla"``, ``"pemsbay"``, or ``"pems08f"``.
        adj_mx: N×N float32 adjacency matrix from ``adjacency.load_adjacency``.
            Used to build the double-transition supports
            ``[D^-1 @ A,  D^-1 @ A^T]`` passed into GWN, plus ``aptinit =
            supports[0]`` for adaptive-adjacency initialisation. Matches
            the original GWN training recipe (adjtype="doubletransition",
            randomadj=False).
        device: Requested torch device.

    Returns:
        Untrained model wrapper on the effective device.

    Raises:
        ValueError: Unknown backbone or dataset name.
        FileNotFoundError: third_party repo not cloned yet.
    """
    name = name.lower()
    dataset = dataset.lower()

    if name != "gwn":
        raise ValueError(f"Unknown backbone '{name}'. Only 'gwn' is supported.")
    if dataset not in _NUM_NODES:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from {sorted(_NUM_NODES)}."
        )

    return _load_gwn(dataset, adj_mx, device)
