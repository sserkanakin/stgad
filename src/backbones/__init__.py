# Thin wrappers around official DCRNN and Graph WaveNet repos (third_party/).
# Do not modify backbone source. All downstream code uses load_backbone().
#
# Wraps:
#   DCRNN_PyTorch — github.com/chnsh/DCRNN_PyTorch  at commit d92490b
#   Graph-WaveNet — github.com/nnzhan/Graph-WaveNet  at commit 6b162e8
# License: MIT (both repos)
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"

# Per-dataset node counts
_NUM_NODES = {"metrla": 207, "pemsbay": 325}

# DCRNN model kwargs  (note: DCRNN source reads 'rnn_units', not 'hidden_state_size')
_DCRNN_BASE = {
    "input_dim": 2,
    "output_dim": 1,
    "seq_len": 12,
    "horizon": 12,
    "rnn_units": 64,
    "num_rnn_layers": 2,
    "filter_type": "dual_random_walk",
    "max_diffusion_step": 2,
    "cl_decay_steps": 1000,
    "use_curriculum_learning": False,
}

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
    "aptinit": None,
}


# ---------------------------------------------------------------------------
# DCRNN wrapper
# ---------------------------------------------------------------------------

class _DCRNNWrapper(nn.Module):
    """Reshape adapter so DCRNN fits the (B, T, N, C) → (B, T, N, 1) contract.

    DCRNN internally expects:
        inputs:  (T, B, N*C)     — time-first, nodes and features flattened
        outputs: (T, B, N)       — time-first, one output per node

    This wrapper translates to/from the standard (B, T, N, C) convention
    used by TrafficDataset and the training / verify scripts.
    """

    def __init__(self, dcrnn_model: nn.Module) -> None:
        super().__init__()
        self.model = dcrnn_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 12, N, 2)  normalised speed + time_of_day

        Returns:
            (B, 12, N, 1)  predicted normalised speed per horizon step
        """
        B, T, N, C = x.shape
        # (B, T, N, C) → (T, B, N*C)
        x_in = x.permute(1, 0, 2, 3).reshape(T, B, N * C)
        out = self.model(x_in)              # (12, B, N)
        # (12, B, N) → (B, 12, N, 1)
        return out.permute(1, 0, 2).unsqueeze(-1)


def _mock_tensorflow() -> None:
    """Pre-install a minimal TensorFlow stub so lib/utils.py can be imported.

    lib/utils.py in DCRNN_PyTorch has a top-level ``import tensorflow as tf``
    and uses tf only in two functions (add_simple_summary, get_total_trainable
    _parameter_size) that we never call.  We stub out the tf module so the
    rest of lib/utils.py (calculate_random_walk_matrix, etc.) is importable
    without TensorFlow being installed.
    """
    import types

    if "tensorflow" not in sys.modules:
        import importlib.machinery

        tf_stub = types.ModuleType("tensorflow")
        # torch._dynamo.trace_rules calls find_spec() on every module in sys.modules;
        # a ModuleSpec with loader=None satisfies find_spec() without raising.
        tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
        # Stub only the symbols that lib/utils.py references at module level
        tf_stub.Summary = None
        tf_stub.trainable_variables = lambda: []
        sys.modules["tensorflow"] = tf_stub


def _load_dcrnn(dataset: str, adj_mx: np.ndarray, device: torch.device) -> nn.Module:
    """Instantiate DCRNN from third_party/DCRNN_PyTorch.

    DCRNN's dcrnn_cell.py allocates sparse support tensors on a module-level
    ``device`` (cuda | cpu) at import time.  MPS is not supported for sparse
    ops.  We therefore always use ``cuda`` if available, else ``cpu``.
    See DECISIONS.md: "DCRNN falls back to CPU on MPS machines".
    """
    dcrnn_path = _THIRD_PARTY / "DCRNN_PyTorch"
    if not dcrnn_path.exists():
        raise FileNotFoundError(
            f"DCRNN repo not found at {dcrnn_path}.\n"
            "Run: bash scripts/download_data.sh"
        )

    dcrnn_str = str(dcrnn_path)
    if dcrnn_str not in sys.path:
        sys.path.insert(0, dcrnn_str)

    _mock_tensorflow()   # must run before the DCRNN import chain

    from model.pytorch.dcrnn_model import DCRNNModel  # noqa: PLC0415

    # DCRNN sparse tensors are created on the module-level device.
    # We must match it to avoid device mismatch in forward.
    dcrnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = dict(_DCRNN_BASE)
    kwargs["num_nodes"] = _NUM_NODES[dataset]

    logger = logging.getLogger(f"dcrnn.{dataset}")
    model = DCRNNModel(adj_mx, logger, **kwargs)
    model = model.to(dcrnn_device)

    wrapper = _DCRNNWrapper(model)
    # Store the effective device so training scripts can use it
    wrapper.effective_device = dcrnn_device
    return wrapper


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

    model = gwnet(
        device=device,
        num_nodes=num_nodes,
        supports=None,
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
    """Instantiate a graph backbone from third_party/ and return it on device.

    Both wrappers expose a uniform forward signature::

        pred = model(x)   # x: (B, 12, N, 2) → pred: (B, 12, N, 1)

    Args:
        name: ``"dcrnn"`` or ``"gwn"``.
        dataset: ``"metrla"`` or ``"pemsbay"``.
        adj_mx: N×N float32 adjacency matrix from ``adjacency.load_adjacency``.
        device: Requested torch device.  Note: DCRNN ignores MPS and uses
            CPU (see DECISIONS.md).

    Returns:
        Untrained model wrapper on the effective device.

    Raises:
        ValueError: Unknown backbone or dataset name.
        FileNotFoundError: third_party repo not cloned yet.
    """
    name = name.lower()
    dataset = dataset.lower()

    if name not in ("dcrnn", "gwn"):
        raise ValueError(f"Unknown backbone '{name}'. Choose 'dcrnn' or 'gwn'.")
    if dataset not in ("metrla", "pemsbay"):
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'metrla' or 'pemsbay'.")

    if name == "dcrnn":
        return _load_dcrnn(dataset, adj_mx, device)
    else:
        return _load_gwn(dataset, adj_mx, device)
