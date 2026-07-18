"""Backbone factory + shared I/O contract for the point-forecast ladder.

Three backbones, each faithful to a real reference (see each module), spanning
graph-awareness and capacity:

    mlp  = DLinear   graph-blind, ~300 params  (Zeng et al. 2023)
    gcn  = STGCN     graph-aware, ~30-90k       (Yu et al. 2018, compact, direct head)
    gwn  = GWN       graph-aware, ~300k         (Wu et al. 2019, vendored)

Shared contract so the SAME training loop drives all three:
- input  x: (B, C_in, N, T_in)   channel 0 = z-scored value; channel 1 = tod (GWN only)
- output  : (B, T_out, N, 1)     z-scored value forecast, all horizons at once
- graph-aware backbones receive their adjacency at construction.

Each backbone consumes its CANONICAL input form (that is "using them correctly"):
GWN takes [value, tod] (in_dim=2); DLinear and STGCN take the value channel only,
as in their references. The residual r = y - ŷ is each backbone's honest error.
"""
import numpy as np

from .dlinear import DLinear
from .stgcn import STGCN
from .gwn import GWN
from .graph import kipf_norm_adj, to_tensor

# which backbones consume time-of-day as an extra input channel
IN_DIM = {"mlp": 1, "gcn": 1, "gwn": 2}
NEEDS_GRAPH = {"mlp": False, "gcn": True, "gwn": True}


def build_backbone(name: str, n_nodes: int, A: np.ndarray, device="cpu",
                   t_in: int = 12, t_out: int = 12):
    """construct a backbone by name. A is the (repaired) weighted/binary adjacency."""
    if name == "mlp":
        return DLinear(n_nodes, t_in, t_out)
    if name == "gcn":
        model = STGCN(n_nodes, t_in, t_out)
        model._A_hat = to_tensor(kipf_norm_adj(A), device=device)   # cached 1-hop adj
        return model
    if name == "gwn":
        return GWN(n_nodes, A, device=device, t_in=t_in, t_out=t_out)
    raise ValueError(f"unknown backbone {name!r} (mlp|gcn|gwn)")


def forward_backbone(name: str, model, x):
    """uniform forward: STGCN needs its adjacency threaded in; others don't."""
    if name == "gcn":
        return model(x, model._A_hat)
    return model(x)
