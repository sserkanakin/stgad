"""Graph WaveNet backbone — the strong graph-AWARE top of the ladder.

Thin wrapper around the vendored `third_party/Graph-WaveNet/model.py::gwnet`
(Wu et al., IJCAI 2019, arXiv:1906.00121) used as a MODEL CLASS ONLY, under our
own training loop. Paper defaults: 8 layers (blocks=4, layers=2), dilations 1,2,
K=2, in_dim=2 = [value_z, time_of_day], out_dim=12 direct multi-horizon.

Canonical supports = `doubletransition` = [asym_adj(A), asym_adj(A^T)] (forward +
backward random-walk transition), plus the learned adaptive adjacency
(addaptadj=True). This is the paper's best METR-LA row. GWN is the ONLY backbone
that consumes time-of-day, exactly as in the reference (in_dim=2).

Requires the model.py Conv1d->Conv2d patch (THIRD_PARTY_PATCHES.md) on torch>=2.x.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2] / "third_party" / "Graph-WaveNet"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from model import gwnet          # noqa: E402
import util as _gwn_util         # noqa: E402


def _double_transition(A: np.ndarray):
    """GWN's doubletransition supports: forward + backward random-walk adjacency."""
    A = np.asarray(A, dtype=np.float64)
    return [np.asarray(_gwn_util.asym_adj(A)),
            np.asarray(_gwn_util.asym_adj(A.T))]


class GWN(nn.Module):
    def __init__(self, n_nodes, A, device="cpu", t_in=12, t_out=12,
                 dropout=0.3, nhid=32, in_dim=2):
        super().__init__()
        supports = [torch.as_tensor(s, dtype=torch.float32, device=device)
                    for s in _double_transition(A)]
        self.net = gwnet(
            device=device, num_nodes=n_nodes, dropout=dropout, supports=supports,
            gcn_bool=True, addaptadj=True, aptinit=None,
            in_dim=in_dim, out_dim=t_out,
            residual_channels=nhid, dilation_channels=nhid,
            skip_channels=nhid * 8, end_channels=nhid * 16,
            kernel_size=2, blocks=4, layers=2,
        )
        # guard the silent-failure trap: gcn_bool w/o supports/addaptadj disables graph conv
        assert self.net.gcn_bool and (self.net.supports is not None or self.net.addaptadj), \
            "GWN graph conv would be silently disabled"

    def forward(self, x):                       # x: (B, C, N, T) with C=in_dim
        return self.net(x)                      # (B, t_out, N, 1)
