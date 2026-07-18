"""DLinear backbone — the graph-BLIND floor of the ladder.

Faithful to Zeng et al., "Are Transformers Effective for Time Series
Forecasting?", AAAI 2023 (arXiv:2205.13504); reference impl cure-lab/LTSF-Linear
`models/DLinear.py`. Series-decomposition (moving-average trend + seasonal
remainder), then two independent linear maps seq_len -> pred_len, summed.

Graph-blind by construction: each node's forecast depends ONLY on that node's
own history (weights shared across nodes, `individual=False`, the BasicTS traffic
default). This is the honest control -- if the graph prior helps on residuals of
a backbone that never saw the graph, that is the cleanest possible evidence.

Deviation from the paper's defaults, stated: the moving-average kernel is 25 in
the paper (long-horizon LTSF, seq_len 336). At seq_len 12 a kernel of 25 would
pad the whole window and collapse the trend to a near-constant, so we use
kernel_size=5 (odd, < seq_len). Everything else is the reference model.
"""
import torch
import torch.nn as nn


class _MovingAvg(nn.Module):
    """moving average with edge replication padding (LTSF-Linear series_decomp)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):                       # x: (B*N, T, 1)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - pad, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class DLinear(nn.Module):
    def __init__(self, n_nodes: int, t_in: int = 12, t_out: int = 12,
                 kernel_size: int = 5, individual: bool = False):
        super().__init__()
        self.n_nodes = n_nodes
        self.t_in, self.t_out = t_in, t_out
        self.decomp = _MovingAvg(kernel_size)
        self.individual = individual
        if individual:
            self.lin_season = nn.ModuleList([nn.Linear(t_in, t_out) for _ in range(n_nodes)])
            self.lin_trend = nn.ModuleList([nn.Linear(t_in, t_out) for _ in range(n_nodes)])
        else:
            self.lin_season = nn.Linear(t_in, t_out)
            self.lin_trend = nn.Linear(t_in, t_out)

    def forward(self, x):                       # x: (B, C, N, T) -- uses channel 0 (value)
        B, C, N, T = x.shape
        v = x[:, 0].permute(0, 2, 1).reshape(B * N, T, 1)   # (B*N, T, 1)
        trend = self.decomp(v)
        season = v - trend
        s = season.squeeze(-1)                  # (B*N, T)
        t = trend.squeeze(-1)
        if self.individual:
            s = s.view(B, N, T); t = t.view(B, N, T)
            out_s = torch.stack([self.lin_season[i](s[:, i]) for i in range(N)], dim=1)
            out_t = torch.stack([self.lin_trend[i](t[:, i]) for i in range(N)], dim=1)
            out = (out_s + out_t)               # (B, N, t_out)
        else:
            out = (self.lin_season(s) + self.lin_trend(t)).view(B, N, self.t_out)
        return out.permute(0, 2, 1).unsqueeze(-1)   # (B, t_out, N, 1)
