"""STGCN backbone — the graph-AWARE middle rung.

Faithful to Yu, Yin, Zhu, "Spatio-Temporal Graph Convolutional Networks", IJCAI
2018 (arXiv:1709.04875), reference impl hazdzz/STGCN. The ST-Conv block is the
paper's sandwich: temporal gated conv (GLU) -> spatial graph conv -> temporal
gated conv -> LayerNorm -> dropout. We use the paper's first-order (Kipf)
graph conv Θ(D̃^-1/2 (A+I) D̃^-1/2)x — the simplest, most GCN-like variant, and
the one the paper offers as its ~20% cheaper option.

Two deliberate, stated deviations from the reference:
  1. Direct multi-horizon head. Canonical STGCN predicts ONE step and rolls out
     recursively for 12 (the Graph WaveNet paper contrasts exactly this). We
     output all 12 horizons from one forward pass (a Linear over the flattened
     post-block features), so STGCN numbers here are NOT comparable to published
     recursive-STGCN rows -- but the model IS a faithful ST-Conv stack.
  2. Narrow channels (32/16/32) to keep it a genuinely small "simple GCN"
     (~30-90k params) well below GWN, so the ladder spans real capacity.

Temporal convs are 'valid' (no padding), as in the paper: each GLU eats (Kt-1)
steps, so the time axis shrinks through the stack; the head consumes whatever
remains.
"""
import torch
import torch.nn as nn


class _TemporalGLU(nn.Module):
    """gated temporal conv (STGCN GLU): valid conv, halves channels via gate."""

    def __init__(self, c_in: int, c_out: int, kt: int = 3):
        super().__init__()
        self.kt = kt
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(1, kt))  # over time axis
        self.align = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else None

    def forward(self, x):                       # x: (B, C, N, T)
        res = x[..., self.kt - 1:]              # align residual to valid output length
        if self.align is not None:
            res = self.align(res)
        pq = self.conv(x)                       # (B, 2*c_out, N, T-kt+1)
        p, q = pq.chunk(2, dim=1)
        return (p + res) * torch.sigmoid(q)     # GLU with residual


class _GraphConv(nn.Module):
    """first-order Kipf graph conv: X' = Â X W, Â = D̃^-1/2(A+I)D̃^-1/2 (1-hop)."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.theta = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x, A_hat):                # x: (B,C,N,T), A_hat: (N,N)
        x = torch.einsum("nm,bcmt->bcnt", A_hat, x)   # propagate over nodes
        return self.theta(x)


class _STConvBlock(nn.Module):
    def __init__(self, c_in, c_spatial, c_out, kt=3, dropout=0.0):
        super().__init__()
        self.t1 = _TemporalGLU(c_in, c_spatial, kt)
        self.gc = _GraphConv(c_spatial, c_spatial)
        self.t2 = _TemporalGLU(c_spatial, c_out, kt)
        self.norm = nn.LayerNorm(c_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, A_hat):
        x = self.t1(x)
        x = torch.relu(self.gc(x, A_hat))
        x = self.t2(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)   # LN over channels
        return self.drop(x)


class STGCN(nn.Module):
    def __init__(self, n_nodes, t_in=12, t_out=12, kt=3,
                 channels=(32, 16, 32), n_blocks=2, dropout=0.0):
        super().__init__()
        self.t_out = t_out
        c_in, c_s, c_out = channels
        self.start = nn.Conv2d(1, c_in, kernel_size=(1, 1))
        blocks = []
        t = t_in
        cin = c_in
        for _ in range(n_blocks):
            blocks.append(_STConvBlock(cin, c_s, c_out, kt, dropout))
            t -= 2 * (kt - 1)                   # two valid temporal convs per block
            cin = c_out
        self.blocks = nn.ModuleList(blocks)
        if t < 1:
            raise ValueError(f"temporal axis exhausted (t={t}); reduce n_blocks or kt")
        self.t_remain = t
        self.head = nn.Linear(c_out * t, n_nodes and t_out)   # per-node flatten -> t_out

    def forward(self, x, A_hat):                # x: (B, C, N, T) -- uses channel 0
        v = x[:, :1]                            # (B,1,N,T) value channel
        h = self.start(v)
        for blk in self.blocks:
            h = blk(h, A_hat)                   # (B, C, N, t_remain)
        B, C, N, T = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, N, C * T)   # (B, N, C*t_remain)
        out = self.head(h)                      # (B, N, t_out)
        return out.permute(0, 2, 1).unsqueeze(-1)        # (B, t_out, N, 1)
