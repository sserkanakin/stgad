"""Unit tests for src/diffusion/score_net.py.

We test:
  - Sinusoidal embedding: shape + deterministic + distinct τ → distinct vectors
  - FiLM: identity at init; learns affine modulation
  - GraphConv: shape with adj; behaves with adj=None (no-graph ablation)
  - ScoreNetBlock: shape preservation + residual connection
  - ScoreNet: end-to-end shape; τ conditioning changes output; autograd flows
  - Parameter count is reasonable (< ~10⁶ for default config)
"""
from __future__ import annotations

import pytest
import torch

from src.diffusion.score_net import (
    FiLM,
    GraphConv,
    ScoreNet,
    ScoreNetBlock,
    SinusoidalEmbedding,
)

# Small test config — keeps tests fast on Mac MPS / CPU.
B, H, N = 3, 12, 8
HIDDEN = 16
COND = 32


# ─────────────────────────────────────────────────────────────────────────────
# SinusoidalEmbedding
# ─────────────────────────────────────────────────────────────────────────────

def test_sinusoidal_embedding_shape():
    emb = SinusoidalEmbedding(embed_dim=COND)
    tau = torch.linspace(0.01, 1.0, B)
    out = emb(tau)
    assert out.shape == (B, COND)
    assert torch.isfinite(out).all()


def test_sinusoidal_embedding_distinct_taus_distinct_outputs():
    emb = SinusoidalEmbedding(embed_dim=COND)
    out = emb(torch.tensor([0.01, 0.5, 1.0]))
    # Pairwise differences should be non-zero
    assert (out[0] - out[1]).abs().max() > 1e-3
    assert (out[1] - out[2]).abs().max() > 1e-3


def test_sinusoidal_embedding_deterministic():
    emb = SinusoidalEmbedding(embed_dim=COND)
    tau = torch.tensor([0.3])
    assert torch.allclose(emb(tau), emb(tau))


def test_sinusoidal_embedding_rejects_odd_dim():
    with pytest.raises(ValueError, match="even"):
        SinusoidalEmbedding(embed_dim=33)


# ─────────────────────────────────────────────────────────────────────────────
# FiLM
# ─────────────────────────────────────────────────────────────────────────────

def test_film_identity_at_init():
    """At init, γ=1, β=0, so x' = 1 · x + 0 = x for any cond."""
    film = FiLM(cond_dim=COND, channels=HIDDEN)
    x = torch.randn(B, HIDDEN, H, N)
    cond = torch.randn(B, COND)
    out = film(x, cond)
    assert torch.allclose(out, x, atol=1e-6)


def test_film_shape():
    film = FiLM(cond_dim=COND, channels=HIDDEN)
    x = torch.randn(B, HIDDEN, H, N)
    cond = torch.randn(B, COND)
    assert film(x, cond).shape == x.shape


def test_film_learns_to_modulate():
    """After a gradient step, FiLM is no longer the identity."""
    torch.manual_seed(0)
    film = FiLM(cond_dim=COND, channels=HIDDEN)
    x = torch.randn(B, HIDDEN, H, N)
    cond = torch.randn(B, COND)
    target = torch.randn_like(x)
    opt = torch.optim.SGD(film.parameters(), lr=0.1)
    for _ in range(20):
        opt.zero_grad()
        out = film(x, cond)
        loss = (out - target).pow(2).mean()
        loss.backward()
        opt.step()
    # After training, output should differ from x
    diff = (film(x, cond) - x).abs().max()
    assert diff.item() > 0.01, "FiLM didn't learn to modulate"


# ─────────────────────────────────────────────────────────────────────────────
# GraphConv
# ─────────────────────────────────────────────────────────────────────────────

def test_graphconv_shape_with_adj():
    gcn = GraphConv(in_channels=HIDDEN, out_channels=HIDDEN)
    x = torch.randn(B, HIDDEN, H, N)
    adj = torch.eye(N) + 0.1 * torch.randn(N, N)
    out = gcn(x, adj)
    assert out.shape == x.shape


def test_graphconv_no_adj_still_runs():
    """adj=None falls back to self-feature only (no graph mixing)."""
    gcn = GraphConv(in_channels=HIDDEN, out_channels=HIDDEN)
    x = torch.randn(B, HIDDEN, H, N)
    out = gcn(x, adj=None)
    assert out.shape == x.shape


def test_graphconv_distinct_adj_changes_output():
    torch.manual_seed(0)
    gcn = GraphConv(in_channels=HIDDEN, out_channels=HIDDEN)
    x = torch.randn(B, HIDDEN, H, N)
    adj1 = torch.eye(N)
    adj2 = torch.eye(N) + 0.5 * torch.randn(N, N)
    out1 = gcn(x, adj1)
    out2 = gcn(x, adj2)
    assert (out1 - out2).abs().max() > 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# ScoreNetBlock
# ─────────────────────────────────────────────────────────────────────────────

def test_block_preserves_shape():
    block = ScoreNetBlock(channels=HIDDEN, cond_dim=COND)
    x = torch.randn(B, HIDDEN, H, N)
    adj = torch.eye(N)
    cond = torch.randn(B, COND)
    out = block(x, adj, cond)
    assert out.shape == x.shape


def test_block_no_adj_runs():
    block = ScoreNetBlock(channels=HIDDEN, cond_dim=COND)
    x = torch.randn(B, HIDDEN, H, N)
    cond = torch.randn(B, COND)
    out = block(x, None, cond)
    assert out.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# Full ScoreNet
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def model():
    adj = torch.eye(N) + 0.1 * torch.randn(N, N)
    return ScoreNet(
        num_horizons=H,
        num_sensors=N,
        hidden_dim=HIDDEN,
        num_blocks=2,
        tau_embed_dim=COND,
        adj=adj,
    )


def _make_inputs(B: int = B, H: int = H, N: int = N):
    return (
        torch.randn(B, H, N),               # r_tau
        torch.rand(B) * 0.999 + 0.001,      # tau in (0.001, 1.0)
        torch.randn(B, 2, H, N),            # x_history
        torch.randn(B, H, N),               # y_pred_backbone
    )


def test_score_net_output_shape(model):
    r_tau, tau, x_history, y_pred = _make_inputs()
    out = model(r_tau, tau, x_history, y_pred)
    assert out.shape == (B, H, N)
    assert torch.isfinite(out).all()


def test_score_net_distinct_tau_distinct_output(model):
    """Conditioning works: same r_τ at different τ should give different r̂₀."""
    # Hack: force the FiLM projections to have non-trivial weights so the
    # network isn't τ-invariant at init.
    for block in model.blocks:
        with torch.no_grad():
            block.film.proj.weight.normal_(0.0, 0.1)
    r_tau, _, x_history, y_pred = _make_inputs(B=1)
    out_low = model(r_tau, torch.tensor([0.05]), x_history, y_pred)
    out_high = model(r_tau, torch.tensor([0.95]), x_history, y_pred)
    diff = (out_low - out_high).abs().mean()
    assert diff.item() > 1e-4, f"τ conditioning is dead: mean diff {diff.item():.2e}"


def test_score_net_distinct_r_tau_distinct_output(model):
    """Different r_τ should give different r̂₀."""
    _, tau, x_history, y_pred = _make_inputs(B=1)
    r1 = torch.randn(1, H, N)
    r2 = torch.randn(1, H, N)
    out1 = model(r1, tau, x_history, y_pred)
    out2 = model(r2, tau, x_history, y_pred)
    assert (out1 - out2).abs().mean() > 1e-3


def test_score_net_autograd(model):
    r_tau, tau, x_history, y_pred = _make_inputs()
    r_tau.requires_grad_(True)
    out = model(r_tau, tau, x_history, y_pred)
    loss = out.pow(2).sum()
    loss.backward()
    # Gradient flows into the input
    assert r_tau.grad is not None
    assert torch.isfinite(r_tau.grad).all()
    # And into the model parameters
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms)


def test_score_net_no_adj_runs():
    """ScoreNet should still build + run with adj=None (isotropic-style ablation)."""
    model = ScoreNet(
        num_horizons=H,
        num_sensors=N,
        hidden_dim=HIDDEN,
        num_blocks=2,
        tau_embed_dim=COND,
        adj=None,
    )
    r_tau, tau, x_history, y_pred = _make_inputs()
    out = model(r_tau, tau, x_history, y_pred)
    assert out.shape == (B, H, N)


def test_score_net_rejects_wrong_shapes(model):
    r_tau, tau, x_history, y_pred = _make_inputs()
    # Wrong H in r_tau
    with pytest.raises(ValueError, match=r"r_tau shape"):
        model(torch.randn(B, H + 1, N), tau, x_history, y_pred)
    # tau as 0-d
    with pytest.raises(ValueError, match=r"tau shape"):
        model(r_tau, torch.tensor(0.5), x_history, y_pred)


def test_score_net_param_count_reasonable():
    """Default config should fit comfortably on a Mac MPS / one A40."""
    # Realistic config for METR-LA: hidden=64, blocks=4
    full = ScoreNet(
        num_horizons=12,
        num_sensors=207,
        hidden_dim=64,
        num_blocks=4,
        tau_embed_dim=128,
        adj=torch.eye(207),
    )
    n_params = full.num_parameters
    assert 50_000 < n_params < 5_000_000, f"unexpected param count: {n_params:,}"


def test_score_net_rejects_wrong_adj_shape():
    with pytest.raises(ValueError, match=r"adj shape"):
        ScoreNet(num_sensors=N, adj=torch.eye(N + 1))
