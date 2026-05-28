"""Smoke tests for the R₀ denoiser adapter."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.diffusion.denoiser import R0Denoiser, num_parameters


def _toy_adjacency(N: int = 16, seed: int = 0) -> np.ndarray:
    """Small directed adjacency for tests; row-stochastic-ish."""
    rng = np.random.default_rng(seed)
    A = (rng.random((N, N)) < 0.2).astype(np.float32)
    np.fill_diagonal(A, 0)
    A = A * rng.random((N, N)).astype(np.float32)
    return A


@pytest.fixture
def denoiser():
    N, H = 16, 12
    A = _toy_adjacency(N)
    return R0Denoiser(N=N, H=H, adjacency=A, d_h=16, channel_multipliers=(1, 2), n_blocks=1)


def test_forward_shape(denoiser):
    B, N, H = 4, denoiser.N, denoiser.H
    R_t = torch.randn(B, N, H)
    Y_hat = torch.randn(B, N, H)
    t = torch.rand(B)
    out = denoiser(R_t, t, Y_hat)
    assert out.shape == (B, N, H), f"out shape {out.shape} != {(B, N, H)}"


def test_gradient_flow(denoiser):
    B = 2
    R_t = torch.randn(B, denoiser.N, denoiser.H)
    Y_hat = torch.randn(B, denoiser.N, denoiser.H)
    t = torch.full((B,), 0.5)
    out = denoiser(R_t, t, Y_hat)
    loss = (out ** 2).mean()
    loss.backward()
    # At least one parameter should have a non-zero gradient.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in denoiser.parameters()
        if p.requires_grad
    )
    assert has_grad, "no gradients flowed to denoiser parameters"


def test_batch_independence(denoiser):
    """Two batches with identical inputs should give identical outputs."""
    torch.manual_seed(0)
    R = torch.randn(1, denoiser.N, denoiser.H)
    Y = torch.randn(1, denoiser.N, denoiser.H)
    t_single = torch.tensor([0.3])

    denoiser.eval()
    with torch.no_grad():
        out1 = denoiser(R, t_single, Y)
        # Process the same sample inside a batch of size 2.
        R2 = torch.cat([R, torch.randn_like(R)], dim=0)
        Y2 = torch.cat([Y, torch.randn_like(Y)], dim=0)
        t2 = torch.tensor([0.3, 0.7])
        out2 = denoiser(R2, t2, Y2)

    assert torch.allclose(out1[0], out2[0], atol=1e-5), (
        "first batch element changes when other batch elements differ"
    )


def test_continuous_time_smooth(denoiser):
    """Output should vary continuously in t (small t change → small output change)."""
    torch.manual_seed(1)
    R = torch.randn(1, denoiser.N, denoiser.H)
    Y = torch.randn(1, denoiser.N, denoiser.H)

    denoiser.eval()
    with torch.no_grad():
        out_a = denoiser(R, torch.tensor([0.5]), Y)
        out_b = denoiser(R, torch.tensor([0.50001]), Y)
    # Outputs should be very close (Lipschitz in t since the embedding is smooth).
    diff = (out_a - out_b).abs().max().item()
    assert diff < 1e-2, f"|f(0.5) - f(0.5+1e-5)| = {diff} too large"


def test_param_count_logged(denoiser):
    n = num_parameters(denoiser)
    assert n > 0
    print(f"R0Denoiser params: {n:,}")
