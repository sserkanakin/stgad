"""Unit tests for src/diffusion/sampling.py.

We test that:
  - reverse_learned returns the right shape and produces finite outputs
  - With a trivially-correct oracle r̂₀ = r₀, the reverse SDE recovers
    something close to r₀ from an initial sample drawn from p(r_τ|r_0)
    (the oracle round-trip, mirrors exp-004's Check 2)
  - sample_per_window stacks K trajectories correctly and produces
    finite outputs with shape (B, K, H, N)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.diffusion import sde as sde_np
from src.diffusion.sampling import reverse_learned, sample_per_window
from src.diffusion.sde_torch import eig_to_torch, forward_sample

H, N = 4, 5


class _OracleX0Net(torch.nn.Module):
    """Cheating 'model' that ignores its noisy input and returns the stored r_0.

    Used to verify the reverse-SDE plumbing in isolation: if the score
    network were perfect (r̂₀ = r₀), reverse sampling should land near r₀.
    """

    def __init__(self, r_0_truth: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("r_0", r_0_truth.clone())

    def forward(self, r_tau, tau, x_history, y_pred):
        # Ignore all inputs except the static buffer.
        return self.r_0


@pytest.fixture(scope="module")
def prior():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((H, H))
    Q_H = A @ A.T + 0.5 * np.eye(H)
    B = rng.standard_normal((N, N))
    Sigma_S = B @ B.T + 0.5 * np.eye(N)
    eig_np = sde_np.eigendecompose_prior(Q_H, Sigma_S)
    return eig_to_torch(eig_np, device="cpu", dtype=torch.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Plumbing: shape + finite
# ─────────────────────────────────────────────────────────────────────────────

def test_reverse_learned_shape_and_finite(prior):
    B = 3
    r_0 = torch.randn(B, H, N, dtype=torch.float64)
    model = _OracleX0Net(r_0)
    g = torch.Generator().manual_seed(0)
    r_init = torch.randn(B, H, N, dtype=torch.float64, generator=g)
    x_history = torch.randn(B, 2, H, N, dtype=torch.float64)
    y_pred = torch.randn(B, H, N, dtype=torch.float64)
    out = reverse_learned(
        model, r_init, x_history, y_pred,
        n_steps=20, eig=prior, beta_min=0.01, beta_max=50.0,
        tau_start=1.0, tau_eps=1e-3, generator=g,
    )
    assert out.shape == (B, H, N)
    assert torch.isfinite(out).all()


# ─────────────────────────────────────────────────────────────────────────────
# Oracle round-trip — exp-004 Check 2 in torch
# ─────────────────────────────────────────────────────────────────────────────

def test_oracle_reverse_recovers_r_0(prior):
    """With an oracle r̂₀=r₀, the reverse SDE from r_τ=1 should land near r₀.

    Mirrors the exp-004 Check-2 PASS-strict result but in the torch
    implementation. We accept rel.recovery < 0.20 (the PASS-acceptable
    threshold) because we're using a smaller synthetic prior and float64.
    """
    B = 16
    torch.manual_seed(0)
    r_0 = torch.randn(B, H, N, dtype=torch.float64)
    model = _OracleX0Net(r_0)

    # Forward to τ=1
    g = torch.Generator().manual_seed(1)
    r_tau1 = forward_sample(r_0, tau=1.0, eig=prior, beta_min=0.01, beta_max=50.0,
                            generator=g)

    # Reverse with the oracle x₀
    g2 = torch.Generator().manual_seed(2)
    r_recovered = reverse_learned(
        model, r_tau1,
        x_history=torch.zeros(B, 2, H, N, dtype=torch.float64),
        y_pred=torch.zeros(B, H, N, dtype=torch.float64),
        n_steps=100, eig=prior, beta_min=0.01, beta_max=50.0,
        tau_eps=1e-3, generator=g2,
    )

    # Per-sample relative recovery
    err = (r_recovered - r_0).reshape(B, -1).norm(dim=-1)
    norm_truth = r_0.reshape(B, -1).norm(dim=-1)
    rel = (err / norm_truth).mean().item()
    assert rel < 0.20, f"oracle round-trip rel.recovery = {rel:.3f} (expected < 0.20)"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-sample wrapper
# ─────────────────────────────────────────────────────────────────────────────

def test_sample_per_window_shape(prior):
    B, K = 2, 7
    model = _OracleX0Net(torch.zeros(B * K, H, N, dtype=torch.float64))
    g = torch.Generator().manual_seed(0)
    x_history = torch.randn(B, 2, H, N, dtype=torch.float64)
    y_pred = torch.randn(B, H, N, dtype=torch.float64)
    out = sample_per_window(
        model, x_history, y_pred,
        n_samples=K, n_steps=10, eig=prior,
        beta_min=0.01, beta_max=50.0, generator=g,
    )
    assert out.shape == (B, K, H, N)
    assert torch.isfinite(out).all()


def test_sample_per_window_different_samples_differ(prior):
    """K samples per window should differ — the noise injection makes each
    reverse trajectory unique."""
    B, K = 1, 8
    model = _OracleX0Net(torch.zeros(B * K, H, N, dtype=torch.float64))
    g = torch.Generator().manual_seed(0)
    x_history = torch.randn(B, 2, H, N, dtype=torch.float64)
    y_pred = torch.randn(B, H, N, dtype=torch.float64)
    out = sample_per_window(
        model, x_history, y_pred,
        n_samples=K, n_steps=20, eig=prior,
        beta_min=0.01, beta_max=50.0, generator=g,
    )
    # All K samples shouldn't be identical
    sample_std = out[0].std(dim=0)
    assert (sample_std > 1e-6).any(), "K samples are identical — noise injection broken"
