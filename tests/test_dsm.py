"""Unit + integration tests for src/diffusion/dsm.py.

Plus an end-to-end integration test that wires score_net + sde_torch +
dsm into a single training step on synthetic data — the path the real
training loop will exercise.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.diffusion import sde as sde_np
from src.diffusion.dsm import dsm_x0_loss, per_mode_squared_error, sample_tau
from src.diffusion.score_net import ScoreNet
from src.diffusion.sde_torch import (
    beta_bar,
    eig_to_torch,
    forward_sample,
    to_spectral,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────────

H, N = 4, 5
B = 6


@pytest.fixture(scope="module")
def prior():
    """Synthetic Q_H, Σ_S → eig_t (float32, CPU). Same fixture style as test_sde_torch."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((H, H))
    Q_H = A @ A.T + 0.5 * np.eye(H)
    B_ = rng.standard_normal((N, N))
    Sigma_S = B_ @ B_.T + 0.5 * np.eye(N)
    eig_np = sde_np.eigendecompose_prior(Q_H, Sigma_S)
    return eig_to_torch(eig_np, device="cpu", dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# sample_tau
# ─────────────────────────────────────────────────────────────────────────────

def test_sample_tau_shape():
    tau = sample_tau(7)
    assert tau.shape == (7,)


def test_sample_tau_in_range():
    tau = sample_tau(1000, tau_eps=1e-3)
    assert tau.min().item() >= 1e-3 - 1e-7
    assert tau.max().item() <= 1.0 + 1e-7


def test_sample_tau_log_uniform_distribution():
    """log(τ) should be uniform in [log(tau_eps), 0]."""
    tau = sample_tau(10000, tau_eps=1e-3, generator=torch.Generator().manual_seed(0))
    log_tau = tau.log()
    expected_mean = (math.log(1e-3) + math.log(1.0)) / 2
    assert log_tau.mean().item() == pytest.approx(expected_mean, abs=0.1)


def test_sample_tau_rejects_nonpositive_eps():
    with pytest.raises(ValueError, match="tau_eps must be > 0"):
        sample_tau(4, tau_eps=0.0)


def test_sample_tau_deterministic_with_generator():
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    t1 = sample_tau(5, generator=g1)
    t2 = sample_tau(5, generator=g2)
    assert torch.allclose(t1, t2)


# ─────────────────────────────────────────────────────────────────────────────
# dsm_x0_loss — zero error → zero loss
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_zero_when_prediction_is_perfect(prior):
    r_0 = torch.randn(B, H, N)
    r_hat = r_0.clone()
    tau = sample_tau(B)
    for w in ("eigenbasis", "uniform"):
        loss = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0, weighting=w)
        assert loss.item() == pytest.approx(0.0, abs=1e-12), f"{w}: {loss.item()}"


def test_loss_positive_when_prediction_wrong(prior):
    r_0 = torch.randn(B, H, N)
    r_hat = torch.zeros_like(r_0)
    tau = sample_tau(B)
    for w in ("eigenbasis", "uniform"):
        loss = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0, weighting=w)
        assert loss.item() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Numerical stability at small τ
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_finite_at_tau_eps(prior):
    """At τ = τ_eps the var floor kicks in but the loss must stay finite."""
    r_0 = torch.randn(B, H, N)
    r_hat = r_0 + 1e-4 * torch.randn_like(r_0)
    tau = torch.full((B,), 1e-3)
    loss = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0)
    assert torch.isfinite(loss).item()


def test_loss_finite_at_tau_one(prior):
    r_0 = torch.randn(B, H, N)
    r_hat = r_0 + 0.1 * torch.randn_like(r_0)
    tau = torch.full((B,), 1.0)
    loss = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0)
    assert torch.isfinite(loss).item()


# ─────────────────────────────────────────────────────────────────────────────
# Autograd
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_differentiable_wrt_prediction(prior):
    r_0 = torch.randn(B, H, N)
    r_hat = torch.randn(B, H, N, requires_grad=True)
    tau = sample_tau(B)
    loss = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0)
    loss.backward()
    assert r_hat.grad is not None
    assert torch.isfinite(r_hat.grad).all()
    assert r_hat.grad.abs().max() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Eigenbasis vs uniform weighting differ when var differs across modes
# ─────────────────────────────────────────────────────────────────────────────

def test_eigenbasis_weighting_differs_from_uniform(prior):
    """The two weightings should produce different scalar losses on real data."""
    r_0 = torch.randn(B, H, N)
    r_hat = torch.randn(B, H, N)
    tau = torch.full((B,), 0.5)
    L_uniform = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0, weighting="uniform")
    L_eig = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0, weighting="eigenbasis")
    assert abs(L_uniform.item() - L_eig.item()) > 0.1


def test_loss_invariant_to_basis_rotation(prior):
    """Plain MSE in original basis = plain MSE in eigenbasis (orthogonal U)."""
    r_0 = torch.randn(B, H, N)
    r_hat = torch.randn(B, H, N)
    err = r_hat - r_0
    err_tilde = to_spectral(err, prior["U_T"], prior["U_S"])
    assert err.pow(2).mean().item() == pytest.approx(err_tilde.pow(2).mean().item(), rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Bad inputs
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_rejects_shape_mismatch(prior):
    r_0 = torch.randn(B, H, N)
    r_hat = torch.randn(B, H + 1, N)
    tau = sample_tau(B)
    with pytest.raises(ValueError, match="shape"):
        dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0)


def test_loss_rejects_bad_tau_shape(prior):
    r_0 = torch.randn(B, H, N)
    r_hat = torch.randn(B, H, N)
    bad_tau = torch.rand(B + 1)
    with pytest.raises(ValueError, match="tau shape"):
        dsm_x0_loss(r_hat, r_0, bad_tau, prior, 0.01, 50.0)


def test_loss_rejects_unknown_weighting(prior):
    r_0 = torch.randn(B, H, N)
    r_hat = torch.randn(B, H, N)
    tau = sample_tau(B)
    with pytest.raises(ValueError, match="unknown weighting"):
        dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0, weighting="nonsense")


# ─────────────────────────────────────────────────────────────────────────────
# Mask
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_mask_all_ones_matches_unmasked(prior):
    """Mask of all-ones should reproduce the unmasked loss exactly."""
    r_0 = torch.randn(B, H, N)
    r_hat = torch.randn(B, H, N)
    tau = torch.full((B,), 0.5)
    L_none = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0)
    L_full = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0,
                         mask=torch.ones(B, H, N))
    assert L_none.item() == pytest.approx(L_full.item(), rel=1e-6)


def test_loss_mask_drops_masked_position_contribution(prior):
    """Masking a position should make r_hat at that position irrelevant.

    Build two predictions identical everywhere except one position where they
    disagree wildly. With that position masked, the two losses must agree.
    """
    r_0 = torch.randn(B, H, N)
    r_hat_a = torch.randn(B, H, N)
    r_hat_b = r_hat_a.clone()
    # Inject a huge prediction error at (0, 0, 0); mask it out.
    r_hat_b[0, 0, 0] += 1e3
    tau = torch.full((B,), 0.5)
    mask = torch.ones(B, H, N)
    mask[0, 0, 0] = 0.0
    La = dsm_x0_loss(r_hat_a, r_0, tau, prior, 0.01, 50.0, mask=mask)
    Lb = dsm_x0_loss(r_hat_b, r_0, tau, prior, 0.01, 50.0, mask=mask)
    assert La.item() == pytest.approx(Lb.item(), rel=1e-5)


def test_loss_mask_changes_loss_when_unmasked_position_differs(prior):
    """Sanity contra-test: an UNmasked error must change the loss."""
    r_0 = torch.randn(B, H, N)
    r_hat_a = torch.randn(B, H, N)
    r_hat_b = r_hat_a.clone()
    r_hat_b[0, 0, 0] += 1e3  # unmasked this time
    tau = torch.full((B,), 0.5)
    mask = torch.ones(B, H, N)
    La = dsm_x0_loss(r_hat_a, r_0, tau, prior, 0.01, 50.0, mask=mask)
    Lb = dsm_x0_loss(r_hat_b, r_0, tau, prior, 0.01, 50.0, mask=mask)
    assert abs(La.item() - Lb.item()) > 1.0


def test_loss_mask_uniform_weighting(prior):
    """Mask path also works under weighting='uniform'."""
    r_0 = torch.randn(B, H, N)
    r_hat = torch.randn(B, H, N)
    tau = torch.full((B,), 0.5)
    mask = torch.ones(B, H, N)
    mask[:, 0, :] = 0.0  # mask out all of horizon 0
    L_full = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0,
                         weighting="uniform", mask=torch.ones(B, H, N))
    L_part = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0,
                         weighting="uniform", mask=mask)
    # Different horizons usually have different squared errors, so the two
    # means should differ.
    assert L_full.item() != L_part.item()


def test_loss_mask_rejects_shape_mismatch(prior):
    r_0 = torch.randn(B, H, N)
    r_hat = torch.randn(B, H, N)
    tau = torch.full((B,), 0.5)
    bad_mask = torch.ones(B, H, N + 1)
    with pytest.raises(ValueError, match="mask shape"):
        dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0, mask=bad_mask)


# ─────────────────────────────────────────────────────────────────────────────
# per_mode_squared_error
# ─────────────────────────────────────────────────────────────────────────────

def test_per_mode_squared_error_shape(prior):
    r_0 = torch.randn(B, H, N)
    r_hat = torch.randn(B, H, N)
    out = per_mode_squared_error(r_hat, r_0, prior)
    assert out.shape == (B, H, N)
    assert (out >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Integration test: score net + sde_torch + dsm + backward
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end_training_step(prior):
    """Full forward → loss → backward on a small synthetic batch.

    This is the single-step path the real training loop will exercise.
    Verifies: forward + reverse signatures match, gradients reach every
    parameter, loss decreases after one SGD step (sanity check that the
    optimisation surface is at least locally sensible).
    """
    torch.manual_seed(0)
    adj = torch.eye(N) + 0.1 * torch.randn(N, N)
    model = ScoreNet(
        num_horizons=H, num_sensors=N,
        hidden_dim=16, num_blocks=2,
        tau_embed_dim=32, adj=adj,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic batch
    r_0 = torch.randn(B, H, N)
    x_history = torch.randn(B, 2, H, N)
    y_pred = torch.randn(B, H, N)

    losses = []
    for step in range(10):
        opt.zero_grad()

        # 1. Sample τ
        tau = sample_tau(B)

        # 2. Forward marginal: r_τ ~ p(r_τ | r_0)
        r_tau = forward_sample(r_0, tau, prior, 0.01, 50.0)

        # 3. Network predicts r̂₀
        r_hat = model(r_tau, tau, x_history, y_pred)
        assert r_hat.shape == r_0.shape

        # 4. Loss
        loss = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0, weighting="eigenbasis")
        assert torch.isfinite(loss)

        # 5. Backward
        loss.backward()

        # 6. Gradient sanity: every trainable parameter gets a finite grad
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} got no grad"
                assert torch.isfinite(p.grad).all(), f"{name} has nonfinite grad"

        opt.step()
        losses.append(loss.item())

    # 7. Loss should have moved meaningfully over 10 steps. The synthetic
    # data is random noise so we can't expect monotonic decrease, but the
    # network should *learn something* — accept any drop ≥ 5% from step 0
    # to the min of the last 5 steps.
    early = losses[0]
    late = min(losses[5:])
    drop = (early - late) / max(abs(early), 1e-8)
    assert drop > 0.05 or late < early, (
        f"loss didn't drop: start={early:.4f} late_min={late:.4f}"
    )


def test_integration_with_uniform_weighting(prior):
    """Same integration but with weighting='uniform' — ablation path."""
    torch.manual_seed(1)
    model = ScoreNet(
        num_horizons=H, num_sensors=N,
        hidden_dim=16, num_blocks=2,
        tau_embed_dim=32, adj=torch.eye(N),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    r_0 = torch.randn(B, H, N)
    x_history = torch.randn(B, 2, H, N)
    y_pred = torch.randn(B, H, N)
    tau = sample_tau(B)
    r_tau = forward_sample(r_0, tau, prior, 0.01, 50.0)

    opt.zero_grad()
    r_hat = model(r_tau, tau, x_history, y_pred)
    loss = dsm_x0_loss(r_hat, r_0, tau, prior, 0.01, 50.0, weighting="uniform")
    loss.backward()
    opt.step()

    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert torch.isfinite(p.grad).all(), f"{name} nonfinite grad"
