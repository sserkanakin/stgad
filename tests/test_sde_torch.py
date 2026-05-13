"""Unit tests for src/diffusion/sde_torch.py.

The torch adapter should numerically agree with the NumPy reference in
``src/diffusion/sde.py`` to better than 1e-5 (float32) on every
deterministic op (basis rotations, decay/var, mean, Q-application).
Sampling ops are tested by matching empirical moments rather than
bit-equivalent draws — torch and numpy use different RNGs.

Same synthetic 4×5 prior as test_sde.py so the two test files share the
notion of "correct" up to floating-point precision.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.diffusion import sde as sde_np
from src.diffusion.sde_torch import (
    apply_Q,
    eig_to_torch,
    forward_marginal_moments,
    forward_sample,
    from_spectral,
    sample_stationary,
    to_spectral,
    beta_bar,
)

H, N = 4, 5
TORCH_FLOAT_TOL = 1e-5
TORCH_DOUBLE_TOL = 1e-10


@pytest.fixture(scope="module")
def prior():
    """Synthetic (Q_H, Σ_S, eig_np, eig_t_f32, eig_t_f64) from a fixed seed."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((H, H))
    Q_H = A @ A.T + 0.5 * np.eye(H)
    B = rng.standard_normal((N, N))
    Sigma_S = B @ B.T + 0.5 * np.eye(N)
    eig_np = sde_np.eigendecompose_prior(Q_H, Sigma_S)
    eig_t_f32 = eig_to_torch(eig_np, device="cpu", dtype=torch.float32)
    eig_t_f64 = eig_to_torch(eig_np, device="cpu", dtype=torch.float64)
    return Q_H, Sigma_S, eig_np, eig_t_f32, eig_t_f64


# ─────────────────────────────────────────────────────────────────────────────
# Conversion shape + dtype
# ─────────────────────────────────────────────────────────────────────────────

def test_eig_to_torch_shapes_and_dtypes(prior):
    _, _, eig_np, eig_t, _ = prior
    for k in ("U_T", "c_T", "U_S", "c_S", "s", "q"):
        assert eig_t[k].shape == eig_np[k].shape
        assert eig_t[k].dtype == torch.float32
    # values should match
    np.testing.assert_allclose(eig_t["s"].numpy(), eig_np["s"], atol=TORCH_FLOAT_TOL)
    np.testing.assert_allclose(eig_t["q"].numpy(), eig_np["q"], atol=TORCH_FLOAT_TOL)


def test_eig_to_torch_missing_key_raises(prior):
    _, _, eig_np, _, _ = prior
    bad = {k: v for k, v in eig_np.items() if k != "s"}
    with pytest.raises(KeyError, match=r"missing required key 's'"):
        eig_to_torch(bad)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_spectral_roundtrip(prior):
    _, _, _, eig_t, _ = prior
    r = torch.randn(7, H, N)  # batch
    r_back = from_spectral(to_spectral(r, eig_t["U_T"], eig_t["U_S"]),
                           eig_t["U_T"], eig_t["U_S"])
    assert torch.allclose(r_back, r, atol=TORCH_FLOAT_TOL)


def test_spectral_matches_numpy(prior):
    """Bit-for-bit torch ↔ numpy on the same random input."""
    _, _, eig_np, _, eig_t64 = prior
    r = np.random.default_rng(7).standard_normal((3, H, N))
    out_np = sde_np.to_spectral(r, eig_np["U_T"], eig_np["U_S"])
    out_t = to_spectral(torch.as_tensor(r, dtype=torch.float64),
                        eig_t64["U_T"], eig_t64["U_S"])
    np.testing.assert_allclose(out_t.numpy(), out_np, atol=TORCH_DOUBLE_TOL)


# ─────────────────────────────────────────────────────────────────────────────
# apply_Q matches numpy
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_Q_matches_numpy(prior):
    _, _, eig_np, _, eig_t64 = prior
    r = np.random.default_rng(11).standard_normal((2, H, N))
    out_np = sde_np.apply_Q(r, eig_np)
    out_t = apply_Q(torch.as_tensor(r, dtype=torch.float64), eig_t64)
    np.testing.assert_allclose(out_t.numpy(), out_np, atol=TORCH_DOUBLE_TOL)


# ─────────────────────────────────────────────────────────────────────────────
# Schedule
# ─────────────────────────────────────────────────────────────────────────────

def test_beta_bar_scalar_matches_numpy():
    for tau in (0.0, 0.25, 0.5, 1.0):
        bb_np = sde_np.beta_bar(tau, 0.01, 50.0)
        bb_t = beta_bar(torch.tensor(tau, dtype=torch.float64), 0.01, 50.0)
        assert bb_t.item() == pytest.approx(bb_np, abs=1e-12)


def test_beta_bar_tensor_broadcast():
    """Per-batch τ should broadcast cleanly."""
    tau = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    bb = beta_bar(tau, 0.01, 50.0)
    assert bb.shape == (3,)
    assert bb[0].item() == pytest.approx(0.0)
    assert bb[2].item() == pytest.approx(0.01 + 0.5 * (50.0 - 0.01))


# ─────────────────────────────────────────────────────────────────────────────
# Forward marginal: mean (deterministic) matches numpy
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_mean_matches_numpy(prior):
    """μ_τ(r_0) is deterministic; should match numpy to float-prec."""
    _, _, eig_np, _, eig_t64 = prior
    r_0_np = np.random.default_rng(13).standard_normal((4, H, N))
    r_0_t = torch.as_tensor(r_0_np, dtype=torch.float64)
    tau, b_min, b_max = 0.4, 0.01, 50.0

    mean_t, sigma_t = forward_marginal_moments(r_0_t, tau, eig_t64, b_min, b_max)

    # Reproduce μ in numpy by replaying the eigenbasis decay
    decay_np, var_np = sde_np._forward_decay_var(tau, eig_np, b_min, b_max)
    r_0_tilde_np = sde_np.to_spectral(r_0_np, eig_np["U_T"], eig_np["U_S"])
    mean_tilde_np = decay_np * r_0_tilde_np
    mean_np = sde_np.from_spectral(mean_tilde_np, eig_np["U_T"], eig_np["U_S"])

    np.testing.assert_allclose(mean_t.numpy(), mean_np, atol=TORCH_DOUBLE_TOL)
    np.testing.assert_allclose(sigma_t.numpy(), np.sqrt(var_np), atol=TORCH_DOUBLE_TOL)


def test_forward_sample_at_tau_zero_is_identity(prior):
    """At τ=0 the marginal collapses to δ(r_τ - r_0)."""
    _, _, _, eig_t, _ = prior
    r_0 = torch.randn(3, H, N)
    g = torch.Generator(); g.manual_seed(0)
    r_tau = forward_sample(r_0, tau=0.0, eig=eig_t, beta_min=0.01, beta_max=50.0,
                           generator=g)
    assert torch.allclose(r_tau, r_0, atol=TORCH_FLOAT_TOL)


def test_forward_sample_variance_matches_analytic(prior):
    """Empirical per-mode variance should match the closed-form forward variance.

    The forward marginal variance at time τ in the joint eigenbasis is
        var[i,j] = s[i,j] · (1 - exp(-β̄(τ) / s[i,j]))
    For modes with β̄ ≫ s this approaches s (stationary); for modes with
    β̄ ≪ s it stays small. We compare to the analytic value (not to s),
    so the test passes for any β_max — even where the schedule doesn't
    drive every mode to stationarity.
    """
    _, _, _, eig_t, _ = prior
    n = 10000
    tau, b_min, b_max = 1.0, 0.01, 50.0
    r_0 = torch.zeros(n, H, N)
    g = torch.Generator(); g.manual_seed(0)
    r_tau = forward_sample(r_0, tau=tau, eig=eig_t, beta_min=b_min, beta_max=b_max,
                           generator=g)
    r_tilde = to_spectral(r_tau, eig_t["U_T"], eig_t["U_S"])
    var_emp = (r_tilde ** 2).mean(dim=0)
    # Analytic per-mode variance
    bb = beta_bar(torch.tensor(tau, dtype=eig_t["s"].dtype), b_min, b_max)
    var_expected = -eig_t["s"] * torch.expm1(-bb / eig_t["s"])
    ratio = var_emp / var_expected
    # MC noise at n=10000 is ~1.4%; allow ±10% per mode, expect ≥90% to land.
    in_range = ((ratio > 0.90) & (ratio < 1.10)).float().mean()
    assert in_range.item() >= 0.90, (
        f"too few modes within ±10% of analytic var: {in_range.item():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# sample_stationary covariance
# ─────────────────────────────────────────────────────────────────────────────

def test_sample_stationary_covariance(prior):
    """Empirical covariance of N(0, Σ_prior) draws should match Σ_prior."""
    Q_H, Sigma_S, _, eig_t, _ = prior
    n = 20000
    g = torch.Generator(); g.manual_seed(42)
    samples = sample_stationary(n, eig_t, generator=g)
    flat = samples.reshape(n, H * N)
    cov_emp = (flat.T @ flat) / n
    cov_true = torch.as_tensor(np.kron(Q_H, Sigma_S), dtype=cov_emp.dtype)
    rel = (cov_emp - cov_true).norm() / cov_true.norm()
    assert rel.item() < 0.05, f"rel.Frob err = {rel.item():.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Autograd: forward_sample is differentiable wrt r_0
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_sample_differentiable(prior):
    _, _, _, eig_t, _ = prior
    r_0 = torch.randn(2, H, N, requires_grad=True)
    g = torch.Generator(); g.manual_seed(0)
    r_tau = forward_sample(r_0, tau=0.5, eig=eig_t, beta_min=0.01, beta_max=50.0,
                           generator=g)
    loss = r_tau.pow(2).sum()
    loss.backward()
    assert r_0.grad is not None
    assert r_0.grad.shape == r_0.shape
    assert torch.isfinite(r_0.grad).all()
