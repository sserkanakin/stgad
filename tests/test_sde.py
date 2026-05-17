"""Unit tests for src/diffusion/sde.py.

Small synthetic Q_H and Σ_S matrices (H=4, N=5) so we can compare against the
explicit Kronecker construction with `np.kron`. These tests catch sign errors,
basis-rotation inconsistency, and β̄ integral mistakes — the three failure modes
the user flagged for sde-sanity.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.diffusion.sde import (
    apply_Q,
    beta_bar,
    beta_schedule,
    eigendecompose_prior,
    forward_sample,
    from_spectral,
    g_squared,
    oracle_score,
    reverse_oracle,
    sample_stationary,
    to_spectral,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures: small synthetic prior we can fully construct by hand
# ─────────────────────────────────────────────────────────────────────────────

H, N = 4, 5


@pytest.fixture(scope="module")
def synthetic_prior():
    rng = np.random.default_rng(0)
    # Random PD temporal cov
    A = rng.standard_normal((H, H))
    Q_H = A @ A.T + 0.5 * np.eye(H)
    # Random PD spatial cov
    B = rng.standard_normal((N, N))
    Sigma_S = B @ B.T + 0.5 * np.eye(N)
    eig = eigendecompose_prior(Q_H, Sigma_S)
    return Q_H, Sigma_S, eig


# ─────────────────────────────────────────────────────────────────────────────
# Schedule
# ─────────────────────────────────────────────────────────────────────────────

def test_beta_schedule_endpoints():
    assert beta_schedule(0.0, 0.01, 10.0) == pytest.approx(0.01)
    assert beta_schedule(1.0, 0.01, 10.0) == pytest.approx(10.0)


def test_beta_bar_endpoints():
    assert beta_bar(0.0, 0.01, 10.0) == pytest.approx(0.0)
    # ∫β = β_min + ½(β_max−β_min) at τ=1
    expected = 0.01 + 0.5 * (10.0 - 0.01)
    assert beta_bar(1.0, 0.01, 10.0) == pytest.approx(expected)


def test_beta_bar_is_integral_of_schedule():
    # Numerically integrate β(τ) and compare with closed form
    taus = np.linspace(0, 1, 1001)
    integrand = beta_schedule(taus, 0.01, 10.0)
    approx = np.trapezoid(integrand, taus)
    exact = beta_bar(1.0, 0.01, 10.0)
    assert approx == pytest.approx(exact, rel=1e-4)


def test_g_squared_is_beta():
    for tau in [0.0, 0.3, 0.7, 1.0]:
        assert g_squared(tau, 0.01, 10.0) == pytest.approx(beta_schedule(tau, 0.01, 10.0))


# ─────────────────────────────────────────────────────────────────────────────
# Eigendecomposition + Kronecker structure
# ─────────────────────────────────────────────────────────────────────────────

def test_eigendecompose_recovers_original(synthetic_prior):
    Q_H, Sigma_S, eig = synthetic_prior
    Q_H_rebuilt = (eig["U_T"] * eig["c_T"]) @ eig["U_T"].T
    Sigma_S_rebuilt = (eig["U_S"] * eig["c_S"]) @ eig["U_S"].T
    assert np.allclose(Q_H_rebuilt, Q_H, atol=1e-10)
    assert np.allclose(Sigma_S_rebuilt, Sigma_S, atol=1e-10)


def test_joint_eigenvalues_are_pairwise_products(synthetic_prior):
    """Eigvals of A ⊗ B equal pairwise products of eigvals of A and B."""
    _, _, eig = synthetic_prior
    expected_full = np.outer(eig["c_T"], eig["c_S"]).ravel()
    assert np.allclose(np.sort(eig["s"].ravel()),
                       np.sort(expected_full),
                       atol=1e-10)


def test_q_is_inverse_of_s(synthetic_prior):
    _, _, eig = synthetic_prior
    assert np.allclose(eig["s"] * eig["q"], 1.0, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral basis transforms
# ─────────────────────────────────────────────────────────────────────────────

def test_spectral_round_trip(synthetic_prior):
    _, _, eig = synthetic_prior
    rng = np.random.default_rng(1)
    r = rng.standard_normal((7, H, N))
    r_back = from_spectral(to_spectral(r, eig["U_T"], eig["U_S"]),
                           eig["U_T"], eig["U_S"])
    assert np.allclose(r, r_back, atol=1e-10)


def test_spectral_round_trip_unbatched(synthetic_prior):
    """Single (H, N) sample, no batch dim."""
    _, _, eig = synthetic_prior
    r = np.random.default_rng(2).standard_normal((H, N))
    r_back = from_spectral(to_spectral(r, eig["U_T"], eig["U_S"]),
                           eig["U_T"], eig["U_S"])
    assert np.allclose(r, r_back, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# apply_Q
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_Q_matches_explicit_kron(synthetic_prior):
    """apply_Q(r) must equal the result of forming Q = Q_T ⊗ Q_S explicitly."""
    Q_H, Sigma_S, eig = synthetic_prior
    rng = np.random.default_rng(3)
    r = rng.standard_normal((H, N))

    via_module = apply_Q(r, eig)

    Q_T = np.linalg.inv(Q_H)             # temporal precision
    Q_S = np.linalg.inv(Sigma_S)          # spatial precision
    Q_full = np.kron(Q_T, Q_S)            # (HN, HN)
    via_kron = (Q_full @ r.ravel()).reshape(H, N)

    assert np.allclose(via_module, via_kron, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Forward marginal endpoints
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_at_tau_zero_is_identity(synthetic_prior):
    """At τ=0 the forward marginal collapses to a point mass at r_0."""
    _, _, eig = synthetic_prior
    rng = np.random.default_rng(4)
    r_0 = rng.standard_normal((8, H, N))
    r_tau = forward_sample(r_0, tau=0.0, eig=eig,
                           beta_min=0.01, beta_max=10.0, rng=rng)
    assert np.allclose(r_tau, r_0, atol=1e-10)


def test_forward_at_large_tau_loses_signal(synthetic_prior):
    """For β̄ ≫ max(s), forward samples should be near-i.i.d. from N(0, Σ_prior).

    Specifically: per-mode empirical variance in the joint eigenbasis should
    match s[i, j] (the prior eigenvalues), independent of the input r_0.
    """
    _, _, eig = synthetic_prior
    rng = np.random.default_rng(5)
    n = 20000
    r_0 = rng.standard_normal((n, H, N)) * 5.0     # large-magnitude r_0
    # β̄(1) target ≫ max(s) (s here is O(10), so 1e3 is more than enough)
    bb_target = 1e3
    beta_max_huge = 2 * bb_target - 0.01           # β̄(1) = 0.01 + 0.5(b-0.01)
    r_tau = forward_sample(r_0, tau=1.0, eig=eig,
                           beta_min=0.01, beta_max=beta_max_huge, rng=rng)
    r_tau_tilde = to_spectral(r_tau, eig["U_T"], eig["U_S"])  # (n, H, N)
    var_emp = (r_tau_tilde ** 2).mean(0)              # (H, N)
    # Per-mode match: empirical variance / prior variance should be ≈ 1
    ratio = var_emp / eig["s"]
    assert np.all(np.abs(ratio - 1.0) < 0.10), (
        f"per-mode variance ratio out of range: min={ratio.min():.3f}, "
        f"max={ratio.max():.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stationary sampler
# ─────────────────────────────────────────────────────────────────────────────

def test_stationary_sampler_matches_target_covariance(synthetic_prior):
    """Empirical covariance of stationary samples ≈ Σ_prior in vec form."""
    _, _, eig = synthetic_prior
    rng = np.random.default_rng(6)
    samples = sample_stationary(20000, eig, rng)             # (n, H, N)
    flat = samples.reshape(samples.shape[0], -1)              # (n, H*N)
    cov_emp = (flat.T @ flat) / samples.shape[0]
    # Rebuild Σ_prior = Q_H ⊗ Σ_S explicitly
    Q_H_rebuilt = (eig["U_T"] * eig["c_T"]) @ eig["U_T"].T
    Sigma_S_rebuilt = (eig["U_S"] * eig["c_S"]) @ eig["U_S"].T
    target = np.kron(Q_H_rebuilt, Sigma_S_rebuilt)
    rel_frob = np.linalg.norm(cov_emp - target, "fro") / np.linalg.norm(target, "fro")
    assert rel_frob < 0.10, f"rel_frob={rel_frob:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Oracle score
# ─────────────────────────────────────────────────────────────────────────────

def test_oracle_score_zero_when_r_tau_equals_mean(synthetic_prior):
    """If r_τ exactly equals μ_τ(r_0), the conditional score must be zero."""
    _, _, eig = synthetic_prior
    rng = np.random.default_rng(7)
    r_0 = rng.standard_normal((3, H, N))
    # Compute mean directly: μ = U exp(-½β̄·q) Uᵀ r_0  (in the joint eigenbasis)
    bb = beta_bar(0.4, 0.01, 10.0)
    decay = np.exp(-0.5 * bb / eig["s"])
    r_tau = from_spectral(decay * to_spectral(r_0, eig["U_T"], eig["U_S"]),
                          eig["U_T"], eig["U_S"])
    score = oracle_score(r_tau, r_0, tau=0.4, eig=eig, beta_min=0.01, beta_max=10.0)
    assert np.allclose(score, 0.0, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Reverse oracle round-trip on the toy prior
# ─────────────────────────────────────────────────────────────────────────────

def test_reverse_oracle_recovers_r0_on_toy_prior(synthetic_prior):
    """End-to-end: forward → reverse with oracle score should recover r_0."""
    _, _, eig = synthetic_prior
    rng = np.random.default_rng(8)
    r_0 = rng.standard_normal((4, H, N)) * 2.0
    r_tau = forward_sample(r_0, tau=1.0, eig=eig,
                           beta_min=0.01, beta_max=10.0, rng=rng)
    r_recovered = reverse_oracle(r_tau, r_0, n_steps=200, eig=eig,
                                 beta_min=0.01, beta_max=10.0, rng=rng,
                                 tau_eps=1e-3)
    rel = (np.linalg.norm((r_recovered - r_0).reshape(4, -1), axis=1)
           / np.linalg.norm(r_0.reshape(4, -1), axis=1))
    assert rel.mean() < 0.20, f"mean rel-recovery {rel.mean():.4f} > 0.20"
