"""Sanity tests for the GAD forward FCPS sampler.

Four checks:
  1. At t = 0, R_t == R_0 exactly.
  2. At t = T with a large c_0, empirical cov of N samples matches the
     analytic Σ_T ⊗ Σ_S (toy 4×3 prior, 20k samples, Frobenius rel err < 5%).
  3. Eigenbasis sampling matches np.random.multivariate_normal on the joint
     covariance (toy prior, KS test per coordinate + variance match).
  4. Real-data sanity: load the metrla prior, draw 5k samples, check per-
     coordinate empirical variance matches (Σ_T)_hh · (Σ_S)_ii within 5%
     median rel err and < 15% on the 99th percentile.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.diffusion.forward_fcps import (
    FCPSSchedule,
    ULSchedule,
    sample_R_t_given_R0,
    sample_prior,
)
from src.diffusion.prior import KroneckerPrior, load_prior


def _toy_prior(N: int = 4, H: int = 3, seed: int = 0) -> KroneckerPrior:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((N, N))
    L = M @ M.T
    L = (L + L.T) / 2
    lam_S, V_S = np.linalg.eigh(L)
    lam_S = lam_S - lam_S.min()

    K = rng.standard_normal((H, H))
    Sigma_T = K @ K.T + 0.1 * np.eye(H)
    mu_T, V_T = np.linalg.eigh(Sigma_T)

    gamma = 0.5
    theta = (lam_S[:, None] + gamma) / mu_T[None, :]
    return KroneckerPrior(
        dataset="toy",
        gamma=gamma,
        V_S=V_S,
        lam_S=lam_S,
        V_T=V_T,
        mu_T=mu_T,
        theta=theta,
        delta_h=np.diag(Sigma_T),
        R_H=Sigma_T / np.sqrt(np.outer(np.diag(Sigma_T), np.diag(Sigma_T))),
    )


def test_at_t_zero_returns_r0_exactly():
    prior = _toy_prior()
    sched = FCPSSchedule()
    rng = np.random.default_rng(42)
    R0 = rng.standard_normal((5, prior.N, prior.H))
    Rt = sample_R_t_given_R0(R0, t=0.0, prior=prior, schedule=sched, rng=rng)
    assert np.allclose(Rt, R0, atol=1e-12), (
        f"R_t at t=0 must equal R_0; max abs diff = {np.abs(Rt - R0).max()}"
    )


def test_stationary_cov_matches_kron_at_t_T_toy():
    """With a large c_0, samples at t=T should be drawn from N(0, Σ_T ⊗ Σ_S)."""
    prior = _toy_prior()
    sched = FCPSSchedule(c_min=0.05, c_0=500.0, alpha=7.0, T=1.0)
    rng = np.random.default_rng(123)

    n = 20000
    R0 = np.zeros((n, prior.N, prior.H))
    Rt = sample_R_t_given_R0(R0, t=sched.T, prior=prior, schedule=sched, rng=rng)

    R_flat = Rt.transpose(0, 2, 1).reshape(n, prior.N * prior.H)
    cov_emp = np.cov(R_flat, rowvar=False)

    Sigma_S = prior.V_S @ np.diag(1.0 / (prior.lam_S + prior.gamma)) @ prior.V_S.T
    Sigma_T = prior.V_T @ np.diag(prior.mu_T) @ prior.V_T.T
    cov_true = np.kron(Sigma_T, Sigma_S)

    rel_err = np.linalg.norm(cov_emp - cov_true) / np.linalg.norm(cov_true)
    assert rel_err < 0.05, f"empirical-vs-true cov rel err {rel_err:.3f} ≥ 0.05"


def test_eigenbasis_sample_matches_direct_multivariate_normal():
    prior = _toy_prior(N=4, H=3, seed=7)
    rng_eig = np.random.default_rng(2024)
    n = 5000

    samples_eig = sample_prior((n,), prior=prior, rng=rng_eig)
    samples_eig_flat = samples_eig.transpose(0, 2, 1).reshape(n, prior.N * prior.H)

    Sigma_S = prior.V_S @ np.diag(1.0 / (prior.lam_S + prior.gamma)) @ prior.V_S.T
    Sigma_T = prior.V_T @ np.diag(prior.mu_T) @ prior.V_T.T
    Sigma_full = (np.kron(Sigma_T, Sigma_S) + np.kron(Sigma_T, Sigma_S).T) / 2

    rng_dir = np.random.default_rng(2025)
    samples_dir = rng_dir.multivariate_normal(
        np.zeros(prior.N * prior.H), Sigma_full, size=n
    )

    for j in range(samples_eig_flat.shape[1]):
        _, p = stats.ks_2samp(samples_eig_flat[:, j], samples_dir[:, j])
        assert p > 0.001, f"coord {j}: KS p-value {p:.4g} ≤ 0.001"

    var_eig = samples_eig_flat.var(axis=0)
    var_dir = samples_dir.var(axis=0)
    rel = np.abs(var_eig - var_dir) / var_dir
    assert rel.max() < 0.10, f"max rel var diff {rel.max():.3f}"


@pytest.mark.parametrize("dataset", ["metrla"])
def test_real_data_per_coord_variance(dataset):
    """Per-coordinate empirical variance of stationary samples ≈ (Σ_T)_hh·(Σ_S)_ii."""
    prior = load_prior(dataset)
    rng = np.random.default_rng(0)
    n = 5000

    samples = sample_prior((n,), prior=prior, rng=rng)  # (n, N, H)

    # Analytic per-coord variance: diag(Σ_S)[i] · diag(Σ_T)[h].
    Sigma_S_diag = (prior.V_S ** 2 @ (1.0 / (prior.lam_S + prior.gamma)))  # (N,)
    Sigma_T_diag = (prior.V_T ** 2 @ prior.mu_T)                            # (H,)
    var_true = Sigma_S_diag[:, None] * Sigma_T_diag[None, :]               # (N, H)

    var_emp = samples.var(axis=0, ddof=1)                                   # (N, H)
    rel = np.abs(var_emp - var_true) / var_true

    median_rel = float(np.median(rel))
    p99 = float(np.quantile(rel, 0.99))
    assert median_rel < 0.05, f"median rel var err {median_rel:.3f} ≥ 0.05"
    assert p99 < 0.15, f"99th-pct rel var err {p99:.3f} ≥ 0.15"


def test_uls_schedule_basic_shape():
    """ULSchedule.c_bar(T) = slope/2 and c_bar(0) = 0."""
    sched = ULSchedule(slope=200.0, T=1.0)
    assert float(sched.c_bar(0.0)) == 0.0
    assert abs(float(sched.c_bar(1.0)) - 100.0) < 1e-12
    assert abs(float(sched.c(0.5)) - 100.0) < 1e-12
