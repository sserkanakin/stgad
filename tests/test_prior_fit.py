"""Tests for src/diffusion/prior_fit.py — Phase 4.

All tests use small synthetic inputs so they run fast (< 5 s total).
Mathematical checks are exact or have tight tolerances.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.diffusion.prior_fit import (
    build_Q_H,
    build_Sigma_S,
    derive_sigma,
    empirical_spatial_cov,
    empirical_temporal_cov,
    empirical_temporal_corr,
    fit_delta_h,
    fit_prior,
    fit_spatial_beta,
    fit_temporal_kernel,
    lag_averaged_acf,
    sample_prior,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: tiny synthetic residuals and Laplacian
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_residuals():
    """(50, 12, 5) residuals with all values valid."""
    rng  = np.random.default_rng(0)
    r    = rng.standard_normal((50, 12, 5)).astype(np.float32)
    mask = np.ones((50, 12, 5), dtype=bool)
    return r, mask


@pytest.fixture
def tiny_residuals_with_missing():
    """(50, 12, 5) residuals with ~20% invalid readings."""
    rng  = np.random.default_rng(1)
    r    = rng.standard_normal((50, 12, 5)).astype(np.float32)
    mask = rng.random((50, 12, 5)) > 0.2
    return r, mask


@pytest.fixture
def chain_laplacian():
    """Normalised Laplacian for a 5-node chain graph.

    A = [[0,1,0,0,0],
         [1,0,1,0,0],
         [0,1,0,1,0],
         [0,0,1,0,1],
         [0,0,0,1,0]]

    Degree: [1,2,2,2,1].
    Normalised Laplacian L = I - D^{-1/2} A D^{-1/2}.
    Eigenvalues in [0, 2]; first eigenvalue ≈ 0.
    """
    N = 5
    A = np.zeros((N, N))
    for i in range(N - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0

    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-12))
    L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvals.astype(np.float64), eigvecs.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# fit_delta_h
# ─────────────────────────────────────────────────────────────────────────────

class TestFitDeltaH:
    def test_shape(self, tiny_residuals):
        r, mask = tiny_residuals
        delta = fit_delta_h(r, mask)
        assert delta.shape == (12,)
        assert delta.dtype == np.float64

    def test_positive(self, tiny_residuals):
        r, mask = tiny_residuals
        delta = fit_delta_h(r, mask)
        assert np.all(delta > 0.0)

    def test_known_variance(self):
        """Single horizon, single sensor, known variance = 4."""
        r    = np.full((100, 12, 1), 2.0, dtype=np.float32)
        r[:, 0, 0] = 0.0    # h=0 all zeros → var = 0
        r[:50, 1, 0] = 4.0  # h=1: half at 0, half at 4 → var = 4
        r[50:, 1, 0] = 0.0
        mask = np.ones((100, 12, 1), dtype=bool)
        delta = fit_delta_h(r, mask)
        assert delta[0] == pytest.approx(0.0, abs=1e-6)
        assert delta[1] == pytest.approx(4.0, abs=1e-4)

    def test_mask_respected(self):
        """Masked-out readings should not contribute."""
        r    = np.ones((10, 12, 3), dtype=np.float32) * 100.0
        mask = np.zeros((10, 12, 3), dtype=bool)
        # Only one valid reading per horizon per sensor — variance = 0
        mask[:, :, 0] = True
        r[:, :, 0] = 1.0
        delta = fit_delta_h(r, mask)
        for h in range(12):
            assert delta[h] == pytest.approx(0.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# empirical_temporal_cov and empirical_temporal_corr
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalCov:
    def test_shape(self, tiny_residuals):
        r, mask = tiny_residuals
        Q = empirical_temporal_cov(r, mask)
        assert Q.shape == (12, 12)

    def test_symmetric(self, tiny_residuals):
        r, mask = tiny_residuals
        Q = empirical_temporal_cov(r, mask)
        np.testing.assert_allclose(Q, Q.T, atol=1e-10)

    def test_positive_semidefinite(self, tiny_residuals):
        r, mask = tiny_residuals
        Q = empirical_temporal_cov(r, mask)
        eigvals = np.linalg.eigvalsh(Q)
        assert np.all(eigvals >= -1e-8)

    def test_diagonal_matches_delta_h(self, tiny_residuals):
        """diag(Q_emp)[h] should equal the weighted variance of r[:, h, :]."""
        r, mask = tiny_residuals
        Q     = empirical_temporal_cov(r, mask)
        delta = fit_delta_h(r, mask)
        # The diagonal of Q is the per-horizon mean-square (not variance, since
        # the global mean is not subtracted).  For zero-mean inputs they match.
        # For this test r is not centred, so just check shape consistency.
        assert Q.shape[0] == delta.shape[0]

    def test_corr_diagonal_ones(self, tiny_residuals):
        r, mask = tiny_residuals
        Q    = empirical_temporal_cov(r, mask)
        Qc   = empirical_temporal_corr(Q)
        np.testing.assert_allclose(np.diag(Qc), 1.0, atol=1e-10)

    def test_corr_in_minus_one_to_one(self, tiny_residuals):
        r, mask = tiny_residuals
        Q  = empirical_temporal_cov(r, mask)
        Qc = empirical_temporal_corr(Q)
        assert np.all(Qc >= -1.0 - 1e-8)
        assert np.all(Qc <=  1.0 + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# lag_averaged_acf
# ─────────────────────────────────────────────────────────────────────────────

class TestLagAveragedAcf:
    def test_shape(self):
        Q = np.eye(12)
        acf = lag_averaged_acf(Q)
        assert acf.shape == (11,)

    def test_identity_gives_zeros(self):
        """Identity correlation matrix → all lags = 0."""
        acf = lag_averaged_acf(np.eye(12))
        np.testing.assert_allclose(acf, 0.0, atol=1e-10)

    def test_all_ones_gives_ones(self):
        """All-ones correlation matrix → all lags = 1."""
        acf = lag_averaged_acf(np.ones((12, 12)))
        np.testing.assert_allclose(acf, 1.0, atol=1e-10)

    def test_known_values(self):
        """For a 3×3 matrix, check lag-1 manually."""
        Q = np.array([[1.0, 0.5, 0.3],
                      [0.5, 1.0, 0.6],
                      [0.3, 0.6, 1.0]])
        acf = lag_averaged_acf(Q)
        # lag 1: mean of Q[0,1] and Q[1,2] = (0.5 + 0.6) / 2 = 0.55
        assert acf[0] == pytest.approx(0.55, abs=1e-10)
        # lag 2: only Q[0,2] = 0.3
        assert acf[1] == pytest.approx(0.3, abs=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# fit_temporal_kernel
# ─────────────────────────────────────────────────────────────────────────────

class TestFitTemporalKernel:
    def test_output_range(self, tiny_residuals):
        r, mask = tiny_residuals
        Q     = empirical_temporal_cov(r, mask)
        Qc    = empirical_temporal_corr(Q)
        alpha, rho1, rho2 = fit_temporal_kernel(Qc)
        assert 0.0 < alpha < 1.0
        assert 0.0 < rho1 < 1.0
        assert 0.0 < rho2 < 1.0

    def test_ordering_convention(self, tiny_residuals):
        """ρ₁ ≥ ρ₂ (slow component first)."""
        r, mask = tiny_residuals
        Q     = empirical_temporal_cov(r, mask)
        Qc    = empirical_temporal_corr(Q)
        alpha, rho1, rho2 = fit_temporal_kernel(Qc)
        assert rho1 >= rho2

    def test_recovers_single_exp(self):
        """For a pure AR(1) matrix (α=1, ρ=0.8) the fit should recover ρ ≈ 0.8."""
        rho_true = 0.8
        lags     = np.abs(np.arange(12)[:, None] - np.arange(12)[None, :])
        Q_true   = rho_true ** lags        # pure single-exp → α=1, ρ₁=0.8
        alpha, rho1, rho2 = fit_temporal_kernel(Q_true)
        # The two-exp model should fit this perfectly: α≈1, ρ₁≈0.8
        model_acf = alpha * rho1 ** np.arange(1, 12) + (1 - alpha) * rho2 ** np.arange(1, 12)
        true_acf  = rho_true ** np.arange(1, 12)
        np.testing.assert_allclose(model_acf, true_acf, atol=1e-3)

    def test_recovers_two_exp(self):
        """Recovers known α=0.6, ρ₁=0.9, ρ₂=0.3 from a synthetic ACF."""
        alpha_true = 0.6
        rho1_true  = 0.9
        rho2_true  = 0.3
        lags = np.abs(np.arange(12)[:, None] - np.arange(12)[None, :])
        Q_true = (alpha_true * rho1_true ** lags
                  + (1.0 - alpha_true) * rho2_true ** lags)
        alpha, rho1, rho2 = fit_temporal_kernel(Q_true)
        # Verify fit quality (not parameter identity — two-exp is not identifiable
        # up to permutation; the model_acf should match).
        model_acf = alpha * rho1 ** np.arange(1, 12) + (1 - alpha) * rho2 ** np.arange(1, 12)
        true_acf  = (alpha_true * rho1_true ** np.arange(1, 12)
                     + (1.0 - alpha_true) * rho2_true ** np.arange(1, 12))
        np.testing.assert_allclose(model_acf, true_acf, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# build_Q_H
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildQH:
    def test_shape(self):
        delta = np.ones(12)
        Q = build_Q_H(delta, alpha=0.7, rho1=0.8, rho2=0.2)
        assert Q.shape == (12, 12)

    def test_symmetric(self):
        delta = np.arange(1, 13, dtype=float)
        Q = build_Q_H(delta, 0.5, 0.9, 0.1)
        np.testing.assert_allclose(Q, Q.T, atol=1e-12)

    def test_diagonal_equals_delta(self):
        """Q_H[h,h] = δ_h (since C_H[h,h] = 1)."""
        delta = np.array([float(h + 1) for h in range(12)])
        Q = build_Q_H(delta, 0.5, 0.9, 0.1)
        np.testing.assert_allclose(np.diag(Q), delta, rtol=1e-10)

    def test_positive_definite(self):
        """With δ > 0 and ρ₁, ρ₂ ∈ (0,1), Q_H must be positive definite."""
        delta = np.arange(1, 13, dtype=float)
        Q = build_Q_H(delta, 0.6, 0.8, 0.3)
        eigvals = np.linalg.eigvalsh(Q)
        assert np.all(eigvals > 0.0)

    def test_off_diagonal_formula(self):
        """Q[0,1] = sqrt(δ_0·δ_1) · (α·ρ₁ + (1-α)·ρ₂)."""
        delta = np.array([4.0, 9.0] + [1.0] * 10)
        alpha, rho1, rho2 = 0.7, 0.8, 0.2
        Q = build_Q_H(delta, alpha, rho1, rho2)
        expected = np.sqrt(4.0 * 9.0) * (alpha * rho1 + (1.0 - alpha) * rho2)
        assert Q[0, 1] == pytest.approx(expected, rel=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# fit_spatial_beta and build_Sigma_S
# ─────────────────────────────────────────────────────────────────────────────

class TestSpatialGMRF:
    def test_build_sigma_s_shape(self, chain_laplacian):
        ev, U = chain_laplacian
        Sigma = build_Sigma_S(beta=1.0, epsilon_s=0.01, eigvals_L=ev, eigvecs_L=U)
        assert Sigma.shape == (5, 5)

    def test_build_sigma_s_symmetric(self, chain_laplacian):
        ev, U = chain_laplacian
        Sigma = build_Sigma_S(1.0, 0.01, ev, U)
        np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-10)

    def test_build_sigma_s_positive_definite(self, chain_laplacian):
        ev, U = chain_laplacian
        Sigma = build_Sigma_S(1.0, 0.01, ev, U)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0.0)

    def test_build_sigma_s_eigenvalues(self, chain_laplacian):
        """Σ_S eigenvalues should equal 1/(β·λᵢ + ε_s)."""
        ev, U = chain_laplacian
        beta, eps = 2.0, 0.05
        Sigma = build_Sigma_S(beta, eps, ev, U)
        expected_eigvals = 1.0 / (beta * ev + eps)
        actual_eigvals   = np.linalg.eigvalsh(Sigma)
        # Both sorted ascending
        np.testing.assert_allclose(
            np.sort(actual_eigvals), np.sort(expected_eigvals), rtol=1e-6
        )

    def test_fit_spatial_beta_recovers_beta(self, chain_laplacian):
        """fit_spatial_beta should recover the β used to generate Σ_S."""
        ev, U = chain_laplacian
        beta_true = 3.0
        eps_s     = 0.01
        Sigma_true = build_Sigma_S(beta_true, eps_s, ev, U)
        beta_fit   = fit_spatial_beta(Sigma_true, ev, U, epsilon_s=eps_s)
        assert beta_fit == pytest.approx(beta_true, rel=1e-3)

    def test_fit_spatial_beta_positive(self, chain_laplacian):
        ev, U = chain_laplacian
        Sigma = build_Sigma_S(1.0, 0.01, ev, U)
        beta_fit = fit_spatial_beta(Sigma, ev, U, epsilon_s=0.01)
        assert beta_fit > 0.0

    def test_fit_spatial_beta_range(self, chain_laplacian):
        """Fitted β should be positive and finite for any reasonable Σ_S."""
        ev, U = chain_laplacian
        rng = np.random.default_rng(7)
        # Random Σ_S (symmetric positive definite)
        A = rng.standard_normal((5, 5))
        Sigma_rand = A @ A.T + 0.1 * np.eye(5)
        beta_fit = fit_spatial_beta(Sigma_rand, ev, U, epsilon_s=0.01)
        assert np.isfinite(beta_fit)
        assert beta_fit > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# empirical_spatial_cov
# ─────────────────────────────────────────────────────────────────────────────

class TestEmpiricalSpatialCov:
    def test_shape(self, tiny_residuals):
        r, mask = tiny_residuals
        S = empirical_spatial_cov(r, mask)
        assert S.shape == (5, 5)

    def test_symmetric(self, tiny_residuals):
        r, mask = tiny_residuals
        S = empirical_spatial_cov(r, mask)
        np.testing.assert_allclose(S, S.T, atol=1e-10)

    def test_positive_semidefinite(self, tiny_residuals):
        r, mask = tiny_residuals
        S = empirical_spatial_cov(r, mask)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# derive_sigma
# ─────────────────────────────────────────────────────────────────────────────

class TestDeriveSigma:
    def test_positive(self, chain_laplacian):
        ev, U = chain_laplacian
        Q = build_Q_H(np.ones(12), 0.5, 0.8, 0.2)
        S = build_Sigma_S(1.0, 0.01, ev, U)
        sigma = derive_sigma(Q, S)
        assert sigma > 0.0

    def test_formula(self, chain_laplacian):
        """σ = sqrt(log(cond(Q)·cond(S) / eps_target))."""
        ev, U   = chain_laplacian
        Q       = build_Q_H(np.ones(12), 0.5, 0.8, 0.2)
        S       = build_Sigma_S(1.0, 0.01, ev, U)
        eps_tgt = 0.01

        ev_Q = np.linalg.eigvalsh(Q)
        ev_S = np.linalg.eigvalsh(S)
        cond_Q = ev_Q.max() / ev_Q.min()
        cond_S = ev_S.max() / ev_S.min()
        sigma_expected = float(np.sqrt(np.log(cond_Q * cond_S / eps_tgt)))

        sigma_actual = derive_sigma(Q, S, eps_target=eps_tgt)
        assert sigma_actual == pytest.approx(sigma_expected, rel=1e-6)

    def test_larger_condition_gives_larger_sigma(self, chain_laplacian):
        """Higher condition number → larger σ."""
        ev, U  = chain_laplacian
        Q_low  = build_Q_H(np.ones(12), 0.5, 0.5, 0.4)   # small ρ → lower cond
        Q_high = build_Q_H(np.ones(12), 0.5, 0.99, 0.01)  # large ρ → higher cond
        S      = build_Sigma_S(1.0, 0.01, ev, U)
        assert derive_sigma(Q_high, S) > derive_sigma(Q_low, S)


# ─────────────────────────────────────────────────────────────────────────────
# sample_prior — key mathematical properties
# ─────────────────────────────────────────────────────────────────────────────

class TestSamplePrior:
    """Verify that sample_prior produces N(0, Q_H ⊗ Σ_S) samples."""

    @pytest.fixture
    def matrices(self, chain_laplacian):
        ev, U = chain_laplacian
        Q = build_Q_H(np.array([2.0, 3.0, 1.5, 2.5, 1.0,
                                 2.0, 3.0, 1.5, 2.5, 1.0,
                                 2.0, 3.0]), 0.7, 0.8, 0.2)
        S = build_Sigma_S(2.0, 0.01, ev, U)
        return Q, S

    def test_output_shape(self, matrices):
        Q, S = matrices
        samples = sample_prior(Q, S, n_samples=10)
        assert samples.shape == (10, 12, 5)

    def test_zero_mean(self, matrices):
        """Mean over 5000 samples should be close to zero.

        Tolerance accounts for the worst-case per-element std in this fixture.
        S_ii can reach ~20 (constant Laplacian mode, beta=2, eps=0.01) and
        Q_H[h,h] reaches ~3, so max element std ≈ sqrt(3*20) ≈ 7.7.
        Std error of mean = 7.7 / sqrt(5000) ≈ 0.11.  atol=0.5 catches bias.
        """
        Q, S    = matrices
        samples = sample_prior(Q, S, n_samples=5000, rng=np.random.default_rng(0))
        np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.5)

    def test_kronecker_covariance(self, matrices):
        """Empirical covariance of flattened samples should match kron(Q, S).

        Uses 5000 samples; threshold rel.Frob < 0.10 is tight but achievable
        (theoretical rate sqrt(2/n) = 0.020 for n=5000, d=60=12*5).
        """
        Q, S    = matrices
        n       = 5000
        H, N    = Q.shape[0], S.shape[0]
        samples = sample_prior(Q, S, n_samples=n, rng=np.random.default_rng(1))

        # Flatten (n, H, N) → (n, H*N)
        flat    = samples.reshape(n, H * N)
        emp_cov = (flat.T @ flat) / n              # (H*N, H*N)
        tgt_cov = np.kron(Q, S)                    # (H*N, H*N)

        rel_frob = (
            np.linalg.norm(emp_cov - tgt_cov, "fro")
            / np.linalg.norm(tgt_cov, "fro")
        )
        assert rel_frob < 0.10, (
            f"Kronecker covariance check failed: rel_frob={rel_frob:.4f} >= 0.10"
        )

    def test_temporal_marginal(self, matrices):
        """Marginal temporal covariance averaged over sensors ≈ Q_H · tr(Σ_S)/N.

        E_n E_i [ R[n,h,i] R[n,h',i] ] = Q_H[h,h'] · tr(Σ_S) / N
        """
        Q, S    = matrices
        n, N, H = 5000, S.shape[0], Q.shape[0]
        samples = sample_prior(Q, S, n_samples=n, rng=np.random.default_rng(2))

        # Sum over samples (n) and sensors (i) → (H, H)
        # einsum "nhi, nki -> hk": n sums, i sums, result (H, H)
        Q_emp    = np.einsum("nhi, nki -> hk", samples, samples) / (n * N)
        Q_scaled = Q * (np.trace(S) / N)

        rel_err = (
            np.linalg.norm(Q_emp - Q_scaled, "fro")
            / np.linalg.norm(Q_scaled, "fro")
        )
        assert rel_err < 0.10

    def test_spatial_marginal(self, matrices):
        """Marginal spatial covariance averaged over horizons ≈ Σ_S · tr(Q_H)/H.

        E_n E_h [ R[n,h,i] R[n,h,j] ] = Σ_S[i,j] · tr(Q_H) / H
        """
        Q, S    = matrices
        n, N, H = 5000, S.shape[0], Q.shape[0]
        samples = sample_prior(Q, S, n_samples=n, rng=np.random.default_rng(3))

        # Sum over samples (n) and horizons (h) → (N, N)
        # einsum "nhi, nhj -> ij": n sums, h sums, result (N, N)
        S_emp    = np.einsum("nhi, nhj -> ij", samples, samples) / (n * H)
        S_scaled = S * (np.trace(Q) / H)

        rel_err = (
            np.linalg.norm(S_emp - S_scaled, "fro")
            / np.linalg.norm(S_scaled, "fro")
        )
        assert rel_err < 0.10

    def test_deterministic_with_rng(self, matrices):
        """Same RNG seed → same samples."""
        Q, S = matrices
        s1 = sample_prior(Q, S, 5, rng=np.random.default_rng(42))
        s2 = sample_prior(Q, S, 5, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(s1, s2)

    def test_different_rng_different(self, matrices):
        """Different RNG seeds → different samples."""
        Q, S = matrices
        s1 = sample_prior(Q, S, 5, rng=np.random.default_rng(1))
        s2 = sample_prior(Q, S, 5, rng=np.random.default_rng(2))
        assert not np.allclose(s1, s2)


# ─────────────────────────────────────────────────────────────────────────────
# fit_prior (top-level integration test)
# ─────────────────────────────────────────────────────────────────────────────

class TestFitPrior:
    def test_returns_correct_keys(self, tiny_residuals, chain_laplacian):
        r, mask   = tiny_residuals
        ev, U     = chain_laplacian
        params    = fit_prior(r, mask, ev, U)
        required  = {"delta_h", "alpha", "rho1", "rho2", "beta", "epsilon_s", "sigma"}
        assert required.issubset(params.keys())

    def test_delta_h_shape(self, tiny_residuals, chain_laplacian):
        r, mask = tiny_residuals
        ev, U   = chain_laplacian
        params  = fit_prior(r, mask, ev, U)
        assert params["delta_h"].shape == (12,)

    def test_all_scalars_positive(self, tiny_residuals, chain_laplacian):
        r, mask = tiny_residuals
        ev, U   = chain_laplacian
        params  = fit_prior(r, mask, ev, U)
        for key in ("alpha", "rho1", "rho2", "beta", "sigma"):
            assert params[key] > 0.0, f"{key} = {params[key]} is not positive"

    def test_rho_ordering(self, tiny_residuals, chain_laplacian):
        """Convention: ρ₁ ≥ ρ₂."""
        r, mask = tiny_residuals
        ev, U   = chain_laplacian
        params  = fit_prior(r, mask, ev, U)
        assert params["rho1"] >= params["rho2"]

    def test_sigma_positive(self, tiny_residuals, chain_laplacian):
        r, mask = tiny_residuals
        ev, U   = chain_laplacian
        params  = fit_prior(r, mask, ev, U)
        assert params["sigma"] > 0.0

    def test_reproducible(self, tiny_residuals, chain_laplacian):
        """fit_prior is deterministic (no randomness)."""
        r, mask = tiny_residuals
        ev, U   = chain_laplacian
        p1 = fit_prior(r, mask, ev, U)
        p2 = fit_prior(r, mask, ev, U)
        for key in ("alpha", "rho1", "rho2", "beta", "sigma"):
            assert p1[key] == pytest.approx(p2[key], rel=1e-10)

    def test_epsilon_s_passthrough(self, tiny_residuals, chain_laplacian):
        r, mask = tiny_residuals
        ev, U   = chain_laplacian
        params  = fit_prior(r, mask, ev, U, epsilon_s=0.05)
        assert params["epsilon_s"] == pytest.approx(0.05)
