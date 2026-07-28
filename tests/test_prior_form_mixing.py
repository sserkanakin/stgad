"""Form-mixing regression test: the rotate-scale-rotate sampler draws N(0, σ²·Σ_T⊗Σ_S),
NOT any inversion-order variant.

The prior is applied as a two-sided rotation C→V_S C V_T^T of per-mode-scaled iso noise.
There are several plausible-but-wrong assemblies that look almost identical in code — the
classic being scaling by the temporal PRECISION Q_T=Σ_T^{-1} instead of the covariance Σ_T,
or transposing which factor rides which axis. Each is a covariance far from the intended one.
This test pins the correct assembly against a large-sample empirical covariance and asserts
one specific wrong assembly (Q_T ⊗ Σ_S) is clearly rejected, so the inversion-order bug can
never come back silently.

vec convention: h·N+i (temporal factor is the OUTER Kronecker block), matching prior.py's
np.kron(Σ_T, Σ_S) and kronecker_audit.separability.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.diffusion.prior import KroneckerPrior
from src.diffusion.forward_fcps import sample_prior

N, H = 6, 4
N_SAMPLES = 200_000
PASS_TOL = 0.05          # relative Frobenius; comfortably reachable at 2e5 samples
FAIL_FLOOR = 0.5         # the wrong assembly must miss by at least this much


def _spd(rng, n, spread=3.0):
    """random SPD with a decent condition number so wrong assemblies actually differ."""
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    eig = np.exp(rng.uniform(-np.log(spread), np.log(spread), n))    # eigenvalues in [1/spread, spread]
    return (Q * eig) @ Q.T


def _build_prior(rng):
    """a synthetic KroneckerPrior whose stationary law is exactly σ²·Σ_T⊗Σ_S.
    theta = 1/(s_S ⊗ μ_T) with sigma2_scale=1 makes prior.sigma2[k,h] = s_S[k]·μ_T[h]."""
    Sigma_S = _spd(rng, N)
    Sigma_T = _spd(rng, H)
    s_S, V_S = np.linalg.eigh(Sigma_S)
    mu_T, V_T = np.linalg.eigh(Sigma_T)
    theta = 1.0 / np.outer(s_S, mu_T)                                # (N,H); sigma2 = 1/theta = s_S⊗μ_T
    prior = KroneckerPrior(
        dataset="synthetic", gamma=0.8, V_S=V_S, lam_S=s_S, V_T=V_T, mu_T=mu_T,
        theta=theta, delta_h=np.diag(Sigma_T).copy(), R_H=np.eye(H), sigma2_scale=1.0,
    )
    # reconstruct the factors from the SAME eigenpairs the sampler uses (V·diag·Vᵀ)
    Sigma_S_r = (V_S * s_S) @ V_S.T
    Sigma_T_r = (V_T * mu_T) @ V_T.T
    return prior, Sigma_T_r, Sigma_S_r


def _empirical_cov(prior, rng):
    R = sample_prior((N_SAMPLES,), prior, rng)                       # (B, N, H)
    vecR = np.ascontiguousarray(R.transpose(0, 2, 1)).reshape(N_SAMPLES, H * N)   # index h*N+i
    return np.cov(vecR, rowvar=False)


def _rel_fro(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_sampler_draws_product_covariance():
    """empirical Cov(vec r) ≈ Σ_T ⊗ Σ_S to within PASS_TOL."""
    rng = np.random.default_rng(0)
    prior, Sigma_T, Sigma_S = _build_prior(rng)
    emp = _empirical_cov(prior, rng)
    target = np.kron(Sigma_T, Sigma_S)
    assert _rel_fro(emp, target) < PASS_TOL


def test_wrong_inversion_order_is_rejected():
    """the temporal-precision assembly Q_T ⊗ Σ_S is a different covariance and must
    miss the empirical second moment by a wide margin — the inversion-order guard."""
    rng = np.random.default_rng(0)
    prior, Sigma_T, Sigma_S = _build_prior(rng)
    emp = _empirical_cov(prior, rng)
    wrong = np.kron(np.linalg.inv(Sigma_T), Sigma_S)                 # Q_T instead of Σ_T
    correct = np.kron(Sigma_T, Sigma_S)
    assert _rel_fro(emp, wrong) > FAIL_FLOOR
    assert _rel_fro(emp, correct) < _rel_fro(emp, wrong)             # correct wins decisively


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
