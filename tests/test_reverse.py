"""Reverse-sampler smoke tests.

Validates the Euler-Maruyama reverse SDE implementation with an oracle
denoiser (returns the true R_0 verbatim). With a perfect denoiser the
sampler should reconstruct R_0 modulo simulation noise — the per-batch
mean should be close to R_0 and the residual variance should be small.
"""
from __future__ import annotations

import numpy as np
import torch

from src.diffusion.forward_fcps import FCPSSchedule
from src.diffusion.prior import KroneckerPrior
from src.diffusion.reverse import reverse_sample


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


def test_oracle_denoiser_recovers_R0():
    """With an oracle denoiser, reverse sampling should concentrate around the true R_0.

    Uses a mild schedule (small c_0, α=1.5) so vanilla EM is stable. Stress test
    against the production schedule lives in test_reverse_oracle_production.
    """
    prior = _toy_prior(N=4, H=3)
    schedule = FCPSSchedule(c_min=0.05, c_0=10.0, alpha=1.5, T=1.0)

    # Ground-truth R_0 — a fixed signal we want the reverse SDE to recover.
    torch.manual_seed(7)
    B = 64
    R_0 = torch.randn(1, prior.N, prior.H).expand(B, -1, -1).contiguous()
    Y_hat = torch.zeros_like(R_0)  # not used by the oracle, but must have right shape

    # Oracle denoiser returns the true R_0 regardless of (R_t, t, Y_hat).
    def oracle(R_t, t_batch, Y_hat_):
        return R_0.to(device=R_t.device, dtype=R_t.dtype)

    g = torch.Generator().manual_seed(2024)
    R_hat_0, _ = reverse_sample(
        prior=prior, schedule=schedule, denoiser=oracle,
        Y_hat=Y_hat, n_steps=300, generator=g, return_trace=False,
    )

    mean_recon = R_hat_0.mean(dim=0)
    err = (mean_recon - R_0[0]).abs()
    median_err = err.median().item()
    max_err = err.max().item()
    # With a perfect denoiser and 64 samples, recon should be tight.
    assert median_err < 0.2, f"median |mean(R̂_0) - R_0| = {median_err:.3f}"
    assert max_err < 0.6, f"max |mean(R̂_0) - R_0| = {max_err:.3f}"


def test_oracle_denoiser_production_schedule():
    """Oracle reconstruction with the production FCPS schedule (c_0=20000, α=7).

    The DDIM update is unconditionally stable, so this should pass cleanly even
    under the steep production schedule that breaks Euler-Maruyama.
    """
    prior = _toy_prior(N=4, H=3)
    schedule = FCPSSchedule(c_min=0.05, c_0=20000.0, alpha=7.0, T=1.0)

    torch.manual_seed(11)
    B = 64
    R_0 = torch.randn(1, prior.N, prior.H).expand(B, -1, -1).contiguous()
    Y_hat = torch.zeros_like(R_0)

    def oracle(R_t, t_batch, Y_hat_):
        return R_0.to(device=R_t.device, dtype=R_t.dtype)

    g = torch.Generator().manual_seed(2025)
    R_hat_0, _ = reverse_sample(
        prior=prior, schedule=schedule, denoiser=oracle,
        Y_hat=Y_hat, n_steps=300, generator=g, eta=0.0,
    )

    err = (R_hat_0 - R_0).abs()
    assert err.max().item() < 1e-3, (
        f"deterministic DDIM with oracle should reconstruct exactly; max err {err.max():.3g}"
    )


def test_reverse_with_R_T_init_shape():
    """Custom R_T initial state is accepted and shapes propagate."""
    prior = _toy_prior(N=4, H=3)
    schedule = FCPSSchedule(c_min=0.05, c_0=200.0, alpha=7.0, T=1.0)
    B = 3
    R_T = torch.randn(B, prior.N, prior.H)
    Y_hat = torch.zeros(B, prior.N, prior.H)
    R_hat_0, trace = reverse_sample(
        prior=prior, schedule=schedule,
        denoiser=lambda R_t, t, Y: torch.zeros_like(R_t),
        Y_hat=Y_hat, n_steps=20, R_T=R_T, return_trace=True,
    )
    assert R_hat_0.shape == (B, prior.N, prior.H)
    assert trace is not None and len(trace.t) == 20
