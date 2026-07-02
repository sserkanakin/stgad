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
from src.diffusion.reverse import reverse_sample_em


def _toy_prior(N: int = 4, H: int = 3, seed: int = 0, gamma: float = 0.5,
               gad_scale: bool = False) -> KroneckerPrior:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((N, N))
    L = M @ M.T
    L = (L + L.T) / 2
    lam_S, V_S = np.linalg.eigh(L)
    lam_S = lam_S - lam_S.min()

    K = rng.standard_normal((H, H))
    Sigma_T = K @ K.T + 0.1 * np.eye(H)
    mu_T, V_T = np.linalg.eigh(Sigma_T)

    theta = (lam_S[:, None] + gamma) / mu_T[None, :]
    sigma2_scale = 1.0
    if gad_scale:                    # mirror load_prior: rescale so θ_min = γ (GAD scale)
        s = gamma / float(theta.min())
        theta, sigma2_scale = theta * s, float(s)
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
        sigma2_scale=sigma2_scale,
    )


def test_em_oracle_recovers_R0_gad_schedule():
    """GAD's Euler–Maruyama reverse (uniform-in-s grid) with an oracle denoiser must
    concentrate around the true R_0 on the GAD-literal schedule (S=7, α=4, c_min=0.1)
    and the GAD-scale prior (θ_min=γ=0.8). This is the reported/thesis sampler."""
    prior = _toy_prior(N=4, H=3, gamma=0.8, gad_scale=True)
    schedule = FCPSSchedule(c_min=0.1, c_0=7.0, alpha=4.0, T=1.0)

    torch.manual_seed(7)
    B = 64
    R_0 = torch.randn(1, prior.N, prior.H).expand(B, -1, -1).contiguous()
    Y_hat = torch.zeros_like(R_0)

    def oracle(R_t, t_batch, Y_hat_):
        return R_0.to(device=R_t.device, dtype=R_t.dtype)

    g = torch.Generator().manual_seed(2024)
    R_hat_0, _ = reverse_sample_em(
        prior=prior, schedule=schedule, denoiser=oracle,
        Y_hat=Y_hat, n_steps=1000, generator=g,
    )
    err = (R_hat_0.mean(dim=0) - R_0[0]).abs()
    assert err.median().item() < 0.1, f"median |mean(R̂_0) - R_0| = {err.median():.3f}"
    assert err.max().item() < 0.3, f"max |mean(R̂_0) - R_0| = {err.max():.3f}"


def test_em_with_R_T_init_shape_and_trace():
    """EM accepts a custom R_T initial state; shapes propagate and the trace has one
    entry per step (per-node = the s=c̄ grid node)."""
    prior = _toy_prior(N=4, H=3, gamma=0.8, gad_scale=True)
    schedule = FCPSSchedule(c_min=0.1, c_0=7.0, alpha=4.0, T=1.0)
    B = 3
    R_T = torch.randn(B, prior.N, prior.H)
    Y_hat = torch.zeros(B, prior.N, prior.H)
    R_hat_0, trace = reverse_sample_em(
        prior=prior, schedule=schedule,
        denoiser=lambda R_t, t, Y: torch.zeros_like(R_t),
        Y_hat=Y_hat, n_steps=20, R_T=R_T, return_trace=True,
    )
    assert R_hat_0.shape == (B, prior.N, prior.H)
    assert torch.isfinite(R_hat_0).all()
    assert trace is not None and len(trace.t) == 20
