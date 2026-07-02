"""GAD's Euler–Maruyama reverse SDE sampler (forward in forward_fcps).

this is the ONLY reverse process — the same one GAD uses (exp_gasde_test_only_Em.py:
sample_reverse_sde_em_in_s). it integrates the reverse-time SDE with EM, adding fresh
noise at every step, so it IS the stochastic / ancestral (DDPM-equivalent) sampler;
GAD's GASDE has no separate DDIM or deterministic sampler. see reverse_sample_em.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .forward_fcps import FCPSSchedule
from .prior import KroneckerPrior


def _rotate_to_eig_torch(R: torch.Tensor, V_S: torch.Tensor, V_T: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ki,...ih,hj->...kj", V_S.T, R, V_T)


def _rotate_from_eig_torch(C: torch.Tensor, V_S: torch.Tensor, V_T: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ik,...kh,hj->...ij", V_S, C, V_T.T)


def _prior_sample_torch(
    shape_batch: tuple[int, ...],
    prior: KroneckerPrior,
    V_S: torch.Tensor,
    V_T: torch.Tensor,
    sigma2: torch.Tensor,
    generator: Optional[torch.Generator],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    z = torch.randn(
        shape_batch + (prior.N, prior.H),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    c = sigma2.sqrt() * z
    return _rotate_from_eig_torch(c, V_S, V_T)


@dataclass
class ReverseTrace:
    """Per-step trace of the reverse-sampling trajectory."""

    t: list[float]
    R_t_mean_norm: list[float]
    R_hat_0_mean_norm: list[float]


def min_em_steps(prior: KroneckerPrior, schedule: FCPSSchedule) -> int:
    """stability floor on the EM step count for this prior + schedule.

    explicit EM of the linear part dC/dt = -c(t)·θ·C is stable only while c·θ·|dt| ≲ 2.
    the worst mode is θ_max near t=T (c largest); on the s=c̄ grid |dt|≈(S/n)/c(T), so the
    condition reduces to θ_max·S / n ≲ 2, i.e. n ≳ θ_max·S. below this EM diverges to NaN
    (the stiff kronecker/raw priors hit this: raw metrla θ_max≈172 NaNs under ~1000 steps).
    returns a safe minimum (θ_max·S, floored at 200); callers use max(requested, this)."""
    theta_max = float(prior.theta.max())
    return max(200, int(math.ceil(theta_max * schedule.c_0)))


def reverse_sample_em(
    prior: KroneckerPrior,
    schedule: FCPSSchedule,
    denoiser: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    Y_hat: torch.Tensor,
    n_steps: int = 1000,
    *,
    sigma: float = 1.0,
    R_T: Optional[torch.Tensor] = None,
    return_trace: bool = False,
    generator: Optional[torch.Generator] = None,
    step_callback: Optional[Callable[[int, float, torch.Tensor], None]] = None,
) -> tuple[torch.Tensor, Optional[ReverseTrace]]:
    """GAD's Euler–Maruyama reverse SDE (exp_gasde_test_only_Em.py: sample_reverse_sde_em_in_s).

    steps on a grid UNIFORM IN THE HEAT s = c̄ (from S=c_0 down to 0), mapping each
    s back to the diffusion time t via the schedule — exactly GAD's grid (NOT uniform
    in t, and not the geometric-c̄ grid the DDIM sampler uses). at each node the
    denoiser predicts R̂₀, we form the score from it, and take one EM step, all
    per-eigenmode (identical algebra to GAD, generalised to the kronecker θ):

        score  = (H·Ĉ₀ - C_t) / σ²(c̄)                # ∇log p(x_t|x_0), Tweedie
        drift  = -c(t)·θ·C_t - 2·c(t)·σ²·score        # reverse drift (eigenbasis)
        C_next = C_t + drift·dt + sqrt(2 c(t) σ²)·sqrt(|dt|)·z,   dt < 0

    the denoiser is conditioned on t (GAD's convention; t = t_of_c̄(s)). WARNING: EM
    is only conditionally stable; on a stiff prior the step −c(t)·θ·dt can exceed 1
    near t=T and diverge — GAD uses ~5000 steps. use enough n_steps if it blows up.
    """
    device, dtype, B = Y_hat.device, Y_hat.dtype, Y_hat.shape[0]
    V_S = torch.as_tensor(prior.V_S, device=device, dtype=dtype)
    V_T = torch.as_tensor(prior.V_T, device=device, dtype=dtype)
    theta = torch.as_tensor(prior.theta, device=device, dtype=dtype)         # (N, H)
    sigma2_stat = prior.sigma2_scale / theta                                       # prior variance

    if R_T is None:
        R_t = _prior_sample_torch(
            (B,), prior, V_S, V_T, sigma2_stat,
            generator=generator, device=device, dtype=dtype,
        )
    else:
        R_t = R_T.to(device=device, dtype=dtype)

    # GAD's grid: uniform in the heat s = c̄ from S (=c_0) down to 0, each mapped to t.
    s_grid = np.linspace(schedule.c_0, 0.0, n_steps + 1)                     # uniform in s=c̄
    t_grid = schedule.t_of_c_bar(s_grid)                                     # (n_steps+1,)
    trace = ReverseTrace([], [], []) if return_trace else None

    for i in range(n_steps):
        c_bar = float(s_grid[i])                                            # heat s at this node
        t_cur = float(t_grid[i])
        dt = float(t_grid[i + 1] - t_grid[i])                              # < 0
        c_t = float(schedule.c(t_cur))                                       # rate c(t)
        H = torch.exp(-theta * c_bar)
        var = prior.sigma2_scale * (1.0 - torch.exp(-2.0 * theta * c_bar)) / theta
        var = torch.clamp(var, min=1e-20)

        t_batch = torch.full((B,), t_cur, device=device, dtype=dtype)
        R0_hat = denoiser(R_t, t_batch, Y_hat)                               # conditioned on t
        if step_callback is not None:
            step_callback(i, c_bar, R0_hat)

        C_t = _rotate_to_eig_torch(R_t, V_S, V_T)
        C0_hat = _rotate_to_eig_torch(R0_hat, V_S, V_T)
        score = (H * C0_hat - C_t) / var                                     # ∇log p
        drift_rev = -c_t * theta * C_t - 2.0 * c_t * prior.sigma2_scale * score
        C_next = C_t + drift_rev * dt
        if c_t > 0:
            z = torch.randn(C_t.shape, generator=generator, device=device, dtype=dtype)
            C_next = C_next + math.sqrt(2.0 * c_t * prior.sigma2_scale) * math.sqrt(abs(dt)) * z
        R_t = _rotate_from_eig_torch(C_next, V_S, V_T)

        if trace is not None:
            trace.t.append(c_bar)
            trace.R_t_mean_norm.append(float(R_t.norm(dim=(-1, -2)).mean()))
            trace.R_hat_0_mean_norm.append(float(R0_hat.norm(dim=(-1, -2)).mean()))

    return R_t, trace
