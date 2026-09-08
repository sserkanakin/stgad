"""reverse samplers for the GAD forward SDE (forward in forward_fcps).

reverse_sample_em        GAD's Euler-Maruyama reverse SDE (exp_gasde_test_only_Em.py:
                         sample_reverse_sde_em_in_s) — the GAD-parity headline sampler.
                         explicit Euler, so the stiff kronecker prior needs
                         n_steps >= theta_max * S for stability (min_em_steps).
reverse_sample_ancestral the DDPM-ancestral analogue for our LINEAR (per-mode OU)
                         SDE: each step samples the closed-form gaussian posterior
                         q(x_{c'} | x_c, x0_hat) per eigenmode — an exponential
                         integrator, EXACT in distribution given the denoiser, with
                         NO stability floor. ~50-100 steps replace thousands of EM
                         steps; the only approximation is the shared x0_hat within
                         a step (same as DDPM).
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


def reverse_sample_ancestral(
    prior: KroneckerPrior,
    schedule: FCPSSchedule,
    denoiser: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    Y_hat: torch.Tensor,
    n_steps: int = 100,
    *,
    c_lo: Optional[float] = None,
    corrected: bool = True,
    common_mode: str = "learned",
    R_T: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    step_callback: Optional[Callable[[int, float, torch.Tensor], None]] = None,
) -> tuple[torch.Tensor, None]:
    """DDPM-ancestral sampling, exact for the linear per-mode OU forward.

    common_mode: "learned" (default) takes x0_hat from the denoiser for every mode.
    "gauss" replaces the coefficient of the NETWORK-WIDE spatial mode (the V_S column
    closest to the constant vector, every horizon) by the prior's own posterior mean
    E[x0 | x_c] = σ_p²·(a_c/s_c)·x_c -- GAD's optimal graph filter (Prop. 1) for that one
    mode. the chain diagnostic showed the learned net's estimate of this mode is worse
    than the prior's (its ensemble-mean total is worse than predicting zero and its
    spread on the total is 0.58x); with the prior's filter for that mode the total is
    calibrated (1.00x). for the isotropic prior the mode carries 1/N of the variance and
    the swap changes nothing, so the rule is the same for both arms.

    grid: geometric in the heat c̄ from c_0 down to c_lo (log-spacing covers every
    mode's transition c̄·θ ~ O(1) evenly across the stiff θ spread). at each node
    the denoiser predicts R0_hat; the next state is a draw from the closed-form
    posterior q(x_{c'} | x_c, x0=R0_hat), per eigenmode:

        a_d = e^{-θ(c-c')}          s_d  = σ²(1-e^{-2θ(c-c')})/θ   (transition c'→c)
        a'  = e^{-θc'}              s'   = σ²(1-e^{-2θc'})/θ        (marginal at c')
        v   = s_d·s' / (a_d²·s' + s_d)
        mean= v·(a_d·x_c/s_d + a'·x0_hat/s')
        x_{c'} = mean + sqrt(var)·z

    var: with corrected=False, var = v -- the variance of q(x_{c'}|x_c, x0) as if x0
    were KNOWN. that is DDPM's β̃ (lower-bound) choice; it is benign for isotropic
    noise and wrong by up to 2x per step for the slow modes of a stiff prior, and the
    deficit compounds over the chain (the sampler then needs ~800 steps to reach what
    it should reach at 50). given x_c the posterior over x0 has variance
    σ_p² = 1/(a_c²/s_c + 1/v_stat), and propagating it through the mean gives the
    exact reverse transition N(a_d x_c, v_stat(1 − a_d²)) under the model:

        var = v + (v·a'/s')² · σ_p²                       (corrected=True, default)

    verified to 6 decimals against the closed form for θ ∈ [0.8, 5000].

    c_lo: terminal floor of the grid. None = schedule.c_lo, i.e. the SAME floor the
    denoiser was trained down to (a hardcoded 0.02 used to sit 25x above it). the
    final output is the denoiser's R0_hat there. returns (sample, None) to match
    reverse_sample_em's signature.
    """
    if c_lo is None:
        c_lo = float(schedule.c_lo)
    device, dtype, B = Y_hat.device, Y_hat.dtype, Y_hat.shape[0]
    V_S = torch.as_tensor(prior.V_S, device=device, dtype=dtype)
    V_T = torch.as_tensor(prior.V_T, device=device, dtype=dtype)
    theta = torch.as_tensor(prior.theta, device=device, dtype=dtype)        # (N, H)
    s2 = prior.sigma2_scale
    sigma2_stat = s2 / theta

    if R_T is None:
        R_t = _prior_sample_torch(
            (B,), prior, V_S, V_T, sigma2_stat,
            generator=generator, device=device, dtype=dtype,
        )
    else:
        R_t = R_T.to(device=device, dtype=dtype)

    c_grid = np.geomspace(schedule.c_0, c_lo, n_steps + 1)                  # (n_steps+1,)
    t_grid = schedule.t_of_c_bar(c_grid)
    C = _rotate_to_eig_torch(R_t, V_S, V_T)
    if common_mode not in ("learned", "gauss"):
        raise ValueError(f"common_mode must be 'learned' or 'gauss'; got {common_mode!r}")
    k_common = int(torch.argmax(V_S.sum(dim=0).abs())) if common_mode == "gauss" else -1   # column ∝ closest to 1/√N

    R0_hat = R_t
    for i in range(n_steps + 1):
        c_cur = float(c_grid[i])
        R_t = _rotate_from_eig_torch(C, V_S, V_T)
        t_batch = torch.full((B,), float(t_grid[i]), device=device, dtype=dtype)
        R0_hat = denoiser(R_t, t_batch, Y_hat)
        if k_common >= 0 and i == n_steps:                                  # same rule for the returned x0_hat
            a_c = torch.exp(-theta * c_cur)
            s_c = torch.clamp(s2 * (1.0 - torch.exp(-2.0 * theta * c_cur)) / theta, min=1e-20)
            C0_hat = _rotate_to_eig_torch(R0_hat, V_S, V_T).clone()
            C0_hat[:, k_common, :] = ((1.0 / (a_c * a_c / s_c + 1.0 / sigma2_stat)) * a_c / s_c * C)[:, k_common, :]
            R0_hat = _rotate_from_eig_torch(C0_hat, V_S, V_T)
        if step_callback is not None:
            step_callback(i, c_cur, R0_hat)
        if i == n_steps:                                                    # last node → return x0_hat
            break
        c_nxt = float(c_grid[i + 1])
        a_d = torch.exp(-theta * (c_cur - c_nxt))
        s_d = torch.clamp(s2 * (1.0 - torch.exp(-2.0 * theta * (c_cur - c_nxt))) / theta, min=1e-20)
        a_p = torch.exp(-theta * c_nxt)
        s_p = torch.clamp(s2 * (1.0 - torch.exp(-2.0 * theta * c_nxt)) / theta, min=1e-20)
        v = s_d * s_p / (a_d * a_d * s_p + s_d)
        C0_hat = _rotate_to_eig_torch(R0_hat, V_S, V_T)
        a_c = torch.exp(-theta * c_cur)
        s_c = torch.clamp(s2 * (1.0 - torch.exp(-2.0 * theta * c_cur)) / theta, min=1e-20)
        sig_p2 = 1.0 / (a_c * a_c / s_c + 1.0 / sigma2_stat)                # posterior var of x0 | x_c under the prior
        if k_common >= 0:
            C0_hat = C0_hat.clone()
            C0_hat[:, k_common, :] = (sig_p2 * a_c / s_c * C)[:, k_common, :]   # prior's E[x0 | x_c] for that mode
        mean = v * (a_d * C / s_d + a_p * C0_hat / s_p)
        var = v
        if corrected:
            # posterior variance of x0 given x_c under the prior, pushed through the mean
            var = v + (v * a_p / s_p) ** 2 * sig_p2
        z = torch.randn(C.shape, generator=generator, device=device, dtype=dtype)
        C = mean + var.sqrt() * z

    return R0_hat, None
