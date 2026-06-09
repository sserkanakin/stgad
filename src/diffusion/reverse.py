"""ddim-style reverse sampler for the gad diffusion sde (forward in forward_fcps).

at each reverse step t -> t-δ the denoiser predicts R0, and we draw R_{t-δ} from
p(R_{t-δ} | R_t, R0) in closed form, per eigenmode:

    eps_hat  = (C_t - H_t C0_hat) / sqrt(σ²_t)
    C_{t-δ}  = H_{t-δ} C0_hat + sqrt(σ²_{t-δ}) eps_hat

with H_t = exp(-c̄(t) θ) and σ²_t = (1 - exp(-2 c̄(t) θ)) / θ. eta in [0, 1] mixes
in ancestral noise: eta=0 is deterministic ddim, eta=1 is the ddpm noise level.
"""
from __future__ import annotations

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


def _eigenbasis_HSigma(
    theta: torch.Tensor, c_bar_t: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-eigenmode (H_t, σ²_t) at the given c̄(t). Both shape (N, H)."""
    Ht = torch.exp(-theta * c_bar_t)
    sigma2_t = (1.0 - torch.exp(-2.0 * theta * c_bar_t)) / theta
    return Ht, sigma2_t


def reverse_sample(
    prior: KroneckerPrior,
    schedule: FCPSSchedule,
    denoiser: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    Y_hat: torch.Tensor,
    n_steps: int = 100,
    *,
    eta: float = 0.0,
    R_T: Optional[torch.Tensor] = None,
    return_trace: bool = False,
    generator: Optional[torch.Generator] = None,
    step_callback: Optional[Callable[[int, float, torch.Tensor], None]] = None,
) -> tuple[torch.Tensor, Optional[ReverseTrace]]:
    """reverse-sample from t=T down to t=0.

    denoiser is callable(R_t, t_batch, Y_hat) -> R0_hat. Y_hat (B, N, H) is the
    backbone forecast used as conditioning. n_steps is the number of uniform
    reverse steps. eta in [0, 1] mixes in ancestral noise (0 = deterministic).
    R_T is the starting sample at t=T (drawn from the prior if None).
    step_callback, if given, is called as step_callback(i, t_cur, R0_hat) each
    step with the denoiser's R0 estimate (used for trajectory diagnostics).
    returns (R0, trace), where trace is per-step diagnostics or None.
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta must lie in [0, 1]; got {eta}")

    device = Y_hat.device
    dtype = Y_hat.dtype
    B = Y_hat.shape[0]

    V_S = torch.as_tensor(prior.V_S, device=device, dtype=dtype)
    V_T = torch.as_tensor(prior.V_T, device=device, dtype=dtype)
    theta = torch.as_tensor(prior.theta, device=device, dtype=dtype)        # (N, H)
    sigma2_stat = 1.0 / theta

    if R_T is None:
        R_t = _prior_sample_torch(
            (B,), prior, V_S, V_T, sigma2_stat,
            generator=generator, device=device, dtype=dtype,
        )
    else:
        if R_T.shape != (B, prior.N, prior.H):
            raise ValueError(
                f"R_T shape {R_T.shape} must be ({B}, {prior.N}, {prior.H})"
            )
        R_t = R_T.to(device=device, dtype=dtype)

    ts = np.linspace(schedule.T, 0.0, n_steps + 1)

    trace = ReverseTrace([], [], []) if return_trace else None

    for i in range(n_steps):
        t_cur = float(ts[i])
        t_next = float(ts[i + 1])
        c_bar_cur = float(schedule.c_bar(t_cur))
        c_bar_next = float(schedule.c_bar(t_next))

        # Per-mode (H_t, σ²_t) at the current and next time.
        H_cur, sigma2_cur = _eigenbasis_HSigma(theta, c_bar_cur)
        H_next, sigma2_next = _eigenbasis_HSigma(theta, c_bar_next)

        # Denoiser predicts R̂_0 in physical space.
        t_batch = torch.full((B,), t_cur, device=device, dtype=dtype)
        R0_hat = denoiser(R_t, t_batch, Y_hat)                              # (B, N, H)

        if step_callback is not None:
            step_callback(i, t_cur, R0_hat)

        # Rotate to eigenbasis.
        C_t = _rotate_to_eig_torch(R_t, V_S, V_T)
        C0_hat = _rotate_to_eig_torch(R0_hat, V_S, V_T)

        # ε̂ from Tweedie: (C_t - H_t·Ĉ_0) / sqrt(σ²_t). Floor σ² to avoid div-by-zero
        # at very small c̄; this is a no-op at t_cur > 0 with the FCPS schedule.
        sigma_cur = torch.sqrt(torch.clamp(sigma2_cur, min=1e-20))
        eps_hat = (C_t - H_cur * C0_hat) / sigma_cur

        # DDIM update at t_next.
        # Stochastic σ̃² (eta-controlled noise level). DDIM eta=0 ⇒ deterministic.
        # Formula: σ̃² = η² · σ²_next · (1 - (σ²_next/σ²_cur)·(H_cur/H_next)²)
        # Guard: ratio_clip avoids NaN at last step where σ²_next=0.
        ratio = (sigma2_next / torch.clamp(sigma2_cur, min=1e-20)) * (
            H_cur / torch.clamp(H_next, min=1e-20)
        ) ** 2
        sigma_tilde2 = (eta ** 2) * sigma2_next * torch.clamp(1.0 - ratio, min=0.0)
        drift_var = torch.clamp(sigma2_next - sigma_tilde2, min=0.0)
        drift_std = torch.sqrt(drift_var)
        tilde_std = torch.sqrt(sigma_tilde2)

        C_next = H_next * C0_hat + drift_std * eps_hat
        if eta > 0:
            zeta = torch.randn(C_t.shape, generator=generator, device=device, dtype=dtype)
            C_next = C_next + tilde_std * zeta

        R_t = _rotate_from_eig_torch(C_next, V_S, V_T)

        if trace is not None:
            trace.t.append(t_cur)
            trace.R_t_mean_norm.append(float(R_t.norm(dim=(-1, -2)).mean()))
            trace.R_hat_0_mean_norm.append(float(R0_hat.norm(dim=(-1, -2)).mean()))

    return R_t, trace
