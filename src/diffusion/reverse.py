"""DDIM-style ancestral reverse sampler for the GAD diffusion SDE.

Forward SDE (matches src/diffusion/forward_fcps.py):
    dr = -c(t) Q r dt + sqrt(2 c(t)) dW
    closed-form marginal in eigenbasis: c_k(t) | c_k(0) ~ N(H_t·c_0, σ²_t)
        H_t[k]   = exp(-c̄(t)·θ_k)
        σ²_t[k]  = (1 - exp(-2 c̄(t) θ_k)) / θ_k

Sampling strategy: at each reverse step (t → t-δ), the denoiser predicts R̂_0,
then we sample R_{t-δ} from p(R_{t-δ} | R_t, R̂_0). For deterministic DDIM
(η=0) this is closed-form and unconditionally stable — no SDE discretization
error blowups even under the steep α=7 FCPS schedule.

Deterministic DDIM update (in eigenbasis, per coordinate):
    ε̂_k       = (C_t[k] - H_t[k]·Ĉ_0[k]) / sqrt(σ²_t[k])
    C_{t-δ}[k] = H_{t-δ}[k]·Ĉ_0[k] + sqrt(σ²_{t-δ}[k])·ε̂_k

Stochastic DDIM with mixing coefficient eta ∈ [0, 1] interpolates between
deterministic (eta=0) and DDPM ancestral noise levels (eta=1):
    σ̃²_k       = eta² · σ²_{t-δ}[k] · (1 - σ²_{t-δ}[k]·H_t[k]² / (σ²_t[k]·H_{t-δ}[k]²))
    drift_var  = σ²_{t-δ}[k] - σ̃²_k
    C_{t-δ}[k] = H_{t-δ}[k]·Ĉ_0[k] + sqrt(drift_var)·ε̂_k + sqrt(σ̃²_k)·ζ_k
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
) -> tuple[torch.Tensor, Optional[ReverseTrace]]:
    """DDIM-style reverse sampling from t=T down to t=0.

    Parameters
    ----------
    prior : KroneckerPrior
    schedule : FCPSSchedule
    denoiser : callable(R_t, t_batch, Y_hat) -> R̂_0
        Tweedie predictor (must output the same shape as R_t).
    Y_hat : (B, N, H) torch tensor
        Backbone forecast, passed to the denoiser as conditioning.
    n_steps : int
        Reverse-time discretization steps (uniform grid on [0, T]).
    eta : float in [0, 1]
        Stochasticity mixing. eta=0 is deterministic DDIM (default); eta=1 is
        DDPM-ancestral noise level.
    R_T : optional (B, N, H) torch tensor
        Initial sample at t=T. If None, draw from the stationary prior.
    return_trace : bool
        If True, return a ReverseTrace with per-step diagnostics.
    generator : torch.Generator | None
        RNG for noise (and prior sample if R_T is None).

    Returns
    -------
    R̂_0 : (B, N, H) torch tensor
        Final state at t=0 — exactly the denoiser's R̂_0 from the last step
        (since H_0 = 1, σ²_0 = 0).
    trace : ReverseTrace | None
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
