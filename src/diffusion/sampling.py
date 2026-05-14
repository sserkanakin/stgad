"""Reverse-SDE sampler using a learned score net (exp-003 §14.5).

The training-time forward marginal is closed-form ``r_τ | r_0 ~ N(μ_τ, Σ_τ)``.
For sampling we run the reverse SDE (Anderson 1982):

    d r̄_τ = [-½ g² Q r̄_τ - g² ∇log p_τ(r̄_τ)] dτ + g(τ) d W̄_τ

using the network's ``r̂₀ = f_θ(r_τ, τ, c)`` plugged into Tweedie:

    ∇log p_τ(r_τ) ≈ -Σ_τ⁻¹ (r_τ - μ_τ(r̂₀))      with μ_τ(r̂₀) = e^{-½β̄Q} r̂₀

We discretise with Euler-Maruyama from τ_start = 1 down to τ_eps,
matching exp-004's oracle round-trip step layout.
"""
from __future__ import annotations

from typing import Optional

import torch

from src.diffusion.sde_torch import (
    EigTorch,
    _forward_decay_var,
    apply_Q,
    beta_bar,
    from_spectral,
    g_squared,
    sample_stationary,
    to_spectral,
)


# ─────────────────────────────────────────────────────────────────────────────
# One reverse Euler-Maruyama step using the learned score
# ─────────────────────────────────────────────────────────────────────────────

def _reverse_step(
    r_tau: torch.Tensor,
    r_hat_0: torch.Tensor,
    tau: float,
    delta_tau: float,
    eig: EigTorch,
    beta_min: float,
    beta_max: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """One Euler-Maruyama step at scalar τ using the network's r̂₀ prediction.

    drift = -½ g² Q r_τ + g² Σ_τ⁻¹ (r_τ - μ_τ(r̂₀))     (note the second-sign flip
    from the spec because score = -Σ_τ⁻¹ Δ, and drift contains "-g²·score")
    r_τ ← r_τ - drift·Δτ + g·√Δτ·ε
    """
    g2 = float(g_squared(tau, beta_min, beta_max))

    # μ_τ(r̂₀) in original basis via spectral decay
    decay, var = _forward_decay_var(tau, eig, beta_min, beta_max)         # (H, N) each
    r_hat_tilde = to_spectral(r_hat_0, eig["U_T"], eig["U_S"])
    mu_tau_tilde = decay * r_hat_tilde
    r_tau_tilde = to_spectral(r_tau, eig["U_T"], eig["U_S"])

    # score = -(r_τ - μ_τ) / var (eigenbasis), then rotate back
    score_tilde = -(r_tau_tilde - mu_tau_tilde) / var
    score = from_spectral(score_tilde, eig["U_T"], eig["U_S"])

    Q_r = apply_Q(r_tau, eig)
    drift = -0.5 * g2 * Q_r - g2 * score
    eps = torch.randn(r_tau.shape, dtype=r_tau.dtype, device=r_tau.device,
                      generator=generator)
    noise_std = (g2 * delta_tau) ** 0.5
    return r_tau - drift * delta_tau + noise_std * eps


# ─────────────────────────────────────────────────────────────────────────────
# Full reverse sampling with the learned score net
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reverse_learned(
    model: torch.nn.Module,
    r_tau_init: torch.Tensor,
    x_history: torch.Tensor,
    y_pred: torch.Tensor,
    n_steps: int,
    eig: EigTorch,
    beta_min: float,
    beta_max: float,
    tau_start: float = 1.0,
    tau_eps: float = 1e-3,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Reverse-time Euler-Maruyama from τ_start down to τ_eps with a learned r̂₀.

    Args:
        model:      Trained ScoreNet producing r̂₀ from (r_τ, τ, x_hist, y_pred).
        r_tau_init: (B, H, N) initial samples — typically `sample_stationary(...)`.
        x_history:  (B, 2, H, N) backbone-input conditioning.
        y_pred:     (B, H, N)    standardised backbone forecast.
        n_steps:    Number of Euler-Maruyama steps (uniform in τ).
        eig:        Eig dict on the same device as everything else.
        beta_min, beta_max: noise schedule.
        tau_start, tau_eps: bounds (defaults 1.0 → 1e-3).
        generator:  Optional torch.Generator for deterministic noise.

    Returns:
        r ≈ r̂₀ at τ ≈ τ_eps, shape (B, H, N).
    """
    model.eval()
    r = r_tau_init.clone()
    taus = torch.linspace(tau_start, tau_eps, n_steps + 1, dtype=r.dtype, device=r.device)
    for k in range(n_steps):
        tau_k = float(taus[k].item())
        delta_tau = float(taus[k].item() - taus[k + 1].item())

        # Network's r̂₀ prediction at this τ
        tau_batch = torch.full((r.shape[0],), tau_k, dtype=r.dtype, device=r.device)
        r_hat_0 = model(r, tau_batch, x_history, y_pred)

        r = _reverse_step(r, r_hat_0, tau_k, delta_tau, eig,
                          beta_min, beta_max, generator=generator)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Multi-sample wrapper: K trajectories per (window, conditioning) tuple
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_per_window(
    model: torch.nn.Module,
    x_history: torch.Tensor,                # (B, 2, H, N)
    y_pred: torch.Tensor,                    # (B, H, N)
    n_samples: int,
    n_steps: int,
    eig: EigTorch,
    beta_min: float,
    beta_max: float,
    tau_eps: float = 1e-3,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw ``n_samples`` independent reverse-SDE trajectories per input window.

    Stacks all (window × sample) trajectories into a single batch dim of size
    ``B * n_samples`` for one fused reverse-SDE pass through the network.
    Returns ``(B, n_samples, H, N)``.

    Memory note: the fused batch is B·K trajectories; for B=100, K=50, this is
    5,000 trajectories. With H=12, N=325 (PEMS-BAY) that's ~60 MB of r tensor
    state plus the network's activations. Comfortably fits on a 46 GB A40.
    Reduce B in the caller (caller iterates over chunks of windows) if memory
    is tight.
    """
    B, _, H, N = x_history.shape
    K = n_samples

    # Replicate conditioning K times along a new sample axis, then flatten.
    x_history_rep = x_history.unsqueeze(1).expand(B, K, 2, H, N).reshape(B * K, 2, H, N)
    y_pred_rep = y_pred.unsqueeze(1).expand(B, K, H, N).reshape(B * K, H, N)

    # Initial draw from the stationary distribution N(0, Σ_prior)
    r_init = sample_stationary(B * K, eig, generator=generator, dtype=y_pred.dtype)

    r_final = reverse_learned(
        model=model,
        r_tau_init=r_init,
        x_history=x_history_rep,
        y_pred=y_pred_rep,
        n_steps=n_steps,
        eig=eig,
        beta_min=beta_min,
        beta_max=beta_max,
        tau_eps=tau_eps,
        generator=generator,
    )

    return r_final.view(B, K, H, N)
