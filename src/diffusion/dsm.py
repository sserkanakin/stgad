"""Denoising-score-matching loss for x₀-prediction (Tweedie target).

Loss form (per-batch element, eigenbasis weighting):

    L = mean_{ij}  (r̂₀ - r₀)²_{ij} / var_{ij}

where the subscript `ij` indexes the joint Q_H ⊗ Σ_S eigenbasis and

    var[i, j] = s[i, j] · (1 - exp(-β̄(τ) / s[i, j]))

is the forward marginal variance of mode (i, j) at time τ. This is the
natural-parameter / Gaussian-likelihood weighting: each mode contributes
in proportion to its precision at the current noise level.

Alternative `weighting="uniform"` mode falls back to plain MSE in the
original basis, useful as an ablation baseline.

τ is sampled log-uniformly in [τ_eps, 1] for training. Log-uniform
spaces samples evenly in log-τ, which gives more attention to the
small-τ regime where the network is doing the bulk of the denoising.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from src.diffusion.sde_torch import EigTorch, beta_bar, to_spectral


# ─────────────────────────────────────────────────────────────────────────────
# τ sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_tau(
    batch_size: int,
    tau_eps: float = 1e-3,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample τ ~ LogUniform[τ_eps, 1.0] of shape ``(batch_size,)``.

    Args:
        batch_size: Number of τ values to draw (one per batch element).
        tau_eps: Lower bound on τ. Must be > 0; values smaller than
            ``tau_eps`` make the per-mode variance vanish and the
            1/var weighting blow up.
        device, dtype, generator: standard torch flags.

    Returns:
        ``(batch_size,)`` tensor with τ values in ``[tau_eps, 1]``.
    """
    if tau_eps <= 0:
        raise ValueError(f"tau_eps must be > 0, got {tau_eps}")
    u = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    log_lo = math.log(tau_eps)
    log_hi = 0.0   # log(1.0)
    return torch.exp(log_lo + u * (log_hi - log_lo))


# ─────────────────────────────────────────────────────────────────────────────
# DSM x₀-prediction loss
# ─────────────────────────────────────────────────────────────────────────────

# Floor used to prevent division by zero at very small τ. Practical noise
# variance at τ_eps = 1e-3 with β_min = 0.01 is around 1e-5 per mode, well
# above this floor.
_VAR_FLOOR = 1e-12


def dsm_x0_loss(
    r_hat_0: torch.Tensor,
    r_0: torch.Tensor,
    tau: torch.Tensor,
    eig: EigTorch,
    beta_min: float,
    beta_max: float,
    weighting: str = "eigenbasis",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Tweedie x₀-prediction loss with optional eigenbasis weighting.

    Args:
        r_hat_0: ``(B, H, N)`` — network prediction of the clean residual.
        r_0:     ``(B, H, N)`` — ground truth (standardised residual).
        tau:     ``(B,)`` — diffusion time used for the forward sample.
                 Each batch element can have its own τ.
        eig:     Eig dict from ``eig_to_torch``. Must be on the same
                 device as ``r_hat_0``.
        beta_min, beta_max: Noise schedule.
        weighting: ``"eigenbasis"`` (per-mode 1/var, default) or
                   ``"uniform"`` (plain MSE in the original basis).
        mask: Optional ``(B, H, N)`` validity mask. When provided, the
              raw-space error is zeroed at masked positions before the
              orthogonal eigenbasis rotation (which preserves the sum of
              squares, so masked positions contribute zero energy). The
              loss is then normalised by ``mask.sum()`` instead of the
              full ``B·H·N``. Pass ``None`` to recover the original
              unmasked behaviour.

    Returns:
        Scalar loss (mean over unmasked modes, or full mean when mask is None).
    """
    if r_hat_0.shape != r_0.shape:
        raise ValueError(
            f"r_hat_0 shape {tuple(r_hat_0.shape)} != r_0 shape {tuple(r_0.shape)}"
        )
    if tau.shape != (r_0.shape[0],):
        raise ValueError(
            f"tau shape {tuple(tau.shape)} must be (B,) = ({r_0.shape[0]},)"
        )
    if mask is not None and mask.shape != r_0.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} != r_0 shape {tuple(r_0.shape)}"
        )

    err = r_hat_0 - r_0
    if mask is not None:
        # Zero error at masked positions in raw space. The eigenbasis rotation
        # below is orthogonal (Plancherel), so the zero energy stays zero — no
        # gradient contribution from masked positions.
        err = err * mask

    if weighting == "uniform":
        if mask is None:
            return err.pow(2).mean()
        return err.pow(2).sum() / mask.sum().clamp(min=1.0)

    if weighting != "eigenbasis":
        raise ValueError(
            f"unknown weighting {weighting!r}; choose 'eigenbasis' or 'uniform'"
        )

    # Project error into joint eigenbasis (orthogonal rotation; no scale change)
    err_tilde = to_spectral(err, eig["U_T"], eig["U_S"])             # (B, H, N)

    # Per-element variance at each (batch, eigenmode) location
    bb = beta_bar(tau, beta_min, beta_max).view(-1, 1, 1)             # (B, 1, 1)
    s = eig["s"].unsqueeze(0)                                         # (1, H, N)
    var = -s * torch.expm1(-bb / s)                                   # (B, H, N)
    var = var.clamp(min=_VAR_FLOOR)

    if mask is None:
        return (err_tilde.pow(2) / var).mean()
    return (err_tilde.pow(2) / var).sum() / mask.sum().clamp(min=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: per-mode weighted MSE in the *eigenbasis* for inspection
# ─────────────────────────────────────────────────────────────────────────────

def per_mode_squared_error(
    r_hat_0: torch.Tensor,
    r_0: torch.Tensor,
    eig: EigTorch,
) -> torch.Tensor:
    """Return ``(B, H, N)`` of per-mode squared errors in the eigenbasis.

    Useful for diagnostics: lets you inspect which joint eigenmodes the
    network is doing well or poorly on.
    """
    err = r_hat_0 - r_0
    err_tilde = to_spectral(err, eig["U_T"], eig["U_S"])
    return err_tilde.pow(2)
