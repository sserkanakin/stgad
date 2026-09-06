"""the training loss for the residual diffusion.

the denoiser predicts the clean residual R0. the error (R0_hat - R0) is scored by a
masked mse weighted per sample by GAD's snr weight

    w(t) = 1 / (1 + 2 σ² c(t))

where c(t) is the schedule RATE at the sampled noise level (exp_gasde.py, σ = 1).
w ≈ 1 at low noise and → 0 at high noise, so every noise level contributes
comparably to the gradient instead of the high-noise samples — where the best
possible error is ≈ the full signal variance — dominating it.
"""
from __future__ import annotations

import torch


def snr_weighted_mse(
    R0_hat: torch.Tensor, R0: torch.Tensor, mask: torch.Tensor,
    c_t: torch.Tensor, sigma: float = 1.0,
) -> torch.Tensor:
    """masked mse, weighted per sample by w = 1/(1 + 2σ²·c(t)). shapes (B, N, H); c_t (B,)."""
    w = (1.0 / (1.0 + 2.0 * (sigma ** 2) * c_t)).view(-1, 1, 1)     # (B,1,1)
    m = mask.float()
    return ((R0_hat - R0).pow(2) * m * w).sum() / ((m * w).sum() + 1e-8)
