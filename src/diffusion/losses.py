"""training losses + the cross-prior comparison metric for the residual diffusion.

the denoiser predicts the clean residual R0. there are three ways to score the error
(R0_hat - R0), and they play different roles:

  mse           plain masked mse. the naive baseline — dominated by the high-noise
                samples (where the best possible error is ≈ the full signal variance
                and there is nothing to learn). NOT comparable across priors.
  snr           snr-weighted mse (GAD default, training loss). each sample is weighted
                by w(c̄) = 1/(1 + 2·v̄(c̄)) — ≈1 at low noise, →0 at high noise — so every
                noise level contributes comparably to the gradient. v̄ is the mean
                per-mode forward variance, the analogue of GAD's beta = 2σ²c(t).
  mahalanobis   precision-weighted: each mode's error divided by its noise variance
                σ²(c̄). puts every mode in standardised-error units, so the number is
                COMPARABLE across priors (kron vs iso). used as the val metric (and,
                optionally, as a training loss ablation).
"""
from __future__ import annotations

import torch

from .forward_fcps import TorchPriorTensors


def _per_mode_sigma2(c_bar: torch.Tensor, prior_t: TorchPriorTensors) -> torch.Tensor:
    """per-mode forward noise variance σ²(c̄) = (1 - e^{-2c̄θ})/θ. c_bar (B,) -> (B, N, H)."""
    cb = c_bar.view(-1, 1, 1)
    theta = prior_t.theta.unsqueeze(0)
    return (1.0 - torch.exp(-2.0 * theta * cb)) / theta


def masked_mse(R0_hat: torch.Tensor, R0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """plain masked mse over (B, N, H)."""
    m = mask.float()
    return ((R0_hat - R0).pow(2) * m).sum() / (m.sum() + 1e-8)


def snr_weight(c_bar: torch.Tensor, prior_t: TorchPriorTensors) -> torch.Tensor:
    """GAD-style snr weight per sample: w(c̄) = 1/(1 + 2·v̄), v̄ = mean per-mode noise
    variance. c_bar (B,) -> (B,), bounded in (0, 1]: ~1 at low noise, →0 at high noise."""
    v_bar = _per_mode_sigma2(c_bar, prior_t).mean(dim=(1, 2))    # (B,)
    return 1.0 / (1.0 + 2.0 * v_bar)


def snr_weighted_mse(
    R0_hat: torch.Tensor, R0: torch.Tensor, mask: torch.Tensor,
    c_bar: torch.Tensor, prior_t: TorchPriorTensors,
) -> torch.Tensor:
    """masked mse with the per-sample snr weight — the training default."""
    w = snr_weight(c_bar, prior_t).view(-1, 1, 1)               # (B, 1, 1)
    m = mask.float()
    return ((R0_hat - R0).pow(2) * m * w).sum() / ((m * w).sum() + 1e-8)


def mahalanobis_mse(
    R0_hat: torch.Tensor, R0: torch.Tensor, mask: torch.Tensor,
    c_bar: torch.Tensor, prior_t: TorchPriorTensors,
) -> torch.Tensor:
    """precision-weighted error in the eigenbasis: mean of (Ĉ0 − C0)² / σ²(c̄) per mode.

    standardises every mode by its own noise variance, so the value is comparable across
    priors. masked entries are zeroed before the rotation (the rotation mixes nodes, so a
    masked entry cannot be excluded per-mode); call on mostly-observed batches.
    """
    err = (R0_hat - R0) * mask.float()
    C = torch.einsum("ki,bih,hj->bkj", prior_t.V_S.T, err, prior_t.V_T)   # to eigenbasis
    sigma2 = _per_mode_sigma2(c_bar, prior_t).clamp(min=1e-12)
    return (C.pow(2) / sigma2).mean()


def compute_loss(
    name: str, R0_hat: torch.Tensor, R0: torch.Tensor, mask: torch.Tensor,
    c_bar: torch.Tensor, prior_t: TorchPriorTensors,
) -> torch.Tensor:
    """dispatch the training loss by name: 'snr' (default), 'mse', or 'mahalanobis'."""
    if name == "snr":
        return snr_weighted_mse(R0_hat, R0, mask, c_bar, prior_t)
    if name == "mse":
        return masked_mse(R0_hat, R0, mask)
    if name == "mahalanobis":
        return mahalanobis_mse(R0_hat, R0, mask, c_bar, prior_t)
    raise ValueError(f"unknown loss {name!r}; use 'snr', 'mse', or 'mahalanobis'")
