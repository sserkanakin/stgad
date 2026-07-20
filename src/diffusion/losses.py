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
    """per-mode forward noise variance σ²(c̄) = σ²·(1 - e^{-2c̄θ})/θ. c_bar (B,) -> (B, N, H)."""
    cb = c_bar.view(-1, 1, 1)
    theta = prior_t.theta.unsqueeze(0)
    return prior_t.sigma2_scale * (1.0 - torch.exp(-2.0 * theta * cb)) / theta


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


def gad_snr_weighted_mse(
    R0_hat: torch.Tensor, R0: torch.Tensor, mask: torch.Tensor,
    c_t: torch.Tensor, sigma: float = 1.0,
) -> torch.Tensor:
    """GAD's exact training loss (exp_gasde.py): masked mse weighted per sample by
    w = 1/(1 + 2σ²·c(t)), where c(t) is the schedule RATE at the sampled noise level
    (not the cumulative c̄). σ=1 in GAD. add the TV term separately if enabled."""
    w = (1.0 / (1.0 + 2.0 * (sigma ** 2) * c_t)).view(-1, 1, 1)     # (B,1,1)
    m = mask.float()
    return ((R0_hat - R0).pow(2) * m * w).sum() / ((m * w).sum() + 1e-8)


def whitened_mse(
    R0_hat: torch.Tensor, R0: torch.Tensor, mask: torch.Tensor,
    prior_t: TorchPriorTensors, c_t: torch.Tensor, sigma: float = 1.0,
) -> torch.Tensor:
    """PriorGrad-style whitened loss: rotate the x0-error into the prior eigenbasis and
    weight each MODE by the prior precision θ (= 1/stationary-variance), times the GAD
    per-sample rate weight w = 1/(1+2σ²c(t)).

    plain mse lets the loud (high-variance) modes dominate, so a structured/reshaped prior
    — which puts the noise on exactly those loud modes — is strictly harder. weighting by θ
    down-weights the loud modes and up-weights the quiet ones so EVERY mode counts equally
    (standardised-error units). this is the loss that turns 'noise where the signal is' from
    a penalty into the mechanism that forces the model to capture the full covariance.

    masked entries are zeroed before the rotation (rotation mixes nodes); call on
    mostly-observed batches (PEMS flow mask is all-valid).

    NB the denominator carries the per-sample OBSERVED FRACTION. zeroing masked entries
    shrinks the numerator in proportion to how much data is missing, so a plain wt.sum()
    (which counts all B·N·H modes regardless) would scale the whole loss by that fraction
    — silently shrinking the effective LR on gappy datasets (metrla/pemsbay) while leaving
    the all-valid PEMS-flow arms untouched. that would be an invisible per-dataset
    asymmetry, not a fair comparison.
    """
    err = (R0_hat - R0) * mask.float()
    C = torch.einsum("ki,bih,hj->bkj", prior_t.V_S.T, err, prior_t.V_T)   # (B,N,H) eigenbasis
    theta = prior_t.theta.unsqueeze(0)                                     # (1,N,H) prior precision
    w = (1.0 / (1.0 + 2.0 * (sigma ** 2) * c_t)).view(-1, 1, 1)            # (B,1,1) per-sample
    wt = w * theta                                                         # (B,N,H) combined weight
    frac = mask.float().flatten(1).mean(1).view(-1, 1, 1)                  # (B,1,1) observed frac
    return (wt * C.pow(2)).sum() / ((wt * frac).sum() + 1e-8)


def graph_tv(R0_hat: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """graph total-variation penalty on the prediction (GAD's tv_pred): mean over
    edges of A_ij·(x_i - x_j)². A is (N, N); R0_hat is (B, N, H). NOTE: GAD applies
    this to smooth signals; residuals are rough, so this can be misspecified — off
    by default (tv_weight=0). enable to match GAD's 5e-3·tv_pred."""
    # x: (B, N, H) -> pairwise node diffs weighted by A, averaged over edges/batch/horizon
    diff = R0_hat.unsqueeze(2) - R0_hat.unsqueeze(1)               # (B, N, N, H)
    w = A.unsqueeze(0).unsqueeze(-1)                                # (1, N, N, 1)
    return (w * diff.pow(2)).sum() / (A.sum() * R0_hat.shape[0] * R0_hat.shape[2] + 1e-8)


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
