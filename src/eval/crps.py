"""fair (energy-form) ensemble CRPS, shared by training-time selection and test eval."""
from __future__ import annotations

import torch


def empirical_crps(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """fair (unbiased) ensemble CRPS per point, in the same units as samples/y.

    CRPS = E|X - y| - 0.5 E|X - X'|, estimated from the K samples. the pairwise
    term uses the O(K log K) sorted identity ΣΣ|x_i-x_j| = 2 Σ_i (2i-K-1) x_(i),
    so we never form the KxK matrix, and is normalised by K(K-1) -- the FAIR
    estimator. the naive 1/K^2 form has expectation CRPS + E|X-X'|/(2K): a
    positive bias proportional to ensemble SPREAD, +25% at K=4, so under it the
    narrower of two equally skilled ensembles scores better. translation-equivariant,
    so CRPS on residuals equals CRPS on the reconstructed forecast.
    samples (K, *shape), y (*shape) -> (*shape); any trailing shape. needs K >= 2."""
    K = samples.shape[0]
    term1 = (samples - y.unsqueeze(0)).abs().mean(dim=0)            # E|X - y|
    xs, _ = torch.sort(samples, dim=0)                             # ascending
    idx = torch.arange(1, K + 1, device=samples.device, dtype=samples.dtype)
    w = (2 * idx - K - 1) / (K * (K - 1))                          # 0.5 E|X - X'|, fair
    # w broadcasts along the sample axis whatever the trailing shape is: the joint
    # metrics pass (K, B, H) network totals and (K, B, N) cumulative sums, not (K, B, N, H).
    # a fixed view(K, 1, 1, 1) against a 3-d tensor broadcast to (K, K, B, H) and summed
    # the weights to zero -> term2 vanished and the sum came out K times too large.
    term2 = (w.view(K, *([1] * (samples.ndim - 1))) * xs).sum(dim=0)
    return term1 - term2


# 19 quantiles 0.05..0.95 — the CSDI/DiffSTG/SpecSTG normalized-CRPS convention.
_QUANTILES = torch.linspace(0.05, 0.95, 19)


def quantile_crps_num(samples: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """numerator of the NORMALIZED quantile-CRPS (CSDI/DiffSTG), per horizon.

    CRPS_norm = (1/19) · Σ_q [ 2·Σ_pts |(q̂_q − y)·((y≤q̂_q) − q)| ] / Σ_pts |y|
    where q̂_q is the q-quantile of the K forecast samples. this fn returns the inner
    numerator Σ_q 2·Σ_{B,N} |…|·mask per horizon (H,); divide by (Σ|y|·19) for the score.
    samples (K, B, N, H), y (B, N, H), mask (B, N, H). this is THE crps papers report
    (~0.06–0.10), distinct from the raw-unit energy CRPS above."""
    qs = _QUANTILES.to(samples.device, samples.dtype)
    q_pred = torch.quantile(samples, qs, dim=0)                    # (19, B, N, H)
    yb = y.unsqueeze(0)
    pinball = ((q_pred - yb) * ((yb <= q_pred).to(samples.dtype) - qs.view(-1, 1, 1, 1))).abs()
    m = mask.to(samples.dtype).unsqueeze(0)
    return 2.0 * (pinball * m).sum(dim=(0, 1, 2))                  # (H,)
