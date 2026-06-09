"""winkler interval score for central prediction intervals.

the winkler score for a central (1 - alpha) interval [lo, hi] and observation y:

    w = (hi - lo)                          if lo <= y <= hi
      = (hi - lo) + (2/alpha) * (lo - y)   if y < lo
      = (hi - lo) + (2/alpha) * (y - hi)   if y > hi

lower is better: it rewards narrow intervals but penalises a miss by 2/alpha
times how far outside the interval the observation lands. the score is
translation invariant, so scoring residuals == scoring forecasts (the backbone
term cancels: forecast_err = R0_hat - R0_true).
"""
from __future__ import annotations

import torch


def winkler_score(
    y: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, alpha: float
) -> torch.Tensor:
    """elementwise winkler score. y/lo/hi broadcastable, alpha in (0, 1)."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    width = hi - lo
    below = (lo - y).clamp(min=0.0)   # > 0 only when y < lo
    above = (y - hi).clamp(min=0.0)   # > 0 only when y > hi
    return width + (2.0 / alpha) * (below + above)


def masked_winkler_per_horizon(
    y: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """mean winkler per horizon over valid positions.

    y/lo/hi/mask are all (B, N, H); returns a (H,) tensor of per-horizon means.
    """
    w = winkler_score(y, lo, hi, alpha) * mask.float()
    num = w.sum(dim=(0, 1))                          # (H,)
    den = mask.float().sum(dim=(0, 1)).clamp(min=1.0)
    return num / den
