"""Probabilistic forecast metrics: CRPS, PIT, pinball loss.

All functions take NumPy arrays and operate per (window, horizon, sensor).
Aggregation along the (window, sensor) axes is the caller's responsibility.

References:
  Gneiting & Raftery 2007. "Strictly Proper Scoring Rules, Prediction and
                            Estimation." J. Am. Statist. Assoc.
  Hersbach 2000. "Decomposition of the CRPS for Ensemble Prediction Systems."
                 Weather Forecasting.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CRPS
# ─────────────────────────────────────────────────────────────────────────────

def crps_ensemble(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Continuous Ranked Probability Score for an empirical ensemble.

    For K samples ``x_1, …, x_K`` and observation ``y``:

        CRPS = (1/K) Σ_i |x_i - y|  -  (1/2K²) Σ_{i,j} |x_i - x_j|

    The two terms balance sharpness (the second, smaller is sharper) against
    calibration (the first, smaller means samples concentrate near the truth).

    Args:
        y_true:  (...,)        observations, any shape S.
        samples: (..., K)      ensemble, last dim is the sample index.

    Returns:
        CRPS values of shape S (same as y_true).
    """
    if samples.shape[:-1] != y_true.shape:
        raise ValueError(
            f"samples shape {samples.shape} doesn't end with y_true shape "
            f"{y_true.shape}; samples must be (..., K)"
        )
    # |x_i - y|  → mean over K
    term1 = np.mean(np.abs(samples - y_true[..., None]), axis=-1)
    # |x_i - x_j| → mean over K² → multiply by 1/2.
    # Use the broadcast trick: |x_i - x_j| pairwise.
    diffs = np.abs(samples[..., :, None] - samples[..., None, :])           # (..., K, K)
    term2 = 0.5 * np.mean(diffs, axis=(-2, -1))
    return term1 - term2


# ─────────────────────────────────────────────────────────────────────────────
# PIT
# ─────────────────────────────────────────────────────────────────────────────

def pit_values(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Probability Integral Transform of ``y_true`` under the empirical
    CDF of ``samples``.

    PIT(y) = (1/K) |{i : x_i ≤ y}|

    For a well-calibrated forecast, PIT values are uniform on [0, 1].
    Deviations from uniformity diagnose miscalibration:
      - U-shaped histogram → forecast too narrow (under-dispersed)
      - dome-shaped histogram → forecast too wide (over-dispersed)
      - skewed → systematic bias

    Args:
        y_true:  (...,)        observations.
        samples: (..., K)      ensemble.

    Returns:
        PIT values in [0, 1] with the same shape as y_true.
    """
    if samples.shape[:-1] != y_true.shape:
        raise ValueError(
            f"samples shape {samples.shape} doesn't end with y_true shape "
            f"{y_true.shape}"
        )
    return np.mean(samples <= y_true[..., None], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Pinball loss
# ─────────────────────────────────────────────────────────────────────────────

def pinball_loss(y_true: np.ndarray, samples: np.ndarray, quantile: float) -> np.ndarray:
    """Quantile (pinball) loss at level ``q ∈ (0, 1)``.

        ρ_q(y, ŷ_q) = max(q·(y - ŷ_q), (1-q)·(ŷ_q - y))

    where ``ŷ_q`` is the empirical q-th quantile of ``samples``. Pinball is
    minimised in expectation by the true q-th quantile, so it provides a
    quantile-specific sharpness check (lower = better at that quantile).

    Args:
        y_true:   (...,)       observations.
        samples:  (..., K)     ensemble.
        quantile: scalar in (0, 1).

    Returns:
        Pinball values with the same shape as y_true.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if samples.shape[:-1] != y_true.shape:
        raise ValueError(
            f"samples shape {samples.shape} doesn't end with y_true shape "
            f"{y_true.shape}"
        )
    q_hat = np.quantile(samples, quantile, axis=-1)
    diff = y_true - q_hat
    return np.maximum(quantile * diff, (quantile - 1.0) * diff)
