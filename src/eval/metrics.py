"""Probabilistic + point forecast metrics, masked and streaming.

All kernels score an *ensemble* forecast (K Monte-Carlo samples) against a
scalar truth, ignoring cells flagged invalid by ``mask``. The same code path is
used by the GPU diffusion pass (``scripts/eval/generate_samples.py``) and the
cheap CPU baselines (``scripts/eval/evaluate_forecasts.py``) so the numbers are
strictly comparable.

Conventions
-----------
samples : (K, B, N, H) ndarray — K ensemble members per (window, sensor, horizon)
truth   : (B, N, H)    ndarray — observed value
mask    : (B, N, H)    bool    — True where the observation is valid

A full split materialised at K samples is tens of GB, so we never hold it:
:class:`MetricAccumulator` consumes one batch at a time, keeps only running
sums / counts / histograms, and :meth:`MetricAccumulator.finalize` divides them
out at the end. ``update`` is called per batch by both eval scripts.

Metric definitions
-------------------
CRPS (per cell, marginal)
    Continuous Ranked Probability Score with the standard ensemble estimator
        CRPS = (1/K) Σ_k |x_k − y| − 1/(2K²) Σ_i Σ_j |x_i − x_j|
    computed via the O(K log K) sorted form. Reduces to |x−y| (MAE) for a
    point forecast and to the closed-form Gaussian CRPS as K → ∞.

Pinball / quantile loss (per cell)
    Mean over a grid of quantile levels of the asymmetric absolute loss; the
    mean over a fine grid equals CRPS/2 (used as a cross-check in tests).

Energy score (per window, multivariate)
    ES = (1/K) Σ_k ‖x_k − y‖ − 1/(2K²) Σ_i Σ_j ‖x_i − x_j‖   over valid cells.
    The multivariate generalisation of CRPS; sensitive to the joint spatial /
    temporal dependence the Kronecker prior is meant to capture. Reduces to
    CRPS in one dimension.

Variogram score of order p (per window, multivariate)
    VS_p = Σ_(i,j) ( |y_i − y_j|^p − (1/K) Σ_k |x_{k,i} − x_{k,j}|^p )²
    over a fixed random subset of cell pairs (full NH(NH−1)/2 pairs is
    infeasible). The subset size is logged, never silently capped.

Calibration
    PIT — the probability-integral-transform value of the truth under the
    ensemble CDF; a calibrated forecast gives a Uniform[0,1] PIT histogram.
    Reliability — empirical coverage of central prediction intervals at a grid
    of nominal levels; ACE is the mean |empirical − nominal| coverage error.

Sharpness
    Mean predictive standard deviation per horizon, and PINAW (the 90% interval
    width normalised by the data range) — narrower is sharper, only meaningful
    alongside coverage.

Point
    MAE / RMSE / MAPE of the ensemble mean, per horizon and overall.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import pdist

# Default quantile grid for pinball loss (also the CRPS cross-check grid).
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# Default nominal central-interval levels for the reliability curve.
DEFAULT_COVERAGE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# Number of PIT histogram bins (K-independent so runs with different K compare).
DEFAULT_PIT_BINS: int = 20
# Interval level used for the PINAW sharpness summary.
PINAW_LEVEL: float = 0.9


# ─── Per-cell kernels (operate on flattened valid cells) ──────────────────────


def crps_ensemble_1d(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-cell CRPS for an ensemble forecast.

    Parameters
    ----------
    samples : (K, M) ndarray — K members for each of M cells.
    truth   : (M,)   ndarray.

    Returns
    -------
    (M,) ndarray of CRPS values.

    Uses the sorted identity Σ_i Σ_j |x_i − x_j| = 2 Σ_i (2i − K − 1) x_(i)
    (1-indexed, ascending), so the whole thing is O(K log K) per cell.
    """
    K = samples.shape[0]
    term1 = np.abs(samples - truth[None, :]).mean(axis=0)              # (M,)
    xs = np.sort(samples, axis=0)                                      # ascending
    i = np.arange(1, K + 1)[:, None]                                   # (K, 1)
    # Σ_i Σ_j |x_i − x_j| = 2 Σ_i (2i − K − 1) x_(i)
    pair_sum = 2.0 * (((2 * i - K - 1) * xs).sum(axis=0))              # (M,)
    term2 = pair_sum / (2.0 * K * K)
    return term1 - term2


def pinball_loss_1d(
    samples: np.ndarray, truth: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Per-cell pinball (quantile) loss averaged over ``quantiles``.

    Returns (M,) ndarray; also returns nothing else — quantile estimates are the
    plain empirical quantiles of the ensemble.
    """
    qhat = np.quantile(samples, quantiles, axis=0)                     # (Q, M)
    diff = truth[None, :] - qhat                                       # (Q, M)
    q = quantiles[:, None]
    loss = np.maximum(q * diff, (q - 1.0) * diff)                      # (Q, M)
    return loss.mean(axis=0)


def pit_values_1d(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-cell PIT value (mid-rank), in [0, 1]. (M,) ndarray.

    u = ( #{x_k < y} + 0.5·#{x_k == y} ) / K. Calibrated ⇒ Uniform[0,1].
    """
    K = samples.shape[0]
    below = (samples < truth[None, :]).sum(axis=0)
    equal = (samples == truth[None, :]).sum(axis=0)
    return (below + 0.5 * equal) / K


# ─── Per-window multivariate kernels ──────────────────────────────────────────


def energy_score_window(samples: np.ndarray, truth: np.ndarray) -> float:
    """Energy score for a single window's valid cells.

    Parameters
    ----------
    samples : (K, d) ndarray — ensemble over the d valid cells of one window.
    truth   : (d,)   ndarray.
    """
    K = samples.shape[0]
    term1 = np.linalg.norm(samples - truth[None, :], axis=1).mean()
    if K > 1:
        # pdist returns the K(K-1)/2 pairwise distances; full double sum is
        # 2·sum(pdist) and the estimator divides by 2K².
        term2 = pdist(samples).sum() * 2.0 / (2.0 * K * K)
    else:
        term2 = 0.0
    return float(term1 - term2)


def variogram_score_window(
    samples: np.ndarray,
    truth: np.ndarray,
    pair_idx: tuple[np.ndarray, np.ndarray],
    p: float = 0.5,
) -> float:
    """Variogram score of order ``p`` over a fixed subset of cell pairs.

    Parameters
    ----------
    samples : (K, d) ndarray.
    truth   : (d,)   ndarray.
    pair_idx : (i_idx, j_idx) integer arrays selecting the pairs to score.
    """
    i_idx, j_idx = pair_idx
    truth_term = np.abs(truth[i_idx] - truth[j_idx]) ** p                       # (P,)
    samp_term = (np.abs(samples[:, i_idx] - samples[:, j_idx]) ** p).mean(0)    # (P,)
    return float(np.sum((truth_term - samp_term) ** 2))


def make_pair_index(d: int, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed random subset of distinct (i, j) cell pairs for the variogram score.

    If the full set of pairs is smaller than ``n_pairs`` we take all of them.
    """
    rng = np.random.default_rng(seed)
    total = d * (d - 1) // 2
    if total <= n_pairs:
        i_idx, j_idx = np.triu_indices(d, k=1)
        return i_idx, j_idx
    i_idx = rng.integers(0, d, size=n_pairs * 2)
    j_idx = rng.integers(0, d, size=n_pairs * 2)
    keep = i_idx != j_idx
    i_idx, j_idx = i_idx[keep][:n_pairs], j_idx[keep][:n_pairs]
    return i_idx, j_idx


# ─── Streaming accumulator ────────────────────────────────────────────────────


@dataclass
class MetricAccumulator:
    """Streaming accumulator for all forecast metrics.

    Call :meth:`update` once per batch of samples; call :meth:`finalize` at the
    end to get a nested dict of per-horizon and overall metrics. Holds only
    running sums / counts / histograms — never the samples themselves.

    Parameters
    ----------
    H : int — forecast horizon length.
    quantiles, coverage_levels : metric grids (defaults above).
    n_pit_bins : PIT histogram resolution.
    variogram_pairs, variogram_p, variogram_seed : variogram subsampling config.
    """

    H: int
    quantiles: np.ndarray = field(default_factory=lambda: np.asarray(DEFAULT_QUANTILES))
    coverage_levels: np.ndarray = field(
        default_factory=lambda: np.asarray(DEFAULT_COVERAGE_LEVELS)
    )
    n_pit_bins: int = DEFAULT_PIT_BINS
    variogram_pairs: int = 2000
    variogram_p: float = 0.5
    variogram_seed: int = 0

    def __post_init__(self) -> None:
        H, Q, L = self.H, len(self.quantiles), len(self.coverage_levels)
        # Per-horizon running sums + counts.
        self._crps_sum = np.zeros(H)
        self._pinball_sum = np.zeros(H)
        self._mae_sum = np.zeros(H)
        self._se_sum = np.zeros(H)
        self._ape_sum = np.zeros(H)
        self._std_sum = np.zeros(H)
        self._cell_count = np.zeros(H)        # valid cells per horizon
        self._ape_count = np.zeros(H)         # valid cells with truth ≠ 0
        # Coverage: covered count per (horizon, level) + interval-width sums.
        self._cov_count = np.zeros((H, L))
        self._width_sum = np.zeros((H, L))
        # PIT histogram (pooled over all cells).
        self._pit_hist = np.zeros(self.n_pit_bins)
        # Multivariate scores (per window).
        self._energy_sum = 0.0
        self._variogram_sum = 0.0
        self._window_count = 0
        # Truth range for PINAW.
        self._y_min = np.inf
        self._y_max = -np.inf
        # Lazily-built variogram pair index (depends on per-window valid d).
        self._pair_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ── batch update ──────────────────────────────────────────────────────────

    def update(self, samples: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
        """Accumulate one batch. samples (K,B,N,H); truth/mask (B,N,H)."""
        samples = np.asarray(samples, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        K, B, N, H = samples.shape
        if H != self.H:
            raise ValueError(f"horizon mismatch: accumulator H={self.H}, batch H={H}")

        self._update_cellwise(samples, truth, mask, K, H)
        self._update_multivariate(samples, truth, mask, B)

    def _update_cellwise(self, samples, truth, mask, K, H) -> None:
        qs = self.quantiles
        lo = (1.0 - self.coverage_levels) / 2.0
        hi = (1.0 + self.coverage_levels) / 2.0
        for h in range(H):
            m = mask[:, :, h].reshape(-1)                      # (B*N,)
            if not m.any():
                continue
            S = samples[:, :, :, h].reshape(K, -1)[:, m]       # (K, M)
            y = truth[:, :, h].reshape(-1)[m]                  # (M,)
            M = y.shape[0]

            self._crps_sum[h] += crps_ensemble_1d(S, y).sum()
            self._pinball_sum[h] += pinball_loss_1d(S, y, qs).sum()

            mean_pred = S.mean(axis=0)
            self._mae_sum[h] += np.abs(mean_pred - y).sum()
            self._se_sum[h] += ((mean_pred - y) ** 2).sum()
            nz = np.abs(y) > 1e-6
            self._ape_sum[h] += (np.abs(mean_pred[nz] - y[nz]) / np.abs(y[nz])).sum()
            self._ape_count[h] += int(nz.sum())
            self._std_sum[h] += S.std(axis=0).sum()
            self._cell_count[h] += M

            # Coverage + widths at each nominal level.
            q_lo = np.quantile(S, lo, axis=0)                  # (L, M)
            q_hi = np.quantile(S, hi, axis=0)                  # (L, M)
            covered = (y[None, :] >= q_lo) & (y[None, :] <= q_hi)
            self._cov_count[h] += covered.sum(axis=1)
            self._width_sum[h] += (q_hi - q_lo).sum(axis=1)

            # PIT histogram.
            pit = pit_values_1d(S, y)
            hist, _ = np.histogram(pit, bins=self.n_pit_bins, range=(0.0, 1.0))
            self._pit_hist += hist

            self._y_min = min(self._y_min, float(y.min()))
            self._y_max = max(self._y_max, float(y.max()))

    def _update_multivariate(self, samples, truth, mask, B) -> None:
        K = samples.shape[0]
        for b in range(B):
            mb = mask[b].reshape(-1)                            # (N*H,)
            if mb.sum() < 2:
                continue
            Sb = samples[:, b].reshape(K, -1)[:, mb]            # (K, d)
            yb = truth[b].reshape(-1)[mb]                       # (d,)
            d = yb.shape[0]
            self._energy_sum += energy_score_window(Sb, yb)
            if d not in self._pair_cache:
                self._pair_cache[d] = make_pair_index(
                    d, self.variogram_pairs, self.variogram_seed
                )
            self._variogram_sum += variogram_score_window(
                Sb, yb, self._pair_cache[d], p=self.variogram_p
            )
            self._window_count += 1

    # ── finalize ────────────────────────────────────────────────────────────--

    def finalize(self) -> dict:
        """Return a nested dict of finalized metrics (per-horizon + overall)."""
        c = self._cell_count
        c_safe = np.where(c > 0, c, 1.0)
        per_h_crps = self._crps_sum / c_safe
        per_h_pinball = self._pinball_sum / c_safe
        per_h_mae = self._mae_sum / c_safe
        per_h_rmse = np.sqrt(self._se_sum / c_safe)
        per_h_std = self._std_sum / c_safe
        ape_safe = np.where(self._ape_count > 0, self._ape_count, 1.0)
        per_h_mape = 100.0 * self._ape_sum / ape_safe

        total_cells = c.sum()
        overall = {
            "crps": float(self._crps_sum.sum() / max(total_cells, 1)),
            "pinball": float(self._pinball_sum.sum() / max(total_cells, 1)),
            "mae": float(self._mae_sum.sum() / max(total_cells, 1)),
            "rmse": float(np.sqrt(self._se_sum.sum() / max(total_cells, 1))),
            "mape": float(100.0 * self._ape_sum.sum() / max(self._ape_count.sum(), 1)),
            "pred_std": float(self._std_sum.sum() / max(total_cells, 1)),
        }

        # Reliability: empirical coverage per level (pooled over horizons).
        cov_total = c.sum()
        emp_cov = self._cov_count.sum(axis=0) / max(cov_total, 1)        # (L,)
        nominal = self.coverage_levels
        ace = float(np.abs(emp_cov - nominal).mean())
        # PINAW at PINAW_LEVEL.
        y_range = max(self._y_max - self._y_min, 1e-8)
        if PINAW_LEVEL in set(np.round(self.coverage_levels, 6)):
            li = int(np.argmin(np.abs(self.coverage_levels - PINAW_LEVEL)))
            pinaw = float(self._width_sum[:, li].sum() / max(cov_total, 1) / y_range)
        else:
            pinaw = float("nan")

        pit_hist = self._pit_hist / max(self._pit_hist.sum(), 1)
        # PIT calibration error: L1 deviation from uniform.
        pit_unif = 1.0 / self.n_pit_bins
        pit_cal_err = float(np.abs(pit_hist - pit_unif).sum())

        wc = max(self._window_count, 1)
        return {
            "overall": {
                **overall,
                "energy_score": float(self._energy_sum / wc),
                "variogram_score": float(self._variogram_sum / wc),
                "ace": ace,
                "pinaw90": pinaw,
                "pit_cal_err": pit_cal_err,
            },
            "per_horizon": {
                "crps": per_h_crps.tolist(),
                "pinball": per_h_pinball.tolist(),
                "mae": per_h_mae.tolist(),
                "rmse": per_h_rmse.tolist(),
                "mape": per_h_mape.tolist(),
                "pred_std": per_h_std.tolist(),
            },
            "reliability": {
                "nominal": nominal.tolist(),
                "empirical": emp_cov.tolist(),
                "mean_width": (self._width_sum.sum(axis=0) / max(cov_total, 1)).tolist(),
            },
            "pit": {
                "bins": self.n_pit_bins,
                "hist": pit_hist.tolist(),
            },
            "counts": {
                "cells_per_horizon": c.tolist(),
                "windows": int(self._window_count),
                "variogram_pairs": int(self.variogram_pairs),
            },
        }
