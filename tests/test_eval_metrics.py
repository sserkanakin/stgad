"""Unit tests for the evaluation metric kernels.

Each test pins a kernel against a closed form or an independent implementation
(``properscoring``) so the streaming accumulator is trustworthy before any real
forecast is scored.
"""
from __future__ import annotations

import numpy as np
import properscoring as ps
import pytest
from scipy import stats

from src.eval.metrics import (
    MetricAccumulator,
    crps_ensemble_1d,
    energy_score_window,
    make_pair_index,
    pinball_loss_1d,
    pit_values_1d,
    variogram_score_window,
)


def gaussian_crps(mu: float, sigma: float, y: float) -> float:
    """Closed-form CRPS of N(mu, sigma²) at observation y (Gneiting & Raftery)."""
    w = (y - mu) / sigma
    return sigma * (w * (2 * stats.norm.cdf(w) - 1) + 2 * stats.norm.pdf(w) - 1 / np.sqrt(np.pi))


# ─── CRPS ─────────────────────────────────────────────────────────────────────


def test_crps_point_forecast_equals_mae():
    """An ensemble with zero spread (all members = c) has CRPS = |c − y|."""
    K, M = 50, 7
    c = 3.0
    samples = np.full((K, M), c)
    y = np.linspace(-2, 8, M)
    crps = crps_ensemble_1d(samples, y)
    np.testing.assert_allclose(crps, np.abs(c - y), atol=1e-12)


def test_crps_matches_gaussian_closed_form():
    """Ensemble CRPS → closed-form Gaussian CRPS as K grows."""
    rng = np.random.default_rng(0)
    mu, sigma = 2.0, 1.5
    K = 20000
    samples = rng.normal(mu, sigma, size=(K, 1))
    y = np.array([3.25])
    est = crps_ensemble_1d(samples, y)[0]
    exact = gaussian_crps(mu, sigma, y[0])
    assert abs(est - exact) < 0.02, f"est={est:.4f} exact={exact:.4f}"


def test_crps_matches_properscoring():
    """Cross-check against properscoring.crps_ensemble on random ensembles."""
    rng = np.random.default_rng(1)
    K, M = 200, 30
    samples = rng.normal(size=(K, M))
    y = rng.normal(size=M)
    ours = crps_ensemble_1d(samples, y)
    theirs = ps.crps_ensemble(y, samples.T)   # properscoring wants (M, K)
    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-8)


# ─── Pinball / CRPS relation ──────────────────────────────────────────────────


def test_pinball_grid_twice_equals_crps():
    """2 · mean pinball over a fine quantile grid ≈ CRPS."""
    rng = np.random.default_rng(2)
    K, M = 4000, 5
    samples = rng.normal(size=(K, M))
    y = rng.normal(size=M)
    qs = np.linspace(0.001, 0.999, 999)
    pin = pinball_loss_1d(samples, y, qs)
    crps = crps_ensemble_1d(samples, y)
    np.testing.assert_allclose(2.0 * pin, crps, rtol=0.02, atol=0.01)


# ─── Energy score ─────────────────────────────────────────────────────────────


def test_energy_reduces_to_crps_in_1d():
    """In one dimension the energy score equals CRPS."""
    rng = np.random.default_rng(3)
    K = 500
    samples = rng.normal(size=(K, 1))
    y = np.array([0.4])
    es = energy_score_window(samples, y)
    crps = crps_ensemble_1d(samples, y)[0]
    assert abs(es - crps) < 1e-9


def test_energy_and_variogram_nonnegative():
    rng = np.random.default_rng(4)
    K, d = 100, 12
    samples = rng.normal(size=(K, d))
    y = rng.normal(size=d)
    assert energy_score_window(samples, y) > 0
    pair_idx = make_pair_index(d, 50, seed=0)
    assert variogram_score_window(samples, y, pair_idx, p=0.5) >= 0


# ─── PIT / calibration ────────────────────────────────────────────────────────


def test_pit_uniform_when_calibrated():
    """Truth drawn from the same dist as the ensemble ⇒ PIT ≈ Uniform[0,1]."""
    rng = np.random.default_rng(5)
    K, M = 200, 20000
    samples = rng.normal(size=(K, M))
    y = rng.normal(size=M)
    pit = pit_values_1d(samples, y)
    # KS test against Uniform[0,1] should not reject.
    ks = stats.kstest(pit, "uniform").statistic
    assert ks < 0.02, f"PIT not uniform, KS={ks:.4f}"


def test_accumulator_coverage_near_nominal_when_calibrated():
    """A calibrated ensemble yields empirical coverage ≈ nominal (small ACE)."""
    rng = np.random.default_rng(6)
    H, B, N, K = 3, 40, 50, 200
    truth = rng.normal(size=(B, N, H))
    samples = rng.normal(size=(K, B, N, H))   # same dist, independent
    mask = np.ones((B, N, H), dtype=bool)
    acc = MetricAccumulator(H=H)
    acc.update(samples, truth, mask)
    out = acc.finalize()
    emp = np.array(out["reliability"]["empirical"])
    nom = np.array(out["reliability"]["nominal"])
    assert np.abs(emp - nom).mean() < 0.03
    assert out["overall"]["ace"] < 0.03


# ─── Masking & aggregation consistency ────────────────────────────────────────


def test_masked_cells_excluded():
    """Corrupting masked-out cells must not change any metric."""
    rng = np.random.default_rng(7)
    H, B, N, K = 2, 10, 8, 64
    truth = rng.normal(size=(B, N, H))
    samples = rng.normal(loc=truth[None], scale=1.0, size=(K, B, N, H))
    mask = rng.random((B, N, H)) > 0.3

    acc1 = MetricAccumulator(H=H)
    acc1.update(samples.copy(), truth.copy(), mask)
    out1 = acc1.finalize()

    truth2 = truth.copy()
    samples2 = samples.copy()
    truth2[~mask] = 1e6
    samples2[:, ~mask] = -1e6
    acc2 = MetricAccumulator(H=H)
    acc2.update(samples2, truth2, mask)
    out2 = acc2.finalize()

    assert out1["overall"]["crps"] == pytest.approx(out2["overall"]["crps"], rel=1e-9)
    assert out1["overall"]["mae"] == pytest.approx(out2["overall"]["mae"], rel=1e-9)


def test_per_horizon_reconciles_with_overall():
    """Cell-count-weighted mean of per-horizon CRPS equals overall CRPS."""
    rng = np.random.default_rng(8)
    H, B, N, K = 4, 30, 20, 80
    truth = rng.normal(size=(B, N, H))
    samples = rng.normal(loc=truth[None], size=(K, B, N, H))
    mask = np.ones((B, N, H), dtype=bool)
    acc = MetricAccumulator(H=H)
    acc.update(samples, truth, mask)
    out = acc.finalize()
    per_h = np.array(out["per_horizon"]["crps"])
    counts = np.array(out["counts"]["cells_per_horizon"])
    weighted = (per_h * counts).sum() / counts.sum()
    assert weighted == pytest.approx(out["overall"]["crps"], rel=1e-9)


def test_accumulator_streaming_matches_single_batch():
    """Splitting a batch across two update() calls gives identical results."""
    rng = np.random.default_rng(9)
    H, B, N, K = 3, 40, 10, 64
    truth = rng.normal(size=(B, N, H))
    samples = rng.normal(loc=truth[None], size=(K, B, N, H))
    mask = np.ones((B, N, H), dtype=bool)

    acc_full = MetricAccumulator(H=H)
    acc_full.update(samples, truth, mask)
    full = acc_full.finalize()

    acc_split = MetricAccumulator(H=H)
    acc_split.update(samples[:, :15], truth[:15], mask[:15])
    acc_split.update(samples[:, 15:], truth[15:], mask[15:])
    split = acc_split.finalize()

    assert full["overall"]["crps"] == pytest.approx(split["overall"]["crps"], rel=1e-9)
    assert full["overall"]["energy_score"] == pytest.approx(
        split["overall"]["energy_score"], rel=1e-9
    )
