"""Unit tests for src/eval/probabilistic.py — CRPS, PIT, pinball."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.probabilistic import crps_ensemble, pinball_loss, pit_values


# ─────────────────────────────────────────────────────────────────────────────
# CRPS
# ─────────────────────────────────────────────────────────────────────────────

def test_crps_zero_when_all_samples_equal_truth():
    """If every sample is exactly y, CRPS = 0."""
    y = np.array([3.0, -1.0, 0.5])
    samples = np.tile(y[:, None], (1, 10))                  # (3, 10)
    crps = crps_ensemble(y, samples)
    assert crps.shape == y.shape
    np.testing.assert_allclose(crps, 0.0, atol=1e-12)


def test_crps_positive_when_samples_far_from_truth():
    y = np.array([0.0])
    samples = np.full((1, 10), 10.0)
    crps = crps_ensemble(y, samples)
    # |x_i - y| = 10 for all i; (1/(2K²))ΣΣ|x_i - x_j| = 0  → CRPS = 10
    np.testing.assert_allclose(crps, 10.0, atol=1e-12)


def test_crps_handles_high_dim_shapes():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((4, 6, 7))
    samples = rng.standard_normal((4, 6, 7, 20))
    crps = crps_ensemble(y, samples)
    assert crps.shape == (4, 6, 7)
    assert np.isfinite(crps).all()
    assert (crps >= 0).all()


def test_crps_matches_gaussian_analytic():
    """For an N(0,1) ensemble at y=0, empirical CRPS ≈ analytic value.

    CRPS(N(0,1), 0) = √(2/π) - 1/√π ≈ 0.2338
    (Gneiting & Raftery 2007, eq. (5) at z=0.)
    """
    y = np.array([0.0])
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((1, 10000))
    crps = crps_ensemble(y, samples)
    expected = np.sqrt(2.0 / np.pi) - 1.0 / np.sqrt(np.pi)
    np.testing.assert_allclose(crps[0], expected, rtol=0.05)


def test_crps_shape_mismatch_raises():
    y = np.zeros((3,))
    samples = np.zeros((4, 5))  # mismatched leading dim
    with pytest.raises(ValueError, match="shape"):
        crps_ensemble(y, samples)


# ─────────────────────────────────────────────────────────────────────────────
# PIT
# ─────────────────────────────────────────────────────────────────────────────

def test_pit_zero_when_y_below_all_samples():
    y = np.array([-100.0])
    samples = np.random.default_rng(0).standard_normal((1, 50))
    pit = pit_values(y, samples)
    np.testing.assert_allclose(pit, 0.0)


def test_pit_one_when_y_above_all_samples():
    y = np.array([100.0])
    samples = np.random.default_rng(0).standard_normal((1, 50))
    pit = pit_values(y, samples)
    np.testing.assert_allclose(pit, 1.0)


def test_pit_calibrated_ensemble_gives_uniform_histogram():
    """If samples are i.i.d. from the same distribution as y, PIT is uniform."""
    rng = np.random.default_rng(0)
    n_windows, K = 5000, 50
    y = rng.standard_normal(n_windows)
    samples = rng.standard_normal((n_windows, K))
    pit = pit_values(y, samples)
    # Histogram should be roughly uniform; mean ≈ 0.5
    assert pit.shape == y.shape
    assert pit.mean() == pytest.approx(0.5, abs=0.02)
    # Standard deviation of uniform[0,1] is 1/√12 ≈ 0.289
    assert pit.std() == pytest.approx(1 / np.sqrt(12), abs=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# Pinball
# ─────────────────────────────────────────────────────────────────────────────

def test_pinball_zero_at_perfect_quantile():
    y = np.array([0.5])
    # Samples whose empirical 0.5-quantile (median) equals y exactly
    samples = np.array([[0.0, 1.0]])                       # median = 0.5
    pin = pinball_loss(y, samples, quantile=0.5)
    np.testing.assert_allclose(pin, 0.0, atol=1e-12)


def test_pinball_positive_otherwise():
    y = np.array([1.0])
    samples = np.zeros((1, 10))
    pin = pinball_loss(y, samples, quantile=0.5)
    # q_hat = 0, y - q_hat = 1, ρ = max(0.5, -0.5) = 0.5
    np.testing.assert_allclose(pin, 0.5)


def test_pinball_rejects_bad_quantile():
    y = np.array([0.0])
    samples = np.zeros((1, 10))
    for q in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="quantile"):
            pinball_loss(y, samples, quantile=q)


def test_pinball_handles_high_dim():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((3, 4))
    samples = rng.standard_normal((3, 4, 20))
    pin = pinball_loss(y, samples, quantile=0.9)
    assert pin.shape == (3, 4)
    assert np.isfinite(pin).all()
    assert (pin >= 0).all()
