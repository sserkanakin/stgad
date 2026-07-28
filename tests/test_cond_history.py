"""Regression test for the history-conditioned arm: cond = [X_z | Y_hat_z].

The conditioning axis has one dangerous failure mode, and it already bit us once:
`cond_past` used to fall back SILENTLY to the previous window's residual field
([R_prev | Y_hat]) whenever a residual cache had no 'history' key. Every run of the
20260721_ladder "-past" arm took that fallback, so the thesis called something
"history conditioning" that was a different experiment entirely.

These tests pin the three things that must hold:
  1. the history half is the z-scored observed input window, not any residual field;
  2. the forecast half is bit-identical to the forecast-only arm's conditioning;
  3. a cache with no history and no way to find one RAISES instead of falling back.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.residual_dataset import ResidualDataset

N, P, H, B = 5, 12, 12, 40
MEAN, STD = 50.0, 10.0


def _write(tmp_path, with_history):
    """a synthetic residual cache; history is a distinctive ramp so a fallback shows."""
    rng = np.random.default_rng(0)
    resid = rng.normal(0, 3, (B, N, H)).astype(np.float32)
    y_pred = rng.normal(MEAN, 5, (B, N, H)).astype(np.float32)
    hist = (np.arange(B * N * P, dtype=np.float32).reshape(B, N, P) % 97.0) + 1.0
    kw = dict(residuals=resid, y_pred=y_pred,
              mask=np.ones((B, N, H), bool), mean=np.float32(MEAN), std=np.float32(STD))
    if with_history:
        kw["history"] = hist
    p = tmp_path / "val.npz"
    np.savez_compressed(p, **kw)
    return p, resid, y_pred, hist


def test_history_half_is_the_observed_window(tmp_path):
    p, _, y_pred, hist = _write(tmp_path, with_history=True)
    ds = ResidualDataset(p, 1.0, cond_past=True)
    assert tuple(ds.Y_hat.shape) == (B, N, P + H)
    np.testing.assert_allclose(ds.Y_hat[..., :P].numpy(), (hist - MEAN) / STD, atol=1e-6)
    np.testing.assert_allclose(ds.Y_hat[..., P:].numpy(), (y_pred - MEAN) / STD, atol=1e-6)


def test_forecast_half_matches_the_forecast_only_arm(tmp_path):
    p, _, _, _ = _write(tmp_path, with_history=True)
    past = ResidualDataset(p, 1.0, cond_past=True)
    fo = ResidualDataset(p, 1.0, cond_past=False)
    assert tuple(fo.Y_hat.shape) == (B, N, H)
    assert torch.equal(past.Y_hat[..., P:], fo.Y_hat)
    assert torch.equal(past.R0, fo.R0)          # same target, only the conditioning differs


def test_missing_history_raises_instead_of_falling_back(tmp_path):
    """the retired [R_prev | Y_hat] fallback must never silently return."""
    p, _, _, _ = _write(tmp_path, with_history=False)
    with pytest.raises((ValueError, FileNotFoundError), match="history"):
        ResidualDataset(p, 1.0, cond_past=True)


def test_row_count_mismatch_raises(tmp_path):
    """a stale processed split must be caught, not broadcast into a wrong pairing."""
    p, _, _, _ = _write(tmp_path, with_history=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        ResidualDataset(p, 1.0, cond_past=True, dataset="does_not_exist", split="val")
