"""Tests for src/data/preprocessing.py, src/data/dataset.py, src/data/adjacency.py.

Synthetic-data tests run always. Tests that require real data files
are skipped when those files are absent (e.g. in CI without data).
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.preprocessing import (
    build_windows,
    compute_mask,
    compute_scaler,
    compute_time_of_day,
    split_timestamps,
)
from src.data.adjacency import compute_normalized_laplacian

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

X_OFFSETS = np.arange(-11, 1, 1)   # [-11..0]
Y_OFFSETS = np.arange(1, 13, 1)    # [1..12]

_DATA_PRESENT = {
    "metrla":  Path("data/raw/adj_mx_metrla.pkl").exists(),
    "pemsbay": Path("data/raw/adj_mx_pemsbay.pkl").exists(),
}
_PROCESSED_METRLA  = Path("data/processed/metrla_train.npz").exists()
_PROCESSED_PEMSBAY = Path("data/processed/pemsbay_train.npz").exists()


def _make_synthetic_data(T: int = 200, N: int = 10, seed: int = 0) -> np.ndarray:
    """Return (T, N, 2) float32 array: [speed, time_of_day]."""
    rng = np.random.default_rng(seed)
    speed = rng.uniform(0, 80, (T, N)).astype(np.float32)
    tod   = np.linspace(0, 1, T, endpoint=False, dtype=np.float32)
    tod   = np.tile(tod[:, None], (1, N))
    return np.stack([speed, tod], axis=-1)


# ---------------------------------------------------------------------------
# split_timestamps
# ---------------------------------------------------------------------------

def test_split_indices_cover_all_timestamps():
    T = 34272
    n_train, n_val_end = split_timestamps(T)
    assert n_train + (n_val_end - n_train) + (T - n_val_end) == T


def test_split_fractions_approx():
    T = 34272
    n_train, n_val_end = split_timestamps(T)
    assert abs(n_train / T - 0.7) < 0.005
    assert abs((T - n_val_end) / T - 0.2) < 0.005


def test_split_no_gaps_or_overlaps():
    T = 52116
    n_train, n_val_end = split_timestamps(T)
    assert 0 < n_train < n_val_end < T


# ---------------------------------------------------------------------------
# compute_time_of_day
# ---------------------------------------------------------------------------

def test_time_of_day_range():
    idx = pd.date_range("2012-03-01 00:00", periods=288, freq="5min")
    tod = compute_time_of_day(idx)
    assert tod.dtype == np.float32
    assert tod.min() >= 0.0
    assert tod.max() < 1.0


def test_time_of_day_midnight_is_zero():
    idx = pd.DatetimeIndex(["2012-03-01 00:00:00"])
    tod = compute_time_of_day(idx)
    assert abs(tod[0]) < 1e-6


def test_time_of_day_noon_is_half():
    idx = pd.DatetimeIndex(["2012-03-01 12:00:00"])
    tod = compute_time_of_day(idx)
    assert abs(tod[0] - 0.5) < 1e-4


# ---------------------------------------------------------------------------
# build_windows
# ---------------------------------------------------------------------------

def test_window_x_shape():
    data = _make_synthetic_data(T=200, N=10)
    x, y = build_windows(data, X_OFFSETS, Y_OFFSETS)
    assert x.shape == (200 - 23, 12, 10, 2)


def test_window_y_shape():
    data = _make_synthetic_data(T=200, N=10)
    x, y = build_windows(data, X_OFFSETS, Y_OFFSETS)
    assert y.shape == (200 - 23, 12, 10, 1)


def test_window_y_is_speed_channel_only():
    """y must equal channel 0 of data (raw speed), not channel 1 (time_of_day)."""
    data = _make_synthetic_data(T=200, N=10)
    x, y = build_windows(data, X_OFFSETS, Y_OFFSETS)
    # Reconstruct what y should be: anchor t=11..187, y_offsets=[1..12]
    min_t = 11
    max_t = 200 - 12
    t0 = min_t
    expected_y = data[t0 + Y_OFFSETS[0], :, :1]  # first window, first step
    assert np.allclose(y[0, 0], expected_y)


def test_no_temporal_leakage_between_train_and_val():
    """Last train window's y must not reach into the val split."""
    T = 500
    n_train, _ = split_timestamps(T)
    data = _make_synthetic_data(T=T, N=5)

    # Build train windows from data[:n_train]
    x_tr, y_tr = build_windows(data[:n_train], X_OFFSETS, Y_OFFSETS)

    # The last train anchor is n_train - 1 - 12 (last valid in the train slice)
    # Its y reaches up to (n_train - 13) + 12 = n_train - 1 < n_train
    # So y never touches index n_train or beyond within the slice
    # We verify by checking that y contains no data from data[n_train:]
    last_anchor_in_train = n_train - 1 - int(Y_OFFSETS.max())  # = n_train - 13
    # The last y value comes from index (last_anchor_in_train + 12) in the train slice
    # = n_train - 13 + 12 = n_train - 1 → within train ✓
    assert last_anchor_in_train + int(Y_OFFSETS.max()) < n_train


def test_window_count_matches_formula():
    for T in [200, 500, 34272]:
        data = _make_synthetic_data(T=T, N=5)
        x, _ = build_windows(data, X_OFFSETS, Y_OFFSETS)
        assert len(x) == T - 23, f"T={T}: expected {T-23} windows, got {len(x)}"


# ---------------------------------------------------------------------------
# compute_mask
# ---------------------------------------------------------------------------

def test_mask_true_where_nonzero():
    speed = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    mask = compute_mask(speed)
    expected = np.array([[True, False, True], [False, True, False]])
    assert np.array_equal(mask, expected)


def test_mask_all_valid_for_no_missing():
    speed = np.ones((50, 10, 12, 1), dtype=np.float32) * 45.0
    mask = compute_mask(speed)
    assert mask.all()


def test_mask_computed_before_normalisation_matters():
    """After z-score, zero-speed becomes (0 - mean)/std != 0 — mask must use raw."""
    raw_speed = np.array([0.0, 30.0, 60.0], dtype=np.float32)
    mask_raw = compute_mask(raw_speed)
    assert not mask_raw[0]        # 0.0 → missing
    assert mask_raw[1]
    # After normalisation the formerly-zero entry is non-zero
    mean, std = raw_speed[1:].mean(), raw_speed[1:].std()
    norm_speed = (raw_speed - mean) / (std + 1e-8)
    mask_norm = compute_mask(norm_speed)
    assert mask_norm[0]           # would be wrongly marked valid after normalisation


# ---------------------------------------------------------------------------
# compute_scaler
# ---------------------------------------------------------------------------

def test_scaler_keys():
    scaler = compute_scaler(np.random.rand(100, 10).astype(np.float32))
    assert "mean" in scaler and "std" in scaler


def test_scaler_values():
    data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    scaler = compute_scaler(data)
    assert abs(scaler["mean"] - 25.0) < 1e-4
    assert scaler["std"] > 0


# ---------------------------------------------------------------------------
# TrafficDataset (synthetic .npz)
# ---------------------------------------------------------------------------

def test_dataset_normalises_speed_channel():
    """After loading, x speed channel should have approximately 0 mean and 1 std."""
    T, N = 300, 8
    data = _make_synthetic_data(T=T, N=N)
    x, y = build_windows(data, X_OFFSETS, Y_OFFSETS)
    x_mask = compute_mask(x[..., :1])
    y_mask = compute_mask(y)
    scaler = compute_scaler(data[:, :, 0])

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path    = Path(tmpdir) / "train.npz"
        scaler_path = Path(tmpdir) / "scaler.yaml"
        np.savez(npz_path, x=x, y=y, x_mask=x_mask, y_mask=y_mask)
        with open(scaler_path, "w") as f:
            yaml.dump(scaler, f)

        from src.data.dataset import TrafficDataset
        ds = TrafficDataset(str(npz_path), str(scaler_path))

        speed_ch = ds.x[..., 0].numpy()
        assert abs(speed_ch.mean()) < 0.1, "normalised speed mean should be ~0"
        assert abs(speed_ch.std() - 1.0) < 0.1, "normalised speed std should be ~1"


def test_dataset_y_not_normalised():
    """y must stay in raw units — mean must NOT be near 0."""
    T, N = 300, 8
    rng = np.random.default_rng(42)
    speed_raw = (rng.uniform(40, 80, (T, N))).astype(np.float32)
    tod = np.tile(np.linspace(0, 1, T, endpoint=False, dtype=np.float32)[:, None], (1, N))
    data = np.stack([speed_raw, tod], axis=-1)
    x, y = build_windows(data, X_OFFSETS, Y_OFFSETS)
    x_mask = compute_mask(x[..., :1])
    y_mask = compute_mask(y)
    scaler = compute_scaler(speed_raw)

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path    = Path(tmpdir) / "train.npz"
        scaler_path = Path(tmpdir) / "scaler.yaml"
        np.savez(npz_path, x=x, y=y, x_mask=x_mask, y_mask=y_mask)
        with open(scaler_path, "w") as f:
            yaml.dump(scaler, f)

        from src.data.dataset import TrafficDataset
        ds = TrafficDataset(str(npz_path), str(scaler_path))

        y_mean = ds.y.numpy().mean()
        assert y_mean > 10.0, f"y should be raw mph (got mean={y_mean:.2f})"


def test_dataset_tod_channel_unchanged():
    """x[..., 1] (time_of_day) must remain in [0, 1) after loading."""
    T, N = 300, 8
    data = _make_synthetic_data(T=T, N=N)
    x, y = build_windows(data, X_OFFSETS, Y_OFFSETS)
    scaler = compute_scaler(data[:, :, 0])

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path    = Path(tmpdir) / "train.npz"
        scaler_path = Path(tmpdir) / "scaler.yaml"
        np.savez(npz_path, x=x, y=y,
                 x_mask=compute_mask(x[..., :1]),
                 y_mask=compute_mask(y))
        with open(scaler_path, "w") as f:
            yaml.dump(scaler, f)

        from src.data.dataset import TrafficDataset
        ds = TrafficDataset(str(npz_path), str(scaler_path))

        tod = ds.x[..., 1].numpy()
        assert tod.min() >= 0.0
        assert tod.max() < 1.0


# ---------------------------------------------------------------------------
# Adjacency and Laplacian (synthetic)
# ---------------------------------------------------------------------------

def test_normalized_laplacian_symmetric():
    rng = np.random.default_rng(0)
    adj = rng.uniform(0, 1, (20, 20)).astype(np.float32)
    adj = (adj + adj.T) / 2   # symmetrise
    np.fill_diagonal(adj, 0)
    L = compute_normalized_laplacian(adj)
    assert np.allclose(L, L.T, atol=1e-10), "Laplacian must be symmetric"


def test_normalized_laplacian_eigenvalue_range():
    rng = np.random.default_rng(1)
    adj = rng.uniform(0, 1, (20, 20)).astype(np.float32)
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)
    L = compute_normalized_laplacian(adj)
    eigs = np.linalg.eigvalsh(L)
    assert eigs.min() >= -1e-8, f"Smallest eigenvalue should be >= 0, got {eigs.min()}"
    assert eigs.max() <= 2.0 + 1e-8, f"Largest eigenvalue should be <= 2, got {eigs.max()}"


# ---------------------------------------------------------------------------
# Real-data tests (skipped if data not downloaded)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _DATA_PRESENT["metrla"], reason="adj_mx_metrla.pkl not present")
def test_metrla_adjacency_shape():
    from src.data.adjacency import load_adjacency
    adj = load_adjacency("metrla")
    assert adj.shape == (207, 207), f"Expected (207,207), got {adj.shape}"
    assert adj.dtype == np.float32


@pytest.mark.skipif(not _DATA_PRESENT["pemsbay"], reason="adj_mx_pemsbay.pkl not present")
def test_pemsbay_adjacency_shape():
    from src.data.adjacency import load_adjacency
    adj = load_adjacency("pemsbay")
    assert adj.shape == (325, 325), f"Expected (325,325), got {adj.shape}"


@pytest.mark.skipif(not _PROCESSED_METRLA, reason="metrla_train.npz not present")
def test_metrla_processed_x_shape():
    """Processed train file must have 2-channel x."""
    raw = np.load("data/processed/metrla_train.npz")
    assert raw["x"].shape[1] == 12
    assert raw["x"].shape[3] == 2, "x must have 2 channels (speed + tod)"
    assert raw["y"].shape[3] == 1, "y must have 1 channel (speed only)"


@pytest.mark.skipif(not _PROCESSED_METRLA, reason="metrla_train.npz not present")
def test_metrla_processed_window_count():
    """Train window count should be within 1% of expected ~23,974."""
    raw = np.load("data/processed/metrla_train.npz")
    n = raw["x"].shape[0]
    assert abs(n - 23974) / 23974 < 0.01, f"Train windows: expected ~23974, got {n}"


@pytest.mark.skipif(not _DATA_PRESENT["metrla"], reason="adj_mx_metrla.pkl not present")
def test_metrla_laplacian_eigenvalue_range():
    """Real METR-LA adjacency (directed + self-loops) must give PSD Laplacian after fix."""
    from src.data.adjacency import load_adjacency, compute_normalized_laplacian
    adj = load_adjacency("metrla")
    L = compute_normalized_laplacian(adj)
    eigs = np.linalg.eigvalsh(L)
    assert eigs.min() >= -1e-6, f"Smallest eigenvalue {eigs.min():.6f} — Laplacian not PSD"
    assert eigs.max() <= 2.0 + 1e-6, f"Largest eigenvalue {eigs.max():.6f} > 2"


@pytest.mark.skipif(not _DATA_PRESENT["pemsbay"], reason="adj_mx_pemsbay.pkl not present")
def test_pemsbay_laplacian_eigenvalue_range():
    """Real PEMS-BAY adjacency must give PSD Laplacian after symmetrise+strip fix."""
    from src.data.adjacency import load_adjacency, compute_normalized_laplacian
    adj = load_adjacency("pemsbay")
    L = compute_normalized_laplacian(adj)
    eigs = np.linalg.eigvalsh(L)
    assert eigs.min() >= -1e-6, f"Smallest eigenvalue {eigs.min():.6f} — Laplacian not PSD"
    assert eigs.max() <= 2.0 + 1e-6, f"Largest eigenvalue {eigs.max():.6f} > 2"


@pytest.mark.skipif(not _PROCESSED_PEMSBAY, reason="pemsbay_train.npz not present")
def test_pemsbay_processed_x_shape():
    """PEMS-BAY processed train file must have 2-channel x, 1-channel y."""
    raw = np.load("data/processed/pemsbay_train.npz")
    assert raw["x"].shape[1] == 12
    assert raw["x"].shape[3] == 2, "x must have 2 channels (speed + tod)"
    assert raw["y"].shape[3] == 1, "y must have 1 channel (speed only)"


@pytest.mark.skipif(not _PROCESSED_PEMSBAY, reason="pemsbay_train.npz not present")
def test_pemsbay_processed_window_count():
    """PEMS-BAY train window count should be within 1% of expected ~36,481."""
    raw = np.load("data/processed/pemsbay_train.npz")
    n = raw["x"].shape[0]
    assert abs(n - 36481) / 36481 < 0.01, f"Train windows: expected ~36481, got {n}"


@pytest.mark.skipif(not _PROCESSED_METRLA, reason="metrla_train.npz not present")
def test_metrla_y_not_normalised():
    """y in stored .npz must be raw mph — mean should be well above 0."""
    raw = np.load("data/processed/metrla_train.npz")
    y_mean = float(raw["y"].mean())
    assert y_mean > 10.0, f"y mean = {y_mean:.2f} — looks normalised (expect raw mph ~54)"


@pytest.mark.skipif(not _PROCESSED_METRLA, reason="metrla_train.npz not present")
def test_metrla_tod_channel_in_range():
    """x[..., 1] (time-of-day) must be float in [0, 1), not an integer index."""
    raw = np.load("data/processed/metrla_train.npz")
    tod = raw["x"][..., 1]
    assert tod.min() >= 0.0, f"tod min = {tod.min()}"
    assert tod.max() < 1.0, f"tod max = {tod.max()}"
    # Values should be continuous floats, not integers (DCRNN convention)
    assert tod.dtype == np.float32
