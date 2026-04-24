# Sliding-window construction, chronological split, time-of-day encoding, and missing mask.
# Protocol confirmed from DCRNN source: liyaguang/DCRNN/scripts/generate_training_data.py
# and nnzhan/Graph-WaveNet/generate_training_data.py — do NOT deviate.
import numpy as np
import pandas as pd


def split_timestamps(T: int) -> tuple[int, int]:
    """Return (n_train, n_val_end) split boundary indices.

    Matches DCRNN exactly:
        n_test  = round(T * 0.2)
        n_train = round(T * 0.7)
        n_val   = T - n_test - n_train

    Slices: train=[0, n_train), val=[n_train, n_val_end), test=[n_val_end, T)

    Args:
        T: Total number of timestamps.

    Returns:
        Tuple ``(n_train, n_val_end)``.
    """
    n_test = round(T * 0.2)
    n_train = round(T * 0.7)
    n_val_end = T - n_test
    return n_train, n_val_end


def compute_time_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Time-of-day as a continuous float in [0, 1) for each timestamp.

    Convention: (timestamp - midnight) / 24h — matches DCRNN exactly.
    00:00 → 0.0,  06:00 → 0.25,  12:00 → 0.5,  23:55 → ~0.997.

    Args:
        index: DatetimeIndex of length T.

    Returns:
        float32 array of shape (T,) with values in [0, 1).
    """
    seconds = (index - index.normalize()).total_seconds().values
    return (seconds / 86400.0).astype(np.float32)


def build_windows(
    data: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding input/output windows from a single split's data array.

    Operates on an already-sliced split array (train, val, or test).
    Matches DCRNN's ``generate_graph_seq2seq_io_data`` logic:
        min_t = abs(x_offsets.min())          # 11
        max_t = T_split - y_offsets.max()     # T_split - 12  (exclusive)
        for t in range(min_t, max_t): ...

    Args:
        data: Array of shape (T_split, N, C) for one split.
        x_offsets: 1-D array [-11, ..., 0] (12 steps ending at anchor t).
        y_offsets: 1-D array [1, ..., 12]  (12 future steps).

    Returns:
        Tuple (x, y):
            x — float32 (N_windows, 12, N, C)  — all feature channels
            y — float32 (N_windows, 12, N, 1)  — speed channel only
        where N_windows = T_split - len(x_offsets) - len(y_offsets) + 1
                        = T_split - 23.
    """
    T_split = data.shape[0]
    min_t = int(abs(x_offsets.min()))            # 11
    max_t = T_split - int(y_offsets.max())       # exclusive upper bound

    anchors = np.arange(min_t, max_t)            # shape (N_windows,)
    x_idx = anchors[:, None] + x_offsets[None, :]   # (N_windows, 12)
    y_idx = anchors[:, None] + y_offsets[None, :]   # (N_windows, 12)

    x = data[x_idx]              # (N_windows, 12, N, C)
    y = data[y_idx][..., :1]     # (N_windows, 12, N, 1) — speed channel only

    return x.astype(np.float32), y.astype(np.float32)


def compute_mask(raw_speed_windows: np.ndarray) -> np.ndarray:
    """Boolean validity mask: True where speed != 0.

    Must be computed BEFORE normalisation — zeros are meaningful (missing)
    only in the raw data; after normalisation they become non-zero.

    Args:
        raw_speed_windows: float32 array of any shape.

    Returns:
        Boolean array of the same shape; True = valid reading.
    """
    return raw_speed_windows != 0.0


def compute_scaler(train_speed: np.ndarray) -> dict:
    """Global z-score statistics from the training split's speed channel.

    A single (mean, std) pair across all sensors and all training timestamps.
    Applied at load time to x's speed channel only; y is never normalised.

    Args:
        train_speed: float32 array of any shape containing raw training speeds.

    Returns:
        Dict ``{"mean": float, "std": float}``.
    """
    return {
        "mean": float(train_speed.mean()),
        "std": float(train_speed.std()),
    }
