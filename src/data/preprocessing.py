# sliding windows, chronological split, time-of-day, and the missing mask.
# the protocol follows dcrnn / graph-wavenet's generate_training_data.py.
import numpy as np
import pandas as pd


def split_timestamps(
    T: int,
    train_frac: float = 0.7,
    test_frac: float = 0.2,
) -> tuple[int, int]:
    """chronological split boundaries (n_train, n_val_end).

    slices are train=[0, n_train), val=[n_train, n_val_end), test=[n_val_end, T).
    defaults are 70/10/20 (dcrnn); pass 0.6/0.2 for pems08's 60/20/20.
    """
    n_test = round(T * test_frac)
    n_train = round(T * train_frac)
    n_val_end = T - n_test
    return n_train, n_val_end


def compute_time_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """time-of-day as a float in [0, 1): (timestamp - midnight) / 24h.

    matches dcrnn (00:00 -> 0.0, 12:00 -> 0.5). returns float32 (T,).
    """
    seconds = (index - index.normalize()).total_seconds().values
    return (seconds / 86400.0).astype(np.float32)


def build_windows(
    data: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """build sliding (x, y) windows from one split array (T_split, N, C).

    x is (n_windows, 12, N, C) over the 12 input steps; y is (n_windows, 12, N, 1)
    over the 12 future steps (channel 0 only). n_windows = T_split - 23. matches
    dcrnn's generate_graph_seq2seq_io_data.
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
    """validity mask, true where the raw value != 0.

    compute it before normalisation: a raw 0 means missing, but after z-scoring
    it becomes non-zero. only used for datasets where 0 means missing.
    """
    return raw_speed_windows != 0.0


def compute_scaler(train_speed: np.ndarray) -> dict:
    """global z-score stats (single mean, std) from the training values.

    returns {"mean", "std"}; applied to x's speed channel at load time.
    """
    return {
        "mean": float(train_speed.mean()),
        "std": float(train_speed.std()),
    }
