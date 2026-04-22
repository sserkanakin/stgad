# Sliding-window construction, train/val/test split, z-score normalisation, mask and time features.
# Protocol is FIXED per §3 of PROJECT_SPEC.md — do not deviate without updating DECISIONS.md.
import numpy as np


def split_data(data: np.ndarray, split_train: float = 0.7, split_val: float = 0.8):
    """Split a time-series array chronologically into train / val / test.

    Args:
        data: Array of shape (T, N, ...).
        split_train: Cumulative fraction at which train ends (default 0.7).
        split_val: Cumulative fraction at which val ends (default 0.8).

    Returns:
        Tuple (train, val, test) each as numpy slices of ``data``.
    """
    raise NotImplementedError


def make_windows(data: np.ndarray, t_in: int = 12, t_out: int = 12, stride: int = 1):
    """Construct sliding input/output windows.

    Args:
        data: Array of shape (T, N, C).
        t_in: Input window length.
        t_out: Output (forecast) window length.
        stride: Step between consecutive windows.

    Returns:
        Tuple (x, y) each of shape (n_windows, t_in_or_out, N, C).
    """
    raise NotImplementedError


def compute_scaler(train_data: np.ndarray):
    """Compute global z-score statistics from training data.

    Statistics are computed across all sensors and all training timestamps.

    Args:
        train_data: Raw training array of shape (T_train, N, C).

    Returns:
        Dict with keys ``mean`` (float) and ``std`` (float).
    """
    raise NotImplementedError


def normalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply z-score normalisation.

    Args:
        data: Array of any shape.
        mean: Training mean.
        std: Training standard deviation.

    Returns:
        Normalised array (same shape and dtype as input).
    """
    raise NotImplementedError


def compute_mask(raw_data: np.ndarray) -> np.ndarray:
    """Compute a boolean validity mask from raw values.

    Per §3: ``mask = (raw_value != 0.0)``, computed *before* normalisation.

    Args:
        raw_data: Raw array of shape (...).

    Returns:
        Boolean array of the same shape; False where value is missing.
    """
    raise NotImplementedError


def compute_time_features(timestamps: np.ndarray, tod_size: int = 288):
    """Derive time-of-day and day-of-week integer indices for each timestep.

    Args:
        timestamps: 1-D array of pandas Timestamps or datetime64 objects, length T.
        tod_size: Number of time slots per day (288 for 5-min data).

    Returns:
        Tuple (tod, dow) each of shape (T,) with dtype int32.
        tod ∈ [0, tod_size-1], dow ∈ [0, 6].
    """
    raise NotImplementedError
