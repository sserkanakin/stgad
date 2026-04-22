# Deterministic evaluation metrics: MAE, RMSE, MAPE — all mask-aware.
# Protocol fixed per §4.1 of PROJECT_SPEC.md.
import numpy as np


def masked_mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute masked MAE per horizon step.

    Args:
        pred: (N_windows, H, N, 1) predictions (de-normalised).
        true: (N_windows, H, N, 1) ground truth (de-normalised).
        mask: (N_windows, H, N, 1) bool, False = missing.

    Returns:
        (H,) array of MAE values, one per horizon step.
    """
    raise NotImplementedError


def masked_rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute masked RMSE per horizon step.

    Args:
        pred: (N_windows, H, N, 1).
        true: (N_windows, H, N, 1).
        mask: (N_windows, H, N, 1) bool.

    Returns:
        (H,) RMSE array.
    """
    raise NotImplementedError


def masked_mape(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute masked MAPE per horizon step.

    Excludes entries where ``|true| < 1.0`` from the denominator (standard
    traffic-speed convention, per §4.1 of PROJECT_SPEC.md).

    Args:
        pred: (N_windows, H, N, 1).
        true: (N_windows, H, N, 1).
        mask: (N_windows, H, N, 1) bool.

    Returns:
        (H,) MAPE array (values in [0, 1], multiply by 100 for %).
    """
    raise NotImplementedError
