# Regime stratification and per-regime metric computation.
# Free-flow / transition / congested split per §4.4 of PROJECT_SPEC.md.
import numpy as np


def assign_regimes(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Assign each test window to a traffic regime based on mean input speed.

    Tercile split of the mean speed of the input window across the test set:
        free_flow   = top 1/3
        transition  = middle 1/3
        congested   = bottom 1/3

    Args:
        x: (N_windows, T_in, N, 1) input windows in original (de-normalised) speed.
        mask: (N_windows, T_in, N, 1) bool validity mask.

    Returns:
        String array of shape (N_windows,) with values in
        ``{"free_flow", "transition", "congested"}``.
    """
    raise NotImplementedError


def stratified_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    regimes: np.ndarray,
) -> dict:
    """Compute MAE and CRPS per regime per horizon.

    Args:
        pred: (N_windows, H, N, 1) point predictions.
        true: (N_windows, H, N, 1) ground truth.
        mask: (N_windows, H, N, 1) bool.
        regimes: (N_windows,) string regime labels.

    Returns:
        Nested dict ``{regime: {"mae": (H,), "crps": (H,)}}``.
    """
    raise NotImplementedError
