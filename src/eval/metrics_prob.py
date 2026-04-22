# Probabilistic evaluation metrics: CRPS, energy score, variogram score, PIT, PICP.
# Protocol fixed per §4.2–4.3 of PROJECT_SPEC.md.
# CRPS via properscoring.crps_ensemble (with mask wrapper).
# Energy score and variogram score ported from scoringRules R reference — unit-tested.
import numpy as np


def masked_crps(samples: np.ndarray, true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute masked CRPS per horizon step using empirical ensemble CRPS.

    Args:
        samples: (n_samples, N_windows, H, N, 1) drawn from the model.
        true: (N_windows, H, N, 1) ground truth.
        mask: (N_windows, H, N, 1) bool, False = missing.

    Returns:
        (H,) array of mean CRPS values.
    """
    raise NotImplementedError


def energy_score(samples: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    """Compute the multivariate energy score averaged over test windows.

    ES = E||S - Y|| - 0.5 * E||S - S'|| with l2 norm over (H × N) dimensions.
    Masked entries are zeroed out before computing norms.

    Args:
        samples: (n_samples, N_windows, H, N, 1).
        true: (N_windows, H, N, 1).
        mask: (N_windows, H, N, 1) bool.

    Returns:
        Scalar mean energy score.
    """
    raise NotImplementedError


def variogram_score(
    samples: np.ndarray, true: np.ndarray, mask: np.ndarray, p: float = 0.5
) -> float:
    """Compute the variogram score of order p (Scheuerer & Hamill 2015).

    Args:
        samples: (n_samples, N_windows, H, N, 1).
        true: (N_windows, H, N, 1).
        mask: (N_windows, H, N, 1) bool.
        p: Order parameter (default 0.5 per §4.2).

    Returns:
        Scalar mean variogram score.
    """
    raise NotImplementedError


def pit_values(samples: np.ndarray, true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute probability integral transform (PIT) values.

    For each (window, horizon, sensor), PIT = F_pred(y_true) where F_pred is
    the empirical CDF of the ensemble.

    Args:
        samples: (n_samples, N_windows, H, N, 1).
        true: (N_windows, H, N, 1).
        mask: (N_windows, H, N, 1) bool.

    Returns:
        1-D array of PIT values for valid (non-masked) entries.
    """
    raise NotImplementedError


def picp(
    samples: np.ndarray, true: np.ndarray, mask: np.ndarray, levels: list[float] | None = None
) -> dict:
    """Compute prediction interval coverage probability at given nominal levels.

    Args:
        samples: (n_samples, N_windows, H, N, 1).
        true: (N_windows, H, N, 1).
        mask: (N_windows, H, N, 1) bool.
        levels: Nominal coverage levels (default [0.50, 0.90, 0.95]).

    Returns:
        Dict mapping each level to empirical coverage fraction.
    """
    raise NotImplementedError
