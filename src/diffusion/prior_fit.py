# Fit Kronecker prior parameters (delta_h, rho, beta) from empirical residuals.
# Implemented in Phase 4. Stub only.


def fit_prior(residuals_path: str, dataset: str) -> dict:
    """Fit prior parameters from training-set residuals.

    Args:
        residuals_path: Path to ``{dataset}_train_seed0.npz``.
        dataset: One of ``"metrla"`` or ``"pemsbay"``.

    Returns:
        Dict with keys ``delta_h``, ``rho``, ``beta``, ``sigma``.
    """
    raise NotImplementedError
