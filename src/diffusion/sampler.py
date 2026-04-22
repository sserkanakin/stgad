# Heun (2nd-order) reverse SDE sampler with structured noise injection.
# Implemented in Phase 5. Stub only.


def heun_sample(denoiser, x_T, sde, n_steps: int = 50):
    """Draw samples from the reverse SDE using the Heun solver.

    Args:
        denoiser: Trained Denoiser instance.
        x_T: Initial noise tensor of shape (batch, H, N, 1).
        sde: StructuredSDE instance carrying prior parameters.
        n_steps: Number of discretisation steps.

    Returns:
        Denoised residual tensor of the same shape as ``x_T``.
    """
    raise NotImplementedError
