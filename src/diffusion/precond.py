# EDM preconditioning (Karras et al., NeurIPS 2022, arXiv:2206.00364).
# Adapted from the EDM reference implementation (cite commit when porting).
# Implemented in Phase 5. Stub only.


def edm_precond(sigma: float) -> dict:
    """Compute EDM preconditioning scalars for a given noise level sigma.

    Returns:
        Dict with keys ``c_skip``, ``c_out``, ``c_in``, ``c_noise``.
    """
    raise NotImplementedError
