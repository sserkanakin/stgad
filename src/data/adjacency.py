# Adjacency loading and Laplacian computation for both METR-LA and PEMS-BAY.
import numpy as np


def load_adjacency(dataset: str) -> np.ndarray:
    """Load the pre-computed adjacency matrix from ``data/raw/``.

    Args:
        dataset: One of ``"metrla"`` or ``"pemsbay"``.

    Returns:
        Float64 array of shape (N, N) where N is the number of sensors.

    Raises:
        FileNotFoundError: if the pkl file is not present in ``data/raw/``.
    """
    raise NotImplementedError


def compute_normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """Compute the symmetric normalised Laplacian L = D^{-1/2}(D - A)D^{-1/2}.

    This is the matrix used in both ChebConv and the Kronecker spatial prior.

    Args:
        adj: (N, N) non-negative adjacency matrix (can have self-loops or not).

    Returns:
        (N, N) symmetric normalised Laplacian with eigenvalues in [0, 2].
    """
    raise NotImplementedError


def laplacian_eigendecomposition(
    L: np.ndarray, dataset: str, cache: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose the normalised Laplacian.

    Result is cached to ``data/processed/{dataset}_laplacian_eigs.npz`` to
    avoid repeated recomputation (N=325 takes a few seconds).

    Args:
        L: (N, N) symmetric normalised Laplacian.
        dataset: Dataset name used for cache file naming.
        cache: Whether to read from / write to disk cache.

    Returns:
        Tuple (eigenvalues, eigenvectors) with shapes (N,) and (N, N).
        Eigenvalues are sorted ascending.
    """
    raise NotImplementedError
