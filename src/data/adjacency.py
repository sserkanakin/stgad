# Adjacency loading and Laplacian computation for METR-LA and PEMS-BAY.
# pkl files are Python-2 pickles — must use encoding='latin1' in Python 3.
# Format: 3-tuple (sensor_ids_list, sensor_id_to_index_dict, adj_mx_array).
import pickle
from pathlib import Path
import numpy as np

_ADJ_PATHS = {
    "metrla": "data/raw/adj_mx_metrla.pkl",
    "pemsbay": "data/raw/adj_mx_pemsbay.pkl",
}


def load_adjacency(dataset: str) -> np.ndarray:
    """Load the pre-computed adjacency matrix from ``data/raw/``.

    The pkl stores a 3-tuple: (sensor_ids, id_to_index, adj_mx).
    adj_mx is float32, shape (N, N), values in [0, 1].
    Edges with weight < 0.1 are zeroed (normalized_k=0.1 Gaussian kernel).

    Args:
        dataset: ``"metrla"`` (N=207) or ``"pemsbay"`` (N=325).

    Returns:
        float32 ndarray of shape (N, N).

    Raises:
        ValueError: unknown dataset name.
        FileNotFoundError: pkl not present in ``data/raw/``.
    """
    if dataset not in _ADJ_PATHS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'metrla' or 'pemsbay'.")
    path = Path(_ADJ_PATHS[dataset])
    if not path.exists():
        raise FileNotFoundError(
            f"Adjacency file not found: {path}\n"
            "Run  bash scripts/download_data.sh  first."
        )
    with open(path, "rb") as f:
        _, _, adj_mx = pickle.load(f, encoding="latin1")
    return adj_mx.astype(np.float32)


def compute_normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalised Laplacian L = I - D^{-1/2} A_sym D^{-1/2}.

    The DCRNN/PEMS-BAY adjacency matrices are directed (asymmetric) and
    include self-loops (diagonal = 1).  We symmetrise and remove self-loops
    before computing L, so eigenvalues are guaranteed in [0, 2].
    Used in the Kronecker spatial prior and spectral analysis.

    DESIGN DECISION: symmetrise + strip self-loops before Laplacian.
    See DECISIONS.md for rationale.

    Args:
        adj: (N, N) adjacency matrix (may be directed and/or have self-loops).

    Returns:
        (N, N) float64 symmetric normalised Laplacian with eigenvalues in [0, 2].
    """
    adj = adj.astype(np.float64)
    np.fill_diagonal(adj, 0)               # remove self-loops
    adj = (adj + adj.T) / 2                # symmetrise
    d = adj.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return np.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt


def compute_scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    """Scaled Laplacian for ChebConv: L_scaled = 2L / lambda_max - I.

    Rescales eigenvalues from [0, 2] to [-1, 1] as required by
    Chebyshev polynomial graph convolution.

    Args:
        adj: (N, N) non-negative adjacency matrix.

    Returns:
        (N, N) float64 scaled Laplacian with eigenvalues in [-1, 1].
    """
    L = compute_normalized_laplacian(adj)
    lambda_max = np.linalg.eigvalsh(L).max()
    return (2.0 / lambda_max) * L - np.eye(L.shape[0])


def laplacian_eigendecomposition(
    L: np.ndarray, dataset: str, cache: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Full eigendecomposition of the symmetric normalised Laplacian.

    Uses scipy.linalg.eigh (guaranteed real eigenvalues for symmetric input).
    Result cached to ``data/processed/{dataset}_laplacian_eigs.npz``.

    Args:
        L: (N, N) symmetric normalised Laplacian.
        dataset: Dataset name, used as cache key.
        cache: Read from / write to disk cache (default True).

    Returns:
        Tuple (eigenvalues, eigenvectors):
            eigenvalues  float64  (N,)   sorted ascending, ∈ [0, 2]
            eigenvectors float64  (N, N) columns are eigenvectors
    """
    cache_path = Path(f"data/processed/{dataset}_laplacian_eigs.npz")

    if cache and cache_path.exists():
        data = np.load(cache_path)
        return data["eigenvalues"], data["eigenvectors"]

    from scipy.linalg import eigh
    eigenvalues, eigenvectors = eigh(L)   # sorted ascending, real for symmetric L

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, eigenvalues=eigenvalues, eigenvectors=eigenvectors)

    return eigenvalues, eigenvectors
