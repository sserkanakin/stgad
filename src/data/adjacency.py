# Adjacency loading and Laplacian computation.
#
# METR-LA / PEMS-BAY ship the DCRNN .pkl format — Python-2 pickle, 3-tuple
#   (sensor_ids_list, sensor_id_to_index_dict, adj_mx_array). Loaded with
#   encoding='latin1' on Python 3.
#
# PEMS08F (STSGCN) ships a .csv edge list with columns from,to,cost. Sensor
#   indices are 0..N-1 contiguous; we build a binary symmetric adjacency
#   matrix (1 where an edge exists, 0 elsewhere).
import csv
import pickle
from pathlib import Path
import numpy as np

from src.utils import paths

_NUM_NODES = {
    "metrla":  207,
    "pemsbay": 325,
    "pems08f": 170,
}


def _load_adjacency_pkl(path: Path) -> np.ndarray:
    """DCRNN .pkl: (sensor_ids, id_to_index, adj_mx). Returns adj_mx float32."""
    with open(path, "rb") as f:
        _, _, adj_mx = pickle.load(f, encoding="latin1")
    return adj_mx.astype(np.float32)


def _load_adjacency_csv(path: Path, num_nodes: int) -> np.ndarray:
    """STSGCN edge list .csv (columns from,to,cost) → binary symmetric A.

    Sensor ids are contiguous 0..num_nodes-1. Edge weight from the CSV is
    discarded; we use binary connectivity (A[i,j]=1 iff (i,j) appears).
    Symmetrised so A = A.T.
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            i, j = int(row[0]), int(row[1])
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


def load_adjacency(dataset: str) -> np.ndarray:
    """Load the adjacency matrix for the given dataset from ``data/raw/``.

    Dispatches by file extension: .pkl (DCRNN format, METR-LA / PEMS-BAY)
    or .csv (STSGCN edge list, PEMS08F).

    Args:
        dataset: ``"metrla"`` (N=207), ``"pemsbay"`` (N=325), or
                 ``"pems08f"`` (N=170).

    Returns:
        float32 ndarray of shape (N, N).

    Raises:
        ValueError: unknown dataset name.
        FileNotFoundError: source file not present in ``data/raw/``.
    """
    if dataset not in _NUM_NODES:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from {list(_NUM_NODES)}."
        )
    path = paths.raw_adj(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Adjacency file not found: {path}\n"
            "Run  bash scripts/download_data.sh  (METR-LA / PEMS-BAY) or "
            "download PEMS08 from STFGNN (PEMS08F)."
        )

    if path.suffix == ".pkl":
        return _load_adjacency_pkl(path)
    if path.suffix == ".csv":
        return _load_adjacency_csv(path, _NUM_NODES[dataset])
    raise ValueError(f"Unsupported adjacency format: {path.suffix}")


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
    cache_path = paths.laplacian_eigs(dataset)

    if cache and cache_path.exists():
        data = np.load(cache_path)
        return data["eigenvalues"], data["eigenvectors"]

    from scipy.linalg import eigh
    eigenvalues, eigenvectors = eigh(L)   # sorted ascending, real for symmetric L

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, eigenvalues=eigenvalues, eigenvectors=eigenvectors)

    return eigenvalues, eigenvectors
