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
    """load the (N, N) adjacency for a dataset.

    dispatches by file type: .pkl for metr-la/pems-bay (dcrnn format), .csv for
    pems08 (stsgcn edge list).
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
    """symmetric normalised laplacian L = I - D^{-1/2} A D^{-1/2}.

    the dcrnn adjacency matrices are directed and include self-loops, so we
    symmetrise and strip the diagonal first; eigenvalues then lie in [0, 2].
    used for the kronecker spatial prior. adj is (N, N); returns (N, N) float64.
    """
    adj = adj.astype(np.float64)
    np.fill_diagonal(adj, 0)               # remove self-loops
    adj = (adj + adj.T) / 2                # symmetrise
    d = adj.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return np.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt


def compute_scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    """scaled laplacian for chebconv: 2L/λ_max - I, eigenvalues in [-1, 1]."""
    L = compute_normalized_laplacian(adj)
    lambda_max = np.linalg.eigvalsh(L).max()
    return (2.0 / lambda_max) * L - np.eye(L.shape[0])


def laplacian_eigendecomposition(
    L: np.ndarray, dataset: str, cache: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """eigendecompose the symmetric laplacian (ascending eigenvalues).

    cached per dataset (see src.utils.paths.laplacian_eigs). returns
    (eigenvalues (N,), eigenvectors (N, N)).
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
