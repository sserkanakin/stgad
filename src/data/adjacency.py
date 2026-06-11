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

# reconnect isolated sensors (degree 0 in the thresholded road graph) so every
# dataset is a SINGLE connected component — the spatio-temporal heat-equation
# prior needs one connected graph, not per-component blocks. we have no raw
# sensor distances (only the thresholded DCRNN adj), so each isolated sensor is
# linked to its most-correlated neighbour by the train-period signal. set False
# to get the bare road graph (with isolated singletons) back.
REPAIR_ISOLATED = True
_ADJ_CACHE: dict = {}

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


def _load_raw_adjacency(dataset: str) -> np.ndarray:
    """the bare (N, N) road adjacency on disk, no repair.

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


def _isolated_nodes(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """indices of zero-degree sensors in the symmetrised, self-loop-free graph."""
    A = adj.astype(np.float64).copy()
    np.fill_diagonal(A, 0.0)
    A = (A + A.T) / 2.0
    return np.where(A.sum(axis=1) == 0)[0], A


def _train_signal_corr(dataset: str) -> np.ndarray:
    """(N, N) Pearson correlation of the train-period sensor signal (feature 0).

    mask-aware: missing entries are filled with the per-sensor mean before
    centring, so a few gaps don't dominate the correlation used to pick a
    neighbour for an isolated sensor.
    """
    d = np.load(paths.split_npz(dataset, "train"))
    N = d["x"].shape[2]
    X = d["x"][..., 0].astype(np.float64).reshape(-1, N)        # (B*T, N)
    M = d["x_mask"][..., 0].astype(bool).reshape(-1, N)
    X = np.where(M, X, np.nan)
    mu = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), mu, X)
    Xc = X - X.mean(axis=0)
    s = np.sqrt((Xc ** 2).sum(axis=0)); s[s == 0] = 1.0
    return (Xc.T @ Xc) / np.outer(s, s)


def repair_adjacency(dataset: str, adj: np.ndarray) -> np.ndarray:
    """connect each isolated sensor to its most-correlated neighbour (train
    signal), so the graph becomes a single connected component. minimal: adds
    one edge per isolated sensor, weight = median existing edge weight. returns
    the repaired adjacency (unchanged if nothing is isolated or no train signal).
    """
    iso, Asym = _isolated_nodes(adj)
    if len(iso) == 0 or not paths.split_npz(dataset, "train").exists():
        return adj
    C = _train_signal_corr(dataset)
    w = float(np.median(Asym[Asym > 0]))
    connected = np.where(Asym.sum(axis=1) > 0)[0]
    out = adj.astype(np.float64).copy()
    for i in iso:
        j = int(connected[np.argmax(np.abs(C[i, connected]))])
        out[i, j] = out[j, i] = w
        print(f"[adjacency] {dataset}: reconnected isolated sensor {i} -> {j} "
              f"(corr={C[i, j]:.2f}, w={w:.3f})", flush=True)
    return out.astype(adj.dtype, copy=False)        # preserve the raw loader's dtype


def load_adjacency(dataset: str, repair: bool = REPAIR_ISOLATED) -> np.ndarray:
    """load the (N, N) adjacency, reconnecting isolated sensors by default so the
    graph is a single connected component (see repair_adjacency). pass
    repair=False for the bare road graph. cached per (dataset, repair)."""
    key = (dataset, repair)
    if key not in _ADJ_CACHE:
        A = _load_raw_adjacency(dataset)
        _ADJ_CACHE[key] = repair_adjacency(dataset, A) if repair else A
    return _ADJ_CACHE[key].copy()


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
