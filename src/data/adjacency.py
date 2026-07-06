# Adjacency loading and Laplacian computation.
#
# The (N, N) source adjacency is precomputed per variant by
# scripts/build_datasets.py and stored in data/processed/<variant>/adj.npz as
# A (weighted for the speed datasets, binary for the flow datasets), A_sym and
# the symmetric normalised Laplacian L. We read A here and repair it to a single
# connected component for the heat-equation prior (see REPAIR_GRAPH below).
import numpy as np

from src.utils import paths

# graph surgery is ON (2026-07-06, supervisor requirement): the heat-equation
# prior must run on a SINGLE connected component — isolated sensors and multi-node
# clusters (pems04 has 12 components) are reconnected, not merely lifted by the
# γ-shift. repair_adjacency adds exactly (n_components - 1) edges: per non-giant
# component, ONE edge to the giant component at the most train-signal-correlated
# node pair, weight = median existing edge weight. this deviates from GAD-literal
# "no invented edges" (GAD/Vimal rely on γ alone) — every added edge is logged and
# recorded in the experiment's connectivity.json. set False for the bare road graph.
REPAIR_GRAPH = True
_ADJ_CACHE: dict = {}

_NUM_NODES = {
    "metrla":  207,
    "pemsbay": 325,
    "pems04f": 307,
    "pems04s": 307,
    "pems08f": 170,
    "pems08s": 170,
}


def _load_raw_adjacency(dataset: str) -> np.ndarray:
    """the bare (N, N) source adjacency from data/processed/<variant>/adj.npz.

    A is the weighted DCRNN road graph (speed datasets) or the binary SpecSTG
    connectivity (flow datasets); no repair applied here.
    """
    if dataset not in _NUM_NODES:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from {list(_NUM_NODES)}."
        )
    path = paths.adj_npz(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Adjacency file not found: {path}\n"
            "Run  python scripts/build_datasets.py --variant "
            f"{dataset}  (or --all) first."
        )
    return np.load(path)["A"].astype(np.float32)


def _components(adj: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """(n_components, labels, A_sym) on the symmetrised, self-loop-free graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    A = adj.astype(np.float64).copy()
    np.fill_diagonal(A, 0.0)
    A = (A + A.T) / 2.0
    n_comp, labels = connected_components(csr_matrix(A > 0), directed=False)
    return int(n_comp), labels, A


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
    """connect every disconnected component to the giant one, so the graph becomes
    a SINGLE connected component (no clusters, no isolated sensors).

    minimal surgery: per non-giant component, add ONE symmetric edge — the
    (component node, giant node) pair with the highest |train-signal correlation| —
    with weight = median existing edge weight. exactly n_components - 1 edges,
    deterministic given the train split. returns the input unchanged if the graph
    is already connected (pems08) or the train signal is missing (loud warning).
    """
    n_comp, labels, Asym = _components(adj)
    if n_comp == 1:
        return adj
    if not paths.split_npz(dataset, "train").exists():
        print(f"[adjacency] WARNING: {dataset} has {n_comp} components but no train "
              f"split to pick repair edges from — returning UNREPAIRED", flush=True)
        return adj

    C = _train_signal_corr(dataset)
    w = float(np.median(Asym[Asym > 0]))
    sizes = np.bincount(labels)
    giant = int(np.argmax(sizes))
    giant_nodes = np.where(labels == giant)[0]
    out = adj.astype(np.float64).copy()
    for comp in range(n_comp):
        if comp == giant:
            continue
        nodes = np.where(labels == comp)[0]
        # best cross pair between this component and the giant one
        sub = np.abs(C[np.ix_(nodes, giant_nodes)])
        i_loc, j_loc = np.unravel_index(int(np.argmax(sub)), sub.shape)
        i, j = int(nodes[i_loc]), int(giant_nodes[j_loc])
        out[i, j] = out[j, i] = w
        print(f"[adjacency] {dataset}: component of {len(nodes)} node(s) connected "
              f"via edge {i} -- {j} (corr={C[i, j]:.2f}, w={w:.3f})", flush=True)
    return out.astype(adj.dtype, copy=False)        # preserve the raw loader's dtype


def load_adjacency(dataset: str, repair: bool = REPAIR_GRAPH) -> np.ndarray:
    """load the (N, N) adjacency, reconnecting ALL disconnected components by
    default so the graph is a single connected component (see repair_adjacency).
    pass repair=False for the bare road graph. cached per (dataset, repair)."""
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
    with np.errstate(divide="ignore"):                 # d=0 rows are masked below
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return np.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt


def compute_scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    """scaled laplacian for chebconv: 2L/λ_max - I, eigenvalues in [-1, 1]."""
    L = compute_normalized_laplacian(adj)
    lambda_max = np.linalg.eigvalsh(L).max()
    return (2.0 / lambda_max) * L - np.eye(L.shape[0])


def connectivity(dataset: str, repair: bool = REPAIR_GRAPH) -> dict:
    """verify the graph is a SINGLE connected component.

    repairing isolated (degree-0) sensors is not enough: a graph can have every
    node at degree ≥ 1 yet still split into separate clusters. a disconnected graph
    is a real problem for the heat-equation prior — the Laplacian's zero eigenvalue
    gains one extra copy per component, so each cluster has its own un-mixable
    constant mode and the forward process can never spread noise between clusters.

    we check it two independent ways that must agree:
      - graph search: scipy.connected_components (BFS/DFS) on the binary adjacency.
      - spectrum: the multiplicity of the λ=0 Laplacian eigenvalue equals the number
        of components; the second-smallest eigenvalue (the Fiedler value) is > 0
        iff the graph is connected.
    returns n_components, component_sizes, fiedler, zero_eig_multiplicity, ok.
    """
    from scipy.sparse.csgraph import connected_components

    A = load_adjacency(dataset, repair=repair).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    A = (A + A.T) / 2.0
    n_comp, labels = connected_components(A > 0, directed=False)
    sizes = sorted(np.bincount(labels).tolist(), reverse=True)

    eig = np.linalg.eigvalsh(compute_normalized_laplacian(load_adjacency(dataset, repair=repair)))
    fiedler = float(eig[1])                       # second-smallest eigenvalue
    zero_mult = int((eig < 1e-8).sum())
    return {
        "n_components": int(n_comp),
        "component_sizes": sizes,
        "fiedler": fiedler,
        "zero_eig_multiplicity": zero_mult,
        "ok": n_comp == 1 and zero_mult == 1 and fiedler > 1e-8,
    }


def laplacian_eigendecomposition(
    L: np.ndarray, dataset: str, cache: bool = True, repaired: bool = REPAIR_GRAPH
) -> tuple[np.ndarray, np.ndarray]:
    """eigendecompose the symmetric laplacian (ascending eigenvalues).

    cached per (dataset, repaired) — the repaired single-component graph gets its
    own cache file so old raw-graph analyses stay valid. returns
    (eigenvalues (N,), eigenvectors (N, N)).
    """
    cache_path = paths.laplacian_eigs(dataset, repaired)

    if cache and cache_path.exists():
        data = np.load(cache_path)
        return data["eigenvalues"], data["eigenvectors"]

    from scipy.linalg import eigh
    eigenvalues, eigenvectors = eigh(L)   # sorted ascending, real for symmetric L

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, eigenvalues=eigenvalues, eigenvectors=eigenvectors)

    return eigenvalues, eigenvectors
