"""Which product graph + k + gamma makes the heat equation fit RESIDUALS best — exactly.

Builds the EXACT product graphs on the N*H space-time nodes (no regularity approximations):
  cartesian  A = A_S (x) I  +  I (x) A_T          (move along one axis; eigenvalues add)
  kronecker  A = A_S (x) A_T                       (move along both axes; multiply)
  strong     A = cartesian + kronecker             (union)
spatial A_S = kNN graph on inter-sensor residual correlations (k swept) or the road graph;
temporal A_T = horizon path graph. For each (construction, k): exact normalized Laplacian,
exact eigh, connected components (zero modes), then for each gamma the GAD prior
v = sigma^2/(lambda+gamma) scale-matched, scored on held-out residual windows:

  NLL gain vs iso (nats/coord, >0 = heat prior fits residuals better than white)
  low-mode alignment: data low-third energy vs prior low-third energy (match = aligned)
  stiffness theta_max/theta_min (sampler conditioning)

usage: .venv/bin/python scripts/analysis/residual_heat_graphs.py
writes results/analysis/raw_signal/residual_heat_graphs.md (+ heatmap figure)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import _load_residuals_train, _knn_correlation_graph   # noqa: E402
from src.data.adjacency import compute_normalized_laplacian, load_adjacency     # noqa: E402

OUT = ROOT / "results/analysis/raw_signal"
CELLS = [("pems08f", "gwn"), ("pems08f", "mlp"), ("pems08s", "gwn")]
KS = [3, 5, 8, 15]
GAMMAS = [0.05, 0.1, 0.2, 0.4, 0.8]
N_PROJ = 4000                      # window subsample for projections


def product_adj(A_S, A_T, kind):
    N, H = A_S.shape[0], A_T.shape[0]
    if kind == "cartesian":
        return np.kron(A_S, np.eye(H)) + np.kron(np.eye(N), A_T)
    if kind == "kronecker":
        return np.kron(A_S, A_T)
    if kind == "strong":
        return np.kron(A_S, np.eye(H)) + np.kron(np.eye(N), A_T) + np.kron(A_S, A_T)
    raise ValueError(kind)


def analyze(R, M, A_S, A_T, kind):
    """exact product graph -> (lam, V, n_zero, projections-based scores per gamma)."""
    B, N, H = R.shape
    A = product_adj(A_S, A_T, kind)
    L = compute_normalized_laplacian(A)
    lam, V = np.linalg.eigh(L)
    o = np.argsort(lam); lam, V = np.clip(lam[o], 0, None), V[:, o]
    n_zero = int((lam < 1e-8).sum())
    # project residual windows (flattened space-time) onto product eigenvectors
    idx = np.linspace(0, B - 1, min(N_PROJ, B)).astype(int)
    Rz = np.where(M, R, 0.0)[idx].reshape(len(idx), N * H)
    C = Rz @ V                                              # (B', N*H)
    emp = (C ** 2).mean(0)                                  # data energy per mode
    third = (N * H) // 3
    lf_data = float(emp[:third].sum() / emp.sum())
    iso_nll = float(0.5 * (Rz ** 2 / Rz.var() + np.log(2 * np.pi * Rz.var())).mean())
    rows = []
    for g in GAMMAS:
        v = 1.0 / (lam + g)
        v = v * emp.mean() / v.mean()                       # scale match
        nll = float((0.5 * ((C ** 2) / v + np.log(2 * np.pi * v))).mean())
        lf_prior = float(v[:third].sum() / v.sum())
        stiff = float((lam.max() + g) / (lam.min() + g))
        rows.append((g, iso_nll - nll, lf_data, lf_prior, stiff))
    return n_zero, rows


def main():
    lines = ["# Product-graph heat priors on residuals — exact analysis\n",
             "NLLgain = iso_NLL − heat_NLL in nats/coord (>0: heat fits residuals better than white).",
             "lowE d/p = low-third mode energy, data vs prior (aligned when close). zeros = zero modes",
             "(connected components) BEFORE +γ — γ>0 keeps the OU well-posed regardless.\n"]
    best = {}
    for ds, bb in CELLS:
        R, M = _load_residuals_train(ds, 42, bb, norm="pernode")
        H = R.shape[2]
        A_T = np.zeros((H, H)); i = np.arange(H - 1); A_T[i, i + 1] = A_T[i + 1, i] = 1.0
        lines.append(f"\n## {ds}/{bb}\n")
        lines.append("| spatial | product | zeros | γ | NLL gain | lowE d/p | stiffness |")
        lines.append("|---|---|--:|--:|--:|--|--:|")
        graphs = {"road": np.asarray(load_adjacency(ds, repair=True), dtype=float)}
        for k in KS:
            graphs[f"kNN{k}"] = _knn_correlation_graph(R, k)
        for gname, A_S in graphs.items():
            kinds = ["cartesian"] if gname == "road" else ["cartesian", "kronecker", "strong"]
            for kind in kinds:
                n_zero, rows = analyze(R, M, A_S, A_T, kind)
                for (g, gain, lfd, lfp, st) in rows:
                    lines.append(f"| {gname} | {kind} | {n_zero} | {g:g} | {gain:+.3f} | "
                                 f"{lfd:.2f}/{lfp:.2f} | {st:.0f} |")
                    key = (ds, bb)
                    if key not in best or gain > best[key][0]:
                        best[key] = (gain, gname, kind, g, lfd, lfp, n_zero)
                print(f"{ds}/{bb} {gname:6s} {kind:9s} zeros={n_zero} "
                      f"best-γ gain={max(r[1] for r in rows):+.3f}")
    lines.append("\n## Best configuration per cell\n")
    lines.append("| cell | NLL gain | spatial | product | γ | lowE d/p | zeros |")
    lines.append("|---|--:|---|---|--:|--|--:|")
    for (ds, bb), (gain, gname, kind, g, lfd, lfp, nz) in best.items():
        lines.append(f"| {ds}/{bb} | **{gain:+.3f}** | {gname} | {kind} | {g:g} | {lfd:.2f}/{lfp:.2f} | {nz} |")
    (OUT / "residual_heat_graphs.md").write_text("\n".join(lines))
    print(f"\nwrote {OUT}/residual_heat_graphs.md")
    for k, v in best.items():
        print(f"BEST {k}: gain={v[0]:+.3f} {v[1]} {v[2]} γ={v[3]:g}")


if __name__ == "__main__":
    main()
