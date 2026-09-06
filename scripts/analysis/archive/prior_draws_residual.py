"""How do stationary-distribution draws LOOK vs real residuals? (the eyeball test)

For each dataset: draws from the three arms actually in training —
  cartL (learned-graph heat: cartesian, kNN-5 corr graph, gamma=0.05)
  cart  (road-graph heat:    cartesian, road graph,       gamma=0.8)
  iso   (white, scale-matched)
— against REAL residual windows, shown as:
  row 1: space-time heatmaps (sensors ordered by the learned graph's Fiedler vector,
         so graph-smoothness is visible as vertical coherence)
  row 2: sensor profiles at one horizon + texture statistics:
         graph roughness (mean |x_i - mean of neighbors|, learned graph),
         edge correlation (corr of values across learned-graph edges),
         temporal lag-1 correlation across horizons.

usage: .venv/bin/python scripts/analysis/prior_draws_residual.py
writes results/analysis/residual_heat/draws_<ds>.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior, _load_residuals_train, _knn_correlation_graph  # noqa: E402
from src.diffusion.forward_fcps import sample_prior                                         # noqa: E402
from src.data.adjacency import compute_normalized_laplacian                                 # noqa: E402

OUT = ROOT / "results/analysis/residual_heat"; OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["pems08s", "pems08f"]


def texture(X, A):
    """graph roughness + edge correlation + temporal lag-1 corr for fields X (B,N,H)."""
    deg = np.clip(A.sum(1), 1e-9, None)
    nb = np.einsum("ij,bjh->bih", A, X) / deg[None, :, None]
    rough = float(np.abs(X - nb).mean())
    iu, jv = np.triu_indices_from(A, k=1)
    e = A[iu, jv] > 0
    xi = X[:, iu[e], :].ravel(); xj = X[:, jv[e], :].ravel()
    edge_corr = float(np.corrcoef(xi, xj)[0, 1])
    lag = float(np.corrcoef(X[:, :, :-1].ravel(), X[:, :, 1:].ravel())[0, 1])
    return rough, edge_corr, lag


def main():
    rng = np.random.default_rng(7)
    for ds in DATASETS:
        R, M = _load_residuals_train(ds, 42, "gwn", norm="pernode")
        B, N, H = R.shape
        Rz = np.where(M, R, 0.0)
        A_learn = _knn_correlation_graph(R, 5)
        # Fiedler ordering of the learned graph -> smoothness becomes visually coherent
        lam, V = np.linalg.eigh(compute_normalized_laplacian(A_learn))
        order = np.argsort(V[:, np.argsort(lam)[1]])

        arms = {}
        load_prior.cache_clear()
        arms["cartL (learned heat)"] = load_prior(ds, "kronecker", seed=42, backbone="gwn",
                                                  temporal="cartesian", spatial_graph="correlation",
                                                  knn_k=5, gamma=0.05, norm="pernode")
        load_prior.cache_clear()
        arms["cart (road heat)"] = load_prior(ds, "kronecker", seed=42, backbone="gwn",
                                              temporal="cartesian", spatial_graph="road",
                                              gamma=0.8, norm="pernode")
        load_prior.cache_clear()
        arms["iso (white)"] = load_prior(ds, "isotropic", seed=42, backbone="gwn", norm="pernode")

        fields = {"REAL residuals": Rz[rng.integers(0, B, 400)]}
        for name, pr in arms.items():
            fields[name] = sample_prior((400,), pr, rng)

        fig, axes = plt.subplots(2, 4, figsize=(17, 7.2),
                                 gridspec_kw={"height_ratios": [1.25, 1]})
        stats_lines = []
        vmax = np.percentile(np.abs(Rz), 98)
        for j, (name, X) in enumerate(fields.items()):
            ax = axes[0, j]
            ax.imshow(X[0][order], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("horizon"); ax.set_ylabel("sensor (graph-ordered)" if j == 0 else "")
            r, ec, lg = texture(X, A_learn)
            stats_lines.append((name, r, ec, lg))
            ax2 = axes[1, j]
            ax2.plot(X[0][order, 0], lw=0.9, color="#333333")
            ax2.set_ylim(-3.2, 3.2); ax2.set_xlabel("sensor (graph-ordered)")
            ax2.set_title(f"rough={r:.2f}  edge-corr={ec:.2f}  lag1={lg:.2f}", fontsize=9)
        fig.suptitle(f"{ds}: real residual fields vs stationary draws — does the noise LOOK like the data?",
                     fontsize=12)
        fig.tight_layout()
        f = OUT / f"draws_{ds}.png"; fig.savefig(f, dpi=140); plt.close(fig)
        print(f"\n{ds}:  (graph roughness / edge-corr on learned graph / temporal lag-1)")
        for name, r, ec, lg in stats_lines:
            print(f"  {name:22s} rough={r:.3f}  edge-corr={ec:+.3f}  lag1={lg:+.3f}")
        print(f"  wrote {f}")


if __name__ == "__main__":
    main()
