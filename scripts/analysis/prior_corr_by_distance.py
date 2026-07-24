"""Correlation-by-distance: the readable version of 'does the noise look like the data?'

Three panels per dataset, one curve per source (REAL residuals, cartL draw, cart-road draw,
iso draw):
  A  correlation vs hop distance on the LEARNED graph
  B  correlation vs hop distance on the ROAD graph
  C  correlation vs time lag (horizons)
Reading: the closer a prior's curve is to the black REAL curve, the better its noise mimics
the data's correlation structure. iso is flat at zero by construction.

usage: .venv/bin/python scripts/analysis/prior_corr_by_distance.py
writes results/analysis/residual_heat/corr_by_distance_<ds>.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior, _load_residuals_train, _knn_correlation_graph  # noqa: E402
from src.diffusion.forward_fcps import sample_prior                                         # noqa: E402
from src.data.adjacency import load_adjacency                                               # noqa: E402

OUT = ROOT / "results/analysis/residual_heat"; OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["pems08s", "pems08f"]
COL = {"REAL": "#111111", "cartL (learned heat)": "#0072B2",
       "cart (road heat)": "#009E73", "iso (white)": "#D55E00"}
MAXHOP = 5


def corr_by_hop(X, D):
    """mean correlation between sensor pairs at each hop distance. X (B,N,H)."""
    Xf = X.transpose(1, 0, 2).reshape(X.shape[1], -1)        # (N, B*H)
    Xf = (Xf - Xf.mean(1, keepdims=True)) / (Xf.std(1, keepdims=True) + 1e-9)
    C = (Xf @ Xf.T) / Xf.shape[1]
    iu = np.triu_indices_from(C, k=1)
    d, c = D[iu], C[iu]
    out = []
    for h in range(1, MAXHOP + 1):
        sel = d == h
        out.append(float(c[sel].mean()) if sel.sum() > 10 else np.nan)
    far = (d > MAXHOP) & np.isfinite(d)
    out.append(float(c[far].mean()) if far.sum() > 10 else np.nan)
    return out


def corr_by_lag(X):
    """correlation between horizons at each lag. X (B,N,H)."""
    H = X.shape[2]
    out = []
    for lag in range(1, H):
        a = X[:, :, :-lag].ravel(); b = X[:, :, lag:].ravel()
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return out


def main():
    rng = np.random.default_rng(7)
    for ds in DATASETS:
        R, M = _load_residuals_train(ds, 42, "gwn", norm="pernode")
        B, N, H = R.shape
        Rz = np.where(M, R, 0.0)
        A_learn = _knn_correlation_graph(R, 5)
        A_road = np.asarray(load_adjacency(ds, repair=True)).astype(float)
        np.fill_diagonal(A_road, 0.0)
        D_learn = shortest_path((A_learn > 0).astype(float), method="D", unweighted=True)
        D_road = shortest_path((A_road > 0).astype(float), method="D", unweighted=True)

        srcs = {"REAL": Rz[rng.integers(0, B, 600)]}
        for name, kw in {
            "cartL (learned heat)": dict(temporal="cartesian", spatial_graph="correlation",
                                         knn_k=5, gamma=0.05),
            "cart (road heat)": dict(temporal="cartesian", spatial_graph="road", gamma=0.8),
        }.items():
            load_prior.cache_clear()
            pr = load_prior(ds, "kronecker", seed=42, backbone="gwn", norm="pernode", **kw)
            srcs[name] = sample_prior((600,), pr, rng)
        load_prior.cache_clear()
        srcs["iso (white)"] = sample_prior((600,), load_prior(ds, "isotropic", seed=42,
                                                              backbone="gwn", norm="pernode"), rng)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        hops = list(range(1, MAXHOP + 1)) + [MAXHOP + 1]
        hop_labels = [str(h) for h in range(1, MAXHOP + 1)] + [f">{MAXHOP}"]
        for name, X in srcs.items():
            lw = 2.6 if name == "REAL" else 1.8
            axes[0].plot(hops, corr_by_hop(X, D_learn), "o-", color=COL[name], lw=lw, label=name)
            axes[1].plot(hops, corr_by_hop(X, D_road), "o-", color=COL[name], lw=lw)
            axes[2].plot(range(1, H), corr_by_lag(X), "o-", color=COL[name], lw=lw)
        for ax, ttl, xl in [(axes[0], "A. by LEARNED-graph distance", "hops (learned graph)"),
                            (axes[1], "B. by ROAD-graph distance", "hops (road graph)"),
                            (axes[2], "C. by time lag", "lag (horizons, ×5 min)")]:
            ax.axhline(0, color="#999999", lw=0.8)
            ax.set_title(ttl, fontsize=10); ax.set_xlabel(xl)
            ax.grid(alpha=0.2)
        axes[0].set_xticks(hops, hop_labels); axes[1].set_xticks(hops, hop_labels)
        axes[0].set_ylabel("mean correlation between pairs")
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"{ds}: correlation vs distance — which prior's noise decays like the real residuals?",
                     fontsize=11)
        fig.tight_layout()
        f = OUT / f"corr_by_distance_{ds}.png"; fig.savefig(f, dpi=150); plt.close(fig)
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
