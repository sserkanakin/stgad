"""one-figure-per-dataset validation of the predict-refine prior (GWN residual arm).

seven panels, each answering one question with data:
  A  temporal correlation: data R_H  vs  empirical Σ_T (USED)  vs  path-Laplacian (rejected)
     -> the empirical temporal factor looks like the data; the product-graph temporal doesn't.
  B  per-horizon std: data vs kron-draw vs iso-draw -> kron carries the data's horizon
     variance growth; iso is flat.
  C  spatial spectrum (cumulative energy vs λ): data vs kron vs iso -> kron tracks the data.
  D  example fields (sensor × horizon): data | kron draw | iso draw -> the visible difference.
  E  forward signal retained e^{-c̄(t)θ} vs diffusion time t, θ-quantile modes, kron solid /
     iso dashed (thesis forward-analysis style) -> graded, graph-aware corruption; c̄(T)=S.
  F  forward noise level 1 - e^{-2c̄(t)θ} vs t -> all modes reach 1 by t=1 (fully mixed).
  G  stationarity: distance of the forwarded VAL data's mode-variance to the kron prior (→0)
     vs to the scale-matched WHITE prior (plateau >0) -> the forward lands on OUR prior,
     and that stationary law is clearly not white noise.

usage: python scripts/plot_refine_prior_checks.py --out figs \
          --cells metrla:road:0.05 pems08f:correlation:0.05
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.adjacency import compute_normalized_laplacian, load_adjacency
from src.data.residual_dataset import train_residual_sigma
from src.diffusion.forward_fcps import FCPSSchedule, sample_R_t_from_c_bar, sample_prior
from src.diffusion.prior import _temporal_laplacian_eig, load_prior
from src.utils import paths


def load_R0(ds, split, cap=3000):
    d = np.load(paths.residual_npz(ds, split, 42, "gwn"))
    sr = float(np.asarray(train_residual_sigma(ds, 42, "gwn", norm="global")))
    R = d["residuals"].astype(np.float64) / sr
    M = d["mask"].astype(bool)
    keep = M.reshape(M.shape[0], -1).all(1)
    Rv = R[keep]
    if Rv.shape[0] > cap:
        Rv = Rv[np.linspace(0, Rv.shape[0] - 1, cap).astype(int)]
    return Rv


def fig_for(ds, sg, gamma, out: Path):
    load_prior.cache_clear()
    pk = load_prior(ds, "kronecker", seed=42, backbone="gwn", temporal="empirical",
                    winsor=90.0, spatial_graph=sg, gamma=gamma)
    pi = load_prior(ds, "isotropic", seed=42, backbone="gwn")
    Rv = load_R0(ds, "val")
    rng = np.random.default_rng(0)
    Sk = sample_prior((2000,), pk, rng)
    Si = sample_prior((2000,), pi, rng)
    N, H = pk.N, pk.H
    sched = FCPSSchedule()

    fig = plt.figure(figsize=(15, 12.5))
    fig.suptitle(f"{ds} — predict-refine prior checks (graph={sg}, γ={gamma}, winsor=90)",
                 fontweight="bold", fontsize=13)
    gs = fig.add_gridspec(3, 6, hspace=0.52, wspace=0.55)

    # ── A: temporal correlation heatmaps (data vs used vs rejected)
    def corrH(X):
        return np.corrcoef(X.reshape(-1, H).T)
    lapT_mu, lapT_V = _temporal_laplacian_eig(H, 0.6)
    Sig_lap = (lapT_V * lapT_mu) @ lapT_V.T
    d_lap = np.sqrt(np.diag(Sig_lap)); C_lap = Sig_lap / np.outer(d_lap, d_lap)
    C_data = corrH(Rv); C_kron = corrH(Sk)
    for j, (title, Cm) in enumerate([("data (val)", C_data),
                                     ("prior Σ_T — empirical (USED)", C_kron),
                                     ("path-Laplacian (REJECTED)", C_lap)]):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        im = ax.imshow(Cm, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"A{j+1} · {title}", fontsize=10)
        ax.set_xlabel("horizon"); ax.set_ylabel("horizon" if j == 0 else "")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # ── B: per-horizon std
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(np.arange(1, H + 1), Rv.std(axis=(0, 1)), "k-o", lw=2, ms=4, label="data (val)")
    ax.plot(np.arange(1, H + 1), Sk.std(axis=(0, 1)), "-s", color="tab:green", ms=4, label="kron draw")
    ax.plot(np.arange(1, H + 1), Si.std(axis=(0, 1)), "--^", color="tab:red", ms=4, label="iso draw")
    ax.set_title("B · per-horizon std\nkron carries the error growth; iso is flat", fontsize=10)
    ax.set_xlabel("horizon"); ax.set_ylabel("std"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── C: spatial spectrum (in the prior's own graph basis)
    lam, Vs = pk.lam_S, pk.V_S
    o = np.argsort(lam); lam_s, Vs_s = lam[o], Vs[:, o]
    def spec(X):
        C = np.einsum("ki,bih->bkh", Vs_s.T, X); g = (C ** 2).mean((0, 2)); return g / g.sum()
    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot(lam_s, np.cumsum(spec(Rv)), "k-", lw=2.2, label="data (val)")
    ax.plot(lam_s, np.cumsum(spec(Sk)), "-", color="tab:green", lw=1.8, label="kron draw")
    ax.plot(lam_s, np.cumsum(spec(Si)), "--", color="tab:red", lw=1.8, label="iso draw")
    ax.set_title(f"C · spatial spectrum ({sg} graph)\nkron tracks data; iso is white", fontsize=10)
    ax.set_xlabel("graph frequency λ"); ax.set_ylabel("cumulative energy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── D: example fields
    sub = np.random.default_rng(1).choice(N, size=min(N, 60), replace=False)
    vlim = np.percentile(np.abs(Rv[0]), 99)
    subgs = gs[1, 4:6].subgridspec(1, 3, wspace=0.12)
    for j, (title, X) in enumerate([("data", Rv[0]), ("kron", Sk[0]), ("iso", Si[0])]):
        ax = fig.add_subplot(subgs[0, j])
        ax.imshow(X[sub], aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title(f"D · {title}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])

    # ── E/F: forward curves vs diffusion time (thesis style): kron solid, iso dashed
    t = np.linspace(0, 1, 200)
    cbar_t = np.asarray(sched.c_bar(t))
    qs = [0, 25, 50, 75, 100]
    cols = plt.cm.tab10(np.linspace(0, 0.45, len(qs)))
    axE = fig.add_subplot(gs[2, 0:2]); axF = fig.add_subplot(gs[2, 2:4])
    th_k = np.percentile(pk.theta, qs)
    th_i = float(pi.theta.mean())                       # iso: single θ
    for th, c, q in zip(th_k, cols, qs):
        axE.plot(t, np.exp(-cbar_t * th), color=c, lw=2, label=f"kron θ q{q}")
        axF.plot(t, 1 - np.exp(-2 * cbar_t * th), color=c, lw=2)
    axE.plot(t, np.exp(-cbar_t * th_i), "k--", lw=1.6, label="iso (single θ)")
    axF.plot(t, 1 - np.exp(-2 * cbar_t * th_i), "k--", lw=1.6)
    axE.set_title("E · signal retained e^{−c̄(t)θ}\nsmooth modes last longest; iso = one curve", fontsize=10)
    axF.set_title("F · noise level 1−e^{−2c̄(t)θ}\nall modes fully mixed by t=1 (c̄(T)=S=7)", fontsize=10)
    for ax in (axE, axF):
        ax.set_xlabel("diffusion time t"); ax.grid(alpha=0.3)
    axE.set_ylabel("signal retained"); axF.set_ylabel("noise level")
    axE.legend(fontsize=7)

    # ── G: forward → stationary, and stationary ≠ white
    ts = np.linspace(0.05, 1.0, 12)
    d_kron, d_iso = [], []
    def modevar(X):
        C = np.einsum("ki,bih->bkh", pk.V_S.T, X); C = np.einsum("bkh,hg->bkg", C, pk.V_T)
        return (C ** 2).mean(0)
    Sk_var = pk.sigma2                                    # kron stationary per-mode var
    # scale-matched WHITE stationary: same total variance, flat across modes
    Sw_var = np.full_like(Sk_var, Sk_var.mean())
    sub_idx = np.linspace(0, Rv.shape[0] - 1, 600).astype(int)
    for tv in ts:
        cb = float(sched.c_bar(tv))
        Rt = sample_R_t_from_c_bar(Rv[sub_idx], cb, pk, rng)
        Sv = modevar(Rt)
        d_kron.append(float(np.abs(Sv - Sk_var).mean() / Sk_var.mean()))
        d_iso.append(float(np.abs(Sv - Sw_var).mean() / Sw_var.mean()))
    ax = fig.add_subplot(gs[2, 4:6])
    ax.plot(ts, d_kron, "-o", color="tab:green", lw=2, ms=4, label="dist to KRON prior (ours)")
    ax.plot(ts, d_iso, "--^", color="tab:red", lw=2, ms=4, label="dist to WHITE (scale-matched)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("G · forward(data,t) mode-variance distance\n→ 0 to OUR prior; stays far from white",
                 fontsize=10)
    ax.set_xlabel("diffusion time t"); ax.set_ylabel("rel. L1 distance")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out.mkdir(parents=True, exist_ok=True)
    pth = out / f"refine_prior_checks_{ds}.png"
    fig.savefig(pth, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {pth}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["metrla:road:0.05", "pems08f:correlation:0.05"],
                    help="dataset:graph:gamma triples")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    for cell in a.cells:
        ds, sg, g = cell.split(":")
        fig_for(ds, sg, float(g), Path(a.out))


if __name__ == "__main__":
    main()
