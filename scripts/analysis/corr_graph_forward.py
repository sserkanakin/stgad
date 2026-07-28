"""k-NN correlation-graph prior (k=10): build the spatial graph FROM the residual
correlations, inspect its graph spectrum, and run the GAD Fig.1 forward analysis on the
resulting Kronecker prior Σ = Σ_T ⊗ Σ_S with Σ_S=(L_corr+γI)^{-1}. Prints every hyperparameter.

usage: .venv/bin/python scripts/analysis/corr_graph_forward.py [--datasets pems04f metrla] [--k 10]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import (load_prior, _load_residuals_train, _knn_correlation_graph,  # noqa: E402
                                 GAMMA_DEFAULT, SHRINK_T)
from src.diffusion.forward_fcps import (FCPSSchedule, ULSchedule, noise_level_bounds,        # noqa: E402
                                        _SNR_HI, _NOISE_FLOOR, _GAD_SAMPLE_POWER)
from src.data.adjacency import compute_normalized_laplacian                                  # noqa: E402

OUT = ROOT / "results/analysis/paper_results/figs"
OUT.mkdir(parents=True, exist_ok=True)
COLS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
BB, SEED = "gwn", 42


def graph_stats(A):
    N = A.shape[0]
    deg = (A > 0).sum(1)                                          # undirected degree (after symmetrise)
    L = compute_normalized_laplacian(A)
    lam = np.sort(np.linalg.eigvalsh(L))
    ncomp = int((lam < 1e-8).sum())
    nedges = int((A > 0).sum() // 2)
    return dict(N=N, edges=nedges, deg_mean=float(deg.mean()), deg_min=int(deg.min()),
                deg_max=int(deg.max()), ncomp=ncomp, lam=lam)


def sliced_w1(A, B, dirs):
    pa, pb = A @ dirs, B @ dirs
    sd = pb.std(0) + 1e-12
    pa, pb = np.sort(pa / sd, 0), np.sort(pb / sd, 0)
    return float(np.abs(pa - pb).mean())


def analyze(ds, k):
    R, mask = _load_residuals_train(ds, SEED, BB)                # unit-variance residual space
    A = _knn_correlation_graph(R, k)                             # the k-NN correlation graph
    gs = graph_stats(A)
    # the production prior built on THIS graph (correlation), and the road graph for contrast
    pr = load_prior(ds, "kronecker", seed=SEED, backbone=BB, spatial="structured",
                    temporal="empirical", spatial_graph="correlation", knn_k=k)
    pr_road = load_prior(ds, "kronecker", seed=SEED, backbone=BB, spatial="structured",
                         temporal="empirical", spatial_graph="road")
    N, H = pr.N, pr.H
    theta = pr.theta.ravel(); s2 = pr.sigma2_scale
    th_min, th_max = float(theta.min()), float(theta.max())

    # residual energy vs graph frequency on the correlation graph (Dirichlet, cumulative)
    C = np.einsum("ki,bih->bkh", pr.V_S.T, np.where(mask, R, 0.0), optimize=True)
    E = (C ** 2).mean(axis=(0, 2)); o = np.argsort(pr.lam_S)
    dirich = float((pr.lam_S * E).sum() / E.sum() / pr.lam_S.mean())
    cum = np.cumsum(E[o]) / E.sum()

    # ---------- FORWARD ANALYSIS (GAD Fig.1) ----------
    fcps = FCPSSchedule(); uls = ULSchedule(slope=2.0 * fcps.c_0)
    c_lo, c_hi = noise_level_bounds(pr)
    order = np.argsort(theta)
    idx = [order[0], order[len(order)//3], order[2*len(order)//3], order[-1]]
    labs = ["θ_min (smoothest)", "θ~1/3", "θ~2/3", "θ_max (roughest)"]
    t = np.linspace(0, 1, 300); cbF, cbU = fcps.c_bar(t), uls.c_bar(t)

    # MC forward on real residuals in the eigenbasis
    sub = np.linspace(0, len(R) - 1, 2500).astype(int); B0 = len(sub)
    C0 = np.einsum("ki,bih,hj->bkj", pr.V_S.T, R[sub], pr.V_T, optimize=True).reshape(B0, N, H)
    rng = np.random.default_rng(0)
    v_model = pr.sigma2                                          # (N,H) σ²/θ
    d = N * H; dirs = rng.standard_normal((d, 256)); dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    prior_coeff = (np.sqrt(v_model)[None] * rng.standard_normal((B0, N, H))).reshape(B0, d)
    ts = np.linspace(0, 1, 13); w1s, verr = [], []
    for ti in ts:
        cb = float(fcps.c_bar(ti))
        Ht = np.exp(-pr.theta * cb)
        s2t = s2 * (1 - np.exp(-2 * pr.theta * cb)) / pr.theta
        Ct = Ht[None] * C0 + np.sqrt(s2t)[None] * rng.standard_normal((B0, N, H))
        verr.append(float(np.abs((Ct**2).mean(0) - v_model).mean() / v_model.mean()))
        w1s.append(sliced_w1(Ct.reshape(B0, d), prior_coeff, dirs))

    return dict(ds=ds, k=k, gs=gs, pr=pr, pr_road=pr_road, N=N, H=H, s2=s2,
                th_min=th_min, th_max=th_max, dirich=dirich, cum=cum, lam_o=pr.lam_S[o],
                idx=idx, labs=labs, t=t, cbF=cbF, cbU=cbU, ts=ts, w1s=np.array(w1s),
                verr=np.array(verr), fcps=fcps, c_lo=c_lo, c_hi=c_hi)


def print_hparams(a):
    pr, gs = a["pr"], a["gs"]
    print(f"\n================ {a['ds']} / {BB} : k={a['k']} correlation-graph prior ================")
    print("--- GRAPH CONSTRUCTION (spatial factor source) ---")
    print(f"  spatial_graph        = correlation (kNN on residual Pearson corr)")
    print(f"  knn_k                = {a['k']}   (top-k neighbours per node, pre-symmetrise)")
    print(f"  symmetrise           = mutual-OR  max(A, Aᵀ)")
    print(f"  edge weight          = max(corr, 0)   (nonneg correlation)")
    print(f"  self-loops           = removed (diag → -inf before top-k; Laplacian strips them)")
    print(f"  corr source          = train residuals R (B,N,H) flattened to (N, B·H), backbone={BB}, seed={SEED}")
    print(f"  → N nodes            = {gs['N']}")
    print(f"  → undirected edges   = {gs['edges']}")
    print(f"  → degree (after OR)  = mean {gs['deg_mean']:.1f}, min {gs['deg_min']}, max {gs['deg_max']}")
    print(f"  → connected comps    = {gs['ncomp']}  (γ lifts zero modes, no repair needed)")
    print("--- LAPLACIAN + SPATIAL FACTOR Σ_S = (L+γI)^-1 ---")
    print(f"  laplacian            = normalized symmetric  L = I - D^-1/2 A D^-1/2")
    print(f"  γ (GAMMA_DEFAULT)    = {GAMMA_DEFAULT}   (GAD, fixed; Σ_S=(L+γI)^-1)")
    print(f"  τ² trace-calibration = N / Σ(1/(λ+γ))   (keeps mean spatial variance = 1)")
    print(f"  λ_S (graph freq)     = [{a['lam_o'][0]:.3f}, {a['lam_o'][-1]:.3f}], median {np.median(a['lam_o']):.3f}")
    print("--- TEMPORAL FACTOR Σ_T (empirical) ---")
    print(f"  D_H                  = per-horizon 2nd moment (H={a['H']})")
    print(f"  R_H                  = empirical horizon correlation")
    print(f"  ρ (SHRINK_T)         = {SHRINK_T}   (ridge μ←(1-ρ)μ+ρμ̄ toward scaled-I)")
    print(f"  winsor               = None (no clipping)")
    print("--- JOINT PRECISION + SCALE ---")
    print(f"  θ[k,h]               = (λ_S[k]+γ) / (τ² μ_T[h])")
    print(f"  θ range              = [{a['th_min']:.3f}, {a['th_max']:.2f}]   stiffness {a['th_max']/a['th_min']:.0f}")
    print(f"  scale norm           = θ_min → γ={GAMMA_DEFAULT} (GAD scale); σ²_scale absorbs → {a['s2']:.3f}")
    print(f"  Σ_stat (per mode)    = σ²_scale / θ")
    print("--- FORWARD SDE + SCHEDULE (FCPS, GAD-literal) ---")
    print(f"  SDE                  = dr = -c(t)·Q·r dt + √(2c σ²) dW,  Q=Σ⁻¹, dW white")
    print(f"  schedule             = FCPS  c(t)=c_min+k·uᵅ,  c̄(t)=c_min·t+(c_0-c_min T)u^(α+1)")
    print(f"  c_min                = {a['fcps'].c_min}")
    print(f"  c_0 (= GAD's S)      = {a['fcps'].c_0}")
    print(f"  alpha                = {a['fcps'].alpha}")
    print(f"  T                    = {a['fcps'].T}")
    print(f"  mixing 2·θ_min·c_0   = {2*a['th_min']*a['fcps'].c_0:.1f}  (GAD rule ≥ 5)")
    print(f"  noise floor          = {_NOISE_FLOOR}  → c̄_lo = floor²/2 = {a['c_lo']:.4f}")
    print(f"  SNR_HI               = {_SNR_HI}  → c̄_hi = SNR_HI/θ_min = {a['c_hi']:.2f}")
    print(f"  train noise sampler  = GAD power-law c̄=c_0·u^p, p={_GAD_SAMPLE_POWER}")
    print(f"  marginal (per mode)  = H_t=e^(-c̄θ),  σ²_t=σ²(1-e^(-2c̄θ))/θ")
    print("--- MC CONVERGENCE ANALYSIS (this figure) ---")
    print(f"  residual samples     = 2500 (subsampled train), sliced-W1 dirs = 256, time points = 13")
    print(f"  Dirichlet(resid on corr graph) = {a['dirich']:.3f}  (white=1; lower=graph-smooth)")
    print(f"  W1 t=0→1: {a['w1s'][0]:.3f}→{a['w1s'][-1]:.3f}   var-relerr t=0→1: {a['verr'][0]:.3f}→{a['verr'][-1]:.3f}")


def figure(a):
    pr = a["pr"]; theta = pr.theta.ravel(); s2 = a["s2"]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    aS, aL, aM, aR = axes

    # panel 0: graph spectrum (residual energy vs frequency, corr vs road) + eigenvalue dist
    N = a["N"]; rank = np.arange(1, N + 1) / N
    aS.plot(rank, a["cum"], lw=2, color=COLS[0], label=f"resid energy (corr graph, Dir={a['dirich']:.2f})")
    # road for contrast
    Rr, mr = _load_residuals_train(a["ds"], SEED, BB)
    Cr = np.einsum("ki,bih->bkh", a["pr_road"].V_S.T, np.where(mr, Rr, 0.0), optimize=True)
    Er = (Cr ** 2).mean(axis=(0, 2)); orr = np.argsort(a["pr_road"].lam_S)
    dr = float((a["pr_road"].lam_S * Er).sum() / Er.sum() / a["pr_road"].lam_S.mean())
    aS.plot(np.arange(1, len(Er)+1)/len(Er), np.cumsum(Er[orr])/Er.sum(), lw=2, color=COLS[3],
            label=f"resid energy (road graph, Dir={dr:.2f})")
    aS.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.6, label="white")
    aS.set_title(f"{a['ds']}: residual energy vs graph mode (k={a['k']} corr vs road)", fontsize=9)
    aS.set_xlabel("fraction of graph modes (smooth→rough)"); aS.set_ylabel("cumulative share")
    aS.legend(fontsize=7.5, frameon=False); aS.grid(alpha=0.25)

    # panels 1-3: GAD Fig.1 forward
    for m, lb, c in zip(a["idx"], a["labs"], COLS):
        th = theta[m]
        aL.plot(a["t"], np.exp(-a["cbF"] * th), "-", color=c, lw=2, label=f"{lb} θ={th:.2f}")
        aL.plot(a["t"], np.exp(-a["cbU"] * th), "--", color=c, lw=1.3)
        aM.plot(a["t"], (s2/th)*(1-np.exp(-2*a["cbF"]*th)), "-", color=c, lw=2)
        aM.plot(a["t"], (s2/th)*(1-np.exp(-2*a["cbU"]*th)), "--", color=c, lw=1.3)
        aM.axhline(s2/th, color=c, ls=":", lw=0.8, alpha=0.6)
    aL.set_title("mean decay $H_t=e^{-\\bar c_t\\theta}$ (— FCPS, -- ULS)", fontsize=9)
    aL.set_xlabel("t"); aL.set_ylabel("signal retained"); aL.set_ylim(-.02, 1.02)
    aL.legend(fontsize=7, frameon=False); aL.grid(alpha=0.25)
    aM.set_title("covariance growth $\\Sigma_t=(\\sigma^2/\\theta)(1-e^{-2\\bar c_t\\theta})$", fontsize=9)
    aM.set_xlabel("t"); aM.set_ylabel("noise injected"); aM.grid(alpha=0.25)
    aR.plot(a["ts"], a["w1s"], "-o", color=COLS[0], lw=2, ms=4, label="sliced-$W_1$ to prior")
    aR.plot(a["ts"], a["verr"], "-s", color=COLS[3], lw=2, ms=4, label="per-mode var rel err")
    aR.axhline(0, color="k", lw=.6, alpha=.4)
    aR.set_title("MC forward on real residuals → stationary $\\Sigma_T\\!\\otimes\\!\\Sigma_S$", fontsize=9)
    aR.set_xlabel("t (FCPS)"); aR.set_ylabel("distance to prior"); aR.set_ylim(bottom=-.03)
    aR.legend(fontsize=8, frameon=False); aR.grid(alpha=0.25)

    fig.suptitle(f"{a['ds']}/{BB}: k={a['k']} correlation-graph Kronecker prior — spectrum + GAD forward "
                 f"(N={a['N']}, H={a['H']}, θ∈[{a['th_min']:.2f},{a['th_max']:.1f}], stiffness {a['th_max']/a['th_min']:.0f})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT / f"corr_graph_forward_{a['ds']}_k{a['k']}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["pems04f", "metrla"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    for ds in args.datasets:
        a = analyze(ds, args.k)
        print_hparams(a)
        p = figure(a)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
