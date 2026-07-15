"""validation figures for the GAD-ST prior (raw arm): GAD extended to the spatio-temporal
domain via the normalized CARTESIAN product graph. θ = λ_ST + γ, λ_ST = (λ_S ⊕ λ_T)/2 ∈ [0,2],
ONE fixed γ (0.04), FCPS literal. panels:

  A  temporal correlation over horizons: raw data vs GAD-ST draw vs iso draw
  B  per-horizon std: data vs draws (raw signal ≈ flat profile; both should match level)
  C  spatial spectrum: data vs GAD-ST draw vs iso draw (kron tracks, iso white)
  D  example fields (sensor × horizon): data | GAD-ST draw | iso draw — smooth vs noise
  E  forward signal retained e^{−c̄(t)θ} per θ-quantile — CONTINUOUS graded cascade
  F  forward noise level 1−e^{−2c̄(t)θ} — all modes mixed by t=1
  G  forward(data,t) mode-variance distance → 0 to the GAD-ST prior, plateau vs white

usage: python scripts/plot_gadst_checks.py --datasets metrla pems08f --out figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.residual_dataset import train_residual_sigma
from src.diffusion.forward_fcps import FCPSSchedule, sample_R_t_from_c_bar, sample_prior, noise_level_bounds
from src.diffusion.prior import load_prior
from src.utils import paths


def load_raw(ds, split, cap=2500):
    d = np.load(paths.residual_npz(ds, split, 42, "raw"))
    sr = float(np.asarray(train_residual_sigma(ds, 42, "raw", norm="global")))
    R = d["residuals"].astype(np.float64) / sr
    M = d["mask"].astype(bool)
    keep = M.reshape(M.shape[0], -1).all(1)
    Rv = R[keep]
    if Rv.shape[0] > cap:
        Rv = Rv[np.linspace(0, Rv.shape[0] - 1, cap).astype(int)]
    return Rv


def fig_for(ds, out: Path):
    load_prior.cache_clear()
    pk = load_prior(ds, "kronecker", seed=42, backbone="raw", temporal="cartesian")
    pi = load_prior(ds, "isotropic", seed=42, backbone="raw")
    Rv = load_raw(ds, "val")
    rng = np.random.default_rng(0)
    Sk = sample_prior((2000,), pk, rng)
    Si = sample_prior((2000,), pi, rng)
    N, H = pk.N, pk.H
    sched = FCPSSchedule()

    fig = plt.figure(figsize=(15, 12.5))
    fig.suptitle(f"{ds} — GAD-ST prior checks (Cartesian product graph, λ_ST∈[0,2], γ=0.04 fixed)",
                 fontweight="bold", fontsize=13)
    gs = fig.add_gridspec(3, 6, hspace=0.52, wspace=0.55)

    # A: temporal correlation
    def corrH(X):
        return np.corrcoef(X.reshape(-1, H).T)
    for j, (title, Cm) in enumerate([("raw data (val)", corrH(Rv)),
                                     ("GAD-ST draw", corrH(Sk)),
                                     ("iso draw", corrH(Si))]):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        im = ax.imshow(Cm, vmin=-0.1, vmax=1, cmap="viridis")
        ax.set_title(f"A{j+1} · {title}", fontsize=10)
        ax.set_xlabel("horizon"); ax.set_ylabel("horizon" if j == 0 else "")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # B: per-horizon std
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(np.arange(1, H + 1), Rv.std(axis=(0, 1)), "k-o", lw=2, ms=4, label="data (val)")
    ax.plot(np.arange(1, H + 1), Sk.std(axis=(0, 1)), "-s", color="tab:green", ms=4, label="GAD-ST draw")
    ax.plot(np.arange(1, H + 1), Si.std(axis=(0, 1)), "--^", color="tab:red", ms=4, label="iso draw")
    ax.set_ylim(0, None)
    ax.set_title("B · per-horizon std (raw signal ≈ flat)", fontsize=10)
    ax.set_xlabel("horizon"); ax.set_ylabel("std"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # C: spatial spectrum
    lam, Vs = pk.lam_S, pk.V_S
    o = np.argsort(lam); lam_s, Vs_s = lam[o], Vs[:, o]
    def spec(X):
        C = np.einsum("ki,bih->bkh", Vs_s.T, X); g = (C ** 2).mean((0, 2)); return g / g.sum()
    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot(lam_s, np.cumsum(spec(Rv)), "k-", lw=2.2, label="data (val)")
    ax.plot(lam_s, np.cumsum(spec(Sk)), "-", color="tab:green", lw=1.8, label="GAD-ST draw")
    ax.plot(lam_s, np.cumsum(spec(Si)), "--", color="tab:red", lw=1.8, label="iso draw")
    ax.set_title("C · spatial spectrum\nGAD-ST tracks data; iso is white", fontsize=10)
    ax.set_xlabel("graph frequency λ"); ax.set_ylabel("cumulative energy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # D: example fields. rows sorted along the FIEDLER coordinate (2nd Laplacian eigvec),
    # so graph-NEIGHBOR sensors are adjacent rows — otherwise graph-smoothness is invisible
    # by construction (a random row order scrambles it).
    fiedler = Vs_s[:, 1]
    order = np.argsort(fiedler)
    vlim = np.percentile(np.abs(Rv[0]), 99)
    subgs = gs[1, 4:6].subgridspec(1, 3, wspace=0.12)
    for j, (title, X) in enumerate([("data", Rv[0]), ("GAD-ST", Sk[0]), ("iso", Si[0])]):
        ax = fig.add_subplot(subgs[0, j])
        ax.imshow(X[order], aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title(f"D · {title}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("sensors (graph-sorted)", fontsize=8)

    # E/F: forward curves — θ quantiles now FILL the range (no bimodal gap)
    t = np.linspace(0, 1, 300)
    cbar_t = np.asarray(sched.c_bar(t))
    qs = [0, 10, 25, 50, 75, 90, 100]
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(qs)))
    axE = fig.add_subplot(gs[2, 0:2]); axF = fig.add_subplot(gs[2, 2:4])
    th_q = np.percentile(pk.theta, qs)
    th_i = float(pi.theta.mean())
    # legend in PHYSICAL units: θ_phys = λ_ST + γ ∈ [γ, 2+γ]. the internal θ is the same
    # spectrum in schedule units (×γ_GAD/γ so GAD's fixed S=7 applies verbatim; σ² co-scaled
    # — an exact reparameterization). graph eigenvalues never exceed 2.
    phys = pk.gamma / float(pk.theta.min())          # θ_phys = θ_internal · phys
    for th, c, q in zip(th_q, cols, qs):
        axE.plot(t, np.exp(-cbar_t * th), color=c, lw=2,
                 label=f"q{q}: λ+γ={th * phys:.2f}")
        axF.plot(t, 1 - np.exp(-2 * cbar_t * th), color=c, lw=2)
    axE.plot(t, np.exp(-cbar_t * th_i), "k--", lw=1.6, label="iso (single θ)")
    axF.plot(t, 1 - np.exp(-2 * cbar_t * th_i), "k--", lw=1.6)
    axE.set_title("E · signal retained e^{−c̄(t)θ}\ncontinuous cascade: smooth modes last longest", fontsize=10)
    axF.set_title("F · noise level 1−e^{−2c̄(t)θ}\nall modes fully mixed by t=1", fontsize=10)
    for ax in (axE, axF):
        ax.set_xlabel("diffusion time t"); ax.grid(alpha=0.3)
    axE.set_ylabel("signal retained"); axF.set_ylabel("noise level")
    axE.legend(fontsize=6.5)

    # G: stationarity, ours vs white
    ts = np.linspace(0.05, 1.0, 12)
    def modevar(X):
        C = np.einsum("ki,bih->bkh", pk.V_S.T, X); C = np.einsum("bkh,hg->bkg", C, pk.V_T)
        return (C ** 2).mean(0)
    Sk_var = pk.sigma2
    Sw_var = np.full_like(Sk_var, Sk_var.mean())
    d_kron, d_iso = [], []
    sub_idx = np.linspace(0, Rv.shape[0] - 1, 500).astype(int)
    for tv in ts:
        cb = float(sched.c_bar(tv))
        Rt = sample_R_t_from_c_bar(Rv[sub_idx], cb, pk, rng)
        Sv = modevar(Rt)
        d_kron.append(float(np.abs(Sv - Sk_var).mean() / Sk_var.mean()))
        d_iso.append(float(np.abs(Sv - Sw_var).mean() / Sw_var.mean()))
    ax = fig.add_subplot(gs[2, 4:6])
    ax.plot(ts, d_kron, "-o", color="tab:green", lw=2, ms=4, label="dist to GAD-ST prior (ours)")
    ax.plot(ts, d_iso, "--^", color="tab:red", lw=2, ms=4, label="dist to WHITE (scale-matched)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("G · forward(data,t) mode-variance distance\n→ 0 to OUR prior; stays far from white", fontsize=10)
    ax.set_xlabel("diffusion time t"); ax.set_ylabel("rel. L1 distance")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out.mkdir(parents=True, exist_ok=True)
    pth = out / f"gadst_checks_{ds}.png"
    fig.savefig(pth, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {pth}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["metrla", "pems08f"])
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    for ds in a.datasets:
        fig_for(ds, Path(a.out))


if __name__ == "__main__":
    main()
