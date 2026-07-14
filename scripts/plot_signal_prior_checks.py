"""visual pre-flight for the raw-signal GAD prior. one figure per dataset:

  A forward mode-decay by graph band — H(c̄)=exp(-θc̄) averaged over low/mid/high-λ
    modes vs the noise level c̄∈[0,c_0]. graph-aware = the SMOOTH (low-λ) modes decay
    SLOWEST (retained longest); c_0 marked. confirms frequencies don't wash out too fast.
  B forward → stationary — forward the val data to rising c̄; total variance converges to
    the prior's stationary variance (ratio → 1). confirms the forward lands on the prior.
  C prior vs data spatial spectrum — per-graph-mode energy of val data vs draws from the
    kron prior vs the iso prior. kron should track the data; iso is flat (white).
  D example fields — a val sample, a kron draw, an iso draw (N×H heatmaps). kron looks
    spatially smooth like the data; iso is spatial noise. the VISIBLE difference.

usage:  python scripts/plot_signal_prior_checks.py --datasets metrla pems08f \
            --gamma 0.04 --shrink-t 0.5 --out /tmp/.../figs
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.diffusion.prior import load_prior
from src.diffusion.forward_fcps import (
    FCPSSchedule, sample_prior, sample_R_t_from_c_bar, noise_level_bounds,
)
from src.data.residual_dataset import train_residual_sigma
from src.data.adjacency import load_adjacency, compute_normalized_laplacian
from src.utils import paths


def load_val_R0(ds):
    d = np.load(paths.residual_npz(ds, "val", 42, "raw"))
    sr = float(np.asarray(train_residual_sigma(ds, 42, "raw", norm="global")))
    R = d["residuals"].astype(np.float64) / sr
    M = d["mask"].astype(bool)
    keep = M.reshape(M.shape[0], -1).all(1)
    return R[keep]


def fig_for(ds, gamma, shrink_t, out: Path):
    load_prior.cache_clear()
    pk = load_prior(ds, "kronecker", seed=42, backbone="raw", gamma=gamma,
                    temporal="laplacian", gamma_t=shrink_t)
    pi = load_prior(ds, "isotropic", seed=42, backbone="raw")
    lam, Vs = np.linalg.eigh(compute_normalized_laplacian(np.asarray(load_adjacency(ds, repair=True))))
    o = np.argsort(lam); lam, Vs = lam[o], Vs[:, o]
    N, H = pk.N, pk.H
    Rv = load_val_R0(ds); Rv = Rv[:2000]
    rng = np.random.default_rng(0)
    Sk = sample_prior((2000,), pk, rng); Si = sample_prior((2000,), pi, rng)
    sched = FCPSSchedule()
    c0 = sched.c_0

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{ds} — raw-signal GAD prior checks (γ={gamma}, gamma_t={shrink_t})", fontweight="bold")

    # --- A: forward retention by graph band, on a LOG c̄ axis so the graded decay is visible.
    # per spatial mode take its SLOWEST-decaying θ (min over horizon = the common-temporal
    # column); low-λ (smooth) modes then have the smallest θ and are retained the longest.
    cb = np.geomspace(noise_level_bounds(pk)[0], c0, 90)
    lo = lam <= np.quantile(lam, 1/3); hi = lam >= np.quantile(lam, 2/3); mid = ~lo & ~hi
    theta_slow = pk.theta.min(axis=1)  # (N,) slowest θ per spatial mode
    for band, msk, col in [("low-λ (smooth)", lo, "tab:green"), ("mid-λ", mid, "tab:orange"), ("high-λ", hi, "tab:red")]:
        th = theta_slow[msk].mean()
        ax[0, 0].semilogx(cb, np.exp(-th * cb), color=col, lw=2, label=f"{band}  θ={th:.2f}")
    ax[0, 0].axvline(c0, ls="--", c="k", alpha=0.5); ax[0, 0].text(c0, 0.55, " c₀", va="center")
    ax[0, 0].set_title("A · forward signal retained H(c̄) by graph band (log c̄)\nsmooth (low-λ) modes retained longest; c₀ mixes even them")
    ax[0, 0].set_xlabel("noise level c̄ (log)"); ax[0, 0].set_ylabel("H(c̄) = exp(−θ c̄)"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    # --- B: forward -> stationary. forward the data to rising c̄, total var vs prior stationary.
    prior_var = float(pk.sigma2.mean())
    cbs = np.geomspace(noise_level_bounds(pk)[0], c0, 9)
    ratios = []
    for c in cbs:
        Rt = sample_R_t_from_c_bar(Rv[:600], float(c), pk, rng)
        ratios.append((Rt ** 2).mean() / prior_var)
    ax[0, 1].semilogx(cbs, ratios, "o-", color="tab:blue")
    ax[0, 1].axhline(1.0, ls="--", c="k", alpha=0.5); ax[0, 1].text(cbs[0], 1.02, " stationary", fontsize=9)
    ax[0, 1].set_title("B · forward → stationary\n(Var(forward data) / prior Var → 1)")
    ax[0, 1].set_xlabel("noise level c̄"); ax[0, 1].set_ylabel("variance ratio"); ax[0, 1].grid(alpha=0.3)

    # --- C: spatial spectrum, data vs kron vs iso
    def spec(X):
        C = np.einsum("ki,bih->bkh", Vs.T, X); g = (C ** 2).mean((0, 2)); return g / g.sum()
    ax[1, 0].plot(lam, np.cumsum(spec(Rv)), label="val data", lw=2.2, color="k")
    ax[1, 0].plot(lam, np.cumsum(spec(Sk)), label="kron prior draw", lw=1.8, color="tab:green")
    ax[1, 0].plot(lam, np.cumsum(spec(Si)), label="iso prior draw", lw=1.8, ls="--", color="tab:red")
    ax[1, 0].set_title("C · spatial spectrum (cumulative energy vs λ)\nkron tracks data; iso is white")
    ax[1, 0].set_xlabel("graph frequency λ"); ax[1, 0].set_ylabel("cumulative energy"); ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

    # --- D: example fields (N×H). one val sample, one kron draw, one iso draw.
    sub = np.random.default_rng(1).choice(N, size=min(N, 60), replace=False)
    vlim = np.percentile(np.abs(Rv[0]), 99)
    gs = ax[1, 1].get_gridspec(); ax[1, 1].remove()
    sub_axes = gs[1, 1].subgridspec(1, 3, wspace=0.1)
    for j, (title, X) in enumerate([("val data", Rv[0]), ("kron draw", Sk[0]), ("iso draw", Si[0])]):
        a = fig.add_subplot(sub_axes[0, j])
        a.imshow(X[sub], aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        a.set_title(title, fontsize=10); a.set_xlabel("horizon")
        if j == 0: a.set_ylabel("sensors (sorted)")
        else: a.set_yticks([])
    fig.text(0.74, 0.46, "D · example fields (sensor × horizon) — kron is spatially smooth like the data; iso is noise",
             ha="center", fontsize=9)

    out.mkdir(parents=True, exist_ok=True)
    p = out / f"signal_prior_checks_{ds}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[fig] {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["metrla", "pems08f"])
    ap.add_argument("--gamma", type=float, default=0.04)
    ap.add_argument("--shrink-t", type=float, default=0.5)
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    for ds in a.datasets:
        fig_for(ds, a.gamma, a.shrink_t, Path(a.out))


if __name__ == "__main__":
    main()
