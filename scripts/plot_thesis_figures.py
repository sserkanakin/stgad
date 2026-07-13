"""thesis figures for the methodology chapter -> tudelft_thesis/figures/*.pdf.

three figures, the analogues of GAD's fig. 1 computed on OUR fitted priors,
presented energy-first: mode importance is measured by the residual energy a
mode carries, not by its raw precision.

  fig_theta_spectrum.pdf       (a) sorted joint precisions per dataset vs
                               GAD's purely spatial range; (b) cumulative
                               MEASURED residual energy vs precision - shows
                               the data's energy saturates at moderate Theta
  fig_forward_decay.pdf        per-mode mean decay + variance growth vs t
                               under FCPS (and linear ULS for contrast), for
                               modes at the 25/50/75/95th percentiles of the
                               measured energy on metr-la
  fig_forward_convergence.pdf  W1 distance of the noised ensemble to the
                               stationary prior vs heat, from forward_analysis

usage:
    python scripts/plot_thesis_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.residual_dataset import train_residual_sigma
from src.diffusion.forward_fcps import FCPSSchedule, ULSchedule
from src.diffusion.prior import load_prior
from src.utils import paths

OUT = paths.ROOT / "tudelft_thesis" / "figures"
DATASETS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
LABELS = {"metrla": "METR-LA", "pemsbay": "PEMS-BAY", "pems04f": "PEMS04 (flow)",
          "pems04s": "PEMS04 (speed)", "pems08f": "PEMS08 (flow)",
          "pems08s": "PEMS08 (speed)"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _mode_energy(ds: str, pr) -> np.ndarray:
    """measured residual energy per joint mode (unit space, fully observed windows)."""
    z = np.load(paths.ROOT / "results" / "residuals" / ds / "gwn" / "seed42" / "train.npz")
    R = z["residuals"].astype(np.float32)
    m = z["mask"].astype(bool)
    X = R[m.all(axis=(1, 2))][:4000] / float(train_residual_sigma(ds, 42, "gwn"))
    C = np.einsum("ki,bih,hj->bkj", pr.V_S.T, X, pr.V_T)
    return (C ** 2).mean(axis=0).ravel()


def fig_theta_spectrum() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7))
    for ds in DATASETS:
        pr = load_prior(ds, "kronecker", seed=42, backbone="gwn")
        th = pr.theta.ravel()
        order = np.argsort(th)
        axes[0].plot(np.linspace(0, 1, th.size), th[order], lw=1.3,
                     label=LABELS[ds])
        e = _mode_energy(ds, pr)[order]
        axes[1].plot(th[order], np.cumsum(e) / e.sum(), lw=1.3)
    axes[0].axhspan(0.8, 2.8, color="gray", alpha=0.15, lw=0)
    axes[0].text(0.03, 0.95, "GAD (spatial only)", fontsize=7.5,
                 color="dimgray", va="top")
    axes[0].set(yscale="log", xlabel="mode index (sorted)",
                ylabel="joint precision $\\Theta$")
    axes[1].set(xscale="log", xlabel="joint precision $\\Theta$",
                ylabel="cumulative residual energy")
    axes[1].set_ylim(0, 1.02)
    axes[1].axhline(1.0, color="gray", lw=0.6, ls=":")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT / "fig_theta_spectrum.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_theta_spectrum.pdf")


def fig_forward_decay() -> None:
    pr = load_prior("metrla", "kronecker", seed=42, backbone="gwn")
    th = pr.theta.ravel()
    e = _mode_energy("metrla", pr)
    order = np.argsort(th)
    cum = np.cumsum(e[order]) / e.sum()
    # four round precisions from the slowest mode up to Theta = 9; print the
    # measured energy fraction they span (for the caption)
    modes = [float(th.min()), 2.0, 4.5, 9.0]
    names = [f"$\\Theta={m:.1f}$" for m in modes]
    frac = cum[np.searchsorted(th[order], 9.0)]
    print(f"measured energy below Theta=9: {100*frac:.0f}%")
    fcps, uls = FCPSSchedule(), ULSchedule()
    t = np.linspace(1e-4, 1.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for m, name, c in zip(modes, names, colors):
        axes[0].plot(t, np.exp(-fcps.c_bar(t) * m), color=c, lw=1.5, label=name)
        axes[0].plot(t, np.exp(-uls.c_bar(t) * m), color=c, lw=1.0, ls="--",
                     alpha=0.55)
        axes[1].plot(t, 1.0 - np.exp(-2.0 * fcps.c_bar(t) * m), color=c, lw=1.5)
        axes[1].plot(t, 1.0 - np.exp(-2.0 * uls.c_bar(t) * m), color=c, lw=1.0,
                     ls="--", alpha=0.55)
    axes[0].set(xlabel="diffusion time $t$",
                ylabel="signal retained $e^{-\\bar c_t\\Theta}$")
    axes[1].set(xlabel="diffusion time $t$",
                ylabel="noise level $1-e^{-2\\bar c_t\\Theta}$")
    # GAD-style legend: one horizontal strip above both panels, nothing
    # inside the axes. mode entries first, schedule styles last.
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="k", lw=1.5),
                plt.Line2D([], [], color="k", lw=1.0, ls="--", alpha=0.55)]
    labels += ["FCPS", "linear (ULS)"]
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.4,
               handlelength=1.8)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xticks([0.0, 0.5, 1.0])
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT / "fig_forward_decay.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_forward_decay.pdf", "modes:", [round(m, 1) for m in modes])


def fig_forward_convergence() -> None:
    """distance to the stationary prior over diffusion time: flat at the
    sampling-noise floor well before t=1 means the forward process has
    reached the prior."""
    sched = FCPSSchedule()
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    for ds in DATASETS:
        f = paths.ROOT / "results" / "forward_analysis" / f"{ds}.json"
        if not f.exists():
            print("skip (no json):", ds)
            continue
        p = json.loads(f.read_text())
        dist = p["operators"]["normalized"]["distances"]
        cs = np.array(sorted(float(c) for c in dist))
        w1 = np.array([dist[k]["tv"]["wasserstein"]
                       for k in sorted(dist, key=float)])
        t = sched.t_of_c_bar(cs)                 # map heat -> diffusion time
        ax.plot(t, w1 / w1[0], "-o", ms=3.2, lw=1.4, label=LABELS[ds])
    ax.set(yscale="log", xlabel="diffusion time $t$",
           ylabel="distance to stationary prior (rel.)")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlim(0, 1.02)
    ax.annotate("prior reached\n(sampling-noise floor)", xy=(0.97, 3e-3),
                xytext=(0.60, 2.5e-2), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="dimgray"))
    ax.legend(ncol=2, frameon=False, loc="lower left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_forward_convergence.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_forward_convergence.pdf")


def fig_temporal_factor() -> None:
    """the two measured facts the temporal factor encodes, on metr-la:
    (a) per-horizon residual spread sqrt(delta_h) in raw units, growing with
    lead time; (b) the horizon correlation matrix R_H."""
    pr = load_prior("metrla", "kronecker", seed=42, backbone="gwn")
    sig = float(train_residual_sigma("metrla", 42, "gwn"))
    spread = np.sqrt(pr.delta_h) * sig
    h = np.arange(1, pr.H + 1)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))
    axes[0].plot(h, spread, "-o", ms=3.5, lw=1.4)
    axes[0].set(xlabel="lead time $h$", ylabel="residual spread (mph)")
    axes[0].set_xticks([1, 6, 12])
    axes[0].set_ylim(0, None)
    axes[0].grid(alpha=0.25)
    print("R_H min:", pr.R_H.min().round(3))
    im = axes[1].imshow(pr.R_H, vmin=0, vmax=1, cmap="viridis", origin="lower",
                        extent=(0.5, 12.5, 0.5, 12.5))
    axes[1].set(xlabel="lead time $h$", ylabel="lead time $g$")
    axes[1].set_xticks([1, 6, 12])
    axes[1].set_yticks([1, 6, 12])
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="correlation")
    fig.tight_layout()
    fig.savefig(OUT / "fig_temporal_factor.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_temporal_factor.pdf")


def fig_prior_draw() -> None:
    """a measured residual field next to one draw from the fitted prior and
    one from a variance-matched isotropic prior. sensors ordered by the
    fiedler vector so graph-adjacent sensors sit next to each other; the
    window shown is the fully observed one closest to median energy (not
    cherry-picked)."""
    pr = load_prior("metrla", "kronecker", seed=42, backbone="gwn")
    z = np.load(paths.ROOT / "results" / "residuals" / "metrla" / "gwn" / "seed42" / "train.npz")
    R = z["residuals"].astype(np.float64)
    m = z["mask"].astype(bool)
    X = R[m.all(axis=(1, 2))] / float(train_residual_sigma("metrla", 42, "gwn"))
    e = (X ** 2).mean(axis=(1, 2))
    real = X[int(np.argmin(np.abs(e - np.median(e))))]
    rng = np.random.default_rng(0)
    kron = pr.V_S @ (np.sqrt(pr.sigma2) * rng.standard_normal(pr.theta.shape)) @ pr.V_T.T
    iso = float(X.std()) * rng.standard_normal(real.shape)
    assert pr.lam_S[0] <= pr.lam_S[1]                       # ascending -> col 1 = fiedler
    order = np.argsort(pr.V_S[:, 1])
    v = 2.5
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 3.9), sharex=True)
    names = ["measured residual field", "draw from the structured prior",
             "draw from an isotropic prior (matched variance)"]
    for ax, F, name in zip(axes, [real, kron, iso], names):
        im = ax.imshow(F[order].T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                       interpolation="nearest", origin="lower")
        ax.set_ylabel("$h$", fontsize=8.5)
        ax.set_yticks([0, 11])
        ax.set_yticklabels(["1", "12"])
        ax.set_title(name, fontsize=8.5, pad=2)
    axes[-1].set_xlabel("sensor (ordered so graph neighbors are adjacent)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="standardized residual")
    fig.savefig(OUT / "fig_prior_draw.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_prior_draw.pdf")


def print_separability() -> None:
    """nearest-kronecker-product separability test (van loan-pitsianis): the
    empirical joint covariance is rearranged so kronecker structure becomes
    rank-1 structure; the leading singular value's share of the squared
    frobenius norm is the fraction of covariance energy one kronecker term
    explains."""
    print("rank-1 kronecker energy of the empirical residual covariance:")
    for ds in DATASETS:
        z = np.load(paths.ROOT / "results" / "residuals" / ds / "gwn" / "seed42" / "train.npz")
        R = z["residuals"].astype(np.float64)
        m = z["mask"].astype(bool)
        X = (R[m.all(axis=(1, 2))] / float(train_residual_sigma(ds, 42, "gwn")))[:4000]
        B, N, H = X.shape
        Xv = X.transpose(0, 2, 1).reshape(B, H * N)          # vec: node-fast
        C = Xv.T @ Xv / B                                     # second-moment matrix
        Rt = np.stack([C[h * N:(h + 1) * N, g * N:(g + 1) * N].reshape(-1)
                       for h in range(H) for g in range(H)])  # (H^2, N^2)
        ev = np.linalg.eigvalsh(Rt @ Rt.T)
        print(f"  {ds:8s}  {100 * ev[-1] / ev.sum():5.1f}%   (windows: {B})")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    # figures used in the methodology chapter
    fig_forward_decay()
    fig_forward_convergence()
    # numbers used in the chapter tables (fig_temporal_factor / fig_prior_draw
    # were dropped from the thesis; the functions are kept for reference)
    print_separability()
