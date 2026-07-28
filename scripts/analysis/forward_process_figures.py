r"""Two verification figures for the forward process, on the prior the thesis uses.

The arm plotted is the residual campaign's structured prior:
    Sigma = Sigma_T (x) Sigma_S,  Sigma_S = (L_road + gamma I)^-1, gamma = 0.8 (GAD-fixed),
    Sigma_T = the empirical horizon covariance (ridge rho = 0.1),
with GAD's FCPS schedule at c_min = 0.1, c_0 = 7.0, alpha = 4.0.

FIGURE 1  stationary.png -- "the stationary law is the one we designed, and here is
                             exactly how its draws compare to the data being corrupted"
  (a) per-mode variance measured from the ACTUAL forward sampler at c_bar_hi against the
      closed form sigma^2/theta. On the diagonal = the process lands on N(0, Sigma).
  (b) convergence: relative error of the per-mode variance along the forward trajectory,
      with c_bar_hi marked. Shows it converges, and that c_bar_hi is far enough out.
  (c) the cascade: per-mode variance at a ladder of c_bar, from the data's own spectrum
      up to the stationary one.
  (d) TEMPORAL factor. Sigma_T is FIT to the residuals, so per-horizon variance of the
      draws should track the data closely -- this is the panel that can match.
  (e) SPATIAL factor. Sigma_S = (L+gamma I)^-1 is IMPOSED, not fit, so it is a smooth
      heat-kernel envelope through a ragged empirical spectrum. Showing this honestly
      matters: it is the quantitative statement of how much graph structure the prior
      actually asserts.
  (f) the node-domain reading of (e): correlation against graph distance.

FIGURE 2  gad_forward.png -- GAD Fig.1 on our theta, FCPS vs ULS
  (a) c_bar(t) for both schedules. They are matched in TOTAL heat: ULS slope = 2*c_0
      gives c_bar(T) = c_0, so any difference is schedule SHAPE, not budget.
  (b) mean decay      e^(-theta*c_bar_t)              per eigenvalue
  (c) covariance growth (sigma^2/theta)(1 - e^(-2*theta*c_bar_t)) per eigenvalue
  Modes are coloured by theta across the whole spectrum, so the eigenvalue CASCADE
  (high-theta modes erased first, the slow graph modes last) is visible rather than
  sampled at five points.

usage: .venv/bin/python scripts/analysis/forward_process_figures.py \
           [--dataset pems08f] [--backbone gwn]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.colors import LogNorm                                 # noqa: E402
from matplotlib.cm import ScalarMappable                              # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.adjacency import load_adjacency                          # noqa: E402
from src.diffusion.forward_fcps import (FCPSSchedule, ULSchedule,      # noqa: E402
                                        noise_level_bounds, sample_prior,
                                        sample_R_t_from_c_bar)
from src.diffusion.prior import _load_residuals_train, load_prior      # noqa: E402
from src.utils import paths                                           # noqa: E402

OUT = ROOT / "results" / "analysis" / "forward_process"


def mode_var(X, prior):
    """per-mode variance of (B,N,H) samples in the joint V_S (x) V_T eigenbasis."""
    C = np.einsum("ki,bih,hj->bkj", prior.V_S.T, X, prior.V_T, optimize=True)
    return (C ** 2).mean(0)                                            # (N, H)


def graph_mode_var(X, prior):
    """variance per GRAPH eigenmode (averaged over horizon): the spatial spectrum."""
    C = np.einsum("ki,bih->bkh", prior.V_S.T, X, optimize=True)
    return (C ** 2).mean(axis=(0, 2))                                  # (N,)


def hop_distance(A, max_hops=6):
    """unweighted hop distance from the adjacency, capped (BFS by boolean powers)."""
    Adj = (np.asarray(A) > 0)
    np.fill_diagonal(Adj, False)
    N = Adj.shape[0]
    D = np.full((N, N), np.inf)
    np.fill_diagonal(D, 0)
    reach = np.eye(N, dtype=bool)
    frontier = np.eye(N, dtype=bool)
    for h in range(1, max_hops + 1):
        frontier = (frontier @ Adj) & ~reach
        if not frontier.any():
            break
        D[frontier] = h
        reach |= frontier
    return D


def corr_by_hop(X, D, max_hops=5):
    """mean |correlation| between node pairs at each hop distance."""
    F = X.transpose(1, 0, 2).reshape(X.shape[1], -1)                   # (N, B*H)
    F = F - F.mean(1, keepdims=True)
    s = F.std(1)
    s[s < 1e-12] = 1e-12
    C = (F @ F.T) / (F.shape[1] * np.outer(s, s))
    out = []
    for h in range(1, max_hops + 1):
        m = (D == h)
        out.append(float(np.abs(C[m]).mean()) if m.any() else np.nan)
    return out


def figure_stationary(prior, sched, R, A, ds, bb, rng):
    c_lo, c_hi = noise_level_bounds(prior)
    B = min(512, R.shape[0])
    R0 = R[:B]

    # (a) sampler vs closed form at c_bar_hi
    Xhi = sample_R_t_from_c_bar(R0, c_hi, prior, rng)
    v_meas = mode_var(Xhi, prior).ravel()
    v_form = prior.sigma2.ravel()

    # (b) convergence along the trajectory
    grid = np.geomspace(max(c_lo, 1e-3), c_hi * 4, 22)
    err = []
    for cb in grid:
        v = mode_var(sample_R_t_from_c_bar(R0, cb, prior, rng), prior)
        err.append(float(np.abs(v - prior.sigma2).sum() / prior.sigma2.sum()))

    # (c) the cascade: per-mode variance at a ladder of noise levels
    ladder = [c_hi / 64, c_hi / 16, c_hi / 4, c_hi]
    cascade = [np.sort(mode_var(sample_R_t_from_c_bar(R0, cb, prior, rng), prior).ravel())[::-1]
               for cb in ladder]

    # (d,e,f) prior draws vs real residuals
    draws = sample_prior((B,), prior, rng)
    gm_d, gm_r = graph_mode_var(draws, prior), graph_mode_var(R0, prior)
    th_d, th_r = (draws ** 2).mean(axis=(0, 1)), (R0 ** 2).mean(axis=(0, 1))
    D = hop_distance(A)
    hops = list(range(1, 6))
    c_d, c_r = corr_by_hop(draws, D), corr_by_hop(R0, D)

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.6))

    a = ax[0, 0]
    lim = [min(v_form.min(), v_meas.min()) * 0.6, max(v_form.max(), v_meas.max()) * 1.6]
    a.plot(lim, lim, "k--", lw=1, zorder=1, label="y = x")
    a.scatter(v_form, v_meas, s=7, alpha=0.35, edgecolors="none", zorder=2)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
    rel = float(np.abs(v_meas - v_form).sum() / v_form.sum())
    a.set_xlabel(r"closed form  $\sigma^2/\theta$")
    a.set_ylabel(r"measured from the sampler at $\bar c_{hi}$")
    a.set_title(f"(a) the forward lands on $N(0,\\Sigma)$\nall {v_form.size} modes, "
                f"total rel. error {rel:.3f}")
    a.legend(fontsize=8); a.grid(alpha=0.25, which="both")

    b = ax[0, 1]  # noqa: E501
    b.loglog(grid, err, "o-", ms=3.5, lw=1.4)
    b.axvline(c_hi, color="crimson", ls="--", lw=1.2,
              label=rf"$\bar c_{{hi}} = 5/\theta_{{min}} = {c_hi:.2f}$")
    b.set_xlabel(r"noise level $\bar c$")
    b.set_ylabel("rel. error vs stationary variance")
    b.set_title("(b) convergence to the stationary law")
    b.legend(fontsize=8); b.grid(alpha=0.25, which="both")

    c = ax[0, 2]
    rank = np.arange(1, prior.theta.size + 1)
    for cb, cv in zip(ladder, cascade):
        c.loglog(rank, cv, lw=1.3, label=rf"$\bar c={cb:.2f}$")
    c.loglog(rank, np.sort(prior.sigma2.ravel())[::-1], "k--", lw=1.6,
             label=r"stationary $\sigma^2/\theta$")
    c.set_xlabel("mode rank (by variance)"); c.set_ylabel("variance")
    c.set_title("(c) the cascade to the stationary spectrum")
    c.legend(fontsize=7); c.grid(alpha=0.25, which="both")

    d = ax[1, 0]
    h = np.arange(1, len(th_r) + 1)
    d.plot(h, th_r, "o-", ms=4, lw=1.6, label="real residuals")
    d.plot(h, th_d, "s--", ms=4, lw=1.6, label="draws from the prior")
    d.set_xlabel("horizon step"); d.set_ylabel("variance")
    ratio = float(th_d.mean() / th_r.mean())
    d.set_title(r"(d) temporal factor $\Sigma_T$ is FIT — shape tracks the data"
                f"\n(level {ratio:.2f}x: the $\\rho$=0.1 ridge + trace matching)")
    d.set_xticks(h); d.legend(fontsize=8); d.grid(alpha=0.25)

    e = ax[1, 1]
    k = np.arange(len(gm_r))
    e.semilogy(k, gm_r, lw=1.1, alpha=0.8, label="real residuals")
    e.semilogy(k, gm_d, lw=2.0, ls="--", label="draws from the prior")
    e.set_xlabel(r"graph eigenmode index (by $\lambda$)")
    e.set_ylabel("variance")
    e.set_title(r"(e) spatial factor $\Sigma_S=(L+\gamma I)^{-1}$ is IMPOSED —"
                "\na smooth envelope, not a fit")
    e.legend(fontsize=8); e.grid(alpha=0.25, which="both")

    f = ax[1, 2]
    w = 0.38
    f.bar(np.array(hops) - w / 2, c_r, w, label="real residuals")
    f.bar(np.array(hops) + w / 2, c_d, w, label="draws from the prior")
    f.set_xlabel("graph distance (hops)")
    f.set_ylabel("mean |correlation|")
    f.set_title("(f) node domain: the prior asserts more\n1-hop coupling than the data has")
    f.set_xticks(hops); f.legend(fontsize=8); f.grid(alpha=0.25, axis="y")

    fig.suptitle(
        f"Forward process hits the designed stationary law — {ds}/{bb}, "
        rf"$\Sigma_T\otimes\Sigma_S$, $\Sigma_S=(L+{prior.gamma:g}I)^{{-1}}$",
        fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"stationary_{ds}_{bb}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p, rel, err[-1], c_hi


def figure_gad_forward(prior, sched, ds, bb):
    """GAD Fig.1 on our theta: FCPS vs ULS at MATCHED total heat."""
    c0 = sched.c_0
    uls = ULSchedule(slope=2 * c0)                    # c_bar(T) = c_0, same budget
    t = np.linspace(0, 1, 400)
    cbF, cbU = sched.c_bar(t), uls.c_bar(t)

    th = np.unique(prior.theta.ravel())
    # a log-spaced ladder across the spectrum, so the cascade is visible
    idx = np.unique(np.geomspace(1, len(th), 26).astype(int) - 1)
    reps = th[idx]
    norm = LogNorm(vmin=reps.min(), vmax=reps.max())
    cmap = plt.get_cmap("viridis")
    s2 = prior.sigma2_scale

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6),
                           gridspec_kw={"width_ratios": [0.85, 1, 1]})

    ax[0].plot(t, cbF, "-", lw=2, color="k", label="FCPS")
    ax[0].plot(t, cbU, ":", lw=2, color="k", label="ULS")
    ax[0].axhline(c0, color="crimson", ls="--", lw=1,
                  label=rf"$c_0={c0:g}$ (same total heat)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$\bar c_t$")
    ax[0].set_title("(a) the two schedules, matched budget")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    for thv in reps:
        col = cmap(norm(thv))
        ax[1].plot(t, np.exp(-thv * cbF), "-", lw=1.5, color=col)
        ax[1].plot(t, np.exp(-thv * cbU), ":", lw=1.1, color=col)
        ax[2].plot(t, (s2 / thv) * (1 - np.exp(-2 * thv * cbF)), "-", lw=1.5, color=col)
        ax[2].plot(t, (s2 / thv) * (1 - np.exp(-2 * thv * cbU)), ":", lw=1.1, color=col)

    ax[1].set_title(r"(b) mean decay  $e^{-\theta \bar c_t}$")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("signal retained"); ax[1].set_ylim(-0.02, 1.02)
    ax[2].set_title(r"(c) covariance growth  $\frac{\sigma^2}{\theta}(1-e^{-2\theta \bar c_t})$")
    ax[2].set_xlabel("t"); ax[2].set_ylabel("noise injected"); ax[2].set_yscale("log")
    for a in ax[1:]:
        a.grid(alpha=0.25)
    ax[1].plot([], [], "-", color="0.3", lw=1.5, label="FCPS")
    ax[1].plot([], [], ":", color="0.3", lw=1.1, label="ULS")
    ax[1].legend(fontsize=8, loc="upper right")

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax[2], pad=0.02)
    cb.set_label(r"mode precision $\theta$  (slow graph modes at the bottom)")

    fig.suptitle(
        f"Eigenvalue cascade under FCPS vs ULS — {ds}/{bb}, "
        rf"stiffness $\theta_{{max}}/\theta_{{min}}={prior.theta.max()/prior.theta.min():.0f}$; "
        "FCPS holds the slow modes longer at the same total heat", fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"gad_forward_{ds}_{bb}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pems08f")
    ap.add_argument("--backbone", default="gwn")
    ap.add_argument("--seed", type=int, default=42)
    # the residual campaign's locked values; --gamma None -> prior.py's GAMMA_DEFAULT=0.8
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--c-0", type=float, default=7.0)
    ap.add_argument("--c-min", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--norm", default="pernode")
    args = ap.parse_args()

    ds, bb = args.dataset, args.backbone
    prior = load_prior(ds, "kronecker", seed=args.seed, temporal="empirical",
                       spatial="structured", spatial_graph="road", gamma=args.gamma,
                       backbone=bb, norm=args.norm)
    sched = FCPSSchedule(c_min=args.c_min, c_0=args.c_0, alpha=args.alpha)
    R, mask = _load_residuals_train(ds, args.seed, bb, norm=args.norm)
    R = np.where(mask, R, 0.0)
    A = load_adjacency(ds)
    if isinstance(A, tuple):
        A = A[0]
    rng = np.random.default_rng(0)

    print(f"=== {ds}/{bb}  gamma={prior.gamma:g}  c_0={sched.c_0:g} "
          f"alpha={args.alpha:g} c_min={args.c_min:g}  norm={args.norm} ===")
    print(f"    N={prior.N} H={prior.H}  theta in [{prior.theta.min():.3f}, "
          f"{prior.theta.max():.2f}]  stiffness {prior.theta.max()/prior.theta.min():.0f}")

    p1, rel, tail, c_hi = figure_stationary(prior, sched, R, A, ds, bb, rng)
    print(f"    [stationary] rel err at c_hi={c_hi:.2f}: {rel:.4f}   (tail {tail:.4f})")
    print(f"    wrote {p1.relative_to(ROOT)}")
    p2 = figure_gad_forward(prior, sched, ds, bb)
    print(f"    wrote {p2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
