"""visual explanations of the forward-analysis result.

three figures, saved to results/forward_analysis/figs/:
  1. convergence — W1-to-prior vs c̄ (as a fraction of the starting distance);
     every metric on every dataset decays to ~0 ⇒ the forward hits the prior.
  2. morph — the Total-Variation *distribution* sliding off the data and onto the
     prior as c̄ grows (kde overlay).
  3. snapshot — one residual field (metr-la) noised step by step into a draw from
     the prior; the structure dissolves into the prior's correlated texture.

the per-sample metric arrays (the slow part) are cached to feats/<ds>.npz, so
re-rendering is instant. metrics come from gad's exact signal_metrics code.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from src.diffusion.prior import load_prior, _load_residuals_train
from src.diffusion import forward_fcps as ff
from src.utils import paths
from src.eval.forward_analysis import build_operators, _features, METRICS

DATASETS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
OPERATORS = ["normalized", "combinatorial"]

# okabe-ito colourblind-safe categorical order (fixed, not cycled)
OKABE = {"tv": "#0072B2", "sc": "#E69F00", "degcorr": "#009E73"}
METRIC_LABEL = {"tv": "Total Variation", "sc": "Spectral Centroid", "degcorr": "Degree Corr."}

FIGDIR = paths.RESULTS / "forward_analysis" / "figs"
FEATDIR = paths.RESULTS / "forward_analysis" / "feats"


# ─── feature cache (per-sample metric scalars) ───────────────────────────────


def get_feats(dataset: str, seed: int = 42, backbone: str = "gwn",
              n_samples: int = 512, n_steps: int = 7) -> dict:
    """per-sample metric arrays at each noise level, cached to npz.
    returns {grid, c_lo, c_hi, theta_min, levels, feats[op][metric][level] -> (B,)}."""
    cache = FEATDIR / f"{dataset}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["payload"].item()

    prior = load_prior(dataset, prior="kronecker", seed=seed,
                       spatial="structured", backbone=backbone)
    R, _ = _load_residuals_train(dataset, seed=seed, backbone=backbone)
    rng = np.random.default_rng(0)
    idx = rng.choice(R.shape[0], size=min(n_samples, R.shape[0]), replace=False)
    R0 = R[idx]
    B = R0.shape[0]

    c_lo, c_hi = ff.noise_level_bounds(prior)
    grid = ff.reverse_c_bar_grid(c_lo, c_hi, n_steps)[::-1]
    ops = build_operators(dataset)
    R_prior = ff.sample_prior((B,), prior, rng)

    levels = ["data"] + [f"{c:.6g}" for c in grid] + ["prior"]
    fields = {"data": R0, "prior": R_prior}
    for c in grid:
        fields[f"{c:.6g}"] = ff.sample_R_t_from_c_bar(R0, float(c), prior, rng)

    feats = {}
    for op_name, op in ops.items():
        feats[op_name] = {m: {} for m in METRICS}
        for lvl, X in fields.items():
            F = _features(X, op)
            for m in METRICS:
                feats[op_name][m][lvl] = F[m].numpy()

    payload = {"grid": [float(c) for c in grid], "c_lo": float(c_lo), "c_hi": float(c_hi),
               "theta_min": float(prior.theta.min()), "levels": levels, "feats": feats}
    os.makedirs(FEATDIR, exist_ok=True)
    np.savez(cache, payload=np.array(payload, dtype=object))
    return payload


# ─── figure 1: convergence curves ────────────────────────────────────────────


def fig_convergence(operator: str = "normalized", metrics=("tv", "sc")) -> None:
    """W1-to-prior vs c̄, normalised to the starting (loosest-noise) distance so the
    metrics and six datasets share a 0–1 axis. → 0 means the forward converged.
    degree-correlation is omitted here: it is ≈0 in both the residuals and the prior
    (residuals carry no degree signal), so its normalised curve is pure noise — see
    the summary table for its (tiny) absolute distances."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, sharey=True)
    for ax, ds in zip(axes.ravel(), DATASETS):
        res = json.load(open(paths.RESULTS / "forward_analysis" / f"{ds}.json"))
        op = res["operators"][operator]
        grid = res["grid"]
        for m in metrics:
            w = np.array([op["distances"][str(float(c))][m]["wasserstein"] for c in grid])
            w = w / (w[0] + 1e-12)
            ax.plot(grid, w, marker="o", ms=4, lw=1.8, color=OKABE[m], label=METRIC_LABEL[m])
        ax.set_xscale("log")
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        ax.grid(True, which="both", color="0.92", lw=0.6)
        ax.set_ylim(-0.05, 1.08)
    for ax in axes[-1]:
        ax.set_xlabel("noise level  c̄")
    for ax in axes[:, 0]:
        ax.set_ylabel("W1 to prior\n(fraction of start)")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(f"forward process drives the graph-signal metrics onto the prior  "
                 f"(operator: {operator})", fontsize=13, y=1.0)
    fig.tight_layout()
    out = FIGDIR / f"convergence_{operator}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ─── figure 2: distribution morph (TV) ───────────────────────────────────────


def _kde_curve(vals: np.ndarray, xs: np.ndarray) -> np.ndarray:
    vals = vals[np.isfinite(vals)]
    if vals.std() < 1e-9:
        return np.zeros_like(xs)
    return gaussian_kde(vals)(xs)


def fig_morph(metric: str = "tv", operator: str = "normalized") -> None:
    """the metric *distribution* itself walking from data onto prior as c̄ grows."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    # sequential single-hue ramp (light→dark blue) for increasing noise level
    ramp = plt.cm.Blues(np.linspace(0.35, 0.95, 4))
    for ax, ds in zip(axes.ravel(), DATASETS):
        p = get_feats(ds)
        F = p["feats"][operator][metric]
        grid = p["grid"]
        data, prior = F["data"], F["prior"]
        lo = min(np.nanpercentile(data, 1), np.nanpercentile(prior, 1))
        hi = max(np.nanpercentile(data, 99), np.nanpercentile(prior, 99))
        xs = np.linspace(lo, hi, 400)
        # data: filled grey reference
        ax.fill_between(xs, _kde_curve(data, xs), color="0.75", alpha=0.6, lw=0, label="data (c̄=0)")
        # four increasing noise levels
        pick = [grid[1], grid[3], grid[5], grid[7]]
        for c, col in zip(pick, ramp):
            ax.plot(xs, _kde_curve(F[f"{c:.6g}"], xs), color=col, lw=1.6)
        # prior: black dashed target
        ax.plot(xs, _kde_curve(prior, xs), color="k", lw=2.0, ls="--", label="prior (target)")
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.set_yticks([])
        ax.grid(True, axis="x", color="0.92", lw=0.6)
    for ax in axes[-1]:
        ax.set_xlabel(f"{METRIC_LABEL[metric]}  (per-sample)")
    # a single shared legend: data, noise ramp, prior
    from matplotlib.lines import Line2D
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.75", alpha=0.6),
               Line2D([0], [0], color=ramp[0], lw=1.6),
               Line2D([0], [0], color=ramp[-1], lw=1.6),
               Line2D([0], [0], color="k", lw=2.0, ls="--")]
    labels = ["data (c̄=0)", "low noise", "high noise", "prior (target)"]
    axes[0, 2].legend(handles, labels, frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(f"the {METRIC_LABEL[metric]} distribution slides off the data and onto the prior "
                 f"(operator: {operator})", fontsize=13, y=1.0)
    fig.tight_layout()
    out = FIGDIR / f"morph_{metric}_{operator}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ─── figure 3: signal snapshot ───────────────────────────────────────────────


def fig_snapshot(dataset: str = "metrla", seed: int = 42, backbone: str = "gwn") -> None:
    """one residual field noised step by step into a draw from the prior."""
    prior = load_prior(dataset, prior="kronecker", seed=seed, spatial="structured", backbone=backbone)
    R, _ = _load_residuals_train(dataset, seed=seed, backbone=backbone)
    rng = np.random.default_rng(1)
    R0 = R[rng.choice(R.shape[0], size=1)]                 # one field (1,N,H)
    c_lo, c_hi = ff.noise_level_bounds(prior)
    grid = ff.reverse_c_bar_grid(c_lo, c_hi, 7)[::-1]
    picks = [("data", R0), (f"c̄={grid[3]:.2g}", ff.sample_R_t_from_c_bar(R0, float(grid[3]), prior, rng)),
             (f"c̄={grid[5]:.2g}", ff.sample_R_t_from_c_bar(R0, float(grid[5]), prior, rng)),
             (f"c̄={grid[7]:.2g}", ff.sample_R_t_from_c_bar(R0, float(grid[7]), prior, rng)),
             ("prior", ff.sample_prior((1,), prior, rng))]
    vmax = np.nanpercentile(np.abs(picks[-1][1]), 99)
    fig, axes = plt.subplots(1, len(picks), figsize=(13, 5.2), sharey=True)
    for ax, (title, X) in zip(axes, picks):
        im = ax.imshow(X[0], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("horizon h")
    axes[0].set_ylabel("node (sensor)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="residual (unit space)")
    fig.suptitle(f"{dataset}: a residual field dissolving into a prior draw as noise grows",
                 fontsize=13, y=1.0)
    out = FIGDIR / f"snapshot_{dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    for ds in DATASETS:                    # build/refresh feature caches first
        get_feats(ds)
        print(f"feats ready: {ds}")
    for op in OPERATORS:
        fig_convergence(op)
    fig_morph("tv", "normalized")
    fig_morph("tv", "combinatorial")
    fig_snapshot("metrla")
    print("all figures done")


if __name__ == "__main__":
    main()
