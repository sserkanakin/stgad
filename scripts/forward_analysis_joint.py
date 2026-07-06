"""run the joint spatio-temporal spectrum check + save figures.

  python scripts/forward_analysis_joint.py                 # all datasets, tables + figs
  python scripts/forward_analysis_joint.py --dataset metrla # one dataset, table only

answers, per dataset:
  q1  is the residual "mostly low frequency" in the kronecker (space×time) sense
  q2  does the forward drive the joint spectrum onto the stationary distribution

figures mirror the spatial forward_analysis (curves, 2x3 dataset grid), -> figs/:
  joint_spectrum_spatial.png   — energy vs spatial graph-frequency; data curve
                                 sliding onto the prior as c̄ grows (per dataset)
  joint_spectrum_temporal.png  — same, energy vs temporal frequency
  joint_convergence.png        — spectrum distance-to-stationary vs c̄ (spatial +
                                 temporal + joint curves), fraction of start → 0
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.eval.forward_analysis_joint import run, print_report

DATASETS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]

# same okabe-ito palette / labels as the spatial forward_analysis figures
OKABE = {"spatial": "#0072B2", "temporal": "#E69F00", "joint": "#009E73"}
DIST_LABEL = {"spatial": "spatial freq", "temporal": "temporal freq", "joint": "joint (2D)"}


def _figdir():
    from src.utils import paths
    d = paths.RESULTS / "forward_analysis" / "figs"
    os.makedirs(d, exist_ok=True)
    return d


def _fig_spectrum(results: dict, axis: str) -> None:
    """energy-vs-frequency curves, one panel per dataset (2x3), in the style of the
    spatial analysis' `morph` figure: the data spectrum (grey fill) slides through a
    noise ramp onto the prior (black dashed) as c̄ grows. axis = 'spatial'|'temporal'.
    log-y so the actual energy VALUES and where they sit are both readable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = f"_spec_{axis}"
    xlabel = "spatial graph frequency  λ" if axis == "spatial" else "temporal frequency  (sign changes)"
    ramp = plt.cm.Blues(np.linspace(0.35, 0.95, 4))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, ds in zip(axes.ravel(), DATASETS):
        res = results[ds]
        spec = res[key]
        x = res["_lam_s_sorted"] if axis == "spatial" else res["_tfreq_sorted"]
        grid = res["grid"]
        data, prior = spec["data"], spec["prior"]
        pick = [grid[1], grid[3], grid[5], grid[7]]
        # tight y-range: a little padding around the actual curves (no empty decades)
        stack = np.concatenate([data, prior] + [spec[f"{c:.6g}"] for c in pick])
        pos = stack[stack > 0]
        ylo, yhi = pos.min() * 0.3, pos.max() * 3
        ax.fill_between(x, data, ylo, color="0.75", alpha=0.6, lw=0, label="data (c̄=0)")
        for c, col in zip(pick, ramp):
            ax.plot(x, spec[f"{c:.6g}"], color=col, lw=1.6)
        ax.plot(x, prior, color="k", lw=2.0, ls="--", label="prior (target)")
        ax.set_yscale("log")
        ax.set_ylim(ylo, yhi)
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", color="0.92", lw=0.6)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("energy  Σ E[|C|²]")
    from matplotlib.lines import Line2D
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.75", alpha=0.6),
               Line2D([0], [0], color=ramp[0], lw=1.6),
               Line2D([0], [0], color=ramp[-1], lw=1.6),
               Line2D([0], [0], color="k", lw=2.0, ls="--")]
    labels = ["data (c̄=0)", "low noise", "high noise", "prior (target)"]
    axes[0, 2].legend(handles, labels, frameon=False, fontsize=8.5, loc="upper right")
    where = "graph" if axis == "spatial" else "time"
    fig.suptitle(f"energy spectrum over {where} frequency — data sits low and the forward "
                 f"slides it onto the prior", fontsize=13, y=1.0)
    fig.tight_layout()
    out = _figdir() / f"joint_spectrum_{axis}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_convergence(results: dict) -> None:
    """distance of the energy spectrum to the stationary σ²/θ vs c̄, 2x3 dataset grid,
    normalised to the starting distance so every curve shares a 0–1 axis and → 0 means
    the forward reached the stationary distribution. three curves: spatial-marginal,
    temporal-marginal, and the full joint spectrum — mirrors the spatial fig_convergence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, sharey=True)
    for ax, ds in zip(axes.ravel(), DATASETS):
        res = results[ds]
        grid = [r["c_bar"] for r in res["trajectory"]]
        for which in ("spatial", "temporal", "joint"):
            d = np.array([r[f"dist_{which}"] for r in res["trajectory"]])
            d = d / (d[0] + 1e-12)
            ax.plot(grid, d, marker="o", ms=4, lw=1.8, color=OKABE[which], label=DIST_LABEL[which])
        ax.set_xscale("log")
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        ax.axhline(1, color="0.85", lw=0.8, zorder=0)
        ax.grid(True, which="both", color="0.92", lw=0.6)
        ax.set_ylim(-0.05, 1.28)   # headroom: temporal marginal can briefly overshoot start
    for ax in axes[-1]:
        ax.set_xlabel("noise level  c̄")
    for ax in axes[:, 0]:
        ax.set_ylabel("distance to stationary\n(fraction of start)")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("forward drives the energy spectrum onto the stationary distribution σ²/θ",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    out = _figdir() / "joint_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="one dataset (default: all)")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--n-steps", type=int, default=7)
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    dsets = [args.dataset] if args.dataset else DATASETS
    results = {}
    for ds in dsets:
        res = run(ds, n_samples=args.n_samples, n_steps=args.n_steps)
        print_report(res)
        results[ds] = res

    print(f"\n{'='*74}\nsummary — low-freq concentration (energy in lowest-freq quarter)\n{'='*74}")
    print(f"  {'dataset':>10} | {'data':>6} {'prior':>6} {'white':>6} | "
          f"{'data/white':>10} | {'dist→stat @c̄_hi':>15}")
    for ds, res in results.items():
        ll = res["lowlow"]
        print(f"  {ds:>10} | {ll['data']:6.3f} {ll['prior']:6.3f} {ll['white']:6.3f} | "
              f"{ll['data']/ll['white']:10.1f} | {res['trajectory'][-1]['dist_joint']*100:14.2f}%")

    if not args.no_figs and not args.dataset:
        _fig_spectrum(results, "spatial")
        _fig_spectrum(results, "temporal")
        fig_convergence(results)


if __name__ == "__main__":
    main()
