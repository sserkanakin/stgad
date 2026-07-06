"""per-mode (GAD-style) forward analysis + figures: do the modes mix, is the
variance right, do any modes disappear?

  python scripts/forward_analysis_modes.py                 # all datasets, tables + figs
  python scripts/forward_analysis_modes.py --dataset metrla # one dataset, table only

figures -> results/forward_analysis/figs/:
  modes_mixing.png    — mixing fraction 1-exp(-2θc̄) per θ-group vs c̄; every group
                        reaches 1 by c̄_hi (stiff early, slow last) => modes get mixed
  modes_variance.png  — empirical marginal variance / stationary σ²/θ per θ-group,
                        relaxing onto 1.0 => the variance is right
  modes_endstate.png  — per-mode variance at c̄_hi vs stationary σ²/θ, log-log on y=x,
                        across the full θ range => no mode collapses / disappears
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.eval.forward_analysis_modes import run, print_report

DATASETS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
N_GROUPS = 6            # θ-quantile groups shown as curves (stiff → slow)


def _figdir():
    from src.utils import paths
    d = paths.RESULTS / "forward_analysis" / "figs"
    os.makedirs(d, exist_ok=True)
    return d


def _theta_groups(theta: np.ndarray, n: int):
    """equal-count groups of modes ordered by precision θ (ascending = slow→stiff).
    returns list of (index_array, median_theta)."""
    order = np.argsort(theta)
    groups = np.array_split(order, n)
    return [(g, float(np.median(theta[g]))) for g in groups]


def _color_norm(res_all: dict):
    """shared log-θ color scale across datasets."""
    import matplotlib.colors as mc
    lo = min(r["theta_min"] for r in res_all.values())
    hi = max(r["theta_max"] for r in res_all.values())
    return mc.LogNorm(vmin=lo, vmax=hi)


def _grid_fig(results: dict, kind: str) -> None:
    """kind in {'mixing','variance'}: 2x3 dataset grid, one curve per θ-group,
    coloured by θ (viridis). markers = empirical samples, thin line = closed-form."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    norm = _color_norm(results)
    cmap = plt.cm.viridis
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, sharey=(kind == "mixing"))
    for ax, ds in zip(axes.ravel(), DATASETS):
        res = results[ds]
        theta, grid = res["_theta"], res["grid"]
        for g, th_med in _theta_groups(theta, N_GROUPS):
            col = cmap(norm(th_med))
            if kind == "mixing":
                y_emp = res["_mix"][:, g].mean(1)                 # analytic already
                ax.plot(grid, y_emp, marker="o", ms=3.5, lw=1.6, color=col)
            else:
                y_emp = (res["_var_emp"][:, g] / res["_sig2"][g]).mean(1)
                y_ana = (res["_var_ana"][:, g] / res["_sig2"][g]).mean(1)
                ax.plot(grid, y_emp, marker="o", ms=3.5, lw=0, color=col)
                ax.plot(grid, y_ana, lw=1.4, color=col)
        ax.axvline(res["c_hi"], color="0.6", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", color="0.92", lw=0.6)
        if kind == "mixing":
            ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
            ax.set_ylim(-0.03, 1.05)
        else:
            ax.axhline(1.0, color="k", lw=1.0, ls="--", zorder=0)
            ax.set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("noise level  c̄")
    ylab = ("mixing fraction  1-e^(-2θc̄)" if kind == "mixing"
            else "empirical variance / stationary σ²/θ")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylab)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("mode precision θ  (small = slow/low-freq → large = stiff/high-freq)")
    title = ("modes mix by c̄_hi — stiff modes early, the slowest mode last  (dotted = c̄_hi)"
             if kind == "mixing" else
             "every mode's variance relaxes onto the stationary σ²/θ (=1)  "
             "(dots = sampled, line = closed-form)")
    fig.suptitle(title, fontsize=13, y=1.0)
    out = _figdir() / f"modes_{kind}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_endstate(results: dict) -> None:
    """2x3 grid: per-mode empirical variance at c̄_hi vs stationary σ²/θ, log-log.
    points on y=x => variance is right; the spread down to tiny (nonzero) σ²/θ shows
    no mode has vanished."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    for ax, ds in zip(axes.ravel(), DATASETS):
        res = results[ds]
        sig2 = res["_sig2"]
        var_hi = res["_var_emp"][-1]
        lo = min(sig2.min(), var_hi.min()) * 0.6
        hi = max(sig2.max(), var_hi.max()) * 1.6
        ax.plot([lo, hi], [lo, hi], color="k", lw=1.2, ls="--", zorder=1, label="y = x (right variance)")
        sc = ax.scatter(sig2, var_hi, c=np.log10(res["_theta"]), cmap="viridis",
                        s=8, alpha=0.5, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", color="0.92", lw=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("stationary variance  σ²/θ")
    for ax in axes[:, 0]:
        ax.set_ylabel("sampled variance at c̄_hi")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("end of forward: every mode lands on its stationary variance, "
                 "down to the smallest (nonzero) — no mode disappears", fontsize=13, y=1.0)
    fig.tight_layout()
    out = _figdir() / "modes_endstate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
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

    print(f"\n{'='*74}\nsummary — modes mix onto the right variance (none disappear)\n{'='*74}")
    print(f"  {'dataset':>10} | {'slow-mode mix @c̄_hi':>19} | {'within 10%':>10} | "
          f"{'min σ²/θ':>10} | {'θ range':>10}")
    for ds, res in results.items():
        print(f"  {ds:>10} | {res['slow_mix_at_hi']:19.4f} | {res['within10_at_hi']*100:9.1f}% | "
              f"{res['sig2_min']:10.3g} | {res['theta_max']/res['theta_min']:8.0f}x")

    if not args.no_figs and not args.dataset:
        _grid_fig(results, "mixing")
        _grid_fig(results, "variance")
        fig_endstate(results)


if __name__ == "__main__":
    main()
