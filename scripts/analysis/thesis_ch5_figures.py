"""figures for the experiments chapter, built from the locked test evals.

three figures only, each one carrying a claim the chapter actually makes:
  f_law         structured-vs-isotropic margin per backbone, CRPS and MAE,
                under both conditionings -- the ladder result
  f_reliability coverage against nominal level before recalibration
  (the residual-structure figure is produced by paper_figures.py and is
   arm-independent, so it is not rebuilt here)

all numbers are seed means of uncalibrated test metrics under the locked
protocol (best_crps checkpoint, 50 steps, k=100, zero targets masked).
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results"
OUT = ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DS = ["pems08f", "pems08s", "pems04f", "pems04s"]
BB = [("mlp", "linear"), ("gcn", "GCN"), ("gwn", "GWN")]
BLUE, GRAY = "#1f6fb4", "#8a8a80"

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
})


def cells(arm, sub, ds, bb):
    """seed-keyed metrics for one cell."""
    out = {}
    pat = f"{arm}/{ds}/{bb}/{sub}/seed*/eval/test_metrics_bestcrps_ancestral50.json"
    for p in sorted((ROOT / "20260721_ladder").glob(pat)):
        d = json.load(open(p))
        out[p.parts[-3]] = d
    return out


def margin(ds, bb, metric, past):
    """percent improvement of the structured prior over the isotropic control,
    averaged over the seeds both arms share."""
    a, b = ("iso-past", "emp-past") if past else ("iso", "emp")
    I, E = cells(a, "isotropic", ds, bb), cells(b, "kronecker", ds, bb)
    s = sorted(set(I) & set(E))
    if not s:
        return np.nan
    iv = np.mean([I[k][metric] for k in s])
    ev = np.mean([E[k][metric] for k in s])
    return 100.0 * (iv - ev) / iv


def fig_law():
    """margin of the structured prior over the isotropic control, per backbone,
    at the main operating point. one panel per metric; each dataset is a point,
    the line is their mean."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
    x = np.arange(3)
    mk = ["o", "s", "^", "D"]
    for ax, metric, title in [(axes[0], "mae", "ensemble-mean MAE"),
                              (axes[1], "crps", "CRPS")]:
        M = np.array([[margin(ds, bb, metric, True) for bb, _ in BB] for ds in DS])
        for j, ds in enumerate(DS):
            ax.plot(x, M[j], mk[j], color=GRAY, ms=4, alpha=0.75,
                    label=ds.upper() if metric == "mae" else None)
        ax.plot(x, np.nanmean(M, axis=0), "-", color=BLUE, lw=2.2, zorder=3,
                label="mean" if metric == "mae" else None)
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in BB])
        ax.set_title(title, fontsize=9); ax.set_xlabel("backbone")
    axes[0].set_ylabel("structured better $\\rightarrow$ (percent)")
    axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"f_law.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/f_law.pdf")


def fig_reliability(ds="pems08f", bb="gwn"):
    fig, ax = plt.subplots(figsize=(4.0, 3.1))
    for arm, sub, col, lbl in [("iso-past", "isotropic", GRAY, "isotropic"),
                               ("emp-past", "kronecker", BLUE, "structured")]:
        p = ROOT / "20260721_ladder" / arm / ds / bb / sub / "seed42" / "eval" / \
            "test_diag_bestcrps_ancestral50.npz"
        if not p.exists():
            continue
        d = np.load(p)
        lv, cv = d["reliab_levels"], d["reliab_cov"]
        ax.plot(lv, cv, "-o", color=col, lw=1.8, ms=3.5, label=lbl)
    ax.plot([0, 1], [0, 1], "--", color="black", lw=0.9, alpha=0.6,
            label="perfect calibration")
    ax.set_xlabel("nominal level")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"f_reliability.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/f_reliability.pdf")


if __name__ == "__main__":
    print("building chapter-5 figures...")
    fig_law()
    fig_reliability()
    print("done.")
