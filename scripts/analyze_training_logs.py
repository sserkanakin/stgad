"""parse the diffusion training .out logs and plot, per dataset, the iso-vs-kron
curves that explain the val/test puzzle.

the key contrast:
  - val DENOISING loss  -> not comparable across priors (each prior denoises its
    own noise); iso usually looks lower here. this is the "iso wins val" signal.
  - val GENERATIVE metrics (recon gen-MAE, coverage, W1) -> ARE comparable across
    priors (they score the actual reverse-sde forecast). kron should look better
    here, matching the test result.

so a figure with both panels shows: iso only "wins" the incomparable denoising
loss, while kron wins the comparable generative metrics -> consistent val->test.

usage:
    python scripts/analyze_training_logs.py
    python scripts/analyze_training_logs.py --logs logs/diffusion --out results/analysis
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# log line patterns (see scripts/train_diffusion.py prints)
_EPOCH = re.compile(r"epoch\s+(\d+)\s+train=([\d.]+)\s+val=([\d.]+)")
_RECON = re.compile(r"\[recon\] gen mae=([\d.]+) mse=[\d.]+ \(.*?\)\s+WI\d+=([\d.]+)\s+cov=([\d.]+)")
_W1 = re.compile(r"\[w1\].*?mean W1=([\d.]+) ED=([\d.]+)")
# run name -> dataset, prior  (diffusion_<ds>_<prior>_seed<seed>)
_RUN = re.compile(r"diffusion_(metrla|pemsbay|pems08f)_(kronecker|isotropic)_seed(\d+)")


def parse_log(path: str) -> dict:
    """pull the per-epoch train/val loss and the periodic recon/w1 rows."""
    ep, tr, va = [], [], []
    r_ep, r_mae, r_wi, r_cov = [], [], [], []
    w_ep, w_w1, w_ed = [], [], []
    cur = -1
    with open(path, errors="ignore") as f:
        for line in f:
            m = _EPOCH.search(line)
            if m:
                cur = int(m.group(1))
                ep.append(cur); tr.append(float(m.group(2))); va.append(float(m.group(3)))
                continue
            m = _RECON.search(line)
            if m:
                r_ep.append(cur); r_mae.append(float(m.group(1)))
                r_wi.append(float(m.group(2))); r_cov.append(float(m.group(3)))
                continue
            m = _W1.search(line)
            if m:
                w_ep.append(cur); w_w1.append(float(m.group(1))); w_ed.append(float(m.group(2)))
    return dict(ep=ep, tr=tr, va=va, r_ep=r_ep, r_mae=r_mae, r_wi=r_wi, r_cov=r_cov,
                w_ep=w_ep, w_w1=w_w1, w_ed=w_ed)


def latest_per_run(logs_dir: str) -> dict:
    """map (dataset, prior, seed) -> newest .out log (handles flat + subdir layout)."""
    files = glob.glob(os.path.join(logs_dir, "**", "*.out"), recursive=True)
    files += glob.glob(os.path.join(logs_dir, "*.out"))
    by_run: dict = {}
    for f in files:
        m = _RUN.search(f)
        if not m:
            continue
        key = (m.group(1), m.group(2), int(m.group(3)))
        if key not in by_run or os.path.getmtime(f) > os.path.getmtime(by_run[key]):
            by_run[key] = f
    return by_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs/diffusion")
    p.add_argument("--out", default="results/analysis")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    runs = latest_per_run(args.logs)
    if not runs:
        print(f"no diffusion logs found under {args.logs}")
        return
    data = {k: parse_log(v) for k, v in runs.items() if k[2] == args.seed}
    os.makedirs(args.out, exist_ok=True)

    datasets = ["metrla", "pemsbay", "pems08f"]
    col = {"isotropic": "#c0392b", "kronecker": "#2471a3"}
    # rows: val denoising loss (NOT comparable) | gen-MAE | coverage | W1 (comparable)
    panels = [
        ("val denoising loss\n(NOT cross-prior comparable)", "ep", "va", None),
        ("val gen-MAE (comparable)", "r_ep", "r_mae", None),
        ("val coverage (target 0.90)", "r_ep", "r_cov", 0.90),
        ("val W1 (comparable)", "w_ep", "w_w1", None),
    ]
    fig, axes = plt.subplots(len(panels), len(datasets), figsize=(14, 11), squeeze=False)
    for j, ds in enumerate(datasets):
        for i, (label, xk, yk, hline) in enumerate(panels):
            ax = axes[i][j]
            for prior in ("isotropic", "kronecker"):
                d = data.get((ds, prior, args.seed))
                if d and d[xk] and d[yk]:
                    ax.plot(d[xk], d[yk], "-o", ms=3, color=col[prior], label=prior)
            if hline is not None:
                ax.axhline(hline, ls="--", lw=1, color="gray")
            if i == 0:
                ax.set_title(ds, fontweight="bold")
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
            if i == len(panels) - 1:
                ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Training curves (seed {args.seed}): iso vs kron — "
                 "iso wins the incomparable denoising loss, kron wins the comparable "
                 "generative metrics", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(args.out, f"fig_train_curves_seed{args.seed}.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # quick text summary: best val-loss epoch vs best gen-MAE epoch per run
    print("\n=== per-run summary (seed", args.seed, ") ===")
    for ds in datasets:
        for prior in ("isotropic", "kronecker"):
            d = data.get((ds, prior, args.seed))
            if not d or not d["va"]:
                print(f"{ds:8s} {prior:9s}: (no log)")
                continue
            bv = min(range(len(d["va"])), key=lambda i: d["va"][i])
            line = f"{ds:8s} {prior:9s}: best val-loss={d['va'][bv]:.3f}@ep{d['ep'][bv]}"
            if d["r_mae"]:
                bm = min(range(len(d["r_mae"])), key=lambda i: d["r_mae"][i])
                line += f"  best gen-MAE={d['r_mae'][bm]:.3f}@ep{d['r_ep'][bm]}"
            if d["r_cov"]:
                line += f"  last cov={d['r_cov'][-1]:.3f}"
            print(line)


if __name__ == "__main__":
    main()
