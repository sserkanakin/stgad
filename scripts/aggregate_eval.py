"""aggregate the per-seed test_metrics.json of one experiment into mean +/- std
tables (and a figure) comparing kronecker vs isotropic.

reports the metrics the supervisor asked to track: MAE, MSE, RMSE, Winkler,
coverage, CRPS, and the distribution distances W1 (sliced wasserstein-1) and
energy distance. each is averaged over seeds with its std.

usage:
    python scripts/aggregate_eval.py --exp 20260610
    python scripts/aggregate_eval.py --exp 20260610 --out results/analysis
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["metrla", "pemsbay", "pems08f"]
PRIORS = ["kronecker", "isotropic"]
# (json key, label, lower-is-better?). coverage is "closer to 1-alpha is better".
METRICS = [
    ("mae", "MAE", True), ("rmse", "RMSE", True), ("mse", "MSE", True),
    ("winkler", "Winkler", True), ("coverage", "Coverage", None),
    ("crps", "CRPS", True), ("w1", "W1", True), ("energy", "Energy", True),
]


def collect(exp: str) -> dict:
    """(dataset, prior) -> {metric: [per-seed values]}. derives rmse from mse."""
    out: dict = defaultdict(lambda: defaultdict(list))
    seeds: dict = defaultdict(list)
    for f in glob.glob(f"results/eval/{exp}/*/*/seed*/test_metrics.json"):
        d = json.load(open(f))
        key = (d["dataset"], d["prior"])
        for m, _, _ in METRICS:
            if m == "rmse":
                out[key]["rmse"].append(float(d["mse"]) ** 0.5)
            elif m in d:
                out[key][m].append(float(d[m]))
        seeds[key].append(d.get("seed"))
    return out, seeds


def fmt(vals: list) -> str:
    if not vals:
        return "  n/a "
    a = np.array(vals, dtype=float)
    return f"{a.mean():.3f}±{a.std():.3f}" if len(a) > 1 else f"{a.mean():.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--out", default="results/analysis")
    args = p.parse_args()

    data, seeds = collect(args.exp)
    if not data:
        print(f"no eval results under results/eval/{args.exp}/")
        return
    os.makedirs(args.out, exist_ok=True)

    lines = [f"# Eval aggregate — exp `{args.exp}` (mean±std over seeds)\n"]
    print(f"\n=== exp {args.exp}: kron vs iso (mean±std over seeds) ===")
    for ds in DATASETS:
        k, i = (ds, "kronecker"), (ds, "isotropic")
        if k not in data and i not in data:
            continue
        nseed = max(len(seeds.get(k, [])), len(seeds.get(i, [])))
        sk = sorted(s for s in seeds.get(k, []) if s is not None)
        lines.append(f"\n## {ds}  (seeds {sk}, n={nseed})\n")
        lines.append("| metric | kronecker | isotropic | winner |")
        lines.append("|---|---|---|:--:|")
        print(f"\n{ds}  (seeds {sk})")
        for m, label, lower in METRICS:
            kv, iv = data.get(k, {}).get(m, []), data.get(i, {}).get(m, [])
            km, im = (np.mean(kv) if kv else None), (np.mean(iv) if iv else None)
            if km is None or im is None:
                win = ""
            elif lower is None:            # coverage: closer to 0.9 wins
                win = "kron" if abs(km - 0.9) < abs(im - 0.9) else "iso"
            else:
                win = "kron" if km < im else "iso"
            row = f"| {label} | {fmt(kv)} | {fmt(iv)} | {win} |"
            lines.append(row)
            print(f"  {label:9s} kron={fmt(kv):14s} iso={fmt(iv):14s} {win}")

    report = os.path.join(args.out, f"eval_aggregate_{args.exp}.md")
    open(report, "w").write("\n".join(lines) + "\n")
    print(f"\nsaved {report}")

    # figure: one panel per metric, datasets on x, kron vs iso bars + std errorbars
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (m, label, lower) in zip(axes.flat, METRICS):
        x = np.arange(len(DATASETS)); w = 0.38
        for off, prior, c in [(-w/2, "kronecker", "#2471a3"), (w/2, "isotropic", "#c0392b")]:
            means = [np.mean(data.get((ds, prior), {}).get(m, [np.nan])) for ds in DATASETS]
            stds = [np.std(data.get((ds, prior), {}).get(m, [0]) or [0]) for ds in DATASETS]
            ax.bar(x + off, means, w, yerr=stds, capsize=3, color=c, label=prior)
        if lower is None:
            ax.axhline(0.9, ls="--", lw=1, color="gray")
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(DATASETS, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"exp {args.exp}: kron vs iso, mean±std over seeds", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figpath = os.path.join(args.out, f"fig_eval_aggregate_{args.exp}.png")
    fig.savefig(figpath, dpi=130)
    print(f"saved {figpath}")


if __name__ == "__main__":
    main()
