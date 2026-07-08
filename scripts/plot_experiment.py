"""comparison figures for one experiment tree -> <exp>/figures/*.png.

reads the same artifacts as summarize_valsweep (history.csv per run,
val_results.json payloads) and renders the four decision figures:

  1. curves_<ds>.png       sampled val CRPS vs epoch, one line per arm
                           ("is training healthy" — the DiffSTG-style monitor)
  2. leaderboard.png       CRPS per dataset: our arms (dots) vs published
                           baselines from the HySRD-Net table (open markers)
  3. samplers_<ds>.png     step-count sweeps on the selected weights:
                           EM vs ancestral CRPS vs n_steps (agreement check)
  4. horizons_<ds>.png     per-horizon MAE of the best arm vs the backbone-only
                           row implied by the residual cache (if present)

usage:
    python scripts/plot_experiment.py --exp 20260708_r2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from summarize_valsweep import PAPER_BASELINES, collect  # noqa: E402

from src.utils import paths  # noqa: E402

ARM_COLORS = {"kron_pastG": "#1f77b4", "kron_pastPN": "#2ca02c",
              "kron_base": "#9467bd", "iso_pastG": "#d62728"}


def _runs(root: Path):
    """(arm, dataset, run_dir) for every run under the per-arm layout."""
    for hist in sorted(root.rglob("history.csv")):
        run = hist.parent                      # .../<arm>/<ds>/<bb>/<prior>/seedS/train
        rel = run.relative_to(root).parts
        if len(rel) >= 5:
            yield rel[0], rel[1], run


def plot_curves(root: Path, out: Path) -> None:
    by_ds: dict[str, list] = {}
    for arm, ds, run in _runs(root):
        ep, crps = [], []
        with open(run / "history.csv") as f:
            for row in csv.DictReader(f):
                if row.get("val_crps"):
                    ep.append(int(row["epoch"])); crps.append(float(row["val_crps"]))
        if ep:
            by_ds.setdefault(ds, []).append((arm, ep, crps))
    for ds, lines in by_ds.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm, ep, crps in sorted(lines):
            ax.plot(ep, crps, label=arm, lw=1.6, color=ARM_COLORS.get(arm))
        ax.set(xlabel="epoch", ylabel="sampled val CRPS",
               title=f"{ds} — selection signal (EMA weights, fixed windows/noise)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / f"curves_{ds}.png", dpi=150); plt.close(fig)
        print(f"wrote {out / f'curves_{ds}.png'}")


def plot_leaderboard(rows: list[dict], out: Path) -> None:
    ds_order = [d for d in ("metrla", "pems04f", "pems04s", "pems08f", "pems08s",
                            "pemsbay") if any(r["dataset"] == d for r in rows)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for xi, ds in enumerate(ds_order):
        # our arms: best row (min crps) per arm, prefer the k=8 ancestral eval
        ours = {}
        for r in rows:
            if r["dataset"] != ds:
                continue
            key = r["arm"] or r["prior"]
            if key not in ours or r["crps"] < ours[key]["crps"]:
                ours[key] = r
        for arm, r in ours.items():
            ax.scatter(xi, r["crps"], color=ARM_COLORS.get(arm, "k"), zorder=3,
                       label=arm if xi == 0 else None)
        for name, (rm, ma, cr) in PAPER_BASELINES.get(ds, {}).items():
            star = name == "HySRD-Net"
            ax.scatter(xi, cr, marker="*" if star else "o", s=110 if star else 30,
                       facecolors="none", edgecolors="dimgray", zorder=2,
                       label=("HySRD-Net (test)" if star and xi <= 1 else
                              ("published baselines" if xi <= 1 and name == "SpecSTG" else None)))
    ax.set_xticks(range(len(ds_order))); ax.set_xticklabels(ds_order)
    ax.set(ylabel="CRPS (normalized)", title="ours (val) vs published (test) — lower is better")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out / "leaderboard.png", dpi=150); plt.close(fig)
    print(f"wrote {out / 'leaderboard.png'}")


def plot_samplers(sweeps: dict, out: Path) -> None:
    by_ds: dict[str, list] = {}
    for (ds, arm, prior, seed, tag), s in sweeps.items():
        if tag != "best_crps":                 # the shipped model's sweep
            continue
        by_ds.setdefault(ds, []).append((arm or prior, s["sweep"]))
    for ds, entries in by_ds.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm, sweep in sorted(entries):
            c = ARM_COLORS.get(arm)
            em = sorted((int(k), v["crps"]) for k, v in sweep.get("steps", {}).items())
            an = sorted((int(k), v["crps"]) for k, v in sweep.get("steps_anc", {}).items())
            if em:
                ax.plot(*zip(*em), "-o", color=c, ms=4, label=f"{arm} EM")
            if an:
                ax.plot(*zip(*an), "--s", color=c, ms=4, alpha=0.7, label=f"{arm} ancestral")
        ax.set(xscale="log", xlabel="reverse steps", ylabel="CRPS (64-window sweep)",
               title=f"{ds} — EM vs ancestral on best_crps.pt")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / f"samplers_{ds}.png", dpi=150); plt.close(fig)
        print(f"wrote {out / f'samplers_{ds}.png'}")


def plot_horizons(root: Path, out: Path) -> None:
    import json
    by_ds: dict[str, list] = {}
    for f in sorted(root.rglob("val_results.json")):
        p = json.loads(f.read_text())
        arm = f.relative_to(root).parts[0]
        blk = (p.get("checkpoints") or {}).get("best_crps") or {}
        res = blk.get("full_val_anc") or blk.get("full_val")
        if res and res.get("mae_h"):
            by_ds.setdefault(p["dataset"], []).append((arm, res["mae_h"]))
    for ds, entries in by_ds.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm, mae_h in sorted(entries):
            ax.plot(range(1, len(mae_h) + 1), mae_h, "-o", ms=3,
                    color=ARM_COLORS.get(arm), label=arm)
        ax.set(xlabel="horizon (steps of 5 min)", ylabel="MAE",
               title=f"{ds} — per-horizon MAE (best_crps ckpt, full val)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / f"horizons_{ds}.png", dpi=150); plt.close(fig)
        print(f"wrote {out / f'horizons_{ds}.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", required=True)
    args = p.parse_args()
    root = paths.exp_root(args.exp)
    if not root.exists():
        raise SystemExit(f"no such experiment tree: {root}")
    out = root / "figures"
    out.mkdir(exist_ok=True)
    rows, sweeps = collect(root)
    plot_curves(root, out)
    if rows:
        plot_leaderboard(rows, out)
    if sweeps:
        plot_samplers(sweeps, out)
    plot_horizons(root, out)


if __name__ == "__main__":
    main()
