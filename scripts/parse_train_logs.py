"""Parse train/val curves out of DSM-training slurm logs.

A fallback for the wandb-resume bug that prevented some runs' per-epoch
metrics from reaching the dashboard. The script scans
``logs/train-diffusion/dsm-*-*.out`` files, extracts the per-epoch line
that ``train_diffusion.py`` prints (``epoch N/99  train=X val=Y (Zs, B
batches)``), and writes one CSV per run plus an optional combined
overview CSV and a comparison plot.

Usage:
    # Default: scan logs/train-diffusion/, write CSVs + plot to
    # experiments/score-network/training_curves/
    python scripts/parse_train_logs.py

    # Only scrape specific runs:
    python scripts/parse_train_logs.py --pattern 'dsm-metrla-*-s0-*.out'

    # Custom output dir:
    python scripts/parse_train_logs.py --out-dir /tmp/curves
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_RE_RUNID = re.compile(r"^Run ID: (\S+)")
_RE_EPOCH = re.compile(
    r"^epoch\s+(?P<epoch>\d+)/\d+\s+"
    r"train=(?P<train>[\d.]+)\s+"
    r"val=(?P<val>[\d.]+)\s+"
    r"\((?P<seconds>[\d.]+)s,\s+(?P<batches>\d+)\s+batches\)"
)
_RE_DSMID = re.compile(
    r"dsm_(?P<prior>structured|isotropic)_"
    r"(?P<dataset>metrla|pemsbay|pems08f)_"
    r"seed(?P<seed>\d+)_"
    r"(?P<hash>\w+)"
)


def parse_one_log(path: Path) -> dict | None:
    """Parse a single slurm-out log and return run metadata + epoch DataFrame.

    Returns None if the log doesn't look like a DSM training log (no Run ID
    line found, no epoch lines).
    """
    run_id: str | None = None
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if run_id is None:
                m = _RE_RUNID.match(line)
                if m:
                    run_id = m.group(1)
                    continue
            m = _RE_EPOCH.match(line)
            if m:
                rows.append({
                    "epoch":   int(m.group("epoch")),
                    "train":   float(m.group("train")),
                    "val":     float(m.group("val")),
                    "seconds": float(m.group("seconds")),
                    "batches": int(m.group("batches")),
                })
    if run_id is None or not rows:
        return None
    meta = _RE_DSMID.search(run_id)
    return {
        "log_path": path,
        "run_id":   run_id,
        "prior":    meta.group("prior") if meta else None,
        "dataset":  meta.group("dataset") if meta else None,
        "seed":     int(meta.group("seed")) if meta else None,
        "hash":     meta.group("hash") if meta else None,
        "epochs":   pd.DataFrame(rows),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-dir", default="logs/train-diffusion",
                   help="Directory containing slurm .out files.")
    p.add_argument("--pattern", default="dsm-*-*.out",
                   help="Glob pattern within --log-dir (default: dsm-*-*.out).")
    p.add_argument("--out-dir", default="experiments/score-network/training_curves",
                   help="Where to save per-run CSVs + combined plot.")
    p.add_argument("--no-plot", action="store_true", help="Skip the combined plot.")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_paths = sorted(log_dir.glob(args.pattern))
    if not log_paths:
        print(f"[parse_train_logs] no files matched {log_dir}/{args.pattern}")
        return 1

    parsed = []
    for path in log_paths:
        rec = parse_one_log(path)
        if rec is None:
            print(f"  [skip] {path.name}: no Run ID or epoch lines")
            continue
        parsed.append(rec)
        print(f"  [ok]   {path.name}  →  {rec['run_id']}  ({len(rec['epochs'])} epochs)")

    if not parsed:
        print("[parse_train_logs] no parseable logs.")
        return 1

    # Per-run CSVs
    for rec in parsed:
        csv_path = out_dir / f"curves_{rec['run_id']}.csv"
        rec["epochs"].to_csv(csv_path, index=False)
    print(f"\n  Wrote {len(parsed)} per-run CSVs to {out_dir}/")

    # Combined CSV (one row per (run_id, epoch))
    combined_rows = []
    for rec in parsed:
        df = rec["epochs"].copy()
        df["run_id"] = rec["run_id"]
        df["dataset"] = rec["dataset"]
        df["prior"] = rec["prior"]
        df["seed"] = rec["seed"]
        combined_rows.append(df)
    combined = pd.concat(combined_rows, ignore_index=True)
    combined_path = out_dir / "all_curves.csv"
    combined.to_csv(combined_path, index=False)
    print(f"  Wrote combined: {combined_path}  ({len(combined)} rows)")

    # Summary table: final epoch + best val per run
    summary_rows = []
    for rec in parsed:
        df = rec["epochs"]
        summary_rows.append({
            "run_id":          rec["run_id"],
            "dataset":         rec["dataset"],
            "prior":           rec["prior"],
            "seed":            rec["seed"],
            "n_epochs":        len(df),
            "final_train":     float(df["train"].iloc[-1]),
            "final_val":       float(df["val"].iloc[-1]),
            "best_val":        float(df["val"].min()),
            "best_val_epoch":  int(df.loc[df["val"].idxmin(), "epoch"]),
            "mean_epoch_secs": float(df["seconds"].iloc[1:].mean() if len(df) > 1 else df["seconds"].iloc[0]),
        })
    summary = pd.DataFrame(summary_rows).set_index("run_id")
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path)
    print(f"  Wrote summary:  {summary_path}")
    print()
    print(summary.to_string(float_format="%.4f"))

    # ── Plot ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 9})

        datasets = sorted({r["dataset"] for r in parsed if r["dataset"]})
        priors = ("structured", "isotropic")
        if not datasets:
            print("[parse_train_logs] no recognised datasets in parsed logs — skipping plot")
            return 0

        fig, axes = plt.subplots(2, len(datasets), figsize=(4.3 * len(datasets), 6), sharex=True)
        if len(datasets) == 1:
            axes = axes.reshape(2, 1)

        for col, ds in enumerate(datasets):
            ax_t = axes[0, col]
            ax_v = axes[1, col]
            for prior, style, color in (("structured", "-", "tab:blue"),
                                         ("isotropic",  "--", "tab:orange")):
                runs = [r for r in parsed if r["dataset"] == ds and r["prior"] == prior]
                for r in runs:
                    df = r["epochs"]
                    ax_t.plot(df["epoch"], df["train"], style, color=color,
                              alpha=0.85, label=f"{prior}")
                    ax_v.plot(df["epoch"], df["val"], style, color=color,
                              alpha=0.85, label=f"{prior}")
            ax_t.set_title(f"{ds} — train loss", fontsize=10)
            ax_v.set_title(f"{ds} — val loss",   fontsize=10)
            ax_t.grid(alpha=0.3)
            ax_v.grid(alpha=0.3)
            ax_v.set_xlabel("epoch")
            handles, labels = ax_v.get_legend_handles_labels()
            if handles:
                # dedupe while preserving order
                seen = set()
                uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
                ax_v.legend(*zip(*uniq), fontsize=8, loc="upper right")
        axes[0, 0].set_ylabel("train DSM loss")
        axes[1, 0].set_ylabel("val DSM loss")
        fig.suptitle("DSM training curves — recovered from slurm logs", fontsize=11)
        plt.tight_layout()
        fig_path = out_dir / "training_curves.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote plot:     {fig_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
