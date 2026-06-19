"""turn synced run logs + eval jsons into tables + figures for an experiment.

local monitoring tool: parses logs/<exp>/diffusion/*/*.out for the training curves
and the val-diagnostic metrics, and results/eval/<exp>/*/*/*.json for the test
metrics, then writes a markdown table + png figures to notebooks/reports_<exp>/.

usage:  python scripts/analysis/report_runs.py --exp 20260619 --seed 42
robust to partial data — runs still training are shown as far as they've logged.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARMS = {"kronecker": ("kron", "C0"), "isotropic": ("iso", "C2")}
DATASETS = ["metrla", "pems08f", "pemsbay"]


def parse_out(path: Path):
    """(cfg, epochs, vals, early_stop) from one training .out.
    epochs: list of (epoch, train, val, sec); vals: (epoch, recon_mae, mse, w1, ed, maha)."""
    cfg, epochs, vals, early = {}, [], [], None
    cur = None
    for line in path.read_text(errors="ignore").splitlines():
        if "c_0 auto-derived" in line:
            cfg["c0"] = float(re.search(r"=\s*([\d.]+)", line).group(1))
        if "sigma_r=" in line:
            cfg["sigma_r"] = float(re.search(r"sigma_r=([\d.]+)", line).group(1))
        m = re.match(r"epoch (\d+)\s+train=([\d.]+)\s+val=([\d.]+)\s+\(([\d.]+)s\)", line)
        if m:
            cur = int(m.group(1))
            epochs.append((cur, float(m.group(2)), float(m.group(3)), float(m.group(4))))
        mv = re.search(r"recon MAE=([\d.]+) MSE=([\d.]+).*W1=([\d.]+) ED=([\d.]+) \| mahalanobis=([\d.]+)", line)
        if mv:
            vals.append((cur, *[float(mv.group(i)) for i in range(1, 6)]))
        if "early stop at epoch" in line:
            early = int(re.search(r"epoch (\d+)", line).group(1))
    return cfg, epochs, vals, early


def collect(exp: str, seed: int):
    """{(dataset, prior): parsed} for every training log found."""
    out = {}
    for ds in DATASETS:
        for pr in ARMS:
            logs = sorted((ROOT / "logs" / exp / "diffusion" / f"diffusion_{ds}_{pr}_seed{seed}").glob("*.out"))
            if logs:
                out[(ds, pr)] = parse_out(logs[-1])
    return out


def training_table(runs) -> str:
    rows = ["| dataset | arm | epochs | best val | last recon-MAE | last W1 | last mahal | status |",
            "|---|---|---|---|---|---|---|---|"]
    for (ds, pr), (cfg, ep, vals, early) in sorted(runs.items()):
        if not ep:
            continue
        best = min(e[2] for e in ep)
        last_v = vals[-1] if vals else (None,) * 6
        status = f"early-stop @{early}" if early is not None else f"running (ep {ep[-1][0]})"
        rows.append(f"| {ds} | {pr[:4]} | {ep[-1][0]+1} | {best:.4f} | "
                    f"{last_v[1] if last_v[1] else '–'} | {last_v[3] if last_v[3] else '–'} | "
                    f"{last_v[5] if last_v[5] else '–'} | {status} |")
    return "\n".join(rows)


def loss_figure(runs, out_path):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 3.4), squeeze=False)
    for ax, ds in zip(axes[0], DATASETS):
        for pr, (lbl, c) in ARMS.items():
            r = runs.get((ds, pr))
            if not r or not r[1]:
                continue
            ep = r[1]
            e = [x[0] for x in ep]
            ax.plot(e, [x[1] for x in ep], "-", color=c, lw=1, alpha=0.6, label=f"{lbl} train")
            ax.plot(e, [x[2] for x in ep], "-", color=c, lw=2, label=f"{lbl} val")
        ax.set_title(ds); ax.set_xlabel("epoch"); ax.set_ylabel("loss (snr-weighted)")
        ax.legend(fontsize=7)
    fig.suptitle("training curves (loss not comparable across priors — see mahalanobis)", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def metric_figure(runs, out_path):
    metrics = [("mahalanobis (cross-prior comparable)", 5), ("val recon-MAE", 1), ("val W1", 3)]
    fig, axes = plt.subplots(len(DATASETS), len(metrics), figsize=(4 * len(metrics), 3 * len(DATASETS)), squeeze=False)
    for i, ds in enumerate(DATASETS):
        for j, (name, idx) in enumerate(metrics):
            ax = axes[i][j]
            for pr, (lbl, c) in ARMS.items():
                r = runs.get((ds, pr))
                if not r or not r[2]:
                    continue
                vv = r[2]
                ax.plot([v[0] for v in vv], [v[idx] for v in vv], "o-", color=c, ms=3, label=lbl)
            if i == 0:
                ax.set_title(name, fontsize=9)
            if j == 0:
                ax.set_ylabel(ds, fontsize=9)
            ax.legend(fontsize=7)
    axes[-1][0].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def eval_table(exp: str, seed: int):
    """read whatever eval jsons exist -> a flat dict {(ds,pr): metrics} + a markdown table."""
    found = {}
    for ds in DATASETS:
        for pr in ARMS:
            for jf in (ROOT / "results" / "eval" / exp / ds / pr / f"seed{seed}").glob("*.json"):
                try:
                    found[(ds, pr)] = json.loads(jf.read_text())
                except Exception:
                    pass
    if not found:
        return None
    # collect scalar metrics present across the jsons
    keys = sorted({k for d in found.values() for k, v in d.items() if isinstance(v, (int, float))})
    rows = ["| dataset | arm | " + " | ".join(keys) + " |",
            "|---|---|" + "|".join(["---"] * len(keys)) + "|"]
    for (ds, pr), d in sorted(found.items()):
        rows.append(f"| {ds} | {pr[:4]} | " + " | ".join(
            f"{d.get(k, ''):.4g}" if isinstance(d.get(k), (int, float)) else "" for k in keys) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = ROOT / "notebooks" / f"reports_{args.exp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect(args.exp, args.seed)
    tt = training_table(runs)
    loss_figure(runs, out_dir / "training_loss.png")
    metric_figure(runs, out_dir / "val_metrics.png")
    et = eval_table(args.exp, args.seed)

    report = [f"# Run report — {args.exp} / seed {args.seed}\n", "## Training\n", tt,
              "\n![loss](training_loss.png)\n![metrics](val_metrics.png)\n"]
    if et:
        report += ["\n## Eval (test set)\n", et]
    else:
        report += ["\n## Eval\n_(no eval json yet — evals run after each training)_"]
    (out_dir / "report.md").write_text("\n".join(report))

    print("\n".join(report).replace("![loss](training_loss.png)\n![metrics](val_metrics.png)", "[figures saved]"))
    print(f"\n-> wrote {out_dir}/report.md + training_loss.png + val_metrics.png")


if __name__ == "__main__":
    main()
