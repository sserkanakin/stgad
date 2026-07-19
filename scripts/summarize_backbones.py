"""Overview of the backbone -> residual grid: forecast quality AND residual
graph-smoothness, one row per (dataset, backbone, seed).

Idempotent: re-run any time (or watch it); it summarizes whatever cells exist so
far. Reads only the saved artifacts (configs, histories, residual npz) -- no GPU.

The load-bearing column is `resid_low%`: the fraction of the residual's energy in
the lowest 10% of graph modes. The thesis hypothesis is that a WEAKER backbone
leaves a MORE graph-smooth residual (higher resid_low%) -- because a strong
backbone (GWN) eats the smooth part, leaving a near-white residual. This table
lets you read that trend straight off (mlp >= gcn >= gwn within a dataset).

walks   RESULTS/backbone/<ds>/<bb>/seed<s>/{config.json,history.json,best.pt-metrics}
        RESULTS/residuals/<ds>/<bb>/seed<s>/{train,test}.npz
writes  RESULTS/analysis/backbone_overview.csv
usage:  python scripts/summarize_backbones.py [--seed 42]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.adjacency import load_adjacency, compute_normalized_laplacian   # noqa: E402
from src.utils import paths                                                    # noqa: E402

VARIANTS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
BACKBONES = ["mlp", "gcn", "gwn"]
_EIG = {}


def resid_lowfrac(ds, split, seed, bb):
    """fraction of residual energy in the lowest 10% of graph modes (train split)."""
    p = paths.residual_npz(ds, split, seed, bb)
    if not p.exists():
        return None, None
    try:
        d = np.load(p)
        R = d["residuals"].astype(np.float64)      # (B, N, H)
    except Exception as e:                          # truncated / corrupt transfer
        print(f"  !! {ds}/{bb}/{split}.npz unreadable ({str(e)[:40]}) -- skipped")
        return None, None
    M = d["mask"].astype(bool)
    std = float(R[M].std()) if M.any() else float("nan")
    if ds not in _EIG:
        L = compute_normalized_laplacian(load_adjacency(ds, repair=True))
        lam, V = np.linalg.eigh(L); o = np.argsort(lam)
        _EIG[ds] = (lam[o], V[:, o])
    lam, V = _EIG[ds]
    N = len(lam)
    Rz = np.where(M, R, 0.0)
    sub = Rz[np.linspace(0, len(Rz) - 1, min(3000, len(Rz))).astype(int)]
    C = np.einsum("nk,bnh->bkh", V, sub)
    e = (C ** 2).mean((0, 2))
    low = float(e[: max(1, N // 10)].sum() / e.sum())
    return std, low


def read_cell(ds, bb, seed):
    bdir = paths.backbone_dir(ds, seed, bb)
    cfg = bdir / "config.json"
    if not cfg.exists():
        return None
    c = json.loads(cfg.read_text())
    row = {"dataset": ds, "backbone": bb, "seed": seed,
           "params": c.get("n_params"), "epochs_ran": None,
           "val_mae": c.get("best_val_mae"), "test_mae": None, "test_rmse": None}
    hist = bdir / "history.json"
    if hist.exists():
        h = json.loads(hist.read_text())
        row["epochs_ran"] = len(h)
    std, low = resid_lowfrac(ds, "train", seed, bb)
    row["resid_std"] = std
    row["resid_low"] = low
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    for ds in VARIANTS:
        for bb in BACKBONES:
            r = read_cell(ds, bb, args.seed)
            if r:
                rows.append(r)

    if not rows:
        print("no backbone runs found yet (RESULTS/backbone/<ds>/<bb>/seed"
              f"{args.seed}/config.json). run submit_backbone first.")
        return

    hdr = f"{'dataset':9s} {'bb':4s} {'params':>8s} {'ep':>3s} {'val_MAE':>8s} {'resid_std':>9s} {'resid_low%':>10s}"
    print(hdr); print("-" * len(hdr))
    last_ds = None
    for r in rows:
        if last_ds and r["dataset"] != last_ds:
            print()
        last_ds = r["dataset"]
        p = f"{r['params']:,}" if r["params"] else "?"
        vm = f"{r['val_mae']:.3f}" if r["val_mae"] is not None else "  -  "
        rs = f"{r['resid_std']:.3f}" if r.get("resid_std") is not None else "  -  "
        rl = f"{r['resid_low']:.1%}" if r.get("resid_low") is not None else "  -  "
        print(f"{r['dataset']:9s} {r['backbone']:4s} {p:>8s} {str(r['epochs_ran'] or '?'):>3s} "
              f"{vm:>8s} {rs:>9s} {rl:>10s}")

    print("\nresid_low% = residual energy in lowest 10% of graph modes.")
    print("hypothesis: WEAKER backbone -> HIGHER resid_low% (more graph structure left).")
    print("read down each dataset: does mlp >= gcn >= gwn?")

    out = paths.analysis_dir("metrics")
    with open(out / "backbone_overview.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out/'backbone_overview.csv'}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
