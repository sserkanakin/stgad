"""Complete metrics table for every backbone x dataset, computed from the saved
residuals (exact: residual = y - yhat, and yhat is stored, so forecast metrics
are recovered without re-running anything).

Per (dataset, backbone, seed):
  model    : params, epochs trained
  forecast : MAE / RMSE / MAPE on val AND test (masked, raw units)
  horizon  : test MAE at 15 / 30 / 60 min (h3 / h6 / h12)
  residual : std, graph-smoothness (energy in lowest 10% of graph modes)

masking follows the family convention: PEMS flow/speed use null_val=nan (the
stored mask is all-valid); MAPE additionally drops y_true==0 (SpecSTG/DiffSTG).

usage: .venv/bin/python scripts/metrics_table.py [--seed 42] [--split test]
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


def _metrics(R, yhat, M):
    """MAE, RMSE, MAPE over masked entries (raw units). R=residual, y=yhat+R."""
    m = M & np.isfinite(R)
    if not m.any():
        return None
    r = R[m]
    mae = float(np.abs(r).mean())
    rmse = float(np.sqrt((r ** 2).mean()))
    y = (yhat + R)[m]
    nz = np.abs(y) > 1e-3                                  # drop y_true==0 for MAPE
    mape = float((np.abs(r[nz]) / np.abs(y[nz])).mean() * 100) if nz.any() else float("nan")
    return mae, rmse, mape


def _lowfrac(ds, R, M):
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
    return float(e[: max(1, N // 10)].sum() / e.sum())


def _load(ds, bb, seed, split):
    p = paths.residual_npz(ds, split, seed, bb)
    if not p.exists():
        return None
    try:
        d = np.load(p)
    except Exception:
        return None
    return d["residuals"].astype(np.float64), d["y_pred"].astype(np.float64), d["mask"].astype(bool)


def read_cell(ds, bb, seed):
    cfg = paths.backbone_dir(ds, seed, bb) / "config.json"
    if not cfg.exists():
        return None
    c = json.loads(cfg.read_text())
    hist = paths.backbone_dir(ds, seed, bb) / "history.json"
    ep = len(json.loads(hist.read_text())) if hist.exists() else None
    row = {"dataset": ds, "backbone": bb, "params": c.get("n_params"), "epochs": ep}

    for split in ("val", "test"):
        got = _load(ds, bb, seed, split)
        if not got:
            continue
        R, yhat, M = got
        mae, rmse, mape = _metrics(R, yhat, M)
        row[f"{split}_MAE"] = mae
        row[f"{split}_RMSE"] = rmse
        row[f"{split}_MAPE"] = mape
        if split == "test":
            for label, h in (("15m", 2), ("30m", 5), ("60m", 11)):        # 5-min steps
                mh = M[:, :, h] & np.isfinite(R[:, :, h])
                row[f"MAE_{label}"] = float(np.abs(R[:, :, h][mh]).mean()) if mh.any() else None
            row["resid_std"] = float(R[M].std())
            row["resid_low"] = _lowfrac(ds, R, M)
    return row


def _f(v, w=7, p=3):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) and v is not None else " " * (w - 1) + "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = [r for ds in VARIANTS for bb in BACKBONES
            if (r := read_cell(ds, bb, args.seed))]
    if not rows:
        print("no cells found."); return

    hdr = (f"{'dataset':9s} {'bb':4s} {'params':>8s} {'ep':>3s} | "
           f"{'test_MAE':>8s} {'test_RMSE':>9s} {'test_MAPE':>9s} | "
           f"{'val_MAE':>7s} {'val_RMSE':>8s} | "
           f"{'MAE@15m':>7s} {'@30m':>6s} {'@60m':>6s} | {'rstd':>5s} {'low%':>5s}")
    print(hdr); print("-" * len(hdr))
    last = None
    for r in rows:
        if last and r["dataset"] != last:
            print()
        last = r["dataset"]
        p = f"{r['params']:,}" if r["params"] else "?"
        low = f"{r['resid_low']*100:.0f}" if r.get("resid_low") is not None else "-"
        print(f"{r['dataset']:9s} {r['backbone']:4s} {p:>8s} {str(r['epochs'] or '?'):>3s} | "
              f"{_f(r.get('test_MAE'),8)} {_f(r.get('test_RMSE'),9)} {_f(r.get('test_MAPE'),9,2)} | "
              f"{_f(r.get('val_MAE'),7)} {_f(r.get('val_RMSE'),8)} | "
              f"{_f(r.get('MAE_15m'),7)} {_f(r.get('MAE_30m'),6)} {_f(r.get('MAE_60m'),6)} | "
              f"{_f(r.get('resid_std'),5,2)} {low:>5s}")

    print("\nMAE/RMSE raw units (mph for speed, veh/5min for flow). MAPE %, y_true>0 only.")
    print("low% = residual energy in lowest 10% of graph modes (graph-smoothness).")

    out = paths.analysis_dir("metrics")
    keys = ["dataset", "backbone", "params", "epochs", "test_MAE", "test_RMSE", "test_MAPE",
            "val_MAE", "val_RMSE", "MAE_15m", "MAE_30m", "MAE_60m", "resid_std", "resid_low"]
    with open(out / "metrics_full.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out/'metrics_full.csv'}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
