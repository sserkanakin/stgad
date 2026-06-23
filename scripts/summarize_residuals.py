"""Summarize the backbone->residuals grid into one understandable CSV.

Walks results/residuals/<dataset>/<backbone>/seed<seed>/{train,test}.npz and
computes, per cell, the backbone's point-forecast quality and the residual scale
the diffusion stage will use. All errors are in RAW units (mph for speed,
veh/5min for flow), masked with the same convention the cell was extracted with.

residuals = y_true - y_hat, so |residuals| IS the backbone's absolute error.

Usage:  .venv/bin/python scripts/summarize_residuals.py
Writes: results/residuals_summary.csv  (+ prints the table)
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "residuals"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "residuals_summary.csv"

# horizons to report (0-indexed): 15 / 30 / 60 min at 5-min steps
H_REPORT = {"h3_15min": 2, "h6_30min": 5, "h12_60min": 11}


def meta(ds):
    m = json.loads((PROC / ds / "meta.json").read_text())
    return m["modality"], m["units"], m.get("null_val", "0.0")


def masked(arr, m):
    v = arr[m]
    return v if v.size else np.array([0.0])


def cell_row(ds, bb, seed):
    d = RES / ds / bb / f"seed{seed}"
    te = np.load(d / "test.npz")
    tr = np.load(d / "train.npz")
    r_te = te["residuals"].astype(np.float32)          # (B,N,H) raw
    m_te = te["mask"].astype(bool)
    r_tr = tr["residuals"].astype(np.float32)
    m_tr = tr["mask"].astype(bool)
    B, N, H = r_te.shape
    modality, units, null_val = meta(ds)

    mae = float(np.abs(masked(r_te, m_te)).mean())
    rmse = float(np.sqrt((masked(r_te, m_te) ** 2).mean()))
    row = {
        "dataset": ds, "backbone": bb, "seed": seed, "N": N,
        "modality": modality, "units": units,
        "masks_zeros": "yes" if null_val == "0.0" else "no",
        "n_train": int(tr["residuals"].shape[0]),
        "n_test": B,
        "test_MAE": round(mae, 4),
        "test_RMSE": round(rmse, 4),
    }
    for name, h in H_REPORT.items():
        row[f"test_MAE_{name}"] = round(
            float(np.abs(masked(r_te[..., h], m_te[..., h])).mean()), 4)
    row["resid_sigma_train"] = round(float(masked(r_tr, m_tr).std()), 4)
    row["resid_bias_train"] = round(float(masked(r_tr, m_tr).mean()), 4)
    row["missing_pct_test"] = round(100.0 * (1.0 - m_te.mean()), 3)
    return row


def main():
    cells = sorted(p.parent.parent.parent.name + "|" + p.parent.parent.name + "|"
                   + p.parent.name.replace("seed", "")
                   for p in RES.glob("*/*/seed*/test.npz"))
    rows = []
    for c in cells:
        ds, bb, seed = c.split("|")
        try:
            rows.append(cell_row(ds, bb, int(seed)))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {ds}/{bb}/seed{seed}: {e}")
    rows.sort(key=lambda r: (r["dataset"], r["backbone"], r["seed"]))

    cols = list(rows[0].keys())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # pretty print
    print(f"\n{len(rows)} cells -> {OUT}\n")
    show = ["dataset", "backbone", "N", "modality", "test_MAE", "test_RMSE",
            "test_MAE_h3_15min", "test_MAE_h6_30min", "test_MAE_h12_60min",
            "resid_sigma_train", "resid_bias_train"]
    hdr = f"{'dataset':<8} {'bb':<4} {'N':>4} {'mod':<5} {'MAE':>7} {'RMSE':>7} " \
          f"{'MAE@15':>7} {'MAE@30':>7} {'MAE@60':>7} {'sigma':>7} {'bias':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['dataset']:<8} {r['backbone']:<4} {r['N']:>4} {r['modality']:<5} "
              f"{r['test_MAE']:>7.3f} {r['test_RMSE']:>7.3f} "
              f"{r['test_MAE_h3_15min']:>7.3f} {r['test_MAE_h6_30min']:>7.3f} "
              f"{r['test_MAE_h12_60min']:>7.3f} {r['resid_sigma_train']:>7.3f} "
              f"{r['resid_bias_train']:>+6.2f}")


if __name__ == "__main__":
    main()
