"""Compute val and test MAE / RMSE / MAPE per horizon for the trained backbone.

Reads the residual .npz files produced by `extract_residuals.py` (which already
contain `y_pred`, `y_true`, `mask` in raw mph) and writes a JSON summary to

    experiments/backbone-training/metrics.json

The same numbers are also printed to stdout for quick inspection.

Conventions (per the project's data protocol):
- Masked MAE / RMSE: only sensors with `mask == True` contribute.
- MAPE additionally excludes elements with `|y_true| < 1.0` (standard for traffic
  speed; very small ground-truth speeds make the relative error explode).
- Reported in raw mph; predictions are inverse-transformed during residual extraction.

Usage:
    python scripts/compute_test_metrics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESID_DIR = ROOT / "results" / "residuals"
OUT_DIR = ROOT / "experiments" / "backbone-training"

DATASETS = ["metrla", "pemsbay"]
SPLITS = ["val", "test"]
HORIZONS = list(range(1, 13))   # h=1..12

# Wu et al., IJCAI 2019 reported numbers (test set, single seed)
WU2019 = {
    "metrla":  {3: {"MAE": 2.69, "RMSE": 5.15, "MAPE": 6.90},
                6: {"MAE": 3.07, "RMSE": 6.22, "MAPE": 8.37},
                12: {"MAE": 3.53, "RMSE": 7.37, "MAPE": 10.01}},
    "pemsbay": {3: {"MAE": 1.30, "RMSE": 2.74, "MAPE": 2.73},
                6: {"MAE": 1.63, "RMSE": 3.70, "MAPE": 3.67},
                12: {"MAE": 1.95, "RMSE": 4.52, "MAPE": 4.63}},
}


def metrics_for_horizon(y_true: np.ndarray, y_pred: np.ndarray,
                        mask: np.ndarray, h_idx: int) -> dict:
    """Masked MAE/RMSE on horizon h, plus MAPE excluding |y|<1.0."""
    m = mask[:, h_idx, :, :].astype(bool)
    yt = y_true[:, h_idx, :, :][m]
    yp = y_pred[:, h_idx, :, :][m]
    err = yp - yt
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape_mask = np.abs(yt) >= 1.0
    if mape_mask.sum() > 0:
        mape = float((np.abs(err[mape_mask]) / np.abs(yt[mape_mask])).mean()) * 100.0
    else:
        mape = float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "n_valid": int(m.sum())}


def process(dataset: str) -> dict:
    out = {"dataset": dataset, "splits": {}}
    for split in SPLITS:
        path = RESID_DIR / f"gwn_{dataset}_{split}_seed0.npz"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        data = np.load(path)
        y_true, y_pred, mask = data["y_true"], data["y_pred"], data["mask"]
        n_windows = int(y_true.shape[0])

        per_horizon = {}
        for h in HORIZONS:
            per_horizon[str(h)] = metrics_for_horizon(y_true, y_pred, mask, h - 1)

        # Average over horizons (unweighted)
        avg = {
            "MAE":  float(np.mean([per_horizon[str(h)]["MAE"]  for h in HORIZONS])),
            "RMSE": float(np.mean([per_horizon[str(h)]["RMSE"] for h in HORIZONS])),
            "MAPE": float(np.mean([per_horizon[str(h)]["MAPE"] for h in HORIZONS])),
        }

        out["splits"][split] = {
            "n_windows":   n_windows,
            "per_horizon": per_horizon,
            "average_over_horizons": avg,
        }
    return out


def print_summary(results: dict) -> None:
    print("=" * 84)
    print("GWN — masked MAE / RMSE / MAPE per horizon (raw mph; MAPE excludes |y|<1.0)")
    print("=" * 84)
    for ds in DATASETS:
        if ds not in results:
            continue
        print(f"\n[{ds.upper()}]  n_windows: " +
              ", ".join(f"{s}={results[ds]['splits'][s]['n_windows']}"
                        for s in results[ds]["splits"]))
        for split in results[ds]["splits"]:
            d = results[ds]["splits"][split]
            print(f"\n  --- {split} ---")
            wu = WU2019.get(ds, {})
            print("  horizon |   MAE    |   RMSE   |  MAPE   | "
                  "Wu2019 MAE / Δ% | Wu2019 RMSE / Δ% | Wu2019 MAPE / Δ%")
            for h in HORIZONS:
                m = d["per_horizon"][str(h)]
                if h in wu and split == "test":
                    dmae = (m["MAE"] - wu[h]["MAE"]) / wu[h]["MAE"] * 100
                    drmse = (m["RMSE"] - wu[h]["RMSE"]) / wu[h]["RMSE"] * 100
                    dmape = (m["MAPE"] - wu[h]["MAPE"]) / wu[h]["MAPE"] * 100
                    print(f"  h={h:2d} (5×{h:>2d}min) | "
                          f"{m['MAE']:7.3f}  | {m['RMSE']:7.3f}  | {m['MAPE']:6.2f}%  | "
                          f"{wu[h]['MAE']:5.2f} / {dmae:+5.1f}%   | "
                          f"{wu[h]['RMSE']:5.2f} / {drmse:+5.1f}%   | "
                          f"{wu[h]['MAPE']:5.2f}% / {dmape:+5.1f}%")
                else:
                    print(f"  h={h:2d} (5×{h:>2d}min) | "
                          f"{m['MAE']:7.3f}  | {m['RMSE']:7.3f}  | {m['MAPE']:6.2f}%")
            avg = d["average_over_horizons"]
            print(f"  avg(h=1..12)   | {avg['MAE']:7.3f}  | "
                  f"{avg['RMSE']:7.3f}  | {avg['MAPE']:6.2f}%")
    print()


def main() -> None:
    results = {ds: process(ds) for ds in DATASETS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "backbone": "gwn",
                "seed": 0,
                "horizons": HORIZONS,
                "metrics": ["MAE", "RMSE", "MAPE"],
                "wu2019_reference": WU2019,
                "notes": (
                    "MAE/RMSE in raw mph. MAPE in %; excludes elements with |y_true|<1.0 "
                    "per standard traffic-speed convention. Masked: only sensors with "
                    "mask==True contribute."
                ),
            },
            "results": results,
        }, f, indent=2)
    print(f"Wrote {out_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
