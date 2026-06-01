"""Score baselines, load the diffusion metrics, and assemble comparison tables.

Cheap CPU step (no GPU, no checkpoint). It:
  1. scores the two analytic baselines (prior-only Gaussian, per-horizon
     Gaussian) through the *same* MetricAccumulator the diffusion pass uses;
  2. computes GWN point metrics directly from the residual file
     (MAE = mean|residual|, etc.) — the backbone has no distribution;
  3. loads the diffusion ``metrics_{split}.json`` written by generate_samples;
  4. writes ``comparison_{split}.md`` + ``.csv`` (rows = methods, cols = headline
     metrics) and a combined ``metrics_all_{split}.json``.

To keep everything strictly comparable, the baselines mirror the diffusion run's
window count and ensemble size K (read from the diffusion metrics metadata) when
that file is present.

Usage:
    python scripts/eval/evaluate_forecasts.py --dataset metrla --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.diffusion.prior import load_prior                        # noqa: E402
from src.eval.io import load_split_arrays                         # noqa: E402
from src.eval.metrics import MetricAccumulator                    # noqa: E402
from src.eval.samplers import PerHorizonGaussianSampler, PriorOnlySampler  # noqa: E402

# Headline columns for the comparison table (key -> display label).
TABLE_COLS = [
    ("crps", "CRPS↓"),
    ("energy_score", "Energy↓"),
    ("variogram_score", "Variogram↓"),
    ("mae", "MAE↓"),
    ("rmse", "RMSE↓"),
    ("mape", "MAPE%↓"),
    ("ace", "ACE↓"),
    ("pinaw90", "PINAW90"),
    ("pred_std", "PredStd"),
    ("pit_cal_err", "PITerr↓"),
]


def point_metrics(residuals, y_true, mask, H) -> dict:
    """GWN point metrics straight from residuals (= y_true − y_pred)."""
    out_per_h_mae, out_per_h_rmse, out_per_h_mape = [], [], []
    abs_r = np.abs(residuals)
    for h in range(H):
        m = mask[:, :, h]
        r = residuals[:, :, h][m]
        if r.size == 0:
            out_per_h_mae.append(float("nan"))
            out_per_h_rmse.append(float("nan"))
            out_per_h_mape.append(float("nan"))
            continue
        out_per_h_mae.append(float(np.abs(r).mean()))
        out_per_h_rmse.append(float(np.sqrt((r ** 2).mean())))
        yt = y_true[:, :, h][m]
        nz = np.abs(yt) > 1e-6
        out_per_h_mape.append(float(100.0 * (np.abs(r[nz]) / np.abs(yt[nz])).mean()))
    m_all = mask
    r_all = residuals[m_all]
    yt_all = y_true[m_all]
    nz = np.abs(yt_all) > 1e-6
    overall = {
        "mae": float(np.abs(r_all).mean()),
        "rmse": float(np.sqrt((r_all ** 2).mean())),
        "mape": float(100.0 * (np.abs(r_all[nz]) / np.abs(yt_all[nz])).mean()),
        # No predictive distribution for a point forecast.
        "crps": float("nan"), "energy_score": float("nan"),
        "variogram_score": float("nan"), "ace": float("nan"),
        "pinaw90": float("nan"), "pred_std": 0.0, "pit_cal_err": float("nan"),
    }
    return {
        "overall": overall,
        "per_horizon": {"mae": out_per_h_mae, "rmse": out_per_h_rmse, "mape": out_per_h_mape},
        "meta": {"method": "gwn_point"},
    }


def score_sampler(sampler, y_pred, y_true, mask, B, batch_size, H, vpairs, seed) -> dict:
    acc = MetricAccumulator(H=H, variogram_pairs=vpairs, variogram_seed=seed)
    for start in range(0, B, batch_size):
        stop = min(start + batch_size, B)
        samples = sampler.sample(y_pred[start:stop])
        acc.update(samples, y_true[start:stop], mask[start:stop])
    return acc.finalize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems08f"), required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=100, help="baseline K (ignored if diffusion metrics present)")
    p.add_argument("--max-windows", type=int, default=0, help="override window cap")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--variogram-pairs", type=int, default=2000)
    args = p.parse_args()

    out_dir = ROOT / "results" / "eval" / args.dataset
    diff_path = out_dir / f"metrics_{args.split}.json"
    diff_metrics = json.loads(diff_path.read_text()) if diff_path.exists() else None

    # Mirror the diffusion run's window count + K so baselines are comparable.
    K = args.n_samples
    max_windows = args.max_windows
    vpairs = args.variogram_pairs
    if diff_metrics is not None:
        meta = diff_metrics["meta"]
        K = meta["n_samples"]
        max_windows = max_windows or meta["windows_scored"]
        vpairs = diff_metrics["counts"].get("variogram_pairs", vpairs)
        print(f"[evaluate] mirroring diffusion run: K={K} windows={max_windows}", flush=True)
    else:
        print("[evaluate] no diffusion metrics found — scoring baselines only", flush=True)

    arrays = load_split_arrays(args.dataset, args.split, args.seed)
    y_pred, y_true, mask = arrays["y_pred"], arrays["y_true"], arrays["mask"]
    residuals = arrays["residuals"]
    B_total, N, H = y_true.shape
    B = B_total if max_windows <= 0 else min(max_windows, B_total)
    sl = slice(0, B)

    prior = load_prior(args.dataset)
    methods: dict[str, dict] = {}

    methods["gwn_point"] = point_metrics(residuals[sl], y_true[sl], mask[sl], H)
    print("[evaluate] GWN point done", flush=True)

    methods["per_horizon_gaussian"] = score_sampler(
        PerHorizonGaussianSampler(prior.delta_h, n_samples=K, seed=args.seed),
        y_pred, y_true, mask, B, args.batch_size, H, vpairs, args.seed,
    )
    methods["per_horizon_gaussian"]["meta"] = {"method": "per_horizon_gaussian", "n_samples": K}
    print("[evaluate] per-horizon Gaussian done", flush=True)

    methods["prior_only"] = score_sampler(
        PriorOnlySampler(prior, n_samples=K, seed=args.seed),
        y_pred, y_true, mask, B, args.batch_size, H, vpairs, args.seed,
    )
    methods["prior_only"]["meta"] = {"method": "prior_only", "n_samples": K}
    print("[evaluate] prior-only Gaussian done", flush=True)

    if diff_metrics is not None:
        methods["diffusion"] = diff_metrics

    # ── Write combined json + comparison tables.
    (out_dir / f"metrics_all_{args.split}.json").write_text(json.dumps(methods, indent=2))
    order = ["gwn_point", "per_horizon_gaussian", "prior_only", "diffusion"]
    order = [m for m in order if m in methods]
    _write_tables(out_dir, args.split, methods, order)
    _print_summary(methods, order)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.4f}"


def _write_tables(out_dir: Path, split: str, methods: dict, order: list[str]) -> None:
    header = ["method"] + [label for _, label in TABLE_COLS]
    rows = []
    for m in order:
        ov = methods[m]["overall"]
        rows.append([m] + [_fmt(ov.get(k)) for k, _ in TABLE_COLS])

    # Markdown.
    md = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (out_dir / f"comparison_{split}.md").write_text("\n".join(md) + "\n")

    # CSV.
    with (out_dir / f"comparison_{split}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[evaluate] wrote comparison_{split}.md / .csv to {out_dir}", flush=True)


def _print_summary(methods: dict, order: list[str]) -> None:
    print("\n=== Overall comparison ===")
    header = ["method"] + [label for _, label in TABLE_COLS]
    print("  ".join(f"{h:>11}" for h in header))
    for m in order:
        ov = methods[m]["overall"]
        cells = [f"{m:>11}"] + [f"{_fmt(ov.get(k)):>11}" for k, _ in TABLE_COLS]
        print("  ".join(cells))


if __name__ == "__main__":
    main()
