"""experiment leaderboard: collect every run's val/test metrics under one EXP tree.

walks <OUTPUTS>/<exp>/ for val_results.json (the final-val payload: best.pt step
sweep + full-val metrics at the chosen step count) and any {val,test}_metrics_steps*.json,
then writes, Vimal-style (summarize_diffstg_runs.py):

  <exp>/summary.md   markdown leaderboard sorted by [CRPS, RMSE, MAE]
  <exp>/summary.csv  same rows as csv
  <exp>/<dataset>/summary.txt   per-dataset table incl. the step-sweep evidence

usage:
    python scripts/summarize_valsweep.py --exp 20260706_gadmse
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils import paths

COLS = ["dataset", "prior", "seed", "split", "n_steps", "best_epoch",
        "mae", "rmse", "mape_pct", "crps", "winkler", "coverage", "w1", "energy",
        "n_windows", "k"]


def _row(res: dict, *, best_epoch=None) -> dict:
    return {
        "dataset": res["dataset"], "prior": res["prior"], "seed": res["seed"],
        "split": res.get("split", "test"), "n_steps": res["n_steps"],
        "best_epoch": best_epoch if best_epoch is not None else res.get("ckpt_epoch"),
        "mae": round(res["mae"], 4), "rmse": round(res["rmse"], 4),
        "mape_pct": round(100 * res["mape"], 2), "crps": round(res["crps"], 4),
        "winkler": round(res["winkler"], 4), "coverage": round(res["coverage"], 3),
        "w1": round(res["w1"], 4), "energy": round(res["energy"], 4),
        "n_windows": res["n_test_windows"], "k": res["k_samples"],
    }


def collect(root: Path) -> tuple[list[dict], dict]:
    """(leaderboard rows, {run key -> steps_sweep table}) from the exp tree."""
    rows, sweeps = [], {}
    for f in sorted(root.rglob("val_results.json")):
        p = json.loads(f.read_text())
        rows.append(_row(p["full_val"], best_epoch=p.get("best_epoch")))
        key = (p["dataset"], p["prior"], p["seed"])
        sweeps[key] = {"chosen_steps": p["chosen_steps"], "sweep": p["steps_sweep"]}
    seen = {(r["dataset"], r["prior"], r["seed"], r["split"], r["n_steps"]) for r in rows}
    for f in sorted(root.rglob("*_metrics_steps*.json")):
        res = json.loads(f.read_text())
        k = (res["dataset"], res["prior"], res["seed"], res.get("split", "test"), res["n_steps"])
        if k not in seen:                       # standalone evals not in a val_results payload
            rows.append(_row(res))
            seen.add(k)
    return rows, sweeps


def write_leaderboard(root: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["crps"], r["rmse"], r["mae"]))   # Vimal's sort
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    lines = ["# " + root.name + " — leaderboard (sorted by CRPS, RMSE, MAE)", "",
             "| " + " | ".join(COLS) + " |",
             "|" + "|".join("---" for _ in COLS) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in COLS) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {root / 'summary.md'} + summary.csv ({len(rows)} rows)")


def write_per_dataset(root: Path, rows: list[dict], sweeps: dict) -> None:
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, rs in by_ds.items():
        L = [f"=== {ds} ===", ""]
        L.append(f"{'prior':10s} {'split':5s} {'steps':>6s} {'MAE':>8s} {'RMSE':>8s} "
                 f"{'MAPE%':>7s} {'CRPS':>7s} {'Wink':>8s} {'cov':>5s}")
        for r in sorted(rs, key=lambda r: (r["prior"], r["n_steps"])):
            L.append(f"{r['prior']:10s} {r['split']:5s} {r['n_steps']:>6d} {r['mae']:>8.3f} "
                     f"{r['rmse']:>8.3f} {r['mape_pct']:>7.2f} {r['crps']:>7.4f} "
                     f"{r['winkler']:>8.3f} {r['coverage']:>5.3f}")
        for (d, prior, seed), s in sorted(sweeps.items()):
            if d != ds:
                continue
            L.append("")
            L.append(f"step sweep on best.pt ({prior}, seed {seed}) — chosen: {s['chosen_steps']}")
            for ns in sorted(s["sweep"]["steps"], key=int):
                r = s["sweep"]["steps"][ns]
                L.append(f"  steps={int(ns):>5d}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  "
                         f"MAPE%={100*r['mape']:.2f}  CRPS={r['crps']:.4f}")
        out = root / ds / "summary.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(L) + "\n")
        print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", required=True, help="experiment tag (folder under OUTPUTS)")
    args = p.parse_args()

    root = paths.exp_root(args.exp)
    if not root.exists():
        raise SystemExit(f"no such experiment tree: {root}")
    rows, sweeps = collect(root)
    if not rows:
        raise SystemExit(f"no val_results.json / *_metrics_steps*.json under {root} yet")
    write_leaderboard(root, rows)
    write_per_dataset(root, rows, sweeps)


if __name__ == "__main__":
    main()
