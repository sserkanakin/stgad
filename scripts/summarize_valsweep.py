"""experiment leaderboard: collect every run's val/test metrics under one EXP tree.

walks <OUTPUTS>/<exp>/ for val_results.json (final-val payload; both the old
single-checkpoint schema and the round-2 schema with per-checkpoint blocks
{best_loss, best_crps} x {em, ancestral}) and any standalone *_metrics_*.json,
then writes, Vimal-style (summarize_diffstg_runs.py):

  <exp>/summary.md   markdown leaderboard sorted by [CRPS, RMSE, MAE]
  <exp>/summary.csv  same rows as csv
  <exp>/<dataset>/summary.txt   per-dataset table incl. the step-sweep evidence
  <exp>/comparison.md  our best rows vs the published baselines (HySRD-Net Table 1,
                       TEST split — our val rows are a proxy; caveat printed)

usage:
    python scripts/summarize_valsweep.py --exp 20260708_r2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils import paths

COLS = ["dataset", "arm", "prior", "seed", "split", "select", "sampler", "n_steps",
        "epoch", "mae", "rmse", "mape_pct", "crps", "winkler", "coverage",
        "w1", "energy", "n_windows", "k"]

# published numbers to beat — HySRD-Net (ESWA 2026, Table 1; TEST split, S=100
# samples, 60/20/20, 12->12) + the strongest baselines it reports. avg-horizon.
PAPER_BASELINES = {
    # dataset: {model: (rmse, mae, crps)}
    "pems04f": {"HySRD-Net": (31.78, 20.20, 0.0723), "SpecSTG": (33.15, 21.53, 0.0766),
                "PriSTI": (33.74, 22.46, 0.0772), "DiffSTG": (37.62, 24.90, 0.0904),
                "TimeGrad": (35.58, 21.70, 0.0801), "GCRDD": (36.28, 22.16, 0.0779)},
    "pems08f": {"HySRD-Net": (25.49, 16.30, 0.0547), "SpecSTG": (25.59, 17.06, 0.0572),
                "PriSTI": (26.35, 17.30, 0.0576), "DiffSTG": (28.26, 18.99, 0.0692),
                "TimeGrad": (33.09, 20.47, 0.0705), "GCRDD": (28.83, 18.72, 0.0626)},
    "pems04s": {"HySRD-Net": (3.82, 1.79, 0.0241), "SpecSTG": (4.06, 1.93, 0.0245),
                "PriSTI": (4.92, 1.96, 0.0222), "DiffSTG": (4.33, 2.15, 0.0262),
                "TimeGrad": (5.92, 2.38, 0.0307), "GCRDD": (4.46, 1.94, 0.0245)},
    "pems08s": {"HySRD-Net": (3.39, 1.52, 0.0204), "SpecSTG": (3.45, 1.63, 0.0211),
                "PriSTI": (4.22, 1.70, 0.0217), "DiffSTG": (3.97, 2.36, 0.0325),
                "TimeGrad": (4.98, 1.98, 0.0267), "GCRDD": (3.75, 1.72, 0.0223)},
}


def _row(res: dict, *, arm="", select="", epoch=None) -> dict:
    return {
        "dataset": res["dataset"], "arm": arm, "prior": res["prior"], "seed": res["seed"],
        "split": res.get("split", "test"), "select": select,
        "sampler": res.get("sampler", "em"), "n_steps": res["n_steps"],
        "epoch": epoch if epoch is not None else res.get("ckpt_epoch"),
        "mae": round(res["mae"], 4), "rmse": round(res["rmse"], 4),
        "mape_pct": round(100 * res["mape"], 2), "crps": round(res["crps"], 4),
        "winkler": round(res["winkler"], 4), "coverage": round(res["coverage"], 3),
        "w1": round(res["w1"], 4), "energy": round(res["energy"], 4),
        "n_windows": res["n_test_windows"], "k": res["k_samples"],
    }


def _arm_of(f: Path, root: Path) -> str:
    """arm name = first path component under the exp root (round-2 nested layout);
    old flat layouts have the dataset there instead — return '' for those."""
    part = f.relative_to(root).parts[0]
    return "" if part in PAPER_BASELINES or part in ("metrla", "pemsbay") else part


def collect(root: Path) -> tuple[list[dict], dict]:
    """(leaderboard rows, {run key -> sweep evidence}) from the exp tree."""
    rows, sweeps = [], {}
    seen = set()
    for f in sorted(root.rglob("val_results.json")):
        p = json.loads(f.read_text())
        arm = _arm_of(f, root)
        if "checkpoints" in p:                      # round-2 schema
            for tag, blk in p["checkpoints"].items():
                for key, res in (("full_val", blk.get("full_val")),
                                 ("full_val_anc", blk.get("full_val_anc"))):
                    if res:
                        rows.append(_row(res, arm=arm, select=tag, epoch=blk.get("epoch")))
                        seen.add((res["dataset"], arm, res["prior"], res["seed"],
                                  res.get("split", "test"), res.get("sampler", "em"),
                                  res["n_steps"], tag))
                sweeps[(p["dataset"], arm, p["prior"], p["seed"], tag)] = {
                    "chosen_steps": blk.get("chosen_steps"),
                    "chosen_anc_steps": blk.get("chosen_anc_steps"),
                    "sweep": blk["steps_sweep"]}
        else:                                       # pre-round-2 schema
            rows.append(_row(p["full_val"], arm=arm, select="best_loss",
                             epoch=p.get("best_epoch")))
            sweeps[(p["dataset"], arm, p["prior"], p["seed"], "best_loss")] = {
                "chosen_steps": p.get("chosen_steps"), "chosen_anc_steps": None,
                "sweep": p["steps_sweep"]}
    for f in sorted(root.rglob("*_metrics_*.json")):
        res = json.loads(f.read_text())
        arm = _arm_of(f, root)
        tag = "best_crps" if res.get("ckpt") == "best_crps.pt" else "best_loss"
        k = (res["dataset"], arm, res["prior"], res["seed"], res.get("split", "test"),
             res.get("sampler", "em"), res["n_steps"], tag)
        if k not in seen:                       # standalone evals not in a payload
            rows.append(_row(res, arm=arm, select=tag))
            seen.add(k)
    return rows, sweeps


def write_leaderboard(root: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["dataset"], r["crps"], r["rmse"], r["mae"]))
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    lines = ["# " + root.name + " — leaderboard (per dataset, sorted by CRPS, RMSE, MAE)", "",
             "| " + " | ".join(COLS) + " |",
             "|" + "|".join("---" for _ in COLS) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in COLS) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {root / 'summary.md'} + summary.csv ({len(rows)} rows)")


def write_comparison(root: Path, rows: list[dict]) -> None:
    """our best row per dataset vs the published baselines (HySRD Table 1)."""
    L = ["# vs published baselines (HySRD-Net Table 1)", "",
         "caveats: paper numbers are TEST split with S=100 samples; ours below are "
         "whatever split/k the rows carry (val during development). same 12->12 task, "
         "same normalized-CRPS convention (SpecSTG numbers match their paper).", ""]
    for ds, base in PAPER_BASELINES.items():
        ours = [r for r in rows if r["dataset"] == ds]
        if not ours:
            continue
        L.append(f"## {ds}")
        L.append("| model | RMSE | MAE | CRPS | note |")
        L.append("|---|---|---|---|---|")
        for name, (rm, ma, cr) in sorted(base.items(), key=lambda kv: kv[1][2]):
            L.append(f"| {name} | {rm} | {ma} | {cr} | paper, test, S=100 |")
        for r in sorted(ours, key=lambda r: r["crps"])[:3]:
            note = f"ours, {r['split']}, k={r['k']}, {r['select']}/{r['sampler']}{r['n_steps']}"
            arm = r["arm"] or r["prior"]
            L.append(f"| **ours:{arm}** | {r['rmse']} | {r['mae']} | {r['crps']} | {note} |")
        best_ours = min(ours, key=lambda r: r["crps"])
        gap = 100 * (best_ours["crps"] / base["HySRD-Net"][2] - 1)
        L.append(f"\nbest-ours CRPS vs HySRD-Net: {gap:+.1f}%")
        L.append("")
    (root / "comparison.md").write_text("\n".join(L) + "\n")
    print(f"wrote {root / 'comparison.md'}")


def write_per_dataset(root: Path, rows: list[dict], sweeps: dict) -> None:
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, rs in by_ds.items():
        L = [f"=== {ds} ===", ""]
        L.append(f"{'arm':12s} {'select':10s} {'sampler':9s} {'steps':>6s} {'k':>3s} "
                 f"{'MAE':>8s} {'RMSE':>8s} {'MAPE%':>7s} {'CRPS':>7s} {'Wink':>8s} {'cov':>5s}")
        for r in sorted(rs, key=lambda r: r["crps"]):
            L.append(f"{(r['arm'] or r['prior']):12s} {r['select']:10s} {r['sampler']:9s} "
                     f"{r['n_steps']:>6d} {r['k']:>3d} {r['mae']:>8.3f} {r['rmse']:>8.3f} "
                     f"{r['mape_pct']:>7.2f} {r['crps']:>7.4f} "
                     f"{r['winkler']:>8.3f} {r['coverage']:>5.3f}")
        for (d, arm, prior, seed, tag), s in sorted(sweeps.items()):
            if d != ds:
                continue
            L.append("")
            L.append(f"sweep [{arm or prior} / {tag}] — chosen em={s['chosen_steps']} "
                     f"anc={s['chosen_anc_steps']}")
            for bucket, s_tag in (("steps", "em"), ("steps_anc", "anc")):
                for ns in sorted(s["sweep"].get(bucket, {}), key=int):
                    r = s["sweep"][bucket][ns]
                    L.append(f"  {s_tag}={int(ns):>5d}  MAE={r['mae']:.3f}  "
                             f"RMSE={r['rmse']:.3f}  MAPE%={100*r['mape']:.2f}  "
                             f"CRPS={r['crps']:.4f}")
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
        raise SystemExit(f"no val_results.json / *_metrics_*.json under {root} yet")
    write_leaderboard(root, rows)
    write_per_dataset(root, rows, sweeps)
    write_comparison(root, rows)


if __name__ == "__main__":
    main()
