#!/usr/bin/env python
"""aggregate diffusion test_results.json into a CRPS-sorted leaderboard (csv + md).

mirrors vimal's tools/summarize_diffstg_runs.py: scan results/eval/<exp> for every
test_results.json, flatten the per-sample-steps rows, sort by CRPS, and print/save a
table plus the kron-vs-iso delta per dataset.

  python scripts/summarize_runs.py --exp 20260624
  python scripts/summarize_runs.py --exp 20260624 --out-csv summary.csv --out-md summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils import paths

# columns pulled from each test_results entry, in display order.
COLS = ["dataset", "prior", "backbone", "seed", "n_steps", "best_epoch",
        "mae", "rmse", "mape", "crps", "winkler", "coverage", "w1", "energy"]


def _rows(root: Path) -> list[dict]:
    rows = []
    for jf in sorted(root.rglob("test_results.json")):
        payload = json.loads(jf.read_text())
        for r in payload.get("test_results", []):
            rows.append({
                "dataset": payload.get("dataset", r.get("dataset")),
                "prior": payload.get("prior", r.get("prior")),
                "backbone": payload.get("backbone", "gwn"),
                "seed": payload.get("seed"),
                "best_epoch": payload.get("best_epoch"),
                "n_steps": r.get("n_steps"),
                "mae": r.get("mae"), "rmse": r.get("rmse", r.get("mse", 0) ** 0.5),
                "mape": r.get("mape"), "crps": r.get("crps"),
                "winkler": r.get("winkler"), "coverage": r.get("coverage"),
                "w1": r.get("w1"), "energy": r.get("energy"),
            })
    return rows


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", required=True, help="experiment tag (results/eval/<exp>)")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-md", default=None)
    args = p.parse_args()

    root = paths.OUTPUTS / "eval" / args.exp
    if not root.exists():
        raise SystemExit(f"no eval results under {root}")
    rows = _rows(root)
    if not rows:
        raise SystemExit(f"no test_results.json found under {root}")

    # leaderboard: sort by CRPS (primary), RMSE, MAE — same habit as vimal's summarizer.
    rows.sort(key=lambda r: (r.get("crps") or 1e9, r.get("rmse") or 1e9, r.get("mae") or 1e9))

    # print
    hdr = "  ".join(c[:8].ljust(8) for c in COLS)
    print(f"\n=== {args.exp}: {len(rows)} rows (sorted by CRPS) ===")
    print(hdr)
    for r in rows:
        print("  ".join(_fmt(r[c])[:8].ljust(8) for c in COLS))

    # kron-vs-iso delta per dataset (best CRPS row per (dataset, prior))
    best: dict = {}
    for r in rows:
        key = (r["dataset"], r["prior"])
        if key not in best or (r.get("crps") or 1e9) < (best[key].get("crps") or 1e9):
            best[key] = r
    print("\n=== kron vs iso (best-CRPS row per dataset; Δ% = (kron-iso)/iso) ===")
    print("dataset    metric    kron        iso         winner   Δ%")
    for ds in sorted({d for d, _ in best}):
        k, i = best.get((ds, "kronecker")), best.get((ds, "isotropic"))
        if not (k and i):
            continue
        for m, lower in [("crps", True), ("rmse", True), ("mae", True), ("coverage", None)]:
            kv, iv = k.get(m), i.get(m)
            if kv is None or iv is None:
                continue
            if lower is None:   # coverage: closer to 0.9 wins
                win = "kron" if abs(kv - 0.9) < abs(iv - 0.9) else "iso"
                d = ""
            else:
                win = "kron" if kv < iv else "iso"
                d = f"{100*(kv-iv)/iv:+.1f}" if iv else ""
            print(f"{ds:9s} {m:9s} {kv:<11.4f} {iv:<11.4f} {win:8s} {d:>6s}")
        print("-" * 56)

    out_csv = Path(args.out_csv) if args.out_csv else root / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows([{c: r[c] for c in COLS} for r in rows])
    print(f"\nwrote {out_csv}")

    out_md = Path(args.out_md) if args.out_md else root / "summary.md"
    with open(out_md, "w") as f:
        f.write(f"# {args.exp} — diffusion test leaderboard (sorted by CRPS)\n\n")
        f.write("| " + " | ".join(COLS) + " |\n")
        f.write("|" + "|".join("---" for _ in COLS) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r[c]) for c in COLS) + " |\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
