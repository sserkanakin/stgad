"""Driver for the Tier 1-4 residual analysis (see notebooks/tier_analysis.ipynb).

Computes one tier at a time over the ladder signal -> mlp -> gcn -> gwn x datasets
and caches the results to results/analysis/tier<N>.json so the notebook renders
instantly. Heavy tiers can be run per-dataset (for DAIC submission).

Usage:
    .venv/bin/python scripts/run_tier_analysis.py --tier 1
    .venv/bin/python scripts/run_tier_analysis.py --tier 2 --datasets metrla pems08f
"""
import argparse
import json
import time
from pathlib import Path

from src.eval import tiers

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "analysis"


def run_tier1(datasets, rungs):
    rows = []
    for ds in datasets:
        ctx = tiers.graph_context(ds)
        for rung in rungs:
            t = time.time()
            row = tiers.tier1_cell(ds, rung, ctx)
            rows.append(row)
            print(f"  tier1 {ds:<8} {rung:<6}  rho_ratio={row['rho_ratio']:.2f} "
                  f"gft_low={row['gft_low']:.2f} eta={row['eta_separability']:.2f} "
                  f"kurt={row['excess_kurtosis']:.1f}  ({time.time()-t:.1f}s)", flush=True)
    return rows


def run_tier2(datasets, rungs):
    rows = []
    for ds in datasets:
        ctx = tiers.graph_context(ds)
        for rung in rungs:
            t = time.time()
            row = tiers.tier2_cell(ds, rung, ctx)
            rows.append(row)
            print(f"  tier2 {ds:<8} {rung:<6}  winner={row['winner']:<10} "
                  f"rank={'<'.join(row['ranking'])}  graph-iso ΔNLL={row['graph_vs_iso_nll']:.2f} "
                  f"({time.time()-t:.1f}s)", flush=True)
    return rows


def run_tier3(datasets, rungs):
    rows = []
    for ds in datasets:
        ctx = tiers.graph_context(ds)
        for rung in rungs:
            t = time.time()
            row = tiers.tier3_cell(ds, rung, ctx)
            rows.append(row)
            print(f"  tier3 {ds:<8} {rung:<6}  math_ok={row['math_validated']} "
                  f"var_err={row['marginal_var_relerr']:.3f}  "
                  f"c0·θmin={row['c0_theta_min']:.2f} mixed@chi={row['frac_modes_mixed_at_chi']:.2f} "
                  f"({time.time()-t:.0f}s)", flush=True)
    return rows


def run_tier4(datasets, rungs):
    rows = []
    for ds in datasets:
        ctx = tiers.graph_context(ds)
        for rung in rungs:
            t = time.time()
            row = tiers.tier4_cell(ds, rung, ctx)
            rows.append(row)
            g = row["crps_gap_pct"]
            print(f"  tier4 {ds:<8} {rung:<6}  CRPS struct/iso by steps; "
                  f"gap@low={g.get(str(row['steps'][0]),'?')}  ({time.time()-t:.0f}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="+", default=tiers.DATASETS)
    ap.add_argument("--rungs", nargs="+", default=tiers.RUNGS)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    runner = {1: run_tier1, 2: run_tier2, 3: run_tier3, 4: run_tier4}[args.tier]
    rows = runner(args.datasets, args.rungs)

    out = OUTDIR / f"tier{args.tier}.json"
    # merge with any existing rows for datasets/rungs we didn't recompute
    existing = json.loads(out.read_text()) if out.exists() else []
    key = lambda r: (r["dataset"], r["rung"])
    done = {key(r) for r in rows}
    merged = rows + [r for r in existing if key(r) not in done]
    out.write_text(json.dumps(merged, indent=2))
    print(f"\nwrote {len(merged)} cells -> {out}")


if __name__ == "__main__":
    main()
