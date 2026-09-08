"""Aggregate the 20260907_matched campaign: seed means/sd per (dataset, arm) for the per-point
and joint metrics, the kron-vs-iso deltas, and the GWN+Gaussian floor. reads the synced
results/20260907_matched tree, writes results/analysis/matched_summary.json and prints the
thesis tables in markdown. cells whose eval predates the joint-crps fix (b452482, 15:41:48
on 2026-09-07) contribute no joint-CRPS until re-evaluated; they are listed.
"""
import json, glob, collections, sys
from pathlib import Path
import numpy as np
ROOT = Path("results/20260907_matched")
PRE_FIX = {"pems08f/isotropic/seed42", "pems08f/isotropic/seed2", "pems08f/kronecker/seed42", "pems08f/kronecker/seed2", "pems08s/isotropic/seed42"}
KEYS = ["crps", "mae", "rmse", "coverage", "winkler", "joint_tot_crps", "joint_tot_coverage", "joint_tot_spread_ratio",
        "joint_cum_crps", "joint_cum_coverage", "joint_cum_spread_ratio", "ckpt_epoch", "n_steps", "gen_seconds"]
LOWER = {"crps", "mae", "rmse", "winkler", "joint_tot_crps", "joint_cum_crps"}
DS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]

cells = collections.defaultdict(dict); stale = []
for f in sorted(glob.glob(str(ROOT / "*/*/gwn/*/seed*/eval/test_metrics_*.json"))):
    m = json.load(open(f)); p = f.split("/"); ds, arm, seed = p[3], p[5], p[6]
    key = f"{ds}/{arm}/{seed}"
    from datetime import datetime
    if key in PRE_FIX and Path(f).stat().st_mtime < datetime(2026, 9, 7, 15, 45).timestamp():
        stale.append(key)
        for k in ("joint_tot_crps", "joint_cum_crps"): m[k] = None
    cells[(ds, arm)][seed] = {k: m.get(k) for k in KEYS}
gauss = json.load(open("results/analysis/gaussian_baseline.json"))

summary = {"gaussian_floor": gauss, "stale_joint_crps_cells": stale, "cells": {}, "deltas": {}}
def agg(vals):
    v = [x for x in vals if x is not None]
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)} if v else None
for (ds, arm), seeds in cells.items():
    summary["cells"][f"{ds}/{arm}"] = {k: agg([s[k] for s in seeds.values()]) for k in KEYS}
    summary["cells"][f"{ds}/{arm}"]["seeds"] = sorted(seeds)
for ds in DS:
    a, b = summary["cells"].get(f"{ds}/isotropic"), summary["cells"].get(f"{ds}/kronecker")
    if not a or not b: continue
    d = {}
    for k in KEYS:
        if a[k] and b[k] and k not in ("ckpt_epoch", "n_steps", "gen_seconds"):
            rel = (b[k]["mean"] - a[k]["mean"]) / abs(a[k]["mean"]) * 100
            d[k] = {"rel_pct": rel, "abs": b[k]["mean"] - a[k]["mean"], "kron_better": bool(rel < 0) if k in LOWER else bool(rel > 0)}
    summary["deltas"][ds] = d
Path("results/analysis/matched_summary.json").write_text(json.dumps(summary, indent=1))

def fmt(c, k, d=4):
    return "–" if not c or not c.get(k) else f"{c[k]['mean']:.{d}f} ± {c[k]['sd']:.{d}f}"
print("## Per point (test, k=100, seed mean ± sd)\n")
print("| dataset | GWN+Gaussian | iso CRPS | kron CRPS | iso MAE | kron MAE | iso cov90 | kron cov90 |\n|---|---|---|---|---|---|---|---|")
for ds in DS:
    a, b = summary["cells"].get(f"{ds}/isotropic"), summary["cells"].get(f"{ds}/kronecker")
    print(f"| {ds} | {gauss[ds]['crps']:.4f} | {fmt(a,'crps')} | {fmt(b,'crps')} | {fmt(a,'mae',2)} | {fmt(b,'mae',2)} | {fmt(a,'coverage',3)} | {fmt(b,'coverage',3)} |")
print("\n## Joint — network total per horizon (test, seed mean ± sd)\n")
print("| dataset | iso CRPS | kron CRPS | Δ | iso cov90 | kron cov90 | iso spread/err | kron spread/err |\n|---|---|---|---|---|---|---|---|")
for ds in DS:
    a, b, d = summary["cells"].get(f"{ds}/isotropic"), summary["cells"].get(f"{ds}/kronecker"), summary["deltas"].get(ds, {})
    dl = d.get("joint_tot_crps"); dl = f"{dl['rel_pct']:+.1f} %" if dl else "–"
    print(f"| {ds} | {fmt(a,'joint_tot_crps',1)} | {fmt(b,'joint_tot_crps',1)} | {dl} | {fmt(a,'joint_tot_coverage',3)} | {fmt(b,'joint_tot_coverage',3)} | {fmt(a,'joint_tot_spread_ratio',3)} | {fmt(b,'joint_tot_spread_ratio',3)} |")
print("\n## Joint — cumulative per sensor over the hour\n")
print("| dataset | iso CRPS | kron CRPS | Δ | iso cov90 | kron cov90 | iso spread/err | kron spread/err |\n|---|---|---|---|---|---|---|---|")
for ds in DS:
    a, b, d = summary["cells"].get(f"{ds}/isotropic"), summary["cells"].get(f"{ds}/kronecker"), summary["deltas"].get(ds, {})
    dl = d.get("joint_cum_crps"); dl = f"{dl['rel_pct']:+.1f} %" if dl else "–"
    print(f"| {ds} | {fmt(a,'joint_cum_crps',2)} | {fmt(b,'joint_cum_crps',2)} | {dl} | {fmt(a,'joint_cum_coverage',3)} | {fmt(b,'joint_cum_coverage',3)} | {fmt(a,'joint_cum_spread_ratio',3)} | {fmt(b,'joint_cum_spread_ratio',3)} |")
won = collections.Counter(); n = collections.Counter()
for ds, d in summary["deltas"].items():
    for k, v in d.items(): won[k] += v["kron_better"]; n[k] += 1
print("\nkron better than iso (datasets won / compared): " + ", ".join(f"{k}: {won[k]}/{n[k]}" for k in n))
if stale: print("stale joint-CRPS (re-eval pending): " + ", ".join(stale))
