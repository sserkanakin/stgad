r"""Emit the thesis's tab:exp_main / tab:exp_dist rows from the CURRENT results.

Why this exists: the draft's tab:exp_main is captioned "history conditioning" but its
numbers are the 20260721_ladder "-past" arm, which conditioned on the previous window's
RESIDUAL (the silent fallback, see docs/conditioning_report.md). The real
history-conditioned numbers are campaign 20260728_histcond. This script regenerates the
rows so the table can be replaced rather than hand-edited.

Sign convention matches the draft: Delta% is positive when the STRUCTURED prior improves
on the isotropic control, i.e. 100*(iso - kron)/iso. Lower is better for all three
metrics, and the better value in each pair is bolded.

usage:
  .venv/bin/python scripts/analysis/thesis_tables_conditioning.py [--cond hist|fo|prev]
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CELLS = os.path.join(ROOT, "results", "analysis", "paper_results", "master_cells.csv")

# (csv dataset key, thesis label) in the draft's row order
DATASETS = [("pems08f", "PEMS08 flow"), ("pems08s", "PEMS08 speed"),
            ("pems04f", "PEMS04 flow"), ("pems04s", "PEMS04 speed"),
            ("metrla", "METR-LA"), ("pemsbay", "PEMS-BAY")]
RULE_AFTER = "pems04s"                       # \midrule before the speed pair
BACKBONES = [("mlp", "linear"), ("gcn", "GCN"), ("gwn", "GWN")]
COND_CAMPAIGN = {"fo": ("20260721_ladder", "fo"),
                 "hist": ("20260728_histcond", "hist"),
                 "prev": ("20260721_ladder", "past")}


def load(cond):
    camp, ckey = COND_CAMPAIGN[cond]
    cell = defaultdict(dict)
    with open(CELLS) as f:
        for r in csv.DictReader(f):
            if r["whit"] == "1" or r["ugnet"] == "1":
                continue
            if r["campaign"] != camp or r["cond"] != ckey or r["base"] not in ("iso", "emp"):
                continue
            for m in ("crps", "mae", "rmse", "coverage", "winkler", "energy"):
                if r.get(m):
                    cell[(r["base"], r["dataset"], r["backbone"], m)][int(r["seed"])] = float(r[m])
    return cell


def paired(cell, ds, bb, metric):
    """(iso_mean, kron_mean, n) over seeds BOTH arms evaluated."""
    i = cell.get(("iso", ds, bb, metric), {})
    k = cell.get(("emp", ds, bb, metric), {})
    common = sorted(set(i) & set(k))
    if not common:
        return None
    return (statistics.mean(i[s] for s in common),
            statistics.mean(k[s] for s in common), len(common))


def fmt(v, nd, better):
    s = f"{v:.{nd}f}"
    return rf"\textbf{{{s}}}" if better else s


def dist_rows(cell):
    """tab:exp_dist -- coverage (no Delta), Winkler and energy score (Delta each)."""
    print(r"% ---- tab:exp_dist rows ----")
    for ds, label in DATASETS:
        first = True
        for bb, bblabel in BACKBONES:
            cov = paired(cell, ds, bb, "coverage")
            wnk = paired(cell, ds, bb, "winkler")
            eng = paired(cell, ds, bb, "energy")
            if cov is None and wnk is None:
                continue
            cols = []
            # coverage: both under-cover, so higher (closer to 0.90) is better
            i, k, _ = cov
            cols += [fmt(i, 3, i >= k), fmt(k, 3, k > i)]
            for r, nd in ((wnk, 2 if ds in ("pems08f", "pems04f") else 2), (eng, 2)):
                i, k, _ = r
                d = 100.0 * (i - k) / i
                cols += [fmt(i, nd, i <= k), fmt(k, nd, k < i),
                         (r"$+$" if d >= 0 else r"$-$") + f"{abs(d):.1f}"]
            head = f"{label:<12}" if first else " " * 12
            first = False
            print(f"{head} & {bblabel:<6} & " + " & ".join(cols) + r" \\")
        if ds == RULE_AFTER:
            print(r"\midrule")


def paired_counts(cell):
    """tab:exp_paired -- per-run wins, structured vs isotropic, over matched runs."""
    from math import comb
    print(r"% ---- tab:exp_paired counts ----")
    PEMS = {"pems08f", "pems08s", "pems04f", "pems04s"}
    for metric, lower_better in (("mae", True), ("rmse", True), ("crps", True),
                                 ("energy", True), ("coverage", False)):
        for scope, keep in (("PEMS", PEMS), ("all", {d for d, _ in DATASETS})):
            w = t = 0
            for ds, _ in DATASETS:
                if ds not in keep:
                    continue
                for bb, _ in BACKBONES:
                    i = cell.get(("iso", ds, bb, metric), {})
                    k = cell.get(("emp", ds, bb, metric), {})
                    for s in sorted(set(i) & set(k)):
                        t += 1
                        w += (k[s] < i[s]) if lower_better else (k[s] > i[s])
            if t:
                p = sum(comb(t, x) for x in range(w, t + 1)) / 2 ** t
                print(f"%   {metric:<9}{scope:<5}{w}/{t}  p={p:.1g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="hist", choices=list(COND_CAMPAIGN))
    ap.add_argument("--table", default="main", choices=("main", "dist", "paired"))
    args = ap.parse_args()
    cell = load(args.cond)
    if args.table == "dist":
        return dist_rows(cell)
    if args.table == "paired":
        return paired_counts(cell)

    # decimals per metric, per dataset family (flow is O(10), speed O(1))
    def nd_for(ds, metric):
        if metric == "crps":
            return 4
        return 2 if ds in ("pems08f", "pems04f") else 3

    print(f"% ---- tab:exp_main rows, conditioning = {args.cond} "
          f"(campaign {COND_CAMPAIGN[args.cond][0]}) ----")
    seedcounts = set()
    for ds, label in DATASETS:
        first = True
        for bb, bblabel in BACKBONES:
            cols = []
            ok = False
            for metric in ("crps", "mae", "rmse"):
                r = paired(cell, ds, bb, metric)
                if r is None:
                    cols += ["--", "--", "--"]
                    continue
                ok = True
                i, k, n = r
                seedcounts.add(n)
                nd = nd_for(ds, metric)
                d = 100.0 * (i - k) / i                    # + favours structure
                cols += [fmt(i, nd, i <= k), fmt(k, nd, k < i),
                         (r"$+$" if d >= 0 else r"$-$") + f"{abs(d):.1f}"]
            if not ok:
                continue
            head = f"{label:<12}" if first else " " * 12
            first = False
            print(f"{head} & {bblabel:<6} & " + " & ".join(cols) + r" \\")
        if ds == RULE_AFTER:
            print(r"\midrule")
    print(f"% seed counts present: {sorted(seedcounts)}")


if __name__ == "__main__":
    main()
