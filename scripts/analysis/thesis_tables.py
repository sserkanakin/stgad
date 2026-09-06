r"""Emit the LaTeX bodies of every results table in Chapters 5-7, from the eval jsons.

WHY THIS EXISTS. Until 2026-08-12 the thesis's headline tables were aggregated from
`20260721_ladder/{iso,emp}-past`. That arm did not condition on the observed history:
`--cond-past` fell back silently to the previous window's RESIDUAL field, so it trained
C = [R_prev | Yhat] while Chapter 4 Eq. (4.2) defines C = [X | Yhat]. See commit 1e253cc
and docs/conditioning_report.md. The replacement campaign `20260728_histcond` is the arm
the methodology describes, and this script is the single place the chapter's numbers come
from, so the mistake cannot recur silently.

    history conditioning  ->  results/20260728_histcond/{iso,emp,rsh}-hist
    forecast-only         ->  results/20260721_ladder/{iso,emp,rsh}
    prev-residual         ->  results/20260721_ladder/{iso,emp,rsh}-past   (appendix only)

Locked protocol for every number: best_crps.pt, ancestral@50, k=100, y==0 masked, test
split, seed-mean over the diffusion seeds that both arms of a pair evaluated.

usage: python scripts/analysis/thesis_tables.py            # all tables to stdout
       python scripts/analysis/thesis_tables.py main dist  # just those
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results"

HIST = "20260728_histcond"
LADDER = "20260721_ladder"

# display order and pretty names
DS_MAIN = ["pems08f", "pems08s", "pems04f", "pems04s"]
DS_EXTRA = ["metrla", "pemsbay"]
DS_ALL = DS_MAIN + DS_EXTRA
BB = ["mlp", "gcn", "gwn"]
DS_NAME = {"pems08f": "PEMS08 flow", "pems08s": "PEMS08 speed",
           "pems04f": "PEMS04 flow", "pems04s": "PEMS04 speed",
           "metrla": "METR-LA", "pemsbay": "PEMS-BAY"}
BB_NAME = {"mlp": "linear", "gcn": "GCN", "gwn": "GWN"}
PRIOR_DIR = {"iso": "isotropic", "emp": "kronecker", "rsh": "kronecker"}


def _read(p):
    """zero-byte jsons exist where the cluster killed a run mid-write."""
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def load(campaign: str, arm: str) -> dict:
    """cell[(ds, bb)][seed] = metrics dict, for one arm directory."""
    out = defaultdict(dict)
    base = ROOT / campaign / arm
    for p in base.glob("*/*/*/seed*/eval/test_metrics_bestcrps_ancestral50.json"):
        ds, bb, seed = p.parts[-6], p.parts[-5], int(p.parts[-3][4:])
        d = _read(p)
        if d:
            out[(ds, bb)][seed] = d
    return out


ARMS = {
    ("hist", "iso"): (HIST, "iso-hist"), ("hist", "emp"): (HIST, "emp-hist"),
    ("hist", "rsh"): (HIST, "rsh-hist"),
    ("fo", "iso"): (LADDER, "iso"), ("fo", "emp"): (LADDER, "emp"),
    ("fo", "rsh"): (LADDER, "rsh"),
    ("prev", "iso"): (LADDER, "iso-past"), ("prev", "emp"): (LADDER, "emp-past"),
    ("prev", "rsh"): (LADDER, "rsh-past"),
}
_CACHE: dict = {}


def arm(cond: str, prior: str) -> dict:
    if (cond, prior) not in _CACHE:
        spec = ARMS.get((cond, prior))
        _CACHE[(cond, prior)] = load(*spec) if spec else {}
    return _CACHE[(cond, prior)]


def paired(cond, ds, bb, metric):
    """(iso_mean, emp_mean, n_common_seeds, per-seed emp-minus-iso list)."""
    a, b = arm(cond, "iso").get((ds, bb), {}), arm(cond, "emp").get((ds, bb), {})
    common = sorted(set(a) & set(b))
    common = [s for s in common if metric in a[s] and metric in b[s]]
    if not common:
        return None
    av = [a[s][metric] for s in common]
    bv = [b[s][metric] for s in common]
    return float(np.mean(av)), float(np.mean(bv)), len(common), list(zip(av, bv))


def signtest(w, n):
    """one-sided P(X >= w) under Binom(n, 1/2)."""
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def fmt_p(p):
    if p > 0.99:
        return "$>0.99$"
    if p >= 0.01:
        return f"{p:.2f}".lstrip("0")
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    if round(m) == 10:                  # 9.7e-7 must print as 1e-6, not 10e-7
        m, e = 1.0, e + 1
    return f"${m:.0f}\\times10^{{{e}}}$"


def bold_lower(x, y, nd):
    """bold the smaller of the pair; ties bold neither."""
    fx, fy = f"{x:.{nd}f}", f"{y:.{nd}f}"
    if fx == fy:
        return fx, fy
    return (f"\\textbf{{{fx}}}", fy) if x < y else (fx, f"\\textbf{{{fy}}}")


def rows(cond, datasets=DS_ALL):
    for ds in datasets:
        for bb in BB:
            if paired(cond, ds, bb, "crps"):
                yield ds, bb


# ─── tab:exp_main — accuracy ──────────────────────────────────────────────────

def t_main(cond="hist"):
    nd = {"crps": 4}
    print(f"% ---- tab:exp_main  ({cond} conditioning) ----")
    last = None
    for ds, bb in rows(cond):
        head = DS_NAME[ds] if ds != last else ""
        if ds in DS_EXTRA and last not in DS_EXTRA and last is not None:
            print("\\midrule")
        last = ds
        cells = []
        for m, dg in (("crps", 4), ("mae", None), ("rmse", None)):
            i, e, n, _ = paired(cond, ds, bb, m)
            d = 4 if m == "crps" else (3 if max(i, e) < 10 else 2)
            si, se = bold_lower(i, e, d)
            cells += [si, se, f"${'+' if i >= e else '-'}${abs(100 * (i - e) / i):.1f}"]
        print(f"{head:12s} & {BB_NAME[bb]:6s} & " + " & ".join(cells) + " \\\\")
    print()


# ─── tab:exp_dist — calibration + joint ──────────────────────────────────────

def t_dist(cond="hist"):
    print(f"% ---- tab:exp_dist  ({cond} conditioning) ----")
    last = None
    for ds, bb in rows(cond):
        head = DS_NAME[ds] if ds != last else ""
        if ds in DS_EXTRA and last not in DS_EXTRA and last is not None:
            print("\\midrule")
        last = ds
        ci, ce, _, _ = paired(cond, ds, bb, "coverage")
        wi, we, _, _ = paired(cond, ds, bb, "winkler")
        ei, ee, _, _ = paired(cond, ds, bb, "energy")
        # coverage: both arms under-cover, so HIGHER is closer to nominal
        sci = f"\\textbf{{{ci:.3f}}}" if ci > ce else f"{ci:.3f}"
        sce = f"\\textbf{{{ce:.3f}}}" if ce > ci else f"{ce:.3f}"
        dw = 3 if max(wi, we) < 100 else 1
        swi, swe = bold_lower(wi, we, dw)
        sei, see = bold_lower(ei, ee, 2)
        print(f"{head:12s} & {BB_NAME[bb]:6s} & {sci} & {sce} & {swi} & {swe} & "
              f"${'+' if wi >= we else '-'}${abs(100 * (wi - we) / wi):.1f} & "
              f"{sei} & {see} & "
              f"${'+' if ei >= ee else '-'}${abs(100 * (ei - ee) / ei):.1f} \\\\")
    print()


# ─── tab:exp_paired — sign tests ─────────────────────────────────────────────

def t_paired(cond="hist"):
    print(f"% ---- tab:exp_paired  ({cond} conditioning) ----")
    spec = [("MAE (ensemble mean)", "mae", "lo"),
            ("RMSE (ensemble mean)", "rmse", "lo"),
            ("CRPS (full distribution)", "crps", "lo"),
            ("Energy (joint)", "energy", "lo"),
            ("Winkler (intervals)", "winkler", "lo"),
            ("Lower coverage", "coverage", "cov")]
    for label, m, direction in spec:
        out = []
        for datasets in (DS_MAIN, DS_ALL):
            w = tot = 0
            for ds, bb in rows(cond, datasets):
                r = paired(cond, ds, bb, m)
                if not r:
                    continue
                for a, b in r[3]:
                    tot += 1
                    w += (b < a) if direction == "lo" else (b < a)
            if direction == "cov":
                out += [f"{w} / {tot}", "---"]
            else:
                out += [f"{w} / {tot}", fmt_p(signtest(w, tot))]
        print(f"{label:26s} & " + " & ".join(out) + " \\\\")
    print()


# ─── tab:exp_cond — conditioning against the prior ───────────────────────────

def t_cond():
    print("% ---- tab:exp_cond  (forecast-only vs history, and the prior margin under each) ----")
    last = None
    for ds in DS_ALL:
        for bb in BB:
            fo = paired("fo", ds, bb, "crps")
            hi = paired("hist", ds, bb, "crps")
            if not (fo and hi):
                continue
            head = DS_NAME[ds] if ds != last else ""
            if ds in DS_EXTRA and last not in DS_EXTRA and last is not None:
                print("\\midrule")
            last = ds
            gain = 100 * (fo[0] - hi[0]) / fo[0]
            mfo = 100 * (fo[0] - fo[1]) / fo[0]
            mhi = 100 * (hi[0] - hi[1]) / hi[0]
            print(f"{head:12s} & {BB_NAME[bb]:6s} & {fo[0]:.4f} & {hi[0]:.4f} & "
                  f"${'+' if gain >= 0 else '-'}${abs(gain):.1f}\\,\\% & "
                  f"${'+' if mfo >= 0 else '-'}${abs(mfo):.1f}\\,\\% & "
                  f"${'+' if mhi >= 0 else '-'}${abs(mhi):.1f}\\,\\% \\\\")
    print()


# ─── appendix: seed spread ───────────────────────────────────────────────────

def t_seedspread(cond="hist"):
    print(f"% ---- appendix seed spread  ({cond} conditioning) ----")
    last = None
    for ds, bb in rows(cond):
        a = arm(cond, "iso")[(ds, bb)]
        b = arm(cond, "emp")[(ds, bb)]
        common = sorted(set(a) & set(b))
        av = np.array([a[s]["crps"] for s in common])
        bv = np.array([b[s]["crps"] for s in common])
        head = DS_NAME[ds] if ds != last else ""
        last = ds
        f = lambda v, win: (f"$\\mathbf{{{v.mean():.4f}}}\\pm{v.std():.4f}$" if win
                            else f"${v.mean():.4f}\\pm{v.std():.4f}$")
        print(f"{head:12s} & {BB_NAME[bb]:6s} & {f(av, av.mean() < bv.mean())} & "
              f"{f(bv, bv.mean() < av.mean())} & {len(common)} \\\\")
    print()


# ─── tab:exp_sota — our two rows, hour-average + 15/30/60 min ────────────────

def t_sota(cond="hist"):
    print(f"% ---- tab:exp_sota  our rows ({cond} conditioning, GWN backbone) ----")
    H = [2, 5, 11]                                   # 15, 30, 60 min at 5-min bins
    for ds in DS_MAIN:
        print(f"% {DS_NAME[ds]}")
        for prior, label in (("emp", "\\method{} (ours)"), ("iso", "isotropic control")):
            cells = arm(cond, prior).get((ds, "gwn"))
            if not cells:
                print(f"%   MISSING {ds}/{prior}")
                continue
            seeds = sorted(cells)
            def avg(key):
                return np.mean([cells[s][key] for s in seeds], axis=0)
            # Avg RMSE is the root of the horizon-averaged squared error
            rmse_avg = float(np.sqrt(np.mean(avg("mse_h"))))
            rmse_h = np.sqrt(avg("mse_h"))
            mae_avg, mae_h = float(avg("mae")), avg("mae_h")
            crps_avg, crps_h = float(avg("crps")), avg("crps_h")
            nd = 2 if rmse_avg >= 10 else 2
            vals = ([f"{rmse_avg:.{nd}f}"] + [f"{rmse_h[h]:.{nd}f}" for h in H]
                    + [f"{mae_avg:.2f}"] + [f"{mae_h[h]:.2f}" for h in H]
                    + [f"{crps_avg:.4f}"] + [f"{crps_h[h]:.4f}" for h in H])
            print(f"{label} & " + " & ".join(vals) + " \\\\")
    print()


# ─── prose facts the chapters quote ──────────────────────────────────────────

def t_facts(cond="hist"):
    print(f"% ---- prose facts ({cond}) ----")
    for m, direction in (("crps", "lo"), ("mae", "lo"), ("coverage", "cov")):
        for datasets, name in ((DS_MAIN, "PEMS"), (DS_ALL, "all 6")):
            w = tot = 0
            cellw = cellt = 0
            for ds, bb in rows(cond, datasets):
                r = paired(cond, ds, bb, m)
                cellt += 1
                cellw += (r[1] < r[0]) if direction != "cov" else (r[1] < r[0])
                for a, b in r[3]:
                    tot += 1
                    w += b < a
            print(f"  {m:9s} {name:6s}: seeds {w}/{tot} (p={signtest(w, tot):.3g})  "
                  f"cells {cellw}/{cellt}")
    print("  per-cell CRPS margin, structured minus isotropic (+ = structure better):")
    for ds in DS_ALL:
        line = f"    {ds:9s}"
        for bb in BB:
            r = paired(cond, ds, bb, "crps")
            line += f"  {BB_NAME[bb]:6s} {100 * (r[0] - r[1]) / r[0]:+6.2f}" if r else f"  {BB_NAME[bb]:6s}   ---- "
        print(line)
    # backbone-averaged conditioning gain, for the three-channel figure
    print("  conditioning gain (iso arm, fo -> hist), per backbone:")
    for bb in BB:
        g = [100 * (paired("fo", ds, bb, "crps")[0] - paired("hist", ds, bb, "crps")[0])
             / paired("fo", ds, bb, "crps")[0]
             for ds in DS_ALL if paired("fo", ds, bb, "crps") and paired("hist", ds, bb, "crps")]
        if g:
            print(f"    {BB_NAME[bb]:6s} mean {np.mean(g):+6.2f}%  range [{min(g):+.2f}, {max(g):+.2f}]  n={len(g)}")
    print("  prior margin (emp vs iso), per backbone and conditioning:")
    for cnd in ("fo", "hist"):
        for bb in BB:
            g = [100 * (paired(cnd, ds, bb, "crps")[0] - paired(cnd, ds, bb, "crps")[1])
                 / paired(cnd, ds, bb, "crps")[0]
                 for ds in DS_ALL if paired(cnd, ds, bb, "crps")]
            if g:
                print(f"    {cnd:4s} {BB_NAME[bb]:6s} mean {np.mean(g):+6.2f}%  "
                      f"cells won {sum(x > 0 for x in g)}/{len(g)}")
    print()


TABLES = {"main": t_main, "dist": t_dist, "paired": t_paired, "cond": t_cond,
          "seedspread": t_seedspread, "sota": t_sota, "facts": t_facts}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TABLES)
    for name in want:
        if name not in TABLES:
            sys.exit(f"unknown table {name!r}; pick from {', '.join(TABLES)}")
        TABLES[name]()
