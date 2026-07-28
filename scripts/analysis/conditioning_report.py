r"""The conditioning picture: history+forecast (past) vs forecast-only (fo).

Reads results/analysis/paper_results/master_cells.csv (written by
paper_results_aggregate.py) and answers one question across the whole grid:

    what does adding the observed history window to the conditioning buy,
    per dataset x backbone x prior, under the locked test protocol?

  cond=fo    C = Yhat            (B,N,H)      -- forecast-only
  cond=past  C = [X_z | Yhat_z]  (B,N,P+H)    -- history + forecast

Only the 17k GCNN denoiser cells of the 20260721_ladder campaign are used, so
the denoiser, loss, schedule, sampler and normalization are identical across the
two arms; the conditioning flag is the only difference (verified separately).

Comparisons are PAIRED on the diffusion seed -- a cell contributes only the
seeds that both arms evaluated. Deltas are (past - fo)/fo in percent, so
negative = history conditioning is better.

usage: .venv/bin/python scripts/analysis/conditioning_report.py
"""
from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CELLS = os.path.join(ROOT, "results", "analysis", "paper_results", "master_cells.csv")
OUT = os.path.join(ROOT, "results", "analysis", "conditioning")
os.makedirs(OUT, exist_ok=True)

CAMPAIGN = "20260721_ladder"
DATASETS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
BACKBONES = ["mlp", "gcn", "gwn"]
PRIORS = ["iso", "emp", "rsh"]
# lower is better for all of these except coverage (nominal 0.90, closer is better)
METRICS = ["crps", "mae", "rmse", "crps_energy", "w1", "energy", "coverage"]


def load():
    """cell[(prior, cond, ds, bb, seed)] = {metric: value} for the 17k-GCNN ladder."""
    cell = {}
    with open(CELLS) as f:
        for r in csv.DictReader(f):
            if r["campaign"] != CAMPAIGN:
                continue
            if r["whit"] == "1" or r["ugnet"] == "1":
                continue  # whitened-loss and 996k-UGnet variants are separate axes
            if r["base"] not in PRIORS:
                continue
            k = (r["base"], r["cond"], r["dataset"], r["backbone"], int(r["seed"]))
            cell[k] = {m: float(r[m]) for m in METRICS if r.get(m) not in (None, "")}
    return cell


def paired(cell, prior, ds, bb, metric):
    """(fo_mean, past_mean, pct_delta, n_seeds, n_seeds_past_better) on common seeds."""
    fo = {s: v[metric] for (p, c, d, b, s), v in cell.items()
          if (p, c, d, b) == (prior, "fo", ds, bb) and metric in v}
    pa = {s: v[metric] for (p, c, d, b, s), v in cell.items()
          if (p, c, d, b) == (prior, "past", ds, bb) and metric in v}
    common = sorted(set(fo) & set(pa))
    if not common:
        return None
    f = statistics.mean(fo[s] for s in common)
    p = statistics.mean(pa[s] for s in common)
    if metric == "coverage":  # closer to 0.90 wins; report signed distance change
        wins = sum(1 for s in common if abs(pa[s] - 0.90) < abs(fo[s] - 0.90))
        return f, p, None, len(common), wins
    wins = sum(1 for s in common if pa[s] < fo[s])
    return f, p, 100.0 * (p - f) / f, len(common), wins


def fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    cell = load()
    lines = []
    w = lines.append

    w("# Conditioning: history+forecast vs forecast-only\n")
    w(f"campaign `{CAMPAIGN}`, 17k GCNN denoiser, locked test protocol")
    w("(best_crps.pt, ancestral@50, k=100, y==0 masked). Paired on diffusion seed;")
    w("residual/backbone seed pinned to 42 everywhere. **Negative delta = history helps.**\n")

    # ---- per-metric grids ------------------------------------------------
    for metric in ["crps", "mae", "rmse", "energy", "w1"]:
        w(f"\n## {metric.upper()}  (past vs forecast-only, % change)\n")
        w("| dataset | backbone | " + " | ".join(
            f"{p} fo | {p} past | {p} d% | n" for p in PRIORS) + " |")
        w("|" + "---|" * (2 + 4 * len(PRIORS)))
        for ds in DATASETS:
            for bb in BACKBONES:
                cols, any_ = [], False
                for p in PRIORS:
                    r = paired(cell, p, ds, bb, metric)
                    if r is None:
                        cols += ["-", "-", "-", "0"]
                        continue
                    any_ = True
                    f, pm, d, n, _ = r
                    nd = 4 if metric == "crps" else 3
                    cols += [fmt(f, nd), fmt(pm, nd), f"{d:+.1f}", str(n)]
                if any_:
                    w(f"| {ds} | {bb} | " + " | ".join(cols) + " |")

    # ---- coverage --------------------------------------------------------
    w("\n## Coverage @ nominal 0.90 (higher = less under-coverage)\n")
    w("| dataset | backbone | " + " | ".join(f"{p} fo | {p} past" for p in PRIORS) + " |")
    w("|" + "---|" * (2 + 2 * len(PRIORS)))
    for ds in DATASETS:
        for bb in BACKBONES:
            cols, any_ = [], False
            for p in PRIORS:
                r = paired(cell, p, ds, bb, "coverage")
                if r is None:
                    cols += ["-", "-"]
                    continue
                any_ = True
                cols += [fmt(r[0], 3), fmt(r[1], 3)]
            if any_:
                w(f"| {ds} | {bb} | " + " | ".join(cols) + " |")

    # ---- backbone rollup -------------------------------------------------
    w("\n## Rollup: mean CRPS delta by backbone (the substitution law)\n")
    w("| backbone | " + " | ".join(f"{p} mean d% | cells past-better" for p in PRIORS) + " |")
    w("|" + "---|" * (1 + 2 * len(PRIORS)))
    for bb in BACKBONES:
        cols = []
        for p in PRIORS:
            ds_ = [paired(cell, p, ds, bb, "crps") for ds in DATASETS]
            ds_ = [r for r in ds_ if r]
            if not ds_:
                cols += ["-", "-"]
                continue
            cols += [f"{statistics.mean(r[2] for r in ds_):+.1f}",
                     f"{sum(1 for r in ds_ if r[2] < 0)}/{len(ds_)}"]
        w(f"| {bb} | " + " | ".join(cols) + " |")

    # ---- prior gap under each conditioning --------------------------------
    w("\n## Does conditioning change the prior verdict? (structured vs iso, same cond)\n")
    w("emp-vs-iso CRPS delta, %; negative = the structured Kronecker prior beats iso.\n")
    w("| dataset | backbone | emp-vs-iso (fo) | emp-vs-iso (past) | rsh-vs-iso (fo) | rsh-vs-iso (past) |")
    w("|---|---|---|---|---|---|")
    for ds in DATASETS:
        for bb in BACKBONES:
            cols, any_ = [], False
            for p in ["emp", "rsh"]:
                for cond in ["fo", "past"]:
                    a = {s: v["crps"] for (pp, c, d, b, s), v in cell.items()
                         if (pp, c, d, b) == (p, cond, ds, bb)}
                    i = {s: v["crps"] for (pp, c, d, b, s), v in cell.items()
                         if (pp, c, d, b) == ("iso", cond, ds, bb)}
                    common = sorted(set(a) & set(i))
                    if not common:
                        cols.append("-")
                        continue
                    any_ = True
                    am = statistics.mean(a[s] for s in common)
                    im = statistics.mean(i[s] for s in common)
                    cols.append(f"{100 * (am - im) / im:+.1f}")
            if any_:
                w(f"| {ds} | {bb} | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} |")

    # ---- coverage of the grid --------------------------------------------
    w("\n## Grid coverage (paired seeds available per cell)\n")
    w("| dataset | backbone | " + " | ".join(PRIORS) + " |")
    w("|" + "---|" * (2 + len(PRIORS)))
    holes = []
    for ds in DATASETS:
        for bb in BACKBONES:
            cols = []
            for p in PRIORS:
                r = paired(cell, p, ds, bb, "crps")
                cols.append(str(r[3]) if r else "**0**")
                if not r:
                    holes.append((ds, bb, p))
            w(f"| {ds} | {bb} | " + " | ".join(cols) + " |")
    w(f"\nmissing cells: {len(holes)} -> " +
      (", ".join(f"{d}/{b}/{p}" for d, b, p in holes) if holes else "none"))

    txt = "\n".join(lines) + "\n"
    path = os.path.join(OUT, "conditioning_report.md")
    with open(path, "w") as f:
        f.write(txt)
    print(txt)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
