r"""The conditioning picture: what the diffusion gains from each conditioning field.

Reads results/analysis/paper_results/master_cells.csv (written by
paper_results_aggregate.py) and reports, across dataset x backbone x prior, the
locked-protocol test numbers for every conditioning arm on disk:

  fo     C = Yhat_z             forecast only                     20260721_ladder/<p>
  hist   C = [X_z | Yhat_z]     observed history + forecast       20260728_histcond/<p>-hist
  prev   C = [R_prev | Yhat_z]  previous window's RESIDUAL        20260721_ladder/<p>-past

`prev` is RETIRED from the thesis (user decision, 2026-07-28). It exists because
--cond-past silently fell back to the previous window's residual field whenever a
residual cache had no history channel, and no cache ever had one -- so the whole
20260721_ladder "-past" arm trained that, not the history conditioning its name
implies (fixed in commit 1e253cc). Its numbers are still printed, clearly labelled,
because they are real runs and the fallback is a defensible ablation -- but they are
not the "history conditioning" result.

Only 17k-GCNN-denoiser cells are used, so denoiser, loss, schedule, sampler and
normalization are identical across arms; the conditioning field is the only
difference. Comparisons are PAIRED on the diffusion seed. Deltas are (arm - fo)/fo
in percent, so negative = that conditioning beats forecast-only.

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

LADDER = "20260721_ladder"
HISTCAMP = "20260728_histcond"
DATASETS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
BACKBONES = ["mlp", "gcn", "gwn"]
PRIORS = ["iso", "emp"]                       # the two arms the campaign covers
PRIOR_LABEL = {"iso": "iso", "emp": "kron(L⊗Σ_T)", "rsh": "rsh"}
CONDS = ["fo", "hist", "prev"]
COND_LABEL = {"fo": "forecast-only", "hist": "history+forecast", "prev": "prev-resid (RETIRED)"}
METRICS = ["crps", "mae", "rmse", "crps_energy", "w1", "energy", "coverage"]


def load():
    """cell[(prior, cond, ds, bb, seed)] = {metric: value}."""
    cell = {}
    with open(CELLS) as f:
        for r in csv.DictReader(f):
            if r["whit"] == "1" or r["ugnet"] == "1":
                continue
            camp, base, cond = r["campaign"], r["base"], r["cond"]
            if camp == LADDER:
                cond = {"fo": "fo", "past": "prev"}.get(cond)
            elif camp == HISTCAMP:
                cond = {"past": "hist", "hist": "hist", "fo": "fo"}.get(cond)
            else:
                continue
            if cond is None or base not in PRIOR_LABEL:
                continue
            k = (base, cond, r["dataset"], r["backbone"], int(r["seed"]))
            cell[k] = {m: float(r[m]) for m in METRICS if r.get(m) not in (None, "")}
    return cell


def seeds_of(cell, prior, cond, ds, bb, metric):
    return {s: v[metric] for (p, c, d, b, s), v in cell.items()
            if (p, c, d, b) == (prior, cond, ds, bb) and metric in v}


def paired(cell, prior, ds, bb, metric, cond):
    """(fo_mean, arm_mean, pct_delta, n_common) on seeds both arms evaluated."""
    fo = seeds_of(cell, prior, "fo", ds, bb, metric)
    ar = seeds_of(cell, prior, cond, ds, bb, metric)
    common = sorted(set(fo) & set(ar))
    if not common:
        return None
    f = statistics.mean(fo[s] for s in common)
    a = statistics.mean(ar[s] for s in common)
    d = None if metric == "coverage" else 100.0 * (a - f) / f
    return f, a, d, len(common)


def present_conds(cell):
    have = {c for (_, c, _, _, _) in cell}
    return [c for c in CONDS if c in have]


def main():
    cell = load()
    conds = present_conds(cell)
    lines = []
    w = lines.append

    w("# Conditioning: what each conditioning field buys\n")
    w("17k GCNN denoiser, locked test protocol (best_crps.pt, ancestral@50, k=100,")
    w("y==0 masked). Paired on diffusion seed; backbone/residual seed pinned to 42.")
    w("**Negative delta = better than forecast-only.**\n")
    w("| arm | conditioning | shape | status |")
    w("|---|---|---|---|")
    w("| `fo` | `C = Ŷ` | (B,N,H) | headline |")
    w("| `hist` | `C = [X \\| Ŷ]` | (B,N,P+H) | headline |")
    w("| `prev` | `C = [R_prev \\| Ŷ]` | (B,N,2H) | RETIRED — silent fallback, not history |")
    w("")
    if "hist" not in conds:
        w("> **The `hist` arm has no test evals on disk yet.** Everything below is the")
        w("> forecast-only picture plus the retired prev-residual arm. Re-run this script")
        w(f"> once `{HISTCAMP}` lands and the history columns fill in automatically.\n")

    # ---- absolute forecast-only table (the standing result) ---------------
    w("\n## Forecast-only, absolute (test CRPS)\n")
    w("| dataset | " + " | ".join(f"{PRIOR_LABEL[p]} / {bb}" for bb in BACKBONES for p in PRIORS) + " |")
    w("|" + "---|" * (1 + len(BACKBONES) * len(PRIORS)))
    for ds in DATASETS:
        cols = []
        for bb in BACKBONES:
            for p in PRIORS:
                s = seeds_of(cell, p, "fo", ds, bb, "crps")
                cols.append(f"{statistics.mean(s.values()):.4f} ({len(s)})" if s else "—")
        w(f"| {ds} | " + " | ".join(cols) + " |")
    w("\n(n seeds in parentheses)")

    # ---- per-metric deltas vs forecast-only -------------------------------
    for metric in ["crps", "mae", "energy"]:
        for cond in [c for c in conds if c != "fo"]:
            w(f"\n## {metric.upper()}: {COND_LABEL[cond]} vs forecast-only (% change)\n")
            w("| dataset | backbone | " + " | ".join(
                f"{PRIOR_LABEL[p]} fo | {PRIOR_LABEL[p]} {cond} | Δ% | n" for p in PRIORS) + " |")
            w("|" + "---|" * (2 + 4 * len(PRIORS)))
            for ds in DATASETS:
                for bb in BACKBONES:
                    cols, any_ = [], False
                    for p in PRIORS:
                        r = paired(cell, p, ds, bb, metric, cond)
                        if r is None:
                            cols += ["—", "—", "—", "0"]
                            continue
                        any_ = True
                        f, a, d, n = r
                        nd = 4 if metric == "crps" else 3
                        cols += [f"{f:.{nd}f}", f"{a:.{nd}f}", f"{d:+.1f}", str(n)]
                    if any_:
                        w(f"| {ds} | {bb} | " + " | ".join(cols) + " |")

    # ---- backbone rollup ---------------------------------------------------
    w("\n## Rollup: mean CRPS delta vs forecast-only, by backbone\n")
    hdr = "| backbone | " + " | ".join(
        f"{PRIOR_LABEL[p]} {c}" for c in conds if c != "fo" for p in PRIORS) + " |"
    w(hdr)
    w("|" + "---|" * (1 + len([c for c in conds if c != 'fo']) * len(PRIORS)))
    for bb in BACKBONES:
        cols = []
        for cond in [c for c in conds if c != "fo"]:
            for p in PRIORS:
                rs = [paired(cell, p, ds, bb, "crps", cond) for ds in DATASETS]
                rs = [r for r in rs if r]
                cols.append(f"{statistics.mean(r[2] for r in rs):+.1f} ({sum(1 for r in rs if r[2] < 0)}/{len(rs)})"
                            if rs else "—")
        w(f"| {bb} | " + " | ".join(cols) + " |")
    w("\n(mean % change, and how many datasets it beat forecast-only in)")

    # ---- does the conditioning move the prior verdict? --------------------
    w("\n## Does the conditioning change the iso-vs-structured verdict?\n")
    w("kron-minus-iso CRPS, %; negative = the structured prior wins under that conditioning.\n")
    w("| dataset | backbone | " + " | ".join(COND_LABEL[c] for c in conds) + " |")
    w("|" + "---|" * (2 + len(conds)))
    for ds in DATASETS:
        for bb in BACKBONES:
            cols, any_ = [], False
            for cond in conds:
                a = seeds_of(cell, "emp", cond, ds, bb, "crps")
                i = seeds_of(cell, "iso", cond, ds, bb, "crps")
                common = sorted(set(a) & set(i))
                if not common:
                    cols.append("—")
                    continue
                any_ = True
                am = statistics.mean(a[s] for s in common)
                im = statistics.mean(i[s] for s in common)
                cols.append(f"{100 * (am - im) / im:+.1f}")
            if any_:
                w(f"| {ds} | {bb} | " + " | ".join(cols) + " |")

    # ---- coverage ---------------------------------------------------------
    w("\n## Coverage @ nominal 0.90\n")
    w("| dataset | backbone | " + " | ".join(
        f"{PRIOR_LABEL[p]} {c}" for c in conds for p in PRIORS) + " |")
    w("|" + "---|" * (2 + len(conds) * len(PRIORS)))
    for ds in DATASETS:
        for bb in BACKBONES:
            cols, any_ = [], False
            for cond in conds:
                for p in PRIORS:
                    s = seeds_of(cell, p, cond, ds, bb, "coverage")
                    if s:
                        any_ = True
                        cols.append(f"{statistics.mean(s.values()):.3f}")
                    else:
                        cols.append("—")
            if any_:
                w(f"| {ds} | {bb} | " + " | ".join(cols) + " |")

    # ---- grid coverage ----------------------------------------------------
    w("\n## Grid completeness (seeds with a locked-protocol test eval)\n")
    w("| dataset | backbone | " + " | ".join(
        f"{PRIOR_LABEL[p]} {c}" for c in conds for p in PRIORS) + " |")
    w("|" + "---|" * (2 + len(conds) * len(PRIORS)))
    holes = []
    for ds in DATASETS:
        for bb in BACKBONES:
            cols = []
            for cond in conds:
                for p in PRIORS:
                    n = len(seeds_of(cell, p, cond, ds, bb, "crps"))
                    cols.append(str(n) if n else "**0**")
                    if not n:
                        holes.append(f"{ds}/{bb}/{PRIOR_LABEL[p]}/{cond}")
            w(f"| {ds} | {bb} | " + " | ".join(cols) + " |")
    w(f"\nmissing: {len(holes)}" + (" -> " + ", ".join(holes) if holes else " — grid complete"))

    txt = "\n".join(lines) + "\n"
    path = os.path.join(OUT, "conditioning_report.md")
    with open(path, "w") as f:
        f.write(txt)
    print(txt)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
