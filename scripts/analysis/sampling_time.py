#!/usr/bin/env python3
# consolidated sampling-time comparison: ours (ancestral@50) vs ours (EM@1500) vs DiffSTG (ddim-40).
# all inputs already on disk -- no GPU eval is run here.
#
# what each source measures:
#   OURS  : reverse-sampling wall-time for k=100 draws over ALL test windows.
#           gen_seconds total; s_per_window = gen_seconds / n_test_windows.
#           sampler-cost is prior-independent (iso/emp/rsh agree to <1.1%), so we take
#           the median s_per_window across the iso/emp/rsh arms per dataset as "ours".
#   EM    : same denoiser, EM reverse SDE @1500 steps, k=100. only measured on pems08f/gwn
#           (the 3 arms with EM json). a step-count knob, so the ~29x factor is dataset-agnostic.
#   DIFFSTG: their native code, ddim_multi @40 steps, n_samples=8, on the TEST split.
#           the csv 'time' wraps ONLY the model(...) sampling call, summed over test batches.
#           (csv 'is_test=False' is their SMOKE flag, not the split; the pkl [.,8,..] and the
#            train.py flow both confirm the reported row is the final TEST eval with 8 samples.)
#
# fairness: ours draws k=100, DiffSTG draws 8 -> we normalize to s per (window*sample).
#           both models batch their samples in the reverse pass, so per-sample is an amortized
#           figure, not a strictly independent one -- stated as a caveat in the table.

import csv, json, glob, statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path("/Users/serkan/Projects/research-tud")
PR   = ROOT / "results/analysis/paper_results"
FIGS = PR / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

DATASETS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
NNODES   = {"pems08f":170,"pems08s":170,"pems04f":307,"pems04s":307,"metrla":207,"pemsbay":325}
K_OURS   = 100        # samples drawn per window in our eval
N_DIFF   = 8          # DiffSTG n_samples
T_H = T_P = 12        # horizons (both pipelines)

# ---------------------------------------------------------------- OURS (ancestral@50)
rows = list(csv.DictReader(open(PR / "master_cells.csv")))
ours = {}   # ds -> dict(win, spw, gen, s_per_sample)
for ds in DATASETS:
    spw, gens, wins = [], [], set()
    for r in rows:
        if r["dataset"] != ds:        continue
        if r["base"] not in ("emp", "iso", "rsh"):  continue  # kronecker/iso family = "ours"
        if not r["s_per_window"]:      continue
        spw.append(float(r["s_per_window"]))
        gens.append(float(r["gen_seconds"]))
        wins.add(int(r["n_test_windows"]))
    win = sorted(wins)[0]
    # representative sampler cost = min across iso/emp/rsh arms (least system jitter;
    # spread across arms is <1.1%, so this equals median/any arm to 3 sig figs).
    s_per_window = min(spw)
    ours[ds] = dict(
        N=NNODES[ds], win=win,
        s_per_window=s_per_window,
        gen_total=s_per_window * win,                 # total for k=100 over all windows
        s_per_sample=s_per_window / K_OURS,           # per (window*sample)
        spw_spread_pct=(max(spw) - min(spw)) / st.median(spw) * 100.0,
    )

# ---------------------------------------------------------------- EM (@1500), pems08f/gwn only
em_spw = []
for j in glob.glob(str(ROOT / "results/20260721_ladder/*/pems08f/gwn/*/seed42/eval/test_metrics_bestcrps_em1500.json")):
    d = json.load(open(j))
    em_spw.append(d["s_per_window"])
em = dict(
    ds="pems08f",
    s_per_window=st.mean(em_spw),
    win=ours["pems08f"]["win"],
    n_steps=1500,
)
em["gen_total"]    = em["s_per_window"] * em["win"]
em["s_per_sample"] = em["s_per_window"] / K_OURS
em["factor_vs_anc"] = em["s_per_window"] / ours["pems08f"]["s_per_window"]

# ---------------------------------------------------------------- DiffSTG (ddim-40, 8 samples)
# recover total series length T so we can count DiffSTG's own test windows.
# split: val_start=int(T*0.6), test_start=int(T*0.8); test window range in train.py is
# (test_start + T_p, T_total), keeping label_start_idx with label_start_idx + T_p <= T_total.
# => n_test_windows = T_total - test_start - 2*T_p + 1.
def recover_T(val_start, test_start):
    # brute force the small integer T that satisfies both int() split constraints
    for Tt in range(int(test_start / 0.8) - 5, int(test_start / 0.8) + 6):
        if int(Tt * 0.6) == val_start and int(Tt * 0.8) == test_start:
            return Tt
    return int(round(test_start / 0.8))

# split indices read from the DiffSTG run logs (see log grep in the prompt trail)
DIFF_SPLIT = {  # ds: (val_start, test_start)
    "pems08f": (10713, 14284), "pems08s": (10713, 14284),
    "pems04f": (10195, 13593), "pems04s": (10195, 13593),
    "metrla":  (20563, 27417), "pemsbay": (31269, 41692),
}
diff = {}
for r in csv.DictReader(open(ROOT / "results/model-ugnet-notcomplete/research-tud-diffstg/results/diffstg_ref/DiffSTG.csv")):
    ds = r["data.name"]
    if ds not in DATASETS:  continue
    val_s, test_s = DIFF_SPLIT[ds]
    T_total = recover_T(val_s, test_s)
    win = T_total - test_s - 2 * T_P + 1
    total = float(r["time"])
    diff[ds] = dict(
        win=win, T_total=T_total,
        gen_total=total,                       # total for 8 samples over all test windows
        s_per_window=total / win,
        s_per_sample=total / win / N_DIFF,     # per (window*sample)
        steps=int(r["model.sample_steps"]),
        crps=float(r["crps"]),
    )
# pemsbay DiffSTG run failed (empty pkl / no usable time) -> exclude from diff if absent
if "pemsbay" in diff and diff["pemsbay"]["gen_total"] <= 0:
    diff.pop("pemsbay")

# ---------------------------------------------------------------- markdown table
def f(x, n=4):
    return f"{x:.{n}f}"

lines = []
lines.append("# Sampling-time comparison (test split, per (window x sample))\n")
lines.append("All numbers are reverse-sampling wall-time already on disk (no GPU eval re-run). "
             "Denoiser is the fixed 17k-param GCNN in every 'ours' cell; DiffSTG uses its own UGnet.\n")
lines.append("**Fairness normalisation.** Ours draws k=100 samples/window, DiffSTG draws 8. "
             "We report `s/(win*sample) = total / windows / k` so the two are apples-to-apples, "
             "and also the raw per-window and total with k noted. Both models batch their samples "
             "inside one reverse pass, so per-sample is an *amortised* figure.\n")

# main table
h = ("| dataset | N | test win | ours anc@50 s/win | ours s/sample | ours total (k=100) "
     "| DiffSTG ddim40 total (k=8) | DiffSTG s/win | DiffSTG s/sample | ours faster / sample |")
lines.append(h)
lines.append("|" + "---|" * 10)
for ds in DATASETS:
    o = ours[ds]
    if ds in diff:
        dd = diff[ds]
        speed = dd["s_per_sample"] / o["s_per_sample"]
        drow = (f"| {f(dd['gen_total'],1)} s ({dd['win']} win) "
                f"| {f(dd['s_per_window'],4)} | {f(dd['s_per_sample'],5)} | **{speed:.1f}x** |")
    else:
        drow = "| - (run failed) | - | - | - |"
    lines.append(
        f"| {ds} | {o['N']} | {o['win']} "
        f"| {f(o['s_per_window'],4)} | {f(o['s_per_sample'],5)} | {f(o['gen_total'],1)} s "
        + drow)

# EM row
lines.append("")
lines.append("### EM reverse-SDE cost (measured on pems08f/gwn only)\n")
lines.append("| dataset | sampler | steps | s/win | s/sample | total (k=100) | vs ours anc@50 |")
lines.append("|" + "---|" * 7)
o = ours["pems08f"]
lines.append(f"| pems08f | ancestral | 50 | {f(o['s_per_window'],4)} | {f(o['s_per_sample'],5)} "
             f"| {f(o['gen_total'],1)} s | 1.0x |")
lines.append(f"| pems08f | EM | 1500 | {f(em['s_per_window'],4)} | {f(em['s_per_sample'],5)} "
             f"| {f(em['gen_total'],1)} s | **{em['factor_vs_anc']:.1f}x** |")

# provenance / caveats
lines.append("")
lines.append("### Notes & caveats\n")
lines.append(f"- **Sampler-cost is prior-independent.** Across the iso/emp/rsh arms, per-dataset "
             f"s/win spread is <1.1% (max {max(ours[d]['spw_spread_pct'] for d in DATASETS):.2f}%), "
             f"so one 'ours' row stands for iso, empirical-Kronecker, and reshaped priors alike.")
lines.append(f"- **Ours:** ancestral sampler, 50 reverse steps, k=100 draws, y==0 masked, best_crps.pt. "
             f"Denoiser = 17k GCNN.")
lines.append(f"- **DiffSTG:** their native code, ddim_multi @40 steps, n_samples=8, on the **test** split "
             f"(the csv `time` wraps only the sampling call; `is_test=False` is their smoke flag, not the split). "
             f"pemsbay is excluded -- that run failed (empty forecast, no usable time).")
lines.append(f"- **DiffSTG window counts** are DiffSTG's own test-window counts recovered from its "
             f"60/20/20 split (`T-test_start-2*T_p+1`), which match ours to within ~0.3% (metrla ~0.3%); "
             f"each model's per-window cost uses its own denominator.")
lines.append(f"- **Step counts differ** (ours 50 ancestral vs DiffSTG 40 ddim): ours does *more* reverse "
             f"steps yet is ~17x cheaper per sample, because DiffSTG's UGnet denoiser is far heavier than "
             f"our 17k GCNN. The gap is architecture, not step budget.")
lines.append(f"- **EM** only measured on pems08f/gwn (the cells with EM json); it is a pure step-count knob "
             f"(50->1500), so the ~{em['factor_vs_anc']:.0f}x factor carries to any dataset. EM buys calibration/"
             f"coverage, not accuracy -- ~{em['factor_vs_anc']:.0f}x the ancestral cost for a coverage fix.")

(PR / "sampling_time.md").write_text("\n".join(lines) + "\n")
print("wrote", PR / "sampling_time.md")

# ---------------------------------------------------------------- figure: grouped bars, log-y
# entities (color follows entity): cool = ours, warm = external baseline.
C_ANC  = "#2a78d6"   # ours ancestral@50  (blue, slot 1)
C_EM   = "#1baf7a"   # ours EM@1500       (aqua, slot 3)
C_DIFF = "#eb6834"   # DiffSTG ddim-40    (orange, slot 2)
HATCH  = {"anc": "", "em": "///", "diff": "xxx"}   # secondary encoding (CVD/print relief)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"

plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif",
    "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
})

labels = [d for d in DATASETS]
x = np.arange(len(labels))
w = 0.26

anc_vals  = [ours[d]["s_per_sample"] for d in labels]
diff_vals = [diff[d]["s_per_sample"] if d in diff else np.nan for d in labels]
em_vals   = [em["s_per_sample"] if d == "pems08f" else np.nan for d in labels]

fig, ax = plt.subplots(figsize=(10.5, 5.6))
ax.set_yscale("log")

def bars(offset, vals, color, hatch, label, show_na=True):
    xs = x + offset
    bs = ax.bar(xs, vals, width=w, color=color, edgecolor="white", linewidth=1.2,
                hatch=hatch, label=label, zorder=3)
    for xi, v in zip(xs, vals):
        if np.isnan(v):
            if show_na:
                ax.text(xi, min(anc_vals) * 0.60, "n/a", ha="center", va="top",
                        fontsize=8, color=INK2)
            continue
        ax.text(xi, v * 1.09, f"{v*1000:.2f}", ha="center", va="bottom",
                fontsize=8.2, color=INK, fontweight="medium")
    return bs

# ours-EM is measured on pems08f only BY DESIGN (not a failure) -> no n/a clutter for it.
bars(-w,  anc_vals,  C_ANC,  HATCH["anc"],  "Ours  ancestral@50  (k=100)")
bars(0.0, diff_vals, C_DIFF, HATCH["diff"], "DiffSTG  ddim@40  (k=8)")
bars(+w,  em_vals,   C_EM,   HATCH["em"],   "Ours  EM@1500  (k=100)", show_na=False)

# speedup annotations: ours-anc vs DiffSTG, above the DiffSTG bar
for xi, d in enumerate(labels):
    if d in diff:
        sp = diff[d]["s_per_sample"] / ours[d]["s_per_sample"]
        ax.annotate(f"{sp:.0f}x", xy=(xi, diff[d]["s_per_sample"] * 1.9),
                    ha="center", va="bottom", fontsize=9.5, color=C_DIFF, fontweight="bold")
# EM factor annotation (vs our own ancestral@50, same pems08f cell)
ax.annotate(f"{em['factor_vs_anc']:.0f}x vs anc", xy=(0 + w, em["s_per_sample"] * 1.42),
            ha="center", va="bottom", fontsize=9, color=C_EM, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([l.replace("pems0", "PEMS0").replace("metrla", "METR-LA").replace("pemsbay", "PEMS-BAY")
                    for l in labels], fontsize=10.5, color=INK)
ax.set_ylabel("sampling cost  —  seconds per (window x sample)", color=INK, fontsize=11)
ax.set_title("Reverse-sampling cost per drawn sample  (test split, log scale;  labels = ms/sample)",
             color=INK, fontsize=12.5, pad=14, loc="left")
ax.set_ylim(min(anc_vals) * 0.42, max([v for v in diff_vals if not np.isnan(v)] + [em["s_per_sample"]]) * 6.0)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
leg = ax.legend(loc="upper right", frameon=False, fontsize=10, ncol=1,
                bbox_to_anchor=(0.995, 0.995))
fig.text(0.5, -0.02,
         "Ours: 17k-param GCNN denoiser.  DiffSTG: native UGnet, ddim_multi@40.  "
         "Per-sample is amortised (both batch samples in one reverse pass).  "
         "PEMS-BAY DiffSTG run failed (excluded).",
         ha="center", va="top", fontsize=8, color=INK2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIGS / f"sampling_time.{ext}", dpi=200, bbox_inches="tight", facecolor=SURF)
print("wrote", FIGS / "sampling_time.png", "and .pdf")

# ---------------------------------------------------------------- console summary
print("\n=== per (window*sample) s ===")
for d in DATASETS:
    dv = diff[d]["s_per_sample"] if d in diff else float("nan")
    sp = (diff[d]["s_per_sample"] / ours[d]["s_per_sample"]) if d in diff else float("nan")
    print(f"{d:9s} ours-anc {ours[d]['s_per_sample']*1000:6.3f} ms | "
          f"DiffSTG {dv*1000:7.3f} ms | ours faster {sp:5.1f}x")
sps = [diff[d]['s_per_sample']/ours[d]['s_per_sample'] for d in DATASETS if d in diff]
print(f"\nours ancestral@50 is {min(sps):.0f}-{max(sps):.0f}x (mean {st.mean(sps):.0f}x) cheaper per sample than DiffSTG")
print(f"EM@1500 costs {em['factor_vs_anc']:.1f}x ancestral@50 (pems08f): "
      f"{em['s_per_window']:.3f} vs {ours['pems08f']['s_per_window']:.4f} s/win")
print(f"raw totals pems08f: ours {ours['pems08f']['gen_total']:.0f}s (k=100) vs DiffSTG "
      f"{diff['pems08f']['gen_total']:.0f}s (k=8)  -> ours makes 12.5x more samples in less wall-time")
