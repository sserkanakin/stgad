"""figures of the experiments chapter, drawn from the aggregated result files.
one figure per section, few panels, thesis colours. cpu, mac.

    .venv/bin/python scripts/analysis/ch5_figures.py            # all figures
    .venv/bin/python scripts/analysis/ch5_figures.py prior      # only the ones whose name contains 'prior'

inputs:  results/analysis/prior/<ds>.json, ladder_summary.json, pooled_summary.json,
         results_dossier.json, prior_structures.json (exact prior fit), prior_alone.json
outputs: tudelft_thesis_paper/figures/results/exp_*.pdf
"""
import sys, json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

INK, MUTED, RULE, ACC, WARM, AMBER, GREEN, VIOLET = "#1B2430", "#8A94A0", "#D6DCE2", "#00A6D6", "#C8321E", "#D99A1E", "#2E8B57", "#6A4FBF"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42, "axes.titlesize": 8})
names = {"pems08f": "PEMS08 flow", "pems08s": "PEMS08 speed", "pems04f": "PEMS04 flow", "pems04s": "PEMS04 speed", "metrla": "METR-LA", "pemsbay": "PEMS-BAY"}
short = {"pems08f": "P08 flow", "pems08s": "P08 speed", "pems04f": "P04 flow", "pems04s": "P04 speed", "metrla": "METR-LA", "pemsbay": "PEMS-BAY"}
DS = list(names); cols = [ACC, VIOLET, GREEN, AMBER, WARM, INK]; x = np.arange(6)
OUT = "tudelft_thesis_paper/figures/results"; os.makedirs(OUT, exist_ok=True)
only = sys.argv[1] if len(sys.argv) > 1 else ""

def load(name):
    p = f"results/analysis/{name}"
    return json.load(open(p)) if os.path.exists(p) else None
prior = {ds: load(f"prior/{ds}.json") for ds in DS}; lad = load("ladder_summary.json"); pooled = load("pooled_summary.json")
dos = load("results_dossier.json"); struct = load("prior_structures.json"); alone = load("prior_alone.json")

def xticks(ax):
    ax.set_xticks(x); ax.set_xticklabels([short[d] for d in DS], rotation=35, ha="right", fontsize=6.5)
def pct(a, b): return 100 * (b / a - 1)
def m(c, k): return c[k]["mean"]

# ---------------- 5.2 the residual field: three panels ----------------
if "residual" in only or not only:
    fig, axs = plt.subplots(1, 3, figsize=(6.2, 2.3), constrained_layout=True)
    ax = axs[0]
    for j, ds in enumerate(DS):
        d = np.array(prior[ds]["A1_delta_h_test"]); ax.plot(range(1, 13), np.sqrt(d / d[0]), "-", color=cols[j], lw=1.1, label=names[ds])
    ax.set_xlabel("lead time (steps of 5 min)"); ax.set_ylabel("error spread ÷ spread at 5 min"); ax.set_xticks([1, 6, 12])
    ax.set_title("(a) the error grows along the horizon", loc="left"); ax.legend(fontsize=5.5, ncol=2, loc="upper left", handlelength=1.5)
    ax = axs[1]
    for j, ds in enumerate(DS):
        c = prior[ds]["A2_corr_by_hops"]; ax.plot(range(1, 6), c["test"], "o-", color=cols[j], lw=1.0, ms=2.5)
    ax.axhline(0, color=MUTED, lw=0.8, ls="--"); ax.set_xticks(range(1, 6)); ax.set_xticklabels(["1", "2", "3", "4", "5+"])
    ax.set_xlabel("distance on the road graph (hops)"); ax.set_ylabel("correlation between errors"); ax.set_title("(b) neighbours err alike", loc="left")
    ax = axs[2]
    ax.bar(x - 0.18, [100 * prior[ds]["A3_common_share"]["test"] for ds in DS], 0.36, color=WARM, label="measured")
    ax.bar(x + 0.18, [100 * prior[ds]["A3_common_share"]["independent"] for ds in DS], 0.36, color=MUTED, label="if errors were independent")
    xticks(ax); ax.set_ylabel("% of error variance shared\nby the whole network"); ax.set_title("(c) a network-wide part", loc="left")
    ax.set_ylim(0, 9.5); ax.legend(fontsize=6, loc="upper right", handlelength=1.2)
    fig.savefig(f"{OUT}/exp_residual_structure.pdf"); plt.close(fig); print("exp_residual_structure")

# ---------------- 5.3 the prior's fit: exact formula, four priors ----------------
if ("prior" in only or not only) and struct is not None and len(struct) == 6:
    scales = struct["pems08f"]["scales"]; lab = {"iso": "white noise", "road08": "road graph only", "matched": "Kronecker prior (used)", "cartesian": "product-graph prior"}
    pc = {"iso": MUTED, "road08": AMBER, "matched": ACC, "cartesian": VIOLET}
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.4), constrained_layout=True)
    ax = axs[0]
    for p in ("iso", "road08", "cartesian", "matched"):
        R = np.array([struct[ds]["priors"][p]["ratio_test"] for ds in DS]); ax.plot(range(6), R.mean(0), "o-", color=pc[p], lw=1.2, ms=2.5, label=lab[p])
        ax.fill_between(range(6), R.min(0), R.max(0), color=pc[p], alpha=0.15, lw=0)
    ax.axhline(1, color=INK, lw=0.7, ls=":"); ax.set_xticks(range(6)); ax.set_xticklabels(["sensor", "1-hop", "64 reg.", "16 reg.", "4 reg.", "network"], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("spread of the sum: prior ÷ measured"); ax.set_title("(a) six sizes of sum, test windows", loc="left"); ax.legend(fontsize=5.5, loc="lower left", handlelength=1.5)
    ax = axs[1]
    for j, ds in enumerate(DS):
        ax.plot(range(1, 13), struct[ds]["priors"]["matched"]["total_profile_test"], "-", color=cols[j], lw=1.1, label=names[ds])
    ax.axhline(1, color=INK, lw=0.7, ls=":"); ax.set_xticks([1, 6, 12]); ax.set_xlabel("lead time (steps of 5 min)"); ax.set_ylabel("network total: prior ÷ measured")
    ax.set_title("(b) network total, by lead time", loc="left"); ax.legend(fontsize=5, ncol=1, loc="upper right", handlelength=1.5)
    ax = axs[2]
    for k, p in enumerate(("road08", "cartesian", "matched")):
        ax.bar(x + (k - 1) * 0.26, [struct[ds]["priors"]["iso"]["nll_test"] - struct[ds]["priors"][p]["nll_test"] for ds in DS], 0.26, color=pc[p], label=lab[p])
    xticks(ax); ax.set_ylabel("nats per entry"); ax.set_title("(c) likelihood gain", loc="left"); ax.set_ylim(0, 1.3); ax.legend(fontsize=5.5, loc="upper left", handlelength=1.2)
    fig.savefig(f"{OUT}/exp_prior_fit.pdf"); plt.close(fig); print("exp_prior_fit")

# ---------------- 5.5 the network total: three panels ----------------
if "total" in only or not only:
    g = lad["gwn"]; fig, axs = plt.subplots(1, 3, figsize=(6.2, 2.5), constrained_layout=True)
    ax = axs[0]; v = [pct(m(g[f"{ds}/iso"], "joint_tot_crps"), m(g[f"{ds}/kron"], "joint_tot_crps")) for ds in DS]
    ax.bar(x, v, 0.55, color=ACC); ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylim(min(v) * 1.25, 0.4)
    for i, val in enumerate(v): ax.text(i, val - 0.12, f"{val:.1f}", ha="center", va="top", fontsize=6.5, color=INK)
    ax.set_title("(a) CRPS of the network total\nchange under the prior (%)", loc="left")
    for ax, key, title, ref in ((axs[1], "joint_tot_coverage", "(b) coverage of the 90 % interval\nnetwork total", 0.9), (axs[2], "joint_tot_spread_ratio", "(c) ensemble spread ÷ error\nnetwork total", 1.0)):
        ax.bar(x - 0.18, [m(g[f"{ds}/iso"], key) for ds in DS], 0.36, color=MUTED); ax.bar(x + 0.18, [m(g[f"{ds}/kron"], key) for ds in DS], 0.36, color=ACC)
        ax.axhline(ref, color=INK, lw=0.8, ls=":"); ax.text(5.6, ref + 0.015, "target", fontsize=6, ha="right", color=INK); ax.set_ylim(0, ref * 1.12); ax.set_title(title, loc="left")
    for ax in axs: xticks(ax)
    fig.legend(handles=[Patch(color=MUTED, label="white noise"), Patch(color=ACC, label="Kronecker prior")], loc="outside lower center", ncol=2, fontsize=7)
    fig.savefig(f"{OUT}/exp_total.pdf"); plt.close(fig); print("exp_total")

# ---------------- 5.5 sums over regions: two panels ----------------
if "pooled" in only or not only:
    g = lad["gwn"]; fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4), constrained_layout=True); sc = ["c64", "c16", "c4", "tot"]
    ax = axs[0]
    for j, ds in enumerate(DS):
        p = pooled[ds]; y = [pct(m(g[f"{ds}/iso"], "crps"), m(g[f"{ds}/kron"], "crps"))] + [pct(p["iso"][f"joint_{s}_crps"], p["kron"][f"joint_{s}_crps"]) for s in sc]
        ax.plot(range(5), y, "o-", color=cols[j], lw=1.1, ms=2.5, label=names[ds])
    ax.axhline(0, color=MUTED, lw=0.8, ls="--"); ax.set_ylabel("CRPS change under the prior (%)"); ax.set_title("(a) the advantage grows with the sum", loc="left"); ax.legend(fontsize=5.5, ncol=2, loc="lower left", handlelength=1.5)
    ax = axs[1]
    for j, ds in enumerate(DS):
        p = pooled[ds]; y = [100 * (m(g[f"{ds}/kron"], "coverage") - m(g[f"{ds}/iso"], "coverage"))] + [100 * (p["kron"][f"joint_{s}_coverage"] - p["iso"][f"joint_{s}_coverage"]) for s in sc]
        ax.plot(range(5), y, "o-", color=cols[j], lw=1.1, ms=2.5)
    ax.axhline(0, color=MUTED, lw=0.8, ls="--"); ax.set_ylabel("coverage gain under the prior (points)"); ax.set_title("(b) so does the coverage gain", loc="left")
    for ax in axs: ax.set_xticks(range(5)); ax.set_xticklabels(["single\nsensor", "64 regions", "16 regions", "4 regions", "whole\nnetwork"], fontsize=6.5)
    fig.savefig(f"{OUT}/exp_pooled.pdf"); plt.close(fig); print("exp_pooled")

# ---------------- 5.6 three backbones: what the prior buys, and why ----------------
if "ladder" in only or not only:
    fig, axs = plt.subplots(1, 3, figsize=(6.2, 2.4), constrained_layout=True); bb = [("mlp", "linear", "#F2B8B0"), ("gcn", "GCN", "#E26E5D"), ("gwn", "Graph WaveNet", WARM)]
    for ax, key, title in ((axs[0], "crps", "(a) per-node CRPS\nchange under the prior (%)"), (axs[1], "joint_tot_crps", "(b) CRPS of the network total\nchange under the prior (%)")):
        for k, (b, lab_, col) in enumerate(bb):
            ax.bar(x + (k - 1) * 0.26, [pct(m(lad[b][f"{ds}/iso"], key), m(lad[b][f"{ds}/kron"], key)) for ds in DS], 0.26, color=col, label=lab_)
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_title(title, loc="left")
    ax = axs[2]; extra = load("common_share_extra.json") or {}   # the two pems-bay cells without a prior-fit file
    def share(ds, b):                                              # test windows, same split as the residual figure
        pf = load(f"prior/{ds}.json" if b == "gwn" else f"prior/{ds}_{b}.json")
        return pf["A3_common_share"]["test"] if pf else extra[f"{ds}/{b}"]["common_share"]
    for k, (b, lab_, col) in enumerate(bb):
        ax.bar(x + (k - 1) * 0.26, [100 * share(ds, b) for ds in DS], 0.26, color=col, label=lab_)
    ax.set_title("(c) network-wide share of\nthe error variance (%)", loc="left")
    for ax in axs: xticks(ax)
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in bb], loc="outside lower center", ncol=3, fontsize=7)
    fig.savefig(f"{OUT}/exp_ladder.pdf"); plt.close(fig); print("exp_ladder")
