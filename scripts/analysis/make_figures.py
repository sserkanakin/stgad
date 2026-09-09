"""Thesis figures from the measured JSONs (no recomputation). writes results/figures/<name>.{pdf,png}.
prior/field figures read results/analysis/prior/<ds>.json (prior_fidelity.py);
result figures read results/20260907_matched/**/test_metrics_*.json, results/analysis/matched_summary.json,
results/analysis/gaussian_baseline.json and the archived histcond evals for the old-vs-new figure.
each figure is skipped (with a note) when its inputs are missing, so this can be re-run as data lands."""
import json, glob, sys, collections
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
DS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
NAME = {"pems08f": "PEMS08 flow", "pems08s": "PEMS08 speed", "pems04f": "PEMS04 flow", "pems04s": "PEMS04 speed", "metrla": "METR-LA", "pemsbay": "PEMS-BAY"}
PRIORS = [("iso", "isotropic (white)", "#888888", "--"), ("temporal", "temporal only", "#c9a227", ":"), ("road08", "road graph, γ=0.8", "#1f77b4", "-."), ("matched", "road + global edge, matched", "#d62728", "-")]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 110})

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", bbox_inches="tight"); plt.close(fig); print("  wrote", name)

prior = {ds: json.load(open(f)) for ds in DS if (f := Path(f"results/analysis/prior/{ds}.json")).exists()}
print(f"prior jsons: {list(prior)}")

# ── 1. the residual field: horizon growth + horizon correlation ─────────────────────────
if prior:
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={"width_ratios": [1.3, 1]})
    for ds, j in prior.items():
        d = np.array(j["A1_delta_h_train"]); ax[0].plot(range(1, 13), np.sqrt(d / d[0]), "o-", ms=3, label=NAME[ds])
    ax[0].set_xlabel("horizon step h (5 min)"); ax[0].set_ylabel("residual sd at h / sd at h=1"); ax[0].set_title("error grows along the horizon"); ax[0].legend(ncol=2)
    ds0 = "pems08f" if "pems08f" in prior else next(iter(prior)); R = np.array(prior[ds0]["R_H"])
    im = ax[1].imshow(R, vmin=0, vmax=1, cmap="viridis"); ax[1].set_title(f"horizon correlation R_H ({NAME[ds0]})"); ax[1].set_xlabel("h'"); ax[1].set_ylabel("h"); fig.colorbar(im, ax=ax[1], fraction=0.046)
    save(fig, "field_horizon")

# ── 2. correlation vs graph distance + common-mode share ─────────────────────────────────
if prior:
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for ds, j in prior.items():
        c = j["A2_corr_by_hops"]; ax[0].plot(range(1, 6), c["train"], "o-", ms=3, label=NAME[ds])
    ax[0].set_xticks(range(1, 6)); ax[0].set_xticklabels(["1", "2", "3", "4", "5+"]); ax[0].set_xlabel("road-graph distance (hops)"); ax[0].set_ylabel("mean error correlation"); ax[0].set_title("neighbours err together, weakly"); ax[0].axhline(0, color="k", lw=0.5); ax[0].legend(ncol=2)
    x = np.arange(len(prior)); w = 0.25
    ax[1].bar(x - w, [100 * j["A3_common_share"]["train"] for j in prior.values()], w, label="data (train)", color="#d62728")
    ax[1].bar(x, [100 * j["A3_common_share"]["test"] for j in prior.values()], w, label="data (test)", color="#ff9896")
    ax[1].bar(x + w, [100 * j["A3_common_share"]["independent"] for j in prior.values()], w, label="independent errors (1/N)", color="#888888")
    ax[1].set_xticks(x); ax[1].set_xticklabels([NAME[d] for d in prior], rotation=20); ax[1].set_ylabel("% of error variance on\n'all sensors together'"); ax[1].set_title("the shared network-wide component"); ax[1].legend()
    save(fig, "field_common_mode")

# ── 3. the network total: measured sd per horizon vs independence ────────────────────────
if prior:
    n = len(prior); fig, axs = plt.subplots(1, n, figsize=(2.3 * n, 2.6), sharex=True)
    for a, (ds, j) in zip(np.atleast_1d(axs), prior.items()):
        t = j["A4_total_sd"]; a.plot(range(1, 13), t["train"], "o-", ms=3, color="#d62728", label="measured (train)"); a.plot(range(1, 13), t["test"], "s--", ms=3, color="#ff9896", label="measured (test)")
        a.axhline(t["independent"], color="#888888", ls="--", label="if errors were independent"); a.set_title(NAME[ds]); a.set_xlabel("horizon h")
        a.text(0.03, 0.9, f"×{np.mean(t['train']) / t['independent']:.1f}", transform=a.transAxes, fontsize=9, color="#d62728")
    np.atleast_1d(axs)[0].set_ylabel("sd of the network-total error"); np.atleast_1d(axs)[0].legend(fontsize=7, loc="lower right")
    save(fig, "field_total_sd")

# ── 4. graph-spectral variance: data vs priors (GAD view) ────────────────────────────────
if prior:
    n = len(prior); fig, axs = plt.subplots(1, n, figsize=(2.4 * n, 2.8), sharey=False)
    for a, (ds, j) in zip(np.atleast_1d(axs), prior.items()):
        B = j["B1_spectral"]; lam = np.array(B["lambda"]); bins = np.linspace(0, 2.0, 25); idx = np.clip(np.digitize(lam, bins) - 1, 0, len(bins) - 2); xc = 0.5 * (bins[1:] + bins[:-1])
        def binned(v): v = np.array(v); return np.array([v[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bins) - 1)])
        a.plot(xc, binned(B["data_test"]), "k.", ms=5, label="residual (test)")
        for key, lab, col, ls in PRIORS:
            if key in B: a.plot(xc, binned(B[key]), ls=ls, color=col, label=lab)
        a.plot([0], [B["data_test"][0]], "k*", ms=9)                       # exact λ=0 (constant pattern): data ...
        for key, lab, col, ls in PRIORS:                                    # ... and each prior, unbinned
            if key in B: a.plot([0], [B[key][0]], marker="*", ms=7, color=col, ls="none")
        a.set_yscale("log"); a.set_title(NAME[ds]); a.set_xlabel("graph frequency λ")
    np.atleast_1d(axs)[0].set_ylabel("variance per graph frequency"); np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.suptitle("where the error variance sits on the graph spectrum — data vs each prior (★ = the constant pattern, λ=0)", y=1.04)
    save(fig, "prior_spectrum")

# ── 5. multi-scale variance ratio (the key prior figure) ────────────────────────────────
if prior:
    n = len(prior); fig, axs = plt.subplots(1, n, figsize=(2.4 * n, 2.8), sharey=True)
    for a, (ds, j) in zip(np.atleast_1d(axs), prior.items()):
        B = j["B4_multiscale_ratio"]; sc = B["scales"]; x = np.arange(len(sc))
        for key, lab, col, ls in PRIORS:
            if key + "_test" in B: a.plot(x, B[key + "_test"], "o", ls=ls, ms=3, color=col, label=lab)
        a.axhline(1.0, color="k", lw=0.6); a.set_xticks(x); a.set_xticklabels(["sensor", "1-hop", "64 cl.", "16 cl.", "4 cl.", "network"], rotation=45, ha="right"); a.set_title(NAME[ds]); a.set_ylim(0.2, 1.3)
    np.atleast_1d(axs)[0].set_ylabel("prior-implied / measured spread (test)"); np.atleast_1d(axs)[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("is the prior's uncertainty right at every aggregation scale?  (1 = yes)", y=1.04)
    save(fig, "prior_multiscale")

# ── 6. held-out likelihood + separability ───────────────────────────────────────────────
if prior:
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8)); x = np.arange(len(prior)); w = 0.2
    for i, (key, lab, col, ls) in enumerate(PRIORS):
        ax[0].bar(x + (i - 1.5) * w, [j["B5_nll"][key]["nll_test"] - j["B5_nll"]["iso"]["nll_test"] for j in prior.values()], w, label=lab, color=col)
    ax[0].set_xticks(x); ax[0].set_xticklabels([NAME[d] for d in prior], rotation=20); ax[0].set_ylabel("test NLL per point, minus isotropic\n(nats; lower = closer to the data)"); ax[0].legend(fontsize=7); ax[0].axhline(0, color="k", lw=0.5)
    ax[1].bar(x, [100 * j["B3_separability"]["kron_rank1_fraction"] for j in prior.values()], color="#1f77b4"); ax[1].set_xticks(x); ax[1].set_xticklabels([NAME[d] for d in prior], rotation=20); ax[1].set_ylabel("% of the residual covariance\ncaptured by ONE Kronecker product"); ax[1].set_ylim(0, 100)
    save(fig, "prior_nll_separability")

# ── 7. joint results: CRPS_sum, PICP, spread ratio — iso vs kron (+ DC rule) per dataset ─
cells = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob("results/20260907_matched/*/*/gwn/*/seed*/eval/test_metrics_bestcrps_ancestral*.json"):
    m = json.load(open(f)); p = f.split("/"); ds, arm = p[3], "kron" if p[5] == "kronecker" else "iso"
    if f.endswith("_cmgauss.json"): arm += "+DC"
    for k in ("crps", "mae", "coverage", "joint_tot_crps", "joint_tot_coverage", "joint_tot_spread_ratio", "joint_cum_crps", "joint_cum_coverage", "joint_cum_spread_ratio"):
        if m.get(k) is not None: cells[(ds, arm)][k].append(m[k])
if cells:
    arms = [("iso", "#888888", "o"), ("kron", "#d62728", "s"), ("kron+DC", "#9467bd", "^")]
    panels = [("joint_tot_crps", "CRPS_sum, relative to iso mean", True), ("joint_tot_coverage", "90 % interval coverage of the total", False), ("joint_tot_spread_ratio", "spread / error of the total", False)]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    for a, (k, lab, rel) in zip(axs, panels):
        for i, ds in enumerate(DS):
            base = np.mean(cells[(ds, "iso")][k]) if cells[(ds, "iso")][k] else None
            for j, (arm, col, mk) in enumerate(arms):
                v = cells[(ds, arm)][k]
                if not v or base is None: continue
                y = np.array(v) / base if rel else np.array(v)
                a.scatter(np.full(len(y), i + (j - 1) * 0.22), y, color=col, marker=mk, s=18, alpha=0.6); a.plot([i + (j - 1) * 0.22 - 0.08, i + (j - 1) * 0.22 + 0.08], [y.mean()] * 2, color=col, lw=2)
        a.set_xticks(range(len(DS))); a.set_xticklabels([NAME[d] for d in DS], rotation=25, ha="right"); a.set_title(lab)
        a.axhline(1.0 if rel or k.endswith("spread_ratio") else 0.9, color="k", lw=0.6, ls=":")
    for arm, col, mk in arms: axs[0].scatter([], [], color=col, marker=mk, label={"iso": "isotropic", "kron": "Kronecker (matched)", "kron+DC": "Kronecker + prior's filter for the DC mode"}[arm])
    axs[0].legend(fontsize=7, loc="lower left"); fig.suptitle("network-total uncertainty on the test set (seed points and means)", y=1.03)
    save(fig, "results_joint_total")
    # cumulative panel
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    for a, (k, lab, rel) in zip(axs, [("joint_cum_crps", "cumulative CRPS, relative to iso mean", True), ("joint_cum_coverage", "90 % coverage, cumulative per sensor", False), ("joint_cum_spread_ratio", "spread / error, cumulative", False)]):
        for i, ds in enumerate(DS):
            base = np.mean(cells[(ds, "iso")][k]) if cells[(ds, "iso")][k] else None
            for j, (arm, col, mk) in enumerate(arms):
                v = cells[(ds, arm)][k]
                if not v or base is None: continue
                y = np.array(v) / base if rel else np.array(v)
                a.scatter(np.full(len(y), i + (j - 1) * 0.22), y, color=col, marker=mk, s=18, alpha=0.6); a.plot([i + (j - 1) * 0.22 - 0.08, i + (j - 1) * 0.22 + 0.08], [y.mean()] * 2, color=col, lw=2)
        a.set_xticks(range(len(DS))); a.set_xticklabels([NAME[d] for d in DS], rotation=25, ha="right"); a.set_title(lab); a.axhline(1.0 if rel or k.endswith("spread_ratio") else 0.9, color="k", lw=0.6, ls=":")
    save(fig, "results_joint_cumulative")

# ── 8. per point: per-horizon CRPS and coverage, iso vs kron, all datasets ───────────────
perh = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob("results/20260907_matched/*/*/gwn/*/seed*/eval/test_metrics_bestcrps_ancestral[0-9]*.json"):
    if "_cm" in f.split("/")[-1]: continue
    m = json.load(open(f)); p = f.split("/"); ds, arm = p[3], "kron" if p[5] == "kronecker" else "iso"
    for k in ("crps_h", "coverage_h", "mae_h"): perh[(ds, arm)][k].append(m[k])
if perh:
    fig, axs = plt.subplots(2, 6, figsize=(13, 4.2), sharex=True)
    for i, ds in enumerate(DS):
        for arm, col in (("iso", "#888888"), ("kron", "#d62728")):
            if not perh[(ds, arm)]["crps_h"]: continue
            c = np.mean(perh[(ds, arm)]["crps_h"], 0); cv = np.mean(perh[(ds, arm)]["coverage_h"], 0)
            axs[0, i].plot(range(1, 13), c, color=col, label=arm); axs[1, i].plot(range(1, 13), cv, color=col)
        axs[0, i].set_title(NAME[ds]); axs[1, i].axhline(0.9, color="k", lw=0.5, ls=":"); axs[1, i].set_xlabel("horizon h")
    axs[0, 0].set_ylabel("per-point CRPS (normalised)"); axs[1, 0].set_ylabel("90 % coverage per point"); axs[0, 0].legend(fontsize=7)
    save(fig, "results_perpoint_horizon")

# ── 9. old protocol vs new: coverage and CRPS per arm ───────────────────────────────────
old = collections.defaultdict(list)
for f in glob.glob("archive/20260902/results/20260728_histcond/*-hist/*/gwn/*/seed*/eval/test_metrics_*.json"):
    m = json.load(open(f)); p = f.split("/"); arm = "kron" if p[4].startswith("emp") else "iso"; ds = p[5]
    if m.get("crps") is not None: old[(ds, arm)].append(m)
if old and cells:
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3))
    for i, ds in enumerate(DS):
        for j, (arm, col) in enumerate((("iso", "#888888"), ("kron", "#d62728"))):
            if not old[(ds, arm)] or not cells[(ds, arm)]["coverage"]: continue
            o = np.mean([m["coverage"] for m in old[(ds, arm)]]); nn = np.mean(cells[(ds, arm)]["coverage"])
            ax[0].plot([i + (j - 0.5) * 0.3] * 2, [o, nn], color=col, lw=1.5); ax[0].scatter([i + (j - 0.5) * 0.3], [o], color="white", edgecolor=col, zorder=3); ax[0].scatter([i + (j - 0.5) * 0.3], [nn], color=col, zorder=3)
            oc = np.mean([m["crps"] for m in old[(ds, arm)]]); nc = np.mean(cells[(ds, arm)]["crps"])
            ax[1].bar(i + (j - 0.5) * 0.3, 100 * (nc - oc) / oc, 0.28, color=col)
    ax[0].axhline(0.9, color="k", lw=0.5, ls=":"); ax[0].set_ylabel("90 % coverage per point (open = old, filled = new)"); ax[0].set_xticks(range(6)); ax[0].set_xticklabels([NAME[d] for d in DS], rotation=25, ha="right")
    ax[1].set_ylabel("per-point CRPS change, old → new (%)"); ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xticks(range(6)); ax[1].set_xticklabels([NAME[d] for d in DS], rotation=25, ha="right")
    ax[0].scatter([], [], color="#888888", label="isotropic"); ax[0].scatter([], [], color="#d62728", label="Kronecker"); ax[0].legend(fontsize=7)
    fig.suptitle("what the sampler and selection fixes changed (old protocol → new), same backbone and test set", y=1.03)
    save(fig, "results_old_vs_new")

# ── 10. training curves: val CRPS per epoch, pems08f ────────────────────────────────────
hist = glob.glob("results/20260907_matched/*/pems08f/gwn/*/seed*/train/history.csv")
if hist:
    import csv
    fig, ax = plt.subplots(figsize=(5, 3))
    for f in sorted(hist):
        arm = "kron" if "/kronecker/" in f else "iso"; rows = list(csv.DictReader(open(f)))
        key = next((k for k in rows[0] if "crps" in k.lower()), None)
        if key is None: continue
        ep = [int(float(r.get("epoch", i))) for i, r in enumerate(rows)]; v = [float(r[key]) if r[key] not in ("", "nan") else np.nan for r in rows]
        ax.plot(ep, v, color="#d62728" if arm == "kron" else "#888888", alpha=0.7, lw=1)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation CRPS (k=16, sampled)"); ax.set_title("PEMS08 flow: training curves, 4 seeds per arm"); ax.plot([], [], color="#888888", label="isotropic"); ax.plot([], [], color="#d62728", label="Kronecker"); ax.legend(fontsize=7)
    save(fig, "training_curves_pems08f")
print("done ->", OUT)
