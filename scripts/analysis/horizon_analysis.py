"""scores along the horizon: per-node MAE / CRPS / coverage by lead time for the backbone, the two
trained models (mean over seeds) and the prior alone, and the network total by lead time where
available. reads the per-cell metrics files of 20260907_matched (gwn), the residual caches and
results/analysis/prior_alone.json. writes results/analysis/horizon_summary.json and the figure
tudelft_thesis_paper/figures/results/exp_horizon.pdf. cpu, mac."""
import sys, json, glob, os, numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.utils import paths
INK, MUTED, ACC, WARM, AMBER, GREEN, VIOLET = "#1B2430", "#8A94A0", "#00A6D6", "#C8321E", "#D99A1E", "#2E8B57", "#6A4FBF"
plt.rcParams.update({"figure.constrained_layout.use": True, "figure.constrained_layout.w_pad": 0.06, "figure.constrained_layout.wspace": 0.08, "font.size": 8, "font.family": "serif", "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "axes.titlesize": 8})
names = {"pems08f": "PEMS08 flow", "pems08s": "PEMS08 speed", "pems04f": "PEMS04 flow", "pems04s": "PEMS04 speed", "metrla": "METR-LA", "pemsbay": "PEMS-BAY"}; DS = list(names)
alone = json.load(open("results/analysis/prior_alone.json")) if os.path.exists("results/analysis/prior_alone.json") else {}
out = {}
for ds in DS:
    d = np.load(paths.residual_npz(ds, "test", 42, "gwn")); R = d["residuals"]; Y = d["y_pred"] + R; M = d["mask"].astype(bool) & (np.abs(Y) > 1e-6)
    bb_mae_h = [float(np.abs(R[:, :, h][M[:, :, h]]).mean()) for h in range(R.shape[2])]
    cell = {"backbone_mae_h": bb_mae_h}
    for arm, prior in (("iso", "isotropic"), ("kron", "kronecker")):
        fs = [f for f in glob.glob(f"results/20260907_matched/{arm}-hist/{ds}/gwn/{prior}/seed*/eval/test_metrics_bestcrps_ancestral*.json") if "_pooled" not in f and "_cmgauss" not in f and "_horizon" not in f]
        ms = [json.load(open(f)) for f in fs]
        cell[arm] = {k: np.mean([m[k] for m in ms], axis=0).tolist() for k in ("mae_h", "crps_h", "coverage_h", "winkler_h")}; cell[arm]["rmse_h"] = np.sqrt(np.mean([m["mse_h"] for m in ms], axis=0)).tolist(); cell[arm]["n_seeds"] = len(ms)
    if ds in alone and "crps_h" in alone[ds]["kron"]:
        cell["alone"] = {k: alone[ds]["kron"][k] for k in ("mae_h", "crps_h", "coverage_h", "joint_tot_crps_h", "joint_tot_coverage_h", "joint_tot_spread_ratio_h")}
        cell["alone_iso"] = {k: alone[ds]["iso"][k] for k in ("joint_tot_crps_h", "joint_tot_coverage_h", "joint_tot_spread_ratio_h")}
    out[ds] = cell
json.dump(out, open("results/analysis/horizon_summary.json", "w"), indent=1)
h = np.arange(1, 13)
# ---- the figure: per-node scores by lead time, two datasets side by side would be too much: show all six as thin lines relative to h=1? no: pick the ratios ----
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
cols = [ACC, VIOLET, GREEN, AMBER, WARM, INK]
ax = axs[0]   # per-node coverage by lead time: white, kron, prior alone (mean over datasets with range)
for key, lab, col, ls in (("iso", "white noise", MUTED, "-"), ("kron", "Kronecker prior", ACC, "-"), ("alone", "Kronecker prior alone", WARM, "--")):
    A = np.array([out[ds][key]["coverage_h"] for ds in DS if key in out[ds]])
    if len(A): ax.plot(h, A.mean(0), ls, color=col, lw=1.3, label=lab); ax.fill_between(h, A.min(0), A.max(0), color=col, alpha=0.15, lw=0)
ax.axhline(0.9, color=INK, lw=0.7, ls=":"); ax.text(1.2, 0.897, "nominal level", fontsize=6, ha="left", va="top", color=INK); ax.set_ylabel("coverage of the 90 % interval"); ax.set_title("(a) per node", loc="left"); ax.legend(fontsize=6, loc="lower left", handlelength=1.5)
ax = axs[1]   # the network total by lead time: trained models (horizon eval pass) and the prior alone
ht = json.load(open("results/analysis/horizon_totals.json")) if os.path.exists("results/analysis/horizon_totals.json") else {}
for key, lab, col, ls in (("iso", "white noise", MUTED, "-"), ("kron", "Kronecker prior", ACC, "-")):
    A = np.array([ht[f"{ds}/{key}"]["joint_tot_spread_ratio_h"] for ds in DS if f"{ds}/{key}" in ht])
    if len(A): ax.plot(h, A.mean(0), ls, color=col, lw=1.3, label=lab); ax.fill_between(h, A.min(0), A.max(0), color=col, alpha=0.15, lw=0)
A = np.array([out[ds]["alone"]["joint_tot_spread_ratio_h"] for ds in DS if "alone" in out[ds]])
if len(A): ax.plot(h, A.mean(0), "--", color=WARM, lw=1.3, label="Kronecker prior alone")
ax.axhline(1, color=INK, lw=0.7, ls=":"); ax.text(12, 1.01, "right width", fontsize=6, ha="right", va="bottom", color=INK); ax.set_ylabel("network total: spread ÷ error"); ax.set_title("(b) the network total", loc="left"); ax.legend(fontsize=6, loc="upper right", handlelength=1.5)
for ax in axs: ax.set_xticks([1, 6, 12]); ax.set_xlabel("lead time (steps of 5 min)")
fig.savefig("tudelft_thesis_paper/figures/results/exp_horizon.pdf", bbox_inches="tight", pad_inches=0.03); print("written; seeds per cell:", {ds: (out[ds]["iso"]["n_seeds"], out[ds]["kron"]["n_seeds"]) for ds in DS})
