"""Where does each arm lose CRPS? Node / window / horizon / calibration, iso vs emp vs rsh.

Merges two sources:
  - local per-node sampling (scripts/analysis/error_localization.py npz: per-node,
    per-window CRPS at k=32/anc@25 on a 600-window stride subset of test)
  - the DAIC protocol evals (per-horizon crps_h, 5-seed; PIT + reliability npz)

One figure per cell:
  A. per-node CRPS, structured vs iso (log-log): on the diagonal = same places
  B. per-node relative delta, sorted: is the loss uniform or driven by a few nodes
  C. per-window relative delta across the test period: is it a regime/time thing
  D. per-horizon CRPS (DAIC, 5-seed mean): short vs long lead
  E. PIT histogram + coverage: calibration shape (U = under-dispersed)

usage: .venv/bin/python scripts/analysis/where_errors.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                                       # noqa: E402

LOCAL = paths.analysis_dir("error_localization")
FIG = paths.analysis_dir("error_localization")
C_ARM = {"iso": "#666666", "emp": "#0072B2", "rsh": "#D55E00"}
PRIOR = {"iso": "isotropic", "emp": "kronecker", "rsh": "kronecker"}
CELLS = [("pems08f", "gwn"), ("pems08s", "gwn"), ("pems08f", "mlp")]


def daic_crps_h(ds, bb, arm):
    fs = glob.glob(str(ROOT / f"results/20260721_ladder/{arm}/{ds}/{bb}/{PRIOR[arm]}"
                              "/seed*/eval/test_metrics_bestcrps_ancestral50.json"))
    hs = [json.load(open(f))["crps_h"] for f in fs]
    cov = [json.load(open(f))["coverage"] for f in fs]
    return (np.mean(hs, axis=0), np.mean(cov)) if hs else (None, None)


def daic_pit(ds, bb, arm):
    f = ROOT / f"results/20260721_ladder/{arm}/{ds}/{bb}/{PRIOR[arm]}" \
               "/seed42/eval/test_diag_bestcrps_ancestral50.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["pit_hist_all"]


def main():
    for ds, bb in CELLS:
        z = {}
        for arm in ("iso", "emp", "rsh"):
            f = LOCAL / f"{ds}_{bb}_seed42_{arm}.npz"
            if not f.exists():
                print(f"missing {f}, skip cell"); break
            z[arm] = np.load(f)
        else:
            node = {a: z[a]["num_nh"].sum(1) / (z[a]["den_nh"].sum(1) * 19) for a in z}
            win = {a: z[a]["num_w"] / (z[a]["den_w"] * 19) for a in z}
            fig, axes = plt.subplots(1, 5, figsize=(22, 4.0))
            fig.suptitle(f"{ds}/{bb} — where the arms win and lose "
                         f"(local k=32 anc@25 seed42; DAIC 5-seed for D/E)", fontsize=11)

            ax = axes[0]
            lo = min(node["iso"].min(), node["emp"].min()) * 0.9
            hi = max(node["iso"].max(), node["emp"].max()) * 1.1
            ax.plot([lo, hi], [lo, hi], "k-", lw=1, alpha=0.5)
            for arm in ("emp", "rsh"):
                r = np.corrcoef(node["iso"], node[arm])[0, 1]
                ax.scatter(node["iso"], node[arm], s=10, alpha=0.55, color=C_ARM[arm],
                           label=f"{arm}  (corr {r:.3f})")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("iso per-node CRPS"); ax.set_ylabel("structured per-node CRPS")
            ax.legend(fontsize=8, frameon=False)
            ax.set_title("A. same nodes hard for everyone?", fontsize=9)

            ax = axes[1]
            for arm in ("emp", "rsh"):
                d = np.sort((node[arm] - node["iso"]) / node["iso"]) * 100
                ax.plot(d, color=C_ARM[arm], lw=2,
                        label=f"{arm}: {np.mean(d < 0) * 100:.0f}% of nodes win")
            ax.axhline(0, color="#888888", ls=":", lw=1.5)
            ax.set_xlabel("nodes, sorted by delta"); ax.set_ylabel("Δ CRPS vs iso (%)")
            ax.legend(fontsize=8, frameon=False)
            ax.set_title("B. uniform loss or a few bad nodes?", fontsize=9)

            ax = axes[2]
            for arm in ("emp", "rsh"):
                d = (win[arm] - win["iso"]) / win["iso"] * 100
                k = 25
                roll = np.convolve(d, np.ones(k) / k, mode="valid")
                ax.plot(z[arm]["window_idx"][k - 1:], roll, color=C_ARM[arm], lw=1.5, label=arm)
            ax.axhline(0, color="#888888", ls=":", lw=1.5)
            ax.set_xlabel("test window index (time order)")
            ax.set_ylabel("Δ CRPS vs iso (%), rolling 25")
            ax.legend(fontsize=8, frameon=False)
            ax.set_title("C. a period thing?", fontsize=9)

            ax = axes[3]
            for arm in ("iso", "emp", "rsh"):
                ch, cov = daic_crps_h(ds, bb, arm)
                if ch is not None:
                    ax.plot(np.arange(1, len(ch) + 1) * 5, ch, "o-", color=C_ARM[arm],
                            lw=2, ms=4, label=f"{arm} (cov {cov:.2f})")
            ax.set_xlabel("lead time (min)"); ax.set_ylabel("CRPS (DAIC 5-seed)")
            ax.legend(fontsize=8, frameon=False)
            ax.set_title("D. short vs long horizon", fontsize=9)

            ax = axes[4]
            for arm in ("iso", "emp", "rsh"):
                p = daic_pit(ds, bb, arm)
                if p is not None:
                    ax.plot(np.linspace(0.025, 0.975, len(p)), p / p.mean(), "-",
                            color=C_ARM[arm], lw=2, label=arm)
            ax.axhline(1.0, color="#888888", ls=":", lw=1.5)
            ax.set_xlabel("PIT"); ax.set_ylabel("density / uniform")
            ax.legend(fontsize=8, frameon=False)
            ax.set_title("E. calibration (U-shape = under-dispersed)", fontsize=9)

            fig.tight_layout()
            f = FIG / f"where_errors_{ds}_{bb}.png"
            fig.savefig(f, dpi=150); plt.close(fig)
            print(f"wrote {f}")


if __name__ == "__main__":
    main()
