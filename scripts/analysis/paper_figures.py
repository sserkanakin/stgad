"""Paper figures from the banked results — line graphs + distributions, not tables.

Reads local results/ (jsons + diag npz), writes results/analysis/figures/*.png+pdf.
Palette validated (dataviz six-checks): emp #2a78d6, axes-green #2e8540, rsh #8557d6;
iso = neutral gray reference with direct labels (secondary encoding).

usage: .venv/bin/python scripts/analysis/paper_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results"
OUT = ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, PURPLE, GRAY = "#2a78d6", "#2e8540", "#8557d6", "#6e6e66"
INK, MUT = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUT, "axes.labelcolor": INK, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "xtick.color": MUT, "ytick.color": MUT,
})


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def test_crps(exp, arm, ds, bb):
    vs = [jload(p).get("crps") for p in
          (ROOT / exp / arm / ds / bb).glob("*/seed*/eval/test_metrics_bestcrps_ancestral50.json")]
    vs = [v for v in vs if v]
    return float(np.mean(vs)) if vs else None


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png (+pdf)")


# ── F1: one law, four information axes ───────────────────────────────────────
def fig_mechanism():
    bbs = ["mlp", "gcn", "gwn"]
    x = np.arange(3)
    ds_avg = lambda f: [float(np.mean([v for v in (f(ds, bb) for ds in
                        ("pems08f", "pems08s", "pems04f", "pems04s")) if v is not None]))
                        for bb in bbs]
    # gains (%) from adding each information source, per backbone (test, raw protocol)
    prior = ds_avg(lambda ds, bb: (lambda i, e: 100 * (i - e) / i if i and e else None)(
        test_crps("20260721_ladder", "iso", ds, bb), test_crps("20260721_ladder", "emp", ds, bb)))
    cond = [float(np.mean([100 * (test_crps("20260721_ladder", "iso", ds, bb)
                                  - test_crps("20260721_ladder", "iso-past", ds, bb))
                           / test_crps("20260721_ladder", "iso", ds, bb)
                           for ds in ("pems08f", "pems08s")])) for bb in bbs]
    # capacity: gcnn->ugnet on past arms (pems08, partial mlp/gwn only)
    cap = []
    for bb in bbs:
        vals = []
        for ds in ("pems08f", "pems08s"):
            g = test_crps("20260721_ladder", "iso-past", ds, bb)
            # ugnet cells were not test-evaluated; use val ratio as proxy? -> skip gcn
            vals.append(None)
        cap.append(None)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.axhline(0, color=GRAY, lw=1, ls="--")
    ax.annotate("iso baseline", (2.02, 0.4), color=GRAY, fontsize=8)
    ax.plot(x, prior, "-o", color=BLUE, lw=2, ms=5, label="+ structured prior (iso→emp)")
    ax.plot(x, cond, "-s", color=GREEN, lw=2, ms=5, label="+ residual history (fo→past)")
    for xi, yi, t in [(0, prior[0], f"+{prior[0]:.0f}%"), (0, cond[0], f"+{cond[0]:.0f}%")]:
        ax.annotate(t, (xi + 0.05, yi), fontsize=8, color=INK)
    ax.set_xticks(x, ["MLP\n(312 params)", "GCN\n(17k)", "GWN\n(309k)"])
    ax.set_ylabel("test CRPS improvement (%)")
    ax.set_xlabel("backbone (information already absorbed →)")
    ax.set_title("Structure helps in proportion to what the model lacks", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "f1_mechanism")


# ── F2: fan chart — the distribution, visually ──────────────────────────────
def fig_fanchart():
    cells = {"emp (structured)": ("emp", "kronecker", BLUE),
             "iso (baseline)": ("iso", "isotropic", GRAY)}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9), sharey=True)
    for ax, (label, (arm, pr, col)) in zip(axes, cells.items()):
        d = np.load(ROOT / "20260721_ladder" / arm / "pems08f" / "mlp" / pr /
                    "seed42/eval/test_diag_bestcrps_ancestral50.npz", allow_pickle=True)
        w, s = 1, 0                      # example window, sensor
        q = d["ex_quantiles"][:, w, s, :]        # (7 qprobs, H)
        yt = d["ex_ytrue"][w, s, :]
        yp = d["ex_yhat"][w, s, :]
        h = np.arange(1, 13) * 5
        ax.fill_between(h, q[0], q[6], color=col, alpha=0.16, label="90% interval")
        ax.fill_between(h, q[2], q[4], color=col, alpha=0.32, label="50% interval")
        ax.plot(h, q[3], color=col, lw=2, label="predictive median")
        ax.plot(h, yp, color=MUT, lw=1.2, ls="--", label="backbone point forecast")
        ax.plot(h, yt, color=INK, lw=1.4, marker="o", ms=3.5, label="observed")
        ax.set_title(label, fontsize=9, color=col if col != GRAY else MUT)
        ax.set_xlabel("horizon (min)")
    axes[0].set_ylabel("flow (veh / 5 min)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("From a point forecast to a calibrated predictive distribution (PEMS08-F, MLP backbone)",
                 fontsize=10, y=1.04)
    save(fig, "f2_fanchart")


# ── F3: reliability — the calibration story in one panel ─────────────────────
def fig_reliability():
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=MUT, lw=1, ls=":")
    ax.annotate("perfect calibration", (0.52, 0.47), rotation=41, color=MUT, fontsize=7)
    for arm, pr, col, lab in [("iso", "isotropic", GRAY, "iso"),
                              ("emp", "kronecker", BLUE, "emp"),
                              ("rsh", "kronecker", PURPLE, "rsh (unregularized)")]:
        d = np.load(ROOT / "20260721_ladder" / arm / "pems08f" / "gwn" / pr /
                    "seed42/eval/test_diag_bestcrps_ancestral50.npz", allow_pickle=True)
        ax.plot(d["reliab_levels"], d["reliab_cov"], "-o", color=col, lw=1.8, ms=4, label=lab)
    ax.set_xlabel("nominal central-interval level")
    ax.set_ylabel("empirical coverage (test)")
    ax.set_title("Why recalibration: all arms under-cover;\nthe tighter the fit, the more it under-covers", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "f3_reliability")


# ── F4: few-step curves (GAD parity) ────────────────────────────────────────
def fig_fewstep():
    steps = [5, 10, 25, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, bb in zip(axes, ["mlp", "gwn"]):
        for arm, pr, col in [("iso", "isotropic", GRAY), ("emp", "kronecker", BLUE),
                             ("rsh", "kronecker", PURPLE)]:
            ys = []
            for n in steps:
                f = list((ROOT / "20260721_ladder" / arm / "pems08f" / bb).glob(
                    f"*/seed42/eval/val_metrics_bestcrps_ancestral{n}.json"))
                ys.append(jload(f[0])["crps"] if f else np.nan)
            ax.plot(steps, ys, "-o", color=col, lw=1.8, ms=4, label=arm)
            ax.annotate(arm, (steps[-1] * 1.08, ys[-1]), color=col, fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(steps, [str(s) for s in steps])
        ax.set_xlabel("reverse steps (NFE)")
        ax.set_title(f"{bb.upper()} backbone", fontsize=9)
    axes[0].set_ylabel("val CRPS")
    fig.suptitle("Structured priors degrade less at few sampling steps (PEMS08-F)", fontsize=10, y=1.03)
    save(fig, "f4_fewstep")


# ── F5: speed–accuracy (measured, same hardware) ────────────────────────────
def fig_pareto():
    steps = [5, 10, 25, 50, 100]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for arm, pr, col in [("emp", "kronecker", BLUE), ("iso", "isotropic", GRAY)]:
        ts, cs = [], []
        for n in steps:
            f = list((ROOT / "20260721_ladder" / arm / "pems08f" / "gwn" / pr).glob(
                f"seed42/eval/val_metrics_bestcrps_ancestral{n}.json"))
            d = jload(f[0]) if f else {}
            if d.get("s_per_window"):
                ts.append(d["s_per_window"] / d["k_samples"] * 1000)  # ms per sample-window
                cs.append(d["crps"])
        ax.plot(ts, cs, "-o", color=col, lw=1.8, ms=4)
        ax.annotate(arm, (ts[0], cs[0] * 1.004), color=col, fontsize=8)
        for t, c, n in zip(ts, cs, steps):
            if n in (5, 100):
                ax.annotate(f"{n} steps", (t, c * 0.9985), fontsize=7, color=MUT)
    # DiffSTG same-cluster anchor: 815.9 s / 3548 windows @ k=8 (val CRPS 0.063 for pems08f)
    ax.plot([815.9 / 3548 / 8 * 1000], [0.0630], "D", color=GREEN, ms=7)
    ax.annotate("DiffSTG (their code,\nsame cluster)", (815.9 / 3548 / 8 * 1000 * 0.55, 0.0605),
                fontsize=8, color=GREEN)
    ax.set_xscale("log")
    ax.set_xlabel("sampling time (ms per sample-window, measured)")
    ax.set_ylabel("CRPS")
    ax.set_title("Accuracy vs sampling cost (PEMS08-F, GWN)", fontsize=10)
    save(fig, "f5_speed_pareto")


if __name__ == "__main__":
    print("building figures ->", OUT)
    fig_mechanism()
    fig_fanchart()
    fig_reliability()
    fig_fewstep()
    fig_pareto()
    print("done.")


# ── F7: per-horizon analysis — where in the hour the win lives ──────────────
def fig_horizons():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    h = np.arange(1, 13) * 5
    for ax, bb, ttl in [(axes[0], "mlp", "MLP backbone"), (axes[1], "gwn", "GWN backbone")]:
        for arm, pr, col in [("iso", "isotropic", GRAY), ("emp", "kronecker", BLUE)]:
            cs = []
            cell = ROOT / "20260721_ladder" / arm / "pems08f" / bb / pr
            hits = list(cell.glob("seed*/eval/test_metrics_bestcrps_ancestral50_srecal.json")) \
                or list(cell.glob("seed*/eval/test_metrics_bestcrps_ancestral50.json"))
            for p in hits:
                cs.append(jload(p)["crps_h"])
            m = np.mean(cs, axis=0)
            ax.plot(h, m, "-o", color=col, lw=1.8, ms=3.5)
            ax.annotate(arm, (h[-1] + 1.5, m[-1]), color=col, fontsize=8)
        ax.set_xlabel("horizon (min)")
        ax.set_title(ttl, fontsize=9)
    axes[0].set_ylabel("test CRPS")
    fig.suptitle("Per-horizon CRPS (PEMS08-F): the structured margin grows with lead time",
                 fontsize=10, y=1.03)
    save(fig, "f7_horizons")


# ── F8: the diffusion improves the point forecast itself ────────────────────
def fig_backbone_improvement():
    # backbone MAE from the test residuals (mean |R| masked, raw units);
    # ensemble-mean MAE from the eval jsons (recal is mean-preserving).
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bbs = ["mlp", "gcn", "gwn"]
    width = 0.27
    ds = "pems08f"
    bb_mae, iso_mae, emp_mae = [], [], []
    for bb in bbs:
        d = np.load(ROOT / "residuals" / ds / bb / "seed42" / "test.npz")
        R, M = d["residuals"], d["mask"].astype(bool)
        bb_mae.append(float(np.abs(R[M]).mean()))
        for arm, pr, acc in [("iso", "isotropic", iso_mae), ("emp", "kronecker", emp_mae)]:
            vals = [jload(p)["mae"] for p in
                    (ROOT / "20260721_ladder" / arm / ds / bb / pr).glob(
                        "seed*/eval/test_metrics_bestcrps_ancestral50.json")]
            acc.append(float(np.mean(vals)))
    x = np.arange(3)
    ax.bar(x - width, bb_mae, width, color="#c3c2b7", label="backbone point forecast")
    ax.bar(x, iso_mae, width, color=GRAY, label="+ diffusion (iso)")
    ax.bar(x + width, emp_mae, width, color=BLUE, label="+ diffusion (structured)")
    for xi, (b, e) in enumerate(zip(bb_mae, emp_mae)):
        imp = 100 * (b - e) / b
        lab = "±0%" if abs(imp) < 1 else f"{-imp:+.0f}%".replace("+", "+").replace("-", "−") if imp < 0 else f"−{imp:.0f}%"
        ax.annotate(lab, (xi + width, e), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, ["MLP", "GCN", "GWN"])
    ax.set_ylabel("test MAE (veh / 5 min)")
    ax.set_title("The ensemble mean improves the point forecast it refines (PEMS08-F)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "f8_backbone_improvement")


# ── F9: what the prior buys at the strong backbone — training cost ──────────
def fig_convergence():
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    ds = "pems08f"
    for arm, pr, col in [("iso", "isotropic", GRAY), ("emp", "kronecker", BLUE)]:
        curves = []
        for f in (ROOT / "20260721_ladder" / arm / ds / "gwn" / pr).glob(
                "seed*/train/validation_history.json"):
            h = jload(f).get("history", [])
            c = [r.get("val_crps") for r in h if r.get("val_crps")]
            if len(c) >= 30:
                curves.append(c[:100])
        L = min(len(c) for c in curves)
        m = np.mean([c[:L] for c in curves], axis=0)
        ax.plot(np.arange(1, L + 1), m, color=col, lw=2)
        ax.annotate("structured" if arm == "emp" else "isotropic",
                    (L * 0.72, m[int(L * 0.7)] * (0.985 if arm == "emp" else 1.015)),
                    color=col, fontsize=8.5)
    iso_final = np.mean([jload(f).get("best_crps") for f in
                         (ROOT / "20260721_ladder/iso" / ds / "gwn/isotropic").glob(
                             "seed*/train/validation_history.json")])
    ax.axhline(iso_final, color=GRAY, lw=1, ls=":")
    ax.annotate("isotropic final quality", (44, iso_final * 0.999), color=MUT, fontsize=7.5,
                va="top")
    ax.annotate("reached at epoch ~11", (11, iso_final * 1.004), color=INK, fontsize=8,
                ha="left")
    ax.plot([10.6], [iso_final], "o", color=BLUE, ms=6)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("val CRPS (sampled, EMA)")
    ax.set_ylim(iso_final * 0.97, iso_final * 1.25)
    ax.set_title("GWN backbone: the structured prior reaches the isotropic\n"
                 "control's converged quality with ~7× less training (PEMS08-F)", fontsize=9.5)
    save(fig, "f9_convergence")


# ── F10: the residual — what the prior is built on ──────────────────────────
def fig_residual_stats():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ds = "pems08f"
    # (a) per-horizon std growth per backbone
    for bb, col in [("mlp", "#9ec5f4"), ("gcn", "#5598e7"), ("gwn", "#1c5cab")]:
        d = np.load(ROOT / "residuals" / ds / bb / "seed42" / "train.npz")
        R, M = d["residuals"], d["mask"].astype(bool)
        s = [R[:, :, h][M[:, :, h]].std() for h in range(12)]
        s = np.array(s) / s[0]
        axes[0].plot(np.arange(1, 13) * 5, s, "-o", color=col, lw=1.8, ms=3.5)
        axes[0].annotate(bb.upper(), (61, s[-1]), color=col, fontsize=8)
    axes[0].set_xlabel("horizon (min)"); axes[0].set_ylabel("residual std (relative to 5 min)")
    axes[0].set_title("scale grows with lead time", fontsize=9)
    # (b) horizon correlation matrix (gwn)
    d = np.load(ROOT / "residuals" / ds / "gwn" / "seed42" / "train.npz")
    R, M = d["residuals"].astype(np.float32), d["mask"].astype(bool)
    keep = M.all(axis=(1, 2)); X = R[keep][:4000].reshape(-1, 12)
    Ch = np.corrcoef(X.T)
    im = axes[1].imshow(Ch, cmap="Blues", vmin=0, vmax=1, origin="lower")
    axes[1].set_xlabel("horizon"); axes[1].set_ylabel("horizon")
    axes[1].set_title("horizons err together (GWN)", fontsize=9)
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.suptitle("The residual field the prior models (PEMS08-F, training split)", fontsize=10, y=1.04)
    save(fig, "f10_residual_stats")


# ── F1 v2: the four information axes at one dataset (recal protocol) ────────
def fig_mechanism_v2():
    x = np.arange(3)
    prior = [22.9, 3.5, 0.1]           # iso->emp, recal test, pems08f
    cond = [63.9, 19.1, 0.2]           # fo->past, iso arm, test, pems08f
    cap = [28.0, np.nan, 1.0]          # gcnn->ugnet, val, pems08f
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.axhline(0, color=GRAY, lw=1, ls="--")
    ax.plot(x, cond, "-s", color=GREEN, lw=2, ms=5, label="conditioning on history")
    ax.plot(x, prior, "-o", color=BLUE, lw=2, ms=5, label="structured prior (ours)")
    ax.plot([0, 2], [cap[0], cap[2]], ":D", color=PURPLE, lw=1.8, ms=5,
            label="58× larger denoiser")
    ax.set_xticks(x, ["MLP\n(312)", "GCN\n(17k)", "GWN\n(309k)"])
    ax.set_ylabel("CRPS improvement from adding it (%)")
    ax.set_xlabel("backbone (information already absorbed →)")
    ax.set_title("Every information source obeys the same law (PEMS08-F)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "f1_mechanism")
