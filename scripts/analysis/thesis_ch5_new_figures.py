"""the two figures the rewritten experiments chapter needs.

f_channels     the prior against the other two ways of giving the model the same
               information -- history conditioning and denoiser capacity -- on one
               axis, per backbone. the point is the relative size of the bars.
f_attribution  what each factor of the prior is worth on held-out data: the fitted
               temporal factor against the imposed graph factor on top of it.

writes into the thesis figures directory as pdf (and png for quick viewing).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "tudelft_thesis_draft" / "figures"

# validated categorical triple (scripts/validate_palette.js, light surface): all six
# checks pass. marker and hatch differ too, so the figures survive greyscale printing.
BLUE, RUST, PURPLE = "#1f6fb4", "#b5591f", "#7a5ea8"
INK, MUTED = "#222222", "#666666"

plt.rcParams.update({
    "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 200,
})

PEMS = ["pems08f", "pems08s", "pems04f", "pems04s"]
BB = [("mlp", "linear"), ("gcn", "GCN"), ("gwn", "GWN")]


def read(p):
    try:
        return json.loads(Path(p).read_text())
    except (ValueError, OSError):
        return None


def test_cells(arm):
    """seed-mean test metrics per (ds, bb) for one arm directory."""
    out = defaultdict(list)
    for p in (RES / "20260721_ladder" / arm).glob(
            "*/*/*/seed*/eval/test_metrics_bestcrps_ancestral50.json"):
        d = read(p)
        if d:
            out[(p.parts[-6], p.parts[-5])].append(d["crps"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def val_cells(arm):
    out = defaultdict(list)
    for p in (RES / "20260721_ladder" / arm).glob("*/*/*/seed*/train/validation_history.json"):
        d = read(p)
        if d and d.get("best_crps"):
            out[(p.parts[-6], p.parts[-5])].append(d["best_crps"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def gain(base, better, ds, bb):
    """percent CRPS reduction of `better` relative to `base` on one cell."""
    a, b = base.get((ds, bb)), better.get((ds, bb))
    return np.nan if a is None or b is None else 100.0 * (a - b) / a


# ───────────────────────────────────────────────── figure 1: three channels
def fig_channels():
    iso_fo, iso_pa = test_cells("iso"), test_cells("iso-past")
    emp_pa = test_cells("emp-past")
    v17, v996 = val_cells("iso-past"), val_cells("iso-past-ugnet")

    series = [
        ("structured prior over isotropic", lambda d, b: gain(iso_pa, emp_pa, d, b), BLUE, "//"),
        ("history conditioning over forecast-only", lambda d, b: gain(iso_fo, iso_pa, d, b), RUST, ""),
        ("996k denoiser over 17k (validation)", lambda d, b: gain(v17, v996, d, b), PURPLE, ".."),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x = np.arange(len(BB))
    w = 0.26
    for i, (label, fn, colour, hatch) in enumerate(series):
        per_ds = np.array([[fn(d, bb) for d in PEMS] for bb, _ in BB], dtype=float)
        m = np.nanmean(per_ds, axis=1)
        pos = x + (i - 1) * w
        ax.bar(pos, m, w * 0.86, label=label, color=colour, alpha=0.85,
               hatch=hatch, edgecolor="white", linewidth=0.6, zorder=2)
        for j in range(len(BB)):
            pts = per_ds[j][~np.isnan(per_ds[j])]
            ax.plot(np.full(len(pts), pos[j]), pts, "o", ms=2.6, mfc="white",
                    mec=colour, mew=0.8, zorder=3)
            top = max(m[j], 0.0, *(pts if len(pts) else [0.0]))
            ax.annotate(f"{m[j]:+.1f}", (pos[j], top), textcoords="offset points",
                        xytext=(0, 4.5), ha="center", fontsize=7, color=INK)

    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xticks(x, [n for _, n in BB])
    ax.set_xlabel("backbone whose residual is modelled")
    ax.set_ylabel("CRPS reduction (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_ylim(-4, 68)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"f_channels.{ext}", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────── figure 2: factor attribution
def fig_attribution():
    rows = list(csv.DictReader(
        open(RES / "analysis" / "paper_results" / "residual_structure.csv")))
    order = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
    cells = [r for ds in order for r in rows if r["ds"] == ds]
    short = {"pems08f": "PEMS08 flow", "pems08s": "PEMS08 speed",
             "pems04f": "PEMS04 flow", "pems04s": "PEMS04 speed",
             "metrla": "METR-LA", "pemsbay": "PEMS-BAY"}
    lab = [{"mlp": "lin", "gcn": "GCN", "gwn": "GWN"}[r["bb"]] for r in cells]
    dT = np.array([float(r["dT"]) for r in cells])
    gT = np.array([float(r["gT"]) for r in cells])
    x = np.arange(len(cells))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12})
    a1.bar(x, dT, 0.66, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.6, zorder=2)
    a1.set_ylabel("temporal factor\n(nats / mode)")
    a1.set_ylim(0, 1.55)

    a2.bar(x, gT, 0.66, color=RUST, alpha=0.85, hatch="//", edgecolor="white",
           linewidth=0.6, zorder=2)
    a2.axhline(0, color=MUTED, lw=0.8, zorder=1)
    a2.set_ylabel("graph factor,\non top of time")
    a2.set_ylim(-0.06, 0.12)
    a2.set_xticks(x, lab, fontsize=7)

    a2.annotate("note the scale: this panel spans one tenth of the one above",
                xy=(0.985, 0.06), xycoords="axes fraction", ha="right",
                fontsize=6.8, color=MUTED)
    for a in (a1, a2):
        a.tick_params(axis="x", length=0)
        for b in range(3, len(cells), 3):          # a hairline between datasets
            a.axvline(b - 0.5, color=MUTED, lw=0.5, alpha=0.35, zorder=0)
    for g, ds in enumerate(order):                 # dataset name once per group
        a2.annotate(short[ds], xy=(3 * g + 1, -0.125), ha="center", va="top",
                    fontsize=7, color=INK, annotation_clip=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"f_attribution.{ext}", bbox_inches="tight")
    plt.close(fig)


fig_channels()
fig_attribution()
print(f"wrote f_channels and f_attribution to {FIG}")
