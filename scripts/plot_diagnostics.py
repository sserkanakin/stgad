#!/usr/bin/env python3
"""render the calibration diagrams + example forecasts from the eval diagnostics.

reads the companion `*_diag_*.npz` files that eval_diffusion.py --save-diagnostics
writes (PIT histogram, reliability curve, example fan charts) and produces, per
(dataset, sampler):
  <root>/<exp>/figures/calibration_<ds>_<sampler>.png   PIT + reliability, arms overlaid
  <root>/<exp>/figures/examples_<ds>_<sampler>_<arm>.png example forecast fan charts

run LOCALLY (matplotlib is local-only; sync the small .npz files back from DAIC first).

usage:  python scripts/plot_diagnostics.py --exp 20260708_r2 --root results/pulled
        python scripts/plot_diagnostics.py --exp 20260708_r2 --root out --ds pems08f
"""
import argparse, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_COLOR = {"kron_base": "#1b6ca8", "iso_base": "#d1495b",
             "kron_pastG": "#66a182", "iso_pastG": "#e8998d", "kron_pastPN": "#8d6a9f"}
HL = (3, 6, 12)


def _s(a):  # unwrap a 1-element object/str array
    v = np.asarray(a).ravel()[0]
    return v.decode() if isinstance(v, bytes) else str(v)


def load_diags(root: Path, exp: str, only_ds=None):
    out = {}
    for f in sorted(glob.glob(str(root / exp / "**" / "*_diag_*.npz"), recursive=True)):
        d = np.load(f, allow_pickle=True)
        ds, arm, smp = _s(d["dataset"]), _s(d["exp"]).split("/")[-1], _s(d["sampler"])
        if only_ds and ds != only_ds:
            continue
        out[(ds, smp, arm)] = d
    return out


def plot_calibration(figdir: Path, ds, smp, cells):
    fig, (axp, axr) = plt.subplots(1, 2, figsize=(10, 4))
    for (arm, d) in cells:
        c = ARM_COLOR.get(arm, "gray")
        # PIT histogram (normalized to a density; flat dashed line = calibrated)
        h = np.asarray(d["pit_hist_all"], float)
        h = h / max(h.sum(), 1e-9)
        edges = np.linspace(0, 1, len(h) + 1)
        axp.step(edges[1:], h, where="pre", color=c, label=arm, lw=1.8)
        # reliability: empirical central-interval coverage vs nominal
        lv = np.asarray(d["reliab_levels"], float)
        cov = np.asarray(d["reliab_cov"], float)
        axr.plot(lv, cov, "-o", color=c, ms=4, label=arm, lw=1.6)
    axp.axhline(1.0 / (len(h)), ls="--", color="k", lw=1, alpha=.6)
    axp.set(title=f"PIT — {ds} ({smp})", xlabel="PIT", ylabel="density", xlim=(0, 1))
    axp.legend(fontsize=8)
    axr.plot([0, 1], [0, 1], "--", color="k", lw=1, alpha=.6)
    axr.set(title=f"reliability — {ds} ({smp})", xlabel="nominal coverage",
            ylabel="empirical coverage", xlim=(0, 1), ylim=(0, 1))
    axr.legend(fontsize=8)
    fig.tight_layout()
    p = figdir / f"calibration_{ds}_{smp}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_examples(figdir: Path, ds, smp, arm, d, max_win=3):
    if "ex_quantiles" not in d:
        return None
    q = np.asarray(d["ex_quantiles"], float)     # (nq, ne, S, H)
    qp = np.asarray(d["ex_qprobs"], float)
    yt = np.asarray(d["ex_ytrue"], float)         # (ne, S, H)
    yhat = np.asarray(d.get("ex_yhat", np.zeros_like(yt)), float)
    sidx = np.asarray(d["ex_sensor_idx"]).ravel()
    nq, ne, S, H = q.shape
    ne = min(ne, max_win); S = min(S, 4)
    x = np.arange(1, H + 1)
    fig, axes = plt.subplots(ne, S, figsize=(3.2 * S, 2.4 * ne), squeeze=False)
    med = np.argmin(np.abs(qp - 0.5))
    lo1, hi1 = np.argmin(np.abs(qp - 0.05)), np.argmin(np.abs(qp - 0.95))
    lo2, hi2 = np.argmin(np.abs(qp - 0.25)), np.argmin(np.abs(qp - 0.75))
    c = ARM_COLOR.get(arm, "#1b6ca8")
    for wi in range(ne):
        for si in range(S):
            ax = axes[wi][si]
            ax.fill_between(x, q[lo1, wi, si], q[hi1, wi, si], color=c, alpha=.18, lw=0)
            ax.fill_between(x, q[lo2, wi, si], q[hi2, wi, si], color=c, alpha=.30, lw=0)
            ax.plot(x, q[med, wi, si], color=c, lw=1.5, label="forecast median")
            ax.plot(x, yt[wi, si], "k-o", ms=2.5, lw=1.2, label="truth")
            if wi == 0 and si == 0:
                ax.legend(fontsize=6)
            if wi == 0:
                ax.set_title(f"sensor {sidx[si]}", fontsize=8)
            if si == 0:
                ax.set_ylabel(f"window {wi}", fontsize=8)
    fig.suptitle(f"example forecasts — {ds} / {arm} ({smp})  90% + 50% bands", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = figdir / f"examples_{ds}_{smp}_{arm}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--root", default="out")
    ap.add_argument("--ds", default=None, help="restrict to one dataset")
    a = ap.parse_args()
    root = Path(a.root)
    diags = load_diags(root, a.exp, a.ds)
    if not diags:
        print(f"[plot_diagnostics] no *_diag_*.npz under {root/a.exp} yet")
        return
    figdir = root / a.exp / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    # group by (ds, sampler) for the overlaid calibration plots
    groups = {}
    for (ds, smp, arm), d in diags.items():
        groups.setdefault((ds, smp), []).append((arm, d))
    made = []
    for (ds, smp), cells in sorted(groups.items()):
        made.append(plot_calibration(figdir, ds, smp, cells))
        for arm, d in cells:
            p = plot_examples(figdir, ds, smp, arm, d)
            if p:
                made.append(p)
    for p in made:
        print(f"  wrote {p}")
    print(f"[plot_diagnostics] {len(made)} figure(s) -> {figdir}")


if __name__ == "__main__":
    main()
