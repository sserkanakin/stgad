"""SEE the data distribution — what values there are, how many, and how heavy the tails
are — for the RAW future signal and the backbone RESIDUAL, across train / val / test.

Table per split: count of valid entries, mean, std, key percentiles, max, excess kurtosis,
and how many entries fall beyond 3σ / 4σ / 5σ (the 'tail' — what a Gaussian says is rare).
Figure: residual histogram on a LOG y-axis (so the tails are visible) for the three splits,
overlaid with the Gaussian of the same variance — the gap at the edges IS the heavy tail.

usage: .venv/bin/python scripts/analysis/distribution.py [--dataset pems08s --backbone gwn]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                                       # noqa: E402
FIG = paths.analysis_dir("distribution")


def load(ds, split, bb):
    p = paths.residual_npz(ds, split, 42, bb)
    if not p.exists():
        return None
    d = np.load(p)
    R = d["residuals"].astype(np.float64)
    Yh = d["y_pred"].astype(np.float64)
    M = d["mask"].astype(bool)
    Yt = Yh + R
    valid = M & (np.abs(Yt) > 1e-3)                 # drop missing (y_true==0): else residual=-ŷ
    return R[valid], Yt[valid]                       # fakes a huge tail (SpecSTG/DiffSTG convention)


def stats(x):
    m, s = x.mean(), x.std()
    z = (x - m) / (s + 1e-12)
    k = float((z ** 4).mean() - 3.0)
    pct = np.percentile(x, [0.1, 1, 50, 99, 99.9])
    return dict(n=len(x), mean=m, std=s, p01=pct[0], p1=pct[1], p50=pct[2],
                p99=pct[3], p999=pct[4], mx=x.max(), kurt=k,
                g3=float((np.abs(z) > 3).mean()), g4=float((np.abs(z) > 4).mean()),
                g5=float((np.abs(z) > 5).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pems08s")
    ap.add_argument("--backbone", default="gwn")
    args = ap.parse_args()
    ds, bb = args.dataset, args.backbone

    print(f"=== {ds} / {bb} — data distribution across splits ===\n")
    resid = {}
    for kind, idx in (("RESIDUAL (y - ŷ)", 0), ("RAW signal (y)", 1)):
        print(f"--- {kind} ---")
        print(f"{'split':6s} {'N':>9s} {'mean':>7s} {'std':>7s} {'p0.1':>8s} {'p50':>7s} "
              f"{'p99.9':>8s} {'max':>8s} {'kurt':>6s} | {'>3σ':>7s} {'>4σ':>7s} {'>5σ':>7s}")
        for split in ("train", "val", "test"):
            got = load(ds, split, bb)
            if got is None:
                print(f"{split:6s}  (missing)"); continue
            s = stats(got[idx])
            if idx == 0:
                resid[split] = got[0]
            print(f"{split:6s} {s['n']:9,d} {s['mean']:7.2f} {s['std']:7.2f} {s['p01']:8.2f} "
                  f"{s['p50']:7.2f} {s['p999']:8.1f} {s['mx']:8.1f} {s['kurt']:6.1f} | "
                  f"{s['g3']:7.2%} {s['g4']:7.2%} {s['g5']:7.3%}")
        # Gaussian reference for the tail fractions
        print(f"{'GAUSS':6s} {'':9s} {'':7s} {'':7s} {'':8s} {'':7s} {'':8s} {'':8s} {0.0:6.1f} | "
              f"{0.0027:7.3%} {6.33e-5:7.4%} {5.73e-7:7.5%}")
        print()

    # figure: LEFT linear (the real shape = a spike at 0), RIGHT log (magnifies the tails).
    tr = resid.get("train", np.concatenate(list(resid.values())))
    mu, sd = tr.mean(), tr.std()
    mae = np.abs(tr).mean(); med = np.median(np.abs(tr))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    lim = np.percentile(np.abs(tr), 99.95)
    zoom = min(4 * sd, lim)                                    # linear panel: zoom to the peak
    binsL = np.linspace(-zoom, zoom, 120)
    binsR = np.linspace(-lim, lim, 160)
    xsL = np.linspace(-zoom, zoom, 400); xsR = np.linspace(-lim, lim, 400)
    gaussL = np.exp(-(xsL - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))
    gaussR = np.exp(-(xsR - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))
    for split, r in resid.items():
        axL.hist(r, bins=binsL, density=True, histtype="step", lw=1.8, label=f"{split} (n={len(r):,})")
        axR.hist(r, bins=binsR, density=True, histtype="step", lw=1.8)
    axL.plot(xsL, gaussL, "k--", lw=2, label=f"Gaussian (σ={sd:.1f})")
    axR.plot(xsR, gaussR, "k--", lw=2)
    axL.set_title(f"LINEAR — the real shape: a SPIKE at 0\n(median |err|={med:.2f}, MAE={mae:.2f}; "
                  f"not uniform!)", fontsize=11)
    axL.set_xlabel("residual value (raw units)"); axL.set_ylabel("density"); axL.legend(fontsize=8)
    axL.grid(alpha=0.25)
    axR.set_yscale("log")
    axR.set_title("LOG — magnifies the rare tails\n(flat-looking ≠ uniform; ~1000× below the peak)",
                  fontsize=11)
    axR.set_xlabel("residual value (raw units)"); axR.set_ylabel("density (log)"); axR.grid(alpha=0.25, which="both")
    fig.suptitle(f"{ds}/{bb}: residual = a sharp peak (good MAE) + rare fat tails (bad calibration). "
                 f"Same data, two axes.", fontsize=12)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / f"distribution_{ds}_{bb}.png", dpi=140, bbox_inches="tight")
    print(f"wrote {FIG / f'distribution_{ds}_{bb}.png'}")
    print("\nread: >4σ column vs GAUSS row — a 4σ residual should be ~0.006% of the time under a")
    print("Gaussian; if it's 0.5-1%, that's the heavy tail (≈100x more extreme events than Gaussian).")


if __name__ == "__main__":
    main()
