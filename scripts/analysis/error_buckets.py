"""The residual distribution in plain counts — no density, no log axis.

For each backbone, what fraction of ALL predictions land in each |error| bucket? Most are
near-perfect (huge first bucket); a small but real fraction are big misses (the tail). The
'Gaussian' row is what a bell curve with the same spread would put there — near-zero in the
big-miss buckets. The gap in those buckets IS the heavy tail, in counts you can read.

usage: .venv/bin/python scripts/analysis/error_buckets.py [--dataset pems08s]
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                                       # noqa: E402
FIG = paths.analysis_dir("distribution")
BACKBONES = ["mlp", "gcn", "gwn"]


def abs_resid(ds, bb):
    p = paths.residual_npz(ds, "test", 42, bb)
    if not p.exists():
        p = paths.residual_npz(ds, "train", 42, bb)
    if not p.exists():
        return None
    d = np.load(p)
    R = d["residuals"].astype(np.float64); Yh = d["y_pred"].astype(np.float64)
    M = d["mask"].astype(bool); Yt = Yh + R
    v = M & (np.abs(Yt) > 1e-3)
    return np.abs(R[v]), R[v].std()


def gauss_bucket_frac(edges, sigma):
    """fraction of a N(0,σ) in each |x| bucket [e_i, e_{i+1}] (both tails)."""
    # P(|x| < t) = erf(t/(σ√2)); bucket = P(<hi) - P(<lo)
    cdf = lambda t: special.erf(t / (sigma * np.sqrt(2))) if np.isfinite(t) else 1.0
    return np.array([cdf(edges[i + 1]) - cdf(edges[i]) for i in range(len(edges) - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pems08s")
    args = ap.parse_args()
    ds = args.dataset

    # buckets in σ units, each backbone standardized by ITS OWN std → a fair heavy-tail
    # comparison (a Gaussian is the same in σ units regardless of scale). std/MAE printed too.
    edges = np.array([0, 1, 2, 3, 5, 10, np.inf])
    labels = ["<1σ", "1–2σ", "2–3σ", "3–5σ", "5–10σ", ">10σ"]
    print(f"=== {ds}: % of predictions in each |error| bucket, each backbone in ITS OWN σ ===\n")
    print(f"{'backbone':9s} {'σ(raw)':>7s} {'MAE':>6s} | " + "".join(f"{l:>9s}" for l in labels))
    fracs = {}
    for bb in BACKBONES:
        got = abs_resid(ds, bb)
        if got is None:
            continue
        a, sd = got
        z = a / sd                                          # standardize by own std
        f = np.array([((z >= edges[i]) & (z < edges[i + 1])).mean() for i in range(len(edges) - 1)])
        fracs[bb] = f
        print(f"{bb:9s} {sd:7.2f} {a.mean():6.2f} | " + "".join(f"{100*x:8.2f}%" for x in f))
    gf = gauss_bucket_frac(edges, 1.0)
    print(f"{'GAUSSIAN':9s} {'':7s} {'':6s} | " + "".join(f"{100*x:8.2f}%" for x in gf))
    print("\nread the last 3 columns (3σ, 5σ, 10σ): the backbones put a FEW % there — real big")
    print("misses (incidents). a Gaussian puts ~0% there. that gap is the heavy tail, in counts.")

    # figure: grouped bars, tail buckets only (where the story is), + Gaussian
    tail = slice(2, 6)                                       # 2–3σ onward
    x = np.arange(len(labels[tail])); w = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, bb in enumerate(fracs):
        ax.bar(x + (i - 1) * w, 100 * fracs[bb][tail], w, label=f"{bb} backbone")
    ax.bar(x + 2 * w, 100 * gf[tail], w, color="k", alpha=0.6, label="Gaussian (would predict)")
    ax.set_xticks(x); ax.set_xticklabels(labels[tail])
    ax.set_xlabel("prediction error size"); ax.set_ylabel("% of ALL predictions")
    ax.set_title(f"{ds}: big-miss tail — gwn makes {100*fracs['gwn'][3:].sum():.1f}% errors ≥3σ; a "
                 f"Gaussian predicts only {100*gf[3:].sum():.2f}%. (At 2–3σ the Gaussian is higher — the\n"
                 f"backbones are MORE peaked there: fewer medium errors, more extreme ones = heavy tail. "
                 f"Rare, so MAE stays good; but a Gaussian prior can't reproduce it.)", fontsize=9.5)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / f"error_buckets_{ds}.png", dpi=140, bbox_inches="tight")
    print(f"\nwrote {FIG / f'error_buckets_{ds}.png'}")


if __name__ == "__main__":
    main()
