"""Compare the STRUCTURE of the raw future signal vs the backbone RESIDUAL — the crux of
why GAD's low-frequency prior fits raw graph signals but not our residuals.

Both come from the SAME residual npz: y_true = Y_hat + residual, residual = residual.
For each we measure, masked, in the SAME graph eigenbasis:
  spatial : share of graph-spectral energy in low/mid/high λ bands (how graph-smooth)
  temporal: horizon autocorrelation ρ(1) (how smooth in time) + temporal-mode energy
  tails   : excess kurtosis (how heavy-tailed / non-Gaussian)

Figure: spatial spectrum (energy vs graph frequency λ) raw vs residual, per dataset —
the raw signal piles at low λ (smooth field), the residual spreads to mid/high (the
backbone removed the smooth part). That mismatch is why a low-pass prior suits raw, not
residual.

usage: .venv/bin/python scripts/analysis/raw_vs_residual.py [--datasets metrla pems08f]
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
from src.data.adjacency import load_adjacency, compute_normalized_laplacian   # noqa: E402
from src.utils import paths                                                    # noqa: E402

VARIANTS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
FIG = paths.analysis_dir("structure")
_EIG = {}


def eig(ds):
    if ds not in _EIG:
        L = compute_normalized_laplacian(load_adjacency(ds, repair=True))
        lam, V = np.linalg.eigh(L); o = np.argsort(lam)
        _EIG[ds] = (lam[o], V[:, o])
    return _EIG[ds]


def graph_spec(X, V):
    """energy share per graph mode of X (B,N,H), sorted by λ. masked entries pre-zeroed."""
    C = np.matmul(V.T, X)                            # (B,N,H) spatial coeffs
    e = (C ** 2).mean((0, 2))                        # per spatial mode
    return e / e.sum()


def kurt(x):
    x = x[np.isfinite(x)]; m = x.mean(); s = x.std() + 1e-12
    return float(((x - m) ** 4).mean() / s ** 4 - 3.0)


def analyze(ds, bb="gwn"):
    d = np.load(paths.residual_npz(ds, "train", 42, bb))
    R = d["residuals"].astype(np.float64)
    Yh = d["y_pred"].astype(np.float64)
    M = d["mask"].astype(bool)
    Yt = Yh + R                                       # raw future signal
    Rz = np.where(M, R, 0.0)
    Ytz = np.where(M, Yt, 0.0)
    sub = np.linspace(0, len(R) - 1, min(2000, len(R))).astype(int)
    lam, V = eig(ds)

    out = {"ds": ds}
    for tag, X in (("raw", Ytz[sub]), ("resid", Rz[sub])):
        e = graph_spec(X, V)
        out[f"{tag}_low"] = float(e[lam < 2/3].sum())
        out[f"{tag}_mid"] = float(e[(lam >= 2/3) & (lam < 4/3)].sum())
        out[f"{tag}_high"] = float(e[lam >= 4/3].sum())
        out[f"{tag}_spec"] = (lam, e)
        # temporal: lag-1 horizon autocorrelation, averaged over nodes/windows (masked)
        Xv = X[M[sub].all(2)] if False else X
        flat = X.reshape(-1, X.shape[-1])
        valid = (flat != 0).all(1)
        f = flat[valid]
        if len(f) > 10:
            f = f - f.mean(1, keepdims=True)
            num = (f[:, 1:] * f[:, :-1]).mean()
            den = (f ** 2).mean() + 1e-12
            out[f"{tag}_rho1"] = float(num / den)
        else:
            out[f"{tag}_rho1"] = float("nan")
        out[f"{tag}_kurt"] = kurt(X[M[sub]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=VARIANTS)
    ap.add_argument("--figure-datasets", nargs="*", default=["metrla", "pems08f"])
    args = ap.parse_args()

    rows = [analyze(ds) for ds in args.datasets]
    print(f"{'dataset':9s} | {'RAW low/mid/high':>22s}  {'ρ1':>5s} {'kurt':>6s} | "
          f"{'RESID low/mid/high':>22s}  {'ρ1':>5s} {'kurt':>6s}")
    print("-" * 100)
    md = ["| dataset | raw low/mid/high λ | raw ρ1 | raw kurt | resid low/mid/high λ | resid ρ1 | resid kurt |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        rw = f"{r['raw_low']:.0%}/{r['raw_mid']:.0%}/{r['raw_high']:.0%}"
        rs = f"{r['resid_low']:.0%}/{r['resid_mid']:.0%}/{r['resid_high']:.0%}"
        print(f"{r['ds']:9s} | {rw:>22s}  {r['raw_rho1']:5.2f} {r['raw_kurt']:6.1f} | "
              f"{rs:>22s}  {r['resid_rho1']:5.2f} {r['resid_kurt']:6.1f}")
        md.append(f"| {r['ds']} | {rw} | {r['raw_rho1']:.2f} | {r['raw_kurt']:.1f} | "
                  f"{rs} | {r['resid_rho1']:.2f} | {r['resid_kurt']:.1f} |")
    print("\nlow=λ<0.67 (graph-smooth), high=λ>1.33 (oscillatory). ρ1=lag-1 horizon autocorr (1=smooth in time).")
    print("kurt=excess kurtosis (0=Gaussian, >0 heavy tails).")
    out = FIG / "raw_vs_residual.md"
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")

    # figure
    figrows = [r for r in rows if r["ds"] in args.figure_datasets]
    fig, axes = plt.subplots(1, len(figrows), figsize=(6 * len(figrows), 4.4), squeeze=False)
    for ax, r in zip(axes[0], figrows):
        lam, er = r["raw_spec"]; _, es = r["resid_spec"]
        ax.plot(lam, er, lw=2, label="raw future signal")
        ax.plot(lam, es, lw=2, label="backbone residual")
        ax.set_title(f"{r['ds']}: graph spectrum (raw low-freq {r['raw_low']:.0%} → "
                     f"resid {r['resid_low']:.0%})", fontsize=11)
        ax.set_xlabel("graph frequency λ  (0=smooth, 2=oscillatory)")
        ax.set_ylabel("share of energy"); ax.legend(); ax.grid(alpha=0.25); ax.set_xlim(0, 2)
    fig.suptitle("Raw signal is graph-SMOOTH (low-freq); the residual is not — the backbone "
                 "already took the smooth part. A low-pass prior fits raw, not residual.", fontsize=12)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / "raw_vs_residual.png", dpi=140, bbox_inches="tight")
    print(f"wrote {FIG / 'raw_vs_residual.png'}")


if __name__ == "__main__":
    main()
