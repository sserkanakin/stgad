"""Mode-variance slope diagnostic: is the residual second moment a PRODUCT (Σ_T⊗Σ_S,
multiplicative precision) or a SUM (Cartesian L_□+γI, additive precision)?

This is the discriminator the Van-Loan/Kronecker separability index could not be. Both
forms live in the SAME joint eigenbasis V_S ⊗ V_T, so separability can't tell them apart —
they differ only in how the per-mode variance depends on the graph frequency λ_k:

    product : v[k,h] = d_h / (λ_k + γ)                 → log v = c_h − 1·log(λ_k+γ)   slope −1
    sum     : v[k,h] = σ² / (½(λ_k+ω_h) + γ_c)         → shallow, h-dependent, curved

so per temporal mode h we regress log v[·,h] on log(λ_k+γ) across the N spatial modes.
product predicts slope −1 for EVERY h; sum predicts a shallow slope −(λ+γ)/((λ+γ)+ν_h)
that flattens toward 0 at low frequency. 12 slopes per dataset, plus the SSE of the two
parametric fits on the (N×H) grid. product's slope −1 is basis-invariant on the temporal
axis, so we rotate temporally into the path-graph basis V_T (the SUM model's own basis) —
fair to product, natural for sum. Action 2 (scale audit of Σ_S) rides along on the same
prior load.

decision rule (pre-registered, see DECISIONS.md):
  slopes in [−1.2, −0.8] for ≥9 of 12 horizons AND product SSE < sum SSE  → lock product.
  ambiguous → supervisor discussion before any redesign.

usage: .venv/bin/python scripts/analysis/mode_slope_diagnostic.py [--datasets ...] [--backbones ...] [--split train]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior                                          # noqa: E402
from src.data.adjacency import compute_normalized_laplacian                         # noqa: E402
from src.utils import paths                                                         # noqa: E402

VARIANTS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
BACKBONES = ["gwn", "mlp"]
OUT = ROOT / "results/analysis/paper_results"
FIG = OUT / "figs"
COV_CAP = 8000
GAMMA_CART = 0.1        # the Cartesian arm's fixed γ (prior.py temporal="cartesian")
COLS = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]


def _load(ds: str, split: str, bb: str, seed: int = 42):
    """residual field (B,N,H) in the prior's unit-variance space, mask-zeroed, subsampled.
    identical to kronecker_audit._load so the two audits share a fit space."""
    from src.diffusion.prior import _load_residuals_train
    from src.data.residual_dataset import train_residual_sigma
    if split == "train":
        R, M = _load_residuals_train(ds, seed, bb)
    else:
        d = np.load(paths.residual_npz(ds, split, seed, bb))
        R = d["residuals"].astype(np.float64)
        M = d["mask"].astype(bool)
        sig = np.asarray(train_residual_sigma(ds, seed, bb, norm="global"))
        R = R / (sig if sig.ndim == 2 else sig.reshape(-1, 1))
    R = np.where(M, R, 0.0)
    if len(R) > COV_CAP:
        R = R[np.linspace(0, len(R) - 1, COV_CAP).astype(int)]
    return R


def _path_graph_temporal(H: int):
    """normalized-Laplacian eigenpairs of the H-node path graph — the SUM model's
    temporal factor (prior.py temporal='cartesian'). ω ∈ [0,2], smooth→rough."""
    A = np.zeros((H, H)); idx = np.arange(H - 1)
    A[idx, idx + 1] = 1.0; A[idx + 1, idx] = 1.0
    omega, V_T = np.linalg.eigh(compute_normalized_laplacian(A))
    omega = np.clip(omega, 0.0, None)
    return omega, V_T


def _ols_slope(x, y):
    """slope + R² of y on x (single regressor). x,y are 1-D over spatial modes."""
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom < 1e-30:
        return 0.0, 0.0
    b = float((x * (y - y.mean())).sum() / denom)
    yhat = y.mean() + b * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-30
    return b, 1.0 - ss_res / ss_tot


def _fixed_slope_sse(logv, predictor):
    """SSE of a per-h fit log v[·,h] = c_h + predictor[·,h], with the ONLY free
    parameter the per-h intercept c_h (absorbs σ² and per-mode normalization). the
    functional SHAPE is fixed — this is what makes product vs sum a fair, non-nested
    comparison rather than one nesting the other."""
    c = (logv - predictor).mean(0, keepdims=True)          # (1,H) best intercept per h
    resid = logv - (c + predictor)
    return float((resid ** 2).sum())


def slope_diagnostic(ds, bb, split):
    prior = load_prior(ds, "kronecker", seed=42, backbone=bb,
                       spatial="structured", temporal="empirical")
    N, H = prior.N, prior.H
    lam_S, gamma = prior.lam_S, prior.gamma
    omega, V_T = _path_graph_temporal(H)                   # rotate into the SUM model's basis

    R = _load(ds, split, bb)
    C = np.einsum("ki,bih->bkh", prior.V_S.T, R, optimize=True)   # spatial rotation
    C = np.einsum("bkh,hj->bkj", C, V_T, optimize=True)           # temporal rotation (path basis)
    v = np.clip((C ** 2).mean(0), 1e-30, None)             # (N,H) per-mode variance
    logv = np.log(v)

    xg = np.log(lam_S + gamma)                             # regressor: log(λ_k+γ), (N,)
    # 1. free per-h slopes
    slopes = np.array([_ols_slope(xg, logv[:, h])[0] for h in range(H)])
    r2s = np.array([_ols_slope(xg, logv[:, h])[1] for h in range(H)])

    # 2. parametric SSE, product vs sum, both with only a free per-h intercept
    pred_prod = -xg[:, None] * np.ones((1, H))                              # slope −1 everywhere
    theta_sum = 0.5 * (lam_S[:, None] + omega[None, :]) + GAMMA_CART        # (N,H) Cartesian precision
    pred_sum = -np.log(theta_sum)
    sse_prod = _fixed_slope_sse(logv, pred_prod)
    sse_sum = _fixed_slope_sse(logv, pred_sum)

    in_band = int(np.sum((slopes >= -1.2) & (slopes <= -0.8)))
    verdict = ("PRODUCT" if (in_band >= 9 and sse_prod < sse_sum)
               else "SUM" if (in_band <= 3 and sse_sum < sse_prod)
               else "AMBIGUOUS")
    return dict(ds=ds, bb=bb, N=N, H=H, gamma=gamma, slopes=slopes, r2s=r2s,
                in_band=in_band, sse_prod=sse_prod, sse_sum=sse_sum, verdict=verdict,
                v=v, lam_S=lam_S, xg=xg, pred_sum=pred_sum)


def scale_audit(ds, bb):
    """Action 2: tr(Σ_S)/N and diag(Σ_S) spread, Σ_S=(L+γI)^{-1} at the locked γ."""
    prior = load_prior(ds, "kronecker", seed=42, backbone=bb,
                       spatial="structured", temporal="empirical")
    s_S = 1.0 / (prior.lam_S + prior.gamma)               # eigenvalues of Σ_S
    diag = (prior.V_S ** 2) @ s_S                          # diag(V_S diag(s_S) V_S^T)
    return dict(ds=ds, bb=bb, N=prior.N, gamma=prior.gamma,
                tr_over_N=float(s_S.mean()),
                dmin=float(diag.min()), dmed=float(np.median(diag)), dmax=float(diag.max()))


def figure(rows, split):
    """one panel per (ds,bb): the 12 free slopes with the product band [-1.2,-0.8] and
    the sum-model expected slope at each temporal mode overlaid."""
    n = len(rows)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for r, ax in zip(rows, axes.flat):
        ax.axis("on")
        H = r["H"]
        h = np.arange(H)
        # sum-model local slope at each temporal mode (mean over spatial modes), for reference
        # d log v / d log(λ+γ) under sum = -(λ+γ)/θ_sum, averaged over k
        lamg = np.exp(r["xg"])
        theta_sum = np.exp(-r["pred_sum"])
        sum_slope = -(lamg[:, None] / theta_sum).mean(0)         # (H,)
        ax.axhspan(-1.2, -0.8, color=COLS[0], alpha=0.15, label="product band")
        ax.axhline(-1.0, color=COLS[0], lw=1, ls="--")
        ax.plot(h, r["slopes"], "o-", color=COLS[1], lw=1.5, ms=4, label="empirical slope")
        ax.plot(h, sum_slope, "s:", color=COLS[2], lw=1.2, ms=3, label="sum-model slope")
        ax.set_ylim(-1.6, 0.2)
        ax.set_title(f"{r['ds']}/{r['bb']}  [{r['verdict']}]\n"
                     f"{r['in_band']}/{H} in band · SSE prod {r['sse_prod']:.1f} vs sum {r['sse_sum']:.1f}",
                     fontsize=8.5)
        ax.set_xlabel("temporal mode h (smooth→rough)", fontsize=8)
        ax.set_ylabel("d log v / d log(λ+γ)", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle(f"mode-variance slope diagnostic [{split}]: product (slope −1) vs sum (shallow)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / "mode_slope_diagnostic.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    args = ap.parse_args()
    datasets = args.datasets or VARIANTS
    backbones = args.backbones or BACKBONES

    slope_rows, scale_rows = [], []
    for ds in datasets:
        for bb in backbones:
            if not paths.residual_npz(ds, "train", 42, bb).exists():
                continue
            slope_rows.append(slope_diagnostic(ds, bb, args.split))
            scale_rows.append(scale_audit(ds, bb))

    # ── Action 1 table ──
    print(f"\n=== ACTION 1. MODE-VARIANCE SLOPE ({args.split} split) ===")
    print("product: v∝1/(λ+γ) → slope −1 every h.  sum: shallow curved slope.")
    hdr = f"{'dataset':9s} {'bb':4s} | {'in-band/H':>10s} | {'SSE prod':>9s} {'SSE sum':>9s} | {'median β':>9s} | verdict"
    print(hdr); print("-" * len(hdr))
    for r in slope_rows:
        print(f"{r['ds']:9s} {r['bb']:4s} | {r['in_band']:>4d}/{r['H']:<5d} | "
              f"{r['sse_prod']:9.2f} {r['sse_sum']:9.2f} | {np.median(r['slopes']):9.3f} | {r['verdict']}")

    # ── Action 2 table ──
    print(f"\n=== ACTION 2. SCALE AUDIT of Σ_S=(L+γI)^-1 (locked γ) ===")
    print("if tr(Σ_S)/N deviates materially from 1, normalize Σ_S←Σ_S/(trΣ_S/N) and fold into σ.")
    hdr2 = f"{'dataset':9s} {'bb':4s} | {'γ':>5s} {'N':>4s} | {'tr/N':>7s} | {'diag min':>9s} {'diag med':>9s} {'diag max':>9s}"
    print(hdr2); print("-" * len(hdr2))
    for r in scale_rows:
        print(f"{r['ds']:9s} {r['bb']:4s} | {r['gamma']:5.2f} {r['N']:>4d} | {r['tr_over_N']:7.3f} | "
              f"{r['dmin']:9.3f} {r['dmed']:9.3f} {r['dmax']:9.3f}")

    # ── markdown ──
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# Mode-variance slope diagnostic + Σ_S scale audit\n",
          f"split: **{args.split}**  ·  regressor log(λ_k+γ)  ·  temporal basis = path-graph (sum's own)\n",
          "## Action 1 — product (slope −1) vs sum (shallow)\n",
          "| dataset | bb | in-band / H | SSE product | SSE sum | median β | verdict |",
          "|---|---|--:|--:|--:|--:|---|"]
    for r in slope_rows:
        md.append(f"| {r['ds']} | {r['bb']} | {r['in_band']}/{r['H']} | {r['sse_prod']:.2f} | "
                  f"{r['sse_sum']:.2f} | {np.median(r['slopes']):.3f} | {r['verdict']} |")
    md += ["\nper-h slopes (product⟺−1):\n"]
    for r in slope_rows:
        md.append(f"- `{r['ds']}/{r['bb']}`: " + " ".join(f"{s:+.2f}" for s in r["slopes"]))
    md += ["\n## Action 2 — Σ_S scale audit (locked γ)\n",
           "| dataset | bb | γ | N | tr(Σ_S)/N | diag min | diag med | diag max |",
           "|---|---|--:|--:|--:|--:|--:|--:|"]
    for r in scale_rows:
        md.append(f"| {r['ds']} | {r['bb']} | {r['gamma']:.2f} | {r['N']} | {r['tr_over_N']:.3f} | "
                  f"{r['dmin']:.3f} | {r['dmed']:.3f} | {r['dmax']:.3f} |")
    (OUT / "mode_slope_diagnostic.md").write_text("\n".join(md) + "\n")
    png = figure(slope_rows, args.split)
    print(f"\nwrote {OUT / 'mode_slope_diagnostic.md'}\nwrote {png}")


if __name__ == "__main__":
    main()
