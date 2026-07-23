"""Test HySRD-Net's claim that trend-removed residuals match an isotropic
Gaussian prior in the graph-spectral domain.

HySRD (ESWA 2026, sec 4.3) POSITS -- without measuring -- that after trend
removal, amplify x100, node-wise z-score, and GFT, the spectral residual
"more closely matches the fundamental assumption of diffusion models" (a
simple N(0,I) prior). their SpecRD-Net then diffuses with a plain isotropic
DDPM. if the claim held, the preprocessed spectral residual would have:

  1. a FLAT variance spectrum across graph-frequency modes (v_k = 1 for all k)
  2. zero excess kurtosis
  3. no cross-mode / cross-horizon correlation

this script applies THEIR exact preprocessing to residual caches (theirs from
scripts/baselines/hysrd_trendnet.py under backbone name "hysrd", plus our
mlp/gcn/gwn backbone residuals) and measures 1-3. note the amplify step is
absorbed exactly by the node-wise z-score, so alpha_amp cannot affect any
distributional property -- we apply it anyway for fidelity.

GFT basis: sym-normalized laplacian of the UNREPAIRED road graph (the paper
does no component surgery), same U for every arm of a dataset.

usage:
  .venv/bin/python scripts/analysis/hysrd_residual_check.py \
      --datasets pems04f pems04s pems08f pems08s --split train
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                          # noqa: E402
from src.data.adjacency import (load_adjacency,                      # noqa: E402
                                compute_normalized_laplacian)

ARMS = ("hysrd", "mlp", "gcn", "gwn")
# dataviz categorical slots 1-4 (validated adjacent-pair order)
COLORS = {"hysrd": "#2a78d6", "mlp": "#eb6834", "gcn": "#1baf7a", "gwn": "#eda100"}
ALPHA_AMP = 100.0


def gft_basis(dataset):
    # their setup: no graph repair, sym-norm laplacian, ascending eigenvalues
    L = compute_normalized_laplacian(load_adjacency(dataset, repair=False))
    lam, U = np.linalg.eigh(L)
    return lam, U


def hysrd_preprocess(r, r_train):
    """their eq: X'' = (X * alpha - mu_res) / sigma_res, node-wise, train stats."""
    a = ALPHA_AMP * r_train.astype(np.float64)
    mu = a.mean(axis=(0, 2), keepdims=True)               # (1,N,1)
    sd = a.std(axis=(0, 2), keepdims=True)
    return ((ALPHA_AMP * r.astype(np.float64)) - mu) / sd


def excess_kurtosis(x):
    x = x.reshape(-1).astype(np.float64)
    x = x - x.mean()
    m2 = (x ** 2).mean()
    return float((x ** 4).mean() / m2 ** 2 - 3.0)


def offdiag_corr_rms(x, rng, max_samples=40000):
    """x (M,K): rms of off-diagonal correlations over a sample of rows."""
    if len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    c = np.corrcoef(x.T.astype(np.float64))
    off = c[~np.eye(len(c), dtype=bool)]
    return float(np.sqrt((off ** 2).mean())), len(x)


def analyze_arm(dataset, arm, split, lam, U, rng):
    f = paths.residual_npz(dataset, split, 42, arm)
    if not f.exists():
        return None
    r = np.load(f)["residuals"]                           # (B,N,H) raw
    r_train = np.load(paths.residual_npz(dataset, "train", 42, arm))["residuals"]
    rz = hysrd_preprocess(r, r_train)                     # their preprocessed residual
    c = np.einsum("kn,bnh->bkh", U.T, rz)                 # spectral coefficients

    v = (c ** 2).mean(axis=(0, 2))                        # (N,) per-mode variance
    B, N, H = c.shape
    m = B * H                                             # samples per mode
    flat_sd = np.sqrt(2.0 / m)                            # chi2 noise on v if iso

    # cross-mode and cross-node correlation (iso-gaussian -> ~1/sqrt(samples))
    spec_off, ns = offdiag_corr_rms(c.transpose(0, 2, 1).reshape(-1, N), rng)
    node_off, _ = offdiag_corr_rms(rz.transpose(0, 2, 1).reshape(-1, N), rng)

    # lag-1 correlation along the horizon, averaged over modes
    a, b = c[:, :, :-1].reshape(-1), c[:, :, 1:].reshape(-1)
    lag1 = float(np.corrcoef(a, b)[0, 1])

    return {
        "arm": arm, "B": B, "v": v,
        "v_cv": float(v.std() / v.mean()),
        "v_ratio_95_5": float(np.percentile(v, 95) / np.percentile(v, 5)),
        "v_max_min": float(v.max() / v.min()),
        "tv_iso": float(0.5 * np.abs(v / v.sum() - 1.0 / N).sum()),
        "flat_noise_sd": float(flat_sd),
        "kurt_spec": excess_kurtosis(c),
        "kurt_node": excess_kurtosis(rz),
        "offdiag_spec": spec_off, "offdiag_node": node_off,
        "corr_samples": ns, "lag1": lag1,
        "resid_std_raw": float(r.std()),
    }


def plot_dataset(dataset, lam, results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    order = np.argsort(lam)
    for res in results:
        ax.plot(lam[order], res["v"][order], lw=1.4, alpha=0.85,
                color=COLORS[res["arm"]], label=res["arm"])
    ax.axhline(1.0, color="#666666", lw=1.0, ls="--", label="iso $N(0,I)$")
    ax.set_yscale("log")
    ax.set_xlabel("graph frequency  $\\lambda_k$")
    ax.set_ylabel("per-mode variance $v_k$ (after their preprocessing)")
    ax.set_title(f"{dataset}: variance spectrum of HySRD-preprocessed residuals")
    ax.grid(True, which="both", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = out_dir / f"spectrum_{dataset}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["pems04f", "pems04s", "pems08f", "pems08s"])
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out_dir = paths.analysis_dir("hysrd_check")
    lines = ["# HySRD residual check: is the preprocessed spectral residual iso-Gaussian?",
             "",
             f"split={args.split}, seed42 caches, their preprocessing "
             f"(amplify x{ALPHA_AMP:.0f} + node-wise z-score + GFT, unrepaired graph).",
             "iso-N(0,I) would give: v_k = 1 for all modes (cv ~ flat_noise_sd), "
             "kurtosis 0, offdiag corr ~ 1/sqrt(samples), lag1 = 0.", ""]

    for ds in args.datasets:
        lam, U = gft_basis(ds)
        results = [r for arm in ARMS
                   if (r := analyze_arm(ds, arm, args.split, lam, U, rng))]
        if not results:
            print(f"[{ds}] no residual caches found, skipping")
            continue
        fig_path = plot_dataset(ds, lam, results, out_dir)
        lines += [f"## {ds}", "",
                  "| arm | v_cv | v p95/p5 | v max/min | TV vs flat | kurt(spec) "
                  "| kurt(node) | offdiag spec | offdiag node | lag1 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        print(f"\n[{ds}]  (flat-spectrum noise floor on v_cv: "
              f"{results[0]['flat_noise_sd']:.4f}; corr floor ~"
              f"{1/np.sqrt(results[0]['corr_samples']):.4f})")
        hdr = (f"{'arm':6s} {'v_cv':>7s} {'p95/p5':>8s} {'max/min':>9s} {'TV':>6s} "
               f"{'k_spec':>7s} {'k_node':>7s} {'od_spec':>8s} {'od_node':>8s} {'lag1':>6s}")
        print(hdr)
        for r in results:
            row = (f"{r['arm']:6s} {r['v_cv']:7.3f} {r['v_ratio_95_5']:8.2f} "
                   f"{r['v_max_min']:9.1f} {r['tv_iso']:6.3f} {r['kurt_spec']:7.2f} "
                   f"{r['kurt_node']:7.2f} {r['offdiag_spec']:8.4f} "
                   f"{r['offdiag_node']:8.4f} {r['lag1']:6.3f}")
            print(row)
            lines.append(f"| {r['arm']} | {r['v_cv']:.3f} | {r['v_ratio_95_5']:.2f} "
                         f"| {r['v_max_min']:.1f} | {r['tv_iso']:.3f} "
                         f"| {r['kurt_spec']:.2f} | {r['kurt_node']:.2f} "
                         f"| {r['offdiag_spec']:.4f} | {r['offdiag_node']:.4f} "
                         f"| {r['lag1']:.3f} |")
        lines += ["", f"noise floors: v_cv {results[0]['flat_noise_sd']:.4f}, "
                      f"offdiag {1/np.sqrt(results[0]['corr_samples']):.4f}",
                  f"figure: {fig_path.name}", ""]
        # per-arm raw residual scale for context
        for r in results:
            lines.append(f"- {r['arm']}: raw residual std {r['resid_std_raw']:.3f} "
                         f"({r['B']} windows)")
        lines.append("")

    (out_dir / f"summary_{args.split}.md").write_text("\n".join(lines))
    print(f"\n[done] {out_dir / f'summary_{args.split}.md'}")


if __name__ == "__main__":
    main()
