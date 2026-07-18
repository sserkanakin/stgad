"""Graph-spectral analysis of the RAW signal, per dataset — the decisive offline
check for the structured-prior arm.

Two questions, answered per dataset with no GPU:
  1. Is the raw signal mostly LOW graph-frequency (graph-smooth)?  -> project each
     snapshot onto the graph-Laplacian eigenbasis, measure the fraction of energy
     in the lowest 10% of modes and the spectral centroid (white = flat).
  2. Does the KRONECKER prior concentrate energy the same way?  -> its spatial
     factor Σ_S = (L+γI)^-1 puts variance 1/(λ+γ) on graph mode λ, so its energy
     is analytically low-λ-weighted. We compare its low-λ fraction to the data's,
     against the ISOTROPIC prior (flat, low-λ fraction == fraction of modes).

If data low-λ fraction ≈ kron low-λ fraction ≫ iso, the structured prior is
justified for that dataset. Traffic residuals were only 4-10% graph-explained;
this measures the RAW signal, where the prior should fare better.

PREPROCESSING (the load-bearing choice): we center two ways and report both.
  - per-node: subtract each sensor's TRAIN mean. The principled GSP/GAD convention.
  - global:   subtract one scalar train mean. Leaves each sensor's static mean in
              the low-λ modes (nearby sensors ~ similar means -> smooth field), so
              global centering INFLATES low-λ energy artificially. Reporting both
              exposes how much of the "smoothness" is real dynamics vs mean spread.

Usage:  .venv/bin/python scripts/analyze_raw_spectrum.py
        .venv/bin/python scripts/analyze_raw_spectrum.py --gamma 0.8 0.04 --max-snap 5000
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.adjacency import load_adjacency, compute_normalized_laplacian   # noqa: E402
from src.utils import paths                                                    # noqa: E402

TRAFFIC = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]


def graph_eig(dataset):
    """repaired single-component sym-norm Laplacian -> sorted (λ, V)."""
    if dataset == "molene":
        L = np.load(paths.processed_dir("molene") / "adj.npz")["L"]
    else:
        L = compute_normalized_laplacian(load_adjacency(dataset, repair=True))
    lam, V = np.linalg.eigh(L)
    order = np.argsort(lam)
    return lam[order], V[:, order]


def raw_snapshots(dataset, max_snap):
    """(S, N) raw graph signals + per-node train mean + global train mean.

    traffic: value at each window origin (y[:,0]) ~ the underlying series.
    molene:  the 744 hourly temperature snapshots (train split = first 670).
    """
    if dataset == "molene":
        d = np.load(paths.processed_dir("molene") / "signals.npz")
        X = d["X"].astype(np.float64)                       # (744, N) raw kelvin
        n_train = int(d["train_idx"].shape[0])
    else:
        tr = np.load(paths.split_npz(dataset, "train"))
        X = tr["y"][:, 0, :, 0].astype(np.float64)          # (B, N) raw value
        n_train = X.shape[0]
    node_mean = X[:n_train].mean(0)                          # (N,) per-node train mean
    glob_mean = float(X[:n_train].mean())                    # scalar train mean
    if X.shape[0] > max_snap:                                # subsample for speed
        idx = np.linspace(0, X.shape[0] - 1, max_snap).astype(int)
        X = X[idx]
    return X, node_mean, glob_mean


def low_frac_and_centroid(energy, lam, low_k):
    """energy per mode (sorted low->high λ). low-λ fraction + spectral centroid."""
    e = energy / energy.sum()
    low = float(e[:low_k].sum())
    centroid = float((lam * e).sum())                        # raw λ centroid
    return low, centroid


def analyze(dataset, gammas, max_snap):
    lam, V = graph_eig(dataset)
    N = len(lam)
    low_k = max(1, N // 10)                                   # lowest 10% of modes
    X, node_mean, glob_mean = raw_snapshots(dataset, max_snap)

    out = {"dataset": dataset, "N": N, "low_k": low_k, "lam_max": float(lam.max())}
    for tag, mean in (("pernode", node_mean), ("global", glob_mean)):
        Xc = X - mean                                        # center
        C = Xc @ V                                           # (S, N) graph-fourier coeffs
        energy = (C ** 2).mean(0)                            # energy per mode
        low, cen = low_frac_and_centroid(energy, lam, low_k)
        out[f"data_{tag}_lowfrac"] = low
        out[f"data_{tag}_centroid"] = cen

    # prior spectra (analytic): energy per mode = variance per mode
    for g in gammas:
        var_kron = 1.0 / (lam + g)                           # Σ_S=(L+γI)^-1 mode variance
        low, cen = low_frac_and_centroid(var_kron, lam, low_k)
        out[f"kron_g{g}_lowfrac"] = low
        out[f"kron_g{g}_centroid"] = cen
    # isotropic: uniform variance
    var_iso = np.ones_like(lam)
    low, cen = low_frac_and_centroid(var_iso, lam, low_k)
    out["iso_lowfrac"] = low
    out["iso_centroid"] = cen
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=TRAFFIC + ["molene"])
    ap.add_argument("--gamma", nargs="+", type=float, default=[0.8, 0.04])
    ap.add_argument("--max-snap", type=int, default=5000)
    args = ap.parse_args()

    rows = [analyze(ds, args.gamma, args.max_snap) for ds in args.datasets]

    g0 = args.gamma[0]
    hdr = (f"{'dataset':9s} {'N':>4s} | {'data low-λ%':>22s} | "
           f"{'kron γ='+str(g0):>10s} {'iso':>6s} | verdict")
    print("\nfraction of raw-signal energy in the lowest 10% of graph modes")
    print("(pernode = principled; global = inflated by static sensor means)\n")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        pn, gl = r["data_pernode_lowfrac"], r["data_global_lowfrac"]
        kr, iso = r[f"kron_g{g0}_lowfrac"], r["iso_lowfrac"]
        # graph-smooth if pernode-centered data is well above the flat iso baseline
        smooth = "graph-smooth" if pn > 2 * iso else ("mild" if pn > 1.3 * iso else "~white")
        match = "kron≈data" if abs(kr - pn) < 0.12 else ("kron>data" if kr > pn else "kron<data")
        print(f"{r['dataset']:9s} {r['N']:4d} | pernode {pn:5.1%}  global {gl:5.1%} | "
              f"{kr:8.1%} {iso:5.1%} | {smooth}, {match}")
    print(f"\niso low-λ% == 10% by construction (flat spectrum). kron γ values: {args.gamma}")

    out = paths.RESULTS / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw_spectrum.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out/'raw_spectrum.json'}")


if __name__ == "__main__":
    main()
