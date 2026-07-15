"""export our spatio-temporal windows in the GAD reference repo's input format,
so their released scripts (exp_gasde.py / exp_vpsde.py / exp_vesde.py) run UNMODIFIED
on our data with the space-time product graph:

  x_train.npy  (2000, N*H)  centered raw windows, flattened (vec: node-major, h fastest)
  x_test.npy   ( 500, N*H)
  L.npy        (N*H, N*H)   normalized Cartesian product Laplacian  λ ∈ [0, 2]
  L_sym.npy    same matrix (it is symmetric; kept for their loader's naming)

matches their METR-LA protocol: 2000 train / 500 test randomly sampled signals.
fully-observed windows only (no masked-coordinate poisoning).

usage: python scripts/export_gad_repo_data.py --dataset metrla --out third_party_data/gadst_metrla
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.adjacency import compute_normalized_laplacian, load_adjacency
from src.data.residual_dataset import train_residual_sigma
from src.utils import paths


def load_windows(ds, split, cap, seed=0):
    d = np.load(paths.residual_npz(ds, split, 42, "raw"))
    sr = float(np.asarray(train_residual_sigma(ds, 42, "raw", norm="global")))
    R = d["residuals"].astype(np.float64) / sr                 # centered, unit variance
    M = d["mask"].astype(bool)
    keep = M.reshape(M.shape[0], -1).all(1)
    X = R[keep]
    rng = np.random.default_rng(seed)
    if X.shape[0] > cap:
        X = X[rng.choice(X.shape[0], cap, replace=False)]
    return X                                                    # (B, N, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metrla")
    ap.add_argument("--n-train", type=int, default=2000)        # GAD's METR-LA protocol
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Xtr = load_windows(args.dataset, "train", args.n_train)
    Xte = load_windows(args.dataset, "test", args.n_test)
    B, N, H = Xtr.shape

    lam_S_L = compute_normalized_laplacian(np.asarray(load_adjacency(args.dataset, repair=True)))
    A_T = np.zeros((H, H)); i = np.arange(H - 1)
    A_T[i, i + 1] = 1.0; A_T[i + 1, i] = 1.0
    L_T = compute_normalized_laplacian(A_T)
    # normalized Cartesian product Laplacian: (L_S ⊕ L_T)/2, spectrum ⊂ [0, 2].
    L_ST = 0.5 * (np.kron(lam_S_L, np.eye(H)) + np.kron(np.eye(N), L_T))

    # flatten windows with the SAME vec convention as the kron above: index = i*H + h.
    xtr = Xtr.reshape(Xtr.shape[0], N * H)
    xte = Xte.reshape(Xte.shape[0], N * H)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "x_train.npy", xtr.astype(np.float32))
    np.save(out / "x_test.npy", xte.astype(np.float32))
    np.save(out / "L.npy", L_ST.astype(np.float64))
    np.save(out / "L_sym.npy", L_ST.astype(np.float64))
    ev = np.linalg.eigvalsh(L_ST)
    print(f"[export] {args.dataset}: x_train {xtr.shape}, x_test {xte.shape}, "
          f"L_ST {L_ST.shape}, λ∈[{ev.min():.4f}, {ev.max():.4f}] (must be ⊂ [0,2])")
    print(f"[export] wrote {out}/")


if __name__ == "__main__":
    main()
