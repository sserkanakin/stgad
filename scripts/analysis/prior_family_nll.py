r"""Held-out NLL of the whole prior family. Fit on TRAIN residuals, scored on VAL.

WHY THIS EXISTS. Section 4.2.3 argues in prose why the prior is a Kronecker PRODUCT
of precisions rather than a product graph's Laplacian, and why the graph factor is
kept despite the attribution table saying it carries almost nothing. Neither claim
had an experiment. This script is that experiment, and it needs no GPU: every family
in the paper lives in the SAME separable basis V_S (x) V_T, so a family is nothing
but a per-mode variance profile v[k,h], and a Gaussian's held-out score is

    NLL = 0.5 * mean_modes( log v + vhat / v ),   vhat = held-out per-mode 2nd moment

Each family gets ONE free scale fitted on train, so families are compared at matched
total variance -- the same rule the trained arms are matched under.

Everything is computed in the space the diffusion actually runs in: per-sensor
normalised residuals (norm="pernode"), fully-observed windows only, seed 42.

usage: python scripts/analysis/prior_family_nll.py [--out results/analysis/...json]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from src.data.adjacency import compute_normalized_laplacian, load_adjacency
from src.data.residual_dataset import train_residual_sigma
from src.utils import paths

DS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
BB = ["mlp", "gcn", "gwn"]
NORM = "pernode"
CAP = 8000          # deterministic thinning; the mode variances converge long before this
SHRINK_T = 0.1      # matches prior.SHRINK_T


def unit_residuals(ds, bb, split, seed=42):
    """(B, N, H) train-scaled residuals, fully-observed windows only."""
    d = np.load(paths.residual_npz(ds, split, seed, bb))
    R = d["residuals"].astype(np.float64)
    M = d["mask"].astype(bool)
    sig = np.asarray(train_residual_sigma(ds, seed, bb, norm=NORM))
    R = R / (sig.reshape(-1, 1) if sig.ndim == 1 else sig)
    R = R[M.reshape(M.shape[0], -1).all(1)]
    if len(R) > CAP:
        R = R[np.linspace(0, len(R) - 1, CAP).astype(int)]
    return R


def mode_variance(R, V_S, V_T):
    """per-mode second moment in the V_S (x) V_T basis. BLAS, not einsum: the
    einsum form is O(B N^2 H) unblocked and takes minutes on pemsbay."""
    B, N, H = R.shape
    X = R.transpose(1, 0, 2).reshape(N, B * H)
    C = (V_S.T @ X).reshape(N, B, H).transpose(1, 0, 2)
    C = (C.reshape(B * N, H) @ V_T).reshape(B, N, H)
    return (C ** 2).mean(0)


def nll(v_model, v_hold):
    v = np.maximum(v_model, 1e-12)
    return 0.5 * float(np.mean(np.log(v) + v_hold / v))


def rescale(v, v_train):
    """the one free scale every family gets, fitted on train."""
    return v * (v_train.mean() / max(v.mean(), 1e-12))


def cell(ds, bb, lam, V_S, omega):
    R_tr, R_va = unit_residuals(ds, bb, "train"), unit_residuals(ds, bb, "val")
    H = R_tr.shape[2]
    Rk = R_tr.reshape(-1, H)
    mu_T, V_T = np.linalg.eigh((Rk.T @ Rk) / Rk.shape[0])
    mu_T = np.clip(mu_T, 1e-12, None)
    mu_T = (1 - SHRINK_T) * mu_T + SHRINK_T * mu_T.mean()
    v_tr, v_va = mode_variance(R_tr, V_S, V_T), mode_variance(R_va, V_S, V_T)
    N = v_tr.shape[0]

    fam = {
        "iso":       np.outer(np.ones(N), np.ones(H)),
        "temporal":  np.outer(np.ones(N), mu_T),              # Sigma_T (x) I
        "spatial":   np.outer(1 / (lam + 0.8), np.ones(H)),   # I (x) Sigma_S
        "kron_g0.8": np.outer(1 / (lam + 0.8), mu_T),         # the shipped prior
    }
    # gamma fitted per cell on TRAIN, then scored held-out
    gs = np.geomspace(0.01, 20, 60)
    g_hat = float(gs[int(np.argmin([nll(rescale(np.outer(1 / (lam + g), mu_T), v_tr), v_tr)
                                    for g in gs]))])
    fam["kron_gfit"] = np.outer(1 / (lam + g_hat), mu_T)
    # gamma AND a free exponent on the spatial precision: (lam+g)^-b
    best = None
    for g in np.geomspace(0.01, 20, 25):
        for b in np.linspace(0.2, 4.0, 20):
            s = nll(rescale(np.outer(1 / (lam + g) ** b, mu_T), v_tr), v_tr)
            if best is None or s < best[0]:
                best = (s, float(g), float(b))
    _, g2, b2 = best
    fam["kron_gbfit"] = np.outer(1 / (lam + g2) ** b2, mu_T)
    # data-fit spatial factor: how much is lost by IMPOSING the Laplacian shape
    fam["kron_empS"] = np.outer(v_tr.mean(1), mu_T)
    # the product-graph family (GTCNN Sec. 2.1): Q = L_diamond + gamma I, rescaled [0,2]
    for name, L in (("cartesian", lam[:, None] + omega[None, :]),
                    ("strong", (1 + lam[:, None]) * (1 + omega[None, :]) - 1),
                    ("tensor", lam[:, None] * omega[None, :])):
        fam[f"prod_{name}"] = 1 / (2 * L / L.max() + 0.1)
    fam["reshaped"] = v_tr.copy()                             # empirical per-mode = the ceiling

    base = nll(rescale(fam["iso"], v_tr), v_va)
    out = {k: base - nll(rescale(v, v_tr), v_va) for k, v in fam.items()}   # nats gained vs iso
    out.update(ds=ds, bb=bb, g_hat=g_hat, g_fit2=g2, b_fit2=b2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/analysis/prior_family_nll.json")
    p.add_argument("--datasets", nargs="*", default=DS)
    args = p.parse_args()

    H = 12
    A_T = np.zeros((H, H))
    i = np.arange(H - 1)
    A_T[i, i + 1] = A_T[i + 1, i] = 1.0
    omega = np.clip(np.linalg.eigvalsh(compute_normalized_laplacian(A_T)), 0, None)

    rows = []
    for ds in args.datasets:
        lam, V_S = np.linalg.eigh(compute_normalized_laplacian(load_adjacency(ds)))
        o = np.argsort(lam)
        lam, V_S = lam[o], V_S[:, o]
        for bb in BB:
            try:
                r = cell(ds, bb, lam, V_S, omega)
            except FileNotFoundError as e:
                print(f"skip {ds}/{bb}: {e}", flush=True)
                continue
            rows.append(r)
            print(f"{ds:8s} {bb:4s} | T {r['temporal']:+.3f}  S {r['spatial']:+.3f}  "
                  f"kron.8 {r['kron_g0.8']:+.3f}  gfit {r['kron_gfit']:+.3f}"
                  f"(g={r['g_hat']:.2f})  empS {r['kron_empS']:+.3f}  "
                  f"cart {r['prod_cartesian']:+.3f}  strong {r['prod_strong']:+.3f}  "
                  f"resh {r['reshaped']:+.3f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
