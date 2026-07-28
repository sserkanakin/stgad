"""fit GTCNN's PARAMETRIC product graph (their eq. 10) as a diffusion forward operator.

GTCNN generalises the three fixed product graphs with a learned coupling

    S_o = sum_{i,j in {0,1}} s_ij (S_T^i (x) S^j)

which, on the factor eigenbases, gives a per-mode precision

    theta[k,h] = s00 + s01*lam_S[k] + s10*lam_T[h] + s11*lam_S[k]*lam_T[h]

Four scalars. s00 is a self-loop term and plays exactly the role of GAD's gamma, so the
family strictly CONTAINS cartesian (s=[0,1,1,0]), tensor ([0,0,0,1]) and strong
([0,1,1,1]) -- fitting it asks the data where on the product-graph continuum it sits.

Why this and not another sweep: the pre-registered mode-slope test (DECISIONS.md
2026-07-27) came back AMBIGUOUS -- neither the pure product nor the pure sum form fit,
0/12 cells, both with huge SSE. A four-parameter family that interpolates between them,
fit by held-out likelihood, is the principled way out of an ambiguous binary test.

We fit the SHAPE only. The stationary variance per mode is sigma^2/theta, so an overall
scale on theta is absorbed by sigma^2; we therefore normalise the fitted spectrum to
[0,2] at load time exactly like the fixed forms, which makes theta_min, the stiffness and
the GAD mixing number identical across every arm. The fit is over the raw SIGNAL's
measured per-mode variance on TRAIN, scored on VAL.

usage:
    python scripts/analysis/fit_product_graph.py --dataset pems08f --backbone raw
    python scripts/analysis/fit_product_graph.py --all --backbone raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.adjacency import compute_normalized_laplacian  # noqa: E402
from src.diffusion.prior import _load_residuals_train, _spatial_graph_eig  # noqa: E402
from src.utils import paths  # noqa: E402

DATASETS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
GAMMA = 0.1                       # the campaign's fixed gamma; must match load_prior's default
# the three fixed forms, as coefficient vectors in the SAME parameterisation --
# these are the reference points the fit is scored against.
FIXED = {"cartesian": (0.0, 1.0, 1.0, 0.0),
         "tensor": (0.0, 0.0, 0.0, 1.0),
         "strong": (0.0, 1.0, 1.0, 1.0)}
# a flat spectrum (all couplings zero) is the isotropic control in this parameterisation
ISO = (1.0, 0.0, 0.0, 0.0)


def _mode_variance(dataset, seed, backbone, split):
    """measured per-mode variance v[k,h] of the field in the (V_S (x) V_T) basis."""
    if split == "train":
        R, mask = _load_residuals_train(dataset, seed, backbone, "global")
    else:
        d = np.load(paths.residual_npz(dataset, split, seed, backbone))
        R = d["residuals"].astype(np.float64)
        R = R / R.std()                       # same unit space as the train loader
        mask = d["mask"].astype(bool)
    lam_S, V_S = _spatial_graph_eig(dataset, R, "road", 15, True)
    H = R.shape[-1]
    A_T = np.zeros((H, H))
    i = np.arange(H - 1)
    A_T[i, i + 1] = A_T[i + 1, i] = 1.0
    lam_T, V_T = np.linalg.eigh(compute_normalized_laplacian(A_T))
    lam_T = np.clip(lam_T, 0.0, None)
    Rz = np.where(mask, R, 0.0)
    C = np.matmul(V_S.T, Rz)                  # (B,N,H)
    C = np.matmul(C, V_T)
    v = np.clip((C ** 2).mean(0), 1e-10, None)   # (N,H)
    return lam_S, lam_T, v


def _theta(s, lam_S, lam_T, gamma=GAMMA):
    """EXACTLY what prior._product_spectrum + load_prior will build at training time:
    combine the couplings, shift the slowest mode to 0, rescale to [0,2], add gamma.

    note s[0] (GTCNN's self-loop term) is REMOVED by the shift -- under this
    normalisation it is exactly degenerate with gamma, so only the three coupling
    coefficients carry shape. we keep it in the vector for fidelity to their eq. 10 and
    report it as gamma."""
    a, b = lam_S[:, None], lam_T[None, :]
    lam = s[0] + s[1] * a + s[2] * b + s[3] * a * b
    lam = lam - lam.min()
    m = float(lam.max())
    lam = 2.0 * lam / m if m > 0 else lam
    return lam + gamma


def _nll(s, lam_S, lam_T, v, gamma=GAMMA):
    """per-mode Gaussian NLL of a field with measured variance v under a prior whose
    stationary variance is sigma^2/theta, with sigma^2 profiled out analytically.

      NLL(s) = mean[ log(sigma^2/theta) + v*theta/sigma^2 ] / 2 + const
    d/d sigma^2 = 0  ->  sigma^2 = mean(v*theta), giving the profiled objective
      NLL(s) = ( log mean(v*theta) - mean(log theta) + 1 ) / 2
    which depends only on the SHAPE of theta -- exactly what we want to fit.
    """
    th = _theta(s, lam_S, lam_T, gamma)
    if not np.all(np.isfinite(th)) or th.min() <= 0:
        return 1e6
    return 0.5 * (np.log((v * th).mean()) - np.log(th).mean() + 1.0)


def fit(dataset, seed=42, backbone="raw", verbose=True):
    lam_S, lam_T, v_tr = _mode_variance(dataset, seed, backbone, "train")
    _, _, v_va = _mode_variance(dataset, seed, backbone, "val")

    # only the three COUPLING coefficients carry shape (s00 is shifted out and equals
    # gamma), so fit those and pin s00 = gamma for the record. multi-start because the
    # profiled objective is non-convex in the coupling ratios.
    def obj(c):
        return _nll(np.concatenate([[GAMMA], c]), lam_S, lam_T, v_tr)

    best = None
    for c0 in ([1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0],
               [0.1, 1.0, 0.0], [1.0, 0.1, 0.0], [0.5, 2.0, 0.5]):
        r = minimize(obj, np.array(c0, float), method="Nelder-Mead",
                     options=dict(maxiter=20000, xatol=1e-10, fatol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    # the [0,2] rescale makes the coupling vector scale-invariant; normalise it so the
    # saved numbers are comparable across datasets and readable as a mixing ratio.
    c = best.x / max(np.abs(best.x).max(), 1e-12)
    s = np.concatenate([[GAMMA], c])

    rows = {"parametric": s}
    rows.update({k: np.array(vv) for k, vv in FIXED.items()})
    rows["iso(flat)"] = np.array(ISO)
    scores = {k: (_nll(vv, lam_S, lam_T, v_tr), _nll(vv, lam_S, lam_T, v_va))
              for k, vv in rows.items()}
    iso = scores["iso(flat)"]

    if verbose:
        print(f"\n=== {dataset} / {backbone} (seed {seed}) ===")
        print(f"  {'form':12} {'s00=g':>8} {'s01(S)':>8} {'s10(T)':>8} {'s11(ST)':>8} "
              f"{'NLL tr':>9} {'NLL val':>9} {'vs iso':>8}")
        for k in ["parametric", "cartesian", "strong", "tensor", "iso(flat)"]:
            cc, (tr, va) = rows[k], scores[k]
            print(f"  {k:12} {cc[0]:8.4f} {cc[1]:8.4f} {cc[2]:8.4f} {cc[3]:8.4f} "
                  f"{tr:9.4f} {va:9.4f} {iso[1]-va:+8.4f}")
        sc = np.abs(c) / max(np.abs(c).max(), 1e-12)
        print(f"  -> normalised couplings  spatial={sc[0]:.3f}  temporal={sc[1]:.3f}  "
              f"product={sc[2]:.3f}   (+ = better than iso in the 'vs iso' column)")
    return s, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--all", action="store_true")
    p.add_argument("--backbone", default="raw")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    todo = DATASETS if a.all else [a.dataset]
    if not todo or todo == [None]:
        p.error("pass --dataset or --all")

    out = paths.analysis_dir("product_graph")
    summary = {}
    for ds in todo:
        s, scores = fit(ds, a.seed, a.backbone)
        np.save(out / f"s_ij_{ds}_{a.backbone}_seed{a.seed}.npy", s)
        summary[ds] = {"s_ij": s.tolist(),
                       "nll": {k: list(map(float, v)) for k, v in scores.items()}}
    (out / f"fit_summary_{a.backbone}_seed{a.seed}.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nwrote coefficients + fit_summary to {out}")


if __name__ == "__main__":
    main()
