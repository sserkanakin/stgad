"""Do the residual's heavy tails call for a Student-t prior? — depends on WHERE the
kurtosis lives. Decompose it:

  pooled          kurtosis of all residual entries together (the 22-87 we saw)
  per-node std    standardize each sensor to unit var, then pool  (removes node-to-node
                  variance heterogeneity — the part per-node norm already handles)
  per-(node,h)    standardize each (sensor,horizon) cell           (removes horizon growth)
  per-MODE        kurtosis of each diffusion mode's coefficient over time, in the prior
                  eigenbasis — THIS is what a per-mode Gaussian prior actually faces.

If per-mode kurtosis ≈ 0, the heavy tails are pure heteroscedasticity (scale varying across
units) → a Gaussian prior with per-unit scaling already covers it, Student-t adds nothing.
If per-mode kurtosis stays high, the tails are GENUINE within each mode → only a heavy-tailed
(Student-t) prior can generate them. We also fit the matching t degrees-of-freedom ν
(excess kurt K = 6/(ν-4) → ν = 4 + 6/K) and quantify the Gaussian tail miss.

usage: .venv/bin/python scripts/analysis/tails.py [--datasets metrla pems08f]
"""
import argparse
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior, _load_residuals_train                 # noqa: E402
from src.utils import paths                                                       # noqa: E402

VARIANTS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]


def exkurt(x):
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return float("nan")
    m, s = x.mean(), x.std() + 1e-12
    return float(((x - m) ** 4).mean() / s ** 4 - 3.0)


def nu_from_kurt(k):
    """Student-t dof matching excess kurtosis k (k=6/(ν-4)); ν→∞ as k→0, ν→4 as k→∞."""
    return 4.0 + 6.0 / k if k and k > 0.05 else float("inf")


def analyze(ds, bb="gwn"):
    R, M = _load_residuals_train(ds, 42, bb)                 # already per... no: unit-space (global)
    # use per-node normalized residual (the diffusion's actual input space)
    from src.data.residual_dataset import train_residual_sigma
    sig = np.asarray(train_residual_sigma(ds, 42, bb, norm="pernode"))   # (N,)
    d = np.load(paths.residual_npz(ds, "train", 42, bb))
    R = d["residuals"].astype(np.float64) / sig.reshape(1, -1, 1)
    M = d["mask"].astype(bool)
    sub = np.linspace(0, len(R) - 1, min(3000, len(R))).astype(int)
    R, M = R[sub], M[sub]
    Rz = np.where(M, R, 0.0)

    out = {"ds": ds}
    # pooled
    out["pooled"] = exkurt(R[M])
    # per-node standardized then pooled
    zn = []
    for n in range(R.shape[1]):
        v = R[:, n, :][M[:, n, :]]
        if len(v) > 50:
            zn.append((v - v.mean()) / (v.std() + 1e-12))
    out["per_node"] = exkurt(np.concatenate(zn)) if zn else float("nan")
    # per-mode: project to prior eigenbasis, kurtosis of each mode over time, take median
    pr = load_prior(ds, prior="kronecker", seed=42, backbone=bb, temporal="empirical")
    C = np.matmul(pr.V_S.T, Rz); C = np.matmul(C, pr.V_T)    # (B,N,H) modes
    mk = np.array([exkurt(C[:, k, h]) for k in range(C.shape[1]) for h in range(C.shape[2])])
    mk = mk[np.isfinite(mk)]
    out["per_mode_med"] = float(np.median(mk))
    out["per_mode_frac_heavy"] = float((mk > 1.0).mean())    # share of modes clearly heavy
    # Gaussian tail miss: empirical P(|z|>4) vs Gaussian 6.33e-5
    z = R[M]; z = (z - z.mean()) / (z.std() + 1e-12)
    out["p_gt4"] = float((np.abs(z) > 4).mean())
    out["nu_pooled"] = nu_from_kurt(out["pooled"])
    out["nu_mode"] = nu_from_kurt(out["per_mode_med"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=VARIANTS)
    args = ap.parse_args()
    rows = [analyze(ds) for ds in args.datasets]

    print(f"{'dataset':9s} | {'pooled':>7s} {'per-node':>8s} {'per-mode':>8s} {'%heavy':>6s} | "
          f"{'ν(pool)':>7s} {'ν(mode)':>7s} | {'P(|z|>4)':>9s} {'×Gauss':>7s}")
    print("-" * 88)
    md = ["| dataset | pooled kurt | per-node kurt | per-mode kurt (median) | % modes heavy | ν(mode) | P(|z|>4) vs Gaussian |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    G4 = 6.334e-5
    for r in rows:
        print(f"{r['ds']:9s} | {r['pooled']:7.1f} {r['per_node']:8.1f} {r['per_mode_med']:8.2f} "
              f"{r['per_mode_frac_heavy']:6.0%} | {r['nu_pooled']:7.2f} {r['nu_mode']:7.1f} | "
              f"{r['p_gt4']:9.2e} {r['p_gt4']/G4:6.1f}×")
        md.append(f"| {r['ds']} | {r['pooled']:.1f} | {r['per_node']:.1f} | {r['per_mode_med']:.2f} | "
                  f"{r['per_mode_frac_heavy']:.0%} | {r['nu_mode']:.1f} | {r['p_gt4']/G4:.0f}× Gaussian |")
    print("\nexcess kurtosis: 0=Gaussian. ν=Student-t dof matching it (∞=Gaussian, →4=very heavy).")
    print("per-mode median is what a per-mode Gaussian prior faces. %heavy = share of modes with kurt>1.")
    print("P(|z|>4): how often a 4σ event happens vs Gaussian's 6.3e-5 (the tail a Gaussian says is ~impossible).")
    out = paths.analysis_dir("distribution") / "tails.md"
    out.write_text("\n".join(md) + "\n")
    print(f"\nwrote {out}")
    print("\nverdict: if per-mode kurtosis ≈0 → heteroscedastic, Student-t won't help (scale it per unit).")
    print("if per-mode kurtosis stays high → genuine tails, only a heavy-tailed prior can generate them.")


if __name__ == "__main__":
    main()
