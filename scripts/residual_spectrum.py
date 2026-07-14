"""settle the 'is the residual graph-smooth?' question with numbers.

two decompositions:
 (1) graph spectral energy of the RESIDUAL: project onto graph laplacian eigvecs,
     variance per graph-frequency mode g[k], averaged over horizon+windows. bucket
     into low/mid/high λ bands. a SMOOTH field concentrates energy at low λ; the
     (L+γI)^-1 prior ASSUMES that (variance ∝ 1/(λ+γ), monotone-decreasing).
 (2) NLL gain decomposition: iso -> temporal-only (pT, no graph) -> full kron.
     tells us how much of 'kron beats iso' is TEMPORAL (horizon variance growth)
     vs a genuine SPATIAL/graph contribution.
"""
from __future__ import annotations

import argparse
import numpy as np

from src.data.residual_dataset import train_residual_sigma
from src.diffusion.prior import _fit_sigma_T, _load_laplacian_eig
from src.utils import paths


def load_resid(ds, split, seed=42, bb="gwn", norm="global"):
    d = np.load(paths.residual_npz(ds, split, seed, bb))
    R = d["residuals"].astype(np.float64)
    mask = d["mask"].astype(bool)
    sig = np.asarray(train_residual_sigma(ds, seed, bb, norm=norm))
    return R / sig.reshape(-1, 1), mask


def run(ds, seed, bb, cap=6000):
    R, M = load_resid(ds, "train", seed, bb)
    keep = M.reshape(M.shape[0], -1).all(axis=1)
    Rv = R[keep]
    if Rv.shape[0] > cap:
        Rv = Rv[np.linspace(0, Rv.shape[0] - 1, cap).astype(int)]
    _, N, H = Rv.shape

    lam_S, V_S = _load_laplacian_eig(ds, repaired=True)
    order = np.argsort(lam_S)
    lam_S, V_S = lam_S[order], V_S[:, order]

    # (1) spatial graph spectrum: c[b,k,h] = (V_S^T R[b])[k,h]; g[k]=mean_{b,h} c^2
    C = np.einsum("ki,bih->bkh", V_S.T, Rv)
    g = (C ** 2).mean(axis=(0, 2))                       # (N,) energy per graph mode
    g_frac = g / g.sum()
    lo = lam_S <= np.quantile(lam_S, 1/3)
    hi = lam_S >= np.quantile(lam_S, 2/3)
    mid = ~lo & ~hi
    # peak location: centroid of energy on the λ axis, vs a flat field's centroid
    centroid = float((g_frac * lam_S).sum())
    flat_centroid = float(lam_S.mean())

    print(f"\n=== {ds} (N={N}, H={H}, windows={Rv.shape[0]}) ===")
    print(f"  graph-spectral energy of the RESIDUAL by λ band:")
    print(f"     low-λ (smooth) : {g_frac[lo].sum()*100:5.1f}%   "
          f"mid-λ : {g_frac[mid].sum()*100:5.1f}%   "
          f"high-λ : {g_frac[hi].sum()*100:5.1f}%")
    print(f"     energy centroid on λ-axis: {centroid:.3f}   (flat/white field = {flat_centroid:.3f}, "
          f"λ_max={lam_S.max():.2f})")
    print(f"     -> residual is {'LOW-PASS (smooth)' if centroid < 0.8*flat_centroid else ('MID/HIGH-shifted (NOT smooth)' if centroid > flat_centroid else 'roughly FLAT')}")
    # does variance actually DECREASE with λ, as (L+γI)^-1 assumes?
    corr = float(np.corrcoef(lam_S, g)[0, 1])
    print(f"     corr(λ, per-mode variance) = {corr:+.3f}   "
          f"((L+γI)^-1 prior needs this strongly NEGATIVE)")

    # (2) NLL gain decomposition (per-dim, held-out val)
    Rva, Mva = load_resid(ds, "val", seed, bb)
    def moments(Rx, Mx):
        k = Mx.reshape(Mx.shape[0], -1).all(axis=1)
        X = Rx[k]
        if X.shape[0] > cap: X = X[np.linspace(0, X.shape[0]-1, cap).astype(int)]
        Cx = np.einsum("ki,bih->bkh", V_S.T, X)
        Cx = np.einsum("bkh,hg->bkg", Cx, V_T)
        return (Cx ** 2).mean(0)
    _, _, Sig_T = _fit_sigma_T(R, M)
    mu_T, V_T = np.linalg.eigh(Sig_T); mu_T = np.clip(mu_T, 1e-12, None)
    pT = 1.0 / mu_T
    S_tr = moments(R, M); S_va = moments(Rva, Mva)
    def nll(phi):
        D = phi.size; s = D / float((phi * S_tr).sum())
        th = s * phi
        return float((th * S_va - np.log(th)).sum()) / D
    iso = nll(np.ones((N, H)))
    temporal = nll(np.ones((N, 1)) * pT[None, :])         # pT only, spatial flat
    kron = nll((lam_S[:, None] + 0.8) * pT[None, :])
    print(f"  NLL/dim decomposition (lower=better):")
    print(f"     iso            {iso:8.4f}")
    print(f"     +temporal(pT)  {temporal:8.4f}   (Δ {temporal-iso:+.4f}  <- horizon variance)")
    print(f"     +graph(kron)   {kron:8.4f}   (Δ {kron-temporal:+.4f}  <- PURE spatial/graph)")
    tot = iso - kron
    if abs(tot) > 1e-9:
        print(f"     share of the iso->kron gain that is SPATIAL: {(temporal-kron)/tot*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pems08s", "pems08f", "metrla", "pemsbay", "pems04f"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="gwn")
    a = ap.parse_args()
    for ds in a.datasets:
        run(ds, a.seed, a.backbone)


if __name__ == "__main__":
    main()
