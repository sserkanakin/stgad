"""offline prior-family search: which spatiotemporal precision structure best fits
the gwn residuals, WITHOUT training the diffusion model.

idea: every candidate prior shares the same eigenbasis V_S ⊗ V_T (graph laplacian
eigvecs ⊗ empirical temporal eigvecs). so a prior is just a choice of per-mode
precision θ[k,h] in that basis, and the residual's fit is a masked gaussian nll we
can evaluate in closed form from the empirical per-mode second moments S[k,h].

we fit each family's scale analytically (nll-optimal s given a shape φ) on TRAIN and
score held-out gaussian nll on VAL, so a richer family can't win just by overfitting.

families:
  empirical   θ = 1/S_tr           per-mode ceiling (best possible in this basis)
  iso         θ = const            scale-matched white noise (the baseline)
  kron(γ)     θ = (λ_S+γ)·pT       CURRENT prior, multiplicative coupling; γ swept
  cart(γ,κ)   θ = (λ_S+γ) + κ·pT   cartesian product-graph, ADDITIVE coupling

usage:  python scripts/prior_family_search.py                 # default 3 datasets
        python scripts/prior_family_search.py --datasets pems08s pems08f metrla pems04f pemsbay
"""
from __future__ import annotations

import argparse

import numpy as np

from src.data.residual_dataset import train_residual_sigma
from src.diffusion.prior import _fit_sigma_T, _load_laplacian_eig
from src.utils import paths


def load_resid(dataset: str, split: str, seed: int = 42, backbone: str = "gwn",
               norm: str = "global") -> tuple[np.ndarray, np.ndarray]:
    """(B,N,H) residuals normalized by the TRAIN scaler (same space the prior fits) + mask."""
    d = np.load(paths.residual_npz(dataset, split, seed, backbone))
    R = d["residuals"].astype(np.float64)
    mask = d["mask"].astype(bool)
    sigma_r = np.asarray(train_residual_sigma(dataset, seed, backbone, norm=norm))
    return R / sigma_r.reshape(-1, 1), mask


def mode_second_moments(R: np.ndarray, mask: np.ndarray, V_S: np.ndarray, V_T: np.ndarray,
                        cap: int = 6000) -> np.ndarray:
    """S[k,h] = mean over fully-valid windows of the squared eigen-coefficient
    c[k,h] = (V_S^T R V_T)[k,h]. only windows valid across all (N,H) are used, so
    the orthogonal projection is exact (no missing entries to fake)."""
    keep = mask.reshape(mask.shape[0], -1).all(axis=1)
    Rv = R[keep]
    if Rv.shape[0] > cap:
        idx = np.linspace(0, Rv.shape[0] - 1, cap).astype(int)
        Rv = Rv[idx]
    # C[b] = V_S^T @ R[b] @ V_T
    C = np.einsum("ki,bih->bkh", V_S.T, Rv)
    C = np.einsum("bkh,hg->bkg", C, V_T)
    return (C ** 2).mean(axis=0), int(keep.sum())


def nll_per_dim(phi: np.ndarray, S_tr: np.ndarray, S_va: np.ndarray) -> float:
    """held-out gaussian nll/dim for θ = s·φ, with s fit on train (analytic optimum).
    up to the same additive const for every family, so differences are meaningful."""
    D = phi.size
    s = D / float((phi * S_tr).sum())          # nll-optimal scale on train
    theta = s * phi
    return float((theta * S_va - np.log(theta)).sum()) / D  # + const (dropped)


def separability(S: np.ndarray) -> tuple[float, float]:
    """how well the variance surface is explained by (a) multiplicative coupling
    (=> log S additively separable => current kronecker is the right form) vs
    (b) additive coupling (=> 1/S additively separable => product graph). returns
    (R²_mult, R²_add): fraction of variance the rank-1 two-way-mean fit captures."""
    def two_way_r2(M: np.ndarray) -> float:
        g = M.mean()
        fit = M.mean(1, keepdims=True) + M.mean(0, keepdims=True) - g  # additive model
        ss_res = ((M - fit) ** 2).sum()
        ss_tot = ((M - g) ** 2).sum()
        return 1.0 - ss_res / ss_tot
    return two_way_r2(np.log(S)), two_way_r2(1.0 / S)


def run_dataset(dataset: str, seed: int, backbone: str) -> None:
    R_tr, M_tr = load_resid(dataset, "train", seed, backbone)
    R_va, M_va = load_resid(dataset, "val", seed, backbone)
    _, N, H = R_tr.shape

    # shared eigenbasis: graph laplacian (spatial) ⊗ empirical Σ_T (temporal)
    lam_S, V_S = _load_laplacian_eig(dataset, repaired=True)
    _, _, Sigma_T = _fit_sigma_T(R_tr, M_tr)
    mu_T, V_T = np.linalg.eigh(Sigma_T)
    mu_T = np.clip(mu_T, 1e-12, None)
    pT = 1.0 / mu_T                                  # empirical temporal precision (H,)

    S_tr, n_tr = mode_second_moments(R_tr, M_tr, V_S, V_T)
    S_va, n_va = mode_second_moments(R_va, M_va, V_S, V_T)

    lamS = lam_S[:, None]                             # (N,1)
    pTh = pT[None, :]                                 # (1,H)

    cands: list[tuple[str, np.ndarray]] = [
        ("empirical(ceiling)", 1.0 / S_tr),
        ("iso", np.ones((N, H))),
    ]
    for g in (0.05, 0.2, 0.8, 2.0):
        cands.append((f"kron  γ={g:<4}", (lamS + g) * pTh))         # current: multiplicative
    # cartesian product graph (additive) — small grid over shift γ and time-weight κ
    best_cart = None
    for g in (0.05, 0.2, 0.8):
        for k in (0.25, 0.5, 1.0, 2.0, 4.0):
            phi = (lamS + g) + k * pTh
            v = nll_per_dim(phi, S_tr, S_va)
            if best_cart is None or v < best_cart[1]:
                best_cart = (f"cart  γ={g},κ={k}", v, phi)
    # score everything
    rows = [(name, nll_per_dim(phi, S_tr, S_va)) for name, phi in cands]
    rows.append((best_cart[0] + " (best)", best_cart[1]))

    iso = dict(rows)["iso"]
    ceil = dict(rows)["empirical(ceiling)"]
    r2m, r2a = separability(S_tr)

    print(f"\n=== {dataset}  (N={N}, H={H}, valid windows train/val={n_tr}/{n_va}) ===")
    print(f"    variance separability:  multiplicative R²={r2m:.3f}   additive R²={r2a:.3f}"
          f"   ({'MULT' if r2m>r2a else 'ADD'} favored)")
    print(f"    {'family':22} {'val-NLL/dim':>12} {'Δ vs iso':>10} {'% of iso→ceil gap':>18}")
    span = iso - ceil
    for name, v in rows:
        pct = (iso - v) / span * 100 if span > 1e-9 else float("nan")
        tag = "  <-- ceiling" if "ceiling" in name else ("  (baseline)" if name == "iso" else "")
        print(f"    {name:22} {v:12.5f} {v-iso:+10.5f} {pct:17.1f}%{tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["pems08s", "pems08f", "metrla"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="gwn")
    args = ap.parse_args()
    print("offline prior-family search — held-out gaussian NLL/dim (lower=better fit)")
    for ds in args.datasets:
        run_dataset(ds, args.seed, args.backbone)


if __name__ == "__main__":
    main()