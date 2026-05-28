"""Section 3 — Experiment 3.3: Fit γ by spectral matching (Approach B).

For each dataset:
  1. Recompute the H4-style mask-aware pairwise covariance Σ̂ (HN × HN).
  2. Van-Loan rearrange + SVD → rank-1 NKP factors → Σ̂_S (N × N), symmetrised.
  3. Project into Laplacian eigenbasis: M̂ = V^T Σ̂_S V.
  4. diag-fraction = Σ_k m̂_k² / ||M̂||_F²    (sanity: is V the right basis?)
  5. Fit γ*_B by log-space matching of the diagonal m̂_k against 1/(λ_k + γ),
     up to an overall scale (i.e. minimise variance over k of
     [log m̂_k + log(λ_k + γ)]).

Outputs:
    results/section3/<ds>/exp33_spectral_match/{summary.json, log_log_scatter.png, sigma_s.npz}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from src.data.adjacency import (
    compute_normalized_laplacian,
    laplacian_eigendecomposition,
    load_adjacency,
)

DATASETS = ("metrla", "pemsbay", "pems08f")
OUT_ROOT = Path("results/section3")
RES_TMPL = "results/residuals/gwn_{ds}_train_seed0.npz"


def empirical_sigma_full(R: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask-aware pairwise covariance (HN, HN). Same convention as H4."""
    B, N, H = R.shape
    R_f = R.astype(np.float64)
    M_f = mask.astype(np.float64)
    count_cell = M_f.sum(axis=0)
    mean_cell = np.where(
        count_cell > 0,
        (R_f * M_f).sum(axis=0) / np.where(count_cell > 0, count_cell, 1.0),
        0.0,
    )
    Rc = (R_f - mean_cell) * M_f
    Rflat = Rc.transpose(0, 2, 1).reshape(B, H * N)
    Mflat = M_f.transpose(0, 2, 1).reshape(B, H * N)
    Cnum = Rflat.T @ Rflat
    Cden = Mflat.T @ Mflat
    with np.errstate(divide="ignore", invalid="ignore"):
        Sigma = np.where(Cden > 0, Cnum / Cden, 0.0)
    return Sigma


def nkp_sigma_s(Sigma: np.ndarray, N: int, H: int) -> tuple[np.ndarray, float, float]:
    """Best rank-1 NKP factor Σ̂_S from Σ (vec convention: flat = h*N + i).

    Returns (Sigma_S (N×N, symmetric), sigma_1, sigma_total_F).
    """
    M_vl = Sigma.reshape(H, N, H, N).transpose(0, 2, 1, 3).reshape(H * H, N * N)
    U, s, Vt = np.linalg.svd(M_vl, full_matrices=False)
    sigma_1 = float(s[0])
    sigma_total = float(np.sqrt((s ** 2).sum()))
    u1, v1 = U[:, 0], Vt[0, :]

    # NKP factors: Σ_T = √σ_1 * reshape(u1, H, H);  Σ_S = √σ_1 * reshape(v1, N, N).
    # Joint sign flip leaves the product unchanged; orient so trace(Σ_S) > 0.
    Sigma_S = np.sqrt(sigma_1) * v1.reshape(N, N)
    Sigma_T_check = np.sqrt(sigma_1) * u1.reshape(H, H)
    if np.trace(Sigma_S) < 0:
        Sigma_S = -Sigma_S
        Sigma_T_check = -Sigma_T_check
    # Symmetrise to absorb numerical noise.
    Sigma_S = 0.5 * (Sigma_S + Sigma_S.T)
    return Sigma_S, sigma_1, sigma_total, Sigma_T_check


def fit_gamma_spectral(m_hat: np.ndarray, eigvals: np.ndarray) -> tuple[float, float]:
    """Fit γ minimising variance over k of [log m̂_k + log(λ_k + γ)].

    Implicit scale c = exp(-mean(log m̂_k + log(λ_k + γ))). Returns (γ*, log_c).
    Drops any k with m̂_k ≤ 0 (a few near-zero modes have tiny negative values
    from rank-1 projection noise; they'd dominate a log fit).
    """
    valid = m_hat > 1e-15
    m = m_hat[valid]
    lam = eigvals[valid]

    def cost(log_gamma: float) -> float:
        gamma = float(np.exp(log_gamma))
        x = np.log(m) + np.log(lam + gamma)
        return float(x.var())

    res = minimize_scalar(cost, bounds=(np.log(1e-6), np.log(1e4)), method="bounded",
                          options={"xatol": 1e-8})
    gamma_star = float(np.exp(res.x))
    log_c = float(-(np.log(m) + np.log(lam + gamma_star)).mean())
    return gamma_star, log_c


def plot_log_log(
    dataset: str, m_hat: np.ndarray, eigvals: np.ndarray, gamma_b: float,
    log_c: float, gamma_a: float, out_dir: Path
) -> None:
    valid = m_hat > 1e-15
    m = m_hat[valid]
    lam = eigvals[valid]

    fig, ax = plt.subplots(figsize=(6, 5))
    x_b = np.log(1.0 / (lam + gamma_b)) + log_c  # log of fitted prior spectral variance
    ax.scatter(np.log(m), x_b, s=8, alpha=0.6, label="per-mode (k)")

    lo = min(np.log(m).min(), x_b.min())
    hi = max(np.log(m).max(), x_b.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, label="y = x (ideal)")
    ax.set_xlabel(r"$\log \hat m_k$  (empirical)")
    ax.set_ylabel(r"$\log c \cdot 1/(\lambda_k + \gamma^\star_B)$  (prior fit)")
    ax.set_title(
        f"{dataset} — Approach B fit\n"
        fr"$\gamma^\star_B = {gamma_b:.4f}$  (vs. $\gamma^\star_A = {gamma_a:.4f}$)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "log_log_scatter.png", dpi=140)
    plt.close(fig)


def analyse(dataset: str, gamma_a: float) -> dict:
    print(f"  [{dataset}] loading residuals ...")
    data = np.load(RES_TMPL.format(ds=dataset))
    R = data["residuals"].astype(np.float32)
    mask = data["mask"].astype(bool)
    B, N, H = R.shape

    print(f"    R.shape = {R.shape}; building HN×HN cov (HN={H*N}) ...")
    Sigma = empirical_sigma_full(R, mask)

    print(f"    NKP rank-1 factorisation ...")
    Sigma_S, sigma_1, sigma_total, Sigma_T_check = nkp_sigma_s(Sigma, N, H)
    print(f"    σ_1/||Σ||_F = {sigma_1 / sigma_total:.4f}, trace(Σ_S) = {np.trace(Sigma_S):.4e}")

    # Eigenbasis projection.
    A = load_adjacency(dataset)
    L = compute_normalized_laplacian(A)
    eigvals, V = laplacian_eigendecomposition(L, dataset, cache=True)

    print(f"    projecting Σ̂_S into Laplacian eigenbasis ...")
    M_hat = V.T @ Sigma_S @ V                                       # (N, N)
    m_hat = np.diag(M_hat).copy()                                   # (N,)

    M_F2 = float((M_hat ** 2).sum())
    diag_F2 = float((m_hat ** 2).sum())
    diag_fraction = diag_F2 / M_F2 if M_F2 > 0 else 0.0
    n_negative = int((m_hat < 0).sum())

    print(f"    fitting γ*_B ...")
    gamma_b, log_c = fit_gamma_spectral(m_hat, eigvals)

    # Save Σ̂_S + diagonal for downstream (3.5 / 3.6 / Section 4).
    out_dir = OUT_ROOT / dataset / "exp33_spectral_match"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "sigma_s.npz",
             Sigma_S=Sigma_S, m_hat=m_hat, eigvals=eigvals,
             gamma_b=gamma_b, log_c=log_c)
    plot_log_log(dataset, m_hat, eigvals, gamma_b, log_c, gamma_a, out_dir)

    return {
        "dataset": dataset,
        "N": int(N),
        "H": int(H),
        "windows": int(B),
        "sigma_1": sigma_1,
        "sigma_total_frobenius": sigma_total,
        "rank1_explained": sigma_1 / sigma_total,
        "trace_Sigma_S": float(np.trace(Sigma_S)),
        "Sigma_S_symmetry_err": float(np.abs(Sigma_S - Sigma_S.T).max()),
        "M_F2": M_F2,
        "diag_F2": diag_F2,
        "diag_fraction": diag_fraction,
        "n_negative_diag_modes": n_negative,
        "m_hat_min": float(m_hat.min()),
        "m_hat_max": float(m_hat.max()),
        "gamma_star_B": gamma_b,
        "log_c": log_c,
        "gamma_star_A_ref": gamma_a,
        "ratio_B_over_A": gamma_b / gamma_a,
    }


def main() -> None:
    gamma_a_lookup = {
        r["dataset"]: r["gamma_star_A"]
        for r in json.loads((OUT_ROOT / "exp32_energy_match_summary.json").read_text())
    }

    rows = []
    for dataset in DATASETS:
        result = analyse(dataset, gamma_a_lookup[dataset])
        out_dir = OUT_ROOT / dataset / "exp33_spectral_match"
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
        rows.append(result)
        print(
            f"[{dataset}]  γ*_B={result['gamma_star_B']:.4f}  "
            f"γ*_A={result['gamma_star_A_ref']:.4f}  ratio={result['ratio_B_over_A']:.2f}  "
            f"diag-frac={result['diag_fraction']:.3f}  "
            f"rank1={result['rank1_explained']:.3f}  "
            f"#neg-modes={result['n_negative_diag_modes']}"
        )

    summary_path = OUT_ROOT / "exp33_spectral_match_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote cross-dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
