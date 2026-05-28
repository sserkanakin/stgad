"""Diagnostics for the locked Kronecker prior (Section 4/5).

Prior:
    Σ = Σ_T ⊗ Σ_S       (Kronecker, never materialised except for Diag 1)
    Σ_S = (L + γ·I)^{-1},  γ = 0.2 fixed across datasets
    Σ_T = D_H^{1/2} R_H D_H^{1/2}
          D_H[h,h] = (1.4826 · MAD_h)^2     (robust per-horizon variance)
          R_H      = corrcoef(R_flat.T)     (empirical correlation, PSD)

vec convention:
    R has shape (N, H). vec(R)[h*N + i] = R[i, h], i.e. column-major over N.
    Σ_kron = np.kron(Σ_T, Σ_S)  ⇒  Σ_kron[h*N+i, h'*N+i'] = Σ_T[h,h']·Σ_S[i,i'].

REVISION (2026-05-22). The locked spec describes a *smoothness* prior, not a
variance prior. Σ_S = (L+γI)^{-1} cannot match the empirical Σ̂_emp in Frobenius
(η_scaled ≈ 0.65–0.75 floor; H4 NKP bound is 0.19–0.26) because GWN residuals
have heterogeneous per-sensor variance that the parametric family cannot
represent. Decision: keep the spec; revise the diagnostic to test correlation
shape only and report the variance gap as an SDE-noise budget.

Diagnostics (revised):
    D1  Structural correlation match — (a) cos similarity between corr-normalised
        Σ_S_ours and corr-normalised empirical spatial covariance.
        (b) cos similarity for temporal side (sanity: should be ≈1).
        (c) Variance gap: relative ||τ²·Σ_kron - Σ̂_emp||_F / ||Σ̂_emp||_F after
        optimal scalar τ². Reported, not pass/fail — quantifies SDE noise budget.
    D2  ratio of empirical Var(R_{i,h}) to (Σ_S)_{ii}·(Σ_T)_{hh} — reframed as
        SDE noise sizing (no pass/fail).
    D3  cluster sanity on 200 joint samples (Fiedler-mode contribution + heatmaps).

Usage:
    python scripts/diag_prior_kron.py --dataset metrla --diag 1
    python scripts/diag_prior_kron.py --dataset all --diag 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse.csgraph

from src.data.adjacency import load_adjacency

GAMMA = 0.2  # fixed Laplacian shift, locked Section 4
SEED = 0
N_SAMPLES = 200
MAX_HOP = 8

DATASETS = ("metrla", "pems08f", "pemsbay")
H4_SEPARABILITY = {
    "metrla": 0.19084460989443597,
    "pems08f": 0.25571597285367736,
    "pemsbay": 0.19922864490397340,
}


# ───────────────────────────── data loading ───────────────────────────────

def load_residuals(dataset: str, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Return (R, mask) with shape (B, N, H) in raw units."""
    p = ROOT / "results" / "residuals" / f"gwn_{dataset}_{split}_seed0.npz"
    d = np.load(p)
    return d["residuals"].astype(np.float64), d["mask"].astype(bool)


def load_eig(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigvals_L, V_S) for the symmetric normalised Laplacian."""
    p = ROOT / "data" / "processed" / f"{dataset}_laplacian_eigs.npz"
    d = np.load(p)
    return d["eigenvalues"].astype(np.float64), d["eigenvectors"].astype(np.float64)


# ───────────────────────────── prior build ────────────────────────────────

def build_prior(R: np.ndarray, mask: np.ndarray, eigvals_L: np.ndarray,
                V_S: np.ndarray, gamma: float = GAMMA) -> dict:
    """Build the locked Kronecker prior. Returns cached factors only."""
    B, N, H = R.shape

    # Spatial: Σ_S = V_S diag(1/(λ+γ)) V_S^T
    eigvals_LgI = eigvals_L + gamma                              # (N,)
    Sigma_S = (V_S * (1.0 / eigvals_LgI)) @ V_S.T                # (N, N)

    # Temporal: flatten to (M, H), drop rows where any horizon is masked.
    R_flat = R.reshape(-1, H)                                    # (B·N, H)
    m_flat = mask.reshape(-1, H)
    valid_row = m_flat.all(axis=1)
    R_valid = R_flat[valid_row]                                  # (M, H)
    M_used = int(valid_row.sum())

    median_h = np.median(R_valid, axis=0)                        # (H,)
    mad_h = np.median(np.abs(R_valid - median_h), axis=0)        # (H,)
    delta_h = (1.4826 * mad_h) ** 2                              # (H,)

    R_H = np.corrcoef(R_valid.T)                                 # (H, H)
    D_sqrt = np.sqrt(delta_h)
    Sigma_T = (D_sqrt[:, None] * R_H) * D_sqrt[None, :]          # (H, H)

    # Eigendecomposition of Σ_T for sampling later.
    mu_T, V_T = np.linalg.eigh(Sigma_T)

    return {
        "N": N, "H": H, "gamma": gamma,
        "eigvals_L": eigvals_L, "V_S": V_S, "eigvals_LgI": eigvals_LgI,
        "Sigma_S": Sigma_S,
        "delta_h": delta_h, "R_H": R_H, "Sigma_T": Sigma_T,
        "mu_T": mu_T, "V_T": V_T,
        "M_temporal_rows": M_used,
    }


# ──────────────────────── empirical joint covariance ──────────────────────

def empirical_joint_cov(R: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Mask-aware Σ̂_emp matching H4 convention: vec index = h*N + i.

    Σ̂_emp[a, b] = Σ_s M[s,a] M[s,b] Rc[s,a] Rc[s,b] / Σ_s M[s,a] M[s,b]
    where Rc is centred at every (i, h) cell using only observed values.
    """
    B, N, H = R.shape
    M_f = mask.astype(np.float64)

    count_cell = M_f.sum(axis=0)                                 # (N, H)
    safe = np.where(count_cell > 0, count_cell, 1.0)
    mean_cell = (R * M_f).sum(axis=0) / safe                      # (N, H)

    Rc = (R - mean_cell) * M_f                                   # (B, N, H)

    Rflat = Rc.transpose(0, 2, 1).reshape(B, H * N)              # (B, HN)
    Mflat = M_f.transpose(0, 2, 1).reshape(B, H * N)             # (B, HN)

    Cnum = Rflat.T @ Rflat                                        # (HN, HN)
    Cden = Mflat.T @ Mflat                                        # (HN, HN)

    with np.errstate(divide="ignore", invalid="ignore"):
        Sigma = np.where(Cden > 0, Cnum / Cden, 0.0)

    info = {
        "pair_count_min": float(Cden[Cden > 0].min()) if (Cden > 0).any() else 0.0,
        "pair_count_median": float(np.median(Cden[Cden > 0])) if (Cden > 0).any() else 0.0,
    }
    return Sigma, info


# ─────────────────────────── Diagnostic 1 (revised) ──────────────────────

def _corr_normalise(M: np.ndarray) -> np.ndarray:
    """Diagonal-scale M to a correlation matrix (unit diagonal)."""
    d = np.sqrt(np.clip(np.diag(M), 1e-12, None))
    return M / np.outer(d, d)


def _matrix_cosine(A: np.ndarray, B: np.ndarray) -> float:
    return float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B)))


def empirical_spatial_corr(R: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask-aware spatial covariance averaged over (window, horizon), then
    normalised to a correlation matrix. Shape (N, N).
    """
    B, N, H = R.shape
    Mf = mask.astype(np.float64)
    # Cell mean (per i across b, h) — use horizon-marginalised mean per sensor.
    cnt = Mf.sum(axis=(0, 2))                                     # (N,)
    safe = np.where(cnt > 0, cnt, 1.0)
    mu_i = (R * Mf).sum(axis=(0, 2)) / safe                       # (N,)
    Rc = (R - mu_i[None, :, None]) * Mf                           # (B, N, H)

    # Σ_S_emp[i, i'] = sum_{b,h} Rc[b,i,h]·Rc[b,i',h] / sum_{b,h} M[b,i,h]·M[b,i',h]
    Rc_flat = Rc.transpose(0, 2, 1).reshape(B * H, N)             # (BH, N)
    Mf_flat = Mf.transpose(0, 2, 1).reshape(B * H, N)
    Cnum = Rc_flat.T @ Rc_flat                                     # (N, N)
    Cden = Mf_flat.T @ Mf_flat
    with np.errstate(divide="ignore", invalid="ignore"):
        Sigma_S_emp = np.where(Cden > 0, Cnum / Cden, 0.0)
    return _corr_normalise(Sigma_S_emp), Sigma_S_emp


def diag1(R: np.ndarray, mask: np.ndarray, prior: dict, dataset: str) -> dict:
    Sigma_emp, info = empirical_joint_cov(R, mask)
    Sigma_kron = np.kron(prior["Sigma_T"], prior["Sigma_S"])

    # (a) Spatial correlation cosine
    K_S_ours = _corr_normalise(prior["Sigma_S"])
    K_S_emp, Sigma_S_emp_raw = empirical_spatial_corr(R, mask)
    cos_spatial_corr = _matrix_cosine(K_S_ours, K_S_emp)

    # Also off-diagonal-only cosine (diagonal is 1 by construction → inflates similarity)
    iu = np.triu_indices_from(K_S_ours, k=1)
    cos_spatial_corr_offdiag = float(
        np.dot(K_S_ours[iu], K_S_emp[iu])
        / (np.linalg.norm(K_S_ours[iu]) * np.linalg.norm(K_S_emp[iu]))
    )

    # (b) Temporal correlation cosine (sanity check; R_H is empirical so ≈ 1)
    R_H_ours = prior["R_H"]
    # empirical temporal correlation (mask-aware), averaged over (b, i)
    H = prior["H"]
    Mf = mask.astype(np.float64)
    cnt_h = Mf.sum(axis=(0, 1))
    safe_h = np.where(cnt_h > 0, cnt_h, 1.0)
    mu_h = (R * Mf).sum(axis=(0, 1)) / safe_h
    Rc = (R - mu_h[None, None, :]) * Mf
    Rc_flat = Rc.reshape(-1, H)
    Mf_flat = Mf.reshape(-1, H)
    Cnum = Rc_flat.T @ Rc_flat
    Cden = Mf_flat.T @ Mf_flat
    with np.errstate(divide="ignore", invalid="ignore"):
        Sigma_T_emp = np.where(Cden > 0, Cnum / Cden, 0.0)
    R_H_emp = _corr_normalise(Sigma_T_emp)
    cos_temporal_corr = _matrix_cosine(R_H_ours, R_H_emp)

    # (c) Variance gap after optimal scalar τ²
    emp_fro = float(np.linalg.norm(Sigma_emp, "fro"))
    kron_fro = float(np.linalg.norm(Sigma_kron, "fro"))
    inner = float((Sigma_kron * Sigma_emp).sum())
    tau2_opt = inner / kron_fro ** 2 if kron_fro > 0 else float("nan")
    eta_raw = float(np.linalg.norm(Sigma_kron - Sigma_emp, "fro") / emp_fro)
    eta_scaled = float(np.linalg.norm(tau2_opt * Sigma_kron - Sigma_emp, "fro") / emp_fro)

    h4_eta = H4_SEPARABILITY[dataset]

    return {
        "diag": "D1_revised",
        "dataset": dataset,
        "spatial_corr_cosine": cos_spatial_corr,
        "spatial_corr_cosine_offdiag": cos_spatial_corr_offdiag,
        "temporal_corr_cosine": cos_temporal_corr,
        "tau2_opt": tau2_opt,
        "eta_raw": eta_raw,
        "eta_scaled": eta_scaled,
        "h4_nkp_floor": h4_eta,
        "kron_fro": kron_fro,
        "emp_fro": emp_fro,
        "pair_count_min": info["pair_count_min"],
        "pair_count_median": info["pair_count_median"],
    }


# ─────────────────────────── Diagnostic 2 ─────────────────────────────────
# Per-(sensor, horizon) marginal variance: empirical vs analytic.
# Reframed (2026-05-22): not a pass/fail; sizes the SDE noise budget.

def diag2(R: np.ndarray, mask: np.ndarray, prior: dict, dataset: str) -> dict:
    B, N, H = R.shape
    Sigma_S = prior["Sigma_S"]
    Sigma_T = prior["Sigma_T"]

    # Analytic marginals: Var_pred(R_{i,h}) = (Σ_S)_{ii} · (Σ_T)_{hh}
    diag_S = np.diag(Sigma_S)                       # (N,)
    diag_T = np.diag(Sigma_T)                       # (H,)
    analytic = np.outer(diag_S, diag_T)             # (N, H)

    # Empirical marginals: mask-aware per-(i,h) variance over windows.
    Mf = mask.astype(np.float64)
    cnt = Mf.sum(axis=0)                            # (N, H)
    safe = np.where(cnt > 0, cnt, 1.0)
    mu = (R * Mf).sum(axis=0) / safe                # (N, H)
    sq = ((R - mu) ** 2 * Mf).sum(axis=0) / safe    # (N, H)
    empirical = sq
    cell_counts = cnt                               # for reporting

    ratio = empirical / np.where(analytic > 0, analytic, np.nan)

    # Original pass criterion (informational, expected to fail under smoothness-prior view)
    mean_ratio = float(np.nanmean(ratio))
    std_ratio = float(np.nanstd(ratio))
    pass_mean_in_band = bool(0.8 <= mean_ratio <= 1.2)
    pass_std_under_03 = bool(std_ratio < 0.3)

    # Distribution
    pct = np.nanpercentile(ratio, [1, 5, 25, 50, 75, 95, 99])

    # Per-horizon breakdown
    per_h_mean = np.nanmean(ratio, axis=0).tolist()
    per_h_std = np.nanstd(ratio, axis=0).tolist()

    # Per-sensor breakdown
    per_i_mean = np.nanmean(ratio, axis=1)
    per_i_pct = np.nanpercentile(per_i_mean, [1, 5, 25, 50, 75, 95, 99])

    return {
        "diag": "D2",
        "dataset": dataset,
        "mean_ratio": mean_ratio,
        "std_ratio": std_ratio,
        "ratio_pct_1_5_25_50_75_95_99": pct.tolist(),
        "pass_mean_in_band": pass_mean_in_band,
        "pass_std_under_03": pass_std_under_03,
        "per_horizon_mean_ratio": per_h_mean,
        "per_horizon_std_ratio": per_h_std,
        "per_sensor_mean_ratio_pct_1_5_25_50_75_95_99": per_i_pct.tolist(),
        "analytic_diag_min": float(analytic.min()),
        "analytic_diag_max": float(analytic.max()),
        "empirical_diag_min": float(empirical.min()),
        "empirical_diag_max": float(empirical.max()),
        "sde_noise_multiplier_tau2_equiv": mean_ratio,  # mean ratio ≈ τ² needed for marginal match
    }


# ─────────────────────────── Diagnostic 3 ─────────────────────────────────

def _hop_distances(A: np.ndarray) -> np.ndarray:
    """All-pairs BFS hop count on symmetric binary adjacency."""
    Asym = A.copy()
    np.fill_diagonal(Asym, 0)
    Asym = (Asym + Asym.T) / 2
    binary = (Asym > 0).astype(np.int8)
    D = scipy.sparse.csgraph.shortest_path(binary, method="D", unweighted=True)
    D[~np.isfinite(D)] = binary.shape[0]
    return D.astype(int)


def _corr_vs_hop_for_field(samples_at_h: np.ndarray, D: np.ndarray,
                           max_hop: int = MAX_HOP) -> np.ndarray:
    """Mean pairwise correlation as function of hop, for (n_samples, N) field."""
    centred = samples_at_h - samples_at_h.mean(axis=0, keepdims=True)
    cov = (centred.T @ centred) / max(samples_at_h.shape[0] - 1, 1)
    std = np.sqrt(np.diag(cov).clip(1e-12))
    corr = cov / (std[:, None] * std[None, :])
    out = np.full(max_hop + 1, np.nan)
    for h in range(1, max_hop + 1):
        m = D == h
        if m.any():
            out[h] = float(corr[m].mean())
    return out


def _empirical_corr_vs_hop_at_horizon(R: np.ndarray, mask: np.ndarray,
                                      h_idx: int, D: np.ndarray,
                                      max_hop: int = MAX_HOP,
                                      n_windows_cap: int = 4000,
                                      rng: np.random.Generator = None) -> np.ndarray:
    """Same metric on empirical residuals at horizon h: pair correlation over windows.

    Only uses windows where every sensor was observed at this horizon.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    ok = mask[:, :, h_idx].all(axis=1)
    if not ok.any():
        return np.full(max_hop + 1, np.nan)
    Rh = R[ok][:, :, h_idx].astype(np.float64)
    if Rh.shape[0] > n_windows_cap:
        idx = rng.choice(Rh.shape[0], size=n_windows_cap, replace=False)
        Rh = Rh[idx]
    centred = Rh - Rh.mean(axis=0, keepdims=True)
    cov = (centred.T @ centred) / max(Rh.shape[0] - 1, 1)
    std = np.sqrt(np.diag(cov).clip(1e-12))
    corr = cov / (std[:, None] * std[None, :])
    out = np.full(max_hop + 1, np.nan)
    for h in range(1, max_hop + 1):
        m = D == h
        if m.any():
            out[h] = float(corr[m].mean())
    return out


def sample_joint(prior: dict, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n_samples joint fields R ~ MN(0, Σ_S, Σ_T) via Kronecker factor sqrts.

    Returns (n_samples, N, H).
    """
    N, H = prior["N"], prior["H"]
    V_S = prior["V_S"]
    eigvals_LgI = prior["eigvals_LgI"]
    V_T = prior["V_T"]
    mu_T = prior["mu_T"]
    # Clip tiny eigvals of Σ_T to zero for safety (R_H is empirical, may have tiny neg)
    sqrt_mu_T = np.sqrt(np.clip(mu_T, 0, None))

    samples = np.empty((n_samples, N, H), dtype=np.float64)
    for k in range(n_samples):
        Z = rng.standard_normal((N, H))
        # Σ_S^{1/2} Z Σ_T^{1/2} using eigenbases
        ZT = V_S.T @ Z @ V_T                                 # (N, H)
        ZT = (1.0 / np.sqrt(eigvals_LgI))[:, None] * ZT * sqrt_mu_T[None, :]
        samples[k] = V_S @ ZT @ V_T.T
    return samples


def diag3(R: np.ndarray, mask: np.ndarray, prior: dict, dataset: str,
          out_dir: Path) -> dict:
    rng = np.random.default_rng(SEED)
    N, H = prior["N"], prior["H"]

    samples = sample_joint(prior, N_SAMPLES, rng)            # (K, N, H)

    # Fiedler vector of L (second-smallest eigval; eigvals_L is sorted ascending)
    v_fiedler = prior["V_S"][:, 1]                            # (N,)
    # Slowest temporal mode = eigvector of Σ_T's largest eigvalue
    idx_top_T = int(np.argmax(prior["mu_T"]))
    v_T_top = prior["V_T"][:, idx_top_T]                      # (H,)

    # Fiedler-mode contribution per sample
    # contrib_k = (v_2^T @ R_k @ v_T1)^2 / ||R_k||_F^2
    projections = np.einsum("i,kih,h->k", v_fiedler, samples, v_T_top)  # (K,)
    energy = (samples ** 2).sum(axis=(1, 2))                  # (K,)
    fiedler_contrib = projections ** 2 / np.clip(energy, 1e-12, None)
    fiedler_mean = float(fiedler_contrib.mean())
    fiedler_p95 = float(np.percentile(fiedler_contrib, 95))
    fiedler_max = float(fiedler_contrib.max())
    pass_fiedler = bool(fiedler_mean < 0.10)

    # Per-horizon corr-vs-hop: samples vs empirical
    A = load_adjacency(dataset)
    Dhop = _hop_distances(A)

    corr_sample_per_h = np.zeros((H, MAX_HOP + 1))
    corr_emp_per_h = np.zeros((H, MAX_HOP + 1))
    for h in range(H):
        corr_sample_per_h[h] = _corr_vs_hop_for_field(samples[:, :, h], Dhop)
        corr_emp_per_h[h] = _empirical_corr_vs_hop_at_horizon(R, mask, h, Dhop, rng=rng)

    # ── Plots ──
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heatmaps: 6 samples, sensors sorted by Fiedler
    order = np.argsort(v_fiedler)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    sel = rng.choice(N_SAMPLES, size=6, replace=False)
    vmax = float(np.percentile(np.abs(samples[sel]), 99))
    for j, k in enumerate(sel):
        ax = axes[j // 3, j % 3]
        im = ax.imshow(samples[k][order], aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xlabel("horizon h")
        ax.set_ylabel("sensor (Fiedler-sorted)")
        ax.set_title(f"sample #{k}  Fiedler-contrib={fiedler_contrib[k]:.1%}")
    fig.suptitle(f"{dataset}: 6 joint samples (N×H), sensors sorted by Fiedler",
                 fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.8, label="residual (raw units)")
    fig.savefig(out_dir / "diag3_heatmaps.png", dpi=110)
    plt.close(fig)

    # Per-horizon corr-vs-hop overlay
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    hops = np.arange(MAX_HOP + 1)
    cm = plt.cm.viridis
    for h in range(H):
        ax.plot(hops[1:], corr_sample_per_h[h][1:],
                color=cm(h / max(H - 1, 1)), lw=1.0, alpha=0.8)
        ax.plot(hops[1:], corr_emp_per_h[h][1:],
                color=cm(h / max(H - 1, 1)), lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel("hop distance")
    ax.set_ylabel("mean pairwise correlation")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{dataset}: corr-vs-hop per horizon\nsolid = prior samples, dashed = empirical")
    fig.tight_layout()
    fig.savefig(out_dir / "diag3_corr_vs_hop.png", dpi=110)
    plt.close(fig)

    # Fiedler-contribution histogram
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(fiedler_contrib, bins=40, alpha=0.7, color="steelblue", edgecolor="k")
    ax.axvline(0.10, color="r", ls="--", lw=1, label="10% threshold")
    ax.axvline(fiedler_mean, color="k", ls=":", lw=1, label=f"mean={fiedler_mean:.1%}")
    ax.set_xlabel(f"Fiedler-mode contribution |v₂ᵀ R v_T⁽¹⁾|² / ‖R‖²_F")
    ax.set_ylabel("count")
    ax.set_title(f"{dataset}: Fiedler contribution across {N_SAMPLES} samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "diag3_fiedler_hist.png", dpi=110)
    plt.close(fig)

    return {
        "diag": "D3",
        "dataset": dataset,
        "n_samples": N_SAMPLES,
        "fiedler_mean": fiedler_mean,
        "fiedler_p95": fiedler_p95,
        "fiedler_max": fiedler_max,
        "pass_fiedler_under_10pct": pass_fiedler,
        "corr_sample_per_h": corr_sample_per_h.tolist(),
        "corr_emp_per_h": corr_emp_per_h.tolist(),
        "fiedler_eigval_L": float(prior["eigvals_L"][1]),
        "top_temporal_eigval": float(prior["mu_T"][idx_top_T]),
        "figures": {
            "heatmaps": str(out_dir / "diag3_heatmaps.png"),
            "corr_vs_hop": str(out_dir / "diag3_corr_vs_hop.png"),
            "fiedler_hist": str(out_dir / "diag3_fiedler_hist.png"),
        },
    }


# ───────────────────────────── CLI / runner ───────────────────────────────

def run_dataset(dataset: str, diags: list[int]) -> dict:
    print(f"\n=== {dataset} ===", flush=True)
    R, mask = load_residuals(dataset, "train")
    eigvals_L, V_S = load_eig(dataset)
    print(f"  residuals: {R.shape}, mask True rate: {mask.mean():.4f}", flush=True)

    prior = build_prior(R, mask, eigvals_L, V_S, gamma=GAMMA)
    print(f"  prior built: N={prior['N']}, H={prior['H']}, "
          f"valid temporal rows={prior['M_temporal_rows']}", flush=True)

    out = {"dataset": dataset, "gamma": GAMMA, "results": {}}

    if 1 in diags:
        r = diag1(R, mask, prior, dataset)
        out["results"]["D1"] = r
        print(f"  D1 (revised, smoothness-prior view):", flush=True)
        print(f"     spatial corr cos        : {r['spatial_corr_cosine']:.4f}  "
              f"(off-diag only: {r['spatial_corr_cosine_offdiag']:.4f})", flush=True)
        print(f"     temporal corr cos       : {r['temporal_corr_cosine']:.4f}", flush=True)
        print(f"     τ²_opt (variance gap)   : {r['tau2_opt']:.4f}  "
              f"η_raw={r['eta_raw']:.4f}  η_scaled={r['eta_scaled']:.4f}  "
              f"(H4 NKP floor={r['h4_nkp_floor']:.4f})", flush=True)

    if 2 in diags:
        r = diag2(R, mask, prior, dataset)
        out["results"]["D2"] = r
        print(f"  D2 (marginal variance per (i,h)):", flush=True)
        print(f"     ratio mean             : {r['mean_ratio']:.3f}  "
              f"(orig band [0.8, 1.2] pass={r['pass_mean_in_band']})", flush=True)
        print(f"     ratio std              : {r['std_ratio']:.3f}  "
              f"(orig <0.3 pass={r['pass_std_under_03']})", flush=True)
        pct = r['ratio_pct_1_5_25_50_75_95_99']
        print(f"     ratio percentiles      : "
              f"p1={pct[0]:.2f}  p5={pct[1]:.2f}  p25={pct[2]:.2f}  "
              f"p50={pct[3]:.2f}  p75={pct[4]:.2f}  p95={pct[5]:.2f}  p99={pct[6]:.2f}",
              flush=True)
        print(f"     per-horizon mean ratio : "
              f"{[round(x,2) for x in r['per_horizon_mean_ratio']]}", flush=True)

    if 3 in diags:
        fig_dir = ROOT / "results" / "prior_params" / f"diag3_{dataset}"
        r = diag3(R, mask, prior, dataset, fig_dir)
        out["results"]["D3"] = r
        print(f"  D3 (cluster sanity, {r['n_samples']} samples):", flush=True)
        print(f"     Fiedler-mode contrib   : mean={r['fiedler_mean']:.2%}  "
              f"p95={r['fiedler_p95']:.2%}  max={r['fiedler_max']:.2%}  "
              f"(pass <10% mean: {r['pass_fiedler_under_10pct']})", flush=True)
        # Compare prior corr-vs-hop at h=0 to empirical at h=0
        sm = np.array(r['corr_sample_per_h'])
        em = np.array(r['corr_emp_per_h'])
        print(f"     corr-vs-hop@h=0  sample (hops 1-4): "
              f"{[round(x,3) for x in sm[0, 1:5].tolist()]}", flush=True)
        print(f"     corr-vs-hop@h=0  empir  (hops 1-4): "
              f"{[round(x,3) for x in em[0, 1:5].tolist()]}", flush=True)
        print(f"     figures saved to       : {fig_dir}", flush=True)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    ap.add_argument("--diag", choices=["1", "2", "3", "all"], default="1")
    ap.add_argument("--out", default=None,
                    help="Optional JSON output path (default: results/prior_params/diag_{dataset}.json)")
    args = ap.parse_args()

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    diags = [1, 2, 3] if args.diag == "all" else [int(args.diag)]

    out_dir = ROOT / "results" / "prior_params"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for ds in datasets:
        res = run_dataset(ds, diags)
        all_results[ds] = res
        out_path = Path(args.out) if args.out else out_dir / f"diag_{ds}.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  → {out_path}", flush=True)

    print("\n=== SUMMARY ===")
    for ds, res in all_results.items():
        for diag_name, r in res["results"].items():
            if diag_name == "D1":
                print(f"  {ds:8s} D1  spatial-corr-cos={r['spatial_corr_cosine']:.4f}  "
                      f"off-diag={r['spatial_corr_cosine_offdiag']:.4f}  "
                      f"temporal-corr-cos={r['temporal_corr_cosine']:.4f}  "
                      f"τ²={r['tau2_opt']:.3f}  η_scaled={r['eta_scaled']:.3f}")


if __name__ == "__main__":
    main()
