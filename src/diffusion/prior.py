"""builds the kronecker prior Σ = Σ_T ⊗ Σ_S used by the diffusion sde.

  Σ_T = D_H^{1/2} R_H D_H^{1/2}, fit from the training residuals:
        D_H[h] = mean(r_h^2)   (true per-horizon variance / second moment)
        R_H    = corrcoef of the residuals   (empirical horizon correlation)
        then the eigenvalues of Σ_T are capped so cond(Σ_T) <= KAPPA_CAP (the
        "γ for time": a deliberate regulariser that keeps the noise schedule
        trainable without throwing away the real horizon correlation).
  Σ_S   the spatial factor. by default flattened to I (SPATIAL_FLATTEN): GWN
        residuals are rough (Dirichlet energy ≈ 0.75), so the (L + γI)^{-1}
        smoothness prior is misspecified. set SPATIAL_FLATTEN=False to use the
        structured Σ_S = (L + γI)^{-1} (for the structured-spatial A/B).
  τ²    a thin scale calibration (= N / tr(Σ_S)) so flattening or structuring
        the spatial factor does not shift the overall variance. the *scale* is
        set by the true-variance D_H above; τ² only trims it (τ² = 1 when flat).

residuals have shape (N, H) with vec(R)[h·N + i] = R[i, h], so np.kron(Σ_T, Σ_S)
matches numpy's vec. the precision Q = Σ^{-1} has eigenvalues θ[k,h] = q_S[k]/(τ² μ_h)
in the V_S ⊗ V_T basis, where q_S[k] = 1 (flat) or λ_k + γ (structured).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.utils import paths

# locked per-dataset γ, fit once from the residual spectrum (see git history).
# only used when SPATIAL_FLATTEN is False (the structured-spatial arm).
GAMMA_LOCKED: dict[str, float] = {
    "metrla": 0.15755546019980676,
    "pemsbay": 0.21271054104199358,
    "pems08f": 0.9239103283128737,
}

# cap on cond(Σ_T) — the temporal twin of γ, locked 2026-06-10 from the hygiene
# analysis. κ=40 only bites pems-bay (cond 157→40); metr-la (37) and pems08f (17)
# keep their real horizon correlation untouched. NOT data-optimal shrinkage
# (ledoit-wolf returns δ≈1e-5 here — the stiffness is real structure, not noise);
# this is a deliberate distortion to keep the per-mode schedule trainable.
KAPPA_CAP = 40.0

# flatten the spatial factor (Σ_S → I). residuals of a frozen GWN are the rough
# leftover (Dirichlet energy ≈ 0.75), so a graph-smoothness prior is misspecified,
# and the small γ also drives c0 up. keep the temporal correlation, drop the
# spatial smoothness. set False for the structured-spatial A/B (on a repaired graph).
SPATIAL_FLATTEN = True

@dataclass(frozen=True)
class KroneckerPrior:
    """the fitted kronecker prior, cached after building from training residuals.

    fields:
      gamma            laplacian shift, locked per dataset
      V_S, lam_S       eigenvectors/eigenvalues of the normalised laplacian (N)
      V_T, mu_T        eigenvectors/eigenvalues of Σ_T (H)
      theta            (N, H) precision eigenvalues q_S[k]/(τ² μ_h); the stationary
                       variance per mode is 1/theta
      delta_h          (H,) per-horizon true variance (diagonal of D_H, pre-cap)
      R_H              (H, H) empirical horizon correlation (pre-cap)
    """

    dataset: str
    gamma: float
    V_S: np.ndarray
    lam_S: np.ndarray
    V_T: np.ndarray
    mu_T: np.ndarray
    theta: np.ndarray
    delta_h: np.ndarray
    R_H: np.ndarray

    @property
    def N(self) -> int:
        return self.V_S.shape[0]

    @property
    def H(self) -> int:
        return self.V_T.shape[0]

    @property
    def sigma2(self) -> np.ndarray:
        """Stationary variance in the joint eigenbasis. Shape (N, H)."""
        return 1.0 / self.theta


def _load_residuals_train(dataset: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Train residuals in unit-variance space, with the mask.

    divides by the single global train std (same scalar ResidualDataset uses), so
    the prior is fit in the same unit space the diffusion runs in. the de-norm
    back to physical units happens at the output (r_raw = r_unit * sigma_r).
    """
    paths.require_inputs([(
        paths.residual_npz(dataset, "train", seed),
        f"run scripts/extract_residuals.py --dataset {dataset} --seed {seed} first",
    )])
    d = np.load(paths.residual_npz(dataset, "train", seed))
    R = d["residuals"].astype(np.float64)
    mask = d["mask"].astype(bool)
    sigma_r = R[mask].std()                       # global train residual std
    return R / sigma_r, mask


def _load_laplacian_eig(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    # self-healing: if the eig cache isn't there yet, build it from the
    # adjacency and cache it, so a missing file never crashes a run.
    cache = paths.laplacian_eigs(dataset)
    if not cache.exists():
        from src.data.adjacency import (
            compute_normalized_laplacian,
            laplacian_eigendecomposition,
            load_adjacency,
        )
        L = compute_normalized_laplacian(load_adjacency(dataset))
        laplacian_eigendecomposition(L, dataset, cache=True)
    d = np.load(cache)
    return d["eigenvalues"].astype(np.float64), d["eigenvectors"].astype(np.float64)


def _fit_sigma_T(R: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (delta_h, R_H, Sigma_T).

    D_H is the true per-horizon variance (second moment). this replaced the old
    MAD estimator, which undercounts the variance 4-11x on these heavy-tailed
    residuals (excess kurtosis 19-40) and flattens the real horizon growth.
    R_H is the empirical horizon correlation.
    """
    B, N, H = R.shape
    R_flat = R.reshape(B * N, H)
    M_flat = mask.reshape(B * N, H)
    keep = M_flat.all(axis=1)
    R_valid = R_flat[keep]

    delta_h = (R_valid ** 2).mean(axis=0)        # true per-horizon variance
    R_H = np.corrcoef(R_valid.T)

    D_sqrt = np.sqrt(delta_h)
    Sigma_T = (D_sqrt[:, None] * R_H) * D_sqrt[None, :]
    return delta_h, R_H, Sigma_T


def _build_isotropic_prior(dataset: str, seed: int = 0) -> KroneckerPrior:
    """Scale-matched white-noise baseline: Σ = σ²·I, σ² = masked variance of the
    training residuals. identity eigenbasis (no spatial/temporal structure) with
    a constant per-mode precision θ = 1/σ², so the stationary variance is σ²
    everywhere. strips all structure but keeps the overall noise energy, so the
    comparison against the kronecker prior isolates the effect of structure.
    """
    R, mask = _load_residuals_train(dataset, seed)
    sigma2 = float(R[mask].var())
    _, N, H = R.shape
    return KroneckerPrior(
        dataset=dataset,
        gamma=0.0,                              # not meaningful for white noise
        V_S=np.eye(N),                          # identity basis -> no spatial structure
        lam_S=np.zeros(N),
        V_T=np.eye(H),                          # identity basis -> no temporal structure
        mu_T=np.full(H, sigma2),
        theta=np.full((N, H), 1.0 / sigma2),    # stationary variance 1/θ = σ² everywhere
        delta_h=np.full(H, sigma2),
        R_H=np.eye(H),
    )


@lru_cache(maxsize=32)
def load_prior(dataset: str, prior: str = "kronecker", seed: int = 0) -> KroneckerPrior:
    """Fit and cache the diffusion noise prior for a dataset.

    prior="kronecker": structured Σ = Σ_T ⊗ Σ_S (graph + temporal correlations),
    fit from training residuals on first call.
    prior="isotropic": scale-matched white noise Σ = σ²·I (the ablation baseline).
    """
    if prior == "isotropic":
        return _build_isotropic_prior(dataset, seed)
    if prior != "kronecker":
        raise ValueError(f"unknown prior {prior!r}; use 'kronecker' or 'isotropic'")
    if dataset not in GAMMA_LOCKED:
        raise KeyError(f"no locked γ for dataset {dataset!r}; valid: {sorted(GAMMA_LOCKED)}")

    R, mask = _load_residuals_train(dataset, seed)
    _, N, H = R.shape
    delta_h, R_H, Sigma_T = _fit_sigma_T(R, mask)

    mu_T, V_T = np.linalg.eigh(Sigma_T)
    mu_T = np.clip(mu_T, a_min=1e-12, a_max=None)
    # deliberate condition cap: floor the small eigenvalues so cond(Σ_T) <= KAPPA_CAP
    mu_T = np.maximum(mu_T, mu_T.max() / KAPPA_CAP)

    # spatial factor: flat (Σ_S = I) by default, else structured (L + γI)^{-1}
    if SPATIAL_FLATTEN:
        gamma = 0.0
        V_S = np.eye(N)
        lam_S = np.zeros(N)
        q_S = np.ones(N)                          # spatial precision = 1 (Σ_S = I)
    else:
        gamma = GAMMA_LOCKED[dataset]
        lam_S, V_S = _load_laplacian_eig(dataset)
        q_S = lam_S + gamma                       # spatial precision = L + γI

    # τ²: trace-calibrate so Σ_S's average eigenvalue doesn't shift the variance
    # (= 1 exactly when flat; the true-variance D_H already carries the scale).
    tau2 = N / (1.0 / q_S).sum()
    theta = q_S[:, None] / (tau2 * mu_T[None, :])

    return KroneckerPrior(
        dataset=dataset,
        gamma=gamma,
        V_S=V_S,
        lam_S=lam_S,
        V_T=V_T,
        mu_T=mu_T,
        theta=theta,
        delta_h=delta_h,
        R_H=R_H,
    )
