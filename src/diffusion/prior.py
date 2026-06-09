"""builds the kronecker prior Σ = Σ_T ⊗ Σ_S used by the diffusion sde.

  Σ_S = (L + γ I)^{-1}, with a per-dataset γ (locked below). the laplacian
        eigendecomposition is read from the cached file (see src.utils.paths).
  Σ_T = D_H^{1/2} R_H D_H^{1/2}, fit from the training residuals:
        D_H[h] = (1.4826 · MAD_h)^2   (robust per-horizon variance)
        R_H    = corrcoef of the residuals   (empirical correlation)

residuals have shape (N, H) with vec(R)[h·N + i] = R[i, h], so np.kron(Σ_T, Σ_S)
matches numpy's vec. the precision Q = Σ^{-1} has eigenvalues θ[k,h] = (λ_k + γ)/μ_h
in the V_S ⊗ V_T basis.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.utils import paths

# locked per-dataset γ, fit once from the residual spectrum (see git history).
GAMMA_LOCKED: dict[str, float] = {
    "metrla": 0.15755546019980676,
    "pemsbay": 0.21271054104199358,
    "pems08f": 0.9239103283128737,
}

@dataclass(frozen=True)
class KroneckerPrior:
    """the fitted kronecker prior, cached after building from training residuals.

    fields:
      gamma            laplacian shift, locked per dataset
      V_S, lam_S       eigenvectors/eigenvalues of the normalised laplacian (N)
      V_T, mu_T        eigenvectors/eigenvalues of Σ_T (H)
      theta            (N, H) precision eigenvalues (λ_k + γ)/μ_h; the stationary
                       variance per mode is 1/theta
      delta_h          (H,) per-horizon MAD variance (diagonal of D_H)
      R_H              (H, H) empirical horizon correlation
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


def _load_residuals_train(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    paths.require_inputs([(
        paths.residual_npz(dataset, "train", 0),
        f"run scripts/extract_residuals.py --dataset {dataset} --seed 0 first",
    )])
    d = np.load(paths.residual_npz(dataset, "train", 0))
    return d["residuals"].astype(np.float64), d["mask"].astype(bool)


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
    """Return (delta_h, R_H, Sigma_T). MAD-based D_H, empirical R_H."""
    B, N, H = R.shape
    R_flat = R.reshape(B * N, H)
    M_flat = mask.reshape(B * N, H)
    keep = M_flat.all(axis=1)
    R_valid = R_flat[keep]

    mad_h = np.median(np.abs(R_valid - np.median(R_valid, axis=0)), axis=0)
    delta_h = (1.4826 * mad_h) ** 2
    R_H = np.corrcoef(R_valid.T)

    D_sqrt = np.sqrt(delta_h)
    Sigma_T = (D_sqrt[:, None] * R_H) * D_sqrt[None, :]
    return delta_h, R_H, Sigma_T


def _build_isotropic_prior(dataset: str) -> KroneckerPrior:
    """Scale-matched white-noise baseline: Σ = σ²·I, σ² = masked variance of the
    training residuals. identity eigenbasis (no spatial/temporal structure) with
    a constant per-mode precision θ = 1/σ², so the stationary variance is σ²
    everywhere. strips all structure but keeps the overall noise energy, so the
    comparison against the kronecker prior isolates the effect of structure.
    """
    R, mask = _load_residuals_train(dataset)
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


@lru_cache(maxsize=8)
def load_prior(dataset: str, prior: str = "kronecker") -> KroneckerPrior:
    """Fit and cache the diffusion noise prior for a dataset.

    prior="kronecker": structured Σ = Σ_T ⊗ Σ_S (graph + temporal correlations),
    fit from training residuals on first call.
    prior="isotropic": scale-matched white noise Σ = σ²·I (the ablation baseline).
    """
    if prior == "isotropic":
        return _build_isotropic_prior(dataset)
    if prior != "kronecker":
        raise ValueError(f"unknown prior {prior!r}; use 'kronecker' or 'isotropic'")
    if dataset not in GAMMA_LOCKED:
        raise KeyError(f"no locked γ for dataset {dataset!r}; valid: {sorted(GAMMA_LOCKED)}")
    gamma = GAMMA_LOCKED[dataset]

    lam_S, V_S = _load_laplacian_eig(dataset)
    R, mask = _load_residuals_train(dataset)
    delta_h, R_H, Sigma_T = _fit_sigma_T(R, mask)

    mu_T, V_T = np.linalg.eigh(Sigma_T)
    mu_T = np.clip(mu_T, a_min=1e-12, a_max=None)

    theta = (lam_S[:, None] + gamma) / mu_T[None, :]

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
