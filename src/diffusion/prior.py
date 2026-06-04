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
    d = np.load(paths.residual_npz(dataset, "train", 0))
    return d["residuals"].astype(np.float64), d["mask"].astype(bool)


def _load_laplacian_eig(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(paths.laplacian_eigs(dataset))
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


@lru_cache(maxsize=4)
def load_prior(dataset: str) -> KroneckerPrior:
    """Fit and cache the Kronecker prior for a dataset.

    First call fits Σ_T from training residuals; subsequent calls return
    the cached instance.
    """
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
