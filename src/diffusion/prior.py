"""builds the kronecker prior Σ = Σ_T ⊗ Σ_S used by the diffusion sde.

  Σ_T = D_H^{1/2} R_H D_H^{1/2}, fit from the training residuals:
        D_H[h] = mean(r_h^2)   (true per-horizon variance / second moment)
        R_H    = corrcoef of the residuals   (empirical horizon correlation)
        then Σ_T is ridge-shifted toward scaled-identity, μ ← (1-ρ)μ + ρ·μ̄
        (SHRINK_T, the "γ for time"): a smooth, trace-preserving regulariser —
        the exact temporal twin of the spatial γ — that keeps the noise schedule
        trainable without throwing away the real horizon correlation.
  Σ_S   the spatial factor: structured Σ_S = (L + γI)^{-1} with fixed γ = 0.8
        (GAD), or flat Σ_S = I (temporal-only ablation, spatial="flat"). γ both
        well-conditions the operator and lifts the Laplacian's zero modes, so
        disconnected graphs (e.g. pems04's 12 components) need no graph surgery.
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

# fixed graph-shift γ in Σ_S = (L + γI)^{-1} for the structured-spatial arm.
# NOT data-fit: we follow GAD (Rozada et al. 2025), which fixes γ = 0.8 across
# datasets. this keeps the prior a principled, paper-aligned hyperparameter
# (cf. Vimal's mas_gamma, also a fixed 1e-3/1e-2 shift) rather than a bespoke
# per-dataset value. override per call via load_prior(gamma=...) for sweeps.
GAMMA_DEFAULT: float = 0.8

# temporal twin of the spatial γ: shrink Σ_T toward scaled-identity by a fixed
# fraction ρ,  μ_T ← (1-ρ)·μ_T + ρ·μ̄  (μ̄ = mean eigenvalue). this is the exact
# analogue of γ — γ→∞ drives Σ_S→isotropic, ρ→1 drives Σ_T→isotropic — applied as
# a smooth, trace-preserving shift instead of the old hard cond-number clamp.
# ρ=0.1 tames the stiffest case (pems-bay cond 116→44, ~matching the old κ=40)
# while barely touching the already-mild datasets (metr-la 23→16, pems08f 8→6).
# override per call via load_prior(shrink_t=...). keeps the real horizon
# correlation (not a path-graph assumption); the c̄-schedule handles any residual stiffness.
SHRINK_T: float = 0.1

# default spatial factor for the "kronecker" prior:
#   "structured" -> Σ_S = (L + γI)^{-1} on the repaired (connected) graph: the
#                   spatio-temporal prior — the thesis contribution.
#   "flat"       -> Σ_S = I (temporal-only): keeps the horizon correlation, drops
#                   the graph smoothness. residuals of a frozen GWN are spatially
#                   ~white (Dirichlet energy ≈ 0.80), so flat is a fair ablation.
# the choice is passed explicitly through load_prior and recorded in config.json
# (do NOT infer it from a module constant — the run must say what it used).
DEFAULT_SPATIAL = "structured"

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
    sigma2_scale: float = 1.0     # σ²: scales the noise so Σ_stat = σ²/θ (see _normalize_scale)

    @property
    def N(self) -> int:
        return self.V_S.shape[0]

    @property
    def H(self) -> int:
        return self.V_T.shape[0]

    @property
    def sigma2(self) -> np.ndarray:
        """Stationary variance in the joint eigenbasis (σ²/θ). Shape (N, H)."""
        return self.sigma2_scale / self.theta


def _load_residuals_train(dataset: str, seed: int = 0, backbone: str = "gwn",
                          norm: str = "global") -> tuple[np.ndarray, np.ndarray]:
    """Train residuals in unit-variance space, with the mask.

    divides by the SAME scaler ResidualDataset uses (global scalar std, or the
    per-node std for norm="pernode"), so the prior is fit in the exact space the
    diffusion runs in. the de-norm back to physical units happens at the output
    (r_raw = r_unit * sigma_r).
    """
    paths.require_inputs([(
        paths.residual_npz(dataset, "train", seed, backbone),
        f"run scripts/gwn_extract_residuals.py --variant {dataset} --seed {seed} "
        f"--backbone {backbone} first",
    )])
    from src.data.residual_dataset import train_residual_sigma
    d = np.load(paths.residual_npz(dataset, "train", seed, backbone))
    R = d["residuals"].astype(np.float64)
    mask = d["mask"].astype(bool)
    sigma_r = np.asarray(train_residual_sigma(dataset, seed, backbone, norm=norm))
    return R / sigma_r.reshape(-1, 1), mask


def _load_laplacian_eig(dataset: str, repaired: bool) -> tuple[np.ndarray, np.ndarray]:
    # self-healing: if the eig cache isn't there yet, build it from the
    # adjacency (repaired to one connected component by default) and cache it,
    # so a missing file never crashes a run.
    cache = paths.laplacian_eigs(dataset, repaired)
    if not cache.exists():
        from src.data.adjacency import (
            compute_normalized_laplacian,
            laplacian_eigendecomposition,
            load_adjacency,
        )
        L = compute_normalized_laplacian(load_adjacency(dataset, repair=repaired))
        laplacian_eigendecomposition(L, dataset, cache=True, repaired=repaired)
    d = np.load(cache)
    return d["eigenvalues"].astype(np.float64), d["eigenvectors"].astype(np.float64)


def _temporal_laplacian_eig(H: int, gamma_t: float) -> tuple[np.ndarray, np.ndarray]:
    """temporal factor from a PATH-graph (chain) normalized Laplacian on the H horizons:
    Σ_T = (L_path + γ_t I)^{-1} — the temporal twin of the spatial (L_S + γI)^{-1}, so the
    prior is a clean spatio-temporal PRODUCT GRAPH (both sides Laplacian, Kronecker/
    multiplicative). a mild γ_t keeps the temporal spread bounded (ratio (2+γ_t)/γ_t),
    unlike the empirical Σ_T whose one giant common mode makes the prior very stiff.
    returns (mu_T, V_T) with mu_T = 1/(ω+γ_t) the covariance eigenvalues."""
    from src.data.adjacency import compute_normalized_laplacian
    A = np.zeros((H, H))
    idx = np.arange(H - 1)
    A[idx, idx + 1] = 1.0
    A[idx + 1, idx] = 1.0
    omega, V_T = np.linalg.eigh(compute_normalized_laplacian(A))
    mu_T = 1.0 / (np.clip(omega, 0.0, None) + gamma_t)
    return mu_T, V_T


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


def _normalize_scale(theta: np.ndarray) -> tuple[np.ndarray, float]:
    """rescale the precision to GAD's scale: put the slowest mode at θ_min = γ (GAD's
    smooth-mode precision λ_min+γ = γ). then the schedule's c_0 ≈ 5/γ ≈ 6 for EVERY
    dataset (GAD fixes S≈7), instead of a per-dataset 30–80. σ² absorbs the rescale,
    so the stationary covariance Σ_stat = σ²/θ is UNCHANGED — an exact reparameterization
    that only moves the scale from c̄ into θ. returns (theta_scaled, sigma2_scale)."""
    s = GAMMA_DEFAULT / float(theta.min())
    return theta * s, s


def _build_isotropic_prior(dataset: str, seed: int = 0, backbone: str = "gwn",
                           norm: str = "global") -> KroneckerPrior:
    """Scale-matched white-noise baseline: Σ = σ²·I, σ² = masked variance of the
    training residuals. identity eigenbasis (no spatial/temporal structure) with
    a constant per-mode precision θ = 1/σ², so the stationary variance is σ²
    everywhere. strips all structure but keeps the overall noise energy, so the
    comparison against the kronecker prior isolates the effect of structure.
    """
    R, mask = _load_residuals_train(dataset, seed, backbone, norm)
    sigma2 = float(R[mask].var())
    _, N, H = R.shape
    theta, sigma2_scale = _normalize_scale(np.full((N, H), 1.0 / sigma2))
    return KroneckerPrior(
        dataset=dataset,
        gamma=0.0,                              # not meaningful for white noise
        V_S=np.eye(N),                          # identity basis -> no spatial structure
        lam_S=np.zeros(N),
        V_T=np.eye(H),                          # identity basis -> no temporal structure
        mu_T=np.full(H, sigma2),
        theta=theta,                            # normalized; Σ_stat = σ²·scale/θ = σ²
        delta_h=np.full(H, sigma2),
        R_H=np.eye(H),
        sigma2_scale=sigma2_scale,
    )


@lru_cache(maxsize=32)
def load_prior(
    dataset: str,
    prior: str = "kronecker",
    seed: int = 0,
    spatial: str = DEFAULT_SPATIAL,
    stiffness_cap: float | None = None,
    gamma: float | None = None,
    shrink_t: float | None = None,
    temporal: str = "empirical",
    gamma_t: float = 0.6,
    backbone: str = "gwn",
    repair: bool | None = None,
    norm: str = "global",
) -> KroneckerPrior:
    """Fit and cache the diffusion noise prior for a dataset.

    prior="kronecker": structured Σ = Σ_T ⊗ Σ_S, fit from training residuals.
      spatial="structured" -> Σ_S = (L + γI)^{-1} (spatio-temporal, the contribution).
      spatial="flat"       -> Σ_S = I (temporal-only ablation).
    prior="isotropic": scale-matched white noise Σ = σ²·I (the baseline).
    stiffness_cap: if set, floor θ_min so θ_max/θ_min <= cap (a joint-spectrum
      twin of the ridge shift; the trainability fallback for stiff structured priors).
    gamma: override the fixed spatial shift γ (structured only). None = use
      GAMMA_DEFAULT (0.8, GAD). exposed for the conditioning trade-off sweep (a larger
      γ tames κ(Q_S) at the cost of a less smoothness-committed prior).
    shrink_t: override the temporal ridge fraction ρ (μ_T ← (1-ρ)μ_T + ρ·μ̄, the
      temporal twin of γ). None = use SHRINK_T (0.1). 0 = faithful empirical Σ_T.
    repair: use the component-repaired (single connected) graph for the laplacian.
      None = follow adjacency.REPAIR_GRAPH (the global default, True).
    norm: residual scaler the prior is fit under — "global" (one scalar std) or
      "pernode" (per-sensor std; must match the dataset's normalization).
    """
    if prior == "isotropic":
        return _build_isotropic_prior(dataset, seed, backbone, norm)
    if prior != "kronecker":
        raise ValueError(f"unknown prior {prior!r}; use 'kronecker' or 'isotropic'")
    if spatial not in ("structured", "flat"):
        raise ValueError(f"unknown spatial {spatial!r}; use 'structured' or 'flat'")

    R, mask = _load_residuals_train(dataset, seed, backbone, norm)
    _, N, H = R.shape

    if temporal == "laplacian":
        # clean spatio-temporal PRODUCT GRAPH: temporal path-graph Laplacian (γ_t) in place
        # of the empirical Σ_T. mild γ_t keeps the temporal spread bounded (no stiff common
        # mode), so no shrink ridge is needed. per-horizon stats kept only for the record.
        mu_T, V_T = _temporal_laplacian_eig(H, gamma_t)
        Rf = R.reshape(-1, H); keep = mask.reshape(-1, H).all(1)
        delta_h = (Rf[keep] ** 2).mean(0)
        R_H = np.corrcoef(Rf[keep].T)
    elif temporal == "empirical":
        delta_h, R_H, Sigma_T = _fit_sigma_T(R, mask)
        mu_T, V_T = np.linalg.eigh(Sigma_T)
        mu_T = np.clip(mu_T, a_min=1e-12, a_max=None)
        # ridge shift toward scaled-identity (temporal twin of γ): μ ← (1-ρ)μ + ρ·μ̄.
        # additive + trace-preserving, not a hard clamp; ρ=0 leaves Σ_T faithful.
        rho = SHRINK_T if shrink_t is None else shrink_t
        if rho:
            mu_T = (1.0 - rho) * mu_T + rho * mu_T.mean()
    else:
        raise ValueError(f"unknown temporal {temporal!r}; use 'empirical' or 'laplacian'")

    # spatial factor: structured (L + γI)^{-1}, or flat (Σ_S = I)
    if spatial == "flat":
        gamma = 0.0
        V_S = np.eye(N)
        lam_S = np.zeros(N)
        q_S = np.ones(N)                          # spatial precision = 1 (Σ_S = I)
    else:
        gamma = GAMMA_DEFAULT if gamma is None else gamma
        if repair is None:
            from src.data.adjacency import REPAIR_GRAPH
            repair = REPAIR_GRAPH
        lam_S, V_S = _load_laplacian_eig(dataset, repaired=repair)
        q_S = lam_S + gamma                       # spatial precision = L + γI

    # τ²: trace-calibrate so Σ_S's average eigenvalue doesn't shift the variance
    # (= 1 exactly when flat; the true-variance D_H already carries the scale).
    tau2 = N / (1.0 / q_S).sum()
    theta = q_S[:, None] / (tau2 * mu_T[None, :])

    # optional joint-stiffness cap: floor θ so θ_max/θ_min <= stiffness_cap. off by
    # default (faithful prior); enabled only if the pilot shows training instability.
    if stiffness_cap is not None:
        theta = np.maximum(theta, theta.max() / stiffness_cap)

    # GAD-scale normalization: θ_min → γ so c_0 ≈ 5/γ ≈ 6 (σ² preserves Σ_stat).
    theta, sigma2_scale = _normalize_scale(theta)
    if temporal == "laplacian":
        # the path-graph μ_T carries no data scale (unlike empirical Σ_T's D_H); match the
        # overall stationary variance to the data (z-scored → ~1) so draws are representative.
        data_var = float(R[mask].var())
        cur = float((sigma2_scale / theta).mean())
        sigma2_scale *= data_var / max(cur, 1e-12)

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
        sigma2_scale=sigma2_scale,
    )


def build_prior_from_R(
    dataset: str,
    R_raw: np.ndarray,
    mask: np.ndarray,
    spatial: str = "structured",
    gamma: float = 0.8,
    stiffness_cap: float | None = None,
    repair: bool | None = None,
) -> tuple["KroneckerPrior", float]:
    """Build a prior directly from a RAW residual field (B,N,H) + mask, in unit-variance
    space. Used by the Tier-3/4 analysis so it works for ANY rung (incl. the climatology
    'signal') and ANY of the 6 datasets with an explicit γ (e.g. GAD's 0.8).

    spatial: 'structured' Σ_S=(L+γI)^-1 · 'flat' Σ_S=I · 'iso' white Σ=σ²I.
    Returns (prior, sigma_r) where sigma_r is the masked train residual std used to
    normalise into unit space.
    """
    sigma_r = float(R_raw[mask].std())
    R = R_raw.astype(np.float64) / sigma_r
    _, N, H = R.shape

    if spatial == "iso":
        sigma2 = float(R[mask].var())
        theta, sigma2_scale = _normalize_scale(np.full((N, H), 1.0 / sigma2))
        return KroneckerPrior(
            dataset=dataset, gamma=0.0, V_S=np.eye(N), lam_S=np.zeros(N),
            V_T=np.eye(H), mu_T=np.full(H, sigma2), theta=theta,
            delta_h=np.full(H, sigma2), R_H=np.eye(H), sigma2_scale=sigma2_scale,
        ), sigma_r

    delta_h, R_H, Sigma_T = _fit_sigma_T(R, mask)
    mu_T, V_T = np.linalg.eigh(Sigma_T)
    mu_T = np.clip(mu_T, a_min=1e-12, a_max=None)
    if SHRINK_T:                              # ridge shift toward scaled-identity (see load_prior)
        mu_T = (1.0 - SHRINK_T) * mu_T + SHRINK_T * mu_T.mean()

    if spatial == "flat":
        gamma, V_S, lam_S, q_S = 0.0, np.eye(N), np.zeros(N), np.ones(N)
    else:
        if repair is None:
            from src.data.adjacency import REPAIR_GRAPH
            repair = REPAIR_GRAPH
        lam_S, V_S = _load_laplacian_eig(dataset, repaired=repair)
        q_S = lam_S + gamma
    tau2 = N / (1.0 / q_S).sum()
    theta = q_S[:, None] / (tau2 * mu_T[None, :])
    if stiffness_cap is not None:
        theta = np.maximum(theta, theta.max() / stiffness_cap)
    theta, sigma2_scale = _normalize_scale(theta)
    return KroneckerPrior(dataset=dataset, gamma=gamma, V_S=V_S, lam_S=lam_S,
                          V_T=V_T, mu_T=mu_T, theta=theta, delta_h=delta_h, R_H=R_H,
                          sigma2_scale=sigma2_scale), sigma_r
