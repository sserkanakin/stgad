"""the two noise priors the diffusion can corrupt with.

  isotropic  Σ = σ²·I            white noise, σ² = the residual's own variance.
                                  the control: same total noise energy, no structure.

  kronecker  Σ = Σ_T ⊗ Σ_S        the structured prior.
      Σ_T = D_H^{1/2} R_H D_H^{1/2}   fit from the training residuals:
            D_H[h] = mean(r_h^2)   diagonal  = the noise variance at each horizon
            R_H    = corrcoef      off-diag  = the pearson correlation between horizons
            then a ridge ρ toward scaled identity on its eigenvalues, μ ← (1-ρ)μ + ρ·μ̄
      Σ_S ∝ (L + γI)^{-1}          the heat kernel of the repaired road graph PLUS a weak
                                    all-to-all edge (adjacency.add_global_edge), then
                                    normalised to UNIT DIAGONAL: per-sensor normalisation
                                    already makes every node's variance 1, so the prior
                                    must say the same and γ controls correlation only.
      (β, γ)                       chosen on the TRAIN residuals so the prior's spread
                                    on the NETWORK TOTAL equals the measured one
                                    (global_edge="match"), or given.

residuals have shape (N, H) with vec(R)[h·N + i] = R[i, h], so np.kron(Σ_T, Σ_S)
matches numpy's vec. the precision Q = Σ^{-1} has eigenvalues θ[k,h] = q_S[k]/(τ² μ_h)
in the V_S ⊗ V_T basis, q_S[k] = λ_k + γ. both arms are rescaled so θ_min = γ (GAD's
scale), with σ² absorbing the rescale so the stationary covariance is unchanged, and
both are then scaled to the SAME total variance -- the masked second moment of the
training residuals -- so the comparison isolates the effect of structure exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.utils import paths

# graph shift γ in Σ_S = (L + γI)^{-1}. inherited from GAD (rozada et al. 2025).
# override per call via load_prior(gamma=...).
GAMMA_DEFAULT: float = 0.8

# temporal ridge ρ: μ_T ← (1-ρ)·μ_T + ρ·μ̄. the temporal twin of γ — γ→∞ drives
# Σ_S→isotropic, ρ→1 drives Σ_T→isotropic — applied as a smooth, trace-preserving
# shift. tames the stiffest case (pems-bay cond 116→44) while barely touching the
# mild datasets. override per call via load_prior(shrink_t=...); 0 = faithful Σ_T.
SHRINK_T: float = 0.1


@dataclass(frozen=True)
class KroneckerPrior:
    """a fitted prior, cached after building from the training residuals.

    fields:
      gamma            laplacian shift (0 for the isotropic arm)
      V_S, lam_S       eigenvectors/eigenvalues of the normalised laplacian (N)
      V_T, mu_T        eigenvectors/eigenvalues of Σ_T (H)
      theta            (N, H) precision eigenvalues q_S[k]/(τ² μ_h); the stationary
                       variance per mode is sigma2_scale/theta
      delta_h          (H,) per-horizon variance (diagonal of D_H)
      R_H              (H, H) horizon correlation
    """

    dataset: str
    gamma: float
    V_S: np.ndarray
    lam_S: np.ndarray             # spatial precision eigenvalues MINUS γ, so q_S = lam_S + γ holds
    V_T: np.ndarray
    mu_T: np.ndarray
    theta: np.ndarray
    delta_h: np.ndarray
    R_H: np.ndarray
    sigma2_scale: float = 1.0     # σ²: scales the noise so Σ_stat = σ²/θ (see _normalize_scale)
    match_ratio: float = float("nan")   # prior-implied / measured sd of the network total (train), kron+match only
    global_edge: float = 0.0      # β: all-to-all edge weight relative to the mean road edge

    @property
    def N(self) -> int:
        return self.V_S.shape[0]

    @property
    def H(self) -> int:
        return self.V_T.shape[0]

    @property
    def sigma2(self) -> np.ndarray:
        """stationary variance in the joint eigenbasis (σ²/θ). shape (N, H)."""
        return self.sigma2_scale / self.theta


def _load_residuals_train(dataset: str, seed: int = 0, backbone: str = "gwn",
                          norm: str = "global") -> tuple[np.ndarray, np.ndarray]:
    """train residuals in unit-variance space, with the mask.

    divides by the SAME scaler ResidualDataset uses (global scalar std, or the
    per-node std for norm="pernode"), so the prior is fit in the exact space the
    diffusion runs in. the de-norm back to physical units happens at the output
    (r_raw = r_unit * sigma_r).
    """
    _how = (f"run scripts/extract_residuals.py --variant {dataset} --seed {seed} "
            f"--backbone {backbone} first")
    paths.require_inputs([(paths.residual_npz(dataset, "train", seed, backbone), _how)])
    from src.data.residual_dataset import train_residual_sigma
    d = np.load(paths.residual_npz(dataset, "train", seed, backbone))
    R = d["residuals"].astype(np.float64)
    mask = d["mask"].astype(bool)
    sigma_r = np.asarray(train_residual_sigma(dataset, seed, backbone, norm=norm))
    if sigma_r.ndim == 2:                                       # (1, H) per-horizon
        return R / sigma_r, mask
    return R / sigma_r.reshape(-1, 1), mask


def _load_laplacian_eig(dataset: str, repaired: bool,
                        global_edge: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(λ, V) of the normalised laplacian, ascending. self-healing: rebuild if the
    eig cache is missing or corrupt, and fall back to an in-memory recompute if the
    cache location is unwritable, so a bad cache never crashes a run. with a global
    edge the graph is run-specific, so it is computed fresh (an eigh of N<=325)."""
    from src.data.adjacency import compute_normalized_laplacian, load_adjacency
    if global_edge:
        L = compute_normalized_laplacian(load_adjacency(dataset, repair=repaired, global_edge=global_edge))
        lam, V = np.linalg.eigh(L)
        o = np.argsort(lam)
        return lam[o].astype(np.float64), V[:, o].astype(np.float64)
    cache = paths.laplacian_eigs(dataset, repaired)
    if cache.exists():
        try:
            d = np.load(cache)
            return d["eigenvalues"].astype(np.float64), d["eigenvectors"].astype(np.float64)
        except Exception as e:                       # BadZipFile, EOFError, OSError, ...
            import warnings
            warnings.warn(f"corrupt laplacian cache {cache} ({e}); recomputing from adjacency")
    L = compute_normalized_laplacian(load_adjacency(dataset, repair=repaired))
    lam, V = np.linalg.eigh(L)
    o = np.argsort(lam)
    lam, V = lam[o].astype(np.float64), V[:, o].astype(np.float64)
    try:                                             # best-effort re-cache (ok if unwritable)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, eigenvalues=lam, eigenvectors=V)
    except Exception:
        pass
    return lam, V


def _fit_sigma_T(R: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(delta_h, R_H, Sigma_T) from the residuals.

    D_H is the per-horizon second moment over ALL observed entries (not a MAD: that
    undercounts the variance 4-11x on these heavy-tailed residuals and flattens the
    real horizon growth). it is deliberately NOT restricted to fully observed rows:
    on metr-la sensors drop out during the hard periods, and the fully observed rows
    carry only 52-63% of the variance -- a prior fitted there is fitted on the easy
    periods. R_H, the horizon correlation, is a SHAPE and is fitted on the fully
    observed rows where it is well defined.
    """
    B, N, H = R.shape
    R_flat = R.reshape(B * N, H)
    M_flat = mask.reshape(B * N, H)
    Rz = np.where(M_flat, R_flat, 0.0)
    delta_h = (Rz ** 2).sum(axis=0) / np.maximum(M_flat.sum(axis=0), 1)   # per-horizon variance
    keep = M_flat.all(axis=1)
    R_H = np.corrcoef(R_flat[keep].T)

    D_sqrt = np.sqrt(delta_h)
    Sigma_T = (D_sqrt[:, None] * R_H) * D_sqrt[None, :]
    return delta_h, R_H, Sigma_T


def _normalize_scale(theta: np.ndarray) -> tuple[np.ndarray, float]:
    """put the slowest mode at θ_min = γ (GAD's smooth-mode precision), so the
    schedule's c_0 ≈ 5/γ mixes it on every dataset instead of a per-dataset 30-80.
    σ² absorbs the rescale, so the stationary covariance Σ_stat = σ²/θ is UNCHANGED —
    an exact reparameterization that only moves scale from c̄ into θ.
    returns (theta_scaled, sigma2_scale)."""
    s = GAMMA_DEFAULT / float(theta.min())
    return theta * s, s


def _build_isotropic_prior(dataset: str, seed: int, backbone: str, norm: str) -> KroneckerPrior:
    """scale-matched white noise: Σ = σ²·I, σ² = masked variance of the training
    residuals. identity eigenbasis (no spatial/temporal structure) with a constant
    per-mode precision, so the stationary variance is σ² everywhere. strips all
    structure but keeps the overall noise energy, so the comparison against the
    kronecker prior isolates the effect of structure."""
    R, mask = _load_residuals_train(dataset, seed, backbone, norm)
    sigma2 = float((R[mask] ** 2).mean())        # the masked second moment: the SAME scale the kron arm is matched to
    _, N, H = R.shape
    theta, sigma2_scale = _normalize_scale(np.full((N, H), 1.0 / sigma2))
    return KroneckerPrior(
        dataset=dataset,
        gamma=0.0,
        V_S=np.eye(N),
        lam_S=np.zeros(N),
        V_T=np.eye(H),
        mu_T=np.full(H, sigma2),
        theta=theta,
        delta_h=np.full(H, sigma2),
        R_H=np.eye(H),
        sigma2_scale=sigma2_scale,
    )


def _spatial_factor(dataset: str, repair: bool, global_edge: float, gamma: float,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mu_S, V_S, lam) : eigenvalues/vectors of the UNIT-DIAGONAL Σ_S ∝ (L+γI)^{-1}, and the
    laplacian eigenvalues the kernel was built from."""
    lam, V = _load_laplacian_eig(dataset, repaired=repair, global_edge=global_edge)
    Sig = (V * (1.0 / (lam + gamma))) @ V.T
    d = np.sqrt(np.diag(Sig))
    Sig = Sig / np.outer(d, d)                     # unit diagonal: γ controls correlation only
    mu_S, V_S = np.linalg.eigh(Sig)
    mu_S = np.clip(mu_S, 1e-12, None)
    return mu_S, V_S, lam


def _implied_total_sd(V_S: np.ndarray, sigma2: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """sd the prior implies for the NETWORK TOTAL error at each horizon, in raw units:
    Var(Σ_i σ_i r_ih) = σᵀ C_h σ with C_h = V_S diag(sigma2[:,h]) V_Sᵀ."""
    return np.array([np.sqrt(sig @ (V_S * sigma2[:, h]) @ V_S.T @ sig) for h in range(sigma2.shape[1])])


def match_global_edge(dataset: str, seed: int, backbone: str, norm: str,
                      repair: bool, mu_T: np.ndarray, var_data: float,
                      betas=(0.003, 0.01, 0.03, 0.1, 0.3),
                      gammas=None) -> tuple[float, float, float]:
    """choose (β, γ) so the prior puts the RIGHT spread on the network total.

    a road graph alone puts 0.5-1.1% of the prior's variance on the 'all sensors
    together' direction where the residual has 3-21%, and no γ fixes that; a global
    edge does. likelihood over-shoots here (it is nearly blind to one mode out of N),
    so the pair is chosen by MATCHING, on the TRAIN residuals only: the ratio of the
    prior-implied sd of the total to the measured one, closest to 1. returns
    (β, γ, ratio)."""
    from src.data.residual_dataset import train_residual_sigma
    gammas = np.geomspace(1e-3, 3.0, 40) if gammas is None else gammas
    d = np.load(paths.residual_npz(dataset, "train", seed, backbone))
    R = d["residuals"].astype(np.float64)
    M = d["mask"].astype(bool)
    sig = np.asarray(train_residual_sigma(dataset, seed, backbone, norm=norm))
    sig = np.broadcast_to(sig.reshape(-1), (R.shape[1],)).astype(np.float64)
    tot = np.where(M, R, 0.0).sum(axis=1)                    # raw total per (window, horizon)
    ok = M.reshape(len(M), -1).mean(1) > 0.5
    truth = tot[ok].std(axis=0)
    N, H = R.shape[1], R.shape[2]
    best = None
    for beta in betas:
        for g in gammas:
            mu_S, V_S, _ = _spatial_factor(dataset, repair, beta, float(g))
            theta = (1.0 / mu_S)[:, None] / mu_T[None, :]
            theta, s2 = _normalize_scale(theta)
            s2 *= var_data / float((s2 / theta).mean())
            r = float((_implied_total_sd(V_S, s2 / theta, sig) / truth).mean())
            if best is None or abs(r - 1.0) < abs(best[2] - 1.0):
                best = (float(beta), float(g), r)
    return best


@lru_cache(maxsize=32)
def load_prior(
    dataset: str,
    prior: str = "kronecker",
    seed: int = 0,
    gamma: float | None = None,
    shrink_t: float | None = None,
    backbone: str = "gwn",
    repair: bool | None = None,
    norm: str = "global",
    global_edge: float | str | None = None,
) -> KroneckerPrior:
    """fit and cache the noise prior for one (dataset, backbone, seed) cell.

    prior="kronecker": Σ = Σ_T ⊗ Σ_S, Σ_T fit from the residuals, Σ_S the road-graph
      heat kernel (L+γI)^{-1}.
    prior="isotropic": scale-matched white noise Σ = σ²·I (the control).
    gamma:    spatial shift γ. None = GAMMA_DEFAULT (ignored when global_edge="match").
    shrink_t: temporal ridge ρ. None = SHRINK_T. 0 = faithful empirical Σ_T.
    global_edge: β for the all-to-all edge (adjacency.add_global_edge); None/0 = road
              graph only; "match" = choose (β, γ) on the train residuals so the prior's
              spread on the network total is right (match_global_edge).
    repair:   use the component-repaired (single connected) graph. None = follow
              adjacency.REPAIR_GRAPH.
    norm:     residual scaler the prior is fit under — "global" or "pernode"; must
              match the dataset's normalisation.
    """
    if prior == "isotropic":
        return _build_isotropic_prior(dataset, seed, backbone, norm)
    if prior != "kronecker":
        raise ValueError(f"unknown prior {prior!r}; use 'kronecker' or 'isotropic'")

    R, mask = _load_residuals_train(dataset, seed, backbone, norm)
    _, N, H = R.shape
    var_data = float((R[mask] ** 2).mean())      # the scale both arms are matched to

    # temporal factor: horizon variance profile and horizon correlation, then ridge
    delta_h, R_H, Sigma_T = _fit_sigma_T(R, mask)
    mu_T, V_T = np.linalg.eigh(Sigma_T)
    mu_T = np.clip(mu_T, a_min=1e-12, a_max=None)
    rho = SHRINK_T if shrink_t is None else shrink_t
    if rho:
        mu_T = (1.0 - rho) * mu_T + rho * mu_T.mean()

    # spatial factor: unit-diagonal heat kernel of the road graph (+ global edge)
    if repair is None:
        from src.data.adjacency import REPAIR_GRAPH
        repair = REPAIR_GRAPH
    match_ratio = float("nan")
    if global_edge == "match":
        beta, gamma, match_ratio = match_global_edge(dataset, seed, backbone, norm, repair, mu_T, var_data)
    else:
        beta = float(global_edge or 0.0)
        gamma = GAMMA_DEFAULT if gamma is None else gamma
    mu_S, V_S, _ = _spatial_factor(dataset, repair, beta, gamma)
    q_S = 1.0 / mu_S                          # spatial precision eigenvalues
    theta = q_S[:, None] / mu_T[None, :]      # (unit diagonal => τ² = 1 exactly)

    theta, sigma2_scale = _normalize_scale(theta)
    sigma2_scale *= var_data / float((sigma2_scale / theta).mean())   # exact variance match
    return KroneckerPrior(
        dataset=dataset,
        gamma=gamma,
        V_S=V_S,
        lam_S=q_S - gamma,
        V_T=V_T,
        mu_T=mu_T,
        theta=theta,
        delta_h=delta_h,
        R_H=R_H,
        sigma2_scale=sigma2_scale,
        global_edge=beta,
        match_ratio=match_ratio,
    )
