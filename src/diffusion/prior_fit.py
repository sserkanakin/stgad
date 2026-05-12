# Fit Kronecker prior parameters (delta_h, alpha/rho1/rho2, beta) from residuals.
# All functions are pure (no I/O).  The entry point script/fit_prior.py handles
# data loading, YAML output, and figures.
#
# Prior structure
# ---------------
#   R[n, h, i] ~ N(0, Σ_prior)
#   Σ_prior = Q_H ⊗ Σ_S   (Kronecker product, H-outer × N-inner)
#
#   Q_H[h1, h2] = sqrt(δ_{h1} · δ_{h2}) · C_H[h1, h2]
#   C_H[h1, h2] = α · ρ₁^|h1-h2| + (1-α) · ρ₂^|h1-h2|    (two-exp correlation)
#   Σ_S         = (β·L + ε_s·I)^{-1}                        (GMRF spatial cov)
#
# Standardised vs raw residuals
# -----------------------------
# By default `fit_prior` operates on **per-horizon standardised** residuals
# r̃[n,h,i] = r[n,h,i] / target_scale[h], with target_scale[h] = sqrt(Var(r_h)).
# This decouples scale (per-horizon σ_h) from structure (correlations), so the
# fitted Σ_prior has eigenvalues O(1)–O(10) instead of O(10⁴) and the SDE noise
# schedule β_max=O(50) reaches stationary at τ=1 (see exp-004 SUMMARY).
# The returned dict contains `target_scale` for de-standardisation at sample time.
# Pass `standardize=False` to recover the raw-mph² behaviour.
#
# Sampling convention (row-major flatten of R):
#   vec(R) = [R[0,:], R[1,:], ..., R[H-1,:]]   has covariance Q_H ⊗ Σ_S
#   np.kron(Q_H, Σ_S)[h*N+i, h'*N+i'] = Q_H[h,h'] · Σ_S[i,i']   ✓
#
# Forward SDE (VP):
#   x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε_struct,  ε_struct ~ N(0, Σ_prior)
#   ᾱ_t = exp(−σ² · t),  t ∈ [0, 1]
#
# σ from eigenvalue analysis:
#   Worst-case SNR at t=1: ᾱ_T·λ_max / ((1−ᾱ_T)·λ_min) ≤ ε_target
#   → σ = sqrt(log(cond(Σ_prior) / ε_target))
import numpy as np
from scipy.optimize import minimize, minimize_scalar


# ── Per-horizon variance ──────────────────────────────────────────────────────

def fit_delta_h(r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Empirical per-horizon variance using valid readings only.

    Args:
        r:    (N_win, 12, N) residuals in mph.
        mask: (N_win, 12, N) bool — True where reading is valid.

    Returns:
        (12,) float64 array of per-horizon variances [mph²].
    """
    delta = np.empty(12, dtype=np.float64)
    for h in range(12):
        vals = r[:, h, :][mask[:, h, :]]
        delta[h] = float(vals.var())
    return delta


# ── Temporal covariance helpers ───────────────────────────────────────────────

def empirical_temporal_cov(r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """12×12 temporal covariance Q averaged over (window, sensor) pairs.

    Args:
        r:    (N_win, 12, N)
        mask: (N_win, 12, N) bool

    Returns:
        (12, 12) float64 covariance matrix.
    """
    r_f = r.transpose(0, 2, 1).reshape(-1, 12).astype(np.float64)  # (M, 12)
    m_f = mask.transpose(0, 2, 1).reshape(-1, 12)
    r_f = r_f * m_f
    num = r_f.T @ r_f                               # (12, 12)
    den = (m_f.astype(np.float64)).T @ m_f.astype(np.float64)
    return num / np.maximum(den, 1.0)


def empirical_temporal_corr(Q: np.ndarray) -> np.ndarray:
    """Normalise Q to a correlation matrix."""
    d_inv = 1.0 / np.sqrt(np.diag(Q).clip(1e-12))
    return d_inv[:, None] * Q * d_inv[None, :]


def lag_averaged_acf(Q_corr: np.ndarray) -> np.ndarray:
    """Average Q_corr[h, h+k] over all h for each lag k=1..11.

    The two-exponential kernel is stationary (depends only on |h1-h2|), but
    the empirical correlation is non-stationary (lag-1 corr increases with
    horizon).  Fitting to the lag-averaged ACF is the correct approach.

    Returns:
        (11,) float64 array — acf[k-1] = mean correlation at lag k.
    """
    H = Q_corr.shape[0]
    return np.array(
        [float(np.mean([Q_corr[h, h + k] for h in range(H - k)]))
         for k in range(1, H)],
        dtype=np.float64,
    )


# ── Two-exponential temporal kernel ──────────────────────────────────────────

def fit_temporal_kernel(
    Q_corr: np.ndarray,
    n_restarts: int = 12,
) -> tuple[float, float, float]:
    """Fit α, ρ₁, ρ₂ for the two-exponential correlation kernel.

    Fits C[h1, h2] = α·ρ₁^|h1-h2| + (1-α)·ρ₂^|h1-h2| to the lag-averaged
    ACF by nonlinear least-squares.  Multiple restarts guard against local
    minima and the degenerate ρ₁=ρ₂ solution.  Returns parameters with
    ρ₁ ≥ ρ₂ by convention (ρ₁ = slow component).

    Args:
        Q_corr:     (12, 12) correlation matrix from empirical_temporal_corr.
        n_restarts: Number of random initialisations (default 12; more guards
                    against the degenerate ρ₁≈ρ₂ local minimum).

    Returns:
        (alpha, rho1, rho2) — all floats in (0, 1).
    """
    acf = lag_averaged_acf(Q_corr)           # (11,)
    lags = np.arange(1, 12, dtype=np.float64)

    def model(alpha: float, rho1: float, rho2: float) -> np.ndarray:
        return alpha * rho1 ** lags + (1.0 - alpha) * rho2 ** lags

    # Sigmoid parameterisation keeps all params in (0, 1) unconstrained.
    def _sigmoid(x: float) -> float:
        # Use the numerically stable form to avoid overflow warnings.
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def loss(p: np.ndarray) -> float:
        alpha = _sigmoid(p[0])
        rho1  = _sigmoid(p[1])
        rho2  = _sigmoid(p[2])
        return float(np.sum((model(alpha, rho1, rho2) - acf) ** 2))

    rng = np.random.default_rng(42)
    best_result = None

    # Deterministic starting points covering the parameter space.
    # Crucially includes starts that force the two components far apart
    # (e.g. rho1_init=0.95, rho2_init=0.10) to escape the degenerate ρ₁=ρ₂
    # local minimum that can trap the solver on near-single-exp ACFs.
    # sigmoid(2.94) ≈ 0.95, sigmoid(1.39) ≈ 0.80, sigmoid(-2.20) ≈ 0.10
    starts = [
        # (alpha_logit, rho1_logit, rho2_logit)
        [ 0.0,  2.94, -2.20],   # α=0.5, ρ₁=0.95, ρ₂=0.10  — well-separated
        [ 0.0,  1.39, -2.20],   # α=0.5, ρ₁=0.80, ρ₂=0.10
        [ 1.39, 2.94, -2.20],   # α=0.8, ρ₁=0.95, ρ₂=0.10  — dominant slow
        [-1.39, 2.94,  0.0 ],   # α=0.2, ρ₁=0.95, ρ₂=0.50  — dominant fast
        [ 0.0,  2.94,  1.39],   # α=0.5, ρ₁=0.95, ρ₂=0.80  — both slow
        [ 0.0,  1.39, -1.39],   # α=0.5, ρ₁=0.80, ρ₂=0.20
        [ 1.39, 1.39, -2.94],   # α=0.8, ρ₁=0.80, ρ₂=0.05
        [-1.39, 2.20, -1.39],   # α=0.2, ρ₁=0.90, ρ₂=0.20
    ]
    for _ in range(n_restarts - len(starts)):
        # Random starts with wide spread, but with ρ₁ and ρ₂ initialised
        # at opposite ends to encourage separated solutions.
        p = rng.uniform(-3, 3, size=3).tolist()
        p[1] += 1.0   # push rho1 toward higher values
        p[2] -= 1.0   # push rho2 toward lower values
        starts.append(p)

    for x0 in starts:
        res = minimize(
            loss, x0=x0, method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 200_000},
        )
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    alpha = float(_sigmoid(best_result.x[0]))
    rho1  = float(_sigmoid(best_result.x[1]))
    rho2  = float(_sigmoid(best_result.x[2]))

    # Convention: ρ₁ is the slow (large) component.
    if rho1 < rho2:
        alpha, rho1, rho2 = 1.0 - alpha, rho2, rho1

    return float(alpha), float(rho1), float(rho2)


def build_Q_H(
    delta_h: np.ndarray,
    alpha: float,
    rho1: float,
    rho2: float,
) -> np.ndarray:
    """Assemble the H×H temporal covariance matrix.

    Q_H[h1, h2] = sqrt(δ_{h1} · δ_{h2}) · (α·ρ₁^|h1-h2| + (1-α)·ρ₂^|h1-h2|)

    This is a scaled version of the two-exponential correlation matrix, where
    δ_h = Var(R_h) provides the per-horizon variance scaling.  The result is
    symmetric positive semi-definite (positive definite when ρ₁, ρ₂ ∈ (0, 1)).

    Args:
        delta_h: (12,) per-horizon variances [mph²].
        alpha:   Mixture weight for the slow component.
        rho1:    Slow-decay correlation (≥ rho2 by convention).
        rho2:    Fast-decay correlation.

    Returns:
        (12, 12) float64 covariance matrix.
    """
    H = len(delta_h)
    lags = np.abs(np.arange(H)[:, None] - np.arange(H)[None, :]).astype(np.float64)
    C_H = alpha * rho1 ** lags + (1.0 - alpha) * rho2 ** lags  # correlation
    D_sqrt = np.sqrt(delta_h.astype(np.float64))
    return D_sqrt[:, None] * C_H * D_sqrt[None, :]              # covariance


# ── Spatial GMRF fit ──────────────────────────────────────────────────────────

def empirical_spatial_cov(r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """N×N spatial covariance averaged across all horizons and windows.

    Args:
        r:    (N_win, 12, N)
        mask: (N_win, 12, N) bool

    Returns:
        (N, N) float64 covariance matrix [mph²].
    """
    N = r.shape[2]
    num = np.zeros((N, N), dtype=np.float64)
    den = np.zeros((N, N), dtype=np.float64)
    for h in range(12):
        r_h = r[:, h, :].astype(np.float64) * mask[:, h, :]
        m_h = mask[:, h, :].astype(np.float64)
        num += r_h.T @ r_h
        den += m_h.T @ m_h
    return num / np.maximum(den, 1.0)


def fit_spatial_beta(
    sig_S: np.ndarray,
    eigvals_L: np.ndarray,
    eigvecs_L: np.ndarray,
    epsilon_s: float = 1e-2,
) -> float:
    """Fit β so that (β·L + ε_s·I)^{-1} best matches Σ_S in the spectral sense.

    The Frobenius distance ||( β·L + ε_s·I)^{-1} − Σ_S||_F equals
    ||diag(1/(β·λᵢ + ε_s)) − diag(U^T Σ_S U)||_F² plus a constant
    (off-diagonal terms of U^T Σ_S U), so we minimise over the diagonal only.
    This is a 1-D optimisation over log(β) for numerical stability.

    Args:
        sig_S:     (N, N) empirical spatial covariance.
        eigvals_L: (N,) eigenvalues of the normalised Laplacian, sorted ascending.
                   The first entry is ≈ 0 (constant mode).
        eigvecs_L: (N, N) eigenvectors (columns), sorted by eigenvalue.
        epsilon_s: Fixed Laplacian regularisation  (default 0.01).

    Returns:
        Fitted β > 0 (float).
    """
    # Project Sigma_S onto Laplacian eigenbasis — only diagonal matters for fit.
    diag_M = np.diag(eigvecs_L.T @ sig_S.astype(np.float64) @ eigvecs_L)

    def objective(log_beta: float) -> float:
        beta = np.exp(log_beta)
        d = 1.0 / (beta * eigvals_L + epsilon_s)   # GMRF eigenvalues
        return float(np.sum((d - diag_M) ** 2))

    result = minimize_scalar(objective, bounds=(-8.0, 8.0), method="bounded",
                             options={"xatol": 1e-10})
    return float(np.exp(result.x))


def build_Sigma_S(
    beta: float,
    epsilon_s: float,
    eigvals_L: np.ndarray,
    eigvecs_L: np.ndarray,
) -> np.ndarray:
    """Build Σ_S = (β·L + ε_s·I)^{-1} via eigendecomposition.

    Using the pre-computed Laplacian eigendecomposition L = V Λ V^T:
        Σ_S = V · diag(1/(β·λᵢ + ε_s)) · V^T

    Args:
        beta:      Spatial precision scaling (> 0).
        epsilon_s: Laplacian regularisation (> 0).
        eigvals_L: (N,) sorted Laplacian eigenvalues.
        eigvecs_L: (N, N) eigenvectors as columns.

    Returns:
        (N, N) float64 GMRF covariance matrix.
    """
    gmrf_eigvals = 1.0 / (beta * eigvals_L + epsilon_s)   # (N,)
    return (eigvecs_L * gmrf_eigvals[None, :]) @ eigvecs_L.T


# ── SDE diffusion coefficient ─────────────────────────────────────────────────

def derive_sigma(
    Q_H: np.ndarray,
    Sigma_S: np.ndarray,
    eps_target: float = 1e-2,
) -> float:
    """Derive σ for the VP-SDE from eigenvalue analysis.

    Forward process: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε_struct
    where ε_struct ~ N(0, Σ_prior = Q_H ⊗ Σ_S),  ᾱ_t = exp(-σ² t), t∈[0,1].

    Worst-case SNR condition (x_0 in direction of λ_max, noise in λ_min):
        ᾱ_T · λ_max / ((1−ᾱ_T) · λ_min) ≤ ε_target
    ≈  ᾱ_T ≤ ε_target / cond(Σ_prior)           (for small ᾱ_T)
    →  σ = sqrt(log(cond(Σ_prior) / ε_target))

    The Kronecker condition number factorises: cond(Q_H ⊗ Σ_S) = cond(Q_H) × cond(Σ_S).

    Args:
        Q_H:       (H, H) temporal covariance.
        Sigma_S:   (N, N) spatial covariance.
        eps_target: Target worst-case SNR at t=T (default 0.01 → 1%).

    Returns:
        σ > 0 (float).
    """
    ev_Q = np.linalg.eigvalsh(Q_H)
    ev_S = np.linalg.eigvalsh(Sigma_S)
    cond_Q = float(ev_Q.max() / ev_Q.min().clip(1e-12))
    cond_S = float(ev_S.max() / ev_S.min().clip(1e-12))
    cond_prior = cond_Q * cond_S
    return float(np.sqrt(np.log(cond_prior / eps_target)))


# ── Prior sampling ────────────────────────────────────────────────────────────

def sample_prior(
    Q_H: np.ndarray,
    Sigma_S: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw i.i.d. samples from N(0, Q_H ⊗ Σ_S).

    Uses the Cholesky square-root of Q_H and the eigendecomposition square-root
    of Σ_S.  For R[n, h, i] = sum_k L_Q[h,k] sum_j Z[n,k,j] V_S[i,j]:

        R = einsum('hk, nkj, ij -> nhi', L_Q, Z, V_S_sqrt)

    where V_S_sqrt[i, j] = V_S[i, j] · sqrt(λᵢ(Σ_S)).

    Covariance check:
        Cov(R[h,i], R[h',i']) = (L_Q L_Q^T)[h,h'] · (V_S_sqrt V_S_sqrt^T)[i,i']
                               = Q_H[h,h'] · Σ_S[i,i']  ✓

    Args:
        Q_H:       (H, H) temporal covariance — must be positive definite.
        Sigma_S:   (N, N) spatial covariance — must be positive semi-definite.
        n_samples: Number of samples to draw.
        rng:       NumPy random Generator (created from seed 0 if None).

    Returns:
        (n_samples, H, N) float64 array.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    H = Q_H.shape[0]
    N = Sigma_S.shape[0]

    # Cholesky of Q_H  (H, H)
    L_Q = np.linalg.cholesky(Q_H)

    # Eigendecomposition square-root of Σ_S  (N, N)
    ev_S, V_S = np.linalg.eigh(Sigma_S)
    ev_S = ev_S.clip(0.0)                        # guard against tiny negatives
    V_S_sqrt = V_S * np.sqrt(ev_S)[None, :]      # V_S[:, j] scaled by sqrt(λ_j)

    Z = rng.standard_normal((n_samples, H, N))   # (n, H, N)
    return np.einsum("hk, nkj, ij -> nhi", L_Q, Z, V_S_sqrt)


# ── Top-level fit ─────────────────────────────────────────────────────────────

def fit_prior(
    r: np.ndarray,
    mask: np.ndarray,
    eigvals_L: np.ndarray,
    eigvecs_L: np.ndarray,
    epsilon_s: float | None = None,
    eps_target: float = 1e-2,
    standardize: bool = True,
) -> dict:
    """Fit all Kronecker prior parameters from training-set residuals.

    With ``standardize=True`` (default), residuals are first divided by
    ``target_scale[h] = sqrt(Var(r_h))``; the rest of the fit then operates on
    a unit-per-horizon-variance signal, which keeps the implied Σ_prior
    eigenvalues O(1)–O(10) and lets a moderate β_max reach stationary at τ=1
    (see exp-004 SUMMARY for the diagnosis). The original per-horizon scale is
    returned in ``target_scale`` for de-standardisation: a sample r̃ in the
    modelled variable maps back to mph as ``r̃ * target_scale[:, None]``.

    Args:
        r:          (N_win, 12, N) residuals in mph.
        mask:       (N_win, 12, N) bool validity mask.
        eigvals_L:  (N,) normalised Laplacian eigenvalues.
        eigvecs_L: (N, N) corresponding eigenvectors (columns).
        epsilon_s:  Laplacian regularisation. Default 1.0 when standardize=True
                    (constant mode has unit variance, matching standardised
                    data); 0.01 when standardize=False (legacy mph² scale).
        eps_target: Worst-case SNR target for σ derivation (default 0.01).
        standardize: If True (default), per-horizon-standardise residuals before
                    fitting; the fitted prior describes a unit-variance variable.
                    If False, fits raw-mph² residuals (legacy behaviour).

    Returns:
        Dict with keys:
            target_scale float64 (12,) — sqrt of per-horizon raw variance [mph];
                                         all-ones when standardize=False.
            delta_h      float64 (12,) — per-horizon variance of the modelled
                                         (possibly-standardised) variable.
                                         ≈ 1.0 across h when standardize=True.
            alpha        float           — slow-component weight
            rho1         float           — slow correlation decay
            rho2         float           — fast correlation decay
            beta         float           — spatial precision scaling
            epsilon_s    float           — Laplacian regularisation
            sigma        float           — VP-SDE diffusion coefficient
            standardized bool            — whether residuals were standardised
    """
    if epsilon_s is None:
        epsilon_s = 1.0 if standardize else 1e-2

    # 0. Per-horizon scale from raw residuals (in mph). Used to standardise the
    #    fit input and to de-standardise samples back to mph at sample time.
    raw_delta_h = fit_delta_h(r, mask)
    target_scale = np.sqrt(np.maximum(raw_delta_h, 1e-12))   # (12,) mph

    if standardize:
        # r̃[n,h,i] = r[n,h,i] / target_scale[h]
        r_used = r.astype(np.float64) / target_scale[None, :, None]
    else:
        r_used = r
        target_scale = np.ones(12, dtype=np.float64)

    # 1. Per-horizon variance of the modelled variable (≈ 1 after standardisation).
    delta_h = fit_delta_h(r_used, mask)

    # 2. Temporal kernel — correlations are scale-invariant, so fitted (α, ρ₁, ρ₂)
    #    are identical with or without standardisation.
    Q_emp = empirical_temporal_cov(r_used, mask)
    Q_corr = empirical_temporal_corr(Q_emp)
    alpha, rho1, rho2 = fit_temporal_kernel(Q_corr)

    # 3. Spatial GMRF on the (possibly-standardised) residuals.
    sig_S = empirical_spatial_cov(r_used, mask)
    beta = fit_spatial_beta(sig_S, eigvals_L, eigvecs_L, epsilon_s)

    # 4. Build matrices for σ derivation
    Q_H = build_Q_H(delta_h, alpha, rho1, rho2)
    Sigma_S = build_Sigma_S(beta, epsilon_s, eigvals_L, eigvecs_L)

    # 5. SDE coefficient
    sigma = derive_sigma(Q_H, Sigma_S, eps_target)

    return {
        "target_scale": target_scale,
        "delta_h":      delta_h,
        "alpha":        alpha,
        "rho1":         rho1,
        "rho2":         rho2,
        "beta":         beta,
        "epsilon_s":    epsilon_s,
        "sigma":        sigma,
        "standardized": bool(standardize),
    }
