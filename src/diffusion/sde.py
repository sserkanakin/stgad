"""Forward + reverse SDE for the Kronecker-structured residual prior.

Math (see prior-construction for the prior fit and sde-sanity for the sanity checks):

  Forward SDE (linear OU with stationary distribution N(0, Σ_prior)):
      dr_τ = -½ g(τ)² Q r_τ dτ + g(τ) dW_τ            with Q = (Σ_prior)⁻¹

  Closed-form forward marginal:
      r_τ | r_0 ~ N(μ_τ(r_0), Σ_τ)
      μ_τ = exp(-½ β̄(τ) Q) r_0
      Σ_τ = Q⁻¹ (I - exp(-β̄(τ) Q))
      β̄(τ) = ∫₀^τ g(s)² ds

  Stationary distribution: as β̄ → ∞, r_τ → N(0, Q⁻¹) = N(0, Σ_prior).

  Reverse SDE (Anderson 1982; g(τ) is a scalar so no Jacobian correction):
      d r̄_τ = [-½ g² Q r̄_τ - g² ∇log p_τ(r̄_τ)] dτ + g(τ) dW̄_τ

  Generalised Tweedie / conditional score (used in the oracle round-trip check):
      ∇_{r_τ} log p(r_τ | r_0) = -Σ_τ⁻¹ (r_τ - μ_τ(r_0))

Kronecker structure
-------------------
Σ_prior = Q_H ⊗ Σ_S, with Q_H ∈ ℝ^{H×H} the temporal covariance and Σ_S ∈ ℝ^{N×N}
the spatial GMRF covariance fit in prior-construction. Eigendecompose once at startup:
    Q_H = U_T diag(c_T) U_T.T,  c_T > 0   (eigvals of the temporal covariance)
    Σ_S = U_S diag(c_S) U_S.T,  c_S > 0   (eigvals of the spatial covariance)
Then Σ_prior in the joint eigenbasis is diagonal with eigenvalues
    s[i, j] = c_T[i] · c_S[j]                          (eigvals of Σ_prior)
    q[i, j] = 1 / s[i, j]                              (eigvals of Q)
Matrix exponentials become elementwise functions of these eigenvalues.

Implementation
--------------
Pure NumPy, float64 throughout. Sizes (H ∈ {12}, N ∈ {207, 325}) are small enough
that LAPACK on CPU dominates GPU; we don't need autograd for the sanity checks.
PyTorch / MPS will be re-introduced in score-network for network training.

The standard tensor shape for residuals throughout is `(..., H, N)` — leading
batch dims are preserved by every transform.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Noise schedule (VP-SDE-style linear β)
# ─────────────────────────────────────────────────────────────────────────────

def beta_schedule(tau, beta_min: float, beta_max: float):
    """β(τ) = β_min + τ (β_max − β_min)."""
    return beta_min + tau * (beta_max - beta_min)


def beta_bar(tau, beta_min: float, beta_max: float):
    """β̄(τ) = ∫₀^τ β(s) ds = β_min · τ + ½ τ² (β_max − β_min)."""
    return beta_min * tau + 0.5 * tau ** 2 * (beta_max - beta_min)


def g_squared(tau, beta_min: float, beta_max: float):
    """g(τ)² = β(τ)."""
    return beta_schedule(tau, beta_min, beta_max)


# ─────────────────────────────────────────────────────────────────────────────
# Eigendecomposition of the Kronecker prior
# ─────────────────────────────────────────────────────────────────────────────

def eigendecompose_prior(Q_H: np.ndarray, Sigma_S: np.ndarray) -> dict:
    """Eigendecompose temporal Q_H and spatial Σ_S; build the joint mode array.

    Args:
        Q_H:     (H, H) temporal covariance, symmetric PD.
        Sigma_S: (N, N) spatial covariance, symmetric PD.

    Returns:
        Dict with keys
            U_T (H, H), c_T (H,)  — eigvecs / eigvals of Q_H, c_T sorted ascending
            U_S (N, N), c_S (N,)  — eigvecs / eigvals of Σ_S, c_S sorted ascending
            s   (H, N)            — eigvals of Σ_prior in joint eigenbasis,
                                    s[i, j] = c_T[i] · c_S[j]   (ALL > 0)
            q   (H, N)            — eigvals of Q in joint eigenbasis = 1 / s

    Tiny negative eigenvalues from float roundoff are clipped to a small positive
    floor to keep `1 / s` finite. In practice both Q_H and Σ_S are well-conditioned
    after the prior fit (see prior-construction) so the clip is defensive only.
    """
    c_T, U_T = np.linalg.eigh(np.asarray(Q_H, dtype=np.float64))
    c_S, U_S = np.linalg.eigh(np.asarray(Sigma_S, dtype=np.float64))
    floor = 1e-12
    c_T = np.clip(c_T, floor, None)
    c_S = np.clip(c_S, floor, None)
    s = np.outer(c_T, c_S)               # (H, N)
    q = 1.0 / s                          # (H, N)
    return {"U_T": U_T, "c_T": c_T, "U_S": U_S, "c_S": c_S, "s": s, "q": q}


# ─────────────────────────────────────────────────────────────────────────────
# Spectral basis transforms
# ─────────────────────────────────────────────────────────────────────────────
#
# For r of shape (..., H, N):
#   r_tilde = U_Tᵀ · r · U_S        (project to joint eigenbasis)
#   r       = U_T  · r_tilde · U_Sᵀ (project back)
#
# We use einsum so leading batch dimensions broadcast naturally.

def to_spectral(r: np.ndarray, U_T: np.ndarray, U_S: np.ndarray) -> np.ndarray:
    """Project r (..., H, N) into the joint Q_H ⊗ Σ_S eigenbasis."""
    return np.einsum("hi, ...hn, nj -> ...ij", U_T, r, U_S)


def from_spectral(r_tilde: np.ndarray, U_T: np.ndarray, U_S: np.ndarray) -> np.ndarray:
    """Inverse of to_spectral (rotate back to the original basis)."""
    return np.einsum("hi, ...ij, nj -> ...hn", U_T, r_tilde, U_S)


# ─────────────────────────────────────────────────────────────────────────────
# Apply the precision matrix Q in the original basis
# ─────────────────────────────────────────────────────────────────────────────

def apply_Q(r: np.ndarray, eig: dict) -> np.ndarray:
    """Compute Q · r where Q = (Σ_prior)⁻¹.

    Diagonal in the joint eigenbasis with eigenvalues q = 1/s.
    """
    r_tilde = to_spectral(r, eig["U_T"], eig["U_S"])
    Qr_tilde = r_tilde * eig["q"]                       # broadcast (..., H, N) * (H, N)
    return from_spectral(Qr_tilde, eig["U_T"], eig["U_S"])


# ─────────────────────────────────────────────────────────────────────────────
# Forward marginal — closed-form
# ─────────────────────────────────────────────────────────────────────────────

def _forward_decay_var(tau: float, eig: dict,
                       beta_min: float, beta_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-mode decay and variance at time τ in the joint eigenbasis.

        decay[i, j] = exp(-½ β̄(τ) · q[i, j])  = exp(-½ β̄ / s[i, j])
        var[i, j]   = (1 - exp(-β̄ q[i, j])) / q[i, j]   = s[i, j] · (1 - exp(-β̄ / s[i, j]))

    The variance is computed via -s * np.expm1(-β̄/s) for numerical stability when
    β̄/s is small (forward process not yet near stationarity in that mode).
    """
    bb = beta_bar(tau, beta_min, beta_max)
    ratio = bb / eig["s"]                                # (H, N)
    decay = np.exp(-0.5 * ratio)
    var = -eig["s"] * np.expm1(-ratio)                   # = s · (1 - exp(-β̄/s)), stable
    return decay, var


def forward_sample(r_0: np.ndarray, tau: float, eig: dict,
                   beta_min: float, beta_max: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Sample r_τ ~ p(r_τ | r_0) from the closed-form forward marginal.

    Args:
        r_0:    (..., H, N) clean residuals (any number of leading batch dims).
        tau:    Scalar time in [0, 1].
        eig:    Output of `eigendecompose_prior`.
        beta_min, beta_max: Schedule parameters.
        rng:    NumPy Generator.

    Returns:
        (..., H, N) — r_τ in the original basis.
    """
    decay, var = _forward_decay_var(tau, eig, beta_min, beta_max)
    r_0_tilde = to_spectral(r_0, eig["U_T"], eig["U_S"])
    mean_tilde = decay * r_0_tilde
    eps = rng.standard_normal(r_0.shape)
    r_tau_tilde = mean_tilde + np.sqrt(var) * eps
    return from_spectral(r_tau_tilde, eig["U_T"], eig["U_S"])


# ─────────────────────────────────────────────────────────────────────────────
# Stationary-distribution sampler — direct draw from N(0, Σ_prior)
# ─────────────────────────────────────────────────────────────────────────────

def sample_stationary(n_samples: int, eig: dict,
                      rng: np.random.Generator) -> np.ndarray:
    """Draw i.i.d. samples from the stationary distribution N(0, Σ_prior).

    Uses the joint eigenbasis: Σ_prior is diagonal there with entries `s`, so a
    draw is sqrt(s) ⊙ z with z ~ N(0, I), then rotated back.
    """
    H, N = eig["s"].shape
    z_tilde = rng.standard_normal((n_samples, H, N))
    r_tilde = np.sqrt(eig["s"]) * z_tilde
    return from_spectral(r_tilde, eig["U_T"], eig["U_S"])


# ─────────────────────────────────────────────────────────────────────────────
# Generalised Tweedie / conditional score
# ─────────────────────────────────────────────────────────────────────────────

def oracle_score(r_tau: np.ndarray, r_0: np.ndarray, tau: float, eig: dict,
                 beta_min: float, beta_max: float) -> np.ndarray:
    """Compute ∇_{r_τ} log p(r_τ | r_0) = -Σ_τ⁻¹ (r_τ - μ_τ(r_0)).

    Σ_τ is diagonal in the joint eigenbasis with entries `var`, so the inverse
    is elementwise division by `var`.
    """
    decay, var = _forward_decay_var(tau, eig, beta_min, beta_max)
    r_tau_tilde = to_spectral(r_tau, eig["U_T"], eig["U_S"])
    r_0_tilde = to_spectral(r_0, eig["U_T"], eig["U_S"])
    mean_tilde = decay * r_0_tilde
    score_tilde = -(r_tau_tilde - mean_tilde) / var
    return from_spectral(score_tilde, eig["U_T"], eig["U_S"])


# ─────────────────────────────────────────────────────────────────────────────
# Reverse SDE — Euler-Maruyama with oracle conditional score
# ─────────────────────────────────────────────────────────────────────────────

def reverse_oracle(r_tau_init: np.ndarray, r_0: np.ndarray,
                   n_steps: int, eig: dict,
                   beta_min: float, beta_max: float,
                   rng: np.random.Generator,
                   tau_start: float = 1.0,
                   tau_eps: float = 1e-3) -> np.ndarray:
    """Reverse-time Euler-Maruyama from τ_start down to τ_eps with oracle score.

    Per-step update (matching the spec):
        drift = -½ g²(τ) · Q · r  -  g²(τ) · score(r, r_0, τ)
        r ← r  -  drift · Δτ  +  g(τ) · √Δτ · ε

    Notes:
      - The reverse SDE goes backward in time; each Δτ here is positive but applied
        with a leading minus sign to the drift (per the user's spec / Anderson 1982).
      - We stop at τ_eps > 0 because Σ_τ → 0 as τ → 0 and the score (1/var) blows
        up. With τ_eps = 1e-3 and β_min = 0.01, β̄(τ_eps) ≈ 1e-5, the residual
        gap between μ_{τ_eps}(r_0) and r_0 is well below 1% in every eigenmode
        (decay = exp(-½ β̄ q), which is ≈ 1 for all q in our setups).

    Args:
        r_tau_init: (B, H, N) state at τ_start (typically obtained from forward_sample).
        r_0:        (B, H, N) clean reference used for the oracle conditional score.
        n_steps:    Number of Euler-Maruyama steps (uniform in τ).
        eig:        Output of `eigendecompose_prior`.
        beta_min, beta_max: Schedule parameters.
        rng:        NumPy Generator.
        tau_start:  Start time (default 1.0).
        tau_eps:    Stop time (default 1e-3) — chosen for numerical stability.

    Returns:
        r_recovered: (B, H, N) at τ ≈ tau_eps.
    """
    if r_tau_init.shape != r_0.shape:
        raise ValueError(
            f"r_tau_init and r_0 must have the same shape; got "
            f"{r_tau_init.shape} vs {r_0.shape}"
        )
    r = r_tau_init.copy().astype(np.float64)
    taus = np.linspace(tau_start, tau_eps, n_steps + 1)
    for k in range(n_steps):
        tau_k = float(taus[k])
        delta_tau = float(taus[k] - taus[k + 1])         # > 0
        g2 = float(g_squared(tau_k, beta_min, beta_max))

        Q_r = apply_Q(r, eig)
        score = oracle_score(r, r_0, tau_k, eig, beta_min, beta_max)

        drift = -0.5 * g2 * Q_r - g2 * score
        noise = np.sqrt(g2 * delta_tau) * rng.standard_normal(r.shape)

        r = r - drift * delta_tau + noise
    return r
