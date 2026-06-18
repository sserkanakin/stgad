"""static checks on a fitted prior — no diffusion, no trained model (stage 0a/0b).

0a (psd sanity): is Σ = Σ_T ⊗ Σ_S actually a valid covariance and numerically sane?
0b (conditioning): how stiff is the precision Q, and what mixing budget c₀ does that
    force on the forward schedule? this is the table that explains, in numbers, why
    the structured prior needs a large c₀ while isotropic needs ≈ 5.
"""
from __future__ import annotations

import numpy as np

from src.diffusion.prior import KroneckerPrior


# ─── 0a: psd / validity gate ─────────────────────────────────────────────────


def psd_checks(prior: KroneckerPrior) -> list[tuple[str, bool, str]]:
    """six checks that the fitted prior is a valid, sane covariance.

    returns a list of (description, passed, detail). all must pass — any failure
    means the prior is broken and stages 1+ are meaningless.
    """
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    # 1. joint precision eigenvalues all positive  ->  Σ = Q⁻¹ is positive definite
    add("θ[k,h] > 0  (precision PD)", (prior.theta > 0).all(),
        f"θ_min = {prior.theta.min():.3e}")

    # 2. stationary per-mode variance 1/θ finite (no infinite-variance mode)
    add("σ² = 1/θ finite", np.isfinite(prior.sigma2).all(),
        f"σ²_max = {prior.sigma2.max():.3e}")

    # 3. spatial eigenbasis orthonormal  ->  Vₛ is a genuine rotation
    err_s = float(np.abs(prior.V_S.T @ prior.V_S - np.eye(prior.N)).max())
    add("Vₛ orthonormal", err_s < 1e-6, f"max|VᵀV − I| = {err_s:.1e}")

    # 4. temporal eigenbasis orthonormal
    err_t = float(np.abs(prior.V_T.T @ prior.V_T - np.eye(prior.H)).max())
    add("V_T orthonormal", err_t < 1e-6, f"max|VᵀV − I| = {err_t:.1e}")

    # 5. temporal factor Σ_T positive definite (μ_h are its eigenvalues)
    add("Σ_T eigenvalues μ_h > 0", (prior.mu_T > 0).all(),
        f"μ_min = {prior.mu_T.min():.3e}")

    # 6. R_H a valid correlation matrix: symmetric, unit diagonal, PSD
    rh_eig = np.linalg.eigvalsh(prior.R_H)
    rh_ok = (np.allclose(prior.R_H, prior.R_H.T)
             and np.allclose(np.diag(prior.R_H), 1.0)
             and (rh_eig > -1e-8).all())
    add("R_H valid corr (sym, unit-diag, PSD)", rh_ok, f"min eig = {rh_eig.min():.3e}")

    return checks


# ─── 0b: conditioning / mixing budget ────────────────────────────────────────


def conditioning_row(prior: KroneckerPrior, mix_target: float = 5.0,
                     mix_margin: float = 2.0) -> dict[str, float]:
    """one row of the conditioning table for a fitted prior.

    κ values are condition numbers (max/min eigenvalue) of the precision factors.
    c₀ is the forward-schedule mixing budget the prior forces (same formula as
    FCPSSchedule.for_prior): c₀ = ½·mix_target·mix_margin / θ_min, so the slowest
    mode reaches 2 c₀ θ_min = mix_target·mix_margin by t=T. the stiffest mode is
    then driven c₀ θ_max = ½·mix_target·mix_margin·κ(Q) — that product is why a
    stiff prior over-drives its fast modes.
    """
    theta = prior.theta
    th_min, th_max = float(theta.min()), float(theta.max())

    kappa_T = float(prior.mu_T.max() / prior.mu_T.min())   # temporal precision ∝ 1/μ_h
    if prior.gamma > 0:                                    # structured spatial factor
        q_S = prior.lam_S + prior.gamma
        kappa_S = float(q_S.max() / q_S.min())
    else:                                                  # flat or isotropic: no spatial structure
        kappa_S = 1.0
    kappa_Q = th_max / th_min                              # = kappa_S · kappa_T

    c0 = 0.5 * mix_target * mix_margin / th_min
    return {
        "θ_min": th_min,
        "θ_max": th_max,
        "κ(Q_S)": kappa_S,
        "κ(Q_T)": kappa_T,
        "κ(Q)": kappa_Q,
        "c₀ (mix budget)": c0,
        "c₀·θ_max (stiffest drive)": c0 * th_max,
    }


def conditioning_table(priors: dict[str, KroneckerPrior], **kw) -> dict[str, dict[str, float]]:
    """conditioning_row for each named prior -> {prior_name: row}."""
    return {name: conditioning_row(p, **kw) for name, p in priors.items()}
