"""PyTorch adapter for the Kronecker-structured SDE prior.

Mirrors the NumPy implementation in ``src/diffusion/sde.py`` with torch
tensors so the forward marginal, basis rotations, and stationary sampler
are differentiable and run on MPS / CUDA.

Math is unchanged — see ``sde.py`` for the derivations. The split is:

  - **Eigendecomposition** stays in NumPy / float64 (one-time startup cost,
    not in the autograd path, and LAPACK on CPU is faster than torch's
    ``torch.linalg.eigh`` for our 12×12 / 207×207 / 325×325 sizes).

  - **Per-step ops** (basis rotations, forward marginal, Q-application,
    stationary sampling) run in PyTorch at the chosen device + dtype.
    Default dtype is float32 (training convention); float64 available for
    diagnostics.

The bridge is ``eig_to_torch(eig_np, device, dtype)``: take the dict from
``sde.eigendecompose_prior(Q_H, Sigma_S)``, convert each entry to a torch
tensor on the target device. Call once at startup; pass the resulting
dict to every subsequent torch op.

Tensor shape convention is the same as the NumPy module: residuals are
``(..., H, N)`` with leading batch dims preserved by every transform.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Schedule — accepts torch tensors or python floats
# ─────────────────────────────────────────────────────────────────────────────

def beta_schedule(tau, beta_min: float, beta_max: float):
    """β(τ) = β_min + τ (β_max − β_min). Works for scalar or tensor τ."""
    return beta_min + tau * (beta_max - beta_min)


def beta_bar(tau, beta_min: float, beta_max: float):
    """β̄(τ) = β_min·τ + ½ τ² (β_max − β_min). Works for scalar or tensor τ."""
    return beta_min * tau + 0.5 * tau ** 2 * (beta_max - beta_min)


def g_squared(tau, beta_min: float, beta_max: float):
    """g(τ)² = β(τ)."""
    return beta_schedule(tau, beta_min, beta_max)


# ─────────────────────────────────────────────────────────────────────────────
# NumPy eig dict → torch tensors
# ─────────────────────────────────────────────────────────────────────────────

EigTorch = Dict[str, torch.Tensor]


def eig_to_torch(
    eig_np: Dict[str, np.ndarray],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EigTorch:
    """Convert the NumPy eig dict (from sde.eigendecompose_prior) to torch.

    Args:
        eig_np: Output of ``sde.eigendecompose_prior``. Keys ``U_T, c_T,
            U_S, c_S, s, q`` with the documented shapes.
        device: Target device — typically ``"mps"``, ``"cuda"``, or ``"cpu"``.
        dtype: Target dtype. Default ``float32`` for training; use
            ``float64`` for diagnostic checks against the NumPy module.

    Returns:
        A dict with the same keys, each a torch.Tensor on the requested
        device + dtype. Tensors are non-trainable (no grad) since they're
        fit-time constants.
    """
    out: EigTorch = {}
    for k in ("U_T", "c_T", "U_S", "c_S", "s", "q"):
        if k not in eig_np:
            raise KeyError(f"eig_np missing required key {k!r}")
        out[k] = torch.as_tensor(eig_np[k], dtype=dtype, device=device)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Spectral basis transforms (differentiable)
# ─────────────────────────────────────────────────────────────────────────────
# For r of shape (..., H, N):
#   r_tilde = U_Tᵀ · r · U_S
#   r       = U_T  · r_tilde · U_Sᵀ

def to_spectral(r: torch.Tensor, U_T: torch.Tensor, U_S: torch.Tensor) -> torch.Tensor:
    """Project r (..., H, N) into the joint Q_H ⊗ Σ_S eigenbasis."""
    return torch.einsum("hi,...hn,nj->...ij", U_T, r, U_S)


def from_spectral(r_tilde: torch.Tensor, U_T: torch.Tensor, U_S: torch.Tensor) -> torch.Tensor:
    """Inverse of `to_spectral` — rotate back to the original basis."""
    return torch.einsum("hi,...ij,nj->...hn", U_T, r_tilde, U_S)


# ─────────────────────────────────────────────────────────────────────────────
# Apply precision matrix Q = Σ_prior⁻¹ in the original basis
# ─────────────────────────────────────────────────────────────────────────────

def apply_Q(r: torch.Tensor, eig: EigTorch) -> torch.Tensor:
    """Compute Q · r where Q is diagonal in the joint eigenbasis with eigvals q=1/s."""
    r_tilde = to_spectral(r, eig["U_T"], eig["U_S"])
    Qr_tilde = r_tilde * eig["q"]
    return from_spectral(Qr_tilde, eig["U_T"], eig["U_S"])


# ─────────────────────────────────────────────────────────────────────────────
# Forward marginal — closed form per mode
# ─────────────────────────────────────────────────────────────────────────────

def _forward_decay_var(
    tau,
    eig: EigTorch,
    beta_min: float,
    beta_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-mode decay and variance at time τ in the joint eigenbasis.

    Returns:
        decay = exp(-½ β̄(τ) / s)
        var   = s · (1 - exp(-β̄(τ) / s))

    Shape rules:
      - τ a Python float / 0-d tensor → decay, var shape ``(H, N)``
      - τ a 1-d tensor ``(B,)``        → decay, var shape ``(B, H, N)``
        (per-batch τ values broadcast against the per-mode ``s`` grid)

    The variance uses ``-s · expm1(-ratio)`` for numerical stability when
    β̄/s is small (mode not yet near stationarity).
    """
    bb = beta_bar(tau, beta_min, beta_max)
    # If bb is a 1-d tensor (per-batch τ), reshape to (B, 1, 1) so it
    # broadcasts against eig["s"] of shape (H, N). Scalar bb is left alone.
    if torch.is_tensor(bb) and bb.dim() >= 1:
        bb = bb.view(*bb.shape, *([1] * eig["s"].dim()))
    ratio = bb / eig["s"]
    decay = torch.exp(-0.5 * ratio)
    var = -eig["s"] * torch.expm1(-ratio)
    return decay, var


def forward_sample(
    r_0: torch.Tensor,
    tau,
    eig: EigTorch,
    beta_min: float,
    beta_max: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample r_τ ~ p(r_τ | r_0) from the closed-form forward marginal.

    Args:
        r_0:  ``(..., H, N)`` clean residuals.
        tau:  Scalar or 0-d tensor in [0, 1]. For per-batch τ pass a
              ``(B,)`` tensor — bb broadcasts via the standard rules.
        eig:  Output of ``eig_to_torch``.
        beta_min, beta_max: schedule parameters.
        generator: Optional torch.Generator for deterministic noise.

    Returns:
        ``(..., H, N)`` — r_τ in the original basis.
    """
    decay, var = _forward_decay_var(tau, eig, beta_min, beta_max)
    r_0_tilde = to_spectral(r_0, eig["U_T"], eig["U_S"])
    mean_tilde = decay * r_0_tilde
    eps = torch.randn(r_0.shape, dtype=r_0.dtype, device=r_0.device, generator=generator)
    r_tau_tilde = mean_tilde + torch.sqrt(var) * eps
    return from_spectral(r_tau_tilde, eig["U_T"], eig["U_S"])


def forward_marginal_moments(
    r_0: torch.Tensor,
    tau,
    eig: EigTorch,
    beta_min: float,
    beta_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (μ_τ(r_0), σ_τ) in the *original* basis for DSM target weighting.

    μ_τ has the same shape as r_0; σ_τ is the (H, N) per-mode std of the
    forward marginal expressed in the joint eigenbasis (not rotated back —
    DSM losses weight in that basis).
    """
    decay, var = _forward_decay_var(tau, eig, beta_min, beta_max)
    r_0_tilde = to_spectral(r_0, eig["U_T"], eig["U_S"])
    mean_tilde = decay * r_0_tilde
    mean = from_spectral(mean_tilde, eig["U_T"], eig["U_S"])
    sigma = torch.sqrt(var)                # (H, N) — eigenbasis std
    return mean, sigma


# ─────────────────────────────────────────────────────────────────────────────
# Stationary-distribution sampler — N(0, Σ_prior)
# ─────────────────────────────────────────────────────────────────────────────

def sample_stationary(
    n_samples: int,
    eig: EigTorch,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Draw i.i.d. samples r ~ N(0, Σ_prior).

    Σ_prior is diagonal in the joint eigenbasis with entries ``s``; a draw
    is ``sqrt(s) ⊙ z`` with ``z ~ N(0, I)``, then rotated back.

    Args:
        n_samples: Batch size.
        eig:       Eigendecomp on the desired device.
        generator: Optional torch.Generator for deterministic noise.
        dtype:     Override the dtype (defaults to ``eig["s"].dtype``).

    Returns:
        ``(n_samples, H, N)`` on the same device as ``eig``.
    """
    s = eig["s"]
    H, N = s.shape
    dtype = dtype or s.dtype
    z_tilde = torch.randn(n_samples, H, N, dtype=dtype, device=s.device, generator=generator)
    r_tilde = torch.sqrt(s) * z_tilde
    return from_spectral(r_tilde, eig["U_T"], eig["U_S"])
