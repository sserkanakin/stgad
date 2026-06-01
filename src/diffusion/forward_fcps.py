"""Forward graph-aware OU SDE with FCPS scheduling, per GAD (Tan et al. 2024).

GAD's forward dynamics:
    dr = -c(t) · Q · r dt + sqrt(2 c(t)) · dW_t,   W_t ~ N(0, I_{NH} dt)

with Q = Σ^{-1} = Q_T ⊗ Q_S (so the stationary cov is Σ = Σ_T ⊗ Σ_S = Q^{-1}).
The c(t) factor is a time-dependent drift coefficient; FCPS shapes it as a
"floor-constrained" polynomial so the slow modes mix only near t=T.

Closed-form marginal in the joint eigenbasis U = V_S ⊗ V_T, with c_k(0) = (U^T r_0)_k
and θ_k = (λ_k + γ)/μ_h:

    c_k(t) | c_k(0) ~ N( exp(-c̄(t) θ_k) · c_k(0),  (1/θ_k)·(1 - exp(-2 c̄(t) θ_k)) )

c̄(t) = ∫_0^t c(s) ds. The mixing condition for the slowest mode is
2 · c̄(T) · θ_min ≳ 5 (mean attenuation < 1%).

Two schedules:

  FCPSSchedule (GAD §3.1): c(t) = c_min + k · u^α with u = t/T,
                           k     = (c_0 - c_min·T)·(α+1)/T
                           c̄(t)  = c_min·t + (c_0 - c_min·T)·u^(α+1)
                           c̄(T)  = c_0

  ULSchedule (baseline):   c(t) = c·t/T   (linear ramp from 0)
                           c̄(t) = c · t² / (2T)
                           c̄(T) = c/2

The "c_0" / "c" naming follows the GAD paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .prior import KroneckerPrior


# ─── Schedules ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FCPSSchedule:
    """Floor-constrained polynomial scheduler per GAD (Tan et al. 2024)."""

    c_min: float = 0.05
    c_0: float = 20_000.0
    alpha: float = 7.0
    T: float = 1.0

    def __post_init__(self) -> None:
        if not (0 < self.c_min < 1):
            raise ValueError(f"c_min must lie in (0, 1); got {self.c_min}")
        if self.alpha <= 1:
            raise ValueError(f"alpha must be > 1; got {self.alpha}")
        if self.c_0 <= self.c_min * self.T:
            raise ValueError(
                f"c_0 ({self.c_0}) must exceed c_min·T ({self.c_min * self.T})"
            )

    def _k(self) -> float:
        return (self.c_0 - self.c_min * self.T) * (self.alpha + 1) / self.T

    def c(self, t):
        u = np.asarray(t) / self.T
        return self.c_min + self._k() * np.power(u, self.alpha)

    def c_bar(self, t):
        u = np.asarray(t) / self.T
        return self.c_min * np.asarray(t) + (self.c_0 - self.c_min * self.T) * np.power(
            u, self.alpha + 1
        )

    @classmethod
    def for_prior(
        cls,
        prior: KroneckerPrior,
        mix_target: float = 5.0,
        mix_margin: float = 2.0,
        c_min: float = 0.05,
        alpha: float = 7.0,
        T: float = 1.0,
    ) -> "FCPSSchedule":
        """Pick c_0 so that 2·θ_min·c_0 ≥ mix_target · mix_margin.

        Picks c_0 large enough to mix the slowest joint mode by t = T.
        """
        theta_min = float(prior.theta.min())
        c_0 = max(c_min * T + 1.0, 0.5 * mix_target * mix_margin / theta_min)
        return cls(c_min=c_min, c_0=c_0, alpha=alpha, T=T)


@dataclass(frozen=True)
class ULSchedule:
    """Uniform linear scheduler (baseline). c(t) = slope · t / T, c̄(t) = slope · t² / (2T).

    `slope` matches GAD's lowercase `c` in "c_t = c·t" (note: at t=T, c_t = slope; at
    t=T the cumulative c̄(T) = slope/2). For parity with an FCPSSchedule at the same
    `c_bar(T) = c_0`, set slope = 2·c_0.
    """

    slope: float = 200.0
    T: float = 1.0

    def c(self, t):  # noqa: A003 — match FCPSSchedule
        return self.slope * np.asarray(t) / self.T

    def c_bar(self, t):
        return self.slope * np.asarray(t) ** 2 / (2.0 * self.T)


# ─── Forward sampler ─────────────────────────────────────────────────────────


def _rotate_to_eigenbasis(R: np.ndarray, prior: KroneckerPrior) -> np.ndarray:
    """(..., N, H) → V_S^T R V_T."""
    return np.einsum("ki,...ih,hj->...kj", prior.V_S.T, R, prior.V_T)


def _rotate_from_eigenbasis(C: np.ndarray, prior: KroneckerPrior) -> np.ndarray:
    """Inverse: V_S C V_T^T."""
    return np.einsum("ik,...kh,hj->...ij", prior.V_S, C, prior.V_T.T)


def sample_R_t_given_R0(
    R0: np.ndarray,
    t: float,
    prior: KroneckerPrior,
    schedule,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample R_t ~ p(R_t | R_0) under the GAD forward SDE.

    Parameters
    ----------
    R0 : (..., N, H) ndarray, initial residual fields.
    t  : float, diffusion time in [0, schedule.T].
    prior, schedule : structure + scheduler.
    rng : np.random.Generator.

    Returns
    -------
    R_t : ndarray same shape as R0.
    """
    if R0.shape[-2:] != (prior.N, prior.H):
        raise ValueError(
            f"R0 last two dims must be ({prior.N}, {prior.H}); got {R0.shape}"
        )

    c_bar = float(schedule.c_bar(t))
    # H_t in eigenbasis: exp(-c̄·θ)
    Ht = np.exp(-prior.theta * c_bar)
    # Σ_t in eigenbasis: (1/θ)·(1 - exp(-2·c̄·θ))
    sigma2_t = (1.0 - np.exp(-2.0 * prior.theta * c_bar)) / prior.theta

    C0 = _rotate_to_eigenbasis(R0, prior)
    Z = rng.standard_normal(R0.shape)
    Ct = Ht * C0 + np.sqrt(sigma2_t) * Z
    return _rotate_from_eigenbasis(Ct, prior)


def sample_prior(
    shape_batch: tuple[int, ...],
    prior: KroneckerPrior,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw from the stationary prior N(0, Σ_T ⊗ Σ_S) in eigenbasis.

    Equivalent to the t → ∞ limit of sample_R_t_given_R0; supplied separately
    so callers don't need to fabricate a schedule.
    """
    Z = rng.standard_normal(shape_batch + (prior.N, prior.H))
    Ct = np.sqrt(prior.sigma2) * Z
    return _rotate_from_eigenbasis(Ct, prior)


# ─── Torch counterparts (training pipeline) ──────────────────────────────────


@dataclass(frozen=True)
class TorchPriorTensors:
    """Pre-converted torch tensors for the prior (one device + dtype).

    Construct once at the start of training; pass to sample_R_t_given_R0_torch
    on every step to avoid converting V_S, V_T, theta from numpy each call.
    """

    V_S: torch.Tensor
    V_T: torch.Tensor
    theta: torch.Tensor


def prior_to_torch(
    prior: KroneckerPrior, device: torch.device, dtype: torch.dtype = torch.float32
) -> TorchPriorTensors:
    return TorchPriorTensors(
        V_S=torch.as_tensor(prior.V_S, device=device, dtype=dtype),
        V_T=torch.as_tensor(prior.V_T, device=device, dtype=dtype),
        theta=torch.as_tensor(prior.theta, device=device, dtype=dtype),
    )


def _fcps_c_bar_torch(
    t: torch.Tensor, schedule: FCPSSchedule
) -> torch.Tensor:
    """Per-element c̄(t) under FCPSSchedule. t shape: any."""
    u = t / schedule.T
    return schedule.c_min * t + (schedule.c_0 - schedule.c_min * schedule.T) * u.pow(
        schedule.alpha + 1
    )


def sample_R_t_given_R0_torch(
    R0: torch.Tensor,
    t: torch.Tensor,
    prior_t: TorchPriorTensors,
    schedule: FCPSSchedule,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample R_t ~ p(R_t | R_0) under the GAD forward SDE — torch on-device.

    Parameters
    ----------
    R0 : (B, N, H) torch tensor.
    t  : (B,) torch tensor of per-sample diffusion times.
    prior_t : pre-converted torch tensors from prior_to_torch.
    schedule : FCPSSchedule (host-side floats; broadcasts to torch).
    generator : optional torch.Generator for the noise term.

    Returns
    -------
    R_t : (B, N, H) torch tensor on R0's device.
    """
    if R0.ndim != 3:
        raise ValueError(f"R0 must be (B, N, H); got shape {tuple(R0.shape)}")
    if t.shape != (R0.shape[0],):
        raise ValueError(f"t shape {tuple(t.shape)} must be ({R0.shape[0]},)")

    c_bar_t = _fcps_c_bar_torch(t, schedule)              # (B,)
    c_bar_b = c_bar_t.view(-1, 1, 1)                       # (B, 1, 1)
    theta_b = prior_t.theta.unsqueeze(0)                   # (1, N, H)

    H_t = torch.exp(-theta_b * c_bar_b)                    # (B, N, H)
    sigma2_t = (1.0 - torch.exp(-2.0 * theta_b * c_bar_b)) / theta_b  # (B, N, H)

    # Rotate to eigenbasis: V_S^T R0 V_T.
    C0 = torch.einsum("ki,bih,hj->bkj", prior_t.V_S.T, R0, prior_t.V_T)
    Z = torch.randn(R0.shape, generator=generator, device=R0.device, dtype=R0.dtype)
    Ct = H_t * C0 + sigma2_t.sqrt() * Z
    # Rotate back: V_S C V_T^T. Output index is `bij` (sum over the spatial mode
    # k and the temporal mode h); writing `bih` would reuse h as the output time
    # index and collapse V_T.T into a column-sum, scrambling the temporal basis.
    return torch.einsum("ik,bkh,hj->bij", prior_t.V_S, Ct, prior_t.V_T.T)
