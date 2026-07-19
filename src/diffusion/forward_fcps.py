"""forward graph-aware OU sde with the fcps schedule, from gad (rozada et al. 2025).

the forward process is
    dr = -c(t) Q r dt + sqrt(2 c(t)) dW,   W ~ N(0, I_{NH} dt)
with precision Q = Q_T ⊗ Q_S, so its stationary covariance is Σ = Σ_T ⊗ Σ_S.
c(t) is a time-dependent rate; fcps shapes it so the slow graph modes only mix
near t=T.

in the joint eigenbasis (θ_k = (λ_k + γ)/μ_h) the marginal is closed-form,
    c_k(t) | c_k(0) ~ N( exp(-c̄(t) θ_k) c_k(0),  (1 - exp(-2 c̄(t) θ_k)) / θ_k ),
where c̄(t) = ∫_0^t c(s) ds. the slowest mode mixes once 2 c̄(T) θ_min ≳ 5.

two schedules:
  FCPSSchedule: c(t) = c_min + k u^α, c̄(t) = c_min t + (c_0 - c_min T) u^(α+1),
                c̄(T) = c_0   (u = t/T)
  ULSchedule:   c(t) = slope t/T, c̄(t) = slope t² / (2T)   (linear baseline)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .prior import KroneckerPrior


# ─── Schedules ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FCPSSchedule:
    """floor-constrained polynomial schedule from gad (rozada et al. 2025).

    GAD-literal defaults: c_min=0.1, c_0 (= GAD's S) = 7.0, alpha=4.0, T=1.0 — the exact
    values in exp_gasde.py. these are FIXED, not auto-derived: the θ→GAD-scale reparam in
    load_prior already puts θ_min=γ=0.8, so a fixed S=7 reproduces GAD's mixing
    (2·S·θ_min ≈ 2·7·0.8 = 11.2) on every dataset. the c(t)/c̄(t) form is GAD's verbatim.
    (for_prior below is legacy and no longer used by the pipeline.)
    """

    c_min: float = 0.1
    c_0: float = 7.0
    alpha: float = 4.0
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

    @property
    def c_lo(self) -> float:
        """noise-floor lower bound for log-uniform training draws (fixed, prior-independent;
        matches noise_level_bounds' floor = NOISE_FLOOR²/2 ≈ 8e-4)."""
        return 0.04 ** 2 / 2.0

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
        c_min: float = 0.1,
        alpha: float = 4.0,
        T: float = 1.0,
    ) -> "FCPSSchedule":
        """pick c_0 (= GAD's S) large enough to mix the slowest mode by t=T
        (so that 2 θ_min c_0 ≥ mix_target · mix_margin). GAD fixes S=7 for θ_min≈0.8
        (2·7·0.8≈11); we reproduce that mixing for our smaller kronecker θ_min."""
        theta_min = float(prior.theta.min())
        c_0 = max(c_min * T + 1.0, 0.5 * mix_target * mix_margin / theta_min)
        return cls(c_min=c_min, c_0=c_0, alpha=alpha, T=T)

    def t_of_c_bar(self, c_bar, n_iter: int = 30):
        """invert c̄(t) → t ∈ [0, T] (newton), to match GAD: it samples the noise
        level s = c̄ then conditions the net on the corresponding diffusion time t.
        c̄(t) = c_min·t + (c_0 - c_min·T)·(t/T)^(α+1) is monotone in t. accepts np/torch."""
        is_torch = isinstance(c_bar, torch.Tensor)
        cb = c_bar if is_torch else np.asarray(c_bar, dtype=float)
        # start from the dominant-term guess t ≈ T·(c̄ / (c_0 - c_min·T))^(1/(α+1))
        k = self.c_0 - self.c_min * self.T
        if is_torch:
            t = self.T * torch.clamp(cb / k, min=0.0) ** (1.0 / (self.alpha + 1))
            t = t.clamp(1e-8, self.T)
            for _ in range(n_iter):
                u = t / self.T
                f = self.c_min * t + k * u.pow(self.alpha + 1) - cb
                fp = self.c_min + k * (self.alpha + 1) / self.T * u.pow(self.alpha)
                t = (t - f / fp).clamp(1e-8, self.T)
            return t
        t = self.T * np.clip(cb / k, 0.0, None) ** (1.0 / (self.alpha + 1))
        t = np.clip(t, 1e-8, self.T)
        for _ in range(n_iter):
            u = t / self.T
            f = self.c_min * t + k * u ** (self.alpha + 1) - cb
            fp = self.c_min + k * (self.alpha + 1) / self.T * u ** self.alpha
            t = np.clip(t - f / fp, 1e-8, self.T)
        return t


@dataclass(frozen=True)
class ULSchedule:
    """linear rate schedule: c(t) = slope t/T, c̄(t) = slope t²/(2T).

    the reverse/training schedule for the ISOTROPIC arm: with a uniform θ there
    are no slow graph modes to protect, so FCPS's floor-constrained polynomial is
    unnecessary — a plain linear rate does the job. for heat parity with an fcps
    schedule at the same total heat c̄(T)=c_0, set slope = 2·c_0 (c_0=7 → slope 14).
    duck-types FCPSSchedule (c_0 / c / c_bar / t_of_c_bar), so the trainer and the
    EM reverse sampler work unchanged.
    """

    slope: float = 14.0
    T: float = 1.0

    @property
    def c_0(self) -> float:
        """total heat c̄(T) = slope·T/2 — the same role as FCPSSchedule.c_0."""
        return self.slope * self.T / 2.0

    def c(self, t):  # noqa: A003 — match FCPSSchedule
        return self.slope * np.asarray(t) / self.T

    def c_bar(self, t):
        return self.slope * np.asarray(t) ** 2 / (2.0 * self.T)

    def t_of_c_bar(self, c_bar):
        """invert c̄(t) → t ∈ [0, T]: t = T·sqrt(c̄/c_0), closed form. np or torch."""
        if isinstance(c_bar, torch.Tensor):
            return self.T * torch.clamp(c_bar / self.c_0, min=0.0).sqrt()
        return self.T * np.sqrt(np.clip(np.asarray(c_bar, dtype=float) / self.c_0, 0.0, None))


@dataclass(frozen=True)
class VPSchedule:
    """standard VP-SDE / DDPM forward: dx = -beta(t)/2 x dt + sqrt(beta(t)) dw with
    LINEAR beta in [beta_min, beta_max] (Song et al. / DDPM defaults 0.1 -> 20).
    in our per-mode OU machinery this is exactly theta=1, sigma=1, c(t) = beta(t)/2.
    duck-types FCPSSchedule (c_0 / c / c_bar / t_of_c_bar)."""

    beta_min: float = 0.1
    beta_max: float = 20.0
    T: float = 1.0

    @property
    def c_0(self) -> float:
        return (self.beta_min * self.T + (self.beta_max - self.beta_min) * self.T / 2.0) / 2.0

    def c(self, t):  # noqa: A003
        u = np.asarray(t) / self.T
        return (self.beta_min + (self.beta_max - self.beta_min) * u) / 2.0

    def c_bar(self, t):
        u = np.asarray(t) / self.T
        return (self.beta_min * np.asarray(t) + (self.beta_max - self.beta_min) * self.T * u ** 2 / 2.0) / 2.0

    def t_of_c_bar(self, c_bar):
        """invert the quadratic c̄(t); closed form. np or torch."""
        a = (self.beta_max - self.beta_min) / (4.0 * self.T)
        b = self.beta_min / 2.0
        if isinstance(c_bar, torch.Tensor):
            disc = (b * b + 4.0 * a * torch.clamp(c_bar, min=0.0)).sqrt()
            return torch.clamp((disc - b) / (2.0 * a), 1e-8, self.T)
        disc = np.sqrt(b * b + 4.0 * a * np.clip(np.asarray(c_bar, dtype=float), 0.0, None))
        return np.clip((disc - b) / (2.0 * a), 1e-8, self.T)


@dataclass(frozen=True)
class VESchedule:
    """VE-SDE (Song et al.): no drift, exploding variance sigma^2(t) with geometric
    sigma(t) = sigma_min (sigma_max/sigma_min)^t. in the theta->0 limit of our OU
    machinery the marginal variance is 2*c̄(t), so c̄(t) = sigma^2(t)/2. terminal draws
    must come from N(0, sigma_max^2) explicitly (NOT the theta->0 stationary)."""

    sigma_min: float = 0.01
    sigma_max: float = 50.0
    T: float = 1.0

    @property
    def c_0(self) -> float:
        return self.sigma_max ** 2 / 2.0

    @property
    def c_lo(self) -> float:
        return self.sigma_min ** 2 / 2.0

    def _lr(self) -> float:
        return math.log(self.sigma_max / self.sigma_min)

    def c(self, t):  # noqa: A003
        return 2.0 * self._lr() * np.asarray(self.c_bar(t))

    def c_bar(self, t):
        u = np.asarray(t) / self.T
        return (self.sigma_min ** 2 / 2.0) * np.exp(2.0 * self._lr() * u)

    def t_of_c_bar(self, c_bar):
        if isinstance(c_bar, torch.Tensor):
            return torch.clamp(torch.log(torch.clamp(2.0 * c_bar, min=1e-20) / self.sigma_min ** 2)
                               / (2.0 * self._lr()) * self.T, 1e-8, self.T)
        v = np.log(np.clip(2.0 * np.asarray(c_bar, dtype=float), 1e-20, None) / self.sigma_min ** 2)
        return np.clip(v / (2.0 * self._lr()) * self.T, 1e-8, self.T)


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
    """sample R_t ~ p(R_t | R_0) under the forward sde (numpy).

    R0 is (..., N, H); t is a scalar time in [0, schedule.T]; the result has the
    same shape as R0.
    """
    if R0.shape[-2:] != (prior.N, prior.H):
        raise ValueError(
            f"R0 last two dims must be ({prior.N}, {prior.H}); got {R0.shape}"
        )

    c_bar = float(schedule.c_bar(t))
    # H_t in eigenbasis: exp(-c̄·θ)
    Ht = np.exp(-prior.theta * c_bar)
    # Σ_t in eigenbasis: σ²·(1 - exp(-2·c̄·θ))/θ
    sigma2_t = prior.sigma2_scale * (1.0 - np.exp(-2.0 * prior.theta * c_bar)) / prior.theta

    C0 = _rotate_to_eigenbasis(R0, prior)
    Z = rng.standard_normal(R0.shape)
    Ct = Ht * C0 + np.sqrt(sigma2_t) * Z
    return _rotate_from_eigenbasis(Ct, prior)


def sample_prior(
    shape_batch: tuple[int, ...],
    prior: KroneckerPrior,
    rng: np.random.Generator,
) -> np.ndarray:
    """draw from the stationary prior N(0, Σ_T ⊗ Σ_S).

    same as the t→∞ limit of sample_R_t_given_R0, but without needing a schedule.
    """
    Z = rng.standard_normal(shape_batch + (prior.N, prior.H))
    Ct = np.sqrt(prior.sigma2) * Z
    return _rotate_from_eigenbasis(Ct, prior)


# ─── Torch counterparts (training pipeline) ──────────────────────────────────


@dataclass(frozen=True)
class TorchPriorTensors:
    """prior tensors pre-moved to one device/dtype, built once before training
    so the per-step torch sampler doesn't re-convert from numpy each call."""

    V_S: torch.Tensor
    V_T: torch.Tensor
    theta: torch.Tensor
    sigma2_scale: float = 1.0


def prior_to_torch(
    prior: KroneckerPrior, device: torch.device, dtype: torch.dtype = torch.float32
) -> TorchPriorTensors:
    return TorchPriorTensors(
        V_S=torch.as_tensor(prior.V_S, device=device, dtype=dtype),
        V_T=torch.as_tensor(prior.V_T, device=device, dtype=dtype),
        theta=torch.as_tensor(prior.theta, device=device, dtype=dtype),
        sigma2_scale=float(prior.sigma2_scale),
    )


# ─── noise-level (c̄) parameterization — the EDM-style training/sampling axis ──
#
# the control variable is the cumulative noise c̄ (not diffusion-time t), because
# the whole process depends only on c̄ and the per-mode precision θ:
#     H(c̄)  = exp(-c̄ θ)                    (signal retained, per mode)
#     σ²(c̄) = (1 - exp(-2 c̄ θ)) / θ        (noise variance, per mode)
# a mode is "in transition" (learnable) when c̄·θ ~ O(1). because θ spans orders of
# magnitude on the stiff prior, we sample / step c̄ LOG-uniformly so every mode's
# transition is covered evenly — uniform-in-t (steep schedule) misses the middle,
# uniform-in-c̄ over-weights the loose modes. bounds come from the spectrum:
#     c̄_hi = mix the slowest mode (5/θ_min) ; c̄_lo = barely noise the fastest (0.05/θ_max).

_SNR_HI = 5.0          # c̄_hi·θ_min: slowest mode fully mixed (H = e^-5 ≈ 0.007)
_NOISE_FLOOR = 0.04    # target std of the noise left in the final sample (unit space):
                       # σ(c̄_lo) ≈ sqrt(2·c̄_lo), so c̄_lo = floor²/2. fixed (not tied to
                       # θ_max) so every dataset's final sample is equally clean (~4%),
                       # and c̄_lo stays below the stiffest mode's transition start.


def noise_level_bounds(prior: KroneckerPrior) -> tuple[float, float]:
    """(c̄_lo, c̄_hi): the noise-level range training samples and sampling steps span.
    c̄_hi mixes the slowest mode (5/θ_min); c̄_lo gives a clean ~4% final noise floor."""
    c_lo = _NOISE_FLOOR ** 2 / 2.0
    c_hi = _SNR_HI / float(prior.theta.min())
    return c_lo, c_hi


def c_bar_to_tau(c_bar, c_lo: float, c_hi: float):
    """normalized log-noise-level in [0, 1] — what the denoiser conditions on
    (replaces raw t; even resolution across the noise range). accepts np or torch."""
    if isinstance(c_bar, torch.Tensor):
        log = torch.log
    else:
        log = np.log
        c_bar = np.asarray(c_bar)
    return (log(c_bar) - math.log(c_lo)) / (math.log(c_hi) - math.log(c_lo))


def sample_log_c_bar_torch(
    c_lo: float, c_hi: float, B: int, device, dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """draw B noise levels log-uniformly in [c̄_lo, c̄_hi]. EDM-style alternative to
    GAD's power-law sampler; covers stiff modes evenly but deviates from GAD."""
    u = torch.rand(B, device=device, dtype=dtype, generator=generator)
    return c_lo * (c_hi / c_lo) ** u


# GAD's training noise sampler: s = S·u^power (u uniform), the default we use.
_GAD_SAMPLE_POWER = 0.65    # exp_gasde.py: u.pow(0.65) — biases toward higher noise


def sample_c_bar_powerlaw_torch(
    c_0: float, B: int, device, dtype,
    power: float = _GAD_SAMPLE_POWER,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """draw B noise levels c̄ = c_0·u^power, u ~ U[0,1] — GAD's scheme (exp_gasde.py).
    spans [0, c_0]; power<1 biases toward higher noise. c_0 is the schedule's c̄(T)."""
    u = torch.rand(B, device=device, dtype=dtype, generator=generator)
    return c_0 * u.pow(power)


def reverse_c_bar_grid(c_lo: float, c_hi: float, n_steps: int) -> np.ndarray:
    """descending geometric grid of n_steps+1 noise levels from c̄_hi down to c̄_lo
    (the reverse trajectory; even in log-c̄ so steps are well-spaced for all modes)."""
    return np.geomspace(c_hi, c_lo, n_steps + 1)


def sample_R_t_from_c_bar_torch(
    R0: torch.Tensor,
    c_bar: torch.Tensor,
    prior_t: TorchPriorTensors,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """forward marginal at noise level c̄ (no schedule / no t). R0 (B, N, H),
    c_bar (B,) -> R_t (B, N, H). H and σ² are per-mode functions of c̄ and θ."""
    if R0.ndim != 3:
        raise ValueError(f"R0 must be (B, N, H); got {tuple(R0.shape)}")
    cb = c_bar.view(-1, 1, 1)
    theta = prior_t.theta.unsqueeze(0)
    H = torch.exp(-theta * cb)
    sigma2 = prior_t.sigma2_scale * (1.0 - torch.exp(-2.0 * theta * cb)) / theta
    C0 = torch.einsum("ki,bih,hj->bkj", prior_t.V_S.T, R0, prior_t.V_T)
    Z = torch.randn(R0.shape, generator=generator, device=R0.device, dtype=R0.dtype)
    Ct = H * C0 + sigma2.sqrt() * Z
    return torch.einsum("ik,bkh,hj->bij", prior_t.V_S, Ct, prior_t.V_T.T)


def sample_R_t_from_c_bar(
    R0: np.ndarray, c_bar: float, prior: KroneckerPrior, rng: np.random.Generator,
) -> np.ndarray:
    """numpy twin of sample_R_t_from_c_bar_torch, for tests/diagnostics. R0 (..., N, H)."""
    H = np.exp(-prior.theta * c_bar)
    sigma2 = prior.sigma2_scale * (1.0 - np.exp(-2.0 * prior.theta * c_bar)) / prior.theta
    C0 = _rotate_to_eigenbasis(R0, prior)
    Z = rng.standard_normal(R0.shape)
    return _rotate_from_eigenbasis(H * C0 + np.sqrt(sigma2) * Z, prior)


def _fcps_c_bar_torch(
    t: torch.Tensor, schedule: FCPSSchedule
) -> torch.Tensor:
    """Per-element c̄(t) under FCPSSchedule. t shape: any."""
    u = t / schedule.T
    return schedule.c_min * t + (schedule.c_0 - schedule.c_min * schedule.T) * u.pow(
        schedule.alpha + 1
    )


def fcps_rate_torch(t: torch.Tensor, schedule: FCPSSchedule) -> torch.Tensor:
    """Per-element rate c(t) = c_min + k·u^α (torch). used for GAD's SNR weight
    w = 1/(1 + 2σ²·c(t)). k = (c_0 - c_min·T)(α+1)/T."""
    u = t / schedule.T
    k = (schedule.c_0 - schedule.c_min * schedule.T) * (schedule.alpha + 1) / schedule.T
    return schedule.c_min + k * u.pow(schedule.alpha)


def sample_R_t_given_R0_torch(
    R0: torch.Tensor,
    t: torch.Tensor,
    prior_t: TorchPriorTensors,
    schedule: FCPSSchedule,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """sample R_t ~ p(R_t | R_0) under the forward sde, on-device (torch).

    R0 is (B, N, H), t is (B,), prior_t comes from prior_to_torch; returns R_t
    as (B, N, H) on R0's device.
    """
    if R0.ndim != 3:
        raise ValueError(f"R0 must be (B, N, H); got shape {tuple(R0.shape)}")
    if t.shape != (R0.shape[0],):
        raise ValueError(f"t shape {tuple(t.shape)} must be ({R0.shape[0]},)")

    c_bar_t = _fcps_c_bar_torch(t, schedule)              # (B,)
    c_bar_b = c_bar_t.view(-1, 1, 1)                       # (B, 1, 1)
    theta_b = prior_t.theta.unsqueeze(0)                   # (1, N, H)

    H_t = torch.exp(-theta_b * c_bar_b)                    # (B, N, H)
    sigma2_t = prior_t.sigma2_scale * (1.0 - torch.exp(-2.0 * theta_b * c_bar_b)) / theta_b

    # Rotate to eigenbasis: V_S^T R0 V_T.
    C0 = torch.einsum("ki,bih,hj->bkj", prior_t.V_S.T, R0, prior_t.V_T)
    Z = torch.randn(R0.shape, generator=generator, device=R0.device, dtype=R0.dtype)
    Ct = H_t * C0 + sigma2_t.sqrt() * Z
    # rotate back: V_S C V_T^T. the output index must be `bij`, not `bih` —
    # reusing h would collapse V_T^T and scramble the temporal covariance.
    return torch.einsum("ik,bkh,hj->bij", prior_t.V_S, Ct, prior_t.V_T.T)
