"""Ensemble samplers producing ``Y_sample = Ŷ + R_sample`` for evaluation.

All samplers expose the same interface::

    samples = sampler.sample(y_hat)        # y_hat (b, N, H) -> (K, b, N, H)

so the streaming driver in the eval scripts is sampler-agnostic. The diffusion
sampler runs the trained denoiser on-device; the two baselines are analytic and
cheap (CPU).

Baselines
---------
PriorOnlySampler
    Ŷ + a draw from the structured Kronecker prior N(0, Σ_T⊗Σ_S), *no denoiser*.
    Isolates how much of the predictive distribution comes from the learned
    diffusion model versus the prior covariance alone.
PerHorizonGaussianSampler
    Ŷ + independent N(0, δ_h) noise per cell (δ_h = MAD-based per-horizon
    variance from the prior fit). The simplest possible uncertainty model — no
    spatial or temporal correlation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.diffusion.forward_fcps import FCPSSchedule, sample_prior
from src.diffusion.prior import KroneckerPrior
from src.diffusion.reverse import ReverseTrace, reverse_sample


class DiffusionSampler:
    """Draw K residual samples from the trained reverse diffusion process.

    K replicas of each window are sampled in one ``reverse_sample`` call (the
    replicas get independent prior starts and per-step noise, so they are genuine
    Monte-Carlo samples).
    """

    def __init__(
        self,
        denoiser,
        prior: KroneckerPrior,
        schedule: FCPSSchedule,
        *,
        n_samples: int = 100,
        n_steps: int = 100,
        eta: float = 1.0,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ) -> None:
        self.denoiser = denoiser
        self.prior = prior
        self.schedule = schedule
        self.K = n_samples
        self.n_steps = n_steps
        self.eta = eta
        self.device = device or next(denoiser.parameters()).device
        self.seed = seed

    @torch.no_grad()
    def sample(self, y_hat: np.ndarray, return_trace: bool = False):
        """y_hat (b, N, H) -> samples (K, b, N, H). Optionally a ReverseTrace."""
        b, N, H = y_hat.shape
        y_t = torch.as_tensor(y_hat, dtype=torch.float32, device=self.device)
        # Replicate to (K*b, N, H): rows [0..b) = sample 0, [b..2b) = sample 1, ...
        y_rep = y_t.repeat(self.K, 1, 1)
        # Generator must live on the sampling device for randn; CPU gen is fine
        # for CPU/MPS, CUDA needs a CUDA generator.
        gen = self._device_generator()
        R, trace = reverse_sample(
            prior=self.prior,
            schedule=self.schedule,
            denoiser=self.denoiser,
            Y_hat=y_rep,
            n_steps=self.n_steps,
            eta=self.eta,
            return_trace=return_trace,
            generator=gen,
        )
        Y_sample = (y_rep + R).reshape(self.K, b, N, H)
        out = Y_sample.detach().cpu().numpy()
        return (out, trace) if return_trace else out

    def _device_generator(self):
        """A seeded generator on the sampling device.

        ``torch.randn(device=d, generator=g)`` requires ``g`` to live on device
        ``d``. CUDA/CPU generators are reliable; on MPS (local-dev only) generator
        support is patchy, so we seed the global RNG and return None there.
        """
        if self.device.type in ("cuda", "cpu"):
            return torch.Generator(device=self.device.type).manual_seed(self.seed)
        torch.manual_seed(self.seed)   # mps / other: rely on global RNG
        return None


class PriorOnlySampler:
    """Ŷ + draws from the Kronecker prior (no learned model)."""

    def __init__(self, prior: KroneckerPrior, *, n_samples: int = 100, seed: int = 0) -> None:
        self.prior = prior
        self.K = n_samples
        self._rng = np.random.default_rng(seed)

    def sample(self, y_hat: np.ndarray) -> np.ndarray:
        b = y_hat.shape[0]
        R = sample_prior((self.K, b), self.prior, self._rng)   # (K, b, N, H)
        return y_hat[None] + R


class PerHorizonGaussianSampler:
    """Ŷ + independent N(0, δ_h) noise; no spatial/temporal correlation."""

    def __init__(self, delta_h: np.ndarray, *, n_samples: int = 100, seed: int = 0) -> None:
        self.std = np.sqrt(np.asarray(delta_h, dtype=np.float64))   # (H,)
        self.K = n_samples
        self._rng = np.random.default_rng(seed)

    def sample(self, y_hat: np.ndarray) -> np.ndarray:
        b, N, H = y_hat.shape
        noise = self._rng.standard_normal((self.K, b, N, H)) * self.std[None, None, None, :]
        return y_hat[None] + noise
