"""Tweedie R₀ denoiser adapted from DiffSTG's UGnet.

We wrap DiffSTG's `UGnet` with a clean signature for our setup:

    R̂₀ = denoiser(R_t, t, Y_hat)

  R_t   : (B, N, H) residual diffusion state at time t
  t     : (B,) continuous diffusion time ∈ [0, schedule.T]
  Y_hat : (B, N, H) deterministic backbone forecast (the conditioning we have)

UGnet expects:
  x : (B, F, V, T)
  t : (B,) integer-like timesteps (used in a sinusoidal embedding)
  c = (x_masked, pos_w, pos_d)   pos_w/pos_d unused for our setup

so we:
  - unsqueeze the single feature channel: R_t → (B, 1, N, H)
  - feed Y_hat as x_masked (same shape after unsqueeze)
  - rescale continuous t to UGnet's discrete-step embedding range (×1000)
  - slice the last H of UGnet's output (which spans T_p+T_h = 2H by design;
    the trailing block corresponds to the "forecast" half).

Notes:
  - We import UGnet from third_party/DiffSTG (see THIRD_PARTY.md). The path
    insertion is local to this module.
  - Output is R̂₀ (Tweedie target), not ε. UGnet's architecture is reused
    unchanged; the difference is purely in the training loss target.
  - X history conditioning is not used yet; it can be added later by widening
    the channel dim of x_masked (F → 1 + extra_cond_channels).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
_DIFFSTG_PATH = _ROOT / "third_party" / "DiffSTG"
if str(_DIFFSTG_PATH) not in sys.path:
    sys.path.insert(0, str(_DIFFSTG_PATH))

import easydict  # noqa: E402

from algorithm.diffstg.ugnet import UGnet as _UGnet  # noqa: E402


# Scale factor mapping continuous t ∈ [0, 1] to UGnet's discrete-step range.
# UGnet's sinusoidal embedding (from DDPM) was designed for integer t in [0, N]
# with N≈1000. We multiply by this scale so the cos/sin frequencies cover the
# same range; the embedding does not require integer values.
T_EMBED_SCALE = 1000.0


def make_ugnet_config(
    N: int,
    H: int,
    A: np.ndarray,
    d_h: int = 32,
    channel_multipliers: tuple[int, ...] = (1, 2),
    n_blocks: int = 2,
    device: str = "cpu",
) -> easydict.EasyDict:
    """Build the EasyDict UGnet expects."""
    return easydict.EasyDict({
        "d_h": d_h,
        # UGnet's internal T = T_p + T_h is the per-input time length (before
        # the concat with x_masked). We're feeding only the forecast horizon,
        # so set T_p = H, T_h = 0; the concat with x_masked then produces 2H
        # along time, and UGnet's final Linear(2T, T) = Linear(2H, H) restores
        # the horizon dim.
        "T_p": H,
        "T_h": 0,
        "F": 1,
        "V": N,
        "A": A.astype(np.float32),
        "channel_multipliers": list(channel_multipliers),
        "n_blocks": n_blocks,
        "device": device,
        "supports_len": 2,
    })


class R0Denoiser(nn.Module):
    """Tweedie R₀ predictor for the GAD diffusion SDE.

    Parameters
    ----------
    N : int
        Number of sensors.
    H : int
        Forecast horizon.
    adjacency : (N, N) array
        Raw (asymmetric for road-graph datasets) adjacency matrix used by
        UGnet's spatial graph convolution. Same convention as GWN backbone.
    d_h : int
        Hidden dim of UGnet residual blocks. Defaults to 32 (DiffSTG default).
    channel_multipliers, n_blocks : architecture knobs.
    device : str
        Where UGnet places its adjacency supports.
    """

    def __init__(
        self,
        N: int,
        H: int,
        adjacency: np.ndarray,
        d_h: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2),
        n_blocks: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if adjacency.shape != (N, N):
            raise ValueError(
                f"adjacency must be ({N}, {N}); got {adjacency.shape}"
            )
        self.N = N
        self.H = H
        cfg = make_ugnet_config(
            N=N, H=H, A=adjacency, d_h=d_h,
            channel_multipliers=channel_multipliers, n_blocks=n_blocks,
            device=device,
        )
        self.ugnet = _UGnet(cfg)

    def forward(
        self,
        R_t: torch.Tensor,
        t: torch.Tensor,
        Y_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Predict R₀ given R_t at diffusion time t, conditioned on Y_hat.

        Shapes:
          R_t   : (B, N, H)
          t     : (B,) — continuous diffusion time
          Y_hat : (B, N, H)
          returns R̂_0 : (B, N, H)
        """
        if R_t.shape[-2:] != (self.N, self.H):
            raise ValueError(
                f"R_t last two dims must be ({self.N}, {self.H}); got {R_t.shape}"
            )
        if Y_hat.shape != R_t.shape:
            raise ValueError(
                f"Y_hat shape {Y_hat.shape} != R_t shape {R_t.shape}"
            )
        if t.shape[0] != R_t.shape[0]:
            raise ValueError(
                f"t batch dim {t.shape[0]} != R_t batch dim {R_t.shape[0]}"
            )

        x = R_t.unsqueeze(1)          # (B, 1, N, H)
        x_masked = Y_hat.unsqueeze(1)  # (B, 1, N, H)
        # pos_w, pos_d are unused inside UGnet's forward; pass 0-tensors of any
        # shape — they're ignored.
        c = (x_masked, torch.zeros(1, device=R_t.device), torch.zeros(1, device=R_t.device))

        t_scaled = t * T_EMBED_SCALE
        out = self.ugnet(x, t_scaled, c)   # (B, 1, N, H)
        return out.squeeze(1)               # (B, N, H)


def num_parameters(module: nn.Module) -> int:
    """Convenience helper for logging param count."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
