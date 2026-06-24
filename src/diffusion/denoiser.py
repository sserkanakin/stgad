"""denoiser that predicts the clean residual R0, built on diffstg's ugnet.

we wrap ugnet with a simple signature:

    R0_hat = denoiser(R_t, t, Y_hat)

    R_t   (B, N, H)  noisy residual at diffusion time t
    t     (B,)       continuous diffusion time in [0, schedule.T]
    Y_hat (B, N, H)  backbone forecast, used as conditioning

ugnet itself wants x as (B, F, V, T), an integer-ish step t for its sinusoidal
embedding, and a tuple c = (x_masked, pos_w, pos_d). so we add a feature axis,
pass Y_hat as x_masked, scale t into ugnet's step range, and ignore pos_w/pos_d.

ugnet is vendored under third_party/DiffSTG (see git history / THIRD_PARTY notes).
we predict R0 rather than the noise eps; the network is otherwise unchanged.
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

# ugnet's sinusoidal step embedding was made for integer steps up to ~1000.
# our t is continuous in [0, 1], so we scale it into that range.
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
    """build the easydict config ugnet expects."""
    return easydict.EasyDict({
        "d_h": d_h,
        # we feed only the H forecast steps, so T_p=H, T_h=0. ugnet concatenates
        # x and x_masked along time (-> 2H) and projects back to H at the end.
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
    """predicts the clean residual R0 for the gad diffusion sde.

    N, H are the sensor count and forecast horizon. adjacency is the (N, N)
    matrix ugnet uses for its spatial graph conv (same one as the gwn backbone).
    d_h / channel_multipliers / n_blocks are ugnet architecture knobs; device is
    where ugnet places its adjacency supports.
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
        """predict R0 from R_t at time t, conditioned on Y_hat.

        R_t, Y_hat and the returned R0_hat are all (B, N, H); t is (B,).
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
        # pos_w / pos_d are ignored by ugnet's forward; pass dummy zeros.
        c = (x_masked, torch.zeros(1, device=R_t.device), torch.zeros(1, device=R_t.device))

        t_scaled = t * T_EMBED_SCALE
        out = self.ugnet(x, t_scaled, c)   # (B, 1, N, H)
        return out.squeeze(1)               # (B, N, H)


class SignalDenoiser(nn.Module):
    """history-conditioned denoiser for the pure-signal arm (DiffSTG-style).

    diffuses only the FUTURE signal Y (H steps) but conditions on the HISTORY window
    (T_h steps), exactly as DiffSTG conditions on x_masked = [history, zeros]:

        x        = [history, Y_t]      (history clean, future noised)
        x_masked = [history, zeros]    (history known, future masked)
        out      = ugnet(x, t, x_masked)[..., -H:]   -> Y0_hat for the future

    signature matches R0Denoiser — denoiser(Y_t, t, history) — so the reverse sampler
    and training loop reuse it unchanged; only the conditioning tensor differs
    (history (B, N, T_h) instead of a forecast (B, N, H))."""

    def __init__(
        self,
        N: int,
        H: int,
        T_h: int,
        adjacency: np.ndarray,
        d_h: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2),
        n_blocks: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if adjacency.shape != (N, N):
            raise ValueError(f"adjacency must be ({N}, {N}); got {adjacency.shape}")
        self.N, self.H, self.T_h = N, H, T_h
        cfg = make_ugnet_config(
            N=N, H=H, A=adjacency, d_h=d_h,
            channel_multipliers=channel_multipliers, n_blocks=n_blocks, device=device,
        )
        cfg.T_h = T_h
        cfg.T_p = H                         # ugnet builds for T = T_h + H (the cat axis)
        self.ugnet = _UGnet(cfg)

    def forward(self, Y_t: torch.Tensor, t: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """predict the clean future signal Y0 from the noised future Y_t (B, N, H),
        conditioned on the history (B, N, T_h). t is (B,)."""
        if Y_t.shape[-2:] != (self.N, self.H):
            raise ValueError(f"Y_t last two dims must be ({self.N}, {self.H}); got {Y_t.shape}")
        if history.shape[-2:] != (self.N, self.T_h):
            raise ValueError(f"history last two dims must be ({self.N}, {self.T_h}); got {history.shape}")
        x = torch.cat([history, Y_t], dim=-1).unsqueeze(1)              # (B,1,N,T_h+H)
        zeros_future = torch.zeros_like(Y_t)
        x_masked = torch.cat([history, zeros_future], dim=-1).unsqueeze(1)  # (B,1,N,T_h+H)
        c = (x_masked, torch.zeros(1, device=Y_t.device), torch.zeros(1, device=Y_t.device))
        out = self.ugnet(x, t * T_EMBED_SCALE, c)                       # (B,1,N,T_h+H)
        return out.squeeze(1)[..., -self.H:]                            # (B,N,H) future only


def num_parameters(module: nn.Module) -> int:
    """Convenience helper for logging param count."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
