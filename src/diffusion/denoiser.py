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
        T_h: int = 0,
    ) -> None:
        super().__init__()
        if adjacency.shape != (N, N):
            raise ValueError(
                f"adjacency must be ({N}, {N}); got {adjacency.shape}"
            )
        self.N = N
        self.H = H
        # T_h > 0: past-residual conditioning. the conditioning tensor is then
        # [R_past | Y_hat_z] along time, (B, N, T_h+H); ugnet is built for
        # T = T_h + T_p exactly like the DiffSTG signal arm (T_h=12/T_p=12).
        self.T_h = int(T_h)
        cfg = make_ugnet_config(
            N=N, H=H, A=adjacency, d_h=d_h,
            channel_multipliers=channel_multipliers, n_blocks=n_blocks,
            device=device,
        )
        if self.T_h:
            cfg.T_h = self.T_h
            cfg.T_p = H
        self.ugnet = _UGnet(cfg)

    def forward(
        self,
        R_t: torch.Tensor,
        t: torch.Tensor,
        Y_hat: torch.Tensor,
    ) -> torch.Tensor:
        """predict R0 from R_t at time t, conditioned on Y_hat.

        R_t and the returned R0_hat are (B, N, H); t is (B,). Y_hat is (B, N, H),
        or (B, N, T_h+H) = [R_past | Y_hat_z] when built with T_h > 0.
        """
        if R_t.shape[-2:] != (self.N, self.H):
            raise ValueError(
                f"R_t last two dims must be ({self.N}, {self.H}); got {R_t.shape}"
            )
        if Y_hat.shape[-1] != self.T_h + self.H:
            raise ValueError(
                f"conditioning last dim must be T_h+H={self.T_h + self.H}; got {Y_hat.shape}"
            )
        if t.shape[0] != R_t.shape[0]:
            raise ValueError(
                f"t batch dim {t.shape[0]} != R_t batch dim {R_t.shape[0]}"
            )

        if self.T_h:
            # diffstg conditioning: x = [known past | noisy future], x_masked =
            # [known past | forecast]. past = the previous window's residuals.
            R_past = Y_hat[..., : self.T_h]
            x = torch.cat([R_past, R_t], dim=-1).unsqueeze(1)       # (B, 1, N, T_h+H)
            x_masked = Y_hat.unsqueeze(1)                            # (B, 1, N, T_h+H)
        else:
            x = R_t.unsqueeze(1)           # (B, 1, N, H)
            x_masked = Y_hat.unsqueeze(1)  # (B, 1, N, H)
        # pos_w / pos_d are ignored by ugnet's forward; pass dummy zeros.
        c = (x_masked, torch.zeros(1, device=R_t.device), torch.zeros(1, device=R_t.device))

        t_scaled = t * T_EMBED_SCALE
        out = self.ugnet(x, t_scaled, c)   # (B, 1, N, T_h+H)
        out = out.squeeze(1)
        return out[..., -self.H:] if self.T_h else out              # (B, N, H)


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


class GCNNDenoiser(nn.Module):
    """GAD's low-capacity denoiser (paper Eq. 16): a cascade of K-hop graph filters
    with ReLUs, adapted to (B, N, H) fields with forecast conditioning.

    DELIBERATELY small (~50-80k params vs UGnet's ~1M): with a weak denoiser the
    corruption prior must carry the structure it cannot learn — the capacity x prior
    interaction PriorGrad documents ("robustness to smaller network capacity") and
    the mechanism our residual study identified (an expressive conditional denoiser
    bypasses the prior). per layer:
        x <- ReLU( FiLM_tau( sum_{k=0..K} (S^k x) W_k  +  depthwise conv1d_H(x) ) )
    with S = D^{-1/2}(A+I)D^{-1/2} (ugnet's gcn_self operator), W_k 1x1 channel mixes.
    """

    def __init__(self, N, H, adjacency, K=3, n_layers=4, hidden=32,
                 device="cpu", T_h=0):
        super().__init__()
        if T_h:
            raise ValueError("GCNNDenoiser supports base conditioning only (T_h=0)")
        self.N, self.H, self.K = N, H, K
        A = np.asarray(adjacency, dtype=np.float64).copy()
        np.fill_diagonal(A, 0.0)
        A = A + np.eye(N)                                     # self loops (gcn_self)
        d = A.sum(1)
        S = A / np.sqrt(np.outer(d, d))
        Sp = [np.eye(N)]
        for _ in range(K):
            Sp.append(Sp[-1] @ S)
        self.register_buffer("S_pow", torch.tensor(np.stack(Sp), dtype=torch.float32))
        cs = [2] + [hidden] * n_layers                        # in: [R_t | Y_hat]
        self.gmix = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(cs[l], cs[l + 1], 1) for _ in range(K + 1)])
            for l in range(n_layers)])
        self.tconv = nn.ModuleList([
            nn.Conv2d(cs[l + 1], cs[l + 1], (1, 3), padding=(0, 1), groups=cs[l + 1])
            for l in range(n_layers)])
        self.film = nn.ModuleList([nn.Linear(64, 2 * cs[l + 1]) for l in range(n_layers)])
        self.head = nn.Conv2d(hidden, 1, 1)

    def _tau_embed(self, t):
        half = 32
        freqs = torch.exp(torch.linspace(0, 6, half, device=t.device))
        ang = t.view(-1, 1) * freqs.view(1, -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)      # (B, 64)

    def forward(self, R_t, t, Y_hat):
        if R_t.shape[-2:] != (self.N, self.H):
            raise ValueError(f"R_t must be (B,{self.N},{self.H}); got {tuple(R_t.shape)}")
        x = torch.stack([R_t, Y_hat], dim=1)                            # (B, 2, N, H)
        emb = self._tau_embed(t * 10.0)                                 # t in [0,1]
        for gm, tc, fl in zip(self.gmix, self.tconv, self.film):
            acc = 0
            for k in range(self.K + 1):
                xk = torch.einsum("ij,bcjh->bcih", self.S_pow[k], x)
                acc = acc + gm[k](xk)
            acc = acc + tc(acc)
            sc, sh = fl(emb).chunk(2, dim=1)
            acc = acc * (1 + sc.unsqueeze(-1).unsqueeze(-1)) + sh.unsqueeze(-1).unsqueeze(-1)
            x = torch.relu(acc)
        return self.head(x).squeeze(1)                                  # (B, N, H)
