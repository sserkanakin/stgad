"""GWN-style score network for residual diffusion (score-network §14.2).

Predicts r̂₀ (Tweedie / x₀-prediction) given:
  - r_τ: noisy residual at time τ, shape (B, H, N)
  - τ:   diffusion times in [τ_eps, 1], shape (B,)
  - x_history:        (B, 2, H, N)  — normalised speed + time-of-day
  - y_pred_backbone:  (B, H, N)     — deterministic forecast (standardised)

The 4 input channels are concatenated, projected to ``hidden_dim``, then
passed through ``num_blocks`` of gated TCN + graph conv + FiLM
modulation on τ. A final 1×1 conv produces the (B, H, N) prediction.

FiLM (Feature-wise Linear Modulation, Perez et al. 2018) injects τ into
every block by computing per-channel affine parameters ``(γ(τ), β(τ))``
and applying ``x ← γ(τ) ⊙ x + β(τ)``. Parameters live in a small linear
projection from the τ embedding to ``2 · C`` scalars; the γ-half is
initialised to 1, the β-half to 0, so every FiLM layer starts as the
identity and learns τ-dependence from there.

The graph conv uses the symmetric normalised adjacency ``Â = D⁻¹ᐟ² A D⁻¹ᐟ²``
(or whatever the caller passes in) as a single static support. K=1-hop
mixing per layer; stacking blocks gives multi-hop receptive fields.
Adaptive adjacency is deferred to a later iteration.

Shape convention: internally we use channel-first ``(B, C, H, N)``,
matching GWN's diffusion-conv layout.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal τ embedding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """DDPM-style sinusoidal embedding of a scalar τ ∈ [0, 1].

    Maps each τ to a vector of dim ``embed_dim`` whose components are
    sin/cos of geometric-progression frequencies. The ``scale`` factor
    (default 1000) spreads the frequencies across the [0,1] τ range so
    nearby τ values map to nearby embeddings while distant τ values
    decorrelate cleanly.
    """

    def __init__(self, embed_dim: int, scale: float = 1000.0) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        self.embed_dim = embed_dim
        self.scale = scale

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: (B,) → (B, embed_dim)."""
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=tau.dtype) / half
        )
        args = tau.unsqueeze(-1) * self.scale * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# FiLM modulation
# ─────────────────────────────────────────────────────────────────────────────

class FiLM(nn.Module):
    """Feature-wise Linear Modulation. ``x ← γ(cond) ⊙ x + β(cond)``.

    γ and β are per-channel scalars (broadcast over H and N). The two
    projections share a single Linear layer outputting ``2 · channels``
    values; the first half is γ, the second β. Initialised to γ=1, β=0
    so the layer starts as the identity and learns conditioning from
    there.
    """

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.proj = nn.Linear(cond_dim, 2 * channels)
        # Identity initialisation: γ=1, β=0 for any cond input.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:channels].fill_(1.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, N), cond: (B, cond_dim) → (B, C, H, N)."""
        gamma_beta = self.proj(cond)                                  # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)                     # (B, C) each
        gamma = gamma.view(-1, self.channels, 1, 1)
        beta = beta.view(-1, self.channels, 1, 1)
        return gamma * x + beta


# ─────────────────────────────────────────────────────────────────────────────
# Graph convolution (K=1 with static adjacency)
# ─────────────────────────────────────────────────────────────────────────────

class GraphConv(nn.Module):
    """Single-support, K=1 graph convolution.

    ``y = Linear([x, x @ Â])`` where ``Â`` is the static normalised
    adjacency. Concatenating self and neighbour features lets the linear
    layer learn the mix; bias terms are handled by the Linear.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor]) -> torch.Tensor:
        """x: (B, C, H, N), adj: (N, N) or None → (B, C_out, H, N)."""
        if adj is None:
            # No-graph ablation: skip neighbour aggregation, only use self.
            return self.proj(torch.cat([x, x], dim=1))
        neigh = torch.einsum("bchn,nm->bchm", x, adj)
        return self.proj(torch.cat([x, neigh], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Score-net block: gated TCN + graph conv + FiLM + residual
# ─────────────────────────────────────────────────────────────────────────────

class ScoreNetBlock(nn.Module):
    """One residual block of the score network.

    Layout::

        x ─► TCN_filter ──┐
                          tanh ⊙ sigmoid  ─► GCN ─► FiLM(τ) ─► GroupNorm ─► + x
        x ─► TCN_gate ────┘                                                  ▲
                                                                             │
                                                                          residual

    Temporal kernel size 3 with padding 1 preserves the horizon dim
    (H=12 in our setup; shape-preserving lets us stack arbitrary depth).
    """

    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.tcn_filter = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0))
        self.tcn_gate = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0))
        self.gcn = GraphConv(channels, channels)
        self.film = FiLM(cond_dim, channels)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(
        self, x: torch.Tensor, adj: Optional[torch.Tensor], cond: torch.Tensor
    ) -> torch.Tensor:
        h = torch.tanh(self.tcn_filter(x)) * torch.sigmoid(self.tcn_gate(x))
        h = self.gcn(h, adj)
        h = self.film(h, cond)
        h = self.norm(h)
        return h + x


# ─────────────────────────────────────────────────────────────────────────────
# Full score network
# ─────────────────────────────────────────────────────────────────────────────

class ScoreNet(nn.Module):
    """GWN-style score network with FiLM conditioning on τ.

    Predicts r̂₀ from (r_τ, τ, x_history, y_pred_backbone). The Kronecker
    prior's Σ_τ is consumed by the DSM loss outside this module — the
    network learns the *raw* x₀-prediction; the DSM loss handles
    eigenbasis weighting.

    Args:
        num_horizons:     H — number of forecast horizons (12 for our datasets).
        num_sensors:      N — sensor count (207 METR-LA, 325 PEMS-BAY).
        hidden_dim:       Internal channel width. Default 64.
        num_blocks:       Stacked score-net blocks. Default 4.
        tau_embed_dim:    Dim of the τ embedding bottleneck. Default 128.
        adj:              (N, N) normalised adjacency tensor. Stored as a buffer.
                          Pass ``None`` to disable the graph conv (sanity ablations).

    Forward signature:
        score_net(r_tau, tau, x_history, y_pred_backbone) → r_hat_0
        shapes:
            r_tau:           (B, H, N)
            tau:             (B,)
            x_history:       (B, 2, H, N)
            y_pred_backbone: (B, H, N)
            r_hat_0 output:  (B, H, N)
    """

    def __init__(
        self,
        num_horizons: int = 12,
        num_sensors: int = 207,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        tau_embed_dim: int = 128,
        adj: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_horizons = num_horizons
        self.num_sensors = num_sensors
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Register adjacency as a non-trainable buffer so it moves with .to(device).
        if adj is not None:
            if adj.shape != (num_sensors, num_sensors):
                raise ValueError(
                    f"adj shape {tuple(adj.shape)} doesn't match (N, N) = "
                    f"({num_sensors}, {num_sensors})"
                )
            self.register_buffer("adj", adj.float())
        else:
            self.adj = None

        # τ embedding: sinusoidal → 2-layer MLP
        self.tau_embed = nn.Sequential(
            SinusoidalEmbedding(tau_embed_dim),
            nn.Linear(tau_embed_dim, tau_embed_dim),
            nn.SiLU(),
            nn.Linear(tau_embed_dim, tau_embed_dim),
        )

        # Input: 1 (r_τ) + 2 (x_history) + 1 (y_pred_backbone) = 4 channels
        self.input_proj = nn.Conv2d(in_channels=4, out_channels=hidden_dim, kernel_size=1)

        # Stacked blocks
        self.blocks = nn.ModuleList(
            [ScoreNetBlock(hidden_dim, tau_embed_dim) for _ in range(num_blocks)]
        )

        # Output: hidden_dim → 1 channel
        self.output_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        r_tau: torch.Tensor,
        tau: torch.Tensor,
        x_history: torch.Tensor,
        y_pred_backbone: torch.Tensor,
    ) -> torch.Tensor:
        """Predict r̂₀ given the noisy residual + conditioning.

        Shapes:
            r_tau:           (B, H, N)
            tau:             (B,) — diffusion time in [τ_eps, 1]
            x_history:       (B, 2, H, N)
            y_pred_backbone: (B, H, N)

        Returns:
            (B, H, N) — r̂₀ (Tweedie / x₀-prediction)
        """
        self._validate_shapes(r_tau, tau, x_history, y_pred_backbone)

        # τ → conditioning vector
        cond = self.tau_embed(tau)                                    # (B, tau_embed_dim)

        # Stack inputs along channel dim
        x = torch.cat(
            [
                r_tau.unsqueeze(1),                                   # (B, 1, H, N)
                x_history,                                            # (B, 2, H, N)
                y_pred_backbone.unsqueeze(1),                         # (B, 1, H, N)
            ],
            dim=1,
        )                                                              # (B, 4, H, N)

        # Input projection
        x = self.input_proj(x)                                        # (B, hidden_dim, H, N)

        # Stacked blocks
        for block in self.blocks:
            x = block(x, self.adj, cond)

        # Output projection
        x = self.output_proj(x)                                       # (B, 1, H, N)
        return x.squeeze(1)                                           # (B, H, N)

    def _validate_shapes(
        self,
        r_tau: torch.Tensor,
        tau: torch.Tensor,
        x_history: torch.Tensor,
        y_pred_backbone: torch.Tensor,
    ) -> None:
        B, H, N = r_tau.shape[0], self.num_horizons, self.num_sensors
        if r_tau.shape != (B, H, N):
            raise ValueError(f"r_tau shape {tuple(r_tau.shape)} != ({B}, {H}, {N})")
        if tau.shape != (B,):
            raise ValueError(f"tau shape {tuple(tau.shape)} != ({B},)")
        if x_history.shape != (B, 2, H, N):
            raise ValueError(
                f"x_history shape {tuple(x_history.shape)} != ({B}, 2, {H}, {N})"
            )
        if y_pred_backbone.shape != (B, H, N):
            raise ValueError(
                f"y_pred_backbone shape {tuple(y_pred_backbone.shape)} != ({B}, {H}, {N})"
            )

    @property
    def num_parameters(self) -> int:
        """Total trainable-parameter count (for logging)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
