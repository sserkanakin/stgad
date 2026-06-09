"""two-sample distribution distances for the W1 generative-fidelity check.

we compare a cloud of generated residual vectors against the true val residual
vectors, per horizon, in node-space (R^N): does the full generative chain
produce realistic residuals (not just locally undo a small perturbation)?

both distances are unpaired (cloud-vs-cloud) and translation aware, so they are
computed directly on residuals (= forecast errors, the backbone term cancels).
"""
from __future__ import annotations

from typing import Optional

import torch

_QGRID = 256  # quantile levels for the 1d wasserstein integral


def sliced_wasserstein1(
    gen: torch.Tensor,
    true: torch.Tensor,
    n_proj: int,
    generator: Optional[torch.Generator] = None,
) -> float:
    """sliced wasserstein-1 between two clouds in R^D.

    gen is (M, D), true is (M2, D) (sizes may differ). we draw n_proj random
    unit directions, project both clouds onto each, and average the 1d
    wasserstein-1 distance over directions. the 1d W1 is the integral of
    |F_gen^{-1} - F_true^{-1}|, approximated on a uniform quantile grid (this
    handles unequal sample sizes cleanly).
    """
    D = gen.shape[1]
    theta = torch.randn(D, n_proj, generator=generator, device=gen.device, dtype=gen.dtype)
    theta = theta / theta.norm(dim=0, keepdim=True).clamp(min=1e-12)
    pg = gen @ theta                                   # (M, P)
    pt = true @ theta                                  # (M2, P)
    qs = torch.linspace(0.0, 1.0, _QGRID, device=gen.device, dtype=gen.dtype)
    qg = torch.quantile(pg, qs, dim=0)                 # (Q, P)
    qt = torch.quantile(pt, qs, dim=0)
    return float((qg - qt).abs().mean())


def energy_distance(gen: torch.Tensor, true: torch.Tensor) -> float:
    """energy distance between two clouds in R^D.

    ED = sqrt( 2·E||X-Y|| - E||X-X'|| - E||Y-Y'|| ), with euclidean norms over
    the R^D vectors. gen (M, D), true (M2, D). pairwise, so keep M, M2 modest.
    """
    d_gt = torch.cdist(gen, true).mean()
    d_gg = torch.cdist(gen, gen).mean()
    d_tt = torch.cdist(true, true).mean()
    ed2 = (2.0 * d_gt - d_gg - d_tt).clamp(min=0.0)
    return float(ed2.sqrt())
