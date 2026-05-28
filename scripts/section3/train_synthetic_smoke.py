"""Stage C.5 synthetic-data smoke test for the diffusion training pipeline.

The cached residuals from extract_residuals carry every real-data quirk
(missingness mask, non-Gaussianity, conditioning correlations). To check the
forward+denoiser+loss plumbing in isolation, we draw R_0 directly from the
locked Kronecker prior (so the data distribution exactly matches the prior
used by the forward SDE) and train the denoiser to recover them.

Expected: loss drops sharply within the first ~100 iterations, since the
optimal R̂_0 = E[R_0 | R_t] is a known linear function of R_t under the
Gaussian joint, and the U-Net can fit this.

Usage:
    python scripts/section3/train_synthetic_smoke.py --dataset metrla
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.data.adjacency import load_adjacency
from src.diffusion.denoiser import R0Denoiser, num_parameters
from src.diffusion.forward_fcps import (
    FCPSSchedule,
    prior_to_torch,
    sample_R_t_given_R0_torch,
    sample_prior as np_sample_prior,
)
from src.diffusion.prior import load_prior


def main(dataset: str, n_samples: int, n_iters: int, batch_size: int,
         d_h: int, lr: float) -> int:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[synthetic smoke] device={device} dataset={dataset}", flush=True)

    prior = load_prior(dataset)
    prior_t = prior_to_torch(prior, device=device, dtype=torch.float32)
    schedule = FCPSSchedule()
    print(
        f"[synthetic smoke] prior N={prior.N} H={prior.H} γ={prior.gamma:.4f}",
        flush=True,
    )

    # Generate synthetic R_0 ~ N(0, Σ_T ⊗ Σ_S) via the numpy prior sampler.
    rng_np = np.random.default_rng(0)
    R0_np = np_sample_prior((n_samples,), prior, rng_np).astype(np.float32)
    R0 = torch.from_numpy(R0_np).to(device)
    Y_hat = torch.zeros_like(R0)  # no conditioning in synthetic test
    print(f"[synthetic smoke] R0 shape={tuple(R0.shape)}  "
          f"empirical std={R0.std().item():.3f}", flush=True)

    A = load_adjacency(dataset)
    denoiser = R0Denoiser(
        N=prior.N, H=prior.H, adjacency=A,
        d_h=d_h, channel_multipliers=(1, 2), n_blocks=2,
        device=str(device),
    ).to(device)
    print(f"[synthetic smoke] denoiser params: {num_parameters(denoiser):,}", flush=True)

    optim = torch.optim.Adam(denoiser.parameters(), lr=lr)

    # Baseline loss: E_t[||R_0 - 0||²] when denoiser predicts 0. We'll compare
    # the trained loss against this baseline.
    baseline_loss = float((R0 ** 2).mean())
    print(f"[synthetic smoke] zero-predictor MSE baseline: {baseline_loss:.4f}",
          flush=True)

    losses: list[float] = []
    t0 = time.time()
    g = torch.Generator(device=device).manual_seed(0) if device.type != "mps" else None
    denoiser.train()
    for it in range(n_iters):
        idx = torch.randint(0, n_samples, (batch_size,), device=device)
        R0_batch = R0[idx]
        Y_hat_batch = Y_hat[idx]
        t = (torch.rand(batch_size, device=device, generator=g) * (schedule.T - 1e-3)
             + 1e-3)
        R_t = sample_R_t_given_R0_torch(R0_batch, t, prior_t, schedule, generator=g)
        R0_hat = denoiser(R_t, t, Y_hat_batch)

        loss = ((R0_hat - R0_batch) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(denoiser.parameters(), 5.0)
        optim.step()

        losses.append(loss.detach().item())
        if (it + 1) % max(n_iters // 10, 1) == 0:
            print(
                f"  iter {it+1:4d}/{n_iters}  loss={loss.item():.4f}  "
                f"({time.time()-t0:.1f}s)",
                flush=True,
            )

    final = float(np.mean(losses[-10:]))
    initial = float(np.mean(losses[:10]))
    reduction = (initial - final) / max(initial, 1e-9)
    print(
        f"\n[synthetic smoke] initial(10)={initial:.4f}  final(10)={final:.4f}  "
        f"reduction={reduction:.1%}  baseline={baseline_loss:.4f}",
        flush=True,
    )

    # Wiring smoke (not a convergence benchmark): pass if final loss is below
    # the zero-predictor baseline and we saw any measurable reduction. Actual
    # convergence is exercised in full training on DAIC with longer schedules
    # and a wider network.
    if np.isnan(final) or final >= baseline_loss:
        print("[synthetic smoke] FAIL — loss did not decrease below baseline", flush=True)
        return 1
    if reduction < 0.05:
        print(
            f"[synthetic smoke] WARN — reduction {reduction:.1%} small "
            f"(< 5%); consider increasing n-iters or d-h. Wiring is OK.",
            flush=True,
        )
    print("[synthetic smoke] PASS — denoiser learned a measurable predictor", flush=True)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems08f"), default="metrla")
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d-h", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    sys.exit(main(
        args.dataset, args.n_samples, args.n_iters,
        args.batch_size, args.d_h, args.lr,
    ))
