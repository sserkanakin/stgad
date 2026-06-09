"""train the diffusion denoiser (predicts the clean residual R0).

inputs are the cached residuals (src.utils.paths.residual_npz) and the kronecker
prior from src.diffusion.prior.load_prior.

each step:
  1. take a batch of (R0, Y_hat, mask) from the residual cache
  2. draw t ~ Uniform[t_eps, T]  (t_eps avoids the singular t=0 where σ²_t -> 0)
  3. forward-sample R_t = sample_R_t_given_R0_torch(R0, t, ...)
  4. predict R0_hat = denoiser(R_t, t, Y_hat)
  5. masked mse between R0_hat and R0, then backprop

usage:
    python scripts/train_diffusion.py --dataset metrla --seed 0
    python scripts/train_diffusion.py --dataset metrla --seed 0 --smoke
    python scripts/train_diffusion.py --dataset metrla --seed 0 --fresh
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.adjacency import load_adjacency
from src.data.residual_dataset import ResidualDataset, residual_path
from src.diffusion.denoiser import R0Denoiser, num_parameters
from src.diffusion.forward_fcps import (
    FCPSSchedule,
    prior_to_torch,
    sample_R_t_given_R0_torch,
)
from src.diffusion.prior import load_prior
from src.diffusion.reverse import reverse_sample
from src.eval.distribution import energy_distance, sliced_wasserstein1
from src.eval.intervals import masked_winkler_per_horizon
from src.utils import paths
from src.utils.checkpoint import (
    find_latest_checkpoint,
    get_rng_states,
    load_checkpoint,
    restore_rng_states,
    save_checkpoint,
)
from src.utils.config import hash_config
from src.utils.seed import set_seed
from src.utils.wandb_setup import init_wandb

_BACKBONE = "gwn"
# only the kronecker prior is wired up so far; the isotropic-baseline arm
# comes with the methodology change and will turn this into a cli flag.
_PRIOR = "kronecker"

# Config keys whose value should NOT contribute to the hash — varying them
# should NOT trigger a fresh wandb run / fresh checkpoint dir.
_CFG_HASH_EXCLUDE = {"seed", "epochs", "batch_size", "lr", "weight_decay", "patience"}


# ─── Loss / metrics ──────────────────────────────────────────────────────────


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean squared error. pred/target/mask all (B, N, H)."""
    err2 = (pred - target).pow(2) * mask.float()
    return err2.sum() / (mask.float().sum() + 1e-8)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean absolute error. pred/target/mask all (B, N, H)."""
    err = (pred - target).abs() * mask.float()
    return err.sum() / (mask.float().sum() + 1e-8)


def masked_mae_per_h(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked mean absolute error per horizon. inputs (B, N, H) -> (H,)."""
    err = (pred - target).abs() * mask.float()
    num = err.sum(dim=(0, 1))
    den = mask.float().sum(dim=(0, 1)).clamp(min=1.0)
    return num / den


# ─── Training step ───────────────────────────────────────────────────────────


def train_step(
    batch: dict[str, torch.Tensor],
    *,
    denoiser: R0Denoiser,
    prior_t,
    schedule: FCPSSchedule,
    device: torch.device,
    t_eps: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    R0 = batch["R0"].to(device, non_blocking=True)         # (B, N, H)
    Y_hat = batch["Y_hat"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    B = R0.shape[0]

    t = torch.rand(B, device=device, generator=generator) * (schedule.T - t_eps) + t_eps
    R_t = sample_R_t_given_R0_torch(R0, t, prior_t, schedule, generator=generator)

    R0_hat = denoiser(R_t, t, Y_hat)
    return masked_mse(R0_hat, R0, mask)


@torch.no_grad()
def validate(
    loader: DataLoader,
    *,
    denoiser: R0Denoiser,
    prior_t,
    schedule: FCPSSchedule,
    device: torch.device,
    t_eps: float,
    n_t_samples: int = 4,
) -> float:
    """Average validation loss over the loader. For each batch we average the
    MSE over `n_t_samples` random t draws to reduce noise from the t sampling.
    """
    denoiser.eval()
    total = 0.0
    n = 0
    for batch in loader:
        R0 = batch["R0"].to(device, non_blocking=True)
        Y_hat = batch["Y_hat"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        B = R0.shape[0]
        batch_loss = 0.0
        for _ in range(n_t_samples):
            t = torch.rand(B, device=device) * (schedule.T - t_eps) + t_eps
            R_t = sample_R_t_given_R0_torch(R0, t, prior_t, schedule)
            R0_hat = denoiser(R_t, t, Y_hat)
            batch_loss += float(masked_mse(R0_hat, R0, mask))
        total += batch_loss / n_t_samples
        n += 1
    denoiser.train()
    return total / max(n, 1)


# ─── End-to-end reconstruction / generation eval ─────────────────────────────


_HL_HORIZONS = (3, 6, 12)  # horizons highlighted in the printout (1-indexed)


@torch.no_grad()
def reconstruction_eval(
    batch: dict[str, torch.Tensor],
    *,
    denoiser: R0Denoiser,
    prior,
    prior_t,
    schedule: FCPSSchedule,
    device: torch.device,
    n_steps: int,
    n_samples: int,
    alpha: float,
    log_every: int,
    generator: Optional[torch.Generator],
) -> dict:
    """end-to-end check that the reverse process actually works, on one val batch.

    all metrics are in mph: forecast_err = (Y_hat + R0_hat) - (Y_hat + R0_true)
    = R0_hat - R0_true, so the backbone term cancels and we score residuals.

    two diagnostics:
      1. reconstruction curve - forward-noise the true R0 to t≈T (almost fully
         noised), then reverse-denoise back to t≈0 with the deterministic (eta=0)
         sampler, logging masked mae/mse every `log_every` steps. mae should drop
         as t -> 0 if the denoiser is inverting the forward process.
      2. generative metric - draw `n_samples` prior-seeded samples, report mae/mse
         of the sample mean and the winkler interval score at level alpha.
    """
    denoiser.eval()
    R0 = batch["R0"].to(device)
    Y_hat = batch["Y_hat"].to(device)
    mask = batch["mask"].to(device)
    B = R0.shape[0]

    # ── 1. reconstruction curve from a noised true R0 ──────────────────────
    t_top = torch.full((B,), schedule.T, device=device, dtype=R0.dtype)
    R_T = sample_R_t_given_R0_torch(R0, t_top, prior_t, schedule, generator=generator)

    curve: list[tuple[float, float, float]] = []  # (t, mae, mse) along the trajectory

    def _record(i: int, t_cur: float, R0_hat: torch.Tensor) -> None:
        if i % log_every == 0:
            curve.append((t_cur, float(masked_mae(R0_hat, R0, mask)),
                          float(masked_mse(R0_hat, R0, mask))))

    R0_recon, _ = reverse_sample(
        prior, schedule, denoiser, Y_hat, n_steps=n_steps,
        eta=0.0, R_T=R_T, generator=generator, step_callback=_record,
    )
    curve.append((0.0, float(masked_mae(R0_recon, R0, mask)),
                  float(masked_mse(R0_recon, R0, mask))))
    recon_mae = curve[-1][1]
    recon_mse = curve[-1][2]

    # ── 2. generative samples from the prior (one batched reverse pass) ─────
    Y_hat_rep = Y_hat.repeat(n_samples, 1, 1)                      # (K*B, N, H)
    samples_flat, _ = reverse_sample(
        prior, schedule, denoiser, Y_hat_rep, n_steps=n_steps,
        eta=0.0, R_T=None, generator=generator,
    )
    samples = samples_flat.view(n_samples, B, prior.N, prior.H)    # (K, B, N, H)

    sample_mean = samples.mean(dim=0)                              # (B, N, H)
    gen_mae_h = masked_mae_per_h(sample_mean, R0, mask)            # (H,)
    gen_mae = float(masked_mae(sample_mean, R0, mask))
    gen_mse = float(masked_mse(sample_mean, R0, mask))

    lo = torch.quantile(samples, alpha / 2.0, dim=0)              # (B, N, H)
    hi = torch.quantile(samples, 1.0 - alpha / 2.0, dim=0)
    wi_h = masked_winkler_per_horizon(R0, lo, hi, mask, alpha)    # (H,)
    wi = float(wi_h.mean())
    # empirical coverage of the interval (should be ~ 1 - alpha if calibrated)
    inside = ((R0 >= lo) & (R0 <= hi) & mask).float().sum()
    coverage = float(inside / mask.float().sum().clamp(min=1.0))

    denoiser.train()
    return {
        "curve": curve,
        "recon_mae": recon_mae,
        "recon_mse": recon_mse,
        "gen_mae": gen_mae,
        "gen_mse": gen_mse,
        "gen_mae_h": gen_mae_h.tolist(),
        "wi": wi,
        "wi_h": wi_h.tolist(),
        "coverage": coverage,
    }


@torch.no_grad()
def generative_fidelity_eval(
    R0: torch.Tensor,
    Y_hat: torch.Tensor,
    *,
    denoiser: R0Denoiser,
    prior,
    schedule: FCPSSchedule,
    device: torch.device,
    n_steps: int,
    n_samples: int,
    n_proj: int,
    eta: float,
    generator: Optional[torch.Generator],
) -> dict:
    """W1 generative-fidelity check (global, distributional).

    R0 / Y_hat are a fixed subsample of fully-observed val windows (complete
    case, so the R^N node vectors are clean - no masked garbage). we draw
    `n_samples` pure-prior samples per window through the full stochastic
    reverse sde (eta=1 by default), then per horizon compare the generated
    residual cloud against the true cloud in node-space via sliced wasserstein-1
    and energy distance. this asks whether the generative chain produces
    realistic residuals, not whether it can locally undo a small perturbation.
    """
    denoiser.eval()
    R0 = R0.to(device)
    Y_hat = Y_hat.to(device)
    W, N, H = R0.shape

    # generate: pure prior -> full reverse sde, n_samples draws per window.
    Y_hat_rep = Y_hat.repeat(n_samples, 1, 1)                  # (S*W, N, H)
    gen_flat, _ = reverse_sample(
        prior, schedule, denoiser, Y_hat_rep, n_steps=n_steps,
        eta=eta, R_T=None, generator=generator,
    )
    gen = gen_flat.view(n_samples, W, N, H)                    # (S, W, N, H)

    w1_h, ed_h = [], []
    for h in range(H):
        true_cloud = R0[:, :, h]                               # (W, N)
        gen_cloud = gen[:, :, :, h].reshape(n_samples * W, N)  # (S*W, N)
        w1_h.append(sliced_wasserstein1(gen_cloud, true_cloud, n_proj, generator))
        ed_h.append(energy_distance(gen_cloud, true_cloud))

    return {
        "w1_h": w1_h,
        "ed_h": ed_h,
        "w1": float(sum(w1_h) / len(w1_h)),
        "ed": float(sum(ed_h) / len(ed_h)),
        "n_windows": W,
        "n_samples": n_samples,
    }


def _log_w1(epoch: int, res: dict, frac: float, *, to_wandb: bool) -> None:
    """print the W1 generative-fidelity result and optionally push to wandb."""
    horizons = [h for h in _HL_HORIZONS if h <= len(res["w1_h"])]
    hl = ", ".join(f"h{h}=W1:{res['w1_h'][h - 1]:.3f}/ED:{res['ed_h'][h - 1]:.3f}"
                   for h in horizons)
    print(f"  [w1] {res['n_windows']} clean windows x{res['n_samples']} "
          f"(={frac:.2f} of val)  mean W1={res['w1']:.3f} ED={res['ed']:.3f}  ({hl})",
          flush=True)
    if not to_wandb:
        return
    import wandb
    log = {"w1/swd": res["w1"], "w1/energy": res["ed"], "epoch": epoch}
    for h in horizons:
        log[f"w1/swd_h{h}"] = res["w1_h"][h - 1]
        log[f"w1/energy_h{h}"] = res["ed_h"][h - 1]
    wandb.log(log)


def _log_reconstruction(epoch: int, res: dict, alpha: float, *, to_wandb: bool) -> None:
    """print the reconstruction-eval result and optionally push scalars to wandb."""
    pct = int(round((1.0 - alpha) * 100))
    horizons = [h for h in _HL_HORIZONS if h <= len(res["gen_mae_h"])]
    print(f"  [recon] curve mae (t->0): "
          + " ".join(f"{t:.2f}:{mae:.3f}" for t, mae, _ in res["curve"]), flush=True)
    hl = ", ".join(f"h{h}={res['gen_mae_h'][h - 1]:.3f}" for h in horizons)
    print(f"  [recon] gen mae={res['gen_mae']:.3f} mse={res['gen_mse']:.3f} "
          f"({hl})  WI{pct}={res['wi']:.3f}  cov={res['coverage']:.3f}", flush=True)
    if not to_wandb:
        return
    import wandb
    log = {
        "recon/recon_mae": res["recon_mae"],
        "recon/recon_mse": res["recon_mse"],
        "recon/gen_mae": res["gen_mae"],
        "recon/gen_mse": res["gen_mse"],
        "recon/wi": res["wi"],
        "recon/coverage": res["coverage"],
        "epoch": epoch,
    }
    for h in horizons:
        log[f"recon/gen_mae_h{h}"] = res["gen_mae_h"][h - 1]
        log[f"recon/wi_h{h}"] = res["wi_h"][h - 1]
    wandb.log(log)


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems08f"), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--d-h", type=int, default=32)
    p.add_argument("--channel-multipliers", type=int, nargs="+", default=[1, 2])
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--c-min", type=float, default=0.05)
    p.add_argument("--c-0", type=float, default=20000.0)
    p.add_argument("--alpha", type=float, default=7.0)
    p.add_argument("--t-eps", type=float, default=1e-3, help="lower bound on t to avoid σ²=0")
    # ── end-to-end reconstruction / generation eval (one val batch)
    p.add_argument("--recon-every", type=int, default=10,
                   help="run the reconstruction+WI eval every N epochs (0 disables)")
    p.add_argument("--recon-steps", type=int, default=100,
                   help="reverse-sampler steps for the reconstruction eval")
    p.add_argument("--recon-samples", type=int, default=20,
                   help="K prior-seeded samples for the generative mae + winkler interval")
    p.add_argument("--recon-alpha", type=float, default=0.1,
                   help="miscoverage for the winkler interval (0.1 -> central 90%%)")
    p.add_argument("--recon-log-every", type=int, default=25,
                   help="record the reconstruction mae/mse every this many reverse steps")
    # ── W1 generative-fidelity eval (fixed subsample of clean val windows)
    p.add_argument("--w1-every", type=int, default=20,
                   help="run the W1 (sliced wasserstein + energy) eval every N epochs (0 disables)")
    p.add_argument("--w1-windows", type=int, default=512,
                   help="number of fully-observed val windows in the W1 cloud")
    p.add_argument("--w1-samples", type=int, default=1,
                   help="prior-seeded samples generated per window for the W1 cloud")
    p.add_argument("--w1-proj", type=int, default=128,
                   help="random projections for the sliced wasserstein-1")
    p.add_argument("--w1-eta", type=float, default=1.0,
                   help="reverse-sde noise level for W1 sampling (1.0 = full stochastic sde)")
    p.add_argument("--smoke", action="store_true", help="1 epoch, tiny batch, wandb disabled")
    p.add_argument("--fresh", action="store_true", help="delete existing checkpoint dir + restart")
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="if > 0, cap training batches per epoch (for smoke runs)")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch_size = 8
        args.max_train_batches = max(args.max_train_batches, 5)
        # exercise the recon + w1 evals cheaply on the single smoke epoch
        args.recon_every = 1
        args.recon_steps = 20
        args.recon_samples = 4
        args.w1_every = 1
        args.w1_windows = 16
        args.w1_proj = 16
        os.environ.setdefault("WANDB_MODE", "disabled")

    cfg = vars(args).copy()
    cfg["backbone"] = _BACKBONE
    cfg["prior"] = _PRIOR
    cfg_hash = hash_config({k: v for k, v in cfg.items() if k not in _CFG_HASH_EXCLUDE})
    run_id = f"diff_{_BACKBONE}_{args.dataset}_seed{args.seed}_{cfg_hash}"

    ckpt_dir = paths.diffusion_dir(args.dataset, _PRIOR, args.seed)
    if args.fresh and ckpt_dir.exists():
        import shutil
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[train_diffusion] device={device}  run_id={run_id}", flush=True)

    set_seed(args.seed)

    # ── Preflight: fail fast (not hours in) if an input is missing.
    _ex = f"run scripts/extract_residuals.py --dataset {args.dataset} --seed {args.seed} first"
    paths.require_inputs([
        (paths.residual_npz(args.dataset, "train", args.seed), _ex),
        (paths.residual_npz(args.dataset, "val", args.seed), _ex),
        (paths.raw_adj(args.dataset), "sync raw data here (scripts/daic/sync_to_daic.sh --raw)"),
    ])

    # ── Data
    train_ds = ResidualDataset(residual_path(args.dataset, "train", args.seed))
    val_ds = ResidualDataset(residual_path(args.dataset, "val", args.seed))
    print(f"[train_diffusion] train={len(train_ds)}  val={len(val_ds)}", flush=True)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=pin,
    )

    # ── Prior + schedule
    prior = load_prior(args.dataset)
    prior_t = prior_to_torch(prior, device=device, dtype=torch.float32)
    schedule = FCPSSchedule(c_min=args.c_min, c_0=args.c_0, alpha=args.alpha, T=1.0)
    print(
        f"[train_diffusion] prior N={prior.N} H={prior.H} γ={prior.gamma:.4f}  "
        f"θ_min={float(prior.theta.min()):.4f}  θ_max={float(prior.theta.max()):.2f}",
        flush=True,
    )

    # ── Model
    A = load_adjacency(args.dataset)
    denoiser = R0Denoiser(
        N=prior.N, H=prior.H, adjacency=A,
        d_h=args.d_h, channel_multipliers=tuple(args.channel_multipliers),
        n_blocks=args.n_blocks, device=str(device),
    ).to(device)
    print(f"[train_diffusion] denoiser params: {num_parameters(denoiser):,}", flush=True)

    optim = torch.optim.Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = None  # add cosine later if needed

    # ── Resume (if a checkpoint exists)
    start_epoch = 0
    best_val = float("inf")
    bad_epochs = 0
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    if ckpt_path is not None and not args.fresh:
        ckpt = load_checkpoint(ckpt_path)
        denoiser.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        if sched is not None and ckpt.get("scheduler_state_dict") is not None:
            sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val_mae"]
        restore_rng_states(ckpt["rng_states"])
        print(f"[train_diffusion] resumed from epoch {start_epoch}  best_val={best_val:.4f}",
              flush=True)

    # ── wandb
    if os.environ.get("WANDB_MODE", "online") != "disabled":
        init_wandb(
            cfg, stage="diffusion", dataset=args.dataset, seed=args.seed,
            prior=_PRIOR, backbone=_BACKBONE,
        )
    import wandb  # safe — always installed

    # ── Reconstruction eval: one fixed val batch + a dedicated rng so the
    # diagnostic is comparable across epochs. mps can't drive randn from a
    # device generator, so fall back to the global rng there (cuda in prod).
    recon_batch = next(iter(val_loader)) if args.recon_every else None
    recon_gen = (
        None if device.type == "mps"
        else torch.Generator(device=device.type).manual_seed(args.seed + 999)
    )

    def _maybe_recon(epoch: int, is_last: bool) -> None:
        if not args.recon_every:
            return
        if epoch % args.recon_every != 0 and not is_last:
            return
        res = reconstruction_eval(
            recon_batch, denoiser=denoiser, prior=prior, prior_t=prior_t,
            schedule=schedule, device=device, n_steps=args.recon_steps,
            n_samples=args.recon_samples, alpha=args.recon_alpha,
            log_every=args.recon_log_every, generator=recon_gen,
        )
        _log_reconstruction(
            epoch, res, args.recon_alpha,
            to_wandb=os.environ.get("WANDB_MODE", "online") != "disabled",
        )

    # ── W1 cloud: a fixed subsample of fully-observed (complete-case) val
    # windows, so the per-horizon R^N node vectors carry no masked garbage.
    if args.w1_every:
        full_valid = val_ds.mask.reshape(len(val_ds), -1).all(dim=1)   # (Nval,)
        valid_idx = full_valid.nonzero(as_tuple=True)[0]
        w1_frac = float(full_valid.float().mean())
        n_w1 = min(args.w1_windows, valid_idx.numel())
        sel = valid_idx[:n_w1]
        w1_R0, w1_Yhat = val_ds.R0[sel], val_ds.Y_hat[sel]
        print(f"[train_diffusion] W1 cloud: {n_w1} clean windows "
              f"({w1_frac:.2f} of val fully observed)", flush=True)
    else:
        w1_R0 = w1_Yhat = None
        w1_frac = 0.0

    def _maybe_w1(epoch: int, is_last: bool) -> None:
        if not args.w1_every or w1_R0 is None or w1_R0.shape[0] == 0:
            return
        if epoch % args.w1_every != 0 and not is_last:
            return
        res = generative_fidelity_eval(
            w1_R0, w1_Yhat, denoiser=denoiser, prior=prior, schedule=schedule,
            device=device, n_steps=args.recon_steps, n_samples=args.w1_samples,
            n_proj=args.w1_proj, eta=args.w1_eta, generator=recon_gen,
        )
        _log_w1(
            epoch, res, w1_frac,
            to_wandb=os.environ.get("WANDB_MODE", "online") != "disabled",
        )

    # ── Train
    for epoch in range(start_epoch, args.epochs):
        denoiser.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for i, batch in enumerate(train_loader):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            optim.zero_grad()
            loss = train_step(
                batch, denoiser=denoiser, prior_t=prior_t, schedule=schedule,
                device=device, t_eps=args.t_eps, generator=None,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 5.0)
            optim.step()
            running += loss.detach().item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_loss = validate(
            val_loader, denoiser=denoiser, prior_t=prior_t, schedule=schedule,
            device=device, t_eps=args.t_eps,
        )
        elapsed = time.time() - t0
        print(
            f"epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        if os.environ.get("WANDB_MODE", "online") != "disabled":
            wandb.log({"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch})

        _maybe_recon(epoch, is_last=(epoch == args.epochs - 1))
        _maybe_w1(epoch, is_last=(epoch == args.epochs - 1))

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        save_checkpoint(
            ckpt_dir / "last.pt", denoiser, optim, sched,
            epoch=epoch, best_val_mae=best_val, rng_states=get_rng_states(),
        )
        if improved:
            save_checkpoint(
                ckpt_dir / "best.pt", denoiser, optim, sched,
                epoch=epoch, best_val_mae=best_val, rng_states=get_rng_states(),
            )

        if bad_epochs >= args.patience:
            print(f"[train_diffusion] early stop at epoch {epoch} "
                  f"(patience={args.patience}, best_val={best_val:.4f})", flush=True)
            break


if __name__ == "__main__":
    main()
