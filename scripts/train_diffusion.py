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
    p.add_argument("--smoke", action="store_true", help="1 epoch, tiny batch, wandb disabled")
    p.add_argument("--fresh", action="store_true", help="delete existing checkpoint dir + restart")
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="if > 0, cap training batches per epoch (for smoke runs)")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch_size = 8
        args.max_train_batches = max(args.max_train_batches, 5)
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
