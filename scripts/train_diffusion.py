"""train the diffusion denoiser (predicts the clean residual R0).

inputs are the cached residuals (src.utils.paths.residual_npz) and the kronecker
prior from src.diffusion.prior.load_prior.

each step (noise-level / c̄ parameterization, EDM-style):
  1. take a batch of (R0, Y_hat, mask) from the residual cache
  2. draw the noise level c̄ ~ log-uniform[c̄_lo, c̄_hi] (even coverage of every
     mode's transition; uniform-in-t on the steep schedule misses the middle)
  3. forward-sample R_t = sample_R_t_from_c_bar_torch(R0, c̄, ...)
  4. predict R0_hat = denoiser(R_t, tau, Y_hat), tau = normalized log-c̄ in [0,1]
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
from src.data.residual_dataset import ResidualDataset, residual_path, train_residual_sigma
from src.diffusion.denoiser import R0Denoiser, num_parameters
from src.diffusion.losses import compute_loss, mahalanobis_mse
from src.diffusion.forward_fcps import (
    FCPSSchedule,
    c_bar_to_tau,
    noise_level_bounds,
    prior_to_torch,
    sample_log_c_bar_torch,
    sample_R_t_from_c_bar_torch,
)
from src.diffusion.prior import load_prior
from src.diffusion.reverse import reverse_sample
from src.eval.distribution import energy_distance, sliced_wasserstein1
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

# Config keys whose value should NOT contribute to the hash — varying them
# should NOT trigger a fresh wandb run / fresh checkpoint dir.
_CFG_HASH_EXCLUDE = {"seed", "epochs", "batch_size", "lr", "weight_decay", "patience",
                     "exp", "lr_find", "lr_find_min", "lr_find_max"}


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


def lr_range_test(loader, *, denoiser, prior_t, c_lo, c_hi, loss_name, optim, device,
                  n_steps, lr_min, lr_max, out_path) -> None:
    """Smith lr range test: sweep lr exponentially lr_min->lr_max over n_steps
    training batches, recording the loss. the good lr band is where the loss falls
    fastest (most negative slope of loss vs log-lr); a common pick is ~1 order
    below the loss minimum. writes (lr, loss) to out_path + prints a suggestion.
    this trains the weights as it goes, so the run exits afterwards (no real fit)."""
    import math
    mult = (lr_max / lr_min) ** (1.0 / max(n_steps - 1, 1))
    lr = lr_min
    lrs: list[float] = []
    losses: list[float] = []
    denoiser.train()
    it = iter(loader)
    best = float("inf")
    for i in range(n_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad()
        loss = train_step(batch, denoiser=denoiser, prior_t=prior_t,
                          c_lo=c_lo, c_hi=c_hi, loss_name=loss_name, device=device, generator=None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 5.0)
        optim.step()
        val = float(loss.detach())
        lrs.append(lr); losses.append(val)
        print(f"  [lr-find] step {i:4d}  lr={lr:.3e}  loss={val:.4f}", flush=True)
        best = min(best, val)
        if math.isnan(val) or val > 4.0 * best:
            print("  [lr-find] loss diverged - stopping early", flush=True)
            break
        lr *= mult

    # suggestion: steepest descent on a lightly smoothed loss-vs-log10(lr) curve
    a = np.asarray(lrs); b = np.asarray(losses)
    suggest = None
    if len(a) > 5:
        k = max(1, len(b) // 20)
        bs = np.convolve(b, np.ones(k) / k, mode="same")
        grad = np.gradient(bs, np.log10(a))
        suggest = float(a[int(np.argmin(grad))])
    out_path.write_text(json.dumps(
        {"lrs": lrs, "losses": losses, "suggested_lr": suggest}, indent=2))
    print(f"  [lr-find] wrote {out_path}", flush=True)
    if suggest is not None:
        print(f"  [lr-find] suggested lr ~ {suggest:.2e}  (steepest descent; "
              f"also eyeball the curve)", flush=True)


# ─── Training step ───────────────────────────────────────────────────────────


def train_step(
    batch: dict[str, torch.Tensor],
    *,
    denoiser: R0Denoiser,
    prior_t,
    c_lo: float,
    c_hi: float,
    loss_name: str,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    R0 = batch["R0"].to(device, non_blocking=True)         # (B, N, H)
    Y_hat = batch["Y_hat"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    B = R0.shape[0]

    # draw the noise level log-uniformly in [c̄_lo, c̄_hi] (even coverage of every
    # mode's transition), forward-noise R0 to it, and condition on the normalized
    # log-noise tau. this is the EDM-style noise-level axis, not uniform-in-t.
    c_bar = sample_log_c_bar_torch(c_lo, c_hi, B, device, R0.dtype, generator=generator)
    R_t = sample_R_t_from_c_bar_torch(R0, c_bar, prior_t, generator=generator)
    tau = c_bar_to_tau(c_bar, c_lo, c_hi)

    R0_hat = denoiser(R_t, tau, Y_hat)
    return compute_loss(loss_name, R0_hat, R0, mask, c_bar, prior_t)


@torch.no_grad()
def validate(
    loader: DataLoader,
    *,
    denoiser: R0Denoiser,
    prior_t,
    c_lo: float,
    c_hi: float,
    loss_name: str,
    device: torch.device,
    n_t_samples: int = 4,
) -> float:
    """Average validation loss over the loader (same loss as training, for early
    stopping). For each batch we average over `n_t_samples` random noise-level draws.
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
            c_bar = sample_log_c_bar_torch(c_lo, c_hi, B, device, R0.dtype)
            R_t = sample_R_t_from_c_bar_torch(R0, c_bar, prior_t)
            R0_hat = denoiser(R_t, c_bar_to_tau(c_bar, c_lo, c_hi), Y_hat)
            batch_loss += float(compute_loss(loss_name, R0_hat, R0, mask, c_bar, prior_t))
        total += batch_loss / n_t_samples
        n += 1
    denoiser.train()
    return total / max(n, 1)


# ─── Validation diagnostic (every few epochs) ────────────────────────────────


_HL_HORIZONS = (3, 6, 12)  # horizons highlighted in the printout (1-indexed)


@torch.no_grad()
def val_diagnostic(
    R0_fo: torch.Tensor,
    Y_hat_fo: torch.Tensor,
    *,
    denoiser: R0Denoiser,
    prior,
    prior_t,
    schedule: FCPSSchedule,
    device: torch.device,
    n_steps: int,
    n_proj: int,
    sigma_r: float,
    c_lo: float,
    c_hi: float,
    generator: Optional[torch.Generator],
) -> dict:
    """end-to-end reverse-process check on fully-observed val windows.

    the flow: take a val batch, forward-noise it close to fully-noised (c̄_hi), run
    the reverse sde back, and score the reconstruction —
      - MSE / MAE vs the true residual (raw units, overall + per horizon),
      - Wasserstein: sliced-W1 + energy distance between the reconstructed and true
        residual clouds, per horizon, in node space,
      - Mahalanobis: the one-shot denoising error in standardised (per-mode) units —
        the metric that is comparable across priors (kron vs iso).
    fully-observed windows only, so the node vectors carry no masked garbage and the
    eigenbasis rotation (Mahalanobis) is clean. metrics are de-normalized (× sigma_r)
    to raw units; the backbone term cancels, so only the residual scaler enters.
    """
    denoiser.eval()
    R0 = R0_fo.to(device)                                    # unit space
    Y_hat = Y_hat_fo.to(device)
    W, N, H = R0.shape
    R0_phys = R0 * sigma_r                                   # raw units
    full = torch.ones_like(R0)                               # fully observed

    # ── noise close to full (c̄_hi), then reverse-denoise back to the data ──
    c_top = torch.full((W,), c_hi, device=device, dtype=R0.dtype)
    R_T = sample_R_t_from_c_bar_torch(R0, c_top, prior_t, generator=generator)
    R0_hat, _ = reverse_sample(prior, schedule, denoiser, Y_hat, n_steps=n_steps,
                               eta=0.0, R_T=R_T, generator=generator)
    R0_hat_phys = R0_hat * sigma_r

    recon_mae = float(masked_mae(R0_hat_phys, R0_phys, full))
    recon_mse = float(masked_mse(R0_hat_phys, R0_phys, full))
    recon_mae_h = masked_mae_per_h(R0_hat_phys, R0_phys, full).tolist()

    # ── Wasserstein on reconstructed vs true residual clouds (per horizon) ──
    w1_h, ed_h = [], []
    for h in range(H):
        w1_h.append(sliced_wasserstein1(R0_hat_phys[:, :, h], R0_phys[:, :, h], n_proj, generator))
        ed_h.append(energy_distance(R0_hat_phys[:, :, h], R0_phys[:, :, h]))

    # ── Mahalanobis: one-shot denoising error in standardised units (cross-prior) ──
    c_bar = sample_log_c_bar_torch(c_lo, c_hi, W, device, R0.dtype)
    R_t = sample_R_t_from_c_bar_torch(R0, c_bar, prior_t)
    R0_hat_1 = denoiser(R_t, c_bar_to_tau(c_bar, c_lo, c_hi), Y_hat)
    maha = float(mahalanobis_mse(R0_hat_1, R0, full, c_bar, prior_t))

    denoiser.train()
    return {
        "recon_mae": recon_mae, "recon_mse": recon_mse, "recon_mae_h": recon_mae_h,
        "w1": float(sum(w1_h) / H), "ed": float(sum(ed_h) / H),
        "w1_h": w1_h, "ed_h": ed_h, "mahalanobis": maha, "n_windows": W,
    }


def _log_val(epoch: int, res: dict, *, to_wandb: bool) -> None:
    """print the val diagnostic and optionally push scalars to wandb."""
    hz = [h for h in _HL_HORIZONS if h <= len(res["recon_mae_h"])]
    hl = ", ".join(f"h{h}={res['recon_mae_h'][h - 1]:.3f}" for h in hz)
    print(f"  [val] {res['n_windows']} clean windows | recon MAE={res['recon_mae']:.3f} "
          f"MSE={res['recon_mse']:.3f} ({hl}) | W1={res['w1']:.3f} ED={res['ed']:.3f} "
          f"| mahalanobis={res['mahalanobis']:.3f}", flush=True)
    if not to_wandb:
        return
    import wandb
    log = {"val/recon_mae": res["recon_mae"], "val/recon_mse": res["recon_mse"],
           "val/w1": res["w1"], "val/energy": res["ed"],
           "val/mahalanobis": res["mahalanobis"], "epoch": epoch}
    for h in hz:
        log[f"val/recon_mae_h{h}"] = res["recon_mae_h"][h - 1]
        log[f"val/w1_h{h}"] = res["w1_h"][h - 1]
    wandb.log(log)


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems04f", "pems04s",
                                          "pems08f", "pems08s"), required=True)
    p.add_argument("--spatial", choices=("structured", "flat"), default="structured",
                   help="kronecker spatial factor: structured Σ_S=(L+γI)^-1 (spatio-temporal) "
                        "or flat Σ_S=I (temporal-only). recorded in config.json.")
    p.add_argument("--stiffness-cap", type=float, default=0.0,
                   help="if >0, floor θ so θ_max/θ_min <= this (joint-spectrum twin of κ-cap; "
                        "trainability fallback for stiff structured priors). 0 = off.")
    p.add_argument("--prior", choices=("kronecker", "isotropic"), default="kronecker",
                   help="noise prior: structured kronecker, or the isotropic σ²I baseline")
    p.add_argument("--exp", default="dev",
                   help="experiment tag; namespaces checkpoints + wandb group "
                        "(results/diffusion/<exp>/..., wandb group <exp>-<dataset>)")
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
    p.add_argument("--loss", choices=("snr", "mse", "mahalanobis"), default="snr",
                   help="training loss: snr-weighted mse (default), plain mse, or the "
                        "precision-weighted mahalanobis ablation")
    # ── learning-rate range test (Smith): sweep lr exponentially over a few
    # hundred steps to locate the good lr band, then exit (no full training).
    p.add_argument("--lr-find", type=int, default=0,
                   help="if >0, run an lr range test for this many steps and exit")
    p.add_argument("--lr-find-min", type=float, default=1e-6)
    p.add_argument("--lr-find-max", type=float, default=1.0)
    # ── val diagnostic (every N epochs): noise a clean val batch close to full,
    #    reverse it, score recon MSE/MAE + Wasserstein + the mahalanobis metric.
    p.add_argument("--val-every", type=int, default=5,
                   help="run the val diagnostic every N epochs (0 disables)")
    p.add_argument("--val-steps", type=int, default=100,
                   help="reverse-sampler steps for the val diagnostic")
    p.add_argument("--val-windows", type=int, default=512,
                   help="number of fully-observed val windows used by the diagnostic")
    p.add_argument("--val-proj", type=int, default=128,
                   help="random projections for the sliced wasserstein-1")
    p.add_argument("--smoke", action="store_true", help="1 epoch, tiny batch, wandb disabled")
    p.add_argument("--fresh", action="store_true", help="delete existing checkpoint dir + restart")
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="if > 0, cap training batches per epoch (for smoke runs)")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch_size = 8
        args.max_train_batches = max(args.max_train_batches, 5)
        # exercise the val diagnostic cheaply on the single smoke epoch
        args.val_every = 1
        args.val_steps = 20
        args.val_windows = 16
        args.val_proj = 16
        os.environ.setdefault("WANDB_MODE", "disabled")

    cfg = vars(args).copy()
    cfg["backbone"] = _BACKBONE
    cfg["prior"] = args.prior
    cfg_hash = hash_config({k: v for k, v in cfg.items() if k not in _CFG_HASH_EXCLUDE})
    run_id = f"diff_{_BACKBONE}_{args.dataset}_seed{args.seed}_{cfg_hash}"

    ckpt_dir = paths.diffusion_dir(args.dataset, args.prior, args.seed, exp=args.exp)
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
    _ex = (f"run scripts/gwn_extract_residuals.py --variant {args.dataset} "
           f"--seed {args.seed} --checkpoint <gwn-ckpt> first")
    paths.require_inputs([
        (paths.residual_npz(args.dataset, "train", args.seed), _ex),
        (paths.residual_npz(args.dataset, "val", args.seed), _ex),
        (paths.adj_npz(args.dataset), "run python scripts/build_datasets.py --all"),
    ])

    # ── Data (modeled in unit space; sigma_r from train, reused for val)
    sigma_r = train_residual_sigma(args.dataset, args.seed)
    train_ds = ResidualDataset(residual_path(args.dataset, "train", args.seed), sigma_r)
    val_ds = ResidualDataset(residual_path(args.dataset, "val", args.seed), sigma_r)
    print(f"[train_diffusion] train={len(train_ds)}  val={len(val_ds)}  "
          f"sigma_r={sigma_r:.4f} (residuals modeled in unit space)", flush=True)

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
    prior = load_prior(args.dataset, args.prior, args.seed,
                       spatial=args.spatial, stiffness_cap=(args.stiffness_cap or None))
    prior_t = prior_to_torch(prior, device=device, dtype=torch.float32)
    # c_0 is auto-derived from the fitted prior (5/θ_min), never a CLI number -
        # the human-typed-c_0 footgun bites the same at 35 as at 27000. --c-0 is ignored.
    schedule = FCPSSchedule.for_prior(prior, c_min=args.c_min, alpha=args.alpha, T=1.0)
    print(f"[train_diffusion] schedule c_0 auto-derived = {schedule.c_0:.1f} "
          f"(--c-0 ignored)", flush=True)
    # persist the derived c_0 so config.json describes the schedule actually used
    # (it used to record the ignored CLI value, which misled eval_diffusion)
    cfg["c_0"] = schedule.c_0
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    # noise-level (c̄) bounds the training sampler and reverse grid span — derived
    # from the prior spectrum. training draws c̄ log-uniformly in [c_lo, c_hi].
    c_lo, c_hi = noise_level_bounds(prior)
    print(
        f"[train_diffusion] prior N={prior.N} H={prior.H} γ={prior.gamma:.4f}  "
        f"θ_min={float(prior.theta.min()):.4f}  θ_max={float(prior.theta.max()):.2f}  "
        f"noise-level c̄∈[{c_lo:.3g}, {c_hi:.3g}] (log-uniform sampling)",
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

    # lr range test: sweep lr, record loss, exit before the real training loop.
    if args.lr_find:
        lr_range_test(
            train_loader, denoiser=denoiser, prior_t=prior_t, c_lo=c_lo, c_hi=c_hi,
            loss_name=args.loss, optim=optim, device=device, n_steps=args.lr_find,
            lr_min=args.lr_find_min, lr_max=args.lr_find_max,
            out_path=ckpt_dir / "lr_find.json",
        )
        return

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
            prior=args.prior, backbone=_BACKBONE, exp=args.exp,
        )
    import wandb  # safe — always installed

    # ── Val diagnostic: a fixed subsample of fully-observed (complete-case) val
    # windows, so the node vectors carry no masked garbage and the eigenbasis
    # rotation (mahalanobis) is clean. a dedicated rng keeps it comparable across
    # epochs; mps can't drive randn from a device generator, so fall back there.
    val_gen = (
        None if device.type == "mps"
        else torch.Generator(device=device.type).manual_seed(args.seed + 999)
    )
    if args.val_every:
        full_valid = val_ds.mask.reshape(len(val_ds), -1).all(dim=1)   # (Nval,)
        valid_idx = full_valid.nonzero(as_tuple=True)[0]
        n_val = min(args.val_windows, valid_idx.numel())
        val_R0, val_Yhat = val_ds.R0[valid_idx[:n_val]], val_ds.Y_hat[valid_idx[:n_val]]
        print(f"[train_diffusion] val diagnostic: {n_val} clean windows "
              f"({float(full_valid.float().mean()):.2f} of val fully observed)", flush=True)
    else:
        val_R0 = val_Yhat = None

    def _maybe_val(epoch: int, is_last: bool) -> None:
        if not args.val_every or val_R0 is None or val_R0.shape[0] == 0:
            return
        if epoch % args.val_every != 0 and not is_last:
            return
        res = val_diagnostic(
            val_R0, val_Yhat, denoiser=denoiser, prior=prior, prior_t=prior_t,
            schedule=schedule, device=device, n_steps=args.val_steps,
            n_proj=args.val_proj, sigma_r=sigma_r, c_lo=c_lo, c_hi=c_hi, generator=val_gen,
        )
        _log_val(epoch, res, to_wandb=os.environ.get("WANDB_MODE", "online") != "disabled")

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
                batch, denoiser=denoiser, prior_t=prior_t, c_lo=c_lo, c_hi=c_hi,
                loss_name=args.loss, device=device, generator=None,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 5.0)
            optim.step()
            running += loss.detach().item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_loss = validate(
            val_loader, denoiser=denoiser, prior_t=prior_t, c_lo=c_lo, c_hi=c_hi,
            loss_name=args.loss, device=device,
        )
        elapsed = time.time() - t0
        print(
            f"epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        if os.environ.get("WANDB_MODE", "online") != "disabled":
            wandb.log({"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch})

        _maybe_val(epoch, is_last=(epoch == args.epochs - 1))

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
