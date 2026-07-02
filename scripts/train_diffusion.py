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
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.adjacency import load_adjacency
from src.data.residual_dataset import make_dataset, residual_path, train_residual_sigma
from src.diffusion.denoiser import R0Denoiser, SignalDenoiser, num_parameters
from src.diffusion.losses import gad_snr_weighted_mse, graph_tv
from src.diffusion.forward_fcps import (
    FCPSSchedule,
    fcps_rate_torch,
    prior_to_torch,
    sample_c_bar_powerlaw_torch,
    sample_R_t_from_c_bar_torch,
)
from src.diffusion.prior import load_prior
from src.diffusion.reverse import reverse_sample_em, min_em_steps
from src.eval.crps import empirical_crps
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
from src.utils.logging_setup import get_logger, setup_logging
from src.utils.seed import set_seed
from src.utils.wandb_setup import init_wandb

# shared timestamped logger (configured in main() once the run dir is known); module
# helpers below log through it so every line lands in both stdout and the run's train.log.
log = get_logger()

# Config keys whose value should NOT contribute to the hash — varying them
# should NOT trigger a fresh wandb run / fresh checkpoint dir.
_CFG_HASH_EXCLUDE = {"seed", "epochs", "batch_size", "lr", "weight_decay", "patience",
                     "exp", "lr_find", "lr_find_min", "lr_find_max"}


# ─── Loss / metrics ──────────────────────────────────────────────────────────


def lr_range_test(loader, *, denoiser, prior_t, schedule, tv_weight, A_t, optim, device,
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
        loss = train_step(batch, denoiser=denoiser, prior_t=prior_t, schedule=schedule,
                          tv_weight=tv_weight, A_t=A_t, device=device, generator=None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
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


def _gad_loss(R0_hat, R0, mask, c_t, *, tv_weight, A_t):
    """GAD loss: rate-weighted masked mse (+ optional graph-TV)."""
    loss = gad_snr_weighted_mse(R0_hat, R0, mask, c_t)
    if tv_weight and A_t is not None:
        loss = loss + tv_weight * graph_tv(R0_hat, A_t)
    return loss


def train_step(
    batch: dict[str, torch.Tensor],
    *,
    denoiser: R0Denoiser,
    prior_t,
    schedule: FCPSSchedule,
    tv_weight: float,
    A_t: Optional[torch.Tensor],
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    R0 = batch["R0"].to(device, non_blocking=True)         # (B, N, H)
    Y_hat = batch["Y_hat"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    B = R0.shape[0]

    # GAD scheme (exp_gasde.py): draw the noise level c̄ = c_0·u^0.65 (power-law),
    # forward-noise R0 to it, recover the diffusion time t, and condition the net on t.
    c_bar = sample_c_bar_powerlaw_torch(schedule.c_0, B, device, R0.dtype, generator=generator)
    R_t = sample_R_t_from_c_bar_torch(R0, c_bar, prior_t, generator=generator)
    t = schedule.t_of_c_bar(c_bar)
    c_t = fcps_rate_torch(t, schedule)                      # rate c(t) for the SNR weight

    R0_hat = denoiser(R_t, t, Y_hat)
    return _gad_loss(R0_hat, R0, mask, c_t, tv_weight=tv_weight, A_t=A_t)


@torch.no_grad()
def validate(
    loader: DataLoader,
    *,
    denoiser: R0Denoiser,
    prior_t,
    schedule: FCPSSchedule,
    tv_weight: float,
    A_t: Optional[torch.Tensor],
    device: torch.device,
    n_t_samples: int = 4,
) -> float:
    """Average validation loss over the loader (same GAD loss as training, for the
    loss curve). For each batch we average over `n_t_samples` random noise-level draws.
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
            c_bar = sample_c_bar_powerlaw_torch(schedule.c_0, B, device, R0.dtype)
            R_t = sample_R_t_from_c_bar_torch(R0, c_bar, prior_t)
            t = schedule.t_of_c_bar(c_bar)
            c_t = fcps_rate_torch(t, schedule)
            R0_hat = denoiser(R_t, t, Y_hat)
            batch_loss += float(_gad_loss(R0_hat, R0, mask, c_t, tv_weight=tv_weight, A_t=A_t))
        total += batch_loss / n_t_samples
        n += 1
    denoiser.train()
    return total / max(n, 1)


# ─── Validation diagnostic (every few epochs) ────────────────────────────────


_HL_HORIZONS = (3, 6, 12)  # horizons highlighted in the printout (1-indexed)


@torch.no_grad()
def periodic_val_eval(
    R0_fo: torch.Tensor,
    Y_hat_fo: torch.Tensor,
    *,
    denoiser,
    prior,
    schedule: FCPSSchedule,
    device: torch.device,
    n_steps: int,
    k: int,
    n_proj: int,
    sigma_r: float,
    generator: Optional[torch.Generator],
    gen_batch: int = 1024,
) -> dict:
    """Vimal-style periodic full-reverse validation (runs every --val-every epochs).

    the "full reverse evaluation and comparison": draw k trajectories FROM THE PRIOR
    via the GAD Euler–Maruyama reverse SDE, conditioned on Y_hat, and compare the sample
    cloud to the true val target —
      - MAE / RMSE of the k-sample mean, overall + per horizon (point accuracy),
      - CRPS of the sample cloud (the probabilistic score best.pt is selected by, Vimal),
      - sliced-W1 + energy distance vs the true cloud, per horizon (distributional match).
    clean (fully-observed) windows only; de-normalized to raw units (backbone cancels,
    so only sigma_r enters). this mirrors Vimal's evals(mode="Val") but with GAD's EM
    sampler (which needs many steps) at a periodic cadence instead of every epoch.
    """
    denoiser.eval()
    W, N, H = R0_fo.shape
    R0 = R0_fo.to(device) * sigma_r                          # raw truth (residual or signal)
    Y = Y_hat_fo.to(device)
    chunk = max(1, gen_batch // k)
    mae_n = torch.zeros(H); mse_n = torch.zeros(H); crps_n = torch.zeros(H)
    gen_first = truth_first = None
    for s in range(0, W, chunk):
        e = min(s + chunk, W); b = e - s
        samp, _ = reverse_sample_em(prior, schedule, denoiser, Y[s:e].repeat(k, 1, 1),
                                    n_steps=n_steps, generator=generator)
        samp = samp.view(k, b, N, H) * sigma_r              # raw sample cloud (k, b, N, H)
        mean = samp.mean(0)
        mae_n += (mean - R0[s:e]).abs().sum(dim=(0, 1)).cpu()
        mse_n += (mean - R0[s:e]).pow(2).sum(dim=(0, 1)).cpu()
        crps_n += empirical_crps(samp, R0[s:e]).sum(dim=(0, 1)).cpu()   # per-horizon energy CRPS
        if gen_first is None:                                # keep first chunk for W1/energy
            gen_first, truth_first = samp, R0[s:e]
    cnt_h = float(W * N)            # per-horizon denominator (mae_n[h] is summed over B,N)
    cnt_all = cnt_h * H            # overall denominator (mae_n.sum() is summed over B,N,H)
    mae_h = (mae_n / cnt_h); mse_h = (mse_n / cnt_h); crps_h = (crps_n / cnt_h)

    # distributional comparison on the first chunk (per horizon, node space)
    w1_h, ed_h = [], []
    for h in range(H):
        gen_c = gen_first[:, :, :, h].reshape(-1, N)
        w1_h.append(sliced_wasserstein1(gen_c, truth_first[:, :, h], n_proj, generator))
        ed_h.append(energy_distance(gen_c, truth_first[:, :, h]))

    denoiser.train()
    return {
        "mae": float(mae_n.sum() / cnt_all), "rmse": float((mse_n.sum() / cnt_all) ** 0.5),
        "crps": float(crps_n.sum() / cnt_all), "mae_h": mae_h.tolist(),
        "w1": float(sum(w1_h) / H), "energy": float(sum(ed_h) / H),
        "w1_h": w1_h, "energy_h": ed_h, "n_windows": W,
    }


def _log_val(epoch: int, res: dict, *, to_wandb: bool) -> None:
    """print the periodic full-reverse val eval and optionally push scalars to wandb."""
    hz = [h for h in _HL_HORIZONS if h <= len(res["mae_h"])]
    hl = ", ".join(f"h{h}={res['mae_h'][h - 1]:.3f}" for h in hz)
    log.info(f"  [val-eval] {res['n_windows']} windows | MAE={res['mae']:.3f} "
             f"RMSE={res['rmse']:.3f} CRPS={res['crps']:.4f} ({hl}) | "
             f"W1={res['w1']:.3f} ED={res['energy']:.3f}")
    if not to_wandb:
        return
    import wandb
    wb = {"val/mae": res["mae"], "val/rmse": res["rmse"], "val/crps": res["crps"],
          "val/w1": res["w1"], "val/energy": res["energy"], "epoch": epoch}
    for h in hz:
        wb[f"val/mae_h{h}"] = res["mae_h"][h - 1]
        wb[f"val/w1_h{h}"] = res["w1_h"][h - 1]
    wandb.log(wb)


def _run_test_eval(args, device, best_epoch, run_id) -> None:
    """combined-job test eval: reload best.pt and evaluate on the test split at each
    --eval-steps value, writing test_results.json next to the eval outputs (Vimal-style:
    one payload with the best_epoch and a per-sample-steps results list)."""
    try:                                                        # heavy import, deferred
        from eval_diffusion import evaluate                      # run as a script (scripts/ on path)
    except ModuleNotFoundError:
        from scripts.eval_diffusion import evaluate              # imported as a package
    out_dir = paths.eval_dir(args.dataset, args.prior, args.seed,
                             exp=args.exp, backbone=args.backbone)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = (None if device.type == "mps"
           else torch.Generator(device=device.type).manual_seed(args.seed))
    steps_list = [int(x) for x in str(args.eval_steps).split(",") if x.strip()]
    log.info(f"[test] best.pt (epoch {best_epoch}); eval at sample_steps={steps_list}")
    results = []
    for ns in steps_list:
        res = evaluate(args.dataset, args.prior, args.seed, exp=args.exp, device=device,
                       k=args.eval_k, n_steps=ns, alpha=0.1,
                       w1_windows=512, w1_samples=4, w1_proj=128, gen_batch=args.eval_gen_batch,
                       max_windows=args.eval_max_windows, generator=gen,
                       backbone=args.backbone, mask_zero=True)
        results.append(res)
        log.info(f"  [test n_steps={ns}] MAE={res['mae']:.4f} RMSE={res['rmse']:.4f} "
                 f"MAPE={res['mape']:.4f} CRPS={res['crps']:.4f} "
                 f"Winkler={res['winkler']:.4f} cov={res['coverage']:.3f}")
    payload = {"run_id": run_id, "dataset": args.dataset, "prior": args.prior,
               "backbone": args.backbone, "seed": args.seed, "exp": args.exp,
               "best_epoch": best_epoch, "select_metric": "crps", "test_results": results}
    (out_dir / "test_results.json").write_text(json.dumps(payload, indent=2))
    log.info(f"[test] wrote {out_dir / 'test_results.json'}")
    _write_run_summary(payload, out_dir)


def _write_run_summary(payload: dict, out_dir) -> None:
    """readable end-of-run summary (Vimal-style): a boxed header + a metric table over
    the eval sample-step settings + per-horizon MAE for the best (largest-n_steps) run.
    printed to the log AND saved to summary.txt next to test_results.json."""
    hl = _HL_HORIZONS
    res_list = payload["test_results"]
    L: list[str] = []
    L.append("=" * 72)
    L.append(f" RUN SUMMARY  {payload['run_id']}")
    L.append("=" * 72)
    L.append(f" dataset={payload['dataset']}  prior={payload['prior']}  "
             f"backbone={payload['backbone']}  seed={payload['seed']}  exp={payload['exp']}")
    L.append(f" reverse=EM (GAD)  select={payload['select_metric']}  "
             f"best_epoch={payload['best_epoch']}")
    L.append("-" * 72)
    L.append(f" {'n_steps':>7} | {'MAE':>8} {'RMSE':>8} {'MAPE':>7} {'CRPS':>7} "
             f"{'Winkler':>8} {'cov':>5}")
    L.append("-" * 72)
    for r in res_list:
        L.append(f" {r['n_steps']:>7} | {r['mae']:>8.4f} {r['rmse']:>8.4f} "
                 f"{r['mape']:>7.4f} {r['crps']:>7.4f} {r['winkler']:>8.4f} "
                 f"{r['coverage']:>5.3f}")
    L.append("-" * 72)
    if res_list:
        best = max(res_list, key=lambda r: r["n_steps"])          # most steps = reference run
        mae_h = best.get("mae_h", [])
        ph = "  ".join(f"h{h}={mae_h[h-1]:.3f}" for h in hl if h <= len(mae_h))
        L.append(f" per-horizon MAE (n_steps={best['n_steps']}):  {ph}")
    L.append("=" * 72)
    text = "\n".join(L)
    (out_dir / "summary.txt").write_text(text + "\n")
    for line in L:
        log.info(line)
    log.info(f"[test] wrote {out_dir / 'summary.txt'}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems04f", "pems04s",
                                          "pems08f", "pems08s"), required=True)
    p.add_argument("--backbone", choices=("gwn", "gcn", "mlp", "raw"), default="gwn",
                   help="which backbone's residuals to model, or 'raw' = pure-signal "
                        "diffusion conditioned on history (DiffSTG-style). namespaces results")
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
    # FCPS schedule = GAD-literal (exp_gasde.py): S=c_0=7, alpha=4, c_min=0.1, sigma=1.
    # the θ→GAD-scale reparam in load_prior puts θ_min=γ=0.8, so 2·S·θ_min=11.2 mixes
    # the slowest mode exactly as GAD does — S is a fixed number, NOT auto-derived.
    p.add_argument("--c-min", type=float, default=0.1, help="GAD c_min (0.1)")
    p.add_argument("--c-0", type=float, default=7.0, help="GAD S = c̄(T), the total heat (7.0)")
    p.add_argument("--alpha", type=float, default=4.0, help="GAD schedule exponent α (4.0)")
    p.add_argument("--t-eps", type=float, default=1e-3, help="lower bound on t to avoid σ²=0")
    p.add_argument("--loss", choices=("snr", "mse", "mahalanobis"), default="snr",
                   help="(legacy) loss name kept for the validate/diagnostic dispatch; "
                        "training now uses GAD's rate-weighted mse directly")
    p.add_argument("--tv-weight", type=float, default=0.0,
                   help="GAD's graph-TV penalty weight on predictions (GAD uses 5e-3). "
                        "off by default: residuals are rough, so TV may be misspecified")
    # ── best.pt is selected by GAD's own validation loss (SNR-weighted MSE, no sampling).
    # EM is the only reverse process (GAD parity) and is too costly/unstable at low step
    # counts to sample every epoch, so we rank epochs by the cheap val loss and reserve
    # EM sampling for the periodic diagnostic + the final locked test eval.
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="ReduceLROnPlateau factor on the val-loss plateau")
    p.add_argument("--lr-patience", type=int, default=5,
                   help="ReduceLROnPlateau patience (epochs) on val-loss")
    # ── combined test eval after training (one job, Vimal-style) ──
    p.add_argument("--eval-after", action="store_true",
                   help="after training, load best.pt and run the test eval in-process, "
                        "writing test_results.json (combined train+eval job)")
    p.add_argument("--eval-k", type=int, default=10, help="test samples/window (EM is costly; Vimal uses 8)")
    p.add_argument("--eval-steps", default="1000",
                   help="comma-separated EM reverse steps for the locked test (GAD uses ~5000)")
    p.add_argument("--eval-max-windows", type=int, default=0,
                   help="cap test windows in the --eval-after path (0=all; bound the EM eval cost)")
    p.add_argument("--eval-gen-batch", type=int, default=1024,
                   help="max trajectories per EM reverse call in the test eval. 1024 fits the "
                        "A40/L40 (46GB) across datasets; 4096 OOMs (the UGnet spatial conv spikes). "
                        "raise cautiously and watch memory")
    # ── learning-rate range test (Smith): sweep lr exponentially over a few
    # hundred steps to locate the good lr band, then exit (no full training).
    p.add_argument("--lr-find", type=int, default=0,
                   help="if >0, run an lr range test for this many steps and exit")
    p.add_argument("--lr-find-min", type=float, default=1e-6)
    p.add_argument("--lr-find-max", type=float, default=1.0)
    # ── periodic full EM reverse eval (Vimal protocol): every --val-every epochs,
    #    sample k trajectories FROM THE PRIOR, compare to val truth (MAE/RMSE/CRPS +
    #    Wasserstein), and select best.pt by CRPS. kept LEAN because the EM reverse is
    #    ~1000+ steps (stiff prior) — a big windows×k here dominates training wall-time.
    p.add_argument("--val-every", type=int, default=10,
                   help="run the full EM reverse val eval every N epochs (0 disables selection)")
    p.add_argument("--val-k", type=int, default=2,
                   help="samples/window for the periodic val eval (kept small — EM is costly)")
    p.add_argument("--val-steps", type=int, default=200,
                   help="EM reverse steps for the val eval (auto-floored per prior stiffness)")
    p.add_argument("--val-windows", type=int, default=48,
                   help="fully-observed val windows for the eval (small — cost = windows·k·steps)")
    p.add_argument("--val-proj", type=int, default=64,
                   help="random projections for the sliced wasserstein-1")
    p.add_argument("--smoke", action="store_true", help="1 epoch, tiny batch, wandb disabled")
    p.add_argument("--fresh", action="store_true", help="delete existing checkpoint dir + restart")
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="if > 0, cap training batches per epoch (for smoke runs)")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 2
        args.batch_size = 8
        args.max_train_batches = max(args.max_train_batches, 5)
        # exercise the full EM reverse val eval cheaply; 200 EM steps keeps it stable on
        # the real prior (EM blows up below ~200) while staying fast on 16 windows.
        args.val_every = 1
        args.val_k = 2
        args.val_steps = 200
        args.val_windows = 16
        args.val_proj = 16
        args.eval_k = 4
        args.eval_steps = "200"
        args.eval_max_windows = 64        # cap the combined-job test eval so the smoke stays fast
        os.environ.setdefault("WANDB_MODE", "disabled")

    cfg = vars(args).copy()
    cfg["backbone"] = args.backbone
    cfg["prior"] = args.prior
    cfg_hash = hash_config({k: v for k, v in cfg.items() if k not in _CFG_HASH_EXCLUDE})
    run_id = f"diff_{args.backbone}_{args.dataset}_seed{args.seed}_{cfg_hash}"

    ckpt_dir = paths.diffusion_dir(args.dataset, args.prior, args.seed, exp=args.exp,
                                   backbone=args.backbone)
    if args.fresh and ckpt_dir.exists():
        import shutil
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    # timestamped logging → stdout (slurm .out) AND the per-run train.log (Vimal-style).
    setup_logging(ckpt_dir / "train.log")
    log.info(f"===== run {run_id} =====")
    log.info(f"run dir: {ckpt_dir}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    log.info(f"device={device}  run_id={run_id}")

    set_seed(args.seed)

    # ── Preflight: fail fast (not hours in) if an input is missing.
    _ex = (f"run scripts/build_signal_cache.py --dataset {args.dataset} first"
           if args.backbone == "raw" else
           f"run scripts/gwn_extract_residuals.py --variant {args.dataset} "
           f"--seed {args.seed} --backbone {args.backbone} --checkpoint <ckpt> first")
    paths.require_inputs([
        (paths.residual_npz(args.dataset, "train", args.seed, args.backbone), _ex),
        (paths.residual_npz(args.dataset, "val", args.seed, args.backbone), _ex),
        (paths.adj_npz(args.dataset), "run python scripts/build_datasets.py --all"),
    ])

    # ── Data (modeled in unit space; sigma_r from train, reused for val). backbone='raw'
    # → SignalDataset (future signal target + history conditioning); else ResidualDataset.
    sigma_r = train_residual_sigma(args.dataset, args.seed, args.backbone)
    train_ds = make_dataset(args.dataset, "train", args.seed, args.backbone, sigma_r)
    val_ds = make_dataset(args.dataset, "val", args.seed, args.backbone, sigma_r)
    log.info(f"train={len(train_ds)}  val={len(val_ds)}  "
             f"sigma_r={sigma_r:.4f} (residuals modeled in unit space)")

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
                       spatial=args.spatial, stiffness_cap=(args.stiffness_cap or None),
                       backbone=args.backbone)
    prior_t = prior_to_torch(prior, device=device, dtype=torch.float32)
    # GAD-literal FCPS schedule: fixed S=c_0, α, c_min (no per-dataset auto-derivation).
    # the θ→GAD-scale reparam already set θ_min=γ=0.8, so a fixed S=7 reproduces GAD's
    # mixing (2·S·θ_min≈11.2) on every dataset.
    schedule = FCPSSchedule(c_min=args.c_min, c_0=args.c_0, alpha=args.alpha, T=1.0)
    log.info(f"schedule (GAD-literal): S=c_0={schedule.c_0:.2f} "
             f"α={schedule.alpha:.1f} c_min={schedule.c_min:.2f}")
    log.info(
        f"prior N={prior.N} H={prior.H} γ={prior.gamma:.4f}  "
        f"θ_min={float(prior.theta.min()):.4f}  θ_max={float(prior.theta.max()):.2f}  "
        f"2·S·θ_min={2*schedule.c_0*float(prior.theta.min()):.2f} (GAD target ≈11.2)  "
        f"train noise c̄ = S·u^0.65 ∈ [0, {schedule.c_0:.3g}]"
    )
    # EM stability floor: explicit EM diverges to NaN when the step count is too low for
    # this prior's stiffness (θ_max·S). floor both the periodic val eval and the test eval
    # so every dataset/backbone runs correctly (the stiff raw prior needs ~1000+ steps).
    em_floor = min_em_steps(prior, schedule)
    val_steps_eff = max(args.val_steps, em_floor)
    if val_steps_eff != args.val_steps:
        log.info(f"EM stability floor: val eval steps "
                 f"{args.val_steps} → {val_steps_eff} (θ_max·S={em_floor})")

    # ── Model. backbone='raw' → history-conditioned SignalDenoiser (T_h from the
    # conditioning width); else the forecast-conditioned R0Denoiser.
    A = load_adjacency(args.dataset)
    if args.backbone == "raw":
        T_h = train_ds.Y_hat.shape[-1]
        denoiser = SignalDenoiser(
            N=prior.N, H=prior.H, T_h=T_h, adjacency=A,
            d_h=args.d_h, channel_multipliers=tuple(args.channel_multipliers),
            n_blocks=args.n_blocks, device=str(device),
        ).to(device)
    else:
        denoiser = R0Denoiser(
            N=prior.N, H=prior.H, adjacency=A,
            d_h=args.d_h, channel_multipliers=tuple(args.channel_multipliers),
            n_blocks=args.n_blocks, device=str(device),
        ).to(device)
    log.info(f"denoiser params: {num_parameters(denoiser):,}")

    # GAD's TV penalty (off by default; residuals are rough, may be misspecified).
    A_t = torch.as_tensor(A, device=device, dtype=torch.float32) if args.tv_weight else None

    # AdamW + grad-clip 1.0, matching GAD (exp_gasde.py). lr stays at our scale-
    # appropriate default (batch/epochs differ from GAD's 128/5000).
    optim = torch.optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # ReduceLROnPlateau on the val-CRPS plateau (Vimal/DiffSTG use a plateau scheduler
    # rather than a flat lr; halve lr after lr_patience epochs without CRPS improvement).
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=args.lr_factor, patience=args.lr_patience)

    # lr range test: sweep lr, record loss, exit before the real training loop.
    if args.lr_find:
        lr_range_test(
            train_loader, denoiser=denoiser, prior_t=prior_t, schedule=schedule,
            tv_weight=args.tv_weight, A_t=A_t, optim=optim, device=device, n_steps=args.lr_find,
            lr_min=args.lr_find_min, lr_max=args.lr_find_max,
            out_path=ckpt_dir / "lr_find.json",
        )
        return

    # ── Resume (if a checkpoint exists)
    start_epoch = 0
    best_crps = float("inf")
    best_epoch = -1
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    if ckpt_path is not None and not args.fresh:
        ckpt = load_checkpoint(ckpt_path)
        denoiser.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        if sched is not None and ckpt.get("scheduler_state_dict") is not None:
            sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_crps = ckpt["best_val_mae"]              # field name is legacy; holds best val-CRPS
        restore_rng_states(ckpt["rng_states"])
        _hist = ckpt_dir / "validation_history.json"
        if _hist.exists():
            best_epoch = int(json.loads(_hist.read_text()).get("best_epoch", -1))
        log.info(f"resumed from epoch {start_epoch}  best_val_CRPS={best_crps:.4f}")

    # ── wandb
    if os.environ.get("WANDB_MODE", "online") != "disabled":
        init_wandb(
            cfg, stage="diffusion", dataset=args.dataset, seed=args.seed,
            prior=args.prior, backbone=args.backbone, exp=args.exp,
        )
    import wandb  # safe — always installed

    # ── Clean (fully-observed) val windows: used BOTH for the CRPS that selects
    # best.pt and for the richer diagnostic. fully observed → clean eigenbasis
    # rotation and no masking needed for CRPS. dedicated rng for cross-epoch
    # comparability (mps can't drive randn from a device generator → cpu fallback).
    val_gen = (
        None if device.type == "mps"
        else torch.Generator(device=device.type).manual_seed(args.seed + 999)
    )
    full_valid = val_ds.mask.reshape(len(val_ds), -1).all(dim=1)       # (Nval,)
    valid_idx = full_valid.nonzero(as_tuple=True)[0]
    n_val = min(args.val_windows, valid_idx.numel())
    val_R0, val_Yhat = val_ds.R0[valid_idx[:n_val]], val_ds.Y_hat[valid_idx[:n_val]]
    log.info(f"per-epoch val_loss (GAD SNR-MSE) for the curve + LR plateau; "
             f"FULL EM reverse eval every {args.val_every} epochs on {n_val} clean windows "
             f"(k={args.val_k}, steps={val_steps_eff}) → best.pt selected by CRPS (Vimal protocol); "
             f"{float(full_valid.float().mean()):.2f} of val fully observed")

    # per-run artefacts (easy navigation): validation_history.json (full, for resume) +
    # history.csv (flat, one row/epoch — open in a spreadsheet or grep/plot directly).
    hist_path = ckpt_dir / "validation_history.json"
    csv_path = ckpt_dir / "history.csv"
    _CSV_COLS = ["timestamp", "epoch", "train_loss", "val_loss", "lr", "time_s",
                 "val_mae", "val_rmse", "val_crps", "val_w1", "val_energy"]
    if not csv_path.exists():                       # keep one continuous file across resumes
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_COLS)
    history: list[dict] = []
    wandb_on = os.environ.get("WANDB_MODE", "online") != "disabled"

    def _maybe_eval(epoch: int, is_last: bool):
        """full EM reverse eval + comparison, every --val-every epochs (and on the last)."""
        if not args.val_every or (epoch % args.val_every != 0 and not is_last):
            return None
        res = periodic_val_eval(
            val_R0, val_Yhat, denoiser=denoiser, prior=prior, schedule=schedule,
            device=device, n_steps=val_steps_eff, k=args.val_k, n_proj=args.val_proj,
            sigma_r=sigma_r, generator=val_gen,
        )
        _log_val(epoch, res, to_wandb=wandb_on)
        return res

    log.info(f"# {run_id} | {args.dataset} prior={args.prior} backbone={args.backbone} "
             f"seed={args.seed} | select=val_CRPS (full EM reverse every {args.val_every} ep) | "
             f"schedule S={schedule.c_0:.1f} α={schedule.alpha:.0f} c_min={schedule.c_min:.2f} | "
             f"reverse=EM")
    log.info(f"epoch | train | val_loss | lr | time   "
             f"(+ full EM eval MAE/RMSE/CRPS every {args.val_every} ep)")

    # ── Train. per-epoch val_loss drives the curve + LR plateau; best.pt is selected by
    # the CRPS of the periodic full EM reverse eval (Vimal's protocol, GAD's sampler).
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
                tv_weight=args.tv_weight, A_t=A_t, device=device, generator=None,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optim.step()
            running += loss.detach().item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        # cheap per-epoch val loss (GAD SNR-weighted MSE, sampler-free): the training
        # curve + the LR plateau signal (NOT the checkpoint-selection metric).
        vloss = validate(val_loader, denoiser=denoiser, prior_t=prior_t, schedule=schedule,
                         tv_weight=args.tv_weight, A_t=A_t, device=device)
        sched.step(vloss)
        lr_now = optim.param_groups[0]["lr"]
        elapsed = time.time() - t0
        log.info(f"{epoch:04d}  | train={train_loss:7.4f} | "
                 f"val_loss={vloss:7.4f} | {lr_now:.1e} | {elapsed:5.1f}s")

        ev = _maybe_eval(epoch, is_last=(epoch == args.epochs - 1))

        row = {"epoch": epoch, "train_loss": train_loss,
               "val_loss": vloss, "lr": lr_now, "time": elapsed}
        improved = False
        if ev is not None:
            row.update({"val_mae": ev["mae"], "val_rmse": ev["rmse"], "val_crps": ev["crps"],
                        "val_w1": ev["w1"], "val_energy": ev["energy"]})
            if ev["crps"] < best_crps:
                best_crps, best_epoch, improved = ev["crps"], epoch, True
        history.append(row)
        hist_path.write_text(json.dumps(
            {"select_metric": "crps", "best_epoch": best_epoch, "history": history}, indent=2))
        # flat per-epoch row for easy scanning/plotting (blank cells on non-eval epochs)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch,
                 f"{train_loss:.6f}", f"{vloss:.6f}", f"{lr_now:.3e}", f"{elapsed:.1f}"]
                + [f"{row[c]:.6f}" if c in row else "" for c in _CSV_COLS[6:]])
        if wandb_on:
            wandb.log({"train/loss": train_loss, "val/loss": vloss,
                       "lr": lr_now, "epoch": epoch})

        save_checkpoint(ckpt_dir / "last.pt", denoiser, optim, sched,
                        epoch=epoch, best_val_mae=best_crps, rng_states=get_rng_states())
        if improved:
            save_checkpoint(ckpt_dir / "best.pt", denoiser, optim, sched,
                            epoch=epoch, best_val_mae=best_crps, rng_states=get_rng_states())
            log.info(f"  [saved best] epoch={epoch} val_CRPS={best_crps:.5f}")

        # early stop (Vimal): no CRPS improvement for `patience` epochs since the best.
        if best_epoch >= 0 and (epoch - best_epoch) >= args.patience:
            log.info(f"[early stop] epoch {epoch} (patience={args.patience}, "
                     f"best_epoch={best_epoch}, best_val_CRPS={best_crps:.5f})")
            break

    log.info(f"[done] best_epoch={best_epoch} best_val_CRPS={best_crps:.5f}")

    # ── Combined test eval (one job): load best.pt, run the locked EM test, write json + summary
    if args.eval_after:
        _run_test_eval(args, device, best_epoch, run_id)


if __name__ == "__main__":
    main()
