"""train the diffusion denoiser (predicts the clean residual R0).

inputs are the cached residuals (src.utils.paths.residual_npz) and the noise
prior from src.diffusion.prior.load_prior (kronecker or isotropic).

each training step (GAD scheme, exp_gasde.py):
  1. take a batch of (R0, cond, mask) from the residual cache. masked coords are
     zeroed at the input; with --cond-past the conditioning is [R_prev | Y_hat_z]
     (the previous window's residuals carry recent backbone-error volatility).
  2. draw the noise level c̄ = c_0·u^0.65 (power-law, biased toward high noise)
  3. forward-sample R_t = sample_R_t_from_c_bar_torch(R0, c̄, ...)
  4. recover t = schedule.t_of_c_bar(c̄) and predict R0_hat = denoiser(R_t, t, cond)
  5. GAD's snr-weighted masked mse (default; --loss mse for the plain ablation)

schedule: kronecker → FCPS (GAD-literal S=7, α=4, c_min=0.1); isotropic → UL
(linear rate, slope=2·S=14, same total heat — no slow graph modes to protect).

selection (DiffSTG-style, see DECISIONS.md 2026-07-08): every --fast-val-every
epochs, sampled forecast CRPS on fixed clean val windows with fixed noise —
ancestral reverse at --fast-val-steps, EMA(--ema) weights. that signal drives
best_crps.pt (EMA weights, the shipped model), early stopping (--patience) and
the LR plateau. the denoising val loss is a DASHBOARD only (best.pt kept as the
val-loss-track artifact). every --val-every epochs a heavier step-count sweep
(EM over --val-steps-list + ancestral over --val-anc-steps, CRN) provides the
step-choice evidence. with --final-val, the job ends by re-running the full
sweep on EACH checkpoint and running the FULL val-set eval per checkpoint: EM
at its chosen count (k=--val-k, GAD parity) and ancestral at its chosen count
(k=--eval-k).

usage:
    python scripts/train_diffusion.py --dataset metrla --seed 42 --final-val
    python scripts/train_diffusion.py --dataset metrla --seed 42 --smoke
    python scripts/train_diffusion.py --dataset metrla --seed 42 --fresh
"""
from __future__ import annotations

import argparse
import math
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

# TF32 matmuls: ~1.5-2x on the L40/A40 denoiser (spatial conv + linears), and the
# EM sampling diagnostics are monte-carlo anyway (precision loss ≪ sampling noise).
# the eval script sets the same flags — keep train + eval numerics consistent.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.data.adjacency import REPAIR_GRAPH, load_adjacency
from src.data.residual_dataset import (
    make_dataset, residual_path, sigma_to_torch, train_residual_sigma,
)
from src.diffusion.denoiser import GCNNDenoiser, R0Denoiser, num_parameters
from src.diffusion.losses import snr_weighted_mse
from src.diffusion.forward_fcps import (
    FCPSSchedule,
    derived_floor,
    derived_steps,
    fcps_rate_torch,
    prior_to_torch,
    sample_log_c_bar_torch,
    sample_R_t_from_c_bar_torch,
)
from src.diffusion.prior import load_prior
from src.diffusion.reverse import min_em_steps, reverse_sample_ancestral, reverse_sample_em
from src.eval.crps import empirical_crps, quantile_crps_num

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
from src.utils.ema import EMA
from src.utils.logging_setup import Throttle, get_logger, setup_logging
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


def lr_range_test(loader, *, denoiser, prior_t, schedule,
                  optim, device, n_steps, lr_min, lr_max, out_path) -> None:
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
                          device=device, generator=None)
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


def _loss(R0_hat, R0, mask, t, schedule):
    """the training loss: GAD's rate-weighted masked mse, w = 1/(1 + 2·c(t))."""
    return snr_weighted_mse(R0_hat, R0, mask, fcps_rate_torch(t, schedule))


def train_step(
    batch: dict[str, torch.Tensor],
    *,
    denoiser: R0Denoiser,
    prior_t,
    schedule,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    R0 = batch["R0"].to(device, non_blocking=True)         # (B, N, H)
    Y_hat = batch["Y_hat"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    B = R0.shape[0]

    # draw the training noise level c̄ log-uniformly over [c̄_lo, c_0]: even coverage
    # across the noise decades, so the stiff modes are trained as often as the slow ones.
    c_bar = sample_log_c_bar_torch(schedule.c_lo, schedule.c_0, B, device, R0.dtype,
                                   generator=generator)
    R_t = sample_R_t_from_c_bar_torch(R0, c_bar, prior_t, generator=generator)
    t = schedule.t_of_c_bar(c_bar)

    R0_hat = denoiser(R_t, t, Y_hat)
    return _loss(R0_hat, R0, mask, t, schedule)


# fixed seed for the val-loss noise draws: every epoch scores the SAME (c̄, Z)
# draws, so the val loss is comparable across epochs and best-by-val-loss
# selection doesn't jitter with the noise lottery.
_VAL_NOISE_SEED = 777


def _parse_steps(spec: str, floor: int) -> list[int]:
    """comma list → ints, each raised to the EM stability floor, deduped in order."""
    out: list[int] = []
    for x in str(spec).split(","):
        if x.strip():
            ns = max(int(x), floor)
            if ns not in out:
                out.append(ns)
    return out


@torch.no_grad()
def validate(
    loader: DataLoader,
    *,
    denoiser: R0Denoiser,
    prior_t,
    schedule,
    device: torch.device,
    n_t_samples: int = 4,
) -> float:
    """Average validation loss over the loader (same loss as training, for the
    curve + LR plateau + best.pt selection). deterministic: a dedicated generator
    is re-seeded every call, so all epochs see identical noise draws (mps has no
    device generator → falls back to fresh draws there; production is cuda).
    """
    denoiser.eval()
    gen = (None if device.type == "mps"
           else torch.Generator(device=device.type).manual_seed(_VAL_NOISE_SEED))
    total = 0.0
    n = 0
    for batch in loader:
        R0 = batch["R0"].to(device, non_blocking=True)
        Y_hat = batch["Y_hat"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        B = R0.shape[0]
        batch_loss = 0.0
        for _ in range(n_t_samples):
            c_bar = sample_log_c_bar_torch(schedule.c_lo, schedule.c_0, B, device,
                                           R0.dtype, generator=gen)
            R_t = sample_R_t_from_c_bar_torch(R0, c_bar, prior_t, generator=gen)
            t = schedule.t_of_c_bar(c_bar)
            R0_hat = denoiser(R_t, t, Y_hat)
            batch_loss += float(_loss(R0_hat, R0, mask, t, schedule))
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
    ypred_fo: torch.Tensor,
    ytrue_fo: torch.Tensor,
    *,
    denoiser,
    prior,
    schedule,
    device: torch.device,
    steps_list: list[int],
    k: int,
    n_proj: int,
    sigma_r,
    crn_seed: int,
    gen_batch: int = 1024,
    anc_list: Optional[list[int]] = None,
) -> dict:
    """full EM reverse diagnostic, swept over reverse step counts (every --val-every
    epochs). for EACH n_steps in steps_list: draw k trajectories from the prior via
    the GAD Euler–Maruyama reverse SDE on the SAME clean val windows, reconstruct the
    forecast (y_pred + σ_r·sample, clipped at 0), and score it against the raw truth
    with Vimal's conventions (y==0 dropped, metrics on de-normalized data):

      MAE / RMSE / MAPE of the k-sample mean · normalized 19-quantile CRPS (the
      leaderboard metric) · raw-unit energy CRPS · sliced-W1 + energy distance
      (residual space, first chunk).

    common random numbers: the generator is re-seeded to crn_seed before every step
    count, so all step counts share the prior draw + noise sequence and the ranking
    reflects discretization error, not the noise lottery. this is how the best
    reverse step count is chosen for the final full-val eval.

    anc_list: also sweep the closed-form ancestral sampler at these step counts
    (cheap; results under out["steps_anc"]) — cross-validates the two samplers.
    """
    denoiser.eval()
    W, N, H = R0_fo.shape
    sig = sigma_to_torch(sigma_r, device)                    # (1,1) or (N,1)
    R0 = R0_fo.to(device) * sig                              # raw residual truth
    Y = Y_hat_fo.to(device)                                  # conditioning (z-scored)
    yp = ypred_fo.to(device)                                 # raw backbone forecast (0 for 'raw')
    yt = ytrue_fo.to(device)                                 # raw ground truth
    Mz = yt.abs() > 1e-6                                     # Vimal null_val=0 mask
    chunk = max(1, gen_batch // k)
    out: dict = {"n_windows": W, "k": k, "steps": {}, "steps_anc": {}}

    jobs = ([("em", ns) for ns in steps_list]
            + [("anc", ns) for ns in (anc_list or [])])
    t_sweep = time.time()
    thr = Throttle(120.0)          # ONE self-contained progress line per ~2 min, total
    for si, (samp_kind, n_steps) in enumerate(jobs):
        gen = (None if device.type == "mps"
               else torch.Generator(device=device.type).manual_seed(crn_seed))
        def _cb(i, _c, _R, _ns=n_steps, _kind=samp_kind, _si=si):
            if thr.ready():
                log.info(f"  [val-eval] {_kind}{_ns} step {i + 1}/{_ns} · "
                         f"job {_si + 1}/{len(jobs)} · {time.time() - t_sweep:.0f}s elapsed")
        rev_fn = reverse_sample_em if samp_kind == "em" else reverse_sample_ancestral
        mae_n = torch.zeros(H); mse_n = torch.zeros(H); mape_n = torch.zeros(H)
        crps_e = torch.zeros(H); qnum = torch.zeros(H); qden = torch.zeros(H)
        cnt = torch.zeros(H)
        gen_first = truth_first = None
        for s in range(0, W, chunk):
            e = min(s + chunk, W); b = e - s
            samp, _ = rev_fn(prior, schedule, denoiser, Y[s:e].repeat(k, 1, 1),
                             n_steps=n_steps, generator=gen, step_callback=_cb)
            samp = samp.view(k, b, N, H) * sig               # raw residual cloud
            fc = (yp[s:e].unsqueeze(0) + samp).clamp(min=0.0)  # raw forecast cloud
            m = Mz[s:e].float()
            mean = fc.mean(0)
            err = mean - yt[s:e]
            mae_n += (err.abs() * m).sum(dim=(0, 1)).cpu()
            mse_n += (err.pow(2) * m).sum(dim=(0, 1)).cpu()
            mape_n += (err.abs() / yt[s:e].abs().clamp(min=1e-6) * m).sum(dim=(0, 1)).cpu()
            crps_e += (empirical_crps(fc, yt[s:e]) * m).sum(dim=(0, 1)).cpu()
            qnum += quantile_crps_num(fc, yt[s:e], Mz[s:e]).cpu()
            qden += (yt[s:e].abs() * m).sum(dim=(0, 1)).cpu()
            cnt += m.sum(dim=(0, 1)).cpu()
            if gen_first is None:                            # first chunk for W1/energy
                gen_first, truth_first = samp, R0[s:e]

        w1_h, ed_h = [], []
        for h in range(H):
            gen_c = gen_first[:, :, :, h].reshape(-1, N)
            w1_h.append(sliced_wasserstein1(gen_c, truth_first[:, :, h], n_proj, gen))
            ed_h.append(energy_distance(gen_c, truth_first[:, :, h]))

        c = cnt.clamp(min=1.0)
        bucket = "steps" if samp_kind == "em" else "steps_anc"
        out[bucket][int(n_steps)] = {
            "mae": float(mae_n.sum() / c.sum()),
            "rmse": float((mse_n.sum() / c.sum()) ** 0.5),
            "mape": float(mape_n.sum() / c.sum()),           # fraction; ×100 to display
            "crps": float(qnum.sum() / (qden.sum().clamp(min=1e-8) * 19.0)),  # normalized (Vimal)
            "crps_energy": float(crps_e.sum() / c.sum()),    # raw units
            "mae_h": (mae_n / c).tolist(),
            "w1": float(sum(w1_h) / H), "energy": float(sum(ed_h) / H),
        }

    denoiser.train()
    return out


def best_steps(sweep: dict, tol: float = 0.01) -> int:
    """fewest reverse steps within `tol` of the sweep's best CRPS — EM improves
    then plateaus with steps, so take the cheap end of the plateau."""
    steps = sorted(int(s) for s in sweep["steps"])
    best = min(sweep["steps"][s]["crps"] for s in steps)
    for s in steps:
        if sweep["steps"][s]["crps"] <= best * (1.0 + tol):
            return s
    return steps[-1]


def _log_val(epoch: int, res: dict, *, to_wandb: bool) -> None:
    """print the step-sweep table of the periodic diagnostic (EM + ancestral rows);
    push to wandb."""
    log.info(f"  [val-eval] {res['n_windows']} windows, k={res['k']} | "
             f"{'sampler':>8s} {'steps':>6s} {'MAE':>8s} {'RMSE':>8s} {'MAPE%':>7s} "
             f"{'CRPS':>7s} {'CRPSe':>7s} {'W1':>7s} {'ED':>7s}")
    for bucket, tag in (("steps", "em"), ("steps_anc", "anc")):
        for ns in sorted(res.get(bucket, {})):
            r = res[bucket][ns]
            log.info(f"  [val-eval] {'':13s}{tag:>8s} {ns:>6d} {r['mae']:>8.3f} "
                     f"{r['rmse']:>8.3f} {100*r['mape']:>7.2f} {r['crps']:>7.4f} "
                     f"{r['crps_energy']:>7.3f} {r['w1']:>7.3f} {r['energy']:>7.3f}")
    bs = best_steps(res)
    log.info(f"  [val-eval] best EM steps (≤1% of min CRPS): {bs} "
             f"(CRPS={res['steps'][bs]['crps']:.4f})")
    if not to_wandb:
        return
    import wandb
    wb = {"epoch": epoch, "val/best_steps": bs}
    for ns, r in res["steps"].items():
        for key in ("mae", "rmse", "mape", "crps", "w1", "energy"):
            wb[f"val/{key}_s{ns}"] = r[key]
    for ns, r in res.get("steps_anc", {}).items():
        wb[f"val/crps_anc{ns}"] = r["crps"]
        wb[f"val/mae_anc{ns}"] = r["mae"]
    b = res["steps"][bs]
    wb.update({"val/mae": b["mae"], "val/rmse": b["rmse"], "val/mape": b["mape"],
               "val/crps": b["crps"], "val/w1": b["w1"], "val/energy": b["energy"]})
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
    # (sampler, n_steps) grid. EM carries the GAD-parity headline but is floored at
    # max(200, θ_max·c₀), so the few-step comparison can only be made on the ancestral
    # sampler — run both families and tag every row with the sampler that produced it.
    em_steps = [int(x) for x in str(args.eval_steps).split(",") if x.strip()]
    anc_steps = [int(x) for x in str(args.eval_anc_steps).split(",") if x.strip()]
    grid = [("ancestral", n) for n in anc_steps] + [("em", n) for n in em_steps]
    log.info(f"[test] {args.eval_ckpt} (epoch {best_epoch}); grid={grid}")
    results = []
    for sampler, ns in grid:
        res = evaluate(args.dataset, args.prior, args.seed, exp=args.exp, device=device,
                       k=args.eval_k, n_steps=ns, alpha=0.1,
                       w1_windows=512, w1_samples=4, w1_proj=128, gen_batch=args.eval_gen_batch,
                       max_windows=args.eval_max_windows, generator=gen,
                       backbone=args.backbone, mask_zero=True,
                       sampler=sampler, ckpt_name=args.eval_ckpt)
        # evaluate() may raise n_steps to the EM floor; record what was ASKED and what RAN
        res = {**res, "sampler": sampler, "requested_steps": ns}
        results.append(res)
        log.info(f"  [test {sampler}@{ns}] MAE={res['mae']:.4f} RMSE={res['rmse']:.4f} "
                 f"MAPE={res['mape']:.4f} CRPS={res['crps']:.4f} "
                 f"Winkler={res['winkler']:.4f} cov={res['coverage']:.3f}")
    payload = {"run_id": run_id, "dataset": args.dataset, "prior": args.prior,
               "backbone": args.backbone, "seed": args.seed, "exp": args.exp,
               "best_epoch": best_epoch, "select_metric": "crps",
               "eval_ckpt": args.eval_ckpt, "test_results": results}
    (out_dir / "test_results.json").write_text(json.dumps(payload, indent=2))
    log.info(f"[test] wrote {out_dir / 'test_results.json'}")
    _write_run_summary(payload, out_dir)


def _run_final_val(args, device, best_epoch, run_id, *, sigma_r, val_R0, val_Yhat,
                   val_ypred, val_ytrue) -> None:
    """final phase of the one-command lifecycle, for BOTH selection tracks
    (best.pt = min val loss, best_crps.pt = min sweep CRPS). per checkpoint:

      1. step sweep on the selected weights — EM over --final-steps-list AND the
         ancestral sampler over --final-anc-steps (CRN, k=--val-k).
      2. FULL val-set eval, twice: EM at the chosen count (k=--val-k, the GAD-parity
         headline) and ancestral at ITS chosen count (k=--eval-k, cheap enough for
         the larger k — the number that predicts what a test eval would give).

    writes one val_results.json with a per-checkpoint block + val_summary.txt.
    the test split is NOT touched."""
    try:                                                        # heavy import, deferred
        from eval_diffusion import evaluate, load_run            # run as a script
    except ModuleNotFoundError:
        from scripts.eval_diffusion import evaluate, load_run    # imported as a package

    ckpt_dir = paths.diffusion_dir(args.dataset, args.prior, args.seed,
                                   exp=args.exp, backbone=args.backbone)
    anc_list = _parse_steps(args.final_anc_steps, 1)
    blocks: dict = {}
    for ckpt_name, tag in (("best.pt", "best_loss"), ("best_crps.pt", "best_crps")):
        if not (ckpt_dir / ckpt_name).exists():
            log.info(f"[final-val] {ckpt_name} not found — skipping the {tag} track")
            continue
        denoiser, prior_obj, schedule_obj, _cfg, ckpt_epoch = load_run(
            args.dataset, args.prior, args.seed, args.exp, device, args.backbone,
            ckpt_name=ckpt_name)
        steps_list = _parse_steps(args.final_steps_list,
                                  min_em_steps(prior_obj, schedule_obj))
        log.info(f"[final-val] {ckpt_name} (epoch {ckpt_epoch}): EM sweep {steps_list} "
                 f"+ ancestral sweep {anc_list} on {val_R0.shape[0]} clean windows")
        sweep = periodic_val_eval(
            val_R0, val_Yhat, val_ypred, val_ytrue, denoiser=denoiser, prior=prior_obj,
            schedule=schedule_obj, device=device, steps_list=steps_list, k=args.val_k,
            n_proj=args.val_proj, sigma_r=sigma_r, crn_seed=args.seed + 999,
            anc_list=anc_list,
        )
        _log_val(ckpt_epoch if ckpt_epoch is not None else -1, sweep, to_wandb=False)
        ns_em = best_steps(sweep)
        ns_anc = best_steps({"steps": sweep["steps_anc"]}) if sweep["steps_anc"] else None
        log.info(f"[final-val] {tag}: chosen EM steps={ns_em}"
                 + (f", ancestral steps={ns_anc}" if ns_anc else ""))

        gen = (None if device.type == "mps"
               else torch.Generator(device=device.type).manual_seed(args.seed))
        res_em = evaluate(args.dataset, args.prior, args.seed, exp=args.exp, device=device,
                          k=args.val_k, n_steps=ns_em, alpha=0.1, split="val",
                          w1_windows=(args.eval_max_windows or 512),
                          w1_samples=4, w1_proj=128,
                          gen_batch=args.eval_gen_batch, max_windows=args.eval_max_windows,
                          generator=gen, backbone=args.backbone, mask_zero=True,
                          sampler="em", ckpt_name=ckpt_name)
        res_anc = None
        if ns_anc:
            gen = (None if device.type == "mps"
                   else torch.Generator(device=device.type).manual_seed(args.seed))
            res_anc = evaluate(args.dataset, args.prior, args.seed, exp=args.exp,
                               device=device, k=args.eval_k, n_steps=ns_anc, alpha=0.1,
                               split="val", w1_windows=(args.eval_max_windows or 512),
                               w1_samples=4, w1_proj=128,
                               gen_batch=args.eval_gen_batch,
                               max_windows=args.eval_max_windows,
                               generator=gen, backbone=args.backbone, mask_zero=True,
                               sampler="ancestral", ckpt_name=ckpt_name)
        blocks[tag] = {"ckpt": ckpt_name, "epoch": ckpt_epoch,
                       "steps_sweep": sweep, "chosen_steps": ns_em,
                       "chosen_anc_steps": ns_anc,
                       "full_val": res_em, "full_val_anc": res_anc}

    out_dir = paths.eval_dir(args.dataset, args.prior, args.seed,
                             exp=args.exp, backbone=args.backbone)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "dataset": args.dataset, "prior": args.prior,
               "backbone": args.backbone, "seed": args.seed, "exp": args.exp,
               "best_epoch": best_epoch,
               "rule": "fewest steps within 1% of min CRPS (per checkpoint, per sampler)",
               "checkpoints": blocks}
    (out_dir / "val_results.json").write_text(json.dumps(payload, indent=2))
    log.info(f"[final-val] wrote {out_dir / 'val_results.json'}")
    _write_val_summary(payload, out_dir)


def _write_val_summary(payload: dict, out_dir) -> None:
    """readable end-of-run summary for the val lifecycle: per selection track
    (best.pt / best_crps.pt), the step-sweep table (EM + ancestral rows) and the
    full-val metrics for both samplers. printed AND saved."""
    L: list[str] = []
    L.append("=" * 74)
    L.append(f" RUN SUMMARY (VAL)  {payload['run_id']}")
    L.append("=" * 74)
    L.append(f" dataset={payload['dataset']}  prior={payload['prior']}  "
             f"backbone={payload['backbone']}  seed={payload['seed']}  exp={payload['exp']}")
    for tag, blk in payload["checkpoints"].items():
        sweep = blk["steps_sweep"]
        L.append("-" * 74)
        L.append(f" [{tag}] {blk['ckpt']} (epoch {blk['epoch']}) — sweep on "
                 f"{sweep['n_windows']} clean windows, k={sweep['k']}:")
        L.append(f" {'sampler':>8} {'steps':>6} | {'MAE':>8} {'RMSE':>8} {'MAPE%':>7} "
                 f"{'CRPS':>7} {'W1':>7}")
        for bucket, s_tag in (("steps", "em"), ("steps_anc", "anc")):
            for ns in sorted(sweep.get(bucket, {})):
                r = sweep[bucket][ns]
                L.append(f" {s_tag:>8} {ns:>6} | {r['mae']:>8.4f} {r['rmse']:>8.4f} "
                         f"{100*r['mape']:>7.2f} {r['crps']:>7.4f} {r['w1']:>7.3f}")
        for key, label in (("full_val", "EM"), ("full_val_anc", "ancestral")):
            res = blk.get(key)
            if not res:
                continue
            L.append(f" FULL VAL [{label} @ {res['n_steps']} steps, k={res['k_samples']}, "
                     f"{res['n_test_windows']} windows]:")
            L.append(f"   MAE={res['mae']:.4f}  RMSE={res['rmse']:.4f}  "
                     f"MAPE%={100*res['mape']:.2f}  CRPS={res['crps']:.4f}  "
                     f"Winkler={res['winkler']:.4f}  cov={res['coverage']:.3f}")
            mae_h = res.get("mae_h", [])
            ph = "  ".join(f"h{h}={mae_h[h-1]:.3f}" for h in _HL_HORIZONS if h <= len(mae_h))
            L.append(f"   per-horizon MAE:  {ph}")
    L.append("=" * 74)
    text = "\n".join(L)
    (out_dir / "val_summary.txt").write_text(text + "\n")
    for line in L:
        log.info(line)
    log.info(f"[final-val] wrote {out_dir / 'val_summary.txt'}")


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
    L.append(f" ckpt={payload.get('eval_ckpt', 'best.pt')}  select={payload['select_metric']}  "
             f"best_epoch={payload['best_epoch']}")
    L.append("-" * 72)
    L.append(f" {'sampler':>9} {'n_steps':>7} | {'MAE':>8} {'RMSE':>8} {'MAPE':>7} {'CRPS':>7} "
             f"{'Winkler':>8} {'cov':>5}")
    L.append("-" * 72)
    for r in res_list:
        # requested_steps differs from n_steps when EM's stability floor raised it
        ns = (f"{r['n_steps']}" if r.get("requested_steps", r["n_steps"]) == r["n_steps"]
              else f"{r['requested_steps']}→{r['n_steps']}")
        L.append(f" {r.get('sampler', 'em'):>9} {ns:>7} | {r['mae']:>8.4f} {r['rmse']:>8.4f} "
                 f"{r['mape']:>7.4f} {r['crps']:>7.4f} {r['winkler']:>8.4f} "
                 f"{r['coverage']:>5.3f}")
    L.append("-" * 72)
    if res_list:
        # reference run = most steps within the headline (EM) family if present, else most steps
        _em = [r for r in res_list if r.get("sampler", "em") == "em"]
        best = max(_em or res_list, key=lambda r: r["n_steps"])
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
    p.add_argument("--backbone", choices=("gwn", "gcn", "mlp"), default="gwn",
                   help="which backbone's residuals to model. namespaces results")
    p.add_argument("--prior", choices=("kronecker", "isotropic", "cartesian", "strong"), default="kronecker",
                   help="noise prior: kronecker Σ_T⊗Σ_S, or the isotropic σ²I control")
    p.add_argument("--gamma", type=float, default=None,
                   help="Laplacian shift γ in Σ_S=(L+γI)^-1 (kronecker only). None = GAMMA_DEFAULT")
    p.add_argument("--shrink-t", type=float, default=None,
                   help="temporal ridge ρ: μ_T ← (1-ρ)μ_T + ρ·μ̄ (kronecker only). None = SHRINK_T")
    p.add_argument("--global-edge", default=None,
                   help="kronecker only. weak all-to-all edge added to the road graph, as a "
                        "fraction of the mean road-edge weight; 'match' = choose (β, γ) on the "
                        "train residuals so the prior's spread on the NETWORK TOTAL is right "
                        "(the road graph alone puts 0.5-1%% of the variance on the all-sensors-"
                        "together direction where the residual has 3-21%%). None = road graph only.")
    p.add_argument("--exp", default="dev",
                   help="experiment tag; namespaces checkpoints + wandb group "
                        "(results/diffusion/<exp>/..., wandb group <exp>-<dataset>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--residual-seed", type=int, default=None,
                   help="stage-1 seed: which residuals/<ds>/<bb>/seed<N>/ to train on, and "
                        "which the prior is fit from. defaults to --seed (the historical "
                        "1:1 coupling). set it to PIN the backbone while --seed varies, so "
                        "multi-seed error bars measure diffusion-training stochasticity "
                        "only. that is the right control for a prior comparison: every arm "
                        "then reads byte-identical residuals, so backbone variance cancels "
                        "in the paired per-seed difference instead of inflating both arms. "
                        "it does NOT cancel if coupled — a different backbone changes the "
                        "residual SPECTRUM, which is the very axis the prior acts on.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--d-h", type=int, default=32)
    p.add_argument("--denoiser", choices=("ugnet", "gcnn"), default="ugnet",
                   help="ugnet (expressive, ~1M params) or gcnn = GAD's low-capacity graph-"
                        "filter cascade (~60k params). with gcnn the network cannot absorb the "
                        "prior's structure, isolating the corruption-prior effect (PriorGrad: "
                        "adaptive priors matter most at small capacity).")
    p.add_argument("--gcnn-hops", type=int, default=5, help="K-hop order (GAD models.py: Ks=[5,5,5])")
    p.add_argument("--gcnn-layers", type=int, default=3)
    p.add_argument("--gcnn-hidden", type=int, default=24)
    p.add_argument("--channel-multipliers", type=int, nargs="+", default=[1, 2])
    p.add_argument("--n-blocks", type=int, default=2)
    # FCPS schedule (GAD's form): S=c_0, alpha, c_min. load_prior puts θ_min=γ, so
    # 2·S·θ_min sets how completely the slowest mode mixes.
    p.add_argument("--c-min", type=float, default=0.1, help="GAD c_min (0.1)")
    p.add_argument("--c-0", type=float, default=7.0, help="GAD S = c̄(T), the total heat (7.0)")
    p.add_argument("--alpha", type=float, default=4.0, help="GAD schedule exponent α (4.0)")
    p.add_argument("--t-eps", type=float, default=1e-3, help="lower bound on t to avoid σ²=0")
    p.add_argument("--cond-past", action="store_true",
                   help="prepend the previous window's residuals to the conditioning "
                        "(recent backbone-error volatility; HySRD-Net conditions on this)")
    p.add_argument("--norm", choices=("global", "pernode", "perhorizon"), default="global",
                   help="residual scaler: one global std; per-sensor std (absorbs static "
                        "spatial heteroscedasticity); or per-horizon std (absorbs the 1.7-2.9x "
                        "error growth so the stationary GAD-ST prior matches — growth re-enters "
                        "samples at de-normalization). prior refit to match either way.")
    # ── best.pt is selected by the val LOSS (deterministic noise draws), checked every
    # epoch. EM is the only reverse process (GAD parity) and too costly to sample every
    # epoch, so EM sampling is reserved for the periodic step-sweep diagnostic + the
    # final full-val eval.
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="ReduceLROnPlateau factor on the val-loss plateau")
    p.add_argument("--lr-patience", type=int, default=5,
                   help="ReduceLROnPlateau patience (epochs) on val-loss")
    # ── combined evals after training (one job, Vimal-style) ──
    p.add_argument("--final-val", action="store_true",
                   help="after training: re-run the step sweep on best.pt, pick the "
                        "fewest steps within 1%% of min CRPS, and run the FULL val-set "
                        "eval at that count (val_results.json + val_summary.txt)")
    p.add_argument("--eval-after", action="store_true",
                   help="after training, load best.pt and run the TEST eval in-process, "
                        "writing test_results.json (only once val is trusted)")
    p.add_argument("--eval-k", type=int, default=8, help="samples/window for the full eval (Vimal uses 8)")
    p.add_argument("--eval-steps", default="1000",
                   help="comma-separated EM reverse steps for the locked test (GAD uses ~5000)")
    p.add_argument("--eval-anc-steps", default="",
                   help="comma-separated ANCESTRAL reverse steps for the test eval, run in "
                        "addition to --eval-steps. the ancestral sampler is exact for the "
                        "linear per-mode OU SDE and has NO stability floor, so it is the only "
                        "family that can answer 'does the prior help at few steps' — EM is "
                        "floored at max(200, θ_max·c₀). empty = EM only (legacy behaviour).")
    p.add_argument("--eval-ckpt", choices=("best.pt", "best_crps.pt"), default="best.pt",
                   help="checkpoint the test eval loads. best_crps.pt (EMA weights, min "
                        "sampled val CRPS) is the shipped model per the 2026-07-08 decision; "
                        "best.pt (min val loss) is the legacy default kept for comparability.")
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
    # ── periodic full EM reverse diagnostic (Vimal metrics): every --val-every epochs,
    #    sweep the reverse step counts on a few clean val windows (common random
    #    numbers), reporting MAE/RMSE/MAPE/CRPS per step count — this is where the
    #    best step count for the final eval is learned. kept LEAN (48 windows):
    #    cost per eval = windows·k·Σsteps forwards.
    p.add_argument("--val-every", type=int, default=10,
                   help="run the EM reverse step-sweep diagnostic every N epochs (0 = off)")
    p.add_argument("--val-k", type=int, default=16,
                   help="samples/window for the diagnostic AND the final full-val eval "
                        "(k=4 keeps the EM budget sane; test runs can use 8+)")
    p.add_argument("--val-steps-list", default="1000,2000,3000",
                   help="EM step counts swept DURING training (the trend; lean). the "
                        "final choice sweeps --final-steps-list on best.pt. auto-floored")
    p.add_argument("--ema", type=float, default=0.999,
                   help="EMA decay for the evaluated/shipped weights (EDM convention); "
                        "0 disables. the sampled validation + best_crps.pt use EMA weights")
    p.add_argument("--fast-val-every", type=int, default=1,
                   help="run the cheap sampled validation (ancestral, --fast-val-steps) "
                        "every N epochs; it drives selection, early stop and the LR plateau")
    p.add_argument("--fast-val-steps", type=int, default=0,
                   help="ancestral step count for the per-epoch sampled validation")
    p.add_argument("--val-anc-steps", default="100",
                   help="ancestral step counts for the periodic diagnostic (cheap trend "
                        "+ cross-validates the two samplers over training)")
    p.add_argument("--final-anc-steps", default="25,50,100,200",
                   help="ancestral step counts swept in the final-val phase")
    p.add_argument("--final-steps-list", default="1000,1500,2000,2500,3000,3500",
                   help="EM step counts swept ONCE on best.pt in --final-val (the "
                        "step-count decision). auto-floored per prior stiffness")
    p.add_argument("--val-windows", type=int, default=64,
                   help="fully-observed val windows for the diagnostic, strided across "
                        "the val period (cost = windows·k·Σsteps)")
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
        # exercise the step-sweep diagnostic + the final-val phase cheaply: ONE periodic
        # eval (last epoch), two small step counts (tests the sweep/choice logic;
        # auto-floored per stiffness), 8 windows, tiny full-val. ~5 min on mps.
        args.val_every = 2
        args.val_k = 2
        args.val_steps_list = "200,300"
        args.final_steps_list = "200,300"
        args.val_anc_steps = "20"
        args.final_anc_steps = "20,30"
        args.fast_val_steps = 20
        args.val_windows = 8
        args.val_proj = 8
        args.final_val = True
        args.eval_k = 4
        args.eval_steps = "200"
        args.eval_max_windows = 16        # caps the full-val point AND W1 phases
        os.environ.setdefault("WANDB_MODE", "disabled")

    cfg = vars(args).copy()
    cfg["backbone"] = args.backbone
    cfg["prior"] = args.prior
    cfg["gamma"] = args.gamma            # None -> load_prior uses GAMMA_DEFAULT
    # record the RESOLVED stage-1 seed, not None, so config.json always states which
    # residuals this run was trained on (it enters the cfg hash: different residuals =>
    # different run identity, while --seed alone does not).
    cfg["residual_seed"] = args.seed if args.residual_seed is None else args.residual_seed
    cfg["shrink_t"] = args.shrink_t      # None -> SHRINK_T
    cfg["denoiser"] = args.denoiser      # ugnet or gcnn (low-capacity, prior-isolating)
    cfg["gcnn_hops"] = args.gcnn_hops
    cfg["gcnn_layers"] = args.gcnn_layers
    cfg["gcnn_hidden"] = args.gcnn_hidden
    cfg["repair_graph"] = REPAIR_GRAPH             # single-component graph surgery
    cfg["zero_masked_input"] = True                # masked coords zeroed at the net INPUT
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
    _ex = (f"run scripts/extract_residuals.py --variant {args.dataset} "
           f"--seed {args.seed} --backbone {args.backbone} first")
    # stage-1 (backbone/residual) seed vs stage-2 (diffusion) seed. --residual-seed pins the
    # former so error bars over --seed isolate diffusion-training stochasticity; every arm
    # then reads byte-identical residuals and backbone variance cancels in the paired
    # difference. defaults to --seed, preserving the original 1:1 coupling.
    res_seed = args.seed if args.residual_seed is None else args.residual_seed
    paths.require_inputs([
        (paths.residual_npz(args.dataset, "train", res_seed, args.backbone), _ex),
        (paths.residual_npz(args.dataset, "val", res_seed, args.backbone), _ex),
        (paths.adj_npz(args.dataset), "run python scripts/build_datasets.py --all"),
    ])

    # ── Data (modeled in unit space; sigma_r from train, reused for val).
    sigma_r = train_residual_sigma(args.dataset, res_seed, args.backbone, norm=args.norm)
    train_ds = make_dataset(args.dataset, "train", res_seed, args.backbone, sigma_r,
                            cond_past=args.cond_past, zero_masked=True)
    val_ds = make_dataset(args.dataset, "val", res_seed, args.backbone, sigma_r,
                          cond_past=args.cond_past, zero_masked=True)
    _sig_desc = (f"{sigma_r:.4f}" if np.ndim(sigma_r) == 0
                 else f"{args.norm} [{np.min(sigma_r):.2f}..{np.max(sigma_r):.2f}]")
    log.info(f"train={len(train_ds)}  val={len(val_ds)}  sigma_r={_sig_desc} "
             f"(unit space, norm={args.norm})  cond_past={args.cond_past}  "
             f"zero_masked_input=True")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=pin,
    )

    # ── Prior + schedule. NB res_seed, not args.seed: the prior is FIT FROM the stage-1
    # residuals, so it must follow the backbone seed. otherwise varying --seed would refit
    # the prior on different data and the arms would stop being comparable across seeds.
    ge = args.global_edge
    if ge is not None and ge != "match":
        ge = float(ge)
    prior = load_prior(args.dataset, args.prior, res_seed,
                       gamma=args.gamma, shrink_t=args.shrink_t,
                       backbone=args.backbone, norm=args.norm, global_edge=ge)
    prior_t = prior_to_torch(prior, device=device, dtype=torch.float32)
    # both ends of the noise range from the prior's own spectrum: the top is already
    # prior-adaptive (θ_min → γ), the bottom is derived here so the fastest mode is
    # still clean where training and sampling stop. the step count keeps the
    # resolution per decade constant across arms.
    c_lo = derived_floor(prior)
    schedule = FCPSSchedule(c_min=args.c_min, c_0=args.c_0, alpha=args.alpha, T=1.0,
                            c_lo_override=c_lo)
    n_anc = derived_steps(prior, schedule)
    if args.fast_val_steps <= 0:
        args.fast_val_steps = n_anc
    # record the RESOLVED prior + schedule so eval rebuilds exactly this run
    cfg.update({"global_edge_beta": float(prior.global_edge), "gamma_resolved": float(prior.gamma),
                "global_edge_match_ratio": float(prior.match_ratio),
                "c_lo": float(c_lo), "anc_steps_derived": int(n_anc),
                "fast_val_steps": int(args.fast_val_steps), "val_k": int(args.val_k)})
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    log.info(f"schedule (FCPS): S=c_0={schedule.c_0:.2f} α={schedule.alpha:.1f} "
             f"c_min={schedule.c_min:.2f} c_lo={c_lo:.3g} (derived) -> "
             f"{math.log10(schedule.c_0 / c_lo):.2f} decades, {n_anc} ancestral steps")
    if args.prior == "kronecker":
        log.info(f"spatial factor: global edge β={prior.global_edge:.4g}  γ={prior.gamma:.4g}"
                 f"{'  (matched on the train residuals)' if args.global_edge == 'match' else ''}")
    log.info(
        f"prior N={prior.N} H={prior.H} γ={prior.gamma:.4f}  "
        f"θ_min={float(prior.theta.min()):.4f}  θ_max={float(prior.theta.max()):.2f}  "
        f"2·S·θ_min={2*schedule.c_0*float(prior.theta.min()):.2f}  "
        f"train noise c̄ ~ logU[{schedule.c_lo:.3g}, {schedule.c_0:.3g}]  "
        f"repair_graph={REPAIR_GRAPH}"
    )
    # EM stability floor: explicit EM diverges to NaN when the step count is too low
    # for this prior's stiffness (θ_max·S). floor every entry of the sweep list (and
    # dedup, keeping order) so all datasets/backbones run correctly.
    em_floor = min_em_steps(prior, schedule)
    requested = [int(x) for x in str(args.val_steps_list).split(",") if x.strip()]
    steps_list = _parse_steps(args.val_steps_list, em_floor)
    if steps_list != requested:
        log.info(f"EM stability floor (θ_max·S={em_floor}): "
                 f"step sweep {requested} → {steps_list}")

    # ── Model.
    A = load_adjacency(args.dataset)
    if args.denoiser == "gcnn":
        # conditioning is n_cond H-wide fields concatenated along time (1 = Ŷ only,
        # 2 = --cond-past's [R_past | Ŷ]); GCNNDenoiser splits them back into channels.
        cond_w = train_ds.Y_hat.shape[-1]
        if cond_w % prior.H:
            raise ValueError(f"gcnn conditioning width {cond_w} is not a multiple of H={prior.H}")
        denoiser = GCNNDenoiser(
            N=prior.N, H=prior.H, adjacency=A, K=args.gcnn_hops,
            n_layers=args.gcnn_layers, hidden=args.gcnn_hidden,
            n_cond=cond_w // prior.H,
        ).to(device)
    else:
        denoiser = R0Denoiser(
            N=prior.N, H=prior.H, adjacency=A,
            d_h=args.d_h, channel_multipliers=tuple(args.channel_multipliers),
            n_blocks=args.n_blocks, device=str(device),
            T_h=(prior.H if args.cond_past else 0),
        ).to(device)
    log.info(f"denoiser params: {num_parameters(denoiser):,}")

    # AdamW + grad-clip 1.0, matching GAD (exp_gasde.py). lr stays at our scale-
    # appropriate default (batch/epochs differ from GAD's 128/5000).
    optim = torch.optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # ReduceLROnPlateau keyed off the SAMPLED val CRPS (DiffSTG keys its scheduler
    # off sampled val MAE; the raw denoising loss is ~93% irreducible floor and
    # never drives a decision).
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=args.lr_factor, patience=args.lr_patience)
    # EMA weights are what gets validated, selected and shipped (EDM convention);
    # the raw weights only compute gradients.
    ema = EMA(denoiser, args.ema) if args.ema > 0 else None

    # lr range test: sweep lr, record loss, exit before the real training loop.
    if args.lr_find:
        lr_range_test(
            train_loader, denoiser=denoiser, prior_t=prior_t, schedule=schedule,
            optim=optim,
            device=device, n_steps=args.lr_find,
            lr_min=args.lr_find_min, lr_max=args.lr_find_max,
            out_path=ckpt_dir / "lr_find.json",
        )
        return

    # ── Resume (if a checkpoint exists)
    start_epoch = 0
    best_vloss = float("inf")
    best_epoch = -1
    best_crps = float("inf")
    best_crps_epoch = -1
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    if ckpt_path is not None and not args.fresh:
        ckpt = load_checkpoint(ckpt_path)
        denoiser.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        if sched is not None and ckpt.get("scheduler_state_dict") is not None:
            sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_vloss = ckpt["best_val_mae"]             # legacy field name; holds best val loss
        if ema is not None and ckpt.get("ema_state"):
            ema.load(ckpt["ema_state"])
        restore_rng_states(ckpt["rng_states"])
        _hist = ckpt_dir / "validation_history.json"
        if _hist.exists():
            _h = json.loads(_hist.read_text())
            best_epoch = int(_h.get("best_epoch", -1))
            best_crps = float(_h.get("best_crps", float("inf")))
            best_crps_epoch = int(_h.get("best_crps_epoch", -1))
        log.info(f"resumed from epoch {start_epoch}  best_val_loss={best_vloss:.4f}")

    # ── wandb
    if os.environ.get("WANDB_MODE", "online") != "disabled":
        init_wandb(
            cfg, stage="diffusion", dataset=args.dataset, seed=args.seed,
            prior=args.prior, backbone=args.backbone, exp=args.exp,
        )
    import wandb  # safe — always installed

    # ── Clean (fully-observed) val windows for the periodic step-sweep diagnostic
    # and the final-val step choice. fully observed → clean eigenbasis rotation;
    # the metrics still drop y==0 points (Vimal null_val=0). the CRN seed
    # (args.seed+999) keeps step counts AND epochs comparable on cuda.
    full_valid = val_ds.mask.reshape(len(val_ds), -1).all(dim=1)       # (Nval,)
    valid_idx = full_valid.nonzero(as_tuple=True)[0]
    n_val = min(args.val_windows, valid_idx.numel())
    # deterministic STRIDED subset of the clean windows: consecutive windows overlap
    # 11/12 steps, so first-N would be one ~4h block at a single time of day. striding
    # spreads the diagnostic across the whole val period — same windows every epoch
    # and every arm (comparability preserved, no rng).
    stride = max(1, valid_idx.numel() // max(n_val, 1))
    sel = valid_idx[::stride][:n_val]
    val_R0, val_Yhat = val_ds.R0[sel], val_ds.Y_hat[sel]
    val_ypred, val_ytrue = val_ds.y_pred_raw[sel], val_ds.y_true_raw[sel]
    log.info(f"per-epoch val_loss (snr, deterministic draws) selects best.pt "
             f"+ drives the LR plateau; EM step-sweep diagnostic every {args.val_every} "
             f"epochs on {n_val} clean windows (k={args.val_k}, steps={steps_list}); "
             f"{float(full_valid.float().mean()):.2f} of val fully observed")

    # per-run artefacts (easy navigation): validation_history.json (full, for resume) +
    # history.csv (flat, one row/epoch — open in a spreadsheet or grep/plot directly).
    hist_path = ckpt_dir / "validation_history.json"
    csv_path = ckpt_dir / "history.csv"
    # csv keeps the sweep's BEST-step metrics per eval epoch; the full per-step
    # tables live in validation_history.json (val_sweep) for the step choice.
    _CSV_COLS = ["timestamp", "epoch", "train_loss", "val_loss", "lr", "time_s",
                 "val_mae", "val_rmse", "val_mape", "val_crps", "val_w1",
                 "val_energy", "val_best_steps"]
    if not csv_path.exists():                       # keep one continuous file across resumes
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_COLS)
    history: list[dict] = []
    wandb_on = os.environ.get("WANDB_MODE", "online") != "disabled"

    anc_list = _parse_steps(args.val_anc_steps, 1)      # ancestral has no stability floor

    from contextlib import nullcontext

    def _eval_weights():
        """sampled validation runs on the EMA weights when EMA is on."""
        return ema.applied(denoiser) if ema is not None else nullcontext()

    def _maybe_eval(epoch: int, is_last: bool):
        """reverse step-sweep diagnostic (EM + ancestral), every --val-every epochs
        (and on the last). runs on the EMA weights — the ones that get selected."""
        if not args.val_every or (epoch % args.val_every != 0 and not is_last):
            return None
        with _eval_weights():
            res = periodic_val_eval(
                val_R0, val_Yhat, val_ypred, val_ytrue,
                denoiser=denoiser, prior=prior, schedule=schedule, device=device,
                steps_list=steps_list, k=args.val_k, n_proj=args.val_proj,
                sigma_r=sigma_r, crn_seed=args.seed + 999, anc_list=anc_list,
            )
        _log_val(epoch, res, to_wandb=wandb_on)
        return res

    def _fast_val(epoch: int):
        """the DiffSTG-style selection signal: sampled forecast metrics on the fixed
        clean-window subset, fixed noise (CRN), EMA weights, cheap ancestral reverse.
        drives best_crps.pt, early stopping and the LR plateau — the denoising val
        loss is a dashboard only."""
        if args.fast_val_every <= 0 or (epoch % args.fast_val_every) != 0:
            return None
        with _eval_weights():
            res = periodic_val_eval(
                val_R0, val_Yhat, val_ypred, val_ytrue,
                denoiser=denoiser, prior=prior, schedule=schedule, device=device,
                steps_list=[], k=args.val_k, n_proj=args.val_proj,
                sigma_r=sigma_r, crn_seed=args.seed + 999,
                anc_list=[args.fast_val_steps],
            )
        return res["steps_anc"][args.fast_val_steps]

    log.info(f"# {run_id} | {args.dataset} prior={args.prior} backbone={args.backbone} "
             f"seed={args.seed} | select=sampled CRPS on EMA({args.ema}) every "
             f"{args.fast_val_every} ep (anc {args.fast_val_steps} steps) → best_crps.pt; "
             f"val_loss = dashboard (best.pt) | schedule=fcps "
             f"c̄(T)={schedule.c_0:.1f} | loss=snr | "
             f"cond_past={args.cond_past} norm={args.norm}")
    log.info(f"epoch | train | val_loss | lr | time | win/s | gpu   "
             f"(+ EM step-sweep MAE/RMSE/MAPE/CRPS every {args.val_every} ep)")

    # ── Train. best.pt = min val_loss, checked every epoch (deterministic draws);
    # the periodic EM step-sweep is a diagnostic + the step-count evidence.
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
                device=device, generator=None,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optim.step()
            if ema is not None:
                ema.update(denoiser)
            running += loss.detach().item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        # per-epoch val loss (same loss as training, deterministic noise draws):
        # DASHBOARD ONLY — ~93% of it is the irreducible high-noise floor, so no
        # decision (selection / early stop / LR) keys off it.
        vloss = validate(val_loader, denoiser=denoiser, prior_t=prior_t, schedule=schedule,
                         device=device)
        # the DiffSTG-style decision signal: sampled forecast CRPS (EMA weights,
        # fixed windows, fixed noise) — drives the LR plateau below and selection.
        fast = _fast_val(epoch)
        if fast is not None:
            sched.step(fast["crps"])
        lr_now = optim.param_groups[0]["lr"]
        elapsed = time.time() - t0
        thru = n_batches * args.batch_size / max(elapsed, 1e-9)      # train windows/s
        gpu_gb = (torch.cuda.max_memory_allocated() / 2**30
                  if device.type == "cuda" else 0.0)
        fast_msg = (f" | crps={fast['crps']:.4f} mae={fast['mae']:.3f}"
                    if fast is not None else "")
        log.info(f"{epoch:04d}  | train={train_loss:7.4f} | val_loss={vloss:7.4f}"
                 f"{fast_msg} | {lr_now:.1e} | {elapsed:5.1f}s | {thru:5.0f} win/s | "
                 f"gpu={gpu_gb:.1f}G")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # dashboard track: min val loss (kept for the floor-analysis plots + as an
        # ablation checkpoint; raw weights).
        improved = vloss < best_vloss
        if improved:
            best_vloss, best_epoch = vloss, epoch

        # SELECTION track: best sampled CRPS on EMA weights — the deliverable metric.
        if fast is not None and fast["crps"] < best_crps:
            best_crps, best_crps_epoch = fast["crps"], epoch
            save_checkpoint(ckpt_dir / "best_crps.pt", denoiser, optim, sched,
                            epoch=epoch, best_val_mae=best_vloss,
                            rng_states=get_rng_states(),
                            model_state=(ema.state_dict(denoiser) if ema else None),
                            extra={"ema": bool(ema), "fast_val_crps": best_crps})
            log.info(f"  [saved best_crps] epoch={epoch} sampled_crps={best_crps:.5f}"
                     f"{' (ema)' if ema else ''}")

        ev = _maybe_eval(epoch, is_last=(epoch == args.epochs - 1))

        row = {"epoch": epoch, "train_loss": train_loss,
               "val_loss": vloss, "lr": lr_now, "time": elapsed}
        if fast is not None:
            row.update({"val_mae": fast["mae"], "val_rmse": fast["rmse"],
                        "val_mape": fast["mape"], "val_crps": fast["crps"],
                        "val_w1": fast["w1"], "val_energy": fast["energy"]})
        if ev is not None:
            row.update({"val_best_steps": best_steps(ev), "val_sweep": ev["steps"],
                        "val_sweep_anc": ev.get("steps_anc", {})})
        history.append(row)
        hist_path.write_text(json.dumps(
            {"select_metric": "val_loss + sweep_crps", "best_epoch": best_epoch,
             "best_crps": (None if best_crps == float("inf") else best_crps),
             "best_crps_epoch": best_crps_epoch, "history": history},
            indent=2))
        # flat per-epoch row for easy scanning/plotting (blank cells on non-eval epochs)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch,
                 f"{train_loss:.6f}", f"{vloss:.6f}", f"{lr_now:.3e}", f"{elapsed:.1f}"]
                + [("" if c not in row else
                    f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]))
                   for c in _CSV_COLS[6:]])
        if wandb_on:
            wandb.log({"train/loss": train_loss, "val/loss": vloss,
                       "lr": lr_now, "epoch": epoch})

        save_checkpoint(ckpt_dir / "last.pt", denoiser, optim, sched,
                        epoch=epoch, best_val_mae=best_vloss, rng_states=get_rng_states(),
                        extra={"ema_state": (ema.shadow if ema else None)})
        if improved:
            save_checkpoint(ckpt_dir / "best.pt", denoiser, optim, sched,
                            epoch=epoch, best_val_mae=best_vloss, rng_states=get_rng_states())
            log.info(f"  [saved best] epoch={epoch} val_loss={best_vloss:.5f}")

        # early stop: patience on the SAMPLED CRPS only (DiffSTG's rule; the val
        # loss saturates at the floor early and must not stop the run). fall back
        # to the val-loss track only if the fast eval is disabled.
        anchor = best_crps_epoch if args.fast_val_every > 0 else best_epoch
        if anchor >= 0 and (epoch - anchor) >= args.patience:
            log.info(f"[early stop] epoch {epoch} (patience={args.patience} on sampled "
                     f"CRPS, best_crps_epoch={best_crps_epoch}, "
                     f"best_crps={best_crps:.5f}, best_val_loss={best_vloss:.5f})")
            break

    log.info(f"[done] best_epoch={best_epoch} best_val_loss={best_vloss:.5f}  "
             f"best_crps_epoch={best_crps_epoch} best_sweep_crps="
             f"{best_crps if best_crps != float('inf') else float('nan'):.5f}")

    # ── final-val (one-command lifecycle): step sweep on best.pt → full val eval
    if args.final_val:
        _run_final_val(args, device, best_epoch, run_id, sigma_r=sigma_r,
                       val_R0=val_R0, val_Yhat=val_Yhat, val_ypred=val_ypred,
                       val_ytrue=val_ytrue)

    # ── locked TEST eval (separate decision, only once val is trusted)
    if args.eval_after:
        _run_test_eval(args, device, best_epoch, run_id)


if __name__ == "__main__":
    main()
