"""evaluate trained diffusion checkpoints on the TEST split.

for each (dataset, prior, seed) this loads best.pt, generates reverse-sde samples
on the test residuals, and reports per-horizon + overall (all in raw units;
forecast_err = Y_hat + R0_hat - (Y_hat + R0_true) = R0_hat - R0_true, so the
backbone term cancels and we score residuals):

  - MAE / MSE of the K-sample mean            (point forecast accuracy)
  - Winkler interval score + empirical coverage at level alpha   (calibration)
  - sliced wasserstein-1 + energy distance vs the true residual cloud   (fidelity)

samples come from GAD's Euler–Maruyama reverse SDE (the only sampler — stochastic /
ancestral, so the intervals are properly dispersed).

writes results/eval/<ds>/<prior>/seed<seed>/test_metrics.json and, with --all,
prints a kronecker-vs-isotropic comparison table.

usage:
    python scripts/eval_diffusion.py --dataset metrla --prior kronecker --seed 0
    python scripts/eval_diffusion.py --all
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np

# reduce fragmentation for the big-N generation batches (set before torch init)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

# eval is sampling-based (k draws of a stochastic reverse sde), so the tiny
# precision loss of TF32 matmuls is far below the monte-carlo noise floor but
# buys ~1.5-2x on the denoiser's graph-conv / linear layers on ampere/ada gpus.
# no-op on cpu/mps and on pre-ampere cards.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.data.adjacency import load_adjacency
from src.data.residual_dataset import make_dataset, residual_path, train_residual_sigma
from src.diffusion.denoiser import R0Denoiser, SignalDenoiser
from src.diffusion.forward_fcps import FCPSSchedule, prior_to_torch
from src.diffusion.prior import load_prior
from src.diffusion.reverse import reverse_sample_em, min_em_steps
from src.eval.distribution import energy_distance, sliced_wasserstein1
from src.eval.intervals import winkler_score
from src.utils import paths
from src.utils.logging_setup import get_logger, setup_logging

# shared timestamped logger. when eval runs inside the combined traineval job the logger
# is already configured (lines land in the run's train.log too); standalone, main() sets
# it up for stdout. formatted comparison tables stay on plain print (kept clean).
log = get_logger()

_HL = (3, 6, 12)  # highlighted horizons (1-indexed)


# ─── small masked helpers (per horizon over (B, N, H)) ───────────────────────


def _masked_sums(err, mask):
    """sum of err*mask and count of mask, per horizon. err/mask (B, N, H)."""
    m = mask.float()
    return (err * m).sum(dim=(0, 1)), m.sum(dim=(0, 1))


from src.eval.crps import empirical_crps, quantile_crps_num   # noqa: E402  (shared w/ training)


# ─── load a trained run ──────────────────────────────────────────────────────


def load_run(dataset, prior, seed, exp, device, backbone="gwn"):
    ckpt_dir = paths.diffusion_dir(dataset, prior, seed, exp=exp, backbone=backbone)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)

    # rebuild the SAME prior the run trained with (spatial + stiffness cap from config;
    # default structured for back-compat with configs written before --spatial existed)
    prior_obj = load_prior(dataset, prior, seed,
                           spatial=cfg.get("spatial", "structured"),
                           stiffness_cap=(cfg.get("stiffness_cap") or None),
                           backbone=backbone)
    # rebuild the GAD-literal schedule exactly as training used it (fixed S/α/c_min
    # from config.json). default S=7/α=4/c_min=0.1 for back-compat with any config
    # written before the schedule was fixed.
    schedule = FCPSSchedule(c_min=cfg.get("c_min", 0.1), c_0=cfg.get("c_0", 7.0),
                            alpha=cfg.get("alpha", 4.0), T=1.0)
    log.info(f"[eval] schedule (GAD-literal): S=c_0={schedule.c_0:.2f} "
             f"α={schedule.alpha:.1f} c_min={schedule.c_min:.2f} "
             f"θ_min={float(prior_obj.theta.min()):.4f}")
    if backbone == "raw":
        import numpy as _np
        T_h = int(_np.load(residual_path(dataset, "test", seed, backbone))["history"].shape[-1])
        denoiser = SignalDenoiser(
            N=prior_obj.N, H=prior_obj.H, T_h=T_h, adjacency=load_adjacency(dataset),
            d_h=cfg["d_h"], channel_multipliers=tuple(cfg["channel_multipliers"]),
            n_blocks=cfg["n_blocks"], device=str(device),
        ).to(device)
    else:
        denoiser = R0Denoiser(
            N=prior_obj.N, H=prior_obj.H, adjacency=load_adjacency(dataset),
            d_h=cfg["d_h"], channel_multipliers=tuple(cfg["channel_multipliers"]),
            n_blocks=cfg["n_blocks"], device=str(device),
        ).to(device)
    denoiser.load_state_dict(ckpt["model_state_dict"])
    denoiser.eval()
    return denoiser, prior_obj, schedule, cfg, ckpt.get("epoch")


# ─── the test eval ───────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(dataset, prior, seed, *, exp, device, k, n_steps, alpha,
             w1_windows, w1_samples, w1_proj, gen_batch, max_windows, generator,
             backbone="gwn", mask_zero=True):
    denoiser, prior_obj, schedule, cfg, ckpt_epoch = load_run(dataset, prior, seed, exp, device, backbone)
    N, H = prior_obj.N, prior_obj.H

    # EM stability floor: explicit EM NaNs when the step count is too low for this prior's
    # stiffness (θ_max·S). floor n_steps so stiff priors (esp. raw) run correctly.
    floor = min_em_steps(prior_obj, schedule)
    if n_steps < floor:
        log.info(f"[eval] EM stability floor: n_steps {n_steps} → {floor} (θ_max·S)")
        n_steps = floor

    # reverse process: GAD's Euler–Maruyama SDE — the only sampler (stochastic /
    # ancestral, the DDPM-equivalent; GAD's GASDE has no DDIM/deterministic variant).
    def _reverse(Y_cond):
        s, _ = reverse_sample_em(prior_obj, schedule, denoiser, Y_cond,
                                 n_steps=n_steps, R_T=None, generator=generator)
        return s

    sigma_r = train_residual_sigma(dataset, seed, backbone)   # same scalar used in training
    test = make_dataset(dataset, "test", seed, backbone, sigma_r)
    R0_all, Y_all, M_all = test.R0, test.Y_hat, test.mask
    Ytrue_all = test.y_true_raw                               # raw ground truth (y_pred+residual)
    # match SpecSTG/DiffSTG: drop y==0 points from the metrics. for the speed datasets
    # the stored mask already excludes zeros (null_val=0); for the flow datasets
    # (null_val=nan) the stored mask keeps them, so this is where it bites.
    if mask_zero:
        M_all = M_all & (Ytrue_all.abs() > 1e-6)
    n_total = R0_all.shape[0] if not max_windows else min(max_windows, R0_all.shape[0])

    # cap trajectories per reverse_sample call so big-N datasets (pemsbay N=325)
    # don't OOM: process chunk windows at a time with chunk*k <= gen_batch.
    chunk = max(1, gen_batch // k)

    import time
    nchunks = (n_total + chunk - 1) // chunk
    log.info(f"[eval] {dataset}/{prior}: {n_total} test windows, k={k}, chunk={chunk}, "
             f"{nchunks} chunks")
    t0 = time.time()

    # ── point accuracy + Winkler: K samples per window, chunked over windows ──
    mae_n = torch.zeros(H); mse_n = torch.zeros(H); cnt = torch.zeros(H)
    mape_n = torch.zeros(H); wink_n = torch.zeros(H); cov_n = torch.zeros(H); crps_n = torch.zeros(H)
    qnum_n = torch.zeros(H); qden_n = torch.zeros(H)
    for ci, start in enumerate(range(0, n_total, chunk)):
        end = min(start + chunk, n_total)
        R0 = R0_all[start:end].to(device) * sigma_r       # raw residual
        Y = Y_all[start:end].to(device)                   # conditioning (z-scored)
        Mk = M_all[start:end].to(device)
        yt = Ytrue_all[start:end].to(device)              # raw ground-truth signal
        yp = yt - R0                                      # raw backbone forecast (0 for 'raw')
        b = R0.shape[0]
        samp = _reverse(Y.repeat(k, 1, 1))
        samp = samp.view(k, b, N, H) * sigma_r            # raw residual samples
        # reconstruct the forecast and clip to [0,∞) (non-negative flow/speed; matches DiffSTG).
        # all metrics are now in raw FORECAST space (= residual space + y_pred, then clip).
        fc = (yp.unsqueeze(0) + samp).clamp(min=0.0)      # (k, b, N, H) raw forecast samples
        mean = fc.mean(dim=0)
        sn, cn = _masked_sums((mean - yt).abs(), Mk);  mae_n += sn.cpu(); cnt += cn.cpu()
        sn, _ = _masked_sums((mean - yt).pow(2), Mk);  mse_n += sn.cpu()
        sn, _ = _masked_sums((mean - yt).abs() / yt.abs().clamp(min=1e-6), Mk);  mape_n += sn.cpu()
        lo = torch.quantile(fc, alpha / 2.0, dim=0)
        hi = torch.quantile(fc, 1.0 - alpha / 2.0, dim=0)
        sn, _ = _masked_sums(winkler_score(yt, lo, hi, alpha), Mk);  wink_n += sn.cpu()
        sn, _ = _masked_sums(((yt >= lo) & (yt <= hi)).float(), Mk);  cov_n += sn.cpu()
        sn, _ = _masked_sums(empirical_crps(fc, yt), Mk);  crps_n += sn.cpu()   # raw-unit energy CRPS
        qnum_n += quantile_crps_num(fc, yt, Mk).cpu()                           # normalized-CRPS num
        sn, _ = _masked_sums(yt.abs(), Mk);  qden_n += sn.cpu()                 # normalized-CRPS denom

        if ci % max(1, nchunks // 10) == 0 or ci == nchunks - 1:
            el = time.time() - t0
            remaining = el / (ci + 1) * (nchunks - ci - 1)   # don't shadow the sde `eta`
            log.info(f"  [{dataset}/{prior}] point {ci+1}/{nchunks} "
                     f"({100*(ci+1)/nchunks:.0f}%)  elapsed={el:.0f}s  eta={remaining:.0f}s")

    log.info(f"  [{dataset}/{prior}] point done in {time.time()-t0:.0f}s; computing W1/energy...")
    cnt = cnt.clamp(min=1.0)
    mae_h = (mae_n / cnt); mse_h = (mse_n / cnt); mape_h = (mape_n / cnt)
    wink_h = (wink_n / cnt); cov_h = (cov_n / cnt); crps_h = (crps_n / cnt)
    qcrps_h = qnum_n / (qden_n.clamp(min=1e-8) * 19.0)    # normalized quantile-CRPS per horizon
    overall = lambda num: float(num.sum() / cnt.sum())

    # ── distribution fidelity: complete-case windows, sliced-W1 + energy ──────
    valid = M_all.reshape(M_all.shape[0], -1).all(dim=1)
    vidx = valid.nonzero(as_tuple=True)[0]
    n_w1 = min(w1_windows, vidx.numel())
    sel = vidx[:n_w1]
    w1_frac = float(valid.float().mean())
    R0v = R0_all[sel].to(device) * sigma_r; Yv = Y_all[sel].to(device)   # de-norm R0 to raw
    # sub-batch windows so w1_samples * sub <= gen_batch (same OOM guard)
    wb = max(1, gen_batch // w1_samples)
    gen_parts = []
    for s in range(0, n_w1, wb):
        e = min(s + wb, n_w1)
        g = _reverse(Yv[s:e].repeat(w1_samples, 1, 1))
        gen_parts.append(g.view(w1_samples, e - s, N, H))
    gen = torch.cat(gen_parts, dim=1) * sigma_r    # (w1_samples, n_w1, N, H), raw units
    w1_h, ed_h = [], []
    for h in range(H):
        true_c = R0v[:, :, h]
        gen_c = gen[:, :, :, h].reshape(w1_samples * n_w1, N)
        w1_h.append(sliced_wasserstein1(gen_c, true_c, w1_proj, generator))
        ed_h.append(energy_distance(gen_c, true_c))

    return {
        "dataset": dataset, "backbone": backbone, "prior": prior, "seed": seed,
        "exp": exp, "ckpt_epoch": ckpt_epoch,
        "n_test_windows": int(n_total), "k_samples": k, "n_steps": n_steps,
        "sampler": "em", "alpha": alpha, "mask_zero": bool(mask_zero),
        "mae": overall(mae_n), "mse": overall(mse_n), "rmse": float((mse_n.sum()/cnt.sum())**0.5),
        "mape": overall(mape_n),
        "winkler": overall(wink_n), "coverage": overall(cov_n),
        "crps_energy": overall(crps_n),                              # raw-unit fair/energy CRPS
        "crps": float(qnum_n.sum() / (qden_n.sum().clamp(min=1e-8) * 19.0)),  # normalized (papers/Vimal)
        "mae_h": mae_h.tolist(), "mse_h": mse_h.tolist(), "mape_h": mape_h.tolist(),
        "winkler_h": wink_h.tolist(), "coverage_h": cov_h.tolist(),
        "crps_h": qcrps_h.tolist(), "crps_energy_h": crps_h.tolist(),
        "w1": float(np.mean(w1_h)), "energy": float(np.mean(ed_h)),
        "w1_h": w1_h, "energy_h": ed_h,
        "w1_clean_windows": int(n_w1), "w1_frac_of_test": w1_frac,
    }


def _save(res):
    out = paths.eval_dir(res["dataset"], res["prior"], res["seed"], exp=res.get("exp"),
                         backbone=res.get("backbone", "gwn"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "test_metrics.json"
    path.write_text(json.dumps(res, indent=2))
    return path


def _print_one(res):
    print(f"\n=== {res['dataset']}/{res['prior']} (best.pt epoch {res['ckpt_epoch']}, "
          f"{res['n_test_windows']} test windows, K={res['k_samples']}, "
          f"sampler={res.get('sampler','em')}, n_steps={res['n_steps']}) ===")
    print(f"  MAE={res['mae']:.4f}  RMSE={res['mse']**0.5:.4f}  MAPE={res['mape']:.4f}  "
          f"Winkler{int((1-res['alpha'])*100)}={res['winkler']:.4f}  coverage={res['coverage']:.3f}")
    print(f"  CRPS(norm)={res['crps']:.4f}  CRPS(energy)={res.get('crps_energy', float('nan')):.4f}  "
          f"W1={res['w1']:.4f}  energy={res['energy']:.4f}  "
          f"({res['w1_clean_windows']} clean windows = {res['w1_frac_of_test']:.2f} of test)")
    hl = "  ".join(f"h{h}:MAE={res['mae_h'][h-1]:.2f}" for h in _HL if h <= len(res['mae_h']))
    print(f"  per-horizon  {hl}")


def _print_comparison(results):
    by = {(r["dataset"], r["prior"]): r for r in results}
    print("\n" + "=" * 78)
    print("KRONECKER vs ISOTROPIC on TEST (lower=better; coverage target = 1-alpha)")
    print("=" * 78)
    print(f"{'dataset':9s} {'metric':9s} {'kronecker':12s} {'isotropic':12s} {'winner':8s} {'Δ%':>7s}")
    for ds in ["metrla", "pems08f", "pemsbay"]:
        rk, ri = by.get((ds, "kronecker")), by.get((ds, "isotropic"))
        if not (rk and ri):
            continue
        for metric, lower in [("mae", True), ("mse", True), ("winkler", True),
                              ("coverage", None), ("w1", True), ("energy", True)]:
            kv, iv = rk[metric], ri[metric]
            if lower is None:  # coverage: closer to (1-alpha) wins
                tgt = 1 - rk["alpha"]
                win = "kron" if abs(kv - tgt) < abs(iv - tgt) else "iso"
                delta = ""
            else:
                win = "kron" if kv < iv else "iso"
                delta = f"{100*(kv-iv)/iv:+.1f}" if iv else ""
            print(f"{ds:9s} {metric:9s} {kv:<12.4f} {iv:<12.4f} {win:8s} {delta:>7s}")
        print("-" * 60)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems04f", "pems04s",
                                          "pems08f", "pems08s"))
    p.add_argument("--prior", choices=("kronecker", "isotropic"))
    p.add_argument("--backbone", choices=("gwn", "gcn", "mlp", "raw"), default="gwn",
                   help="which backbone's diffusion runs to eval (namespaces results)")
    p.add_argument("--exp", default="dev", help="experiment tag (matches training --exp)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true", help="eval all datasets x both priors")
    p.add_argument("--k", type=int, default=50, help="samples per window for MAE/MSE/Winkler")
    p.add_argument("--n-steps", type=int, default=1000,
                   help="GAD Euler–Maruyama reverse steps (stiff prior needs many; GAD uses ~5000)")
    p.add_argument("--alpha", type=float, default=0.1, help="0.1 -> central 90%% interval")
    p.add_argument("--w1-windows", type=int, default=512)
    p.add_argument("--w1-samples", type=int, default=4)
    p.add_argument("--w1-proj", type=int, default=128)
    p.add_argument("--gen-batch", type=int, default=1024,
                   help="max trajectories per reverse_sample call (caps gpu memory)")
    p.add_argument("--max-windows", type=int, default=0, help="cap test windows (0=all)")
    p.add_argument("--no-mask-zero", dest="mask_zero", action="store_false",
                   help="keep y==0 points in the metrics (default: drop them, SpecSTG/DiffSTG convention)")
    args = p.parse_args()

    setup_logging(None)        # standalone: timestamped lines to stdout (traineval configures its own)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    g = None if device.type == "mps" else torch.Generator(device=device.type).manual_seed(args.seed)

    if args.all:
        jobs = [(ds, pr) for ds in ("metrla", "pemsbay", "pems04f", "pems04s",
                                    "pems08f", "pems08s")
                for pr in ("kronecker", "isotropic")]
    else:
        if not (args.dataset and args.prior):
            p.error("give --dataset and --prior, or --all")
        jobs = [(args.dataset, args.prior)]

    print(f"[eval] device={device}  sampler=EM (GAD)  k={args.k}  n_steps={args.n_steps}")
    results = []
    for ds, pr in jobs:
        res = evaluate(ds, pr, args.seed, exp=args.exp, device=device, k=args.k,
                       n_steps=args.n_steps, alpha=args.alpha,
                       w1_windows=args.w1_windows, w1_samples=args.w1_samples,
                       w1_proj=args.w1_proj, gen_batch=args.gen_batch,
                       max_windows=args.max_windows, generator=g, backbone=args.backbone,
                       mask_zero=args.mask_zero)
        _print_one(res)
        print(f"  wrote {_save(res)}")
        results.append(res)

    if len(results) > 1:
        _print_comparison(results)


if __name__ == "__main__":
    main()
