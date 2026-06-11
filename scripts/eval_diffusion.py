"""evaluate trained diffusion checkpoints on the TEST split.

for each (dataset, prior, seed) this loads best.pt, generates reverse-sde samples
on the test residuals, and reports per-horizon + overall (all in raw units;
forecast_err = Y_hat + R0_hat - (Y_hat + R0_true) = R0_hat - R0_true, so the
backbone term cancels and we score residuals):

  - MAE / MSE of the K-sample mean            (point forecast accuracy)
  - Winkler interval score + empirical coverage at level alpha   (calibration)
  - sliced wasserstein-1 + energy distance vs the true residual cloud   (fidelity)

samples use the full stochastic reverse sde (eta=1) by default, so the intervals
are properly dispersed (the during-training Winkler used eta=0 and under-covered).

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
from src.data.residual_dataset import ResidualDataset, residual_path, train_residual_sigma
from src.diffusion.denoiser import R0Denoiser
from src.diffusion.forward_fcps import FCPSSchedule, prior_to_torch
from src.diffusion.prior import load_prior
from src.diffusion.reverse import reverse_sample
from src.eval.distribution import energy_distance, sliced_wasserstein1
from src.eval.intervals import winkler_score
from src.utils import paths

_HL = (3, 6, 12)  # highlighted horizons (1-indexed)


# ─── small masked helpers (per horizon over (B, N, H)) ───────────────────────


def _masked_sums(err, mask):
    """sum of err*mask and count of mask, per horizon. err/mask (B, N, H)."""
    m = mask.float()
    return (err * m).sum(dim=(0, 1)), m.sum(dim=(0, 1))


def empirical_crps(samples, y):
    """fair (energy-form) ensemble CRPS per point, in the same units as samples/y.

    CRPS = E|X - y| - 0.5 E|X - X'|, estimated from the K samples. the pairwise
    term uses the O(K log K) sorted identity ΣΣ|x_i-x_j| = 2 Σ_i (2i-K-1) x_(i),
    so we never form the KxK matrix. translation-equivariant, so CRPS on residuals
    equals CRPS on the reconstructed forecast (the backbone Y_hat cancels per point).
    samples (K, B, N, H), y (B, N, H) -> (B, N, H)."""
    K = samples.shape[0]
    term1 = (samples - y.unsqueeze(0)).abs().mean(dim=0)            # E|X - y|
    xs, _ = torch.sort(samples, dim=0)                             # ascending
    idx = torch.arange(1, K + 1, device=samples.device, dtype=samples.dtype)
    w = (2 * idx - K - 1) / (K * K)                                # 0.5 E|X - X'| weights
    term2 = (w.view(K, 1, 1, 1) * xs).sum(dim=0)
    return term1 - term2


# ─── load a trained run ──────────────────────────────────────────────────────


def load_run(dataset, prior, seed, exp, device):
    ckpt_dir = paths.diffusion_dir(dataset, prior, seed, exp=exp)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)

    prior_obj = load_prior(dataset, prior, seed)
    schedule = FCPSSchedule(c_min=cfg["c_min"], c_0=cfg["c_0"], alpha=cfg["alpha"], T=1.0)
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
def evaluate(dataset, prior, seed, *, exp, device, k, n_steps, eta, alpha,
             w1_windows, w1_samples, w1_proj, gen_batch, max_windows, generator):
    denoiser, prior_obj, schedule, cfg, ckpt_epoch = load_run(dataset, prior, seed, exp, device)
    N, H = prior_obj.N, prior_obj.H

    sigma_r = train_residual_sigma(dataset, seed)        # same scalar used in training
    test = ResidualDataset(residual_path(dataset, "test", seed), sigma_r)
    R0_all, Y_all, M_all = test.R0, test.Y_hat, test.mask
    n_total = R0_all.shape[0] if not max_windows else min(max_windows, R0_all.shape[0])

    # cap trajectories per reverse_sample call so big-N datasets (pemsbay N=325)
    # don't OOM: process chunk windows at a time with chunk*k <= gen_batch.
    chunk = max(1, gen_batch // k)

    import time
    nchunks = (n_total + chunk - 1) // chunk
    print(f"[eval] {dataset}/{prior}: {n_total} test windows, k={k}, chunk={chunk}, "
          f"{nchunks} chunks", flush=True)
    t0 = time.time()

    # ── point accuracy + Winkler: K samples per window, chunked over windows ──
    mae_n = torch.zeros(H); mse_n = torch.zeros(H); cnt = torch.zeros(H)
    wink_n = torch.zeros(H); cov_n = torch.zeros(H); crps_n = torch.zeros(H)
    for ci, start in enumerate(range(0, n_total, chunk)):
        end = min(start + chunk, n_total)
        R0 = R0_all[start:end].to(device) * sigma_r       # de-normalize to raw units
        Y = Y_all[start:end].to(device)                   # conditioning stays z-scored
        Mk = M_all[start:end].to(device)
        b = R0.shape[0]
        samp, _ = reverse_sample(
            prior_obj, schedule, denoiser, Y.repeat(k, 1, 1),
            n_steps=n_steps, eta=eta, R_T=None, generator=generator,
        )
        samp = samp.view(k, b, N, H) * sigma_r            # de-normalize to raw units
        mean = samp.mean(dim=0)
        sn, cn = _masked_sums((mean - R0).abs(), Mk);  mae_n += sn.cpu(); cnt += cn.cpu()
        sn, _ = _masked_sums((mean - R0).pow(2), Mk);  mse_n += sn.cpu()
        lo = torch.quantile(samp, alpha / 2.0, dim=0)
        hi = torch.quantile(samp, 1.0 - alpha / 2.0, dim=0)
        sn, _ = _masked_sums(winkler_score(R0, lo, hi, alpha), Mk);  wink_n += sn.cpu()
        inside = ((R0 >= lo) & (R0 <= hi)).float()
        sn, _ = _masked_sums(inside, Mk);  cov_n += sn.cpu()
        sn, _ = _masked_sums(empirical_crps(samp, R0), Mk);  crps_n += sn.cpu()

        if ci % max(1, nchunks // 10) == 0 or ci == nchunks - 1:
            el = time.time() - t0
            remaining = el / (ci + 1) * (nchunks - ci - 1)   # don't shadow the sde `eta`
            print(f"  [{dataset}/{prior}] point {ci+1}/{nchunks} "
                  f"({100*(ci+1)/nchunks:.0f}%)  elapsed={el:.0f}s  eta={remaining:.0f}s", flush=True)

    print(f"  [{dataset}/{prior}] point done in {time.time()-t0:.0f}s; computing W1/energy...",
          flush=True)
    cnt = cnt.clamp(min=1.0)
    mae_h = (mae_n / cnt); mse_h = (mse_n / cnt)
    wink_h = (wink_n / cnt); cov_h = (cov_n / cnt); crps_h = (crps_n / cnt)
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
        g, _ = reverse_sample(
            prior_obj, schedule, denoiser, Yv[s:e].repeat(w1_samples, 1, 1),
            n_steps=n_steps, eta=eta, R_T=None, generator=generator,
        )
        gen_parts.append(g.view(w1_samples, e - s, N, H))
    gen = torch.cat(gen_parts, dim=1) * sigma_r    # (w1_samples, n_w1, N, H), raw units
    w1_h, ed_h = [], []
    for h in range(H):
        true_c = R0v[:, :, h]
        gen_c = gen[:, :, :, h].reshape(w1_samples * n_w1, N)
        w1_h.append(sliced_wasserstein1(gen_c, true_c, w1_proj, generator))
        ed_h.append(energy_distance(gen_c, true_c))

    return {
        "dataset": dataset, "prior": prior, "seed": seed, "exp": exp, "ckpt_epoch": ckpt_epoch,
        "n_test_windows": int(n_total), "k_samples": k, "n_steps": n_steps,
        "eta": eta, "alpha": alpha,
        "mae": overall(mae_n), "mse": overall(mse_n),
        "winkler": overall(wink_n), "coverage": overall(cov_n), "crps": overall(crps_n),
        "mae_h": mae_h.tolist(), "mse_h": mse_h.tolist(),
        "winkler_h": wink_h.tolist(), "coverage_h": cov_h.tolist(), "crps_h": crps_h.tolist(),
        "w1": float(np.mean(w1_h)), "energy": float(np.mean(ed_h)),
        "w1_h": w1_h, "energy_h": ed_h,
        "w1_clean_windows": int(n_w1), "w1_frac_of_test": w1_frac,
    }


def _save(res):
    out = paths.eval_dir(res["dataset"], res["prior"], res["seed"], exp=res.get("exp"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "test_metrics.json"
    path.write_text(json.dumps(res, indent=2))
    return path


def _print_one(res):
    print(f"\n=== {res['dataset']}/{res['prior']} (best.pt epoch {res['ckpt_epoch']}, "
          f"{res['n_test_windows']} test windows, K={res['k_samples']}, eta={res['eta']}) ===")
    print(f"  MAE={res['mae']:.4f}  MSE={res['mse']:.4f}  RMSE={res['mse']**0.5:.4f}  "
          f"Winkler{int((1-res['alpha'])*100)}={res['winkler']:.4f}  coverage={res['coverage']:.3f}  "
          f"CRPS={res['crps']:.4f}")
    print(f"  W1={res['w1']:.4f}  energy={res['energy']:.4f}  "
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
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems08f"))
    p.add_argument("--prior", choices=("kronecker", "isotropic"))
    p.add_argument("--exp", default="dev", help="experiment tag (matches training --exp)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true", help="eval all datasets x both priors")
    p.add_argument("--k", type=int, default=50, help="samples per window for MAE/MSE/Winkler")
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--eta", type=float, default=1.0, help="1.0 = full stochastic reverse sde")
    p.add_argument("--alpha", type=float, default=0.1, help="0.1 -> central 90%% interval")
    p.add_argument("--w1-windows", type=int, default=512)
    p.add_argument("--w1-samples", type=int, default=4)
    p.add_argument("--w1-proj", type=int, default=128)
    p.add_argument("--gen-batch", type=int, default=1024,
                   help="max trajectories per reverse_sample call (caps gpu memory)")
    p.add_argument("--max-windows", type=int, default=0, help="cap test windows (0=all)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    g = None if device.type == "mps" else torch.Generator(device=device.type).manual_seed(args.seed)

    if args.all:
        jobs = [(ds, pr) for ds in ("metrla", "pems08f", "pemsbay")
                for pr in ("kronecker", "isotropic")]
    else:
        if not (args.dataset and args.prior):
            p.error("give --dataset and --prior, or --all")
        jobs = [(args.dataset, args.prior)]

    print(f"[eval] device={device}  k={args.k}  n_steps={args.n_steps}  eta={args.eta}")
    results = []
    for ds, pr in jobs:
        res = evaluate(ds, pr, args.seed, exp=args.exp, device=device, k=args.k,
                       n_steps=args.n_steps, eta=args.eta, alpha=args.alpha,
                       w1_windows=args.w1_windows, w1_samples=args.w1_samples,
                       w1_proj=args.w1_proj, gen_batch=args.gen_batch,
                       max_windows=args.max_windows, generator=g)
        _print_one(res)
        print(f"  wrote {_save(res)}")
        results.append(res)

    if len(results) > 1:
        _print_comparison(results)


if __name__ == "__main__":
    main()
