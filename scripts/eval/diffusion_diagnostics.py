"""Correctness diagnostics for the diffusion machinery (independent of forecast skill).

Three checks, written to ``results/eval/{dataset}/diagnostics.json``:

  1. Oracle reconstruction — feed the reverse sampler an *oracle* denoiser that
     returns the true R_0. With the exact/DDIM update this must reconstruct R_0
     to < 1e-3 on the real prior dimensions, even under the stiff production
     FCPS schedule that breaks Euler–Maruyama. (PASS/FAIL.)

  2. Forward-marginal exactness — Monte-Carlo draws from
     ``sample_R_t_given_R0_torch`` must match the closed-form per-mode mean
     (H_t·C₀) and variance (σ²_t) in the eigenbasis. Reports max relative error
     across modes at several t. (PASS/FAIL.)

  3. Generated-sample quality (only with --ckpt-dir) — residuals sampled from the
     trained model vs held-out real residuals: per-horizon variance match and
     horizon-correlation R_H match. Complements the prior-vs-empirical checks in
     ``scripts/section3/exp34_36_reconcile_sample_psd.py`` (which validate the
     prior; this validates the trained model).

Usage:
    python scripts/eval/diffusion_diagnostics.py --dataset metrla
    python scripts/eval/diffusion_diagnostics.py --dataset metrla --ckpt-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.diffusion.forward_fcps import (                          # noqa: E402
    FCPSSchedule,
    prior_to_torch,
    sample_R_t_given_R0_torch,
)
from src.diffusion.prior import load_prior                        # noqa: E402
from src.diffusion.reverse import reverse_sample                  # noqa: E402
from src.eval.io import load_eval_checkpoint, load_split_arrays   # noqa: E402
from src.eval.samplers import DiffusionSampler                    # noqa: E402

ORACLE_TOL = 1e-3
# Forward-marginal tolerances. The per-mode mean is checked as a standardized
# z-score (error ÷ MC standard error) because slow modes have a tiny mean but a
# huge variance — a relative error on the mean would be pure MC noise. Variance
# is checked as a ratio (well-posed everywhere).
FORWARD_MEAN_Z_TOL = 6.0      # max |z| over all modes (≈ N(0,1) per mode)
FORWARD_VAR_RATIO_TOL = 0.10  # max |emp_var/σ²_t − 1|


# ─── 1. Oracle reconstruction ─────────────────────────────────────────────────


def oracle_reconstruction(prior, schedule, n_steps=200, B=64, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    R0 = torch.randn(1, prior.N, prior.H).expand(B, -1, -1).contiguous()
    Y_hat = torch.zeros_like(R0)

    def oracle(R_t, t_batch, Y_):
        return R0.to(device=R_t.device, dtype=R_t.dtype)

    R_hat, _ = reverse_sample(
        prior=prior, schedule=schedule, denoiser=oracle,
        Y_hat=Y_hat, n_steps=n_steps, eta=0.0, generator=g,
    )
    max_err = float((R_hat - R0).abs().max())
    return {
        "max_abs_err": max_err,
        "tol": ORACLE_TOL,
        "n_steps": n_steps,
        "schedule": {"c_min": schedule.c_min, "c_0": schedule.c_0, "alpha": schedule.alpha},
        "pass": bool(max_err < ORACLE_TOL),
    }


# ─── 2. Forward-marginal exactness ────────────────────────────────────────────


def forward_marginal_check(prior, schedule, ts=(0.3, 0.6, 0.9), n_draws=20000, seed=0) -> dict:
    """Compare MC mean/var of R_t (rotated to eigenbasis) to the closed form.

    Per mode, R_t's coordinate is N(H_t·C₀, σ²_t). We check:
      * the mean as a standardized z = (emp_mean − H_t·C₀)·√n / σ_t  (≈ N(0,1)),
      * the variance as a ratio emp_var / σ²_t  (≈ 1).
    """
    prior_t = prior_to_torch(prior, device=torch.device("cpu"), dtype=torch.float64)
    rng = np.random.default_rng(seed)
    R0 = torch.as_tensor(
        rng.standard_normal((1, prior.N, prior.H)), dtype=torch.float64
    ).expand(n_draws, -1, -1).contiguous()
    V_S, V_T, theta = prior_t.V_S, prior_t.V_T, prior_t.theta
    C0 = torch.einsum("ki,bih,hj->bkj", V_S.T, R0[:1], V_T)[0]     # (N, H)

    results = {}
    g = torch.Generator().manual_seed(seed)
    worst_z, worst_var = 0.0, 0.0
    for t in ts:
        t_b = torch.full((n_draws,), float(t), dtype=torch.float64)
        R_t = sample_R_t_given_R0_torch(R0, t_b, prior_t, schedule, generator=g)
        C_t = torch.einsum("ki,bih,hj->bkj", V_S.T, R_t, V_T)      # (B, N, H)

        c_bar = float(schedule.c_bar(t))
        H_t = torch.exp(-theta * c_bar)
        sigma2_t = (1.0 - torch.exp(-2.0 * theta * c_bar)) / theta
        sigma_t = sigma2_t.clamp(min=1e-300).sqrt()

        emp_mean = C_t.mean(0)
        emp_var = C_t.var(0, unbiased=True)
        mean_z = float(((emp_mean - H_t * C0).abs() * (n_draws ** 0.5) / sigma_t).max())
        var_ratio_err = float((emp_var / sigma2_t - 1.0).abs().max())
        worst_z = max(worst_z, mean_z)
        worst_var = max(worst_var, var_ratio_err)
        results[f"t={t}"] = {
            "c_bar": c_bar,
            "mean_max_zscore": mean_z,
            "var_max_ratio_err": var_ratio_err,
        }
    passed = worst_z < FORWARD_MEAN_Z_TOL and worst_var < FORWARD_VAR_RATIO_TOL
    return {
        "per_t": results,
        "mean_max_zscore": worst_z,
        "var_max_ratio_err": worst_var,
        "mean_z_tol": FORWARD_MEAN_Z_TOL,
        "var_ratio_tol": FORWARD_VAR_RATIO_TOL,
        "pass": bool(passed),
    }


# ─── 3. Generated-sample quality (trained model) ──────────────────────────────


def generated_sample_quality(bundle, dataset, split, n_windows, K, n_steps, eta, seed) -> dict:
    arrays = load_split_arrays(dataset, split, seed)
    y_pred, y_true, mask = arrays["y_pred"], arrays["y_true"], arrays["mask"]
    B = min(n_windows, y_true.shape[0])
    idx = np.linspace(0, y_true.shape[0] - 1, B, dtype=int)

    sampler = DiffusionSampler(
        bundle.denoiser, bundle.prior, bundle.schedule,
        n_samples=K, n_steps=n_steps, eta=eta, device=bundle.device, seed=seed,
    )
    Y_sample = sampler.sample(y_pred[idx])                          # (K, B, N, H)
    R_sample = Y_sample - y_pred[idx][None]                         # generated residuals
    R_real = (y_true - y_pred)[idx]                                 # real residuals
    H = R_real.shape[-1]

    # Per-horizon variance (pooled over K, windows, sensors for samples; windows,
    # sensors for real). Only over valid cells.
    m = mask[idx]
    var_sample, var_real = [], []
    for h in range(H):
        mh = m[:, :, h]
        rs = R_sample[:, :, :, h][:, mh]      # (K, M)
        rr = R_real[:, :, h][mh]              # (M,)
        var_sample.append(float(rs.var()))
        var_real.append(float(rr.var()))
    var_sample = np.array(var_sample)
    var_real = np.array(var_real)

    # Horizon correlation R_H (use the K=0 sample slice pooled with real layout).
    flat_sample = R_sample.reshape(-1, H)
    keep_s = np.isfinite(flat_sample).all(1)
    RH_sample = np.corrcoef(flat_sample[keep_s].T)
    flat_real = R_real.reshape(-1, H)
    mask_flat = m.reshape(-1, H).all(1)
    RH_real = np.corrcoef(flat_real[mask_flat].T)

    return {
        "n_windows": int(B),
        "per_horizon_variance": {
            "sample": var_sample.tolist(),
            "real": var_real.tolist(),
            "ratio": (var_sample / np.clip(var_real, 1e-8, None)).tolist(),
        },
        "horizon_corr_frobenius_diff": float(np.linalg.norm(RH_sample - RH_real)),
        "RH_sample": RH_sample.tolist(),
        "RH_real": RH_real.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems08f"), required=True)
    p.add_argument("--ckpt-dir", default=None, help="enable generated-sample quality check")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen-windows", type=int, default=64)
    p.add_argument("--gen-samples", type=int, default=64)
    p.add_argument("--gen-steps", type=int, default=50)
    p.add_argument("--gen-eta", type=float, default=1.0)
    args = p.parse_args()

    prior = load_prior(args.dataset)
    schedule = FCPSSchedule()
    print(f"[diagnostics] dataset={args.dataset}  N={prior.N} H={prior.H}", flush=True)

    diag = {"dataset": args.dataset}
    diag["oracle_reconstruction"] = oracle_reconstruction(prior, schedule)
    print(f"  oracle reconstruction: max_err={diag['oracle_reconstruction']['max_abs_err']:.2e} "
          f"-> {'PASS' if diag['oracle_reconstruction']['pass'] else 'FAIL'}", flush=True)

    diag["forward_marginal"] = forward_marginal_check(prior, schedule)
    fm = diag["forward_marginal"]
    print(f"  forward marginal: mean|z|={fm['mean_max_zscore']:.2f} "
          f"var_ratio_err={fm['var_max_ratio_err']:.3f} "
          f"-> {'PASS' if fm['pass'] else 'FAIL'}", flush=True)

    if args.ckpt_dir:
        bundle = load_eval_checkpoint(args.ckpt_dir)
        diag["generated_sample_quality"] = generated_sample_quality(
            bundle, args.dataset, args.split, args.gen_windows,
            args.gen_samples, args.gen_steps, args.gen_eta, args.seed,
        )
        ratio = np.array(diag["generated_sample_quality"]["per_horizon_variance"]["ratio"])
        print(f"  generated quality: var ratio range [{ratio.min():.2f}, {ratio.max():.2f}]  "
              f"R_H Frobenius diff={diag['generated_sample_quality']['horizon_corr_frobenius_diff']:.3f}",
              flush=True)
    else:
        print("  (skip generated-sample quality: pass --ckpt-dir to enable)", flush=True)

    out_dir = ROOT / "results" / "eval" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(f"[diagnostics] wrote {out_dir}/diagnostics.json", flush=True)


if __name__ == "__main__":
    main()
