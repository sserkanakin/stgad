"""GAD's evaluation, one axis up: UNCONDITIONAL spatio-temporal generation quality.

samples S windows from a trained unconditional model (kron GAD-ST vs iso) at several
ancestral step counts and compares the generated distribution against held-out TEST
windows using GAD's own metric suite (vendored verbatim, gad_signal_metrics.py):

  TV_S   spatial smoothness      x^T L_S x   (per horizon, averaged)
  TV_T   temporal smoothness     x^T L_T x   (per sensor, path graph)
  SC     spatial spectral centroid
  aMMD   average MMD between generated-vs-test metric distributions (GAD Fig. 3's score)

usage:
  python scripts/eval_uncond_gadst.py --dataset metrla --exp 20260716_uncond \
      --arms kron_u:kronecker iso_u:isotropic --steps 2 5 10 25 50 --n-samples 512
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root (for scripts.* import)

import numpy as np
import torch

from src.data.adjacency import compute_normalized_laplacian, load_adjacency
from src.data.residual_dataset import train_residual_sigma
from src.diffusion.reverse import min_em_steps, reverse_sample_ancestral, reverse_sample_em
from src.eval.gad_signal_metrics import (
    degree_correlation,
    mmd_distance,
    spectral_centroid,
    total_variation,
    wasserstein_1d,
)
from src.utils import paths


def load_test_windows(ds, cap=2000):
    d = np.load(paths.residual_npz(ds, "test", 42, "raw"))
    sr = float(np.asarray(train_residual_sigma(ds, 42, "raw", norm="global")))
    R = d["residuals"].astype(np.float32) / sr
    M = d["mask"].astype(bool)
    keep = M.reshape(M.shape[0], -1).all(1)
    X = R[keep]
    if X.shape[0] > cap:
        X = X[np.linspace(0, X.shape[0] - 1, cap).astype(int)]
    return torch.from_numpy(X)                                  # (B, N, H)


def path_laplacian(H):
    A = np.zeros((H, H)); i = np.arange(H - 1)
    A[i, i + 1] = 1.0; A[i + 1, i] = 1.0
    return compute_normalized_laplacian(A)


# GAD's headline aMMD averages exactly these three spatial statistics (their Fig. 3).
GAD_KEYS = ["TV_S", "SC_S", "DC_S"]


def metric_set(X, L_comb, L_norm, degrees, L_T):
    """GAD's three graph-signal statistics on (B, N, H) windows, horizons as features:
      TV_S = xᵀL x   with the COMBINATORIAL Laplacian (GAD's QV/TV, signal_metrics.py)
      SC_S = spectral centroid with the NORMALIZED Laplacian
      DC_S = degree correlation (Pearson corr of signal vs node-degree vector)
    plus TV_T (temporal path-graph smoothness) as an extra spatio-temporal column."""
    return {
        "TV_S": total_variation(X, L_comb),                     # (B,)  combinatorial L
        "SC_S": spectral_centroid(X, L=L_norm),                 # normalized L
        "DC_S": degree_correlation(X, degrees),                 # GAD's 3rd metric
        "TV_T": total_variation(X.transpose(1, 2), L_T),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metrla")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--arms", nargs="+", default=["kron_u:kronecker", "iso_u:isotropic"])
    ap.add_argument("--steps", type=int, nargs="+", default=[2, 5, 10, 25, 50])
    ap.add_argument("--sampler", choices=("ancestral", "em"), default="ancestral",
                    help="em = GAD's protocol (their Fig.3 step counts; NO stability floor "
                         "applied — few-step EM error is part of what that figure measures)")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--eval-seeds", type=int, default=1,
                    help=">1: repeat sampling with seeds 0..k-1 and report mean±std aMMD")
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    Xt = load_test_windows(args.dataset)
    N, H = Xt.shape[1], Xt.shape[2]
    A = np.asarray(load_adjacency(args.dataset, repair=True), dtype=np.float64)
    L_comb = torch.tensor(np.diag(A.sum(1)) - A, dtype=torch.float32)      # GAD's TV uses combinatorial L
    L_norm = torch.tensor(compute_normalized_laplacian(A), dtype=torch.float32)  # SC uses normalized L
    degrees = torch.tensor(A.sum(1), dtype=torch.float32)                  # DC uses node degrees
    L_T = torch.tensor(path_laplacian(H), dtype=torch.float32)
    ref = metric_set(Xt, L_comb, L_norm, degrees, L_T)
    print(f"=== {args.dataset}: unconditional GAD-ST vs baselines · sampler={args.sampler} · "
          f"{args.n_samples} samples vs {Xt.shape[0]} test windows · aMMD=mean(TV_S,SC_S,DC_S) [GAD] ===")
    print(f"{'arm':8} {'steps':>5} | {'MMD TV_S':>9} {'MMD SC_S':>9} {'MMD DC_S':>9} | {'aMMD':>7} | {'MMD TV_T':>8}")

    from scripts.eval_diffusion import load_run                  # reuse the loader
    results = {}
    for armspec in args.arms:
        arm, pr = armspec.split(":")
        denoiser, prior_obj, schedule, cfg, ep = load_run(
            args.dataset, pr, 42, f"{args.exp}/{arm}", device, backbone="raw",
            ckpt_name=args.ckpt)
        for st in args.steps:
            reps = []
            for sd in range(args.eval_seeds):
                torch.manual_seed(sd)
                outs = []
                for s0 in range(0, args.n_samples, args.gen_batch):
                    b = min(args.gen_batch, args.n_samples - s0)
                    Y0 = torch.zeros(b, N, H, device=device)     # UNCONDITIONAL
                    # VE has no stationary law (θ→0): terminal draws N(0, σ_max²) explicit.
                    R_T = (schedule.sigma_max * torch.randn(b, N, H, device=device)
                           if hasattr(schedule, "sigma_max") else None)
                    if args.sampler == "em":
                        samp, _ = reverse_sample_em(prior_obj, schedule, denoiser, Y0,
                                                    n_steps=st, R_T=R_T)
                        samp = torch.nan_to_num(samp, nan=0.0, posinf=0.0, neginf=0.0)
                    else:
                        samp, _ = reverse_sample_ancestral(prior_obj, schedule, denoiser, Y0,
                                                           n_steps=st, R_T=R_T)
                    outs.append(samp.cpu())
                G = torch.cat(outs)
                gm = metric_set(G, L_comb, L_norm, degrees, L_T)
                mmds = {k: float(mmd_distance(gm[k].view(-1, 1), ref[k].view(-1, 1)))
                        for k in ref}
                # aMMD = GAD's headline: mean over TV_S, SC_S, DC_S only (TV_T reported extra).
                reps.append((mmds, float(np.mean([mmds[k] for k in GAD_KEYS])),
                             mmds["TV_T"]))
            am = np.array([r[1] for r in reps])
            mmds = {k: float(np.mean([r[0][k] for r in reps])) for k in ref}
            results[(arm, st)] = float(am.mean())
            sd_str = f" ±{am.std():.3f}" if args.eval_seeds > 1 else ""
            print(f"{arm:8} {st:5d} | {mmds['TV_S']:9.4f} {mmds['SC_S']:9.4f} {mmds['DC_S']:9.4f} "
                  f"| {am.mean():7.4f}{sd_str} | {np.mean([r[2] for r in reps]):8.3f}")
    print("\n=== aMMD summary (lower=better, GAD Fig.3 style) ===")
    arms = [a.split(":")[0] for a in args.arms]
    print(f"{'steps':>5} | " + " | ".join(f"{a:>8}" for a in arms) + " | winner")
    for st in args.steps:
        vals = [results.get((a, st), float("nan")) for a in arms]
        win = arms[int(np.nanargmin(vals))]
        print(f"{st:5d} | " + " | ".join(f"{v:8.4f}" for v in vals) + f" | {win}")


if __name__ == "__main__":
    main()
