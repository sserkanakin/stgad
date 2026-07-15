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

import numpy as np
import torch

from src.data.adjacency import compute_normalized_laplacian, load_adjacency
from src.data.residual_dataset import train_residual_sigma
from src.diffusion.reverse import reverse_sample_ancestral
from src.eval.gad_signal_metrics import mmd_distance, spectral_centroid, total_variation, wasserstein_1d
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


def metric_set(X, L_S, L_T):
    """GAD metrics on (B, N, H) windows: spatial TV/SC (horizons as features),
    temporal TV (sensors as features, path graph)."""
    return {
        "TV_S": total_variation(X, L_S),                        # (B,)
        "SC_S": spectral_centroid(X, L=L_S),
        "TV_T": total_variation(X.transpose(1, 2), L_T),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metrla")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--arms", nargs="+", default=["kron_u:kronecker", "iso_u:isotropic"])
    ap.add_argument("--steps", type=int, nargs="+", default=[2, 5, 10, 25, 50])
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    Xt = load_test_windows(args.dataset)
    N, H = Xt.shape[1], Xt.shape[2]
    L_S = torch.tensor(compute_normalized_laplacian(np.asarray(load_adjacency(args.dataset, repair=True))),
                       dtype=torch.float32)
    L_T = torch.tensor(path_laplacian(H), dtype=torch.float32)
    ref = metric_set(Xt, L_S, L_T)
    print(f"=== {args.dataset}: unconditional GAD-ST vs iso · {args.n_samples} samples vs "
          f"{Xt.shape[0]} test windows · metrics TV_S/SC_S/TV_T ===")
    print(f"{'arm':8} {'steps':>5} | {'MMD TV_S':>9} {'MMD SC_S':>9} {'MMD TV_T':>9} | {'aMMD':>7} | {'W1 TV_S':>8}")

    from scripts.eval_diffusion import load_run                  # reuse the loader
    results = {}
    for armspec in args.arms:
        arm, pr = armspec.split(":")
        denoiser, prior_obj, schedule, cfg, ep = load_run(
            args.dataset, pr, 42, f"{args.exp}/{arm}", device, backbone="raw",
            ckpt_name=args.ckpt)
        torch.manual_seed(0)
        for st in args.steps:
            outs = []
            for s0 in range(0, args.n_samples, args.gen_batch):
                b = min(args.gen_batch, args.n_samples - s0)
                Y0 = torch.zeros(b, N, H, device=device)         # UNCONDITIONAL
                samp, _ = reverse_sample_ancestral(prior_obj, schedule, denoiser, Y0, n_steps=st)
                outs.append(samp.cpu())
            G = torch.cat(outs)
            gm = metric_set(G, L_S, L_T)
            mmds = {k: float(mmd_distance(gm[k].view(-1, 1), ref[k].view(-1, 1)))
                    for k in ref}
            ammd = float(np.mean(list(mmds.values())))
            w1 = float(wasserstein_1d(gm["TV_S"], ref["TV_S"]))
            results[(arm, st)] = ammd
            print(f"{arm:8} {st:5d} | {mmds['TV_S']:9.4f} {mmds['SC_S']:9.4f} {mmds['TV_T']:9.4f} "
                  f"| {ammd:7.4f} | {w1:8.3f}")
    print("\n=== aMMD summary (lower=better, GAD Fig.3 style) ===")
    arms = [a.split(":")[0] for a in args.arms]
    print(f"{'steps':>5} | " + " | ".join(f"{a:>8}" for a in arms) + " | winner")
    for st in args.steps:
        vals = [results.get((a, st), float("nan")) for a in arms]
        win = arms[int(np.nanargmin(vals))]
        print(f"{st:5d} | " + " | ".join(f"{v:8.4f}" for v in vals) + f" | {win}")


if __name__ == "__main__":
    main()
