"""gpu smoke test for the daic setup.

infrastructure only — runs in well under a minute and needs no trained
artefacts. checks:

  1. cuda is available and a small matmul runs on the gpu
  2. the project modules import cleanly
  3. the forward sampler and the denoiser run end to end on a toy prior
  4. wandb init + log + finish
  5. optional: the real prior + residual cache load (warned, not failed, since
     they only exist after the backbone + residual jobs have run)

run via submit_smoke.sbatch or directly:

    python scripts/daic/smoke_test.py --dataset metrla
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _toy_prior(N: int = 12, H: int = 12, seed: int = 0):
    # a small but valid kronecker prior, so we can exercise the math without
    # any trained artefacts on disk.
    from src.data.adjacency import compute_normalized_laplacian
    from src.diffusion.prior import KroneckerPrior

    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 1, (N, N))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    lam_S, V_S = np.linalg.eigh(compute_normalized_laplacian(A))

    B = rng.standard_normal((H, H))
    Sigma_T = (B @ B.T) / H + 0.5 * np.eye(H)
    mu_T, V_T = np.linalg.eigh(Sigma_T)

    gamma = 0.2
    theta = (lam_S[:, None] + gamma) / mu_T[None, :]
    return KroneckerPrior(
        dataset="toy", gamma=gamma, V_S=V_S, lam_S=lam_S, V_T=V_T, mu_T=mu_T,
        theta=theta, delta_h=np.diag(Sigma_T).copy(), R_H=np.eye(H),
    ), A


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="metrla", choices=["metrla", "pemsbay", "pems08f"])
    args = parser.parse_args()

    failed: list[str] = []
    warned: list[str] = []

    def step(name, fn, fatal=True):
        t0 = time.time()
        try:
            fn()
            print(f"  ✓ {name}  ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"  {'✗' if fatal else '!'} {name}: {e}")
            (failed if fatal else warned).append(name)

    device = _pick_device()
    print(f"[smoke] dataset={args.dataset}  device={device}")
    print(f"[smoke] python={sys.version.split()[0]}  torch={torch.__version__}")

    # 1. cuda
    print("\n-- 1. device --")
    def _device_matmul():
        if device.type == "cuda":
            print(f"      {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
        x = torch.randn(1024, 1024, device=device)
        (x @ x.T).sum().item()
    step("device matmul", _device_matmul)

    # 2. imports
    print("\n-- 2. project imports --")
    def _imports():
        import src.diffusion.forward_fcps, src.diffusion.prior  # noqa: F401
        import src.diffusion.reverse, src.diffusion.denoiser  # noqa: F401
        import src.data.adjacency, src.data.dataset  # noqa: F401
        from src.utils import paths  # noqa: F401
    step("src.diffusion + src.data + paths", _imports)

    # 3. forward sampler + denoiser on a toy prior
    print("\n-- 3. forward sampler + denoiser (toy prior) --")
    def _forward_and_denoise():
        from src.diffusion.forward_fcps import (
            FCPSSchedule, prior_to_torch, sample_R_t_given_R0_torch,
        )
        from src.diffusion.denoiser import R0Denoiser

        prior, A = _toy_prior()
        N, H, Bsz = prior.N, prior.H, 2
        sched = FCPSSchedule(c_min=0.05, c_0=500.0, alpha=7.0)
        prior_t = prior_to_torch(prior, device=device)

        R0 = torch.randn(Bsz, N, H, device=device)
        t = torch.rand(Bsz, device=device) * sched.T
        R_t = sample_R_t_given_R0_torch(R0, t, prior_t, sched)
        assert R_t.shape == (Bsz, N, H) and torch.isfinite(R_t).all()

        denoiser = R0Denoiser(N=N, H=H, adjacency=A, device=str(device)).to(device)
        R0_hat = denoiser(R_t, t, torch.zeros_like(R_t))
        assert R0_hat.shape == (Bsz, N, H) and torch.isfinite(R0_hat).all()
        print(f"      R_t {tuple(R_t.shape)} -> R0_hat {tuple(R0_hat.shape)}")
    step("forward sample + denoiser forward", _forward_and_denoise)

    # 4. wandb
    print("\n-- 4. wandb --")
    def _wandb():
        import wandb
        mode = os.environ.get("WANDB_MODE", "online")
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "thesis-diffusion"),
            name=f"smoke_{args.dataset}", job_type="smoke",
            tags=["smoke", args.dataset], mode=mode, reinit=True,
        )
        wandb.log({"smoke/ok": 1.0})
        print(f"      mode={mode}  url={getattr(run, 'url', '(offline)') or '(offline)'}")
        wandb.finish()
    step("wandb init + log + finish", _wandb)

    # 5. optional real artefacts (need backbone + residual jobs to have run)
    print("\n-- 5. optional: real prior + residual cache --")
    def _real_prior():
        from src.utils import paths
        from src.diffusion.prior import load_prior
        res = paths.residual_npz(args.dataset, "train", 0)
        if not res.exists():
            raise FileNotFoundError(f"{res} not present yet (run extract_residuals first)")
        prior = load_prior(args.dataset)
        print(f"      load_prior ok: N={prior.N} H={prior.H} gamma={prior.gamma:.4f}")
    step("load_prior from real residuals", _real_prior, fatal=False)

    print()
    if failed:
        print(f"[smoke] FAILED: {', '.join(failed)} — fix before submitting jobs.")
        return 1
    if warned:
        print(f"[smoke] infra OK. pending (expected on first run): {', '.join(warned)}")
        return 0
    print("[smoke] ALL CHECKS PASSED — infrastructure and data ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
