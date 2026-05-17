"""GPU smoke test for the DAIC setup.

Tests *infrastructure only* — does not require residuals or prior YAML
files. Runs in ~30 seconds. Checks:

  1. CUDA available + a small matmul succeeds on the GPU.
  2. Project modules import cleanly.
  3. SDE math runs end-to-end on a synthetic prior (eigendecompose
     + forward marginal + spectral round-trip).
  4. wandb init + log + finish (online mode reaches dashboard).
  5. Optional: residual .npz / prior YAML loads — *warned*, not failed,
     so this passes even before extract_residuals + fit_prior have been
     run on the cluster.

Run via the sbatch wrapper `submit_smoke.sbatch` or directly:

    python scripts/daic/smoke_test.py --dataset metrla
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


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
            tag = "✗" if fatal else "!"
            print(f"  {tag} {name}: {e}")
            (failed if fatal else warned).append(name)

    print(f"[smoke] dataset={args.dataset}")
    print(f"[smoke] python={sys.version.split()[0]}  torch={torch.__version__}")

    # ── 1. CUDA (fatal) ──────────────────────────────────────────────
    print("\n── 1. CUDA ─────────────────────────────────────────────")
    def _cuda():
        assert torch.cuda.is_available(), "torch.cuda not available"
        d = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"      device: {d}  capability: {cap}")
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        assert y.shape == (1024, 1024)
    step("cuda available + matmul", _cuda)

    # ── 2. Project imports (fatal) ───────────────────────────────────
    print("\n── 2. Project imports ──────────────────────────────────")
    def _import_src():
        from src.diffusion import sde, prior_fit  # noqa: F401
        from src.data.adjacency import load_adjacency  # noqa: F401
    step("src.diffusion + src.data", _import_src)

    # ── 3. SDE math with synthetic prior (fatal) ─────────────────────
    print("\n── 3. SDE forward marginal (synthetic prior) ───────────")
    def _sde_synthetic():
        from src.diffusion.sde import (
            eigendecompose_prior, forward_sample,
            to_spectral, from_spectral,
        )
        rng = np.random.default_rng(0)
        # Synthetic 12×N priors: temporal AR(1)-like, spatial Wishart-ish.
        H, N = 12, 20
        lag = np.abs(np.arange(H)[:, None] - np.arange(H)[None, :])
        Q_H = 0.8 ** lag.astype(np.float64)
        A = rng.standard_normal((N, N))
        Sigma_S = (A @ A.T) / N + 0.1 * np.eye(N)
        eig = eigendecompose_prior(Q_H, Sigma_S)
        r_0 = rng.standard_normal((4, H, N))
        r_tau = forward_sample(
            r_0, tau=0.5, eig=eig, beta_min=0.01, beta_max=50.0, rng=rng,
        )
        assert r_tau.shape == r_0.shape
        # Spectral round-trip
        r_back = from_spectral(to_spectral(r_0, eig["U_T"], eig["U_S"]),
                               eig["U_T"], eig["U_S"])
        err = float(np.abs(r_back - r_0).max())
        print(f"      forward marginal: τ=0.5  shape={r_tau.shape}")
        print(f"      spectral round-trip max abs err: {err:.2e}")
        assert err < 1e-10, f"round-trip err too high: {err}"
    step("eigendecompose + forward_sample + spectral round-trip", _sde_synthetic)

    # ── 4. wandb (fatal in online mode, non-fatal otherwise) ─────────
    print("\n── 4. wandb ────────────────────────────────────────────")
    def _wandb():
        import wandb
        mode = os.environ.get("WANDB_MODE", "online")
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "thesis-diffusion"),
            name=f"daic-smoke-{args.dataset}",
            mode=mode,
            tags=["smoke", "daic"],
            reinit=True,
        )
        wandb.log({"smoke/cuda_ok": 1.0, "smoke/dataset": args.dataset})
        url = getattr(run, "url", "(offline)") or "(offline)"
        print(f"      mode={mode}  url={url}")
        wandb.finish()
    step("wandb init + log + finish", _wandb)

    # ── 5. Optional real-data checks (non-fatal) ─────────────────────
    print("\n── 5. Optional: real residuals + prior YAML ────────────")
    def _resid_load():
        path = Path(f"results/residuals/gwn_{args.dataset}_train_seed0.npz")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not present yet — rebuild on DAIC via "
                f"submit_extract_residuals.sbatch (expected for first run)"
            )
        data = np.load(path)
        r = data["residuals"]
        print(f"      residuals: {r.shape} {r.dtype}")
        assert r.ndim == 4 and r.shape[1] == 12
    step("residual npz loads", _resid_load, fatal=False)

    def _prior_load():
        path = Path(f"results/prior_params/gwn_{args.dataset}.yaml")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not present yet — fit on DAIC via "
                f"submit_fit_prior.sbatch (expected for first run)"
            )
        with path.open() as f:
            p = yaml.safe_load(f)
        assert p.get("standardized") is True, "prior must be standardised (v2)"
        assert len(p["target_scale"]) == 12
        print(f"      α={p['alpha']:.4f}  ρ₁={p['rho1']:.4f}  "
              f"ρ₂={p['rho2']:.4f}  β={p['beta']:.4f}  ε_s={p['epsilon_s']:.3f}")
    step("prior yaml loads (v2 standardised)", _prior_load, fatal=False)

    # ── verdict ──────────────────────────────────────────────────────
    print()
    if failed:
        print(f"[smoke] FAILED steps: {', '.join(failed)}")
        print(f"[smoke] Infrastructure is broken — fix before submitting jobs.")
        return 1
    if warned:
        print(f"[smoke] Infrastructure OK. Pending steps: {', '.join(warned)}")
        print(f"[smoke] These run next:  sbatch submit_extract_residuals.sbatch")
        return 0
    print("[smoke] ALL CHECKS PASSED — infrastructure AND data ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
