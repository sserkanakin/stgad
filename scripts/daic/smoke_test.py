"""GPU smoke test for the DAIC setup.

Confirms CUDA works, the prior loads, the SDE module functions correctly,
a residual .npz can be read, and wandb logs a couple of metrics. About
30 seconds end-to-end.

Run directly via:

    python scripts/daic/smoke_test.py --dataset metrla

or via the sbatch wrapper `submit_smoke.sbatch`.
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
    parser.add_argument("--dataset", default="metrla", choices=["metrla", "pemsbay"])
    args = parser.parse_args()

    ok = True
    def step(name, fn):
        nonlocal ok
        t0 = time.time()
        try:
            fn()
            print(f"  ✓ {name}  ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            ok = False

    print(f"[smoke] dataset={args.dataset}")
    print(f"[smoke] python={sys.version.split()[0]}  torch={torch.__version__}")

    # ── CUDA ────────────────────────────────────────────────────────
    print("\n── CUDA ────────────────────────────────────────────────")
    def _cuda():
        assert torch.cuda.is_available(), "torch.cuda not available"
        d = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"      device: {d}  capability: {cap}")
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.T  # ~1 GFLOP, just to exercise the kernel path
        torch.cuda.synchronize()
        assert y.shape == (1024, 1024)
    step("cuda available + matmul", _cuda)

    # ── Project imports ────────────────────────────────────────────
    print("\n── Project imports ────────────────────────────────────")
    def _import_src():
        from src.diffusion import sde, prior_fit  # noqa: F401
        from src.data.adjacency import load_adjacency  # noqa: F401
    step("src.diffusion + src.data", _import_src)

    # ── Data files ─────────────────────────────────────────────────
    print("\n── Data files ─────────────────────────────────────────")
    def _resid_load():
        path = Path(f"results/residuals/gwn_{args.dataset}_train_seed0.npz")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run submit_extract_residuals.sbatch first "
                f"(or sync from local with sync_to_daic.sh)"
            )
        data = np.load(path)
        r = data["residuals"]
        print(f"      residuals: {r.shape} {r.dtype}")
        assert r.ndim == 4 and r.shape[1] == 12
    step("residual npz loads", _resid_load)

    def _prior_load():
        path = Path(f"results/prior_params/gwn_{args.dataset}.yaml")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run submit_fit_prior.sbatch first"
            )
        with path.open() as f:
            p = yaml.safe_load(f)
        assert p.get("standardized") is True, "prior must be standardised (v2)"
        assert len(p["target_scale"]) == 12
        print(f"      α={p['alpha']:.4f}  ρ₁={p['rho1']:.4f}  "
              f"ρ₂={p['rho2']:.4f}  β={p['beta']:.4f}  ε_s={p['epsilon_s']:.3f}")
    step("prior yaml loads (v2 standardised)", _prior_load)

    # ── SDE math ───────────────────────────────────────────────────
    print("\n── SDE forward marginal ───────────────────────────────")
    def _sde_roundtrip():
        from src.diffusion.sde import (
            eigendecompose_prior, forward_sample,
            to_spectral, from_spectral,
        )
        from src.diffusion.prior_fit import build_Q_H, build_Sigma_S
        eig_path = Path(f"data/processed/{args.dataset}_laplacian_eigs.npz")
        if not eig_path.exists():
            raise FileNotFoundError(f"{eig_path} missing")
        eig_data = np.load(eig_path)
        eigvals_L = eig_data["eigenvalues"]
        eigvecs_L = eig_data["eigenvectors"]
        with open(f"results/prior_params/gwn_{args.dataset}.yaml") as f:
            p = yaml.safe_load(f)
        Q_H = build_Q_H(
            delta_h=np.array(p["delta_h"]),
            alpha=p["alpha"], rho1=p["rho1"], rho2=p["rho2"],
        )
        Sigma_S = build_Sigma_S(p["beta"], p["epsilon_s"], eigvals_L, eigvecs_L)
        eig = eigendecompose_prior(Q_H, Sigma_S)
        rng = np.random.default_rng(0)
        r_0 = rng.standard_normal((4, 12, eigvals_L.shape[0]))
        r_tau = forward_sample(
            r_0, tau=0.5, eig=eig, beta_min=0.01, beta_max=50.0, rng=rng,
        )
        assert r_tau.shape == r_0.shape
        # spectral round-trip
        r_t = to_spectral(r_0, eig["U_T"], eig["U_S"])
        r_back = from_spectral(r_t, eig["U_T"], eig["U_S"])
        err = np.abs(r_back - r_0).max()
        print(f"      forward marginal: τ=0.5  shape={r_tau.shape}")
        print(f"      spectral round-trip max abs err: {err:.2e}")
        assert err < 1e-10
    step("eigendecomp + forward marginal + spectral round-trip", _sde_roundtrip)

    # ── wandb ──────────────────────────────────────────────────────
    print("\n── wandb ──────────────────────────────────────────────")
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
        print(f"      mode={mode}  run.url={url}")
        wandb.finish()
    step("wandb init + log + finish", _wandb)

    print()
    if ok:
        print("[smoke] ALL CHECKS PASSED. Cluster is ready for exp-003.")
        return 0
    else:
        print("[smoke] FAILURES above. Fix before submitting training jobs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
