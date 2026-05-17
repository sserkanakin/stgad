"""Pre-flight checker for the Stage-2 eval submission.

Validates that every input the eval pipeline needs is present, configured
correctly, and that a tiny end-to-end sample-and-score round-trip works
without NaN / shape errors. Runs in ~1-2 minutes on a single A40 (or CPU
fall-back; it's tiny). Run this BEFORE submitting the six real eval jobs.

Checks:
  A. For each of the 6 expected (dataset, prior) checkpoints:
     - The checkpoint dir + best.pt file exists
     - The saved cfg matches the expected (dataset, prior) tuple
     - The corresponding residuals npz exists for the test split
     - The prior YAML exists (for the dataset)
     - The Laplacian eigendecomp is cached
  B. End-to-end smoke: for ONE checkpoint per dataset, load model + prior,
     do a 4-window × K=2 × n_steps=10 sample, verify output shapes + no NaN.
  C. Throughput probe: same tiny sample, time the wall, extrapolate to
     full-test-set wall time for each dataset.

Usage:
    python scripts/eval_preflight.py

    # CPU-only (for local sanity):
    python scripts/eval_preflight.py --cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.adjacency import compute_normalized_laplacian, load_adjacency
from src.data.dataset import TrafficDataset
from src.diffusion.prior_fit import build_Q_H, build_Sigma_S
from src.diffusion.sampling import sample_per_window
from src.diffusion.score_net import ScoreNet
from src.diffusion.sde import eigendecompose_prior
from src.diffusion.sde_torch import eig_to_torch
from src.utils.config import load_config

_DATASETS = ("metrla", "pemsbay", "pems08f")
_PRIORS = ("structured", "isotropic")
_NUM_NODES = {"metrla": 207, "pemsbay": 325, "pems08f": 170}
_NUM_HORIZONS = 12
_BACKBONE = "gwn"

# Test-set window counts for extrapolation
_TEST_WINDOWS = {"metrla": 6850, "pemsbay": 10423, "pems08f": 3548}


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def find_checkpoint(dataset: str, prior: str, seed: int = 0) -> Path | None:
    """Return the newest matching checkpoint dir, or None if missing."""
    pattern = f"results/checkpoints/dsm_{prior}_{dataset}_seed{seed}_*"
    candidates = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def check_input_files(dataset: str, prior: str) -> tuple[bool, list[str]]:
    """Validate that everything eval needs is on disk for (dataset, prior)."""
    issues: list[str] = []

    # Checkpoint dir + best.pt
    ckpt_dir = find_checkpoint(dataset, prior)
    if ckpt_dir is None:
        issues.append(f"no checkpoint dir matching dsm_{prior}_{dataset}_seed0_*")
        return False, issues
    best_pt = ckpt_dir / "best.pt"
    if not best_pt.exists():
        issues.append(f"{best_pt} missing")
        return False, issues

    # Saved cfg matches expectation
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    if cfg.get("dataset") != dataset:
        issues.append(f"ckpt cfg dataset={cfg.get('dataset')} != expected {dataset}")
    if cfg.get("prior") != prior:
        issues.append(f"ckpt cfg prior={cfg.get('prior')} != expected {prior}")

    # Residuals
    resid = Path(f"results/residuals/{_BACKBONE}_{dataset}_test_seed0.npz")
    if not resid.exists():
        issues.append(f"missing residuals: {resid}")

    # Prior YAML
    yaml_path = Path(f"results/prior_params/{_BACKBONE}_{dataset}.yaml")
    if not yaml_path.exists():
        issues.append(f"missing prior YAML: {yaml_path}")
    elif prior == "structured":
        with open(yaml_path) as f:
            p = yaml.safe_load(f)
        if not p.get("standardized"):
            issues.append(f"{yaml_path} not flagged standardized=true")
        if "target_scale" not in p or len(p["target_scale"]) != _NUM_HORIZONS:
            issues.append(f"{yaml_path} target_scale missing or wrong length")

    # Laplacian eigs
    eigs = Path(f"data/processed/{dataset}_laplacian_eigs.npz")
    if not eigs.exists():
        issues.append(f"missing Laplacian eigs: {eigs}")

    # Scaler
    scaler = Path(f"data/processed/{dataset}_scaler.yaml")
    if not scaler.exists():
        issues.append(f"missing scaler: {scaler}")

    # Processed test split
    test_npz = Path(f"data/processed/{dataset}_test.npz")
    if not test_npz.exists():
        issues.append(f"missing test split: {test_npz}")

    return len(issues) == 0, issues


def end_to_end_smoke(
    dataset: str, prior: str, device: torch.device,
    batch_size: int = 16, K: int = 50, n_steps: int = 100,
) -> tuple[bool, str, dict]:
    """Realistic-size single-batch smoke at real eval dimensions.

    Uses ``batch_size`` windows × K=50 samples × n_steps=100 on one batch
    of REAL test data so per-batch timing represents real eval throughput.
    Extrapolation: wall_full = setup_overhead + wall_smoke × n_batches_real.

    Returns (ok, message, info). info includes wall_seconds and
    full_test_estimate_min.
    """
    BATCH = batch_size
    N_STEPS = n_steps
    try:
        ckpt_dir = find_checkpoint(dataset, prior)
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        cfg = ckpt["config"]

        prior_yaml = load_config(f"results/prior_params/{_BACKBONE}_{dataset}.yaml")
        target_scale = np.asarray(prior_yaml["target_scale"], dtype=np.float32)

        if prior == "structured":
            delta_h = np.asarray(prior_yaml["delta_h"], dtype=np.float64)
            Q_H = build_Q_H(delta_h=delta_h, alpha=prior_yaml["alpha"],
                            rho1=prior_yaml["rho1"], rho2=prior_yaml["rho2"])
            eig_L = np.load(f"data/processed/{dataset}_laplacian_eigs.npz")
            Sigma_S = build_Sigma_S(prior_yaml["beta"], prior_yaml["epsilon_s"],
                                    eig_L["eigenvalues"].astype(np.float64),
                                    eig_L["eigenvectors"].astype(np.float64))
            eig_np = eigendecompose_prior(Q_H, Sigma_S)
        else:
            H, N = _NUM_HORIZONS, _NUM_NODES[dataset]
            eig_np = {
                "U_T": np.eye(H, dtype=np.float64), "c_T": np.ones(H, dtype=np.float64),
                "U_S": np.eye(N, dtype=np.float64), "c_S": np.ones(N, dtype=np.float64),
                "s": np.ones((H, N), dtype=np.float64), "q": np.ones((H, N), dtype=np.float64),
            }
        eig_t = eig_to_torch(eig_np, device=device, dtype=torch.float32)

        # Adjacency
        adj_mx = load_adjacency(dataset)
        L = compute_normalized_laplacian(adj_mx)
        A_norm = np.eye(L.shape[0]) - L
        adj = torch.as_tensor(A_norm, dtype=torch.float32, device=device)

        # Model
        model = ScoreNet(
            num_horizons=_NUM_HORIZONS,
            num_sensors=_NUM_NODES[dataset],
            hidden_dim=cfg["hidden_dim"],
            num_blocks=cfg["num_blocks"],
            tau_embed_dim=cfg["tau_embed_dim"],
            adj=adj,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Real test conditioning for the first BATCH windows
        scaler_path = f"data/processed/{dataset}_scaler.yaml"
        test_npz = f"data/processed/{dataset}_test.npz"
        traffic = TrafficDataset(test_npz, scaler_path)
        resid = np.load(f"results/residuals/{_BACKBONE}_{dataset}_test_seed0.npz")
        y_pred_raw = resid["y_pred"].squeeze(-1).astype(np.float32)[:BATCH]    # (B, H, N)
        ts = target_scale.reshape(1, _NUM_HORIZONS, 1)
        y_pred_std_np = (y_pred_raw / ts).astype(np.float32)

        # TrafficDataset.x is (H, N, 2); ScoreNet wants (2, H, N).
        x_history = torch.stack([
            traffic.x[i].float().permute(2, 0, 1).contiguous() for i in range(BATCH)
        ], dim=0).to(device)
        y_pred = torch.from_numpy(y_pred_std_np).to(device)

        H, N = _NUM_HORIZONS, _NUM_NODES[dataset]

        g = None
        if device.type != "mps":
            g = torch.Generator(device=device.type).manual_seed(42)

        # Warmup pass to pre-allocate CUDA buffers (don't include in timing)
        _ = sample_per_window(
            model=model, x_history=x_history[:2], y_pred=y_pred[:2],
            n_samples=4, n_steps=4, eig=eig_t,
            beta_min=cfg["beta_min"], beta_max=cfg["beta_max"],
            tau_eps=1e-3, generator=g,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Realistic-size batch with real eval defaults
        t0 = time.time()
        r_hat = sample_per_window(
            model=model, x_history=x_history, y_pred=y_pred,
            n_samples=K, n_steps=N_STEPS, eig=eig_t,
            beta_min=cfg["beta_min"], beta_max=cfg["beta_max"],
            tau_eps=1e-3, generator=g,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0

        # Shape + NaN checks
        expected_shape = (BATCH, K, H, N)
        if tuple(r_hat.shape) != expected_shape:
            return False, f"shape mismatch: got {tuple(r_hat.shape)}, expected {expected_shape}", {}
        n_nans = int(torch.isnan(r_hat).sum())
        n_infs = int(torch.isinf(r_hat).sum())
        if n_nans > 0 or n_infs > 0:
            return False, f"sample has {n_nans} NaNs and {n_infs} Infs", {}

        # De-standardise + add y_pred — real raw units
        ts4 = target_scale.reshape(1, 1, H, 1)
        r_hat_np = r_hat.cpu().numpy()
        samples_raw = y_pred_raw[:, None, :, :] + r_hat_np * ts4              # (B, K, H, N)

        # Call CRPS / PIT / pinball once to verify they run on real samples
        from src.eval.probabilistic import crps_ensemble, pit_values, pinball_loss
        mask = resid["mask"].squeeze(-1).astype(bool)[:BATCH]
        y_true_raw = resid["y_true"].squeeze(-1).astype(np.float32)[:BATCH]
        samples_for_metrics = np.moveaxis(samples_raw, 1, -1)                  # (B, H, N, K)
        # Horizon h=0
        mh = mask[:, 0, :]
        y_h = y_true_raw[:, 0, :][mh]
        s_h = samples_for_metrics[:, 0, :, :][mh]
        if y_h.size == 0:
            return False, "all positions masked at h=0 — unexpected", {}
        crps_h = crps_ensemble(y_h, s_h)
        pit_h = pit_values(y_h, s_h)
        pin_h = pinball_loss(y_h, s_h, quantile=0.5)
        if np.isnan(crps_h).any() or np.isnan(pit_h).any() or np.isnan(pin_h).any():
            return False, "NaN in CRPS/PIT/pinball output", {}

        # Extrapolation: wall_real = setup_overhead + wall_per_batch × n_batches_real.
        # Setup overhead (~30s on cold cache) is amortised. Real eval iterates
        # n_batches_real = ceil(test_windows / BATCH) batches at this exact size.
        n_batches_real = int(np.ceil(_TEST_WINDOWS[dataset] / BATCH))
        # Add ~30s setup overhead (model load + dataset load + first-batch warmup)
        SETUP_OVERHEAD_S = 30.0
        full_seconds = SETUP_OVERHEAD_S + wall * n_batches_real
        return True, "OK", {
            "wall_seconds_per_batch": wall,
            "n_batches_real":         n_batches_real,
            "model_params":           sum(p.numel() for p in model.parameters()),
            "crps_h0_mean":           float(crps_h.mean()),
            "full_test_estimate_min": full_seconds / 60.0,
        }
    except Exception as e:
        import traceback
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", {}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available.")
    p.add_argument("--skip-smoke", action="store_true",
                   help="Only verify files; skip the tiny sample-and-score round-trip.")
    args = p.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"[eval_preflight] device={device}  torch={torch.__version__}")

    # ── A. File presence + cfg validity (all 6 ckpts) ────────────────────────
    print()
    print("─" * 78)
    print("A. File presence + checkpoint config validity")
    print("─" * 78)
    file_failures = 0
    for ds in _DATASETS:
        for prior in _PRIORS:
            ok, issues = check_input_files(ds, prior)
            tag = "OK  " if ok else "FAIL"
            print(f"  [{tag}] {ds:>8} / {prior:>10}")
            if not ok:
                file_failures += 1
                for i in issues:
                    print(f"           - {i}")

    if file_failures:
        print(f"\n  {file_failures} (dataset, prior) pairs FAILED file checks.")
        print("  Fix those before submitting eval. Aborting smoke test.")
        return 1
    print()
    print("  All 6 (dataset, prior) checkpoints + dependencies present.")

    if args.skip_smoke:
        print("\n[eval_preflight] --skip-smoke set, done.")
        return 0

    # ── B. End-to-end smoke per dataset, scanning batch sizes for best GPU saturation ─
    print()
    print("─" * 78)
    print("B. Batch-size scan (per dataset, K=50, n_steps=100, real test data)")
    print("─" * 78)
    print("   Goal: find the batch_size that maximises GPU utilisation. A40 has 46 GB;")
    print("   our model is small (~230k params) so larger batches reduce Python/")
    print("   kernel-launch overhead per window without hitting memory limits.")
    print()
    print(f"   {'dataset':>8}  {'B':>4}  {'per-batch':>10}  {'n_batches':>10}  "
          f"{'full est.':>11}  {'win/s':>7}")
    print("   " + "─" * 64)

    summary = {}     # {dataset: {best_batch_size, info}}
    smoke_failures = 0
    candidate_batch_sizes = (16, 32, 64, 128)

    for ds in _DATASETS:
        best_per_window_s = float("inf")
        best_B = None
        best_info = None
        for B in candidate_batch_sizes:
            ok, msg, info = end_to_end_smoke(ds, "structured", device, batch_size=B)
            if not ok:
                # OOM or other failure at this B — usually means we've gone too big.
                print(f"   {ds:>8}  {B:>4}  FAIL ({msg.split(':')[0]})")
                # Out-of-memory at higher B isn't a fatal preflight failure if a
                # smaller B succeeded; only flag fail if every B failed.
                if best_B is None and B == candidate_batch_sizes[0]:
                    smoke_failures += 1
                break  # stop scanning bigger B once one OOMs
            n_batches = info["n_batches_real"]
            full_min = info["full_test_estimate_min"]
            per_batch = info["wall_seconds_per_batch"]
            per_window_s = per_batch / B
            win_per_s = B / per_batch
            marker = ""
            if per_window_s < best_per_window_s:
                best_per_window_s = per_window_s
                best_B = B
                best_info = info
                marker = " ★"
            print(f"   {ds:>8}  {B:>4}  {per_batch:>7.2f}s  {n_batches:>10d}  "
                  f"{full_min:>8.1f}m  {win_per_s:>6.1f}{marker}")
        if best_B is None:
            print(f"   [FAIL] {ds}: no batch size succeeded")
            continue
        summary[ds] = {"batch_size": best_B, **best_info}

    if smoke_failures or len(summary) < len(_DATASETS):
        print(f"\n  Some datasets failed smoke — investigate before submitting eval.")
        return 1

    # ── C. Recommended (batch_size, --time=) per dataset ─────────────────────
    print()
    print("─" * 78)
    print("C. Recommendation per dataset (best batch_size + --time= with +25% margin)")
    print("─" * 78)
    print(f"   {'dataset':>8}  {'best B':>7}  {'est. min':>9}  "
          f"{'+ margin':>10}  {'--time='}")
    print("   " + "─" * 60)
    for ds, info in summary.items():
        base_min = info["full_test_estimate_min"]
        est_min_with_margin = base_min * 1.25 + 5.0
        hh, mm = int(est_min_with_margin // 60), int(round(est_min_with_margin % 60))
        if mm == 60:
            hh, mm = hh + 1, 0
        print(f"   {ds:>8}  {info['batch_size']:>7d}  {base_min:>7.1f}m  "
              f"{est_min_with_margin:>8.1f}m  {hh:02d}:{mm:02d}:00")

    # ── Generate ready-to-paste launch commands ──────────────────────────────
    print()
    print("─" * 78)
    print("D. Ready-to-paste launch commands for the six real eval jobs")
    print("─" * 78)
    print()
    print("declare -A BS=(")
    for ds, info in summary.items():
        print(f"  [{ds}]={info['batch_size']}")
    print(")")
    print("declare -A T=(")
    for ds, info in summary.items():
        est_min_with_margin = info["full_test_estimate_min"] * 1.25 + 5.0
        hh, mm = int(est_min_with_margin // 60), int(round(est_min_with_margin % 60))
        if mm == 60: hh, mm = hh + 1, 0
        print(f"  [{ds}]={hh:02d}:{mm:02d}:00")
    print(")")
    print("for ds in metrla pemsbay pems08f; do")
    print("  for prior in structured isotropic; do")
    print("    run_dir=$(ls -d results/checkpoints/dsm_${prior}_${ds}_seed0_*/ | head -1)")
    print("    run_id=$(basename \"$run_dir\")")
    print("    [ -f \"experiments/score-network/eval_${run_id}.json\" ] && { echo \"[skip] $run_id\"; continue; }")
    print("    sbatch --time=${T[$ds]} \\")
    print("           --export=RUN_ID=${run_id},BATCH_SIZE=${BS[$ds]} \\")
    print("           scripts/daic/submit_eval.sbatch")
    print("  done")
    print("done")

    print()
    print("[eval_preflight] ALL CHECKS PASSED")
    print("  Bigger batch_size DOES NOT change eval results (each window's K samples")
    print("  are still independent). It only changes how many windows are processed")
    print("  in parallel per forward pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
