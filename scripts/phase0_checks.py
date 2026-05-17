"""Phase 0 verification checks for the Stage 0 code fixes.

Runs on a CUDA GPU (DAIC) to confirm the four Stage 0 code fixes are wired
correctly end-to-end. Mirrors the style of ``scripts/daic/smoke_test.py``:
each check prints PASS/FAIL with timing and the script exits non-zero on
fatal failures.

Currently covered (Steps 1 + 2 of the Stage 0 sequence):

  A. **Val determinism** — ``train_diffusion.val_epoch`` must produce
     identical val loss across two consecutive calls. Exercises the new
     torch.Generator plumbing through ``forward_sample`` on CUDA. Catches
     the pre-fix bug where the ε noise inside ``forward_sample`` was
     drawn from the global RNG and therefore drifted between runs.

  B. **Normalised CRPS** — ``eval_diffusion.py`` must emit the new keys
     (``crps_per_horizon_normalized``, ``crps_average_normalized``,
     ``crps_overall_normalized_pooled``, ``abs_y_mean_*``) and their
     values must satisfy ``Variant2[h] = Variant1[h] / mean|y|[h]``
     to numerical tolerance. Skipped when ``--run-id`` is not provided.

Run via the sbatch wrapper ``scripts/daic-exp/submit_phase0_checks.sbatch``
or directly:

    python scripts/phase0_checks.py --dataset metrla
    python scripts/phase0_checks.py --dataset metrla --run-id <run_id>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Make the project importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_diffusion import (
    DiffusionDataset,
    build_adjacency_torch,
    build_eig_structured,
    val_epoch,
)
from src.diffusion.score_net import ScoreNet
from src.diffusion.sde_torch import eig_to_torch
from src.utils.config import load_config
from src.utils.seed import set_seed
from torch.utils.data import DataLoader

_NUM_NODES = {"metrla": 207, "pemsbay": 325, "pems08f": 170}
_NUM_HORIZONS = 12
_BACKBONE = "gwn"


# ─────────────────────────────────────────────────────────────────────────────
# Stage A — val determinism
# ─────────────────────────────────────────────────────────────────────────────

def check_val_determinism(dataset: str) -> None:
    """Run val_epoch twice on a fresh (untrained) model; assert identical loss.

    Two consecutive val_epoch calls must produce bit-for-bit identical loss:
      - τ comes from a CPU generator re-seeded at the top of val_epoch.
      - ε in forward_sample comes from a device-side generator (CUDA) or a
        torch.manual_seed-reset global RNG (MPS) — both re-set per call.

    Before Step 1's fix, ε was drawn from the global RNG without any reset,
    so a second call would advance and produce a different loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      device={device}")

    set_seed(0)

    prior_yaml = load_config(f"results/prior_params/{_BACKBONE}_{dataset}.yaml")
    target_scale = np.asarray(prior_yaml["target_scale"], dtype=np.float32)
    eig_np = build_eig_structured(dataset, prior_yaml)
    eig_t = eig_to_torch(eig_np, device=device, dtype=torch.float32)

    val_ds = DiffusionDataset(dataset, "val", seed=0, target_scale=target_scale)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"      val_ds size={len(val_ds):,}")

    adj = build_adjacency_torch(dataset, device)
    model = ScoreNet(
        num_horizons=_NUM_HORIZONS,
        num_sensors=_NUM_NODES[dataset],
        hidden_dim=64, num_blocks=4, tau_embed_dim=128,
        adj=adj,
    ).to(device)
    model.eval()
    print(f"      model params={model.num_parameters:,}")

    common = dict(
        eig_t=eig_t, beta_min=0.01, beta_max=50.0, tau_eps=1e-3,
        weighting="eigenbasis", device=device, max_batches=8,
    )
    loss_1 = val_epoch(model, val_loader, **common)
    loss_2 = val_epoch(model, val_loader, **common)
    print(f"      val loss (run 1): {loss_1:.10f}")
    print(f"      val loss (run 2): {loss_2:.10f}")
    if loss_1 != loss_2:
        raise AssertionError(
            f"val loss not deterministic: {loss_1} != {loss_2} "
            f"(Δ={loss_1 - loss_2:.3e})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage B — normalised CRPS in eval JSON
# ─────────────────────────────────────────────────────────────────────────────

_EXPECTED_KEYS = {
    "crps_per_horizon",
    "crps_per_horizon_normalized",
    "crps_average",
    "crps_average_normalized",
    "crps_overall_normalized_pooled",
    "abs_y_mean_per_horizon",
    "abs_y_mean_overall",
}


def check_normalised_crps(dataset: str, run_id: str) -> None:
    """Spawn eval_diffusion.py on a small subset; verify new keys and identities."""
    with tempfile.TemporaryDirectory() as td:
        cmd = [
            sys.executable, "scripts/eval_diffusion.py",
            "--run_id", run_id,
            "--max_windows", "100",
            "--n_samples", "10",
            "--n_steps", "30",
            "--batch_size", "8",
            "--out_dir", td,
        ]
        print(f"      $ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("      STDOUT (tail):", result.stdout[-1500:])
            print("      STDERR (tail):", result.stderr[-1500:])
            raise RuntimeError(f"eval_diffusion.py exited with {result.returncode}")

        json_path = Path(td) / f"eval_{run_id}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"eval output not produced: {json_path}")
        out = json.loads(json_path.read_text())

    missing = _EXPECTED_KEYS - set(out.keys())
    if missing:
        raise AssertionError(f"missing keys in eval JSON: {sorted(missing)}")
    print(f"      all {len(_EXPECTED_KEYS)} expected JSON keys present")

    raw = np.asarray(out["crps_per_horizon"])
    norm = np.asarray(out["crps_per_horizon_normalized"])
    abs_y = np.asarray(out["abs_y_mean_per_horizon"])

    derived = raw / np.maximum(abs_y, 1e-12)
    rel_err = float(np.max(np.abs(norm - derived) / np.maximum(np.abs(norm), 1e-12)))
    print(f"      per-horizon  Variant2 = Variant1 / mean|y|   rel_err = {rel_err:.2e}")
    if rel_err > 1e-6:
        raise AssertionError(f"per-horizon identity violated: rel_err={rel_err:.2e}")

    avg_norm = float(out["crps_average_normalized"])
    pooled_norm = float(out["crps_overall_normalized_pooled"])
    overall_abs_y = float(out["abs_y_mean_overall"])
    avg_raw = float(out["crps_average"])

    print(f"      raw CRPS per horizon (mph): {[f'{x:.2f}' for x in raw]}")
    print(f"      norm CRPS per horizon:      {[f'{x:.4f}' for x in norm]}")
    print(f"      mean|y| per horizon (mph):  {[f'{x:.2f}' for x in abs_y]}")
    print(f"      avg raw CRPS:               {avg_raw:.3f} mph")
    print(f"      avg norm CRPS:              {avg_norm:.4f}")
    print(f"      pooled overall norm CRPS:   {pooled_norm:.4f}")
    print(f"      overall mean|y|:            {overall_abs_y:.2f} mph")

    if dataset == "metrla":
        if not (20.0 < overall_abs_y < 80.0):
            raise AssertionError(
                f"METR-LA overall mean|y|={overall_abs_y:.2f} outside plausible "
                f"mph band [20, 80]"
            )
    if not (0.0 < avg_norm < 1.0):
        raise AssertionError(
            f"avg normalised CRPS = {avg_norm:.4f} outside (0, 1) — suspicious"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="metrla",
                        choices=["metrla", "pemsbay", "pems08f"])
    parser.add_argument("--run-id", default=None,
                        help="Existing trained checkpoint folder name. "
                             "If omitted, Stage B is skipped.")
    args = parser.parse_args()

    print(f"[phase0-checks] dataset={args.dataset}  "
          f"run_id={args.run_id or '(none, Stage B will be skipped)'}")
    print(f"[phase0-checks] python={sys.version.split()[0]}  "
          f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")

    failed: list[str] = []
    skipped: list[str] = []

    def stage(label: str, fn, *args_, **kwargs_) -> None:
        print(f"\n── {label} " + "─" * max(0, 65 - len(label)))
        t0 = time.time()
        try:
            fn(*args_, **kwargs_)
            print(f"  PASS  {label}  ({time.time() - t0:.2f}s)")
        except Exception as e:
            print(f"  FAIL  {label}: {type(e).__name__}: {e}")
            failed.append(label)

    stage("A. Val determinism (val_epoch twice → identical loss)",
          check_val_determinism, args.dataset)

    if args.run_id is None:
        print("\n── B. Normalised CRPS (skipped — no --run-id) "
              + "─" * 21)
        skipped.append("B. Normalised CRPS")
    else:
        stage("B. Normalised CRPS (eval_diffusion JSON keys + identity)",
              check_normalised_crps, args.dataset, args.run_id)

    print()
    if failed:
        print(f"[phase0-checks] FAILED: {', '.join(failed)}")
        return 1
    if skipped:
        print(f"[phase0-checks] PASS (skipped: {', '.join(skipped)})")
        return 0
    print("[phase0-checks] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
