"""Evaluate a trained diffusion checkpoint on test residuals (exp-003 §14.5).

Pipeline:
  1. Load best.pt for a given run_id from results/checkpoints/.
  2. Build the same prior eigendecomp + adjacency the training used.
  3. For each test window (batched), generate K reverse-SDE samples.
  4. De-standardise (multiply by target_scale[h]) and add the backbone
     y_pred to produce samples in raw mph.
  5. Compute per-horizon CRPS / PIT / pinball aggregates against y_true.
  6. Write metrics to experiments/exp-003-score-network/eval_<run_id>.json
     and produce per-horizon CRPS bar + PIT histogram figures.

Usage:
    python scripts/eval_diffusion.py \
        --run_id dsm_structured_metrla_seed0_f07d4ad3 \
        --n_samples 50 --n_steps 200

    # Quick sanity (smaller subset, fewer samples)
    python scripts/eval_diffusion.py --run_id <…> --max_windows 200 \
        --n_samples 20 --n_steps 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.adjacency import compute_normalized_laplacian, load_adjacency
from src.data.dataset import TrafficDataset
from src.diffusion.prior_fit import build_Q_H, build_Sigma_S
from src.diffusion.sampling import sample_per_window
from src.diffusion.score_net import ScoreNet
from src.diffusion.sde import eigendecompose_prior
from src.diffusion.sde_torch import eig_to_torch
from src.eval.probabilistic import crps_ensemble, pinball_loss, pit_values
from src.utils.config import load_config

_NUM_NODES = {"metrla": 207, "pemsbay": 325}
_NUM_HORIZONS = 12
_BACKBONE = "gwn"
_QUANTILES = (0.1, 0.5, 0.9)


# ─────────────────────────────────────────────────────────────────────────────
# Prior eigendecomp (mirrors train_diffusion.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def build_eig_structured(dataset: str, prior_yaml: dict) -> dict:
    delta_h = np.asarray(prior_yaml["delta_h"], dtype=np.float64)
    Q_H = build_Q_H(
        delta_h=delta_h,
        alpha=prior_yaml["alpha"],
        rho1=prior_yaml["rho1"],
        rho2=prior_yaml["rho2"],
    )
    eig_L = np.load(f"data/processed/{dataset}_laplacian_eigs.npz")
    Sigma_S = build_Sigma_S(
        prior_yaml["beta"], prior_yaml["epsilon_s"],
        eig_L["eigenvalues"].astype(np.float64),
        eig_L["eigenvectors"].astype(np.float64),
    )
    return eigendecompose_prior(Q_H, Sigma_S)


def build_eig_isotropic(num_sensors: int) -> dict:
    H, N = _NUM_HORIZONS, num_sensors
    U_T = np.eye(H, dtype=np.float64)
    U_S = np.eye(N, dtype=np.float64)
    c_T = np.ones(H, dtype=np.float64)
    c_S = np.ones(N, dtype=np.float64)
    s = np.ones((H, N), dtype=np.float64)
    q = np.ones((H, N), dtype=np.float64)
    return {"U_T": U_T, "c_T": c_T, "U_S": U_S, "c_S": c_S, "s": s, "q": q}


def build_adjacency_torch(dataset: str, device: torch.device) -> torch.Tensor:
    adj_mx = load_adjacency(dataset)
    L = compute_normalized_laplacian(adj_mx)
    A_norm = np.eye(L.shape[0]) - L
    return torch.as_tensor(A_norm, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Test-set dataset (returns x_history + y_pred standardised + y_true raw + mask)
# ─────────────────────────────────────────────────────────────────────────────

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: str, seed: int, target_scale: np.ndarray) -> None:
        scaler_path = f"data/processed/{dataset}_scaler.yaml"
        npz_path = f"data/processed/{dataset}_test.npz"
        self.traffic = TrafficDataset(npz_path, scaler_path)

        resid_path = f"results/residuals/{_BACKBONE}_{dataset}_test_seed{seed}.npz"
        if not Path(resid_path).exists():
            raise FileNotFoundError(f"{resid_path} not found.")
        resid = np.load(resid_path)
        y_pred_raw = resid["y_pred"].squeeze(-1).astype(np.float32)        # (N_win, H, N), mph
        y_true_raw = resid["y_true"].squeeze(-1).astype(np.float32)
        mask = resid["mask"].squeeze(-1).astype(bool)

        ts = target_scale.astype(np.float32).reshape(1, _NUM_HORIZONS, 1)
        # y_pred fed to the network in standardised units
        self.y_pred_std = torch.from_numpy(y_pred_raw / ts)
        # y_pred raw is needed for final de-standardisation
        self.y_pred_raw = torch.from_numpy(y_pred_raw)
        self.y_true_raw = torch.from_numpy(y_true_raw)
        self.mask = torch.from_numpy(mask)

        assert len(self.traffic) == self.y_pred_std.shape[0]

    def __len__(self) -> int:
        return len(self.traffic)

    def __getitem__(self, idx: int) -> dict:
        x_hn2 = self.traffic.x[idx].float()
        x_history = x_hn2.permute(2, 0, 1).contiguous()                    # (2, H, N)
        return {
            "x_history": x_history,
            "y_pred_std": self.y_pred_std[idx],
            "y_pred_raw": self.y_pred_raw[idx],
            "y_true_raw": self.y_true_raw[idx],
            "mask": self.mask[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_id", required=True,
                   help="Checkpoint folder name under results/checkpoints/.")
    p.add_argument("--n_samples", type=int, default=50,
                   help="K reverse-SDE samples per test window.")
    p.add_argument("--n_steps", type=int, default=200,
                   help="Euler-Maruyama steps in the reverse SDE.")
    p.add_argument("--tau_eps", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Number of windows processed in a single GPU batch. "
                        "Total trajectories per batch = batch_size * n_samples.")
    p.add_argument("--max_windows", type=int, default=None,
                   help="If set, only evaluate the first M test windows.")
    p.add_argument("--sample_seed", type=int, default=42,
                   help="Seed for the reverse-SDE noise.")
    p.add_argument("--out_dir", default="experiments/exp-003-score-network")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ── Load checkpoint + config ───────────────────────────────────────────
    ckpt_path = Path(f"results/checkpoints/{args.run_id}/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    dataset = cfg["dataset"]
    prior_kind = cfg["prior"]
    print(f"Run ID: {args.run_id}")
    print(f"Dataset: {dataset}  Prior: {prior_kind}  "
          f"Epoch saved: {ckpt['epoch']}  val_loss: {ckpt['val_loss']:.4f}")

    # ── Prior + eig ────────────────────────────────────────────────────────
    prior_yaml = load_config(f"results/prior_params/{_BACKBONE}_{dataset}.yaml")
    target_scale = np.asarray(prior_yaml["target_scale"], dtype=np.float32)
    if prior_kind == "structured":
        eig_np = build_eig_structured(dataset, prior_yaml)
    else:
        eig_np = build_eig_isotropic(_NUM_NODES[dataset])
    eig_t = eig_to_torch(eig_np, device=device, dtype=torch.float32)

    # ── Model + weights ────────────────────────────────────────────────────
    adj = build_adjacency_torch(dataset, device)
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
    print(f"Score net parameters: {model.num_parameters:,}")

    # ── Test set ───────────────────────────────────────────────────────────
    test_ds = TestDataset(dataset, seed=0, target_scale=target_scale)
    n_total = len(test_ds)
    if args.max_windows is not None:
        n_total = min(n_total, args.max_windows)
    print(f"Test windows to evaluate: {n_total:,}  "
          f"(K = {args.n_samples}, steps = {args.n_steps})")

    # ── Storage for accumulating metrics ───────────────────────────────────
    H, N = _NUM_HORIZONS, _NUM_NODES[dataset]
    crps_sum = np.zeros((H,), dtype=np.float64)
    crps_count = np.zeros((H,), dtype=np.int64)
    pit_bins = np.zeros((H, 20), dtype=np.int64)
    pin_sums = {q: np.zeros((H,), dtype=np.float64) for q in _QUANTILES}

    # MPS doesn't allow a CPU-side torch.Generator to drive randn() of MPS
    # tensors. Fall back to the global RNG seed in that case (still
    # deterministic if no other process draws in parallel).
    torch.manual_seed(args.sample_seed)
    if device.type == "mps":
        g = None
    else:
        g = torch.Generator(device=device.type).manual_seed(args.sample_seed)

    t0 = time.time()
    processed = 0

    # Use indexed batching so we can support --max_windows cleanly
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)
    for batch in loader:
        if processed >= n_total:
            break
        b = batch["x_history"].shape[0]
        take = min(b, n_total - processed)

        x_history = batch["x_history"][:take].to(device)
        y_pred_std = batch["y_pred_std"][:take].to(device)
        y_pred_raw = batch["y_pred_raw"][:take].numpy()                     # (b, H, N)
        y_true_raw = batch["y_true_raw"][:take].numpy()
        mask = batch["mask"][:take].numpy().astype(bool)

        # Reverse-SDE sampling: (take, K, H, N) in standardised residual units
        r_hat_0 = sample_per_window(
            model=model,
            x_history=x_history, y_pred=y_pred_std,
            n_samples=args.n_samples, n_steps=args.n_steps, eig=eig_t,
            beta_min=cfg["beta_min"], beta_max=cfg["beta_max"],
            tau_eps=args.tau_eps, generator=g,
        )
        # De-standardise and add backbone forecast → samples in mph
        r_hat_0_np = r_hat_0.cpu().numpy().astype(np.float32)               # (b, K, H, N)
        ts = target_scale.reshape(1, 1, H, 1)
        samples_mph = y_pred_raw[:, None, :, :] + r_hat_0_np * ts            # (b, K, H, N)

        # Per-horizon CRPS / PIT / pinball, masking invalid positions
        # samples_mph last axis must be (..., K) for our helpers → reorder
        samples_for_metrics = np.moveaxis(samples_mph, 1, -1)                # (b, H, N, K)

        for h in range(H):
            mh = mask[:, h, :]                                              # (b, N)
            y_h = y_true_raw[:, h, :][mh]                                   # (m,)
            s_h = samples_for_metrics[:, h, :, :][mh]                       # (m, K)
            if y_h.size == 0:
                continue
            crps_h = crps_ensemble(y_h, s_h)                                # (m,)
            crps_sum[h] += crps_h.sum()
            crps_count[h] += y_h.size

            # PIT into 20 bins
            pit_h = pit_values(y_h, s_h)                                    # (m,)
            counts, _ = np.histogram(pit_h, bins=20, range=(0.0, 1.0 + 1e-9))
            pit_bins[h] += counts

            for q in _QUANTILES:
                pin = pinball_loss(y_h, s_h, quantile=q)
                pin_sums[q][h] += pin.sum()

        processed += take
        if processed % (args.batch_size * 10) == 0 or processed >= n_total:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (n_total - processed) / rate if rate > 0 else 0
            print(f"  {processed:>6d}/{n_total} windows  "
                  f"elapsed={elapsed:6.0f}s  rate={rate:5.1f} win/s  eta={eta:6.0f}s")

    total_secs = time.time() - t0

    # ── Aggregate + write JSON ─────────────────────────────────────────────
    per_h_crps = (crps_sum / np.maximum(crps_count, 1)).tolist()
    per_h_pin = {f"q{int(q*100):02d}":
                 (pin_sums[q] / np.maximum(crps_count, 1)).tolist()
                 for q in _QUANTILES}
    avg_crps = float(np.mean(per_h_crps))

    out: dict = {
        "run_id": args.run_id,
        "dataset": dataset,
        "prior": prior_kind,
        "epoch_saved": int(ckpt["epoch"]),
        "val_loss": float(ckpt["val_loss"]),
        "eval": {
            "n_windows": int(processed),
            "n_samples": int(args.n_samples),
            "n_steps": int(args.n_steps),
            "tau_eps": float(args.tau_eps),
            "wall_seconds": float(total_secs),
        },
        "crps_per_horizon": per_h_crps,
        "crps_average": avg_crps,
        "pit_histogram_per_horizon": pit_bins.tolist(),
        "pinball_per_horizon": per_h_pin,
        "config": cfg,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{args.run_id}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    print(f"\nCRPS per horizon (h=1..12):")
    for h, c in enumerate(per_h_crps, start=1):
        marker = " ←" if h in (3, 6, 12) else ""
        print(f"  h={h:2d}  CRPS={c:6.3f}{marker}")
    print(f"\naverage CRPS over horizons: {avg_crps:.3f}")
    print(f"wall-clock: {total_secs:.0f}s")


if __name__ == "__main__":
    main()
