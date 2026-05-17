"""Train the score network for residual diffusion (score-network §14.4).

Wires together:
  - prior YAML (results/prior_params/gwn_{dataset}.yaml)
  - residual + y_pred + x_history (data/processed + results/residuals)
  - eigendecomposition of Σ_prior (NumPy → torch via eig_to_torch)
  - ScoreNet (with normalized adjacency)
  - DSM x₀-prediction loss (eigenbasis-weighted or uniform)
  - Adam + wandb + best.pt by val loss

Usage:
    # Full training
    python scripts/train_diffusion.py --dataset metrla --seed 0 --prior structured

    # Smoke (1 epoch, 5 train + 5 val batches, no checkpoints, wandb off)
    python scripts/train_diffusion.py --dataset metrla --seed 0 --prior structured --smoke

    # Isotropic ablation
    python scripts/train_diffusion.py --dataset metrla --seed 0 --prior isotropic

    # Resume from checkpoint
    python scripts/train_diffusion.py --dataset metrla --seed 0 --prior structured  # auto-resumes if best.pt exists for the same config hash
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.adjacency import compute_normalized_laplacian, load_adjacency
from src.data.dataset import TrafficDataset
from src.diffusion.dsm import dsm_x0_loss, sample_tau
from src.diffusion.prior_fit import build_Q_H, build_Sigma_S
from src.diffusion.sde import eigendecompose_prior
from src.diffusion.sde_torch import eig_to_torch, forward_sample
from src.diffusion.score_net import ScoreNet
from src.utils.config import hash_config, load_config
from src.utils.seed import set_seed
from src.utils.wandb_setup import init_wandb

_NUM_NODES = {"metrla": 207, "pemsbay": 325}
_NUM_HORIZONS = 12
_BACKBONE = "gwn"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: pairs (x_history, y_pred_backbone, r_0_standardised)
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionDataset(Dataset):
    """Pre-loads processed + residual npz into memory; serves training tuples.

    Returns per-window dict:
        x_history (2, H, N) — normalised speed + time-of-day (from TrafficDataset)
        y_pred    (H, N)    — backbone forecast / target_scale[h]   (standardised)
        r_0       (H, N)    — residual           / target_scale[h]  (standardised)
        mask      (H, N)    — validity bool from extract_residuals; masked
                              positions are excluded from the DSM loss.
    """

    def __init__(self, dataset: str, split: str, seed: int, target_scale: np.ndarray) -> None:
        # TrafficDataset gives us x (with speed already normalised) and y/mask.
        scaler_path = f"data/processed/{dataset}_scaler.yaml"
        npz_path = f"data/processed/{dataset}_{split}.npz"
        self.traffic = TrafficDataset(npz_path, scaler_path)

        # Residuals + y_pred + mask from the extracted .npz
        resid_path = f"results/residuals/{_BACKBONE}_{dataset}_{split}_seed{seed}.npz"
        if not Path(resid_path).exists():
            raise FileNotFoundError(
                f"{resid_path} not found. Run scripts/extract_residuals.py first."
            )
        resid = np.load(resid_path)
        # All shapes are (N_win, 12, N, 1) in raw mph.
        residuals = resid["residuals"].squeeze(-1)
        y_pred = resid["y_pred"].squeeze(-1)
        mask = resid["mask"].squeeze(-1).astype(bool)

        # Per-horizon standardisation
        ts = target_scale.astype(np.float32).reshape(1, _NUM_HORIZONS, 1)   # (1, H, 1)
        self.r_0 = torch.from_numpy((residuals / ts).astype(np.float32))   # (N_win, H, N)
        self.y_pred = torch.from_numpy((y_pred / ts).astype(np.float32))   # (N_win, H, N)
        # Stored as float32 so it can broadcast directly into the loss
        # multiplication on GPU without a dtype cast each step.
        self.mask = torch.from_numpy(mask.astype(np.float32))               # (N_win, H, N)

        if self.r_0.shape[0] != len(self.traffic):
            raise RuntimeError(
                f"window count mismatch: traffic={len(self.traffic)}, "
                f"residuals={self.r_0.shape[0]}"
            )

    def __len__(self) -> int:
        return len(self.traffic)

    def __getitem__(self, idx: int) -> dict:
        # TrafficDataset.x is (H, N, 2); ScoreNet wants (2, H, N).
        x_hn2 = self.traffic.x[idx].float()                                # (H, N, 2)
        x_history = x_hn2.permute(2, 0, 1).contiguous()                    # (2, H, N)
        return {
            "x_history": x_history,
            "y_pred": self.y_pred[idx],
            "r_0": self.r_0[idx],
            "mask": self.mask[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Prior eigendecomposition (structured vs isotropic)
# ─────────────────────────────────────────────────────────────────────────────

def build_eig_structured(dataset: str, prior_yaml: dict) -> dict:
    """Build the eigendecomp of the fitted Kronecker prior Σ_prior = Q_H ⊗ Σ_S."""
    if not prior_yaml.get("standardized"):
        raise ValueError(
            f"Prior YAML for {dataset} is not the v2 standardised version. "
            "Re-run scripts/fit_prior.py."
        )
    delta_h = np.asarray(prior_yaml["delta_h"], dtype=np.float64)
    Q_H = build_Q_H(
        delta_h=delta_h,
        alpha=prior_yaml["alpha"],
        rho1=prior_yaml["rho1"],
        rho2=prior_yaml["rho2"],
    )
    eig_path = f"data/processed/{dataset}_laplacian_eigs.npz"
    eig_L = np.load(eig_path)
    Sigma_S = build_Sigma_S(
        prior_yaml["beta"], prior_yaml["epsilon_s"],
        eig_L["eigenvalues"].astype(np.float64),
        eig_L["eigenvectors"].astype(np.float64),
    )
    return eigendecompose_prior(Q_H, Sigma_S)


def build_eig_isotropic(num_sensors: int) -> dict:
    """Trivial 'eigendecomp' for the isotropic ablation prior Σ = I_{HN}."""
    H, N = _NUM_HORIZONS, num_sensors
    U_T = np.eye(H, dtype=np.float64)
    U_S = np.eye(N, dtype=np.float64)
    c_T = np.ones(H, dtype=np.float64)
    c_S = np.ones(N, dtype=np.float64)
    s = np.ones((H, N), dtype=np.float64)
    q = np.ones((H, N), dtype=np.float64)
    return {"U_T": U_T, "c_T": c_T, "U_S": U_S, "c_S": c_S, "s": s, "q": q}


# ─────────────────────────────────────────────────────────────────────────────
# Normalised adjacency for ScoreNet's graph conv
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency_torch(dataset: str, device: torch.device) -> torch.Tensor:
    """A_norm = I - L  where L is the symmetric normalised Laplacian.

    Eigenvalues of L are in [0, 2], so A_norm's eigenvalues are in [-1, 1].
    """
    adj_mx = load_adjacency(dataset)                                        # (N, N) float32
    L = compute_normalized_laplacian(adj_mx)                                # (N, N) float64
    A_norm = np.eye(L.shape[0]) - L
    return torch.as_tensor(A_norm, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Train / val loops
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model, loader, opt, eig_t, beta_min, beta_max, tau_eps, weighting, device,
    max_batches: int | None = None,
) -> tuple[float, int]:
    model.train()
    loss_sum = 0.0
    n = 0
    for batch in loader:
        if max_batches is not None and n >= max_batches:
            break
        r_0 = batch["r_0"].to(device)
        x_history = batch["x_history"].to(device)
        y_pred = batch["y_pred"].to(device)
        mask = batch["mask"].to(device)
        B = r_0.shape[0]

        tau = sample_tau(B, tau_eps=tau_eps, device=device, dtype=r_0.dtype)
        r_tau = forward_sample(r_0, tau, eig_t, beta_min, beta_max)
        r_hat = model(r_tau, tau, x_history, y_pred)
        loss = dsm_x0_loss(
            r_hat, r_0, tau, eig_t, beta_min, beta_max,
            weighting=weighting, mask=mask,
        )

        opt.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        loss_sum += loss.item()
        n += 1
    return loss_sum / max(n, 1), n


@torch.no_grad()
def val_epoch(
    model, loader, eig_t, beta_min, beta_max, tau_eps, weighting, device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    # Deterministic val: fixed τ-sampling RNG and fixed forward-noise RNG.
    # τ uses a CPU generator (sample_tau materialises on CPU then we .to()).
    # Forward noise uses a device-side generator so randn lands on the right
    # device. MPS doesn't accept a non-MPS generator for MPS randn; fall back
    # to torch.manual_seed (same workaround as eval_diffusion.py).
    g_tau = torch.Generator(device="cpu").manual_seed(0)
    if device.type == "mps":
        torch.manual_seed(0)
        g_noise = None
    else:
        g_noise = torch.Generator(device=device.type).manual_seed(0)
    loss_sum = 0.0
    n = 0
    for batch in loader:
        if max_batches is not None and n >= max_batches:
            break
        r_0 = batch["r_0"].to(device)
        x_history = batch["x_history"].to(device)
        y_pred = batch["y_pred"].to(device)
        mask = batch["mask"].to(device)
        B = r_0.shape[0]

        tau_cpu = sample_tau(B, tau_eps=tau_eps, device="cpu", dtype=torch.float32, generator=g_tau)
        tau = tau_cpu.to(device)
        r_tau = forward_sample(r_0, tau, eig_t, beta_min, beta_max, generator=g_noise)
        r_hat = model(r_tau, tau, x_history, y_pred)
        loss = dsm_x0_loss(
            r_hat, r_0, tau, eig_t, beta_min, beta_max,
            weighting=weighting, mask=mask,
        )
        loss_sum += loss.item()
        n += 1
    return loss_sum / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=["metrla", "pemsbay"], required=True)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for residuals (must match the one used by extract_residuals).")
    p.add_argument("--prior", choices=["structured", "isotropic"], default="structured")
    p.add_argument("--weighting", choices=["eigenbasis", "uniform"], default="eigenbasis")

    # Architecture
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--tau_embed_dim", type=int, default=128)

    # Diffusion schedule (locked from sde-sanity)
    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--beta_max", type=float, default=50.0)
    p.add_argument("--tau_eps", type=float, default=1e-3)

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stopping patience in epochs.")
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 5 train + 5 val batches, no checkpoints, wandb off.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.smoke:
        args.epochs = 1
        args.batch_size = 8
        os.environ["WANDB_MODE"] = "disabled"
        smoke_train_batches = 5
        smoke_val_batches = 5
    else:
        smoke_train_batches = None
        smoke_val_batches = None

    # ────────────────────────────────────────────────────────────────────────
    # Config + run id + checkpoint dir
    # ────────────────────────────────────────────────────────────────────────
    cfg = {
        "model": "score_net",
        "prior": args.prior,
        "weighting": args.weighting,
        "dataset": args.dataset,
        "backbone": _BACKBONE,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "tau_embed_dim": args.tau_embed_dim,
        "beta_min": args.beta_min,
        "beta_max": args.beta_max,
        "tau_eps": args.tau_eps,
        # Hash on platform so the same code+config produces different
        # run_ids on Mac (Darwin) vs DAIC (Linux). Lets us launch the
        # same training on both without wandb / checkpoint-dir collisions.
        "platform": platform.system().lower(),
    }
    cfg_hash = hash_config(cfg)
    run_id = f"dsm_{args.prior}_{args.dataset}_seed{args.seed}_{cfg_hash}"
    ckpt_dir = Path(f"results/checkpoints/{run_id}")
    if not args.smoke:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────────
    # Device
    # ────────────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    pin_memory = device.type == "cuda"
    print(f"Device: {device}")
    print(f"Run ID: {run_id}")
    if not args.smoke:
        print(f"Checkpoint dir: {ckpt_dir}")

    # ────────────────────────────────────────────────────────────────────────
    # Prior + eigendecomp
    # ────────────────────────────────────────────────────────────────────────
    prior_yaml_path = f"results/prior_params/{_BACKBONE}_{args.dataset}.yaml"
    prior_yaml = load_config(prior_yaml_path)
    target_scale = np.asarray(prior_yaml["target_scale"], dtype=np.float32)
    print(f"Prior: {args.prior}; target_scale per horizon: {target_scale}")

    if args.prior == "structured":
        eig_np = build_eig_structured(args.dataset, prior_yaml)
    else:
        eig_np = build_eig_isotropic(num_sensors=_NUM_NODES[args.dataset])
    eig_t = eig_to_torch(eig_np, device=device, dtype=torch.float32)
    print(f"Prior eigvals s: min={eig_np['s'].min():.3e}  median={np.median(eig_np['s']):.3e}  "
          f"max={eig_np['s'].max():.3e}")

    # ────────────────────────────────────────────────────────────────────────
    # Datasets + loaders
    # ────────────────────────────────────────────────────────────────────────
    train_ds = DiffusionDataset(args.dataset, "train", args.seed, target_scale)
    val_ds = DiffusionDataset(args.dataset, "val", args.seed, target_scale)
    print(f"Train windows: {len(train_ds):,}   Val windows: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    # ────────────────────────────────────────────────────────────────────────
    # Model + optimiser
    # ────────────────────────────────────────────────────────────────────────
    adj = build_adjacency_torch(args.dataset, device)
    model = ScoreNet(
        num_horizons=_NUM_HORIZONS,
        num_sensors=_NUM_NODES[args.dataset],
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        tau_embed_dim=args.tau_embed_dim,
        adj=adj,
    ).to(device)
    print(f"Score net parameters: {model.num_parameters:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ────────────────────────────────────────────────────────────────────────
    # wandb
    # ────────────────────────────────────────────────────────────────────────
    init_wandb(cfg, run_id)
    if not args.smoke:
        import wandb
        wandb.config.update({"num_parameters": model.num_parameters,
                             "num_train_windows": len(train_ds),
                             "num_val_windows": len(val_ds)}, allow_val_change=True)

    # ────────────────────────────────────────────────────────────────────────
    # Training loop
    # ────────────────────────────────────────────────────────────────────────
    best_val = float("inf")
    patience = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss, n_train = train_epoch(
            model, train_loader, opt, eig_t,
            args.beta_min, args.beta_max, args.tau_eps, args.weighting, device,
            max_batches=smoke_train_batches,
        )
        val_loss = val_epoch(
            model, val_loader, eig_t,
            args.beta_min, args.beta_max, args.tau_eps, args.weighting, device,
            max_batches=smoke_val_batches,
        )
        dt = time.time() - ep_start
        print(f"epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"({dt:.1f}s, {n_train} batches)")

        if not args.smoke:
            import wandb
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "epoch": epoch,
                "epoch_seconds": dt,
            }, step=epoch)

        # Save best by val loss
        if not args.smoke and val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": cfg,
            }, ckpt_dir / "best.pt")
            patience = 0
        else:
            patience += 1

        # Save last every epoch
        if not args.smoke:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": cfg,
            }, ckpt_dir / "last.pt")

        if patience >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    total = time.time() - t0
    print(f"\nDone in {total/60:.1f} min.   best val loss: {best_val:.4f}")
    if not args.smoke:
        import wandb
        wandb.log({"best_val_loss": best_val, "total_minutes": total / 60.0})
        wandb.finish()


if __name__ == "__main__":
    main()
