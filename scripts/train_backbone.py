"""Train the Graph WaveNet backbone and save the best checkpoint.

Full training is run by the user on their Mac (see RUNBOOK.md).
Claude Code only executes smoke-test mode (--smoke) as a wiring check.

Usage:
    # Full training (user runs this):
    python scripts/train_backbone.py --dataset metrla --seed 0

    # Smoke test (1 epoch, batch_size=16, wandb disabled):
    python scripts/train_backbone.py --dataset metrla --seed 0 --smoke

    # Force fresh start (deletes checkpoint and resets wandb run):
    python scripts/train_backbone.py --dataset metrla --seed 0 --fresh
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones import load_backbone
from src.data.adjacency import load_adjacency
from src.data.dataset import TrafficDataset
from src.utils.checkpoint import (
    find_latest_checkpoint,
    get_rng_states,
    load_checkpoint,
    restore_rng_states,
    save_checkpoint,
)
from src.utils.config import hash_config, load_config
from src.utils.seed import set_seed
from src.utils.wandb_setup import init_wandb

# ---------------------------------------------------------------------------
# Target MAE values for post-training sanity check (±5%)
# ---------------------------------------------------------------------------
_MAE_TARGETS = {
    "metrla":  {3: 2.69, 6: 3.07, 12: 3.53},
    "pemsbay": {3: 1.30, 6: 1.63, 12: 1.95},
    # PEMS08F (flow, vehicles/5min) — targets to be populated after first
    # baseline run on DAIC; sanity warning is suppressed until then.
    "pems08f": {},
}

_BACKBONE = "gwn"


# ---------------------------------------------------------------------------
# Loss / metric helpers
# ---------------------------------------------------------------------------

def masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        pred:   (B, T, N, 1) or any broadcastable shape
        target: same shape as pred
        mask:   bool tensor, same shape — True = valid reading

    Returns:
        Scalar loss tensor.
    """
    err = torch.abs(pred - target) * mask.float()
    return err.sum() / (mask.float().sum() + 1e-8)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mean: float,
    std: float,
    device: torch.device,
    batches_seen: int,
    max_batches: int | None = None,
) -> tuple[float, int]:
    """Run one training epoch.

    Args:
        model: Backbone wrapper on ``device``.
        loader: Training DataLoader.
        optimizer: Adam optimiser.
        mean, std: Scaler values from TrafficDataset.
        device: Compute device.
        batches_seen: Running batch counter (for wandb step alignment).
        max_batches: If set, stop after this many batches (for smoke mode).

    Returns:
        (epoch_mean_loss, updated_batches_seen)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, leave=False, desc="  train", total=max_batches or len(loader)):
        x = batch["x"].to(device)          # (B, 12, N, 2)  normalised speed+tod
        y_raw = batch["y"].to(device)       # (B, 12, N, 1)  raw mph
        mask = batch["y_mask"].to(device)   # (B, 12, N, 1)  bool

        # Normalise y internally to match what the model predicts
        y_norm = (y_raw - mean) / std

        pred_norm = model(x)                # (B, 12, N, 1)
        loss = masked_mae(pred_norm, y_norm, mask)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        batches_seen += 1
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    return total_loss / max(n_batches, 1), batches_seen


@torch.no_grad()
def eval_split(
    model: nn.Module,
    loader: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
) -> dict[int, float]:
    """Evaluate MAE at horizons h=3,6,12 (1-indexed).

    Returns:
        Dict mapping horizon index to masked MAE in raw mph.
    """
    model.eval()

    all_pred: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    all_mask: list[torch.Tensor] = []

    for batch in tqdm(loader, leave=False, desc="  eval "):
        x = batch["x"].to(device)
        y_raw = batch["y"].to(device)
        mask = batch["y_mask"].to(device)
        y_norm = (y_raw - mean) / std

        pred_norm = model(x)
        pred_raw = pred_norm * std + mean   # inverse transform

        all_pred.append(pred_raw.cpu())
        all_y.append(y_raw.cpu())
        all_mask.append(mask.cpu())

    preds = torch.cat(all_pred, dim=0)   # (N_windows, 12, N, 1)
    targets = torch.cat(all_y, dim=0)
    masks = torch.cat(all_mask, dim=0)

    mae_per_h: dict[int, float] = {}
    for h in [3, 6, 12]:
        idx = h - 1
        mae_per_h[h] = masked_mae(
            preds[:, idx], targets[:, idx], masks[:, idx]
        ).item()

    return mae_per_h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    """Entry point."""
    parser = argparse.ArgumentParser(description="Train the Graph WaveNet backbone.")
    parser.add_argument("--dataset", choices=["metrla", "pemsbay", "pems08f"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count from backbone config.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch_size from backbone config.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke-test: 1 epoch, batch_size=16, max 50 batches, wandb disabled.")
    parser.add_argument("--smoke_batches", type=int, default=50,
                        help="Max training batches in smoke mode (default 50).")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start from scratch.")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    backbone_cfg = load_config(f"configs/backbone/{_BACKBONE}.yaml")
    data_cfg = load_config(f"configs/data/{args.dataset}.yaml")

    cfg = {
        "backbone": _BACKBONE,
        "dataset": args.dataset,
        "seed": args.seed,
        **backbone_cfg,
        **data_cfg,
    }
    if args.smoke:
        cfg["epochs"] = 1
        cfg["batch_size"] = 16
        os.environ["WANDB_MODE"] = "disabled"
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    epochs = int(cfg.get("epochs", 100))
    batch_size = int(cfg.get("batch_size", 64))
    smoke_batches = args.smoke_batches if args.smoke else None
    lr = float(cfg.get("lr", 1e-3))

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ------------------------------------------------------------------ #
    # Checkpoint directory and run ID
    # ------------------------------------------------------------------ #
    cfg_hash = hash_config({k: v for k, v in cfg.items() if k not in ("seed",)})
    run_id = f"{_BACKBONE}_{args.dataset}_seed{args.seed}_{cfg_hash}"
    ckpt_dir = Path(f"results/checkpoints/{run_id}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh and (ckpt_dir / "last.pt").exists():
        (ckpt_dir / "last.pt").unlink()
        if (ckpt_dir / "best.pt").exists():
            (ckpt_dir / "best.pt").unlink()
        if (ckpt_dir / "state.json").exists():
            (ckpt_dir / "state.json").unlink()
        print(f"[fresh] Deleted checkpoint at {ckpt_dir}")

    # ------------------------------------------------------------------ #
    # Seed
    # ------------------------------------------------------------------ #
    set_seed(args.seed)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    ds_name = args.dataset
    train_ds = TrafficDataset(
        f"data/processed/{ds_name}_train.npz",
        f"data/processed/{ds_name}_scaler.yaml",
    )
    val_ds = TrafficDataset(
        f"data/processed/{ds_name}_val.npz",
        f"data/processed/{ds_name}_scaler.yaml",
    )

    mean = train_ds.mean
    std = train_ds.std

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=pin, drop_last=False,
    )

    # ------------------------------------------------------------------ #
    # Model + optimiser
    # ------------------------------------------------------------------ #
    adj_mx = load_adjacency(ds_name)
    model = load_backbone(_BACKBONE, ds_name, adj_mx, device)
    eff_device = getattr(model, "effective_device", device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone: {_BACKBONE}/{ds_name} | params={n_params:,} | device={eff_device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------ #
    # Resume from checkpoint
    # ------------------------------------------------------------------ #
    start_epoch = 0
    best_val_mae = float("inf")
    batches_seen = 0

    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt is not None and not args.fresh:
        print(f"Resuming from {latest_ckpt} ...")
        ckpt = load_checkpoint(latest_ckpt)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mae = ckpt["best_val_mae"]
        restore_rng_states(ckpt["rng_states"])
        print(f"  Resumed from epoch {ckpt['epoch']} | best_val_mae={best_val_mae:.4f}")
    else:
        print(f"Starting fresh | epochs={epochs} | batch_size={batch_size} | lr={lr}")

    # ------------------------------------------------------------------ #
    # wandb
    # ------------------------------------------------------------------ #
    if not args.smoke:
        run = init_wandb(cfg, run_id)
        if latest_ckpt is not None and not args.fresh:
            import wandb
            wandb.log({"resumed_from_epoch": start_epoch - 1}, step=0)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        ep_t0 = time.time()

        train_loss, batches_seen = train_one_epoch(
            model, train_loader, optimizer, mean, std, eff_device, batches_seen,
            max_batches=smoke_batches,
        )
        val_mae = eval_split(model, val_loader, mean, std, eff_device)

        val_avg = sum(val_mae.values()) / len(val_mae)
        ep_time = time.time() - ep_t0

        print(
            f"epoch {epoch:3d}/{epochs-1}  "
            f"train_loss={train_loss:.4f}  "
            f"val_h3={val_mae[3]:.3f}  "
            f"val_h6={val_mae[6]:.3f}  "
            f"val_h12={val_mae[12]:.3f}  "
            f"t={ep_time:.1f}s"
        )

        # Checkpoint: always save last.pt
        rng_states = get_rng_states()
        save_checkpoint(
            ckpt_dir / "last.pt",
            model, optimizer, sched=None,
            epoch=epoch, best_val_mae=best_val_mae,
            rng_states=rng_states,
        )

        # Save best.pt when val MAE improves
        improved = val_mae[12] < best_val_mae
        if improved:
            best_val_mae = val_mae[12]
            save_checkpoint(
                ckpt_dir / "best.pt",
                model, optimizer, sched=None,
                epoch=epoch, best_val_mae=best_val_mae,
                rng_states=rng_states,
            )

        (ckpt_dir / "state.json").write_text(json.dumps({
            "last_epoch": epoch,
            "best_val_mae": best_val_mae,
            "run_id": run_id,
            "cfg_hash": cfg_hash,
        }, indent=2))

        if not args.smoke:
            import wandb
            wandb.log({
                "train/loss": train_loss,
                "val/mae_h3": val_mae[3],
                "val/mae_h6": val_mae[6],
                "val/mae_h12": val_mae[12],
                "val/mae_avg": val_avg,
                "lr": lr,
                "epoch": epoch,
            }, step=epoch)

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time/60:.1f} min | best val_h12={best_val_mae:.4f}")

    # ------------------------------------------------------------------ #
    # Post-training test-set evaluation (skipped in smoke mode)
    # ------------------------------------------------------------------ #
    if not args.smoke:
        test_ds = TrafficDataset(
            f"data/processed/{ds_name}_test.npz",
            f"data/processed/{ds_name}_scaler.yaml",
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=pin,
        )

        # Load best checkpoint for test evaluation
        best_ckpt = ckpt_dir / "best.pt"
        if best_ckpt.exists():
            ckpt = load_checkpoint(best_ckpt)
            model.load_state_dict(ckpt["model_state_dict"])

        test_mae = eval_split(model, test_loader, mean, std, eff_device)
        print("\n--- Test MAE ---")
        targets = _MAE_TARGETS.get(ds_name, {})
        if not targets:
            print(f"  (no MAE targets defined for '{ds_name}' yet — "
                  f"sanity warnings suppressed)")
        for h, mae in sorted(test_mae.items()):
            label = f"h{h} ({h*5}min)"
            target = targets.get(h, None)
            flag = ""
            if target is not None:
                pct_diff = abs(mae - target) / target * 100
                flag = f"  target={target:.2f}  diff={pct_diff:.1f}%"
                if pct_diff > 5:
                    flag += "  *** WARN: >5% from target ***"
            print(f"  {label}: {mae:.4f}{flag}")

        if not args.smoke:
            import wandb
            wandb.log({f"test/mae_h{h}": v for h, v in test_mae.items()})
            wandb.finish()

    elif args.smoke:
        print(f"\n[smoke] Wallclock: {total_time:.1f}s")


if __name__ == "__main__":
    main()
