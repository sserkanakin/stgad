"""Extract GWN residuals (y_true_raw − y_pred_raw) and save to results/residuals/.

Loads the best GWN checkpoint, runs inference on train/val/test, inverse-transforms
predictions to raw mph, computes residuals, and saves one .npz per split.

Usage:
    python scripts/extract_residuals.py --dataset metrla --seed 0
    python scripts/extract_residuals.py --dataset pemsbay --seed 0

Outputs (one per split):
    results/residuals/gwn_{dataset}_{split}_seed{seed}.npz
        residuals  float32  (N_windows, 12, N_sensors, 1)   y_true − y_pred in mph
        y_pred     float32  (N_windows, 12, N_sensors, 1)   backbone forecast in mph
        y_true     float32  (N_windows, 12, N_sensors, 1)   ground truth in mph
        mask       bool     (N_windows, 12, N_sensors, 1)   True = valid reading
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones import load_backbone

_BACKBONE = "gwn"
from src.data.adjacency import load_adjacency
from src.data.dataset import TrafficDataset
from src.utils.checkpoint import load_checkpoint
from src.utils.config import hash_config, load_config
from src.utils.seed import set_seed


def extract_one_split(
    model: torch.nn.Module,
    loader: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on one split and return (residuals, y_pred, y_true, mask).

    Args:
        model: Trained backbone wrapper (eval mode expected).
        loader: DataLoader for the split.
        mean, std: Scaler values for inverse-transform.
        device: Compute device.
        split_name: e.g. "train", "val", "test" — used only for progress bar.

    Returns:
        Tuple of four float32/bool numpy arrays, each (N_windows, 12, N, 1).
    """
    model.eval()

    preds_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc=f"  {split_name}"):
            x = batch["x"].to(device)
            y_raw = batch["y"].to(device)
            mask = batch["y_mask"].to(device)
            y_norm = (y_raw - mean) / std

            pred_norm = model(x)
            pred_raw = pred_norm * std + mean    # inverse transform to mph

            preds_list.append(pred_raw.cpu().numpy().astype(np.float32))
            y_list.append(y_raw.cpu().numpy().astype(np.float32))
            mask_list.append(mask.cpu().numpy())

    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    residuals = (y_true - y_pred).astype(np.float32)

    return residuals, y_pred, y_true, mask


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Extract GWN residuals.")
    parser.add_argument("--dataset", choices=["metrla", "pemsbay", "pems08f"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Inference batch size (larger = faster).")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Config + checkpoint path
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
    cfg_hash = hash_config({k: v for k, v in cfg.items() if k not in ("seed",)})
    run_id = f"{_BACKBONE}_{args.dataset}_seed{args.seed}_{cfg_hash}"
    ckpt_dir = Path(f"results/checkpoints/{run_id}")
    best_ckpt = ckpt_dir / "best.pt"

    if not best_ckpt.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {best_ckpt}\n"
            f"Run training first: python scripts/train_backbone.py "
            f"--dataset {args.dataset} --seed {args.seed}"
        )

    # ------------------------------------------------------------------ #
    # Device + model
    # ------------------------------------------------------------------ #
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    set_seed(args.seed)

    adj_mx = load_adjacency(args.dataset)
    model = load_backbone(_BACKBONE, args.dataset, adj_mx, device)
    eff_device = getattr(model, "effective_device", device)

    ckpt = load_checkpoint(best_ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']} "
          f"(val_h12={ckpt['best_val_mae']:.4f})")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    ds_name = args.dataset
    scaler_path = f"data/processed/{ds_name}_scaler.yaml"
    train_ds = TrafficDataset(f"data/processed/{ds_name}_train.npz", scaler_path)
    mean = train_ds.mean
    std = train_ds.std

    out_dir = Path("results/residuals")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Extract each split
    # ------------------------------------------------------------------ #
    for split in ("train", "val", "test"):
        print(f"\nExtracting {split} split ...")
        ds = TrafficDataset(f"data/processed/{ds_name}_{split}.npz", scaler_path)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        residuals, y_pred, y_true, mask = extract_one_split(
            model, loader, mean, std, eff_device, split
        )

        out_path = out_dir / f"{_BACKBONE}_{ds_name}_{split}_seed{args.seed}.npz"
        np.savez_compressed(
            out_path,
            residuals=residuals,
            y_pred=y_pred,
            y_true=y_true,
            mask=mask,
        )
        print(f"  Saved {out_path}  shape={residuals.shape}")

        # Sanity prints
        valid = mask.astype(bool)
        r_valid = residuals[valid]
        per_h_std = [
            float(residuals[:, h, :, 0][mask[:, h, :, 0].astype(bool)].std())
            for h in range(12)
        ]
        print(f"  Global mean residual: {r_valid.mean():.4f} mph  (target ≈ 0)")
        print(f"  Per-horizon std (h1→h12): "
              + " ".join(f"{s:.3f}" for s in per_h_std))
        missing_frac = 1.0 - valid.mean()
        print(f"  Missing fraction: {missing_frac:.4f}")

    print("\nDone. Proceed to notebooks/02_residual_analysis.ipynb to inspect residuals.")


if __name__ == "__main__":
    main()
