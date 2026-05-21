"""Extract GWN residuals in raw units. Implements Section 1.9 + §1.10 of the spec.

For each window s in S_split, with frozen θ*:
    Yhat_z^(s) = f_{θ*}(X^(s))               (z-space network output, shape H×N)
    Yhat^(s)   = σ * Yhat_z^(s) + μ          (raw-unit prediction, shape H×N)
    r^(s)      = Y^(s) - Yhat^(s)            (raw-unit residual,   shape H×N)
    R^(s)      = (r^(s))^T                   (reordered,           shape N×H)

Stacked across the split → R ∈ R^{B × N × H} (Kronecker convention
Cov(vec(R)) = Σ_T ⊗ Σ_S, with vec column-major over the leading N axis).

The training-split residuals R_tr are the canonical artifact for the diffusion
prior. Val and test residuals are extracted with the same θ* so we can run the
§1.10 in-sample diagnostic and (later) report final-forecast metrics.

§1.10 diagnostic. Per-horizon residual variance:
    δ_h^(tr) = masked mean of (R_tr[:, :, h])^2
    δ_h^(vl) = masked mean of (R_vl[:, :, h])^2
Pass criterion: |δ_h^(vl)/δ_h^(tr) - 1| < 0.10 ∀ h.

Usage:
    python scripts/extract_residuals.py --dataset metrla --seed 0
    python scripts/extract_residuals.py --dataset pemsbay --seed 0
    python scripts/extract_residuals.py --dataset pems08f --seed 0

Outputs:
    results/residuals/gwn_{dataset}_{split}_seed{seed}.npz   (split=train|val|test)
        residuals  float32  (B, N, H)   r^(s) = Y - Yhat, raw units, reordered N×H
        y_pred     float32  (B, N, H)   Yhat in raw units (for Y_final composition)
        y_true     float32  (B, N, H)   Y    in raw units
        mask       bool     (B, N, H)   True = valid (METR-LA only; ones elsewhere)
        mean       float32  scalar      μ used for backbone input z-score (provenance)
        std        float32  scalar      σ used for backbone input z-score (provenance)

    results/residuals/gwn_{dataset}_seed{seed}_diagnostic.json
        Section 1.10 train-vs-val per-horizon variance ratios + pass/fail flag.
"""
import argparse
import json
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


def _to_BNH(arr: np.ndarray) -> np.ndarray:
    """Drop GWN's trailing channel dim and reorder (B, H, N, 1) → (B, N, H).

    The reorder matches the §1.9 spec: R^(s) = (r^(s))^T so that
    vec(R) is column-major over the leading N axis and Σ = Σ_T ⊗ Σ_S.
    """
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]              # (B, H, N)
    return np.transpose(arr, (0, 2, 1))  # (B, N, H)


def extract_one_split(
    model: torch.nn.Module,
    loader: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on one split and return (residuals, y_pred, y_true, mask)
    in raw units, reordered to (B, N, H) per §1.9.

    Args:
        model: Trained backbone wrapper (eval mode expected).
        loader: DataLoader for the split.
        mean, std: Scaler values for backbone inverse-transform (Yhat = σ Yhat_z + μ).
        device: Compute device.
        split_name: "train" | "val" | "test" — used only for the progress bar.

    Returns:
        residuals  float32  (B, N, H)   r = Y - Yhat in raw units
        y_pred     float32  (B, N, H)   Yhat in raw units (post inverse-transform)
        y_true     float32  (B, N, H)   Y    in raw units (as loaded from npz)
        mask       bool     (B, N, H)   True = valid reading
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

            pred_norm = model(x)                 # (B, H, N, 1) z-space
            pred_raw = pred_norm * std + mean    # raw units; matches §1.7 inverse

            preds_list.append(pred_raw.cpu().numpy().astype(np.float32))
            y_list.append(y_raw.cpu().numpy().astype(np.float32))
            mask_list.append(mask.cpu().numpy().astype(bool))

    y_pred = _to_BNH(np.concatenate(preds_list, axis=0))      # (B, N, H)
    y_true = _to_BNH(np.concatenate(y_list, axis=0))           # (B, N, H)
    mask = _to_BNH(np.concatenate(mask_list, axis=0))          # (B, N, H) bool
    residuals = (y_true - y_pred).astype(np.float32)            # raw units

    return residuals, y_pred, y_true, mask


def _masked_per_horizon_variance(
    residuals: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    """δ_h = Σ_s,i M_h,i^(s) (R_i,h^(s))^2 / Σ_s,i M_h,i^(s), per spec §1.10.

    Args:
        residuals: (B, N, H) float32, raw units.
        mask:      (B, N, H) bool.

    Returns:
        (H,) float64 — per-horizon masked second moment about zero.
    """
    H = residuals.shape[-1]
    out = np.empty(H, dtype=np.float64)
    sq = residuals.astype(np.float64) ** 2
    for h in range(H):
        m = mask[..., h]
        denom = m.sum()
        out[h] = float((sq[..., h] * m).sum() / denom) if denom > 0 else np.nan
    return out


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
    # Extract each split; cache per-horizon variances for the §1.10 diagnostic.
    # ------------------------------------------------------------------ #
    delta_per_split: dict[str, np.ndarray] = {}

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
            mean=np.float32(mean),
            std=np.float32(std),
        )
        print(f"  Saved {out_path}  shape={residuals.shape}  (B, N, H)")

        # Sanity prints (mask-aware, raw units).
        valid_residuals = residuals[mask]
        per_h_var = _masked_per_horizon_variance(residuals, mask)
        per_h_std = np.sqrt(per_h_var)
        delta_per_split[split] = per_h_var

        print(f"  Global mean residual:    {valid_residuals.mean():+.4f}   (target ≈ 0)")
        print(f"  Per-horizon std (h1→h12):  "
              + " ".join(f"{s:.3f}" for s in per_h_std))
        print(f"  Missing fraction:        {1.0 - mask.mean():.4f}")

    # ------------------------------------------------------------------ #
    # §1.10 in-sample-residuals diagnostic
    # ------------------------------------------------------------------ #
    print("\n--- §1.10 train-vs-val per-horizon variance ---")
    delta_tr = delta_per_split["train"]
    delta_vl = delta_per_split["val"]
    ratios = delta_vl / delta_tr - 1.0
    pass_per_h = np.abs(ratios) < 0.10
    overall_pass = bool(pass_per_h.all())

    print(f"  {'h':>3}  {'δ_tr':>10}  {'δ_vl':>10}  {'vl/tr - 1':>10}  ok?")
    for h in range(len(delta_tr)):
        flag = " ✓" if pass_per_h[h] else " ✗"
        print(f"  {h+1:>3}  {delta_tr[h]:>10.4f}  {delta_vl[h]:>10.4f}  "
              f"{ratios[h]:>+10.3%}  {flag}")
    print(f"  pass criterion |vl/tr - 1| < 0.10 ∀ h: "
          f"{'PASS' if overall_pass else 'FAIL'}")

    diag_path = out_dir / f"{_BACKBONE}_{ds_name}_seed{args.seed}_diagnostic.json"
    diag_path.write_text(json.dumps({
        "dataset": ds_name,
        "seed": args.seed,
        "criterion": "|delta_h^(vl) / delta_h^(tr) - 1| < 0.10 for all h",
        "delta_train":     delta_tr.tolist(),
        "delta_val":       delta_vl.tolist(),
        "delta_test":      delta_per_split["test"].tolist(),
        "ratio_vl_over_tr_minus_one": ratios.tolist(),
        "pass_per_horizon":  pass_per_h.tolist(),
        "pass_overall":      overall_pass,
    }, indent=2))
    print(f"  Saved {diag_path}")

    print("\nDone. Residuals are in raw units, shape (B, N, H). Ready for H1–H5.")


if __name__ == "__main__":
    main()
