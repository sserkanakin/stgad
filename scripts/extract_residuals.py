"""extract the backbone residuals r = Y - Yhat for train/val/test.

for each window we run the frozen backbone, undo the z-score to get Yhat in raw
units, and take r = Y - Yhat reordered to (N, H) (so vec(R) is column-major over
N and Cov(vec(R)) = Σ_T ⊗ Σ_S). stacked per split this is (B, N, H).

we also run a quick train-vs-val check on the per-horizon residual variance
δ_h = masked mean of R[:, :, h]^2; it passes if |δ_h^val / δ_h^train - 1| < 0.10
for every horizon h.

usage:
    python scripts/extract_residuals.py --dataset metrla --seed 0

outputs (under results/residuals/<dataset>/seed<seed>/, see src.utils.paths):
    {split}.npz       residuals, y_pred, mask (all (B, N, H)), mean, std
                      (y_true is not stored - it's exactly y_pred + residuals)
    diagnostic.json   the train-vs-val per-horizon variance check
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones import load_backbone
from src.data.adjacency import load_adjacency
from src.data.dataset import TrafficDataset
from src.utils import paths
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import set_seed

_BACKBONE = "gwn"


def _to_BNH(arr: np.ndarray) -> np.ndarray:
    """drop gwn's trailing channel dim and reorder (B, H, N, 1) -> (B, N, H).

    putting N first makes vec(R) column-major over N, so Cov(vec(R)) = Σ_T ⊗ Σ_S.
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
    """run the backbone on one split and return (residuals, y_pred, y_true, mask),
    all (B, N, H) in raw units. mean/std undo the input z-score (Yhat = σ Yhat_z + μ)."""
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
            pred_raw = pred_norm * std + mean    # undo the z-score

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
    """per-horizon masked second moment δ_h = masked mean of R[:, :, h]^2.

    residuals and mask are (B, N, H); returns (H,) float64.
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
    # Checkpoint path
    # ------------------------------------------------------------------ #
    best_ckpt = paths.backbone_dir(args.dataset, args.seed) / "best.pt"

    # preflight: fail fast with a clear message if an input is missing.
    paths.require_inputs([
        (best_ckpt, f"run scripts/train_backbone.py --dataset {args.dataset} --seed {args.seed} first"),
        (paths.split_npz(args.dataset, "train"), f"run scripts/preprocess.py --dataset {args.dataset}"),
        (paths.scaler_yaml(args.dataset), f"run scripts/preprocess.py --dataset {args.dataset}"),
    ])

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
    scaler_path = paths.scaler_yaml(ds_name)
    train_ds = TrafficDataset(paths.split_npz(ds_name, "train"), scaler_path)
    mean = train_ds.mean
    std = train_ds.std

    out_dir = paths.residual_dir(ds_name, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # extract each split; cache per-horizon variances for the train-vs-val check.
    # ------------------------------------------------------------------ #
    delta_per_split: dict[str, np.ndarray] = {}

    for split in ("train", "val", "test"):
        print(f"\nExtracting {split} split ...")
        ds = TrafficDataset(paths.split_npz(ds_name, split), scaler_path)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        residuals, y_pred, y_true, mask = extract_one_split(
            model, loader, mean, std, eff_device, split
        )

        out_path = paths.residual_npz(ds_name, split, args.seed)
        # note: y_true is NOT saved - it's exactly y_pred + residuals, and only
        # residuals / y_pred / mask are read downstream. saves ~1/3 of the disk.
        np.savez_compressed(
            out_path,
            residuals=residuals,
            y_pred=y_pred,
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
    # train-vs-val per-horizon variance check
    # ------------------------------------------------------------------ #
    print("\n--- train-vs-val per-horizon variance ---")
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

    diag_path = out_dir / "diagnostic.json"
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

    print("\ndone. residuals saved as (B, N, H) in raw units.")


if __name__ == "__main__":
    main()
