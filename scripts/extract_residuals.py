"""Extract point-forecast residuals from a trained backbone -> the diffusion target.

Loads a best.pt from train_backbone.py, runs inference NO-SHUFFLE over each split,
and writes r = y - ŷ in RAW units, reshaped (B, N, H) so vec(R) is column-major
over nodes and Cov(vec(R)) = Sigma_T ⊗ Sigma_S (the Kronecker prior's convention).

Output per split (results/residuals/<variant>/<backbone>/seed<seed>/<split>.npz):
    residuals (B,N,H) float32 | y_pred (B,N,H) | mask (B,N,H) bool | mean, std scalars

The mask is the split's stored y_mask (family convention): the diffusion loss and
prior fit consume it, so offline sensors never enter either. y_pred is kept so the
eval stage can reconstruct y = ŷ + (residual sample) in raw units.

Usage:
  .venv/bin/python scripts/extract_residuals.py --variant pems08s --backbone gwn --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbones import build_backbone, forward_backbone, IN_DIM   # noqa: E402
from src.utils import paths                                          # noqa: E402


def _bnh(a):
    """(B,T,N,1) -> (B,N,H): drop feature axis, put horizon last."""
    return np.ascontiguousarray(a[..., 0].transpose(0, 2, 1))        # (B,N,T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--backbone", required=True, choices=("mlp", "gcn", "gwn"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device(args.device)
    variant, name = args.variant, args.backbone
    meta = json.loads((paths.processed_dir(variant) / "meta.json").read_text())
    scaler = json.loads(paths.scaler_json(variant).read_text())
    mean, std = float(scaler["mean"]), float(scaler["std"])
    N = meta["N"]
    A = np.load(paths.adj_npz(variant))["A_sym"]

    ckpt = torch.load(paths.backbone_dir(variant, args.seed, name) / "best.pt",
                      map_location=device)
    model = build_backbone(name, N, A, device=device).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"[{variant}/{name}/seed{args.seed}] loaded best.pt "
          f"(epoch {ckpt.get('epoch')}, val_mae {ckpt.get('val_mae'):.4f})")

    out_dir = paths.residual_dir(variant, args.seed, name)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        d = np.load(paths.split_npz(variant, split))
        x = torch.as_tensor(d["x"], dtype=torch.float32)         # (B,T,N,1) raw
        xt = torch.as_tensor(d["x_tod"], dtype=torch.float32)
        y = d["y"][..., 0]                                       # (B,T,N) raw value
        ym = d["y_mask"]                                         # (B,T,N,1)

        xz = (x - mean) / std
        ch = [xz] if IN_DIM[name] == 1 else [xz, xt]
        xin = torch.cat(ch, dim=-1).permute(0, 3, 2, 1)          # (B,C,N,T)

        preds = []
        with torch.no_grad():
            for i in range(0, xin.shape[0], args.batch_size):
                pb = forward_backbone(name, model, xin[i:i + args.batch_size].to(device))
                preds.append((pb * std + mean).cpu().numpy())    # raw units
        yhat = np.concatenate(preds, 0)[..., 0]                  # (B,T,N)

        resid = _bnh((y - yhat)[..., None])                      # (B,N,H)
        y_pred = _bnh(yhat[..., None])
        mask = _bnh(ym).astype(bool)
        np.savez_compressed(out_dir / f"{split}.npz",
                            residuals=resid.astype(np.float32),
                            y_pred=y_pred.astype(np.float32),
                            mask=mask, mean=np.float32(mean), std=np.float32(std))
        rv = resid[mask]
        print(f"  {split}: residuals {resid.shape}  masked mean={rv.mean():+.3f} "
              f"std={rv.std():.3f}  missing={1 - mask.mean():.4f}")

    print(f"[done] -> {out_dir}")


if __name__ == "__main__":
    main()
