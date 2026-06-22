"""Extract Graph-WaveNet residuals using the vendored repo directly (no wrapper).

Given a checkpoint trained by `third_party/Graph-WaveNet/train.py` on a variant's
`gwn_native/` view, this loads the repo's own `gwnet` model + `util.load_adj`,
runs inference (no shuffle) over train/val/test, undoes the repo's train-only
z-score, and saves residuals r = y - yhat in raw units, reshaped (B, N, H) so
vec(R) is column-major over N and Cov(vec(R)) = Sigma_T (x) Sigma_S.

Output per split (results/residuals/<variant>/seed<seed>/<split>.npz):
    residuals (B,N,H) float32, y_pred (B,N,H), mask (B,N,H) bool, mean, std.

Usage:
    .venv/bin/python scripts/gwn_extract_residuals.py \
        --variant pems08s --seed 0 \
        --checkpoint results/backbone/pems08s/seed0/best.pth

Requires the GWN model.py Conv1d->Conv2d patch (see THIRD_PARTY_PATCHES.md).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Graph-WaveNet"
sys.path.insert(0, str(REPO))
import util                       # noqa: E402  (repo module)
from model import gwnet          # noqa: E402

N_NODES = {"metrla": 207, "pemsbay": 325, "pems04f": 307, "pems04s": 307,
           "pems08f": 170, "pems08s": 170}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(N_NODES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", required=True,
                    help="state_dict saved by GWN train.py")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--nhid", type=int, default=32)
    args = ap.parse_args()

    n = N_NODES[args.variant]
    nat = ROOT / "data" / "processed" / args.variant / "gwn_native"
    out_dir = ROOT / "results" / "residuals" / args.variant / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # repo loaders on the native view + pkl
    _, _, supports = util.load_adj(str(nat / "adj_mx.pkl"), "doubletransition")
    supports = [torch.tensor(s).to(device) for s in supports]
    dl = util.load_dataset(str(nat), args.batch_size, args.batch_size, args.batch_size)
    scaler = dl["scaler"]

    # repo model, matching train.py defaults (gcn_bool + addaptadj, doubletransition)
    model = gwnet(device, n, supports=supports, gcn_bool=True, addaptadj=True,
                  aptinit=supports[0], in_dim=2, out_dim=12,
                  residual_channels=args.nhid, dilation_channels=args.nhid,
                  skip_channels=args.nhid * 8, end_channels=args.nhid * 16).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"loaded {args.checkpoint}; scaler mean={scaler.mean:.3f} std={scaler.std:.3f}")

    for split in ("train", "val", "test"):
        x = dl[f"x_{split}"]                       # (B,12,N,2), ch0 already z-scored
        y = dl[f"y_{split}"]                       # (B,12,N,2) raw
        preds = []
        with torch.no_grad():
            for i in range(0, x.shape[0], args.batch_size):
                xb = torch.Tensor(x[i:i + args.batch_size]).to(device).transpose(1, 3)
                out = model(xb)                    # (b,12,N,1) z-space
                preds.append(scaler.inverse_transform(out).cpu().numpy())
        pred_raw = np.concatenate(preds, axis=0)[:, :, :, 0]   # (B,12,N) raw
        y_real = y[..., 0]                                     # (B,12,N) raw
        # -> (B,N,H)
        y_pred = np.transpose(pred_raw, (0, 2, 1)).astype(np.float32)
        y_true = np.transpose(y_real, (0, 2, 1)).astype(np.float32)
        mask = (y_true != 0.0)
        residuals = (y_true - y_pred).astype(np.float32)

        out_path = out_dir / f"{split}.npz"
        tmp = out_path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, residuals=residuals, y_pred=y_pred, mask=mask,
                                mean=np.float32(scaler.mean), std=np.float32(scaler.std))
        tmp.replace(out_path)
        vr = residuals[mask]
        print(f"  {split}: {residuals.shape} (B,N,H)  masked mean={vr.mean():+.4f} "
              f"std={vr.std():.4f}  missing={1 - mask.mean():.4f}")

    print("done — residuals in raw units (B,N,H).")


if __name__ == "__main__":
    main()
