"""Train one point-forecast backbone (mlp|gcn|gwn) on one variant, one seed.

ONE loop for all three backbones, so the only thing that varies is the
architecture -- no per-model training protocol, no second scaler. This is the
discipline whose absence produced the FCPS/ULS schedule confound in the old code.

Protocol (matches the GWN/DCRNN/BasicTS field convention, verified 2026-07-17):
- normalization: single GLOBAL scalar z-score, TRAIN-ONLY, value channel only
  (from scaler.json). time-of-day fed RAW in [0,1) as GWN's 2nd channel.
- model predicts in z-space; prediction is de-normalized to RAW units before the
  loss (GWN scales x only, inverse-transforms the prediction). masked MAE in raw
  units, Adam, early-stop on val MAE. mask = the family's stored y_mask (speed:
  value!=0 masked; PEMS: nothing masked, null_val=nan).

Usage:
  .venv/bin/python scripts/train_backbone.py --variant pems08s --backbone gwn --seed 42
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbones import build_backbone, forward_backbone, IN_DIM   # noqa: E402
from src.utils import paths                                          # noqa: E402
from src.utils.seed import set_seed                                  # noqa: E402

sys.path.insert(0, str(ROOT / "third_party" / "Graph-WaveNet"))
from util import masked_mae, masked_rmse                             # noqa: E402


def load_split(variant, split, device):
    d = np.load(paths.split_npz(variant, split))
    x = torch.as_tensor(d["x"], dtype=torch.float32)          # (B,T,N,1) raw value
    y = torch.as_tensor(d["y"], dtype=torch.float32)          # (B,T,N,1) raw value
    xt = torch.as_tensor(d["x_tod"], dtype=torch.float32)     # (B,T,N,1) tod [0,1)
    ym = torch.as_tensor(d["y_mask"], dtype=torch.float32)    # (B,T,N,1) validity
    return x, y, xt, ym


def make_input(x_raw, tod, mean, std, in_dim):
    """(B,T,N,1) value + tod -> (B, C_in, N, T) with value z-scored, tod raw."""
    xz = (x_raw - mean) / std                                 # z-score value
    ch = [xz] if in_dim == 1 else [xz, tod]
    x = torch.cat(ch, dim=-1)                                 # (B,T,N,C)
    return x.permute(0, 3, 2, 1)                              # (B,C,N,T)


def evaluate(model, name, loader, mean, std, null_val, device):
    model.eval()
    tot_mae = tot_rmse = n = 0.0
    with torch.no_grad():
        for xin, y in loader:
            xin, y = xin.to(device), y.to(device)
            pred_z = forward_backbone(name, model, xin)       # (B,T,N,1) z-space
            pred = pred_z * std + mean                        # -> raw units
            b = y.shape[0]
            tot_mae += masked_mae(pred, y, null_val).item() * b
            tot_rmse += masked_rmse(pred, y, null_val).item() * b
            n += b
    return tot_mae / n, tot_rmse / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--backbone", required=True, choices=("mlp", "gcn", "gwn"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-train-batches", type=int, default=0)   # >0 = smoke
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    variant, name = args.variant, args.backbone

    meta = json.loads((paths.processed_dir(variant) / "meta.json").read_text())
    scaler = json.loads(paths.scaler_json(variant).read_text())
    mean = torch.tensor(scaler["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(scaler["std"], dtype=torch.float32, device=device)
    N = meta["N"]
    # PEMS flow/speed: null_val=nan (mask nothing); speed pair: 0.0 (mask zeros).
    null_val = float("nan") if variant.startswith("pems") else 0.0
    A = np.load(paths.adj_npz(variant))["A_sym"]

    # build tensors once (z-score value, keep raw y for the loss)
    loaders = {}
    for split in ("train", "val", "test"):
        x, y, xt, ym = load_split(variant, split, device)
        xin = make_input(x, xt, mean.cpu(), std.cpu(), IN_DIM[name])   # (B,C,N,T)
        ds = TensorDataset(xin, y)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size,
                                    shuffle=(split == "train"), drop_last=False)

    model = build_backbone(name, N, A, device=device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    out_dir = paths.backbone_dir(variant, args.seed, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{variant}/{name}/seed{args.seed}] N={N} params={n_params:,} "
          f"null_val={null_val} device={device}")

    best = float("inf"); bad = 0; hist = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for bi, (xin, y) in enumerate(loaders["train"]):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            xin, y = xin.to(device), y.to(device)
            opt.zero_grad()
            pred = forward_backbone(name, model, xin) * std + mean     # raw units
            loss = masked_mae(pred, y, null_val)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += loss.item(); nb += 1
        val_mae, val_rmse = evaluate(model, name, loaders["val"], mean, std, null_val, device)
        sched.step(val_mae)
        hist.append({"epoch": ep, "train_mae": tot / max(nb, 1),
                     "val_mae": val_mae, "val_rmse": val_rmse})
        improved = val_mae < best - 1e-5
        if improved:
            best = val_mae; bad = 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_mae": val_mae,
                        "backbone": name, "variant": variant, "seed": args.seed,
                        "n_params": n_params}, out_dir / "best.pt")
        else:
            bad += 1
        print(f"  ep{ep:03d} train {tot/max(nb,1):.4f}  val_mae {val_mae:.4f} "
              f"rmse {val_rmse:.4f}  {'*' if improved else ''}({time.time()-t0:.1f}s)")
        if bad >= args.patience:
            print(f"  early stop (no val improvement in {args.patience})")
            break

    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        **vars(args), "n_params": n_params, "null_val": ("nan" if null_val != null_val else null_val),
        "best_val_mae": best}, indent=2))
    # final test with the best checkpoint
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
    test_mae, test_rmse = evaluate(model, name, loaders["test"], mean, std, null_val, device)
    print(f"[done] best val_mae {best:.4f} | test_mae {test_mae:.4f} rmse {test_rmse:.4f}")


if __name__ == "__main__":
    main()
