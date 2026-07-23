"""Re-implement HySRD-Net's ST-TrendNet (ESWA 2026) and extract its residuals.

HySRD has NO public code (checked 2026-07-23: no availability statement in the
paper, nothing on github, Awesome-Traffic-Prediction lists it as code N/A), so
this is a faithful re-implementation from the paper (sec 4.2, Alg 1, sec 5.4):

  - node embeddings E (N,d_e); adaptive adjacency A = f(E E^T)   (eq 19)
  - per timestep: X_sp = AdaptiveGCN(X_t, A); h_t = GRU(X_sp, h_{t-1})  (alg 1)
  - FC head: last hidden state -> 12-step trend forecast per node
  - Adam lr 1e-4, batch 32, MSE loss (eq 20)
  - defaults from table 5: D_gru=64, L_gc=1  (sec 5.4 text says hidden 96,
    contradicting table 5's "64 (Default)" -- we follow the table)

paper leaves unspecified (assumptions, all conventional for this family):
  - f in eq 19: softmax(relu(.)) as in Graph WaveNet / AGCRN (their citations)
  - embedding dim d_e = 10 (GWN/AGCRN convention)
  - input normalization: global train z-score (same as our backbones); MSE in
    z-space (Adam is loss-scale invariant, so this matches eq 20 up to a const)
  - epochs / stopping: max 100, early stop on val MSE, patience 10

the residual we extract is the DIFFUSION TARGET residual of their alg 2 line 8:
r = y_true - trend_future, the full 12-step forecast residual, raw units,
(B,N,H) -- byte-compatible with our backbone caches, saved under
results/residuals/<ds>/hysrd/seed<seed>/. their preprocessing (amplify x100 +
node-wise z-score + GFT) happens at analysis time, not here, so the cache stays
in raw units like every other arm.

usage:
  .venv/bin/python scripts/baselines/hysrd_trendnet.py --variant pems04f
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                          # noqa: E402
from src.utils.seed import set_seed                                  # noqa: E402


class STTrendNet(nn.Module):
    """adaptive-GCN + GRU trend extractor (HySRD alg 1), one recurrent layer."""

    def __init__(self, n_nodes, horizon=12, d_embed=10, d_gru=64, l_gc=1):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_nodes, d_embed) * 0.1)
        # l_gc graph-conv layers; first lifts the 1-dim value to d_gru
        dims = [1] + [d_gru] * l_gc
        self.gc = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(l_gc))
        # the graph conv has no recurrence, so it runs on all T steps at once and
        # a fused nn.GRU replaces the per-step GRUCell loop (same math, faster)
        self.gru = nn.GRU(d_gru, d_gru, batch_first=True)
        self.fc = nn.Linear(d_gru, horizon)
        self.d_gru = d_gru

    def adaptive_adj(self):
        # eq 19: nonlinear activation on embedding similarity (GWN convention)
        return torch.softmax(torch.relu(self.E @ self.E.T), dim=1)

    def forward(self, x):
        # x (B,T,N,1) z-scored value -> (B,N,H) z-space trend forecast
        B, T, N, _ = x.shape
        A = self.adaptive_adj()
        v = x
        for lin in self.gc:
            v = torch.relu(lin(torch.einsum("mn,btnf->btmf", A, v)))
        v = v.permute(0, 2, 1, 3).reshape(B * N, T, self.d_gru)
        out, _ = self.gru(v)
        return self.fc(out[:, -1]).reshape(B, N, -1)


def load_split(variant, split):
    d = np.load(paths.split_npz(variant, split))
    x = torch.as_tensor(d["x"], dtype=torch.float32)      # (B,T,N,1) raw
    y = torch.as_tensor(d["y"], dtype=torch.float32)      # (B,T,N,1) raw
    ym = torch.as_tensor(d["y_mask"], dtype=torch.float32)
    return x, y, ym


def evaluate(model, loader, mean, std, device):
    # raw-unit MAE/RMSE over the full horizon (their table 2 convention)
    model.eval()
    se = ae = n = 0.0
    with torch.no_grad():
        for xz, y in loader:
            xz, y = xz.to(device), y.to(device)
            pred = model(xz) * std + mean                 # (B,N,H) raw
            yt = y[..., 0].permute(0, 2, 1)               # (B,T,N,1) -> (B,N,H)
            ae += (pred - yt).abs().sum().item()
            se += ((pred - yt) ** 2).sum().item()
            n += yt.numel()
    return ae / n, (se / n) ** 0.5


def extract(model, variant, split, mean, std, device, batch_size):
    """run no-shuffle inference, save the cache npz in our (B,N,H) convention."""
    x, y, ym = load_split(variant, split)
    xz = (x - mean) / std
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(xz), batch_size):
            preds.append(model(xz[i:i + batch_size].to(device)).cpu())
    pred = torch.cat(preds) * std + mean                  # (B,N,H) raw
    yt = y[..., 0].permute(0, 2, 1)                       # (B,N,H) raw
    mask = ym[..., 0].permute(0, 2, 1).bool()
    return {
        "residuals": (yt - pred).numpy().astype(np.float32),
        "y_pred": pred.numpy().astype(np.float32),
        "mask": mask.numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)     # paper sec 5.4
    ap.add_argument("--lr", type=float, default=1e-4)         # paper sec 5.4
    ap.add_argument("--hidden", type=int, default=64)         # table 5 default
    ap.add_argument("--l-gc", type=int, default=1)            # table 5 default
    ap.add_argument("--d-embed", type=int, default=10)        # assumption
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--max-train-batches", type=int, default=0)   # >0 = smoke
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    variant = args.variant

    meta = json.loads((paths.processed_dir(variant) / "meta.json").read_text())
    scaler = json.loads(paths.scaler_json(variant).read_text())
    mean, std = float(scaler["mean"]), float(scaler["std"])
    N = meta["N"]

    loaders = {}
    for split in ("train", "val"):
        x, y, _ = load_split(variant, split)
        xz = (x - mean) / std
        loaders[split] = DataLoader(TensorDataset(xz, y), batch_size=args.batch_size,
                                    shuffle=(split == "train"), drop_last=False)

    model = STTrendNet(N, d_embed=args.d_embed, d_gru=args.hidden,
                       l_gc=args.l_gc).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = paths.backbone_dir(variant, args.seed, "hysrd")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{variant}/hysrd/seed{args.seed}] N={N} params={n_params:,} device={device}")

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)
    best = float("inf"); bad = 0; hist = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for bi, (xz, y) in enumerate(loaders["train"]):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            xz, y = xz.to(device), y.to(device)
            yz = (y[..., 0].permute(0, 2, 1) - mean_t) / std_t   # (B,N,H) z-space
            opt.zero_grad()
            loss = ((model(xz) - yz) ** 2).mean()                # eq 20 (z-space)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        val_mae, val_rmse = evaluate(model, loaders["val"], mean_t, std_t, device)
        hist.append({"epoch": ep, "train_mse_z": tot / max(nb, 1),
                     "val_mae": val_mae, "val_rmse": val_rmse})
        improved = val_rmse < best - 1e-5                        # mse-trained -> rmse stop
        if improved:
            best = val_rmse; bad = 0
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_rmse": val_rmse, "variant": variant,
                        "seed": args.seed, "n_params": n_params}, out_dir / "best.pt")
        else:
            bad += 1
        print(f"  ep{ep:03d} train_mse {tot/max(nb,1):.4f}  val_mae {val_mae:.4f} "
              f"rmse {val_rmse:.4f}  {'*' if improved else ''}({time.time()-t0:.1f}s)")
        if bad >= args.patience:
            print(f"  early stop (no val improvement in {args.patience})")
            break

    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        **vars(args), "n_params": n_params, "best_val_rmse": best}, indent=2))

    # reload best, report test accuracy (their table 2 deterministic row is the
    # calibration target: pems04f RMSE 32.02 / MAE 20.43, pems08s 3.37 / 1.53)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
    x, y, _ = load_split(variant, "test")
    tl = DataLoader(TensorDataset((x - mean) / std, y), batch_size=args.batch_size)
    test_mae, test_rmse = evaluate(model, tl, mean_t, std_t, device)
    print(f"[done] best val_rmse {best:.4f} | test_mae {test_mae:.4f} rmse {test_rmse:.4f}")

    # residual caches, same convention as extract_residuals.py
    res_dir = paths.residual_dir(variant, args.seed, "hysrd")
    for split in ("train", "val", "test"):
        out = extract(model, variant, split, mean, std, device, args.batch_size)
        np.savez_compressed(res_dir / f"{split}.npz", **out,
                            mean=np.float32(mean), std=np.float32(std))
        r = out["residuals"]
        print(f"  [{split}] residuals {r.shape} std={r.std():.3f} -> "
              f"{res_dir / f'{split}.npz'}")


if __name__ == "__main__":
    main()
