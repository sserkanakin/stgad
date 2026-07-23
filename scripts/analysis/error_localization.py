"""Where does each prior arm lose points? Per-node / per-window / per-horizon CRPS.

Samples the trained ladder checkpoints (iso/emp/rsh, same dataset+backbone+seed) locally
and accumulates the normalized quantile-CRPS numerator/denominator PER NODE, PER WINDOW
and PER HORIZON, plus the ensemble-mean MAE per node. The saved npz lets us compare WHERE
the arms differ, not just the scalar: same places or different, sharpness vs location.

usage: .venv/bin/python scripts/analysis/error_localization.py --dataset pems08f \
          --backbone gwn --seed 42 --max-windows 600 --k 32 --n-steps 25
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                                       # noqa: E402
from src.data.residual_dataset import make_dataset, train_residual_sigma, sigma_to_torch  # noqa: E402
from src.diffusion.reverse import reverse_sample_ancestral                        # noqa: E402
from src.eval.crps import _QUANTILES                                              # noqa: E402

# reuse load_run from the eval script (rebuilds prior+schedule+denoiser from config)
spec = importlib.util.spec_from_file_location("eval_diffusion", ROOT / "scripts" / "eval_diffusion.py")
ed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ed)

ARMS = [("iso", "isotropic"), ("emp", "kronecker"), ("rsh", "kronecker")]


@torch.no_grad()
def run_arm(arm, prior, args, device):
    # the run tree is arm-namespaced: <exp>/<arm>/<ds>/<bb>/<prior>/seed*/train
    denoiser, prior_obj, schedule, cfg, _ = ed.load_run(
        args.dataset, prior, args.seed, f"{args.exp}/{arm}", device, args.backbone,
        ckpt_name="best_crps.pt")
    N, H = prior_obj.N, prior_obj.H
    res_seed = cfg.get("residual_seed", args.seed)
    sigma_r = train_residual_sigma(args.dataset, res_seed, args.backbone, norm=cfg.get("norm", "global"))
    sig = sigma_to_torch(sigma_r, device)
    ds = make_dataset(args.dataset, "test", res_seed, args.backbone, sigma_r,
                      cond_past=bool(cfg.get("cond_past")),
                      zero_masked=bool(cfg.get("zero_masked_input")))
    B = ds.R0.shape[0]
    # stride-subsample the test windows so the subset spans the whole test period
    idx = np.arange(B) if not args.max_windows else \
        np.linspace(0, B - 1, min(args.max_windows, B)).round().astype(int)
    R0_all, Y_all, M_all = ds.R0[idx], ds.Y_hat[idx], ds.mask[idx]
    Yt_all = ds.y_true_raw[idx]
    M_all = M_all & (Yt_all.abs() > 1e-6)                     # mask y==0 (protocol)

    qs = _QUANTILES.to(device)
    num_nh = torch.zeros(N, H, device=device)                 # pinball numerator per (node,horizon)
    den_nh = torch.zeros(N, H, device=device)                 # sum |y| per (node,horizon)
    mae_nh = torch.zeros(N, H, device=device)                 # ensemble-mean abs err
    cnt_nh = torch.zeros(N, H, device=device)
    num_w, den_w = [], []                                     # per window
    chunk = max(1, args.gen_batch // args.k)
    n = len(idx)
    t0 = time.time()
    for ci, s in enumerate(range(0, n, chunk)):
        e = min(s + chunk, n)
        R0 = R0_all[s:e].to(device) * sig
        Y = Y_all[s:e].to(device)
        Mk = M_all[s:e].to(device)
        yt = Yt_all[s:e].to(device)
        yp = yt - R0
        b = R0.shape[0]
        samp, _ = reverse_sample_ancestral(prior_obj, schedule, denoiser,
                                           Y.repeat(args.k, 1, 1), n_steps=args.n_steps)
        fc = (yp.unsqueeze(0) + samp.view(args.k, b, N, H) * sig).clamp(min=0.0)
        q_pred = torch.quantile(fc, qs, dim=0)                # (19,b,N,H)
        yb = yt.unsqueeze(0)
        pin = ((q_pred - yb) * ((yb <= q_pred).to(fc.dtype) - qs.view(-1, 1, 1, 1))).abs()
        ptnum = 2.0 * pin.sum(0) * Mk                         # (b,N,H) numerator per point
        ptden = yt.abs() * Mk
        num_nh += ptnum.sum(0); den_nh += ptden.sum(0)
        mae_nh += ((fc.mean(0) - yt).abs() * Mk).sum(0); cnt_nh += Mk.sum(0)
        num_w.append(ptnum.sum(dim=(1, 2)).cpu()); den_w.append(ptden.sum(dim=(1, 2)).cpu())
        if ci % 5 == 0:
            el = time.time() - t0
            print(f"  [{arm}] chunk {ci + 1}/{(n + chunk - 1) // chunk}  {el:.0f}s", flush=True)
    out = paths.analysis_dir("error_localization")
    num_w = torch.cat(num_w).numpy(); den_w = torch.cat(den_w).numpy()
    f = out / f"{args.dataset}_{args.backbone}_seed{args.seed}_{arm}.npz"
    np.savez(f,
             num_nh=num_nh.cpu().numpy(), den_nh=den_nh.cpu().numpy(),
             mae_nh=mae_nh.cpu().numpy(), cnt_nh=cnt_nh.cpu().numpy(),
             num_w=num_w, den_w=den_w, window_idx=idx,
             crps=float(num_nh.sum().item() / (den_nh.sum().item() * 19.0)),
             k=args.k, n_steps=args.n_steps)
    print(f"[{arm}] saved {f}  overall crps={num_nh.sum().item() / (den_nh.sum().item() * 19.0):.4f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pems08f")
    ap.add_argument("--backbone", default="gwn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp", default="20260721_ladder")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=25)
    ap.add_argument("--max-windows", type=int, default=600)
    ap.add_argument("--gen-batch", type=int, default=1024)
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  dataset={args.dataset}/{args.backbone}  arms={[a for a, _ in ARMS]}")
    for arm, prior in ARMS:
        run_arm(arm, prior, args, device)


if __name__ == "__main__":
    main()
