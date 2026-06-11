"""qualitative forecast-interval plot (task c).

generates reverse-sde samples from the trained diffusion checkpoints for a few
test windows and plots, per (window, sensor): the ground-truth trajectory, the
deterministic backbone point forecast, and the predictive median + 90% band for
the structured (kron) vs isotropic prior side by side. this is the picture that
shows *what the uncertainty looks like* — where the kron band is sharper/better
shaped than iso, and whether truth stays inside the band.

reconstruction (raw units):
    y_true   = y_pred_raw + residual_raw        (both from the residual npz)
    backbone = y_pred_raw
    band     = y_pred_raw + quantiles_K( sample_residual_raw )
    sample_residual_raw = reverse_sample(...) * sigma_r

runs on cpu by default (a handful of windows is cheap). reuses the exact eval
building blocks so the samples match scripts/eval_diffusion.py.

usage:
    python scripts/analysis/qualitative_samples.py --dataset metrla --seed 42 \
        --exp 20260610 --windows 6 --k 100
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.adjacency import load_adjacency
from src.data.residual_dataset import ResidualDataset, residual_path, train_residual_sigma
from src.diffusion.denoiser import R0Denoiser
from src.diffusion.forward_fcps import FCPSSchedule
from src.diffusion.prior import load_prior
from src.diffusion.reverse import reverse_sample
from src.utils import paths

PRIOR_COLOR = {"kronecker": "#1f77b4", "isotropic": "#d62728"}
PRIOR_LABEL = {"kronecker": "kron (ours)", "isotropic": "iso (baseline)"}


def load_run(dataset, prior, seed, exp, device):
    """minimal replica of eval_diffusion.load_run (kept in sync)."""
    ckpt_dir = paths.diffusion_dir(dataset, prior, seed, exp=exp)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    prior_obj = load_prior(dataset, prior, seed)
    # schedule rebuilt from the prior, not config.json's ignored c_0 (see eval_diffusion)
    schedule = FCPSSchedule.for_prior(prior_obj, c_min=cfg["c_min"], alpha=cfg["alpha"], T=1.0)
    denoiser = R0Denoiser(
        N=prior_obj.N, H=prior_obj.H, adjacency=load_adjacency(dataset),
        d_h=cfg["d_h"], channel_multipliers=tuple(cfg["channel_multipliers"]),
        n_blocks=cfg["n_blocks"], device=str(device),
    ).to(device)
    denoiser.load_state_dict(ckpt["model_state_dict"])
    denoiser.eval()
    return denoiser, prior_obj, schedule, cfg


@torch.no_grad()
def gen_samples(dataset, prior, seed, exp, device, win_idx, k, n_steps, eta, generator):
    """raw-unit residual samples for the chosen windows. -> (K, W, N, H)."""
    denoiser, prior_obj, schedule, _ = load_run(dataset, prior, seed, exp, device)
    N, H = prior_obj.N, prior_obj.H
    sigma_r = train_residual_sigma(dataset, seed)
    test = ResidualDataset(residual_path(dataset, "test", seed), sigma_r)
    Y = test.Y_hat[win_idx].to(device)                 # (W, N, H) z-scored conditioning
    samp, _ = reverse_sample(
        prior_obj, schedule, denoiser, Y.repeat(k, 1, 1),
        n_steps=n_steps, eta=eta, R_T=None, generator=generator,
    )
    return (samp.view(k, len(win_idx), N, H) * sigma_r).cpu().numpy()


def pick_windows(dataset, seed, n_windows, rng):
    """choose fully-observed test windows spanning the residual-energy
    distribution: a representative spread from typical (well-forecast, where we
    expect the band to cover) up to the hard tail (sudden congestion, where it
    is stressed). evenly-spaced energy percentiles → an honest mix, not just the
    worst cases."""
    d = np.load(residual_path(dataset, "test", seed))
    r = d["residuals"].astype(np.float32); m = d["mask"].astype(bool)
    full = m.reshape(m.shape[0], -1).all(axis=1)
    energy = (np.abs(r) * m).reshape(r.shape[0], -1).mean(axis=1)
    idx = np.where(full)[0]
    order = idx[np.argsort(energy[idx])]
    # spread across the 40th..96th percentile (skip the very flat low tail —
    # those panels are visually boring — and include one hard case at the top).
    pcts = np.linspace(0.40, 0.96, n_windows)
    sel = [int(order[min(len(order) - 1, int(p * len(order)))]) for p in pcts]
    return sorted(dict.fromkeys(sel))            # dedup, keep order


def pick_nodes(dataset, seed, win_idx, n_nodes, rng):
    """per window, a representative *active* sensor (≈75th percentile of |residual|
    swing) — visible dynamics without cherry-picking the single worst outlier."""
    d = np.load(residual_path(dataset, "test", seed))
    r = d["residuals"].astype(np.float32)
    nodes = {}
    for w in win_idx:
        score = np.abs(r[w]).mean(axis=1)            # (N,)
        ranked = np.argsort(score)                   # ascending
        q = int(0.75 * (len(ranked) - 1))
        pick = ranked[max(0, q - n_nodes // 2): q - n_nodes // 2 + n_nodes]
        nodes[w] = sorted(int(x) for x in pick)
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--nodes-per-window", type=int, default=1)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/analysis/{args.exp}"
    os.makedirs(out, exist_ok=True)
    device = torch.device(args.device)

    rng = np.random.default_rng(0)          # deterministic window/node choice
    gen = torch.Generator(device=device).manual_seed(args.seed)

    d = np.load(residual_path(args.dataset, "test", args.seed))
    y_pred = d["y_pred"].astype(np.float32)         # raw backbone forecast
    resid = d["residuals"].astype(np.float32)       # raw true residual
    y_true = y_pred + resid

    win = pick_windows(args.dataset, args.seed, args.windows, rng)
    nodes = pick_nodes(args.dataset, args.seed, win, args.nodes_per_window, rng)
    print(f"[{args.dataset}] windows={win}")

    samples = {}
    for pr in ("kronecker", "isotropic"):
        print(f"  generating {pr} (K={args.k}, {args.n_steps} steps)...", flush=True)
        samples[pr] = gen_samples(args.dataset, pr, args.seed, args.exp, device,
                                  win, args.k, args.n_steps, args.eta, gen)

    # cache the (expensive) raw residual samples + reconstruction inputs so the
    # plot can be re-tweaked locally without re-running the reverse sde.
    win_arr = np.asarray(win)
    npz = os.path.join(out, f"samples_{args.dataset}_seed{args.seed}.npz")
    np.savez_compressed(
        npz, windows=win_arr,
        nodes=np.asarray([nodes[w] for w in win]),
        y_pred=y_pred[win_arr], y_true=y_true[win_arr],
        kron=samples["kronecker"], iso=samples["isotropic"],
        alpha=args.alpha, k=args.k, n_steps=args.n_steps, eta=args.eta,
    )
    print("saved", npz, flush=True)

    # one panel per (window, node)
    panels = [(w, n) for w in win for n in nodes[w]]
    ncol = min(3, len(panels)); nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow), squeeze=False)
    H = y_true.shape[-1]; h = np.arange(1, H + 1)
    qlo, qhi = args.alpha / 2, 1 - args.alpha / 2
    for pi, (w, n) in enumerate(panels):
        ax = axes[pi // ncol][pi % ncol]
        wi = win.index(w)
        ax.plot(h, y_true[w, n], "k-o", ms=3, lw=1.8, label="ground truth", zorder=5)
        ax.plot(h, y_pred[w, n], color="0.45", ls="--", lw=1.5, label="backbone (point)", zorder=4)
        for pr in ("kronecker", "isotropic"):
            s = samples[pr][:, wi, n, :]                       # (K, H) raw residual
            traj = y_pred[w, n][None, :] + s                   # (K, H) raw forecast
            med = np.median(traj, axis=0)
            lo = np.quantile(traj, qlo, axis=0); hi = np.quantile(traj, qhi, axis=0)
            c = PRIOR_COLOR[pr]
            ax.fill_between(h, lo, hi, color=c, alpha=0.18,
                            label=f"{PRIOR_LABEL[pr]} {int((1-args.alpha)*100)}% band")
            ax.plot(h, med, color=c, lw=1.3, alpha=0.9)
        ax.set_title(f"window {w}, sensor {n}", fontsize=9)
        ax.set_xlabel("horizon"); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
        if pi == 0:
            ax.legend(fontsize=7, loc="best")
    for pi in range(len(panels), nrow * ncol):
        axes[pi // ncol][pi % ncol].axis("off")
    unit = "speed (mph)" if args.dataset in ("metrla", "pemsbay") else "flow (veh/5min)"
    fig.suptitle(f"{args.dataset} — predictive intervals (kron vs iso), {unit}, seed {args.seed}\n"
                 f"K={args.k} reverse-SDE samples, η={args.eta}, {int((1-args.alpha)*100)}% band",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out, f"fig5_intervals_{args.dataset}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
