"""Generate diffusion forecast samples and reduce them to metrics (GPU step).

This is the expensive step — it runs the trained reverse sampler. To stay within
memory it *streams*: each batch of windows is sampled (K members), scored into a
:class:`~src.eval.metrics.MetricAccumulator`, and discarded. Only the finalized
metrics and a small *sample bank* (a handful of windows kept in full, for the
fan-chart / spatial figures) are written to disk.

Outputs (under ``results/eval/{dataset}/``):
  * ``metrics_{split}.json`` — finalized diffusion metrics + run metadata.
  * ``samples_{split}.npz``  — sample bank (Y_sample, y_true, y_hat, mask, idx)
                               + one reverse-sampling trace for the figures.

Usage:
    # DAIC, real run:
    python scripts/eval/generate_samples.py \
        --ckpt-dir results/checkpoints/diffusion/diff_gwn_metrla_seed0_<hash> \
        --split test --n-samples 100 --n-steps 100 --eta 1.0

    # Fast plumbing check (meaningless numbers, proves it runs):
    python scripts/eval/generate_samples.py --ckpt-dir <dir> \
        --split val --smoke --max-windows 64 --n-samples 16 --n-steps 25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.io import load_eval_checkpoint, load_split_arrays   # noqa: E402
from src.eval.metrics import MetricAccumulator                    # noqa: E402
from src.eval.samplers import DiffusionSampler                    # noqa: E402

N_BANK = 12   # windows kept in full for figures


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True, help="results/checkpoints/diffusion/<run_id>")
    p.add_argument("--which", choices=("best", "last"), default="best")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--seed", type=int, default=0, help="residual-file seed + sampler seed")
    p.add_argument("--n-samples", type=int, default=100, help="K ensemble members")
    p.add_argument("--n-steps", type=int, default=100, help="reverse discretization steps")
    p.add_argument("--eta", type=float, default=1.0, help="0=deterministic DDIM, 1=ancestral")
    p.add_argument("--batch-size", type=int, default=16, help="windows per batch")
    p.add_argument("--variogram-pairs", type=int, default=2000)
    p.add_argument("--max-windows", type=int, default=0, help="cap windows (smoke)")
    p.add_argument("--smoke", action="store_true", help="tiny preset for plumbing checks")
    args = p.parse_args()

    if args.smoke:
        args.n_samples = min(args.n_samples, 16)
        args.n_steps = min(args.n_steps, 25)
        args.max_windows = args.max_windows or 64

    bundle = load_eval_checkpoint(args.ckpt_dir, which=args.which)
    dataset = bundle.cfg["dataset"]
    print(
        f"[generate_samples] dataset={dataset} split={args.split} device={bundle.device}\n"
        f"  K={args.n_samples} n_steps={args.n_steps} eta={args.eta} "
        f"ckpt={args.ckpt_dir} ({args.which})",
        flush=True,
    )

    arrays = load_split_arrays(dataset, args.split, args.seed)
    y_pred, y_true, mask = arrays["y_pred"], arrays["y_true"], arrays["mask"]
    B_total, N, H = y_true.shape
    B = B_total if args.max_windows <= 0 else min(args.max_windows, B_total)
    print(f"[generate_samples] scoring {B}/{B_total} windows  (N={N} H={H})", flush=True)

    sampler = DiffusionSampler(
        bundle.denoiser, bundle.prior, bundle.schedule,
        n_samples=args.n_samples, n_steps=args.n_steps, eta=args.eta,
        device=bundle.device, seed=args.seed,
    )
    acc = MetricAccumulator(H=H, variogram_pairs=args.variogram_pairs, variogram_seed=args.seed)

    t0 = time.time()
    for start in range(0, B, args.batch_size):
        stop = min(start + args.batch_size, B)
        samples = sampler.sample(y_pred[start:stop])               # (K, b, N, H)
        acc.update(samples, y_true[start:stop], mask[start:stop])
        done = stop
        print(f"  [{done}/{B}]  ({time.time() - t0:.1f}s)", flush=True)

    metrics = acc.finalize()
    metrics["meta"] = {
        "dataset": dataset,
        "split": args.split,
        "method": "diffusion",
        "n_samples": args.n_samples,
        "n_steps": args.n_steps,
        "eta": args.eta,
        "windows_scored": int(B),
        "ckpt_dir": str(args.ckpt_dir),
        "which": args.which,
        "smoke": bool(args.smoke),
        "wall_seconds": round(time.time() - t0, 1),
    }

    # ── Sample bank: a few evenly-spaced windows kept in full, + one trace.
    bank_idx = np.linspace(0, B - 1, min(N_BANK, B), dtype=int)
    bank_samples, trace = sampler.sample(y_pred[bank_idx], return_trace=True)

    out_dir = ROOT / "results" / "eval" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / f"samples_{args.split}.npz",
        bank_idx=bank_idx,
        Y_sample=bank_samples.astype(np.float32),       # (K, n_bank, N, H)
        y_true=y_true[bank_idx].astype(np.float32),
        y_hat=y_pred[bank_idx].astype(np.float32),
        mask=mask[bank_idx],
        trace_t=np.asarray(trace.t),
        trace_Rt_norm=np.asarray(trace.R_t_mean_norm),
        trace_R0_norm=np.asarray(trace.R_hat_0_mean_norm),
    )
    print(
        f"[generate_samples] wrote {out_dir}/metrics_{args.split}.json "
        f"(overall CRPS={metrics['overall']['crps']:.4f}, "
        f"energy={metrics['overall']['energy_score']:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
