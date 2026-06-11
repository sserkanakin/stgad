#!/usr/bin/env python
"""micro-benchmark the diffusion hot loops on whatever gpu we land on.

the point is a fair gpu-vs-gpu number: we time ONLY the compute (a warmup pass
then `iters` timed passes, each bracketed by torch.cuda.synchronize), never the
process startup / data load / checkpoint read that differ run to run. one
machine-greppable BENCH_RESULT line goes to stdout at the end.

  eval  mode: times one reverse_sample(B=gen_batch, n_steps) — the ddim sampling
              loop that dominates eval wallclock. needs a trained checkpoint
              (--exp must point at one). throughput = windows-equivalent / s.
  train mode: times one optimiser step at --batch-size — the training hot path.
              needs only the fitted prior + adjacency (no checkpoint), so it
              runs even before any diffusion model is trained.

conditioning / batches are synthetic random tensors: we are measuring kernels,
not accuracy, and the shapes (N, H) are what set the cost. tf32 is inherited
from eval_diffusion's import (matches production).

usage (normally launched by scripts/daic/bench_gpu.sbatch):
  python scripts/bench_diffusion.py --mode eval  --dataset metrla --exp 20260610 \
         --prior kronecker --seed 42 --gen-batch 2048 --n-steps 50
  python scripts/bench_diffusion.py --mode train --dataset metrla --seed 42 \
         --batch-size 32
"""
from __future__ import annotations

import argparse
import os
import time

import torch

# importing eval_diffusion also flips on allow_tf32 (same as production) and
# gives us the exact checkpoint loader; train_diffusion gives the exact step.
from eval_diffusion import load_run
from train_diffusion import train_step

from src.data.adjacency import load_adjacency
from src.diffusion.denoiser import R0Denoiser, num_parameters
from src.diffusion.forward_fcps import FCPSSchedule, prior_to_torch
from src.diffusion.prior import load_prior
from src.diffusion.reverse import reverse_sample


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_it(fn, *, warmup: int, iters: int) -> float:
    """run fn() `warmup` times (untimed), then `iters` times timed. returns
    mean seconds per call, with cuda fully synchronised around the timed block."""
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters


def _gpu_tags(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"gpu": device.type, "cap": "-", "cuda": "-"}
    name = torch.cuda.get_device_name(0).replace(" ", "_")
    major, minor = torch.cuda.get_device_capability(0)
    return {"gpu": name, "cap": f"{major}.{minor}", "cuda": torch.version.cuda or "-"}


def bench_eval(args, device, gen) -> dict:
    denoiser, prior_obj, schedule, cfg, _ = load_run(
        args.dataset, args.prior, args.seed, args.exp, device
    )
    N, H = prior_obj.N, prior_obj.H
    # synthetic conditioning at the batch size eval actually feeds reverse_sample
    # (chunk*k ≈ gen_batch). we time the sampler, not the data pipeline.
    Y = torch.randn(args.gen_batch, N, H, device=device)

    def once():
        reverse_sample(
            prior_obj, schedule, denoiser, Y,
            n_steps=args.n_steps, eta=1.0, R_T=None, generator=gen,
        )

    sec = _time_it(once, warmup=args.warmup, iters=args.iters)
    return {
        "params": num_parameters(denoiser),
        "elapsed_s": sec,
        "ms_per_step": sec / args.n_steps * 1000.0,
        "throughput": args.gen_batch / sec,   # windows-equivalent per second
        "N": N, "H": H,
    }


def bench_train(args, device, gen) -> dict:
    prior = load_prior(args.dataset, args.prior, args.seed)
    prior_t = prior_to_torch(prior, device=device, dtype=torch.float32)
    schedule = FCPSSchedule.for_prior(prior, c_min=args.c_min, alpha=args.alpha, T=1.0)
    A = load_adjacency(args.dataset)
    denoiser = R0Denoiser(
        N=prior.N, H=prior.H, adjacency=A, d_h=args.d_h,
        channel_multipliers=tuple(args.channel_multipliers),
        n_blocks=args.n_blocks, device=str(device),
    ).to(device)
    optim = torch.optim.Adam(denoiser.parameters(), lr=args.lr)
    B, N, H = args.batch_size, prior.N, prior.H
    # one fixed synthetic batch, fully observed; we measure the step, not IO.
    batch = {
        "R0": torch.randn(B, N, H, device=device),
        "Y_hat": torch.randn(B, N, H, device=device),
        "mask": torch.ones(B, N, H, device=device),
    }

    def once():
        optim.zero_grad(set_to_none=True)
        loss = train_step(
            batch, denoiser=denoiser, prior_t=prior_t, schedule=schedule,
            device=device, t_eps=args.t_eps, generator=gen,
        )
        loss.backward()
        optim.step()

    sec = _time_it(once, warmup=args.warmup, iters=args.iters)
    return {
        "params": num_parameters(denoiser),
        "elapsed_s": sec,
        "ms_per_step": sec * 1000.0,
        "throughput": B / sec,                # samples per second
        "N": N, "H": H,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("eval", "train"), required=True)
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems08f"), required=True)
    p.add_argument("--prior", choices=("kronecker", "isotropic"), default="kronecker")
    p.add_argument("--exp", default="dev", help="experiment tag; eval needs a trained run here")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    # eval knobs
    p.add_argument("--gen-batch", type=int, default=2048)
    p.add_argument("--n-steps", type=int, default=50)
    # train knobs (mirror train_diffusion defaults so the step is identical)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d-h", type=int, default=32)
    p.add_argument("--channel-multipliers", type=int, nargs="+", default=[1, 2])
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--c-min", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=7.0)
    p.add_argument("--t-eps", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_diffusion is gpu-only; no cuda device visible")

    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    tags = _gpu_tags(device)
    stack = os.environ.get("BENCH_STACK", "?")
    print(f"[bench] mode={args.mode} stack={stack} torch={torch.__version__} "
          f"gpu={tags['gpu']} cap={tags['cap']} cuda={tags['cuda']}", flush=True)

    res = bench_eval(args, device, gen) if args.mode == "eval" else bench_train(args, device, gen)

    # one machine-greppable line. submit_bench.sh / the slurm log get scraped for
    # "BENCH_RESULT" to build the side-by-side table.
    fields = {
        "mode": args.mode, "stack": stack, "torch": torch.__version__,
        "gpu": tags["gpu"], "cap": tags["cap"], "dataset": args.dataset,
        "params": res["params"], "N": res["N"], "H": res["H"],
        "warmup": args.warmup, "iters": args.iters,
    }
    if args.mode == "eval":
        fields.update({"gen_batch": args.gen_batch, "n_steps": args.n_steps})
    else:
        fields.update({"batch_size": args.batch_size})
    fields.update({
        "elapsed_s": f"{res['elapsed_s']:.4f}",
        "ms_per_step": f"{res['ms_per_step']:.3f}",
        "throughput": f"{res['throughput']:.2f}",
    })
    print("BENCH_RESULT " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)


if __name__ == "__main__":
    main()
