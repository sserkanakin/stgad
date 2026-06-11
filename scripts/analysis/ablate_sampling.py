"""ground the reverse-sampler choices (n_steps, K) on one trained checkpoint.

reuses eval_diffusion.evaluate() on a SUBSET of test windows, sweeping n_steps
(fixed K) then K (fixed n_steps), printing metrics vs each so we can pick the
plateau knee. eta is fixed at 1.0 (DDPM reverse sde). writes a markdown table.

    python scripts/analysis/ablate_sampling.py --dataset metrla --prior kronecker \
        --seed 42 --exp 20260612 --max-windows 512
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# repo root on sys.path so `scripts` resolves as a namespace package (src is
# installed, scripts is not — run as a file, sys.path[0] is scripts/analysis/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from scripts.eval_diffusion import evaluate
from src.utils import paths


def _row(tag, val, res):
    return (f"| {tag} {val:>4} | {res['mae']:.3f} | {res['winkler']:.2f} | "
            f"{res['coverage']:.3f} | {res['crps']:.3f} | {res['w1']:.3f} | {res['energy']:.3f} |")


def run(dataset, prior, seed, exp, max_windows, steps_list, k_list,
        k_fixed, steps_fixed, eta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = dict(dataset=dataset, prior=prior, seed=seed, exp=exp, device=device,
                  eta=eta, alpha=0.1, w1_windows=min(512, max_windows), w1_samples=4,
                  w1_proj=128, gen_batch=2048, max_windows=max_windows)
    hdr = "| setting | MAE | Winkler | coverage | CRPS | W1 | energy |\n|---|---|---|---|---|---|---|"

    out = {"dataset": dataset, "prior": prior, "seed": seed, "exp": exp,
           "max_windows": max_windows, "eta": eta, "steps_sweep": [], "k_sweep": []}

    print(f"\n### n_steps sweep (K={k_fixed}, {max_windows} windows, eta={eta})\n{hdr}")
    for ns in steps_list:
        g = torch.Generator(device=device).manual_seed(0)
        r = evaluate(**common, k=k_fixed, n_steps=ns, generator=g)
        print(_row("steps", ns, r), flush=True)
        out["steps_sweep"].append({"n_steps": ns, **{m: r[m] for m in
                                  ("mae", "winkler", "coverage", "crps", "w1", "energy")}})

    print(f"\n### K sweep (n_steps={steps_fixed}, {max_windows} windows, eta={eta})\n{hdr}")
    for k in k_list:
        g = torch.Generator(device=device).manual_seed(0)
        r = evaluate(**common, k=k, n_steps=steps_fixed, generator=g)
        print(_row("K", k, r), flush=True)
        out["k_sweep"].append({"k": k, **{m: r[m] for m in
                              ("mae", "winkler", "coverage", "crps", "w1", "energy")}})

    dest = Path("results") / "analysis" / f"ablation_{dataset}_{prior}_seed{seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--prior", default="kronecker")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp", default="20260612")
    p.add_argument("--max-windows", type=int, default=512)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--steps-list", type=int, nargs="+", default=[20, 50, 100, 200, 400])
    p.add_argument("--k-list", type=int, nargs="+", default=[10, 20, 30, 50, 100])
    p.add_argument("--k-fixed", type=int, default=30)
    p.add_argument("--steps-fixed", type=int, default=100)
    a = p.parse_args()
    run(a.dataset, a.prior, a.seed, a.exp, a.max_windows, a.steps_list, a.k_list,
        a.k_fixed, a.steps_fixed, a.eta)
