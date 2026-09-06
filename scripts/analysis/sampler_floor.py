r"""How much of each prior is still un-denoised where the ancestral sampler stops.

WHY THIS EXISTS. reverse_sample_ancestral runs a geometric c-bar grid from c_0 down
to a terminal floor c_lo (default 0.02) and returns the denoiser's R0_hat there. Any
noise still present at c_lo is therefore never resolved -- the returned ensemble is a
posterior MEAN, so it under-disperses by exactly the conditional variance left over.

For the per-mode OU forward, the fraction of the prior's variance still un-erased at
c-bar is variance-weighted over modes:

    f(c) = sum_m var_m (1 - exp(-2 theta_m c)) / sum_m var_m

theta is CONSTANT for the isotropic control (0.8 by construction) and spans decades
for a structured prior, so one shared c_lo resolves the two arms to very different
fidelities. Training draws go down to 8e-4, so that -- not 0.02 -- is the floor the
denoiser was actually trained to.

usage: python scripts/analysis/sampler_floor.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from src.diffusion.prior import load_prior

DS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
BB = ["mlp", "gcn", "gwn"]
ARMS = {
    "iso":  dict(prior="isotropic"),
    "kron": dict(prior="kronecker"),
}
C_LOCKED, C_TRAIN = 0.02, 8e-4


def undenoised(prior, c):
    theta, var = prior.theta.ravel(), prior.sigma2.ravel()
    return float((var * (1 - np.exp(-2 * theta * c))).sum() / var.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/analysis/sampler_floor.json")
    p.add_argument("--datasets", nargs="*", default=DS)
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    args = p.parse_args()

    rows = []
    print(f"{'ds':9s}{'bb':5s}{'arm':5s}{'f(0.02)':>9s}{'f(8e-4)':>9s}{'theta_max':>11s}")
    for ds in args.datasets:
        for bb in BB:
            for arm in args.arms:
                try:
                    pr = load_prior(ds, seed=42, backbone=bb, norm="pernode", **ARMS[arm])
                except FileNotFoundError as e:
                    print(f"skip {ds}/{bb}/{arm}: {e}", flush=True)
                    continue
                r = dict(ds=ds, bb=bb, arm=arm,
                         f_locked=undenoised(pr, C_LOCKED),
                         f_train=undenoised(pr, C_TRAIN),
                         theta_max=float(pr.theta.max()),
                         theta_min=float(pr.theta.min()))
                rows.append(r)
                print(f"{ds:9s}{bb:5s}{arm:5s}{r['f_locked']:9.4f}{r['f_train']:9.4f}"
                      f"{r['theta_max']:11.1f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
