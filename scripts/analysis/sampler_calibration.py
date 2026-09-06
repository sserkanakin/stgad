r"""Can the sampler produce calibrated samples at all? No training involved.

WHY THIS EXISTS. The thesis reports that the structured prior under-disperses in every
paired seed. This script tests whether that is a property of the PRIOR or of the
SAMPLER, by removing every other source of error:

  * the "data" is drawn FROM the arm's own stationary law, so the model is exact;
  * the denoiser is the analytic Bayes rule for that law, so it is optimal;
  * therefore ANY deviation of the output spread from the target is the sampler's.

It measures two independent sampler defects.

D1 -- the ancestral step drops the posterior variance of x0. The update draws
     x' ~ N(mean, v_post) with v_post = s_d*s_p/(a_d^2 s_p + s_d), the variance of
     q(x'|x, x0) -- i.e. it treats x0_hat as KNOWN. Given x, the posterior over x0
     still has variance sigma_p^2 = 1/(a_c^2/s_c + 1/v), and that propagates through
     the mean. This is DDPM's sigma_t ambiguity (beta-tilde vs beta); the code takes
     the lower bound. Benign for isotropic noise, severe for a stiff spectrum. Under
     the model the exact reverse transition is N(a_d x, v(1-a_d^2)), and

         var = v_post + (v_post*a_p/s_p)^2 * sigma_p^2

     reproduces it to 6 decimals for every theta and step size tested.

D2 -- sampling stops at c_lo=0.02 while training goes to 8e-4. The variance-weighted
     un-erased fraction f(c) = sum_m var_m (1-e^{-2 theta_m c}) / sum_m var_m caps
     what any sampler can deliver, and depends on the theta SPREAD -- so one shared
     floor resolves the arms to very different fidelities.

usage:
  RESULTS_ROOT=$PWD/archive/20260902/results \
    python scripts/analysis/sampler_calibration.py --cells pems08f/gcn metrla/gwn
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from src.diffusion.forward_fcps import FCPSSchedule, prior_to_torch
from src.diffusion.prior import load_prior

ARMS = {
    "iso":  dict(prior="isotropic"),
    "kron": dict(prior="kronecker"),
}
TARGET_F = 1.3e-3          # the un-erased fraction isotropic already gets at 8e-4


def undenoised(prior, c):
    th, v = prior.theta.ravel(), prior.sigma2.ravel()
    return float((v * (1 - np.exp(-2 * th * c))).sum() / v.sum())


def matched_floor(prior, target=TARGET_F):
    """the terminal floor that leaves `target` of THIS prior's variance un-erased."""
    from scipy.optimize import brentq
    try:
        return float(brentq(lambda c: undenoised(prior, c) - target, 1e-14, 1.0))
    except ValueError:
        return 8e-4


def dispersion(prior, schedule, c_lo, n_steps, corrected, B=256, seed=0):
    """Var(sampled)/Var(target) with an exact model and a Bayes-optimal denoiser."""
    g = torch.Generator().manual_seed(seed)
    pt = prior_to_torch(prior, device=torch.device("cpu"), dtype=torch.float32)
    theta, s2 = pt.theta.unsqueeze(0), pt.sigma2_scale
    v = s2 / theta
    C = torch.randn(B, prior.N, prior.H, generator=g) * v.sqrt()      # start at stationarity
    grid = np.geomspace(schedule.c_0, c_lo, n_steps + 1)
    C0_hat = C
    for i in range(n_steps + 1):
        c = float(grid[i])
        a_c = torch.exp(-theta * c)
        s_c = torch.clamp(s2 * (1 - torch.exp(-2 * theta * c)) / theta, min=1e-20)
        C0_hat = a_c * v / (a_c * a_c * v + s_c) * C                  # Bayes-optimal denoiser
        if i == n_steps:
            break
        cp = float(grid[i + 1])
        a_d = torch.exp(-theta * (c - cp))
        s_d = torch.clamp(s2 * (1 - torch.exp(-2 * theta * (c - cp))) / theta, min=1e-20)
        a_p = torch.exp(-theta * cp)
        s_p = torch.clamp(s2 * (1 - torch.exp(-2 * theta * cp)) / theta, min=1e-20)
        v_post = s_d * s_p / (a_d * a_d * s_p + s_d)
        mean = v_post * (a_d * C / s_d + a_p * C0_hat / s_p)
        var = v_post
        if corrected:
            sig_p2 = 1.0 / (a_c * a_c / s_c + 1.0 / v)                # posterior var of x0
            var = v_post + (v_post * a_p / s_p) ** 2 * sig_p2
        C = mean + var.sqrt() * torch.randn(C.shape, generator=g)
    return float((C0_hat.pow(2).mean(0).sum() / v.squeeze(0).sum()).item())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cells", nargs="*", default=["pems08f/gcn", "metrla/gwn"])
    p.add_argument("--steps", nargs="*", type=int, default=[50, 200])
    p.add_argument("--out", default="results/analysis/sampler_calibration.json")
    args = p.parse_args()
    S = FCPSSchedule(c_min=0.1, c_0=7.0, alpha=4.0, T=1.0)

    rows = []
    for cell in args.cells:
        ds, bb = cell.split("/")
        print(f"\n########## {ds} / {bb} ##########")
        print(f"{'arm':6s}{'c_lo':>10s}{'ceiling':>9s}" +
              "".join(f"{'coded@'+str(n):>11s}{'corr@'+str(n):>11s}" for n in args.steps))
        for arm, kw in ARMS.items():
            pr = load_prior(ds, seed=42, backbone=bb, norm="pernode", **kw)
            for lab, c_lo in (("0.02", 0.02), ("8e-4", 8e-4), ("matched", matched_floor(pr))):
                ceil = 1.0 - undenoised(pr, c_lo)
                vals = []
                for n in args.steps:
                    vals += [dispersion(pr, S, c_lo, n, False),
                             dispersion(pr, S, c_lo, n, True)]
                print(f"{arm:6s}{c_lo:10.2e}{ceil:9.4f}" +
                      "".join(f"{x:11.4f}" for x in vals), flush=True)
                rows.append(dict(ds=ds, bb=bb, arm=arm, c_lo=c_lo, floor=lab, ceiling=ceil,
                                 theta_max=float(pr.theta.max()),
                                 **{f"{k}@{n}": x for (k, n), x in
                                    zip([(k, n) for n in args.steps for k in ("coded", "corr")], vals)}))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
