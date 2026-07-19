"""Derive EVERY diffusion arm parameter from the prior spectrum — no defaults.

For each (dataset, backbone, prior arm) this prints/derives, from the fitted prior:
  θ_min, θ_max, stiffness        the per-mode decay rates (γ + normalization fix θ_min)
  σ²_scale                       the single noise-strength scalar (matched to data var)
  mix = 2·θ_min·c₀               GAD's slow-mode mixing number (must be ≥5)   [FCPS c₀=7]
  c̄_lo, c̄_hi                     the noise-level range (clean floor .. slowest-mode mix)
  EM floor = ⌈θ_max·c₀⌉          min Euler–Maruyama steps for stability
  sampler                        EM if floor small (GAD regime) else ancestral

The normalization in load_prior fixes θ_min → 0.8 for EVERY prior, so a fixed FCPS
c₀=7 gives mix = 11.2 everywhere (that is WHY c₀ is not per-dataset). What varies is
θ_max → the EM floor → the sampler choice. γ is stated per arm with the stiffness it
induces, so the choice is visible, not hidden.

usage: .venv/bin/python scripts/analysis/params.py
"""
import math
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior                                        # noqa: E402
from src.diffusion.forward_fcps import FCPSSchedule, noise_level_bounds           # noqa: E402
from src.diffusion.reverse import min_em_steps                                    # noqa: E402
from src.utils import paths                                                       # noqa: E402

VARIANTS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
BACKBONES = ["mlp", "gcn", "gwn"]
# arms as they'd run: iso (white, ULS) + the two structured Kronecker priors (FCPS).
# cartesian is stated at GAD's γ=0.8 (the EM-friendly, paper-faithful value), NOT the
# module's 0.04 default — chosen deliberately.
ARMS = [("iso", dict(prior="isotropic")),
        ("kronE", dict(prior="kronecker", temporal="empirical")),
        ("kronC(γ.8)", dict(prior="kronecker", temporal="cartesian", gamma=0.8))]


def row(ds, bb, arm, kw):
    pr = load_prior(ds, seed=42, backbone=bb, **kw)
    sched = FCPSSchedule()
    th = pr.theta
    thmin, thmax = float(th.min()), float(th.max())
    c_lo, c_hi = noise_level_bounds(pr)
    floor = min_em_steps(pr, sched)
    sampler = "EM" if floor <= 400 else "ancestral"
    return dict(ds=ds, bb=bb, arm=arm, N=pr.N, H=pr.H, gamma=pr.gamma,
                thmin=thmin, thmax=thmax, stiff=thmax / thmin,
                sig2=pr.sigma2_scale, mix=2 * thmin * sched.c_0,
                c_lo=c_lo, c_hi=c_hi, em_floor=floor, sampler=sampler)


def main():
    rows = []
    for ds in VARIANTS:
        for bb in BACKBONES:
            if not paths.residual_npz(ds, "train", 42, bb).exists():
                continue
            for arm, kw in ARMS:
                rows.append(row(ds, bb, arm, kw))

    hdr = (f"{'dataset':9s} {'bb':4s} {'arm':11s} | {'γ':>5s} {'θ_min':>5s} {'θ_max':>7s} "
           f"{'stiff':>6s} {'σ²':>7s} | {'mix':>5s} {'EMfloor':>7s} {'sampler':>9s}")
    print(hdr); print("-" * len(hdr))
    md = ["| dataset | bb | arm | γ | θ_min | θ_max | stiffness | σ² | mix (≥5) | EM floor | sampler |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|--:|:--|"]
    last = None
    for r in rows:
        if last and (r["ds"], r["bb"]) != last:
            print()
        last = (r["ds"], r["bb"])
        print(f"{r['ds']:9s} {r['bb']:4s} {r['arm']:11s} | {r['gamma']:5.2f} {r['thmin']:5.2f} "
              f"{r['thmax']:7.1f} {r['stiff']:6.0f} {r['sig2']:7.2f} | {r['mix']:5.1f} "
              f"{r['em_floor']:7d} {r['sampler']:>9s}")
        md.append(f"| {r['ds']} | {r['bb']} | {r['arm']} | {r['gamma']:.2f} | {r['thmin']:.2f} | "
                  f"{r['thmax']:.1f} | {r['stiff']:.0f} | {r['sig2']:.2f} | {r['mix']:.1f} | "
                  f"{r['em_floor']} | {r['sampler']} |")

    print("\nmix = 2·θ_min·c₀ with FCPS c₀=7 (≥5 required; =11.2 everywhere because the")
    print("normalization fixes θ_min→0.8). EM floor = ⌈θ_max·c₀⌉; sampler = EM if floor ≤400 else ancestral.")
    out = paths.analysis_dir("prior_schedule") / "arm_params.md"
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
