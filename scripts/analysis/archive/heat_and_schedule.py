"""Two supervisor questions, answered with numbers:

(A) "Check the heat equation for the Kronecker prior."
    The forward drift is Q. A true GRAPH HEAT equation needs Q to be a graph Laplacian, and
    then the space-time heat kernel FACTORISES: exp(-(Q_S⊕Q_T)c̄) = exp(-Q_S c̄) ⊗ exp(-Q_T c̄).
    That holds iff the joint precision θ[k,h] is ADDITIVE (θ = a_k + b_h).
      cartesian (product graph): θ = (λ_S+ω_T)/2 + γ   → ADDITIVE → true heat, factorises,
                                 stiffness ≈ spatial + temporal (GENTLE).
      empirical (Σ_T⊗Σ_S):      θ = (λ_S+γ)/(τ²μ_h)    → MULTIPLICATIVE (θ = a_k·b_h) →
                                 NOT a graph heat equation; valid OU but the heat kernel does
                                 NOT factorise, and stiffness ≈ spatial × temporal (VIOLENT).
    We test additivity via the mixed 2nd difference Δ = θ[k,h]-θ[k,0]-θ[0,h]+θ[0,0] (=0 iff additive).

(B) "May FCPS be changed for better generation?"
    Each mode 'transitions' (signal→noise) around c̄* = 1/θ. Good generation needs the reverse
    grid to spend its steps WHERE modes transition. We histogram the transition points 1/θ over
    all modes and overlay the schedule's step density (FCPS-forward and the geometric reverse
    grid). If the modes CLUSTER but the grid is uniform-in-log, steps are wasted at the empty
    ends → a spectrum-matched (mode-density) grid would generate better at the same step count.

usage: .venv/bin/python scripts/analysis/heat_and_schedule.py [--dataset pems08f --backbone gwn]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior                                        # noqa: E402
from src.diffusion.forward_fcps import FCPSSchedule, noise_level_bounds           # noqa: E402
from src.utils import paths                                                       # noqa: E402
FIG = paths.analysis_dir("prior_schedule")


def additivity(theta):
    """mixed 2nd difference Δ[k,h]=θ[k,h]-θ[k,0]-θ[0,h]+θ[0,0]; ~0 iff θ is additive a_k+b_h.
    return relative residual ‖Δ‖ / ‖θ-θ.mean‖ (0=perfectly additive, ~1=not)."""
    t = theta
    D = t - t[:, :1] - t[:1, :] + t[0, 0]
    return float(np.abs(D).mean() / (np.abs(t - t.mean()).mean() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pems08f")
    ap.add_argument("--backbone", default="gwn")
    args = ap.parse_args()
    ds, bb = args.dataset, args.backbone

    print(f"=== (A) heat equation for the Kronecker prior — {ds}/{bb} ===\n")
    pri = {"cartesian(γ=.8)": load_prior(ds, prior="kronecker", seed=42, backbone=bb,
                                         temporal="cartesian", gamma=0.8),
           "empirical": load_prior(ds, prior="kronecker", seed=42, backbone=bb, temporal="empirical")}
    for name, p in pri.items():
        th = p.theta
        add = additivity(th)                                  # additive (heat) test
        mul = additivity(np.log(th))                          # multiplicative test (additive in log)
        stiff = th.max() / th.min()
        kind = "ADDITIVE → true graph heat, kernel FACTORISES" if add < 0.05 else \
               ("MULTIPLICATIVE → NOT graph heat, kernel does NOT factorise" if mul < 0.05 else "neither")
        print(f"  {name:16s}: additivity Δ={add:.3f}  log-additivity Δ={mul:.3f}  stiffness={stiff:6.0f}")
        print(f"                    → θ is {kind}")
    print("\n  meaning: cartesian is a genuine space-time heat equation (spatial ⊕ temporal, gentle);")
    print("  empirical is a Kronecker-PRODUCT-precision OU — correct stationary prior, but the")
    print("  'diffusion rate' is spatial × temporal, which is why it is so stiff.\n")

    # (B) schedule vs mode transitions
    print(f"=== (B) may FCPS be changed for better generation? — {ds}/{bb} ===\n")
    sched = FCPSSchedule()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (name, p) in zip(axes, pri.items()):
        theta = p.theta.ravel()
        c_lo, c_hi = noise_level_bounds(p)
        trans = 1.0 / theta                                   # each mode's transition noise level
        # histogram of transition points (log c̄)
        ax.hist(np.log10(trans), bins=40, density=True, alpha=0.5, color="#2f6fc0",
                label="mode transitions  (1/θ)")
        # reverse grid density: geometric (uniform-in-log-c̄) → FLAT in log10 over [c_lo,c_hi]
        ax.axvspan(np.log10(c_lo), np.log10(c_hi), color="#d1495b", alpha=0.08)
        lo, hi = np.log10(c_lo), np.log10(c_hi)
        ax.plot([lo, hi], [1/(hi-lo)]*2, "--", color="#d1495b", lw=2,
                label="reverse grid (uniform in log c̄)")
        frac_in = float(((trans >= c_lo) & (trans <= c_hi)).mean())
        ax.set_title(f"{name}: {frac_in:.0%} of modes transition inside [c_lo,c_hi]", fontsize=11)
        ax.set_xlabel("log₁₀ noise level  c̄"); ax.set_ylabel("density"); ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        # quantify wasted resolution: entropy gap between the grid (uniform) and the mode density
        h, edges = np.histogram(np.log10(trans[(trans >= c_lo) & (trans <= c_hi)]), bins=20, density=True)
        h = h / (h.sum() + 1e-12)
        u = np.ones_like(h) / len(h)
        kl = float((h * np.log((h + 1e-12) / u)).sum())        # KL(mode-density ‖ uniform grid)
        print(f"  {name:16s}: {frac_in:.0%} of modes transition in-range; "
              f"mode-density vs uniform-grid KL = {kl:.2f} nats (0=matched, >0=grid wastes steps)")
    fig.suptitle(f"{ds}/{bb}: where modes transition (blue) vs where the reverse schedule spends steps "
                 "(red). Mismatch = wasted steps → a mode-density-matched grid generates better.",
                 fontsize=11)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / f"schedule_vs_modes_{ds}_{bb}.png", dpi=140, bbox_inches="tight")
    print(f"\n  wrote results/figures/schedule_vs_modes_{ds}_{bb}.png")
    print("\n  → concrete change: replace the uniform-in-log-c̄ reverse grid with one whose step")
    print("    density matches the mode-transition histogram (more steps in the crowded middle,")
    print("    fewer at the empty ends). same #steps, better coverage of the modes that matter.")


if __name__ == "__main__":
    main()
