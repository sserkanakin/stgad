"""Forward / stationary diagnostic across the backbone ladder and a γ sweep.

Checks, per (dataset, backbone, γ), whether the FCPS schedule + structured Kronecker prior
behave: does the forward REACH the stationary at c̄_hi, does the slow mode still MIX
(2·θ_min·c_0 ≥ 5), and — the thing to verify — do the modes CASCADE (not all collapse to 0
at once). Small γ keeps graph information (steeper Σ_S spectrum) but stiffens the prior.

usage: .venv/bin/python scripts/analysis/forward_grid.py [--gammas 0.1 0.2 0.8] [--datasets ...]
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior, _load_residuals_train, build_prior_from_R
from src.diffusion.forward_fcps import FCPSSchedule, noise_level_bounds, sample_R_t_from_c_bar
from src.utils import paths

VARIANTS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
BACKBONES = ["mlp", "gcn", "gwn"]


def _modevar(X, pr):
    C = np.einsum("ki,bih,hj->bkj", pr.V_S.T, X, pr.V_T, optimize=True)
    return (C ** 2).mean(0)


def diagnose(pr, R0, sched):
    """analytic (no sampling): forward-reach at c̄_hi + the mode cascade in TIME under the
    FCPS schedule (fraction of modes still holding signal at t=0.3/0.5/0.7/0.9)."""
    theta = pr.theta
    th_min, th_max = float(theta.min()), float(theta.max())
    c_lo, c_hi = noise_level_bounds(pr)
    v_emp = _modevar(R0, pr)                                     # per-mode residual var (N,H)
    # forward per-mode var at c̄_hi = H²·v_emp + σ²(1-H²); →σ² means it reached stationary
    H2 = np.exp(-2 * theta * c_hi)
    reach = float(np.abs(H2 * (v_emp - pr.sigma2)).mean() / pr.sigma2.mean())
    # cascade in t-space (FCPS holds c̄ low until late t): fraction of modes alive (H>0.1)
    alive = {t: float((np.exp(-float(sched.c_bar(t)) * theta) > 0.1).mean())
             for t in (0.3, 0.5, 0.7, 0.9)}
    em_floor = int(np.ceil(th_max * sched.c_0))                  # EM stability step count ≈ θ_max·c_0
    return dict(th_min=th_min, th_max=th_max, stiff=th_max / th_min,
                mix=2 * th_min * sched.c_0, reach=reach, alive=alive, em_floor=em_floor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", nargs="*", type=float, default=[0.1, 0.2, 0.8])
    ap.add_argument("--datasets", nargs="*", default=VARIANTS)
    ap.add_argument("--backbones", nargs="*", default=BACKBONES)
    ap.add_argument("--raw", action="store_true", help="also include the raw-signal rung")
    args = ap.parse_args()

    sched = FCPSSchedule()                                          # GAD-literal c_min=.1 c_0=7 α=4
    hdr = (f"{'dataset':9s} {'bb':5s} {'γ':>4s} | {'θ_max':>7s} {'stiff':>6s} {'mix':>5s} {'reach':>6s} | "
           f"{'alive t=.3/.5/.7/.9':>20s} | {'EM steps':>8s}")
    print(hdr); print("-" * len(hdr))

    def row(ds, bb, g, pr, R0):
        d = diagnose(pr, R0, sched)
        a = d["alive"]
        casc = f"{a[0.3]:.2f}/{a[0.5]:.2f}/{a[0.7]:.2f}/{a[0.9]:.2f}"
        print(f"{ds:9s} {bb:5s} {g:4.1f} | {d['th_max']:7.1f} {d['stiff']:6.0f} {d['mix']:5.1f} {d['reach']:6.3f} | "
              f"{casc:>20s} | {d['em_floor']:8d}")

    for ds in args.datasets:
        for bb in args.backbones:
            if not paths.residual_npz(ds, "train", 42, bb).exists():
                continue
            R, mask = _load_residuals_train(ds, 42, bb)
            R0 = R[np.linspace(0, len(R) - 1, 2000).astype(int)]
            for g in args.gammas:
                pr = load_prior(ds, "kronecker", seed=42, backbone=bb, spatial="structured",
                                temporal="empirical", gamma=g)
                row(ds, bb, g, pr, R0)
            print()
        if args.raw:
            d0 = np.load(paths.residual_npz(ds, "train", 42, "gwn"))
            Yt = (d0["y_pred"] + d0["residuals"]).astype(np.float64)
            m = d0["mask"].astype(bool)
            sub = np.linspace(0, len(Yt) - 1, 2000).astype(int)
            R0u = (Yt[sub] - Yt[m].mean()) / Yt[m].std()
            for g in args.gammas:
                pr, _ = build_prior_from_R(ds, Yt[sub], m[sub], spatial="structured", gamma=g)
                row(ds, "raw", g, pr, R0u)
            print()
    print("mix = 2·θ_min·c_0 (θ_min pinned to 0.8 by scale-norm, so this is ALWAYS 11.2 ≥ 5 —")
    print("     the slow mode mixes for every dataset/backbone/γ; the SCHEDULE needs no per-cell tuning).")
    print("reach = per-mode var rel-err at c̄_hi (→0 = forward reaches the stationary prior).")
    print("alive t=.. = fraction of modes still holding signal (H=e^(-c̄(t)θ) > 0.1) at that t under FCPS —")
    print("     a GRADED cascade decreases across t (modes die progressively), NOT all-to-0 at once.")
    print("EM steps ≈ θ_max·c_0 = the EM sampler's STABILITY floor. This is what varies per cell")
    print("     (and blows up for small γ / stiff backbones) — the real per-backbone hyperparameter.")


if __name__ == "__main__":
    main()
