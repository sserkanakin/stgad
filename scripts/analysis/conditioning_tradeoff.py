"""how much structure do we give up to bring the noising budget c₀ toward GAD's?

the stiff structured prior forces a large c₀ (= 5/θ_min) and a large end-rate c(T),
which makes the reverse sde stiff. two levers tame it:
  - raise γ  (the spatial shift): the loose smooth mode gets tighter, κ(Q_S) shrinks.
  - cap the joint stiffness: floor θ_min so θ_max/θ_min ≤ cap.
both lower c₀ — but both also pull the prior toward white noise. this script prints,
per setting, the conditioning (κ, c₀, end-rate) AND the fidelity (stage-0c mean KS to
val: does it still beat isotropic?), so the trade-off is explicit.

usage:  python scripts/analysis/conditioning_tradeoff.py --dataset metrla
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))      # reuse the stage-0 glue

import stage0_lib as s0                            # noqa: E402
from src.diffusion.forward_fcps import FCPSSchedule  # noqa: E402
from src.diffusion.prior import GAMMA_LOCKED, load_prior  # noqa: E402
from src.eval import signal_features as sf  # noqa: E402


def conditioning(prior):
    """(θ_min, κ(Q), c₀, end-rate c(T)) for a fitted prior under the schedule it forces."""
    sch = FCPSSchedule.for_prior(prior)
    th = prior.theta
    return float(th.min()), float(th.max() / th.min()), float(sch.c_0), float(sch.c(sch.T))


def mean_ks_to_val(prior, ops, val_feats, M, rng):
    """stage-0c fidelity: mean KS between this prior's feature cloud and val's (lower=closer)."""
    L, lam, V_S, deg = ops
    feats = sf.feature_matrix(s0.prior_fields(prior, M, rng), L, lam, V_S, deg)
    d = sf.compare_feature_sets(feats, val_feats)
    return float(np.mean([d[n]["ks"] for n in sf.FEATURE_NAMES]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metrla", choices=list(GAMMA_LOCKED))
    ap.add_argument("--M", type=int, default=2000)
    args = ap.parse_args()
    ds = args.dataset
    rng = np.random.default_rng(0)

    g_locked = GAMMA_LOCKED[ds]
    # (label, kwargs to load_prior). each row is one prior configuration.
    configs = [
        ("isotropic (baseline)",        dict(prior="isotropic")),
        ("temporal-flat (Σ_S=I)",       dict(prior="kronecker", spatial="flat")),
        (f"kron γ={g_locked:.2f} (locked)", dict(prior="kronecker", spatial="structured")),
        ("kron γ=0.50",                 dict(prior="kronecker", spatial="structured", gamma=0.50)),
        ("kron γ=0.80 (GAD-like)",      dict(prior="kronecker", spatial="structured", gamma=0.80)),
        ("kron locked + cap κ≤30",      dict(prior="kronecker", spatial="structured", stiffness_cap=30.0)),
        ("kron locked + cap κ≤13",      dict(prior="kronecker", spatial="structured", stiffness_cap=13.0)),
    ]

    val = s0.load_val_fields(ds)
    L, lam, V_S, deg = s0.graph_operators(ds)
    ops = (L, lam, V_S, deg)
    val_feats = sf.feature_matrix(val, L, lam, V_S, deg)

    iso_ks = None
    rows = []
    for label, kw in configs:
        p = load_prior(ds, seed=0, **kw)
        th_min, kappa, c0, cT = conditioning(p)
        ks = mean_ks_to_val(p, ops, val_feats, args.M, rng)
        if "isotropic" in label:
            iso_ks = ks
        rows.append((label, kappa, c0, cT, ks))

    print(f"\nconditioning vs fidelity trade-off — {ds}  (M={args.M} samples)\n")
    print(f"{'configuration':<28} {'κ(Q)':>8} {'c₀':>8} {'c(T)':>8} {'mean KS':>9}  vs iso")
    print("-" * 76)
    for label, kappa, c0, cT, ks in rows:
        verdict = "" if iso_ks is None else ("beats iso ✓" if ks < iso_ks - 1e-6
                                             else "= iso" if abs(ks - iso_ks) <= 1e-6
                                             else "worse ✗")
        print(f"{label:<28} {kappa:>8.1f} {c0:>8.1f} {cT:>8.0f} {ks:>9.3f}  {verdict}")
    print("\nGAD for reference:  κ≈3.5  c₀=7  c(T)≈35  (spatial-only, no temporal axis)")


if __name__ == "__main__":
    main()
