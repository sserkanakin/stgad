"""Stage B mixing diagnostic for the GAD FCPS forward sampler.

For each dataset, fit the Kronecker prior, compute θ_min over the joint
precision eigenvalues, and verify that 2·θ_min·c̄(T) ≥ mix_target with the
default FCPSSchedule. Mean attenuation at t = T is exp(-c̄(T)·θ_min);
a target of 5 means the slowest mode retains < 1% of its initial signal.

Output: results/diffusion/mixing_diagnostic.json with per-dataset numbers
and a recommended c_0 (2× margin) for tuned per-dataset schedules.

Usage:
    python scripts/section3/mixing_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.diffusion.forward_fcps import FCPSSchedule
from src.diffusion.prior import load_prior

DATASETS = ("metrla", "pemsbay", "pems08f")
PASS_THRESHOLD = 5.0
OUT_PATH = ROOT / "results" / "diffusion" / "mixing_diagnostic.json"


def main() -> int:
    schedule = FCPSSchedule()
    c_bar_T = float(schedule.c_bar(schedule.T))

    summary = {
        "schedule": {
            "type": "FCPS (polynomial, GAD)",
            "c_min": schedule.c_min,
            "c_0": schedule.c_0,
            "alpha": schedule.alpha,
            "T": schedule.T,
            "c_bar_T": c_bar_T,
        },
        "pass_threshold": PASS_THRESHOLD,
        "datasets": {},
    }
    all_pass = True
    for ds in DATASETS:
        prior = load_prior(ds)
        theta_min = float(prior.theta.min())
        theta_max = float(prior.theta.max())
        # Variance saturation requires 2·c̄(T)·θ_min ≥ 5; equivalently mean attenuation
        # exp(-c̄_T·θ_min) sets the dominant scale. Use the stricter mean-attenuation check.
        mix_mean = theta_min * c_bar_T
        passed = mix_mean >= PASS_THRESHOLD
        all_pass &= passed
        # Recommended c_0 = c̄(T) for a 2× margin over threshold.
        c_0_recommended = 2.0 * PASS_THRESHOLD / theta_min
        summary["datasets"][ds] = {
            "N": prior.N,
            "H": prior.H,
            "gamma": prior.gamma,
            "theta_min": theta_min,
            "theta_max": theta_max,
            "theta_condition": theta_max / theta_min,
            "mix_score_mean_attenuation": mix_mean,
            "mix_score_variance_saturation": 2.0 * mix_mean,
            "passed": passed,
            "c_0_recommended_2x_margin": c_0_recommended,
        }
        print(
            f"{ds:8s}  θ_min={theta_min:.4f}  θ_max={theta_max:.2f}  "
            f"cond(θ)={theta_max/theta_min:.1f}  "
            f"c̄_T·θ_min={mix_mean:.2f}  "
            f"c_0_rec={c_0_recommended:.0f}  "
            f"{'PASS' if passed else 'FAIL'}"
        )
    summary["all_pass"] = all_pass

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)}")
    print(f"overall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
