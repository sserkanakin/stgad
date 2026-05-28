"""Reproduce GAD (Tan et al. 2024) Figure 1: ULS vs FCPS time-warping effect.

For four representative spatial eigenvalues λ_i (of L_γ = L + γI) from a real
dataset, plot the mean-decay  exp(-c̄_t · λ_i)  (top row) and the eigenbasis
covariance growth  (1/λ_i) · (1 - exp(-2 c̄_t · λ_i))  (bottom row), under
both Uniform Linear (ULS) and Floor-Constrained Polynomial (FCPS) schedulers.

Output: results/diffusion/fcps_schedule_figure.png.

Usage:
    python scripts/section3/plot_fcps_schedule.py --dataset metrla
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.diffusion.forward_fcps import FCPSSchedule, ULSchedule
from src.diffusion.prior import load_prior


def pick_eigenvalues(lam_gamma: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Pick 4 representative eigenvalues mirroring GAD's choices (low / low+1 / N-1 / N)."""
    n = lam_gamma.shape[0]
    idx = [0, 1, n - 2, n - 1]
    labels = [
        f"λ_1 = {lam_gamma[0]:.3f}",
        f"λ_2 = {lam_gamma[1]:.3f}",
        f"λ_{{N-1}} = {lam_gamma[-2]:.3f}",
        f"λ_N = {lam_gamma[-1]:.3f}",
    ]
    return lam_gamma[idx], labels


def main(dataset: str, c_0_viz: float, alpha: float) -> int:
    prior = load_prior(dataset)
    lam_gamma = prior.lam_S + prior.gamma  # spatial eigenvalues of L_γ

    # Use a visualization-tuned c_0 so the qualitative ULS-vs-FCPS contrast is
    # visible across the four eigenvalues. The production c_0 (default 20000
    # in FCPSSchedule) is much larger and would saturate every curve almost
    # immediately at t = 0.
    fcps = FCPSSchedule(c_min=0.05, c_0=c_0_viz, alpha=alpha, T=1.0)
    uls = ULSchedule(slope=2.0 * fcps.c_0, T=fcps.T)

    t = np.linspace(0.0, fcps.T, 401)
    c_bar_fcps = fcps.c_bar(t)
    c_bar_uls = uls.c_bar(t)

    lams_plot, labels = pick_eigenvalues(lam_gamma)

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.6), sharex=True)
    for col, (lam, label) in enumerate(zip(lams_plot, labels)):
        # Mean decay: exp(-c̄_t · λ)
        mean_uls = np.exp(-c_bar_uls * lam)
        mean_fcps = np.exp(-c_bar_fcps * lam)
        # Eigenbasis cov: (1/λ) · (1 - exp(-2 c̄_t · λ))
        cov_uls = (1.0 / lam) * (1.0 - np.exp(-2.0 * c_bar_uls * lam))
        cov_fcps = (1.0 / lam) * (1.0 - np.exp(-2.0 * c_bar_fcps * lam))

        ax_m, ax_c = axes[0, col], axes[1, col]

        ax_m.plot(t, mean_uls, color="tab:blue", linestyle="--", lw=1.6, label="ULS")
        ax_m.plot(t, mean_fcps, color="tab:red", lw=1.8, label="FCPS")
        ax_m.set_title(label, fontsize=10)
        ax_m.set_ylim(-0.02, 1.05)
        ax_m.grid(True, alpha=0.3)
        if col == 0:
            ax_m.set_ylabel(r"Mean decay  $e^{-\bar c_t \lambda_i}$")
        ax_m.legend(loc="upper right", fontsize=8, frameon=False)

        ax_c.plot(t, cov_uls, color="tab:blue", linestyle="--", lw=1.6, label="ULS")
        ax_c.plot(t, cov_fcps, color="tab:red", lw=1.8, label="FCPS")
        ax_c.grid(True, alpha=0.3)
        if col == 0:
            ax_c.set_ylabel(
                r"Cov. growth  $\frac{1}{\lambda_i}(1 - e^{-2\bar c_t \lambda_i})$"
            )
        ax_c.set_xlabel("t")

    fig.suptitle(
        f"FCPS vs ULS — {dataset} (γ={prior.gamma:.3f}, "
        f"FCPS c₀={fcps.c_0:.0f}, α={fcps.alpha}, "
        f"ULS c̄(T)={uls.c_bar(uls.T):.0f})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = ROOT / "results" / "diffusion" / "fcps_schedule_figure.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems08f"), default="metrla")
    p.add_argument("--c-0-viz", type=float, default=10.0,
                   help="c_0 used for the figure (visualization-tuned, not for training)")
    p.add_argument("--alpha", type=float, default=7.0)
    args = p.parse_args()
    sys.exit(main(args.dataset, args.c_0_viz, args.alpha))
