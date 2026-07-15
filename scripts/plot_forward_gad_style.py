"""GAD Fig.1, reproduced for OUR spatio-temporal product graph (GAD-ST).

two panels, exactly the paper's layout:
  left   mean decay  e^{-c̄_t λ_k}                       (H_t eigenvalues)
  right  cov growth  (σ²/λ_k)(1 - e^{-2 c̄_t λ_k})       (Σ_t eigenvalues)
for four modes of L_γ = L_ST + γI — the two smoothest and the two roughest —
under ULS (solid) and FCPS (dashed), same total heat c̄(T) = c_0 = 7.

usage: python scripts/plot_forward_gad_style.py --dataset metrla --out figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.diffusion.forward_fcps import FCPSSchedule, ULSchedule
from src.diffusion.prior import load_prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metrla")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()

    load_prior.cache_clear()
    p = load_prior(a.dataset, "kronecker", seed=42, backbone="raw", temporal="cartesian")
    # physical shifted eigenvalues λ_k = λ_ST + γ (the paper's convention: λ1 = γ, λmax ≤ 2+γ)
    phys = p.gamma / float(p.theta.min())
    lam = np.sort((p.theta * phys).ravel())
    K = lam.size
    picks = [(0, r"$\lambda_1$"), (1, r"$\lambda_2$"),
             (K - 2, r"$\lambda_{K-1}$"), (K - 1, r"$\lambda_K$")]
    sigma2 = p.sigma2_scale * phys        # σ² in physical units (Σ_stat = σ²/λ unchanged)

    fcps = FCPSSchedule()                  # c̄(T) = 7
    uls = ULSchedule(slope=2 * fcps.c_0)   # same total heat
    t = np.linspace(0, 1, 400)
    cb_f = np.asarray(fcps.c_bar(t))
    cb_u = np.asarray(uls.c_bar(t))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.2))
    cols = ["tab:blue", "tab:orange", "tab:purple", "tab:gray"]
    for (k, name), c in zip(picks, cols):
        lk = float(lam[k])
        # the invariant is c̄·θ_internal = (c̄/phys)·λ_physical — label in physical λ,
        # but the exponent uses the internal product (schedule units carry the 1/phys).
        eu, ef = cb_u * lk / phys, cb_f * lk / phys
        axL.plot(t, np.exp(-eu), color=c, lw=1.6, label=f"ULS {name}={lk:.3f}")
        axL.plot(t, np.exp(-ef), color=c, lw=1.6, ls="--", label=f"FCPS {name}={lk:.3f}")
        axR.plot(t, sigma2 / lk * (1 - np.exp(-2 * eu)), color=c, lw=1.6)
        axR.plot(t, sigma2 / lk * (1 - np.exp(-2 * ef)), color=c, lw=1.6, ls="--")
    axL.set_xlabel("$t$"); axL.set_ylabel(r"Mean Decay $e^{-\bar c_t \lambda_k}$")
    axR.set_xlabel("$t$")
    axR.set_ylabel(r"Cov. Growth $\frac{\sigma^2}{\lambda_k}(1-e^{-2\bar c_t \lambda_k})$")
    for ax in (axL, axR):
        ax.grid(alpha=0.25)
    axL.legend(fontsize=6.5, ncol=2, loc="upper right")
    fig.suptitle(f"{a.dataset} — GAD-ST forward: ULS (solid) vs FCPS (dashed), "
                 f"space-time product graph, γ={p.gamma:g}", fontsize=10)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pth = out / f"forward_gadstyle_{a.dataset}.png"
    fig.tight_layout(); fig.savefig(pth, dpi=140); plt.close(fig)
    print(f"[fig] {pth}")


if __name__ == "__main__":
    main()
