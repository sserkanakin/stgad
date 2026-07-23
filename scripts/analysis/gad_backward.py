"""Reproduce GAD Fig.2 (backward-process graph filter frequency response) for our prior,
to verify our BACKWARD process is GAD-faithful — not just the forward (Fig.1).

GAD Eq.12: the closed-form MMSE denoiser g* is a graph filter with response
    h(λ) = e^{(λ+γ)c̄} / (1 + ασ² · (λ/(λ+γ)) · (e^{2(λ+γ)c̄} − 1))
Fig.2 shows: large c̄ → low-pass (recover coarse structure first); small c̄ → toward all-pass
(reintroduce high-freq detail). left panel varies ασ² at fixed c̄; right varies c̄.

usage: .venv/bin/python scripts/analysis/gad_backward.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GAMMA = 0.1


def h_response(lam, cbar, asigma2, gamma=GAMMA):
    lg = lam + gamma
    num = np.exp(lg * cbar)
    den = 1.0 + asigma2 * (lam / lg) * (np.exp(2.0 * lg * cbar) - 1.0)
    return num / den


def main():
    OUT = ROOT / "results/analysis/raw_signal"; OUT.mkdir(parents=True, exist_ok=True)
    lam = np.linspace(0, 2, 400)                      # normalized-Laplacian range, like GAD
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # left: vary ασ² at fixed c̄ (GAD Fig.2 left)
    cbar0 = 1.0
    for a, c in zip([0.0, 0.01, 0.1, 1.0, 3.0], ["#111", "#0072B2", "#009E73", "#E69F00", "#D55E00"]):
        axes[0].plot(lam, h_response(lam, cbar0, a), color=c, lw=2, label=f"ασ²={a}")
    axes[0].set_title(f"backward filter h(λ) vs ασ²   (c̄={cbar0})", fontsize=10)
    axes[0].set_xlabel("graph frequency λ"); axes[0].set_ylabel("h(λ)")
    axes[0].legend(fontsize=8, frameon=False); axes[0].grid(alpha=0.2)

    # right: vary c̄ at fixed ασ² (GAD Fig.2 right) — the staged low-pass→detail denoising
    asig = 1.0
    for cb, c in zip([0.05, 0.2, 0.5, 1.0, 3.0], ["#D55E00", "#E69F00", "#009E73", "#0072B2", "#3B0F70"]):
        axes[1].plot(lam, h_response(lam, cb, asig), color=c, lw=2, label=f"c̄={cb}")
    axes[1].set_title(f"backward filter h(λ) vs c̄   (ασ²={asig})  — large c̄ low-pass, small c̄ detail", fontsize=9)
    axes[1].set_xlabel("graph frequency λ"); axes[1].set_ylabel("h(λ)")
    axes[1].legend(fontsize=8, frameon=False); axes[1].grid(alpha=0.2)

    fig.suptitle("GAD Fig.2 reproduced on our prior (γ=0.1): the backward process is a graph filter", fontsize=11)
    fig.tight_layout()
    f = OUT / "gad_backward_filter.png"; fig.savefig(f, dpi=150); plt.close(fig)
    print(f"wrote {f}")
    # numeric check: at large c̄ the filter is low-pass (h decreases with λ); at c̄→0 all-pass
    for cb in [0.05, 1.0, 3.0]:
        h = h_response(lam, cb, 1.0)
        print(f"  c̄={cb}: h(λ≈0)={h[1]:.3f}  h(λ=1)={h[200]:.3f}  h(λ=2)={h[-1]:.3f}  "
              f"{'LOW-PASS' if h[1] > h[-1] else 'high/all-pass'}")


if __name__ == "__main__":
    main()
