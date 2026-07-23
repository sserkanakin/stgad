"""Reproduce GAD Fig.1 (mean decay + covariance growth per eigenvalue) for our raw-signal
heat prior, and compare the CURRENT config against a GAD-faithful one — so we can pick γ,
the heat c_0 and the schedule 'one by one' by looking at the curves.

left  : mean decay      H_t = exp(-c̄_t · θ_i)                 (signal retained)
right : covariance grow (σ²/θ_i)(1 - exp(-2 c̄_t · θ_i))       (noise injected)
solid = FCPS schedule, dashed = ULS — exactly GAD Fig.1.

θ_i = λ_ST,i + γ  where λ_ST = ½(λ_S ⊕ λ_T) ∈ [0,2] (cartesian product graph).
The CURRENT prior then rescales θ_min→0.8 (×~20) which collapses the cascade; the
GAD-FAITHFUL config keeps θ = λ_ST+γ (no rescale) so θ∈[γ, 2+γ] like GAD.

usage: .venv/bin/python scripts/analysis/gad_forward.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                                       # noqa: E402
from src.diffusion.forward_fcps import FCPSSchedule, ULSchedule                   # noqa: E402
from src.data.adjacency import compute_normalized_laplacian, load_adjacency       # noqa: E402

OUT = paths.analysis_dir("raw_signal") if hasattr(paths, "analysis_dir") else ROOT / "results/analysis/raw_signal"
OUT = ROOT / "results/analysis/raw_signal"; OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["pems08f", "metrla"]


def lam_ST(ds, H=12):
    """cartesian product-graph eigenvalues λ_ST = ½(λ_S ⊕ λ_T) ∈ [0,2]."""
    lam_S, _ = np.linalg.eigh(compute_normalized_laplacian(np.asarray(load_adjacency(ds, repair=True))))
    A_T = np.zeros((H, H)); i = np.arange(H - 1); A_T[i, i + 1] = A_T[i + 1, i] = 1.0
    lam_T, _ = np.linalg.eigh(compute_normalized_laplacian(A_T))
    L = 0.5 * (np.clip(lam_S, 0, None)[:, None] + np.clip(lam_T, 0, None)[None, :])
    return np.sort(L.ravel())


def gad_panel(ax_l, ax_r, theta, sched_f, sched_u, sigma2, title):
    t = np.linspace(0, 1, 200)
    cf = sched_f.c_bar(t); cu = sched_u.c_bar(t)
    # 4 representative modes: smoothest, ~⅓, ~⅔, roughest
    idx = [0, len(theta) // 3, 2 * len(theta) // 3, len(theta) - 1]
    labs = ["λ₁ (smoothest)", "λ⅓", "λ⅔", "λ_N (roughest)"]
    cols = ["#0072B2", "#E69F00", "#7F3FBF", "#999999"]
    for m, lb, c in zip(idx, labs, cols):
        th = theta[m]
        ax_l.plot(t, np.exp(-cf * th), color=c, lw=2, label=f"FCPS {lb} θ={th:.2f}")
        ax_l.plot(t, np.exp(-cu * th), color=c, lw=1.3, ls="--")
        ax_r.plot(t, sigma2 / th * (1 - np.exp(-2 * cf * th)), color=c, lw=2)
        ax_r.plot(t, sigma2 / th * (1 - np.exp(-2 * cu * th)), color=c, lw=1.3, ls="--")
    ax_l.set_ylabel("mean decay  exp(−c̄ₜθ)"); ax_l.set_xlabel("t")
    ax_r.set_ylabel("cov growth  (σ²/θ)(1−e^{−2c̄ₜθ})"); ax_r.set_xlabel("t")
    ax_l.set_title(title + "  (solid=FCPS, dashed=ULS)", fontsize=9)
    ax_l.legend(fontsize=6.5, frameon=False); ax_l.grid(alpha=0.2); ax_r.grid(alpha=0.2)
    ax_l.set_ylim(-0.02, 1.02)


def main():
    for ds in DATASETS:
        L = lam_ST(ds)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{ds}: GAD Fig.1 forward analysis — current vs GAD-faithful prior", fontsize=12)

        # (a) CURRENT: θ=λ_ST+0.04 then rescale θ_min→0.8 (the ×20 that collapses it), c_0=7
        g0 = 0.04; th0 = L + g0; th0 = th0 * (0.8 / th0.min())
        f0 = FCPSSchedule(c_min=0.1, c_0=7.0, alpha=4.0)
        u0 = ULSchedule(slope=2.0 * 7.0)
        gad_panel(axes[0, 0], axes[0, 1], th0, f0, u0, 1.0,
                  f"CURRENT  θ∈[{th0.min():.1f},{th0.max():.0f}] stiff={th0.max()/th0.min():.0f}  c₀=7")

        # (b) GAD-FAITHFUL: θ=λ_ST+0.1, NO rescale → θ∈[0.1,2.1], c_0=25 (mixes θ_min: 2·25·0.1=5)
        g1 = 0.1; th1 = L + g1
        c0 = 25.0
        f1 = FCPSSchedule(c_min=0.1, c_0=c0, alpha=4.0)
        u1 = ULSchedule(slope=2.0 * c0)
        gad_panel(axes[1, 0], axes[1, 1], th1, f1, u1, 1.0,
                  f"GAD-FAITHFUL  θ∈[{th1.min():.2f},{th1.max():.2f}] stiff={th1.max()/th1.min():.0f}  c₀={c0:.0f}")
        fig.tight_layout()
        f = OUT / f"gad_forward_{ds}.png"; fig.savefig(f, dpi=150); plt.close(fig)
        print(f"wrote {f}")
        print(f"  {ds}: CURRENT θ median={np.median(th0):.1f} (collapse — most modes die at once)")
        print(f"        GAD-faithful θ median={np.median(th1):.2f} (graded), θ∈[{th1.min():.2f},{th1.max():.2f}]")


if __name__ == "__main__":
    main()
