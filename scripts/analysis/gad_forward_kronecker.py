"""GAD Fig.1 forward-process analysis for OUR KRONECKER PRIOR (the 'emp' arm).

The emp prior is the spatio-temporal Kronecker covariance
    Σ = Σ_T ⊗ Σ_S ,   Σ_S = (L + γ I)^{-1}  (structured, GAD graph, γ=0.8),
    Σ_T = D_H^{1/2} R_H D_H^{1/2}            (empirical horizon variance + correlation),
loaded verbatim from the production code (src/diffusion/prior.load_prior, temporal=
"empirical", spatial="structured") — the SAME object every emp test cell diffused with.
In the joint eigenbasis V_S ⊗ V_T the per-mode precision is θ[k,h] and the stationary
variance is σ²/θ[k,h], so the forward OU marginal is closed form (forward_fcps.py):
    H_t   = exp(-c̄_t · θ)                        (mean decay,  signal retained)
    Σ_t   = (σ²/θ)(1 - exp(-2 c̄_t · θ))          (covariance growth, noise injected)

Layout mirrors GAD (Rozada et al. 2025) Fig.1:
  panel 1  MEAN DECAY  H_t   per representative mode (smoothest θ_min, ~1/3, ~2/3,
           roughest θ_max), solid = FCPS schedule, dashed = ULS (linear) — exactly
           GAD Fig.1 left.
  panel 2  COVARIANCE GROWTH  Σ_t  for the same modes, solid=FCPS dashed=ULS — GAD
           Fig.1 right (each curve saturates at its stationary σ²/θ).
  panel 3  (verification, added) MONTE-CARLO forward SDE run on REAL training residuals:
           forward-noise them along the FCPS schedule and show two signal-domain
           convergence metrics to the stationary Kronecker prior — sliced-Wasserstein-1
           (prior-std units) and per-mode variance relative error — both → 0 as t→1,
           i.e. the marginal reaches N(0, Σ_T⊗Σ_S). This is the empirical counterpart of
           the closed-form left/right panels.

usage: .venv/bin/python scripts/analysis/gad_forward_kronecker.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior, _load_residuals_train                  # noqa: E402
from src.diffusion.forward_fcps import (FCPSSchedule, ULSchedule,                   # noqa: E402
                                        noise_level_bounds)

OUT = ROOT / "results/analysis/paper_results/figs"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["pems08f", "metrla"]
BACKBONE = "gwn"
SEED = 42

# GAD Fig.1 palette: 4 representative modes, blue/orange/green/red.
COLS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]


def _rotate_to_eig(X, prior):
    """(B,N,H) -> joint eigenbasis coeffs V_S^T X V_T, shape (B,N,H).
    optimize=True picks the pairwise path (V_S⊗V_T is orthonormal); the numpy default
    path is O(B·N²·H²) and pathological for large N."""
    return np.einsum("ki,bih,hj->bkj", prior.V_S.T, X, prior.V_T, optimize=True)


def sliced_w1(A, B, dirs):
    """mean over projection directions of the 1-D Wasserstein-1 distance between the
    rows of A and B (equal sample counts), each direction standardized by B's projected
    std so the result is in prior-std units. A,B are (n, d); dirs is (d, K)."""
    pa = A @ dirs                                    # (n, K)
    pb = B @ dirs
    sd = pb.std(axis=0) + 1e-12                       # prior scale per direction
    pa, pb = pa / sd, pb / sd
    pa.sort(axis=0); pb.sort(axis=0)
    return float(np.abs(pa - pb).mean())              # sorted 1-D W1, averaged over K


def forward_analysis(ds):
    prior = load_prior(ds, "kronecker", seed=SEED, backbone=BACKBONE,
                       spatial="structured", temporal="empirical")   # the emp arm
    N, H = prior.N, prior.H
    theta = prior.theta.ravel()
    s2 = prior.sigma2_scale
    th_min, th_max = float(theta.min()), float(theta.max())
    stiff = th_max / th_min

    fcps = FCPSSchedule()                             # GAD-literal c_min=.1, c_0=7, α=4
    c0 = fcps.c_0
    uls = ULSchedule(slope=2.0 * c0)                  # same total heat c̄(T)=c_0
    c_lo, c_hi = noise_level_bounds(prior)

    # ---- representative modes: smoothest, ~1/3, ~2/3, roughest (GAD picks 4) ----
    order = np.argsort(theta)
    idx = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]]
    labs = ["θ_min (smoothest)", "θ ~1/3", "θ ~2/3", "θ_max (roughest)"]
    rep_theta = theta[idx]

    t = np.linspace(0.0, 1.0, 300)
    cbF, cbU = fcps.c_bar(t), uls.c_bar(t)

    # ---- panel 3 data: MC forward on real residuals, done in the joint eigenbasis ----
    # V_S⊗V_T is orthonormal, so the forward marginal is per-mode independent there:
    #   C_t = e^{-c̄θ}·C_0 + sqrt((σ²/θ)(1-e^{-2c̄θ}))·Z ,  C_0 = V_S^T R_0 V_T.
    # per-mode variance err and (rotation-invariant) sliced-W1 are identical to signal
    # space but avoid rotating every noised batch back — the expensive step.
    R, mask = _load_residuals_train(ds, SEED, BACKBONE)              # unit-var space (prior's)
    sub = np.linspace(0, len(R) - 1, 2500).astype(int)
    B0 = len(sub)
    C0 = _rotate_to_eig(R[sub], prior).reshape(B0, N, H)            # real residuals, eigenbasis
    rng = np.random.default_rng(0)
    theta_m = prior.theta                                           # (N,H)
    v_model = (s2 / theta_m)                                        # (N,H) stationary var
    d = N * H
    dirs = rng.standard_normal((d, 256)); dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    prior_coeff = (np.sqrt(v_model)[None] * rng.standard_normal((B0, N, H))).reshape(B0, d)

    ts = np.linspace(0.0, 1.0, 13)
    w1s, verr = [], []
    for ti in ts:
        cb = float(fcps.c_bar(ti))
        Ht = np.exp(-theta_m * cb)
        s2t = s2 * (1.0 - np.exp(-2.0 * theta_m * cb)) / theta_m
        Ct = Ht[None] * C0 + np.sqrt(s2t)[None] * rng.standard_normal((B0, N, H))
        v_fwd = (Ct ** 2).mean(0)                                   # (N,H)
        verr.append(float(np.abs(v_fwd - v_model).mean() / v_model.mean()))
        w1s.append(sliced_w1(Ct.reshape(B0, d), prior_coeff, dirs))
    w1s, verr = np.array(w1s), np.array(verr)

    # ================= figure =================
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    axL, axM, axR = axes

    for m, lb, c in zip(idx, labs, COLS):
        th = theta[m]
        axL.plot(t, np.exp(-cbF * th), "-", color=c, lw=2.0, label=f"FCPS  {lb}  θ={th:.2f}")
        axL.plot(t, np.exp(-cbU * th), "--", color=c, lw=1.4)
        axM.plot(t, (s2 / th) * (1 - np.exp(-2 * cbF * th)), "-", color=c, lw=2.0)
        axM.plot(t, (s2 / th) * (1 - np.exp(-2 * cbU * th)), "--", color=c, lw=1.4)

    axL.set_title("Mean decay  $H_t=e^{-\\bar c_t\\theta}$   (— FCPS,  -- ULS)", fontsize=10)
    axL.set_xlabel("t"); axL.set_ylabel("signal retained"); axL.set_ylim(-0.02, 1.02)
    axL.legend(fontsize=7.5, frameon=False, loc="upper right"); axL.grid(alpha=0.25)

    axM.set_title("Covariance growth  $\\Sigma_t=(\\sigma^2/\\theta)(1-e^{-2\\bar c_t\\theta})$",
                  fontsize=10)
    axM.set_xlabel("t"); axM.set_ylabel("noise variance injected"); axM.grid(alpha=0.25)
    # dotted stationary levels σ²/θ each curve saturates to
    for m, c in zip(idx, COLS):
        axM.axhline(s2 / theta[m], color=c, ls=":", lw=0.9, alpha=0.6)

    axR.plot(ts, w1s, "-o", color="#0072B2", lw=2.0, ms=4, label="sliced-$W_1$ to prior (prior-std units)")
    axR.plot(ts, verr, "-s", color="#D55E00", lw=2.0, ms=4, label="per-mode variance rel. err.")
    axR.axhline(0.0, color="k", lw=0.7, alpha=0.4)
    axR.set_title("MC forward on real residuals → stationary $\\Sigma_T\\!\\otimes\\!\\Sigma_S$",
                  fontsize=10)
    axR.set_xlabel("t  (FCPS schedule)"); axR.set_ylabel("distance to stationary prior")
    axR.set_ylim(bottom=-0.03); axR.legend(fontsize=8, frameon=False); axR.grid(alpha=0.25)

    fig.suptitle(
        f"forward diffusion under the structured prior "
        f"$\\Sigma=\\Sigma_T\\otimes\\Sigma_S$   "
        f"($N$={N}, $H$={H}, joint precisions spanning "
        f"{th_min:.1f} to {th_max:.1f})",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = OUT / f"gad_forward_kronecker_{ds}.png"
    pdf = OUT / f"gad_forward_kronecker_{ds}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    nums = dict(
        ds=ds, N=N, H=H, th_min=th_min, th_max=th_max, stiff=stiff, s2=s2,
        c0=c0, c_lo=c_lo, c_hi=c_hi, mix=2 * th_min * c0,
        rep_theta=rep_theta,
        w1_t0=float(w1s[0]), w1_t1=float(w1s[-1]),
        verr_t0=float(verr[0]), verr_t1=float(verr[-1]),
    )
    return png, pdf, nums


def main():
    for ds in DATASETS:
        png, pdf, n = forward_analysis(ds)
        print(f"\n=== {n['ds']} : Kronecker prior (emp arm) forward analysis ===")
        print(f"  N={n['N']} H={n['H']}  joint modes={n['N']*n['H']}")
        print(f"  θ ∈ [{n['th_min']:.3f}, {n['th_max']:.2f}]  stiffness {n['stiff']:.0f}   "
              f"σ²_scale={n['s2']:.3f}")
        print(f"  representative θ (smoothest, 1/3, 2/3, roughest): "
              f"{', '.join(f'{v:.2f}' for v in n['rep_theta'])}")
        print(f"  FCPS c_0={n['c0']:.0f}  c̄_lo={n['c_lo']:.4f}  c̄_hi={n['c_hi']:.2f}  "
              f"mixing 2·θ_min·c_0={n['mix']:.1f} (GAD rule ≥5)")
        print(f"  MC forward convergence to stationary prior:")
        print(f"    sliced-W1  t=0: {n['w1_t0']:.3f}  ->  t=1: {n['w1_t1']:.3f}  "
              f"({100*(1-n['w1_t1']/n['w1_t0']):.0f}% closed)")
        print(f"    var rel err t=0: {n['verr_t0']:.3f} -> t=1: {n['verr_t1']:.3f}")
        print(f"  wrote {png}")
        print(f"  wrote {pdf}")


if __name__ == "__main__":
    main()
