"""Verify the GAD forward AND reverse process on OUR spatio-temporal prior, with
numbers (not assertions), for the empirical and the cartesian Kronecker variants.

FORWARD  (the heat equation + scheduler):
  [A] stationary variance is exactly σ²/θ per mode                 (closed form)
  [B] the real forward sampler REACHES it: forward-noise real residuals to c̄_hi,
      compare per-mode variance to σ²/θ                            (rel err → 0)
  [C] the FCPS scheduler mixes the slowest mode: 2·θ_min·c₀ ≥ 5    (GAD's rule)
  [D] heat ordering: high-θ modes decay first, low-θ last          (e^{-θc̄})

REVERSE  (does the sampler invert the forward — independent of any learned net):
  [E] plug in the ANALYTIC optimal (Wiener) denoiser for synthetic Gaussian data
      whose per-mode variance = the REAL residual's per-mode variance, start from
      prior noise, and check the generated samples recover that variance profile.
      Run ancestral (few steps, no stability floor) and EM (needs n ≳ θ_max·c₀).
      This is the standard sampler-correctness test: optimal denoiser ⇒ the reverse
      SDE must reproduce the target distribution.

Also reproduces GAD's Fig. 1 (mean-decay + covariance-growth vs t) on our θ, under
FCPS vs the linear ULS schedule, so the scheduler's effect is visible.

usage: .venv/bin/python scripts/analysis/verify_process.py --dataset metrla \
         --backbone gwn --temporal empirical [--gamma 0.04]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior, _load_residuals_train                 # noqa: E402
from src.diffusion.forward_fcps import (FCPSSchedule, ULSchedule,                 # noqa: E402
                                        noise_level_bounds, sample_R_t_from_c_bar)
from src.diffusion.reverse import (reverse_sample_em, reverse_sample_ancestral,   # noqa: E402
                                   min_em_steps)
from src.utils import paths                                                       # noqa: E402

OK, NO = "PASS", "FAIL"
FIG = paths.analysis_dir("prior_schedule")


def _modevar(X, prior):
    """per-mode variance of a batch (B,N,H) in the prior's joint eigenbasis."""
    C = np.einsum("ki,bih,hj->bkj", prior.V_S.T, X, prior.V_T)
    return (C ** 2).mean(0)                                          # (N,H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metrla")
    ap.add_argument("--backbone", default="gwn")
    ap.add_argument("--temporal", default="empirical", choices=("empirical", "cartesian"))
    ap.add_argument("--gamma", type=float, default=None, help="spatial γ (default: variant default)")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    ds, bb = args.dataset, args.backbone
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    prior = load_prior(ds, "kronecker", seed=42, backbone=bb,
                       temporal=args.temporal, gamma=args.gamma)
    N, H = prior.N, prior.H
    theta = prior.theta
    th_min, th_max = float(theta.min()), float(theta.max())
    s2 = prior.sigma2_scale
    sched = FCPSSchedule()                                            # GAD-literal c_min=.1 c0=7 α=4
    c0 = sched.c_0
    c_lo, c_hi = noise_level_bounds(prior)

    print(f"=== {ds} / {bb} / temporal={args.temporal}  γ={prior.gamma:.3f} ===")
    print(f"    N={N} H={H}   θ: min {th_min:.3f}  max {th_max:.1f}  stiffness {th_max/th_min:.0f}")
    print(f"    σ²_scale={s2:.4g}   FCPS c₀={c0}  c_min={sched.c_min}  α={sched.alpha}")
    print(f"    noise levels: c̄_lo={c_lo:.4f}  c̄_hi={c_hi:.2f}\n")

    # ---- FORWARD ----
    # [A] stationary variance is σ²/θ by construction; confirm sample_prior matches.
    from src.diffusion.forward_fcps import sample_prior
    draws = sample_prior((4000,), prior, rng)
    v_draw, v_model = _modevar(draws, prior), (s2 / theta)
    a_err = float(np.abs(v_draw - v_model).mean() / v_model.mean())
    a_ok = a_err < 0.05
    print(f"[A] stationary var == σ²/θ (sampler vs closed form):  rel err {a_err:.3f}   {OK if a_ok else NO}")

    # [B] the real forward REACHES stationary at c̄_hi
    R, mask = _load_residuals_train(ds, 42, bb)                       # unit-variance space (prior's)
    R0 = R[np.linspace(0, len(R) - 1, 3000).astype(int)]
    Rt = sample_R_t_from_c_bar(R0, c_hi, prior, rng)
    v_fwd = _modevar(Rt, prior)
    b_err = float(np.abs(v_fwd - v_model).mean() / v_model.mean())
    b_ok = b_err < 0.1
    print(f"[B] forward(c̄_hi) reaches stationary:                 rel err {b_err:.3f}   {OK if b_ok else NO}")

    # [C] scheduler mixes the slowest mode
    mix = 2 * th_min * c0
    c_ok = mix >= 5.0
    print(f"[C] scheduler mixes slowest mode: 2·θ_min·c₀ = {mix:.1f} (need ≥5)   {OK if c_ok else NO}")

    # [D] heat ordering
    cb = 0.25 * c_hi
    o = np.argsort(theta.ravel())
    lo_ret = float(np.exp(-theta.ravel()[o[:max(1, len(o)//10)]] * cb).mean())
    hi_ret = float(np.exp(-theta.ravel()[o[-max(1, len(o)//10):]] * cb).mean())
    d_ok = lo_ret > 3 * hi_ret
    print(f"[D] heat ordering @c̄=0.25c̄_hi: low-θ keeps {lo_ret:.2f}, high-θ {hi_ret:.3f}   {OK if d_ok else NO}\n")

    # ---- REVERSE (oracle Wiener denoiser) ----
    # synthetic Gaussian data whose per-mode variance = the REAL residual's, so the
    # reverse must transport prior-noise -> this (different) profile. optimal denoiser
    # is the closed-form Wiener filter, so any error is the SAMPLER's, not a net's.
    B = args.batch
    v_data = torch.as_tensor(_modevar(R0, prior), dtype=torch.float32)           # (N,H) target
    theta_t = torch.as_tensor(theta, dtype=torch.float32)
    V_S = torch.as_tensor(prior.V_S, dtype=torch.float32)
    V_T = torch.as_tensor(prior.V_T, dtype=torch.float32)

    def oracle(R_t, t_batch, Y_hat):
        """E[x0 | x_c] for x0 ~ N(0, diag(v_data)) through the forward, per mode."""
        cbar = float(sched.c_bar(float(t_batch[0].item())))
        Hc = torch.exp(-theta_t * cbar)
        var_c = Hc * Hc * v_data + s2 * (1.0 - Hc * Hc) / theta_t
        wiener = Hc * v_data / torch.clamp(var_c, min=1e-20)                     # (N,H)
        C = torch.einsum("ki,bih,hj->bkj", V_S.T, R_t, V_T)
        C0 = wiener.unsqueeze(0) * C
        return torch.einsum("ik,bkh,hj->bij", V_S, C0, V_T.T)

    Y0 = torch.zeros(B, N, H)                                                    # unconditional
    vd = v_data.numpy()

    def rel_var_err(sample):
        vg = _modevar(sample.numpy(), prior)
        return float(np.abs(vg - vd).mean() / vd.mean())

    # irreducible batch-noise floor: even a PERFECT sample of B draws estimates the
    # N·H per-mode variances imperfectly. measure it so the sampler error is judged
    # against what's achievable, not against 0.
    fl = []
    for s in range(4):
        Zt = torch.randn(B, N, H, generator=torch.Generator().manual_seed(100 + s))
        Cz = v_data.sqrt().unsqueeze(0) * torch.einsum("ki,bih,hj->bkj", V_S.T, Zt, V_T)
        fl.append(rel_var_err(torch.einsum("ik,bkh,hj->bij", V_S, Cz, V_T.T)))
    floor_ref = float(np.mean(fl))
    print(f"[E] reverse recovers the residual's per-mode variance profile (oracle denoiser):")
    print(f"    (irreducible batch-noise floor at B={B}: {floor_ref:.3f}; PASS = err < 1.8× floor)")
    for n in (25, 50, 100):
        s, _ = reverse_sample_ancestral(prior, sched, oracle, Y0, n_steps=n,
                                        c_lo=c_lo, generator=torch.Generator().manual_seed(1))
        e = rel_var_err(s)
        print(f"    ancestral n={n:>4}:  rel var err {e:.3f}  ({e/floor_ref:.1f}× floor)   "
              f"{OK if e < 1.8 * floor_ref else NO}")

    floor = min_em_steps(prior, sched)
    print(f"    EM stability floor n ≳ θ_max·c₀ = {floor}")
    if floor <= 700:                                                            # cheap enough to run
        for n in (floor // 4, floor):
            s, _ = reverse_sample_em(prior, sched, oracle, Y0, n_steps=n,
                                     generator=torch.Generator().manual_seed(1))
            e = rel_var_err(s)
            tag = "NaN/diverged" if not np.isfinite(e) else f"rel var err {e:.3f}  ({e/floor_ref:.1f}× floor)"
            ok = np.isfinite(e) and e < 1.8 * floor_ref
            note = "(below floor → expect divergence)" if n < floor else ""
            print(f"    EM       n={n:>4}:  {tag}   {OK if ok else NO} {note}")
    else:
        print(f"    EM       n={floor}:  SKIP (too stiff for EM on CPU; this variant needs ancestral)")

    # ---- GAD Fig. 1 reproduction on our θ ----
    reps = np.unique(theta.ravel()[o[[0, len(o)//4, len(o)//2, 3*len(o)//4, -1]]])
    t = np.linspace(0, 1, 200)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    for th in reps:
        cbF = sched.c_bar(t)
        cbU = ULSchedule(slope=2 * c0).c_bar(t)
        axL.plot(t, np.exp(-th * cbF), "-", lw=1.8, label=f"θ={th:.2f}")
        axL.plot(t, np.exp(-th * cbU), ":", lw=1.2, color=axL.lines[-1].get_color())
        axR.plot(t, (s2 / th) * (1 - np.exp(-2 * th * cbF)), "-", lw=1.8)
        axR.plot(t, (s2 / th) * (1 - np.exp(-2 * th * cbU)), ":", lw=1.2, color=axR.lines[-1].get_color())
    axL.set_title("mean decay  e^(−c̄_t·θ)   (— FCPS,  ⋯ ULS)"); axL.set_xlabel("t"); axL.set_ylim(0, 1.02)
    axR.set_title("covariance growth  (σ²/θ)(1−e^(−2c̄_t·θ))"); axR.set_xlabel("t")
    axL.legend(fontsize=8); axL.grid(alpha=0.25); axR.grid(alpha=0.25)
    fig.suptitle(f"GAD Fig.1 on our prior — {ds}/{bb}/{args.temporal}: FCPS holds dominant modes "
                 f"longer (stiffness {th_max/th_min:.0f})", fontsize=11)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / f"gad_fig1_{ds}_{bb}_{args.temporal}.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)
    print(f"\n    wrote results/figures/gad_fig1_{ds}_{bb}_{args.temporal}.png")

    allok = all([a_ok, b_ok, c_ok, d_ok])
    print(f"\n{'FORWARD ALL PASS ✓' if allok else 'FORWARD NEEDS REVIEW ✗'}")


if __name__ == "__main__":
    main()
