"""Explanatory figures + numbers for two things:

(1) NOISE SAMPLERS — what "log-uniform" vs "GAD power-law" actually are and why it
    matters. Both pick WHICH noise level c̄ each training step corrupts a sample to.
      GAD (default): c̄ = c₀·u^0.65,  u~U(0,1)   -> density ∝ c̄^0.54, biased to HIGH noise
      log-uniform  : c̄ = c_lo·(c_hi/c_lo)^u     -> density ∝ 1/c̄, even across decades
      uniform-in-t : t~U(0,1) then FCPS c̄(t)     -> what naive diffusion does
    A mode θ is "learnable" at levels where c̄·θ ~ O(1); because θ spans a wide range,
    the sampler decides which modes get trained. Left: the three densities. Right: the
    fraction of training mass landing in each mode's transition zone (c̄·θ∈[0.1,10]).

(2) RESHAPE / the Gaussian picture — the prior is a product of per-mode Gaussians whose
    WIDTH is the per-mode variance. White (iso) = all equal width; Laplacian (cartesian)
    = width ∝ 1/(λ+γ); empirical = data-fit temporal; RESHAPED = the residual's own
    per-mode variance. We overlay the residual's ACTUAL per-mode variance and report the
    per-mode KL(data‖prior) — the extra nats the denoiser must undo, which reshaping →0.

usage: .venv/bin/python scripts/analysis/explain.py [--dataset pems08f --backbone gwn]
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
from src.diffusion.prior import load_prior, _load_residuals_train                 # noqa: E402
from src.diffusion.forward_fcps import FCPSSchedule, noise_level_bounds           # noqa: E402
from src.utils import paths                                                       # noqa: E402
FIG_SCHED = paths.analysis_dir("prior_schedule")     # noise samplers
FIG_STRUCT = paths.analysis_dir("structure")         # reshape concept


def per_mode_var(R, prior):
    C = np.matmul(prior.V_S.T, R); C = np.matmul(C, prior.V_T)
    return (C ** 2).mean(0)                                          # (N,H)


def kl_data_prior(v_emp, v_model):
    """mean per-mode KL(N(0,v_emp)‖N(0,v_model)) = ½(r−1−ln r), r=v_emp/v_model. nats."""
    r = np.clip(v_emp, 1e-30, None) / np.clip(v_model, 1e-30, None)
    return float((0.5 * (r - 1.0 - np.log(r))).mean())


def samplers_fig(prior):
    c_lo, c_hi = noise_level_bounds(prior)
    c0 = FCPSSchedule().c_0
    u = np.random.default_rng(0).random(200000)
    gad = c0 * u ** 0.65                                             # GAD power-law
    logu = c_lo * (c_hi / c_lo) ** u                                # log-uniform
    sched = FCPSSchedule()
    t = u; unit = sched.c_bar(t)                                    # uniform-in-t → c̄
    theta = np.sort(prior.theta.ravel())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    lbins = np.linspace(-3, np.log10(c0), 45)                        # mass per decade of c̄
    for name, s, c in (("GAD  c̄=c₀·u^0.65", gad, "#d1495b"),
                       ("log-uniform", logu, "#2f6fc0"),
                       ("uniform-in-t → FCPS", unit, "#3a923a")):
        axL.hist(np.log10(np.clip(s, 1e-3, None)), bins=lbins, density=True, histtype="step",
                 lw=2, label=name, color=c)
    axL.set_xlabel("noise level  log₁₀ c̄"); axL.set_ylabel("mass per decade")
    axL.set_title("(1) which noise levels training visits  (GAD → high noise; log-u → even)")
    axL.legend(fontsize=8); axL.grid(alpha=0.25)

    # coverage per mode: fraction of draws with c̄·θ in the learnable band [0.1, 10]
    def coverage(s):
        return np.array([np.mean((s * th >= 0.1) & (s * th <= 10.0)) for th in theta])
    for name, s, c in (("GAD", gad, "#d1495b"), ("log-uniform", logu, "#2f6fc0"),
                       ("uniform-in-t", unit, "#3a923a")):
        axR.plot(theta, coverage(s), lw=2, label=name, color=c)
    axR.set_xscale("log"); axR.set_xlabel("mode decay rate  θ  (low=smooth, high=rough)")
    axR.set_ylabel("fraction of training in its transition band")
    axR.set_title("(2) which MODES get trained"); axR.legend(fontsize=8); axR.grid(alpha=0.25)
    fig.suptitle("Noise samplers: the schedule/sampler decides which graph modes the denoiser ever learns",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_SCHED / "noise_samplers.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {FIG_SCHED / 'noise_samplers.png'}")


def reshape_fig(ds, bb):
    R, M = _load_residuals_train(ds, 42, bb)
    R = np.where(M, R, 0.0)[np.linspace(0, len(_load_residuals_train(ds, 42, bb)[0]) - 1, 1500).astype(int)]
    priors = {"iso": load_prior(ds, prior="isotropic", seed=42, backbone=bb),
              "kronC (Laplacian γ=.8)": load_prior(ds, prior="kronecker", seed=42, backbone=bb,
                                                    temporal="cartesian", gamma=0.8),
              "kronE (empirical)": load_prior(ds, prior="kronecker", seed=42, backbone=bb,
                                              temporal="empirical")}
    # residual per-mode variance in the EMPIRICAL prior's basis (the reshape target)
    base = priors["kronE (empirical)"]
    v_res = np.sort(per_mode_var(R, base).ravel())[::-1]            # sorted high→low
    x = np.arange(len(v_res))

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    axT.plot(x, v_res, color="k", lw=2.5, label="residual ACTUAL (=reshaped target)")
    rows = []
    for name, pr in priors.items():
        v_model = np.sort(pr.sigma2.ravel())[::-1]
        axT.plot(x, v_model, lw=1.8, alpha=0.9, label=name)
        # KL in the prior's OWN basis (fair to each)
        kl = kl_data_prior(per_mode_var(R, pr), pr.sigma2)
        rows.append((name, kl))
    axT.set_yscale("log"); axT.set_xlabel("joint space×time mode (sorted by variance)")
    axT.set_ylabel("per-mode variance = noise width²  (log)")
    axT.set_title(f"{ds}/{bb}: the prior is a product of per-mode Gaussians; each curve is how WIDE "
                  f"the noise is per mode", fontsize=11)
    axT.legend(fontsize=9); axT.grid(alpha=0.25)

    # bottom: the actual Gaussians (bell curves) for a smooth, a mid, a rough mode
    picks = [x[2], x[len(x)//2], x[-3]]
    labels = ["smooth mode", "mid mode", "rough mode"]
    xs = np.linspace(-6, 6, 400)
    for j, (idx, lab) in enumerate(zip(picks, labels)):
        for name, pr in [("iso", priors["iso"]), ("kronE (empirical)", priors["kronE (empirical)"])]:
            sd = np.sqrt(np.sort(pr.sigma2.ravel())[::-1][idx])
            sd_res = np.sqrt(v_res[idx])
            axB.plot(xs, np.exp(-xs**2/(2*sd**2))/(sd*np.sqrt(2*np.pi)), lw=1.6,
                     label=f"{lab}: {name}" if j == 0 else None,
                     color="#2f6fc0" if name == "iso" else "#d1495b",
                     ls=["-", "--", ":"][j])
        axB.plot(xs, np.exp(-xs**2/(2*sd_res**2))/(sd_res*np.sqrt(2*np.pi)), lw=1.6, color="k",
                 ls=["-", "--", ":"][j], label=f"{lab}: residual" if j == 0 else None)
    axB.set_xlabel("noise value in one mode"); axB.set_ylabel("density")
    axB.set_title("the Gaussian per mode: iso (blue) uses one width everywhere; empirical (red) & "
                  "residual (black) differ per mode", fontsize=10)
    axB.legend(fontsize=7, ncol=3); axB.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG_STRUCT / f"reshape_concept_{ds}_{bb}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_STRUCT / f'reshape_concept_{ds}_{bb}.png'}")
    print(f"\nper-mode KL(residual ‖ prior)  [nats, lower=less for the denoiser to undo], {ds}/{bb}:")
    for name, kl in rows:
        print(f"    {name:26s}  KL = {kl:.4f}")
    print(f"    {'reshaped (v=v_emp)':26s}  KL = 0.0000   (matched by construction)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pems08f")
    ap.add_argument("--backbone", default="gwn")
    args = ap.parse_args()
    prior = load_prior(args.dataset, prior="kronecker", seed=42, backbone=args.backbone,
                       temporal="empirical")
    samplers_fig(prior)
    reshape_fig(args.dataset, args.backbone)


if __name__ == "__main__":
    main()
