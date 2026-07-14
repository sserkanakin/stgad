"""pre-flight correctness checks for the raw-signal GAD diffusion arm (backbone='raw').

runs BEFORE any training. verifies, per dataset and per prior (kronecker/isotropic):
  [A] grounding  — the centered target is zero-mean unit-scale, and the prior's
                   marginal variance field matches the data's marginal variance.
  [B] stationary — drawing from the prior N(0,Σ) reproduces the data's per-horizon
                   variance and graph-spectral energy (the prior IS the data's law at c̄→∞).
  [C] forward    — running the forward SDE to c̄_hi washes the signal out and lands
                   on the stationary covariance (H→0, Var→prior).
  [D] schedule   — c_0/θ_min mix the slowest mode (2·c_0·θ_min ≥ 5); c̄ bounds sane.
  [E] graph      — normalized-Laplacian spectrum in [0,2], single zero mode (connected),
                   eigenvectors orthonormal.

usage:  python scripts/validate_signal_prior.py --datasets metrla pems08f --priors kronecker isotropic
"""
from __future__ import annotations

import argparse
import numpy as np

from src.diffusion.prior import load_prior
from src.diffusion.forward_fcps import (
    FCPSSchedule, ULSchedule, sample_prior, sample_R_t_from_c_bar, noise_level_bounds,
)
from src.data.residual_dataset import train_residual_sigma
from src.data.adjacency import load_adjacency, compute_normalized_laplacian
from src.utils import paths

OK = "\033[92mPASS\033[0m"
NO = "\033[91mFAIL\033[0m"


def _load_R0(ds, split="train"):
    d = np.load(paths.residual_npz(ds, split, 42, "raw"))
    sr = float(np.asarray(train_residual_sigma(ds, 42, "raw", norm="global")))
    return d["residuals"].astype(np.float64) / sr, d["mask"].astype(bool)


def _marginal_var_field(prior):
    """prior's marginal variance at each (node, horizon) from per-mode variance:
    Var[R_{i,h}] = Σ_{k,l} V_S[i,k]^2 · σ²[k,l] · V_T[h,l]^2 = (V_S∘V_S) σ² (V_T∘V_T)^T."""
    s2 = prior.sigma2                                   # (N, H) per-mode stationary var
    return (prior.V_S ** 2) @ s2 @ (prior.V_T ** 2).T   # (N, H) marginal var per (i,h)


def check(ds, prior_name, gamma=0.04, verbose=True):
    load_prior.cache_clear()
    prior = load_prior(ds, prior_name, seed=42, backbone="raw", gamma=gamma)
    R0, M = _load_R0(ds)
    B, N, H = R0.shape
    v = R0[M]
    results = {}

    # [A] grounding: zero-mean unit-scale + marginal-variance match
    mean0 = abs(float(v.mean()))
    e2 = float((v ** 2).mean())
    data_var = np.stack([ (R0[:, :, h][M[:, :, h]] ** 2).mean() for h in range(H) ])  # per-horizon
    prior_var_field = _marginal_var_field(prior)
    prior_var_h = prior_var_field.mean(0)                                             # per-horizon
    var_match = float(np.abs(prior_var_h - data_var).mean() / data_var.mean())
    results["A grounding"] = (mean0 < 0.02 and abs(e2 - 1) < 0.05 and var_match < 0.15,
                              f"mean={mean0:.4f} E[R0²]={e2:.3f} per-horizon var-match err={var_match:.3f}")

    # [B] stationary: prior samples reproduce data per-horizon variance
    rng = np.random.default_rng(0)
    S = sample_prior((4000,), prior, rng)                     # (4000, N, H)
    samp_var_h = (S ** 2).mean((0, 1))
    b_err = float(np.abs(samp_var_h - data_var).mean() / data_var.mean())
    # graph-spectral energy match (low-λ fraction)
    lam, Vs = np.linalg.eigh(compute_normalized_laplacian(np.asarray(load_adjacency(ds, repair=True))))
    def lowfrac(X):
        C = np.einsum("ki,bih->bkh", Vs.T, X); g = (C ** 2).mean((0, 2))
        return float(g[np.argsort(lam) <= N // 3].sum() / g.sum())
    lf_data = lowfrac(R0[M.reshape(B, -1).all(1)][:4000]) if M.reshape(B, -1).all(1).sum() > 100 else lowfrac(np.nan_to_num(R0)[:4000])
    lf_prior = lowfrac(S)
    # for the structured prior, its spatial spectrum should track the data's (low-λ energy
    # within ~0.1); the isotropic prior is white by construction (no spectral-match gate).
    lf_ok = abs(lf_prior - lf_data) < 0.12 if prior_name == "kronecker" else True
    results["B stationary"] = (b_err < 0.15 and lf_ok,
                               f"per-horizon var err={b_err:.3f}  low-λ energy data={lf_data:.2f} prior={lf_prior:.2f}"
                               f"{'' if lf_ok else '  <-- spectral MISMATCH'}")

    # [C] forward → stationary at c̄_hi: signal washed, variance → prior
    c_lo, c_hi = noise_level_bounds(prior)
    idx = M.reshape(B, -1).all(1)
    R0v = R0[idx][:2000] if idx.sum() > 100 else np.nan_to_num(R0)[:2000]
    Rt = sample_R_t_from_c_bar(R0v, c_hi, prior, rng)
    fwd_var_h = (Rt ** 2).mean((0, 1))
    c_err = float(np.abs(fwd_var_h - samp_var_h).mean() / samp_var_h.mean())
    H_slow = float(np.exp(-prior.theta.min() * c_hi))        # slowest-mode signal retained
    results["C forward"] = (c_err < 0.15 and H_slow < 0.05,
                            f"Var(c̄_hi)/prior err={c_err:.3f}  slow-mode H(c̄_hi)={H_slow:.3f} (→0)")

    # [D] schedule / c: slowest mode mixes, bounds sane
    sched = FCPSSchedule() if prior_name == "kronecker" else ULSchedule()
    theta_min = float(prior.theta.min()); theta_max = float(prior.theta.max())
    mix = 2.0 * sched.c_0 * theta_min                        # GAD mixing of slowest mode
    results["D schedule"] = (mix >= 5.0 and c_hi * theta_min >= 4.0,
                             f"c_0={sched.c_0:.1f} θ_min={theta_min:.2f} θ_max={theta_max:.0f} "
                             f"2·c_0·θ_min={mix:.1f}(≥5)  c̄∈[{c_lo:.3f},{c_hi:.2f}]")

    # [E] graph spectrum
    lam_s = np.sort(lam)
    n_zero = int((lam_s < 1e-6).sum())
    ortho = float(np.abs(Vs.T @ Vs - np.eye(N)).max())
    results["E graph"] = (lam_s.min() > -1e-6 and lam_s.max() <= 2.0 + 1e-6 and n_zero == 1 and ortho < 1e-6,
                          f"λ∈[{lam_s.min():.3f},{lam_s.max():.3f}] zero-modes={n_zero}(→1 connected) orthoErr={ortho:.1e}")

    if verbose:
        allok = all(v[0] for v in results.values())
        print(f"\n=== {ds} · {prior_name} === {'ALL PASS' if allok else 'HAS FAILURES'}")
        for k, (ok, msg) in results.items():
            print(f"  [{OK if ok else NO}] {k:14} {msg}")
    return all(v[0] for v in results.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["metrla", "pems08f"])
    ap.add_argument("--priors", nargs="+", default=["kronecker", "isotropic"])
    ap.add_argument("--gamma", type=float, default=0.04, help="GAD-style small γ for the raw arm")
    a = ap.parse_args()
    allok = True
    for ds in a.datasets:
        for pr in a.priors:
            allok &= check(ds, pr, gamma=a.gamma)
    print(f"\n{'='*50}\n{'ALL CHECKS PASS ✓' if allok else 'SOME CHECKS FAILED ✗'}")


if __name__ == "__main__":
    main()
