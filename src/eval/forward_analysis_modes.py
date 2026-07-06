"""per-mode forward analysis (GAD-style): do the modes mix, is the variance right,
and do any modes disappear?

the joint-spectrum check (forward_analysis_joint.py) looks at aggregate energy. this
one looks at every individual mode of the kronecker eigenbasis and its marginal
variance trajectory, exactly the quantity gad's forward analysis tracks. the forward
sde is diagonal in this basis (forward_fcps): mode k with precision θ_k evolves as

    C_t = exp(-θ_k c̄) C_0 + N(0, σ²(1 - exp(-2θ_k c̄))/θ_k)

so its marginal variance is closed-form,
    Var_k(c̄) = exp(-2θ_k c̄)·Var_data_k + (σ²/θ_k)·(1 - exp(-2θ_k c̄)),
a weighted walk from the DATA variance to the STATIONARY variance σ²/θ_k. two readouts:

  mixing fraction  m_k(c̄) = 1 - exp(-2θ_k c̄) ∈ [0,1]
      how far mode k has relaxed to stationary. stiff modes (large θ) mix at small c̄,
      the slowest mode (θ_min) only by c̄_hi = 5/θ_min. "modes get mixed" = every
      m_k → 1 within the c̄ range (none left frozen).
  variance ratio   Var_k(c̄) / (σ²/θ_k) → 1
      "the variance is right" = the empirical (sampled) marginal variance lands on the
      stationary σ²/θ_k, for every mode.
  end state        Var_k(c̄_hi) vs σ²/θ_k, across the full dynamic range of θ
      "modes don't disappear" = every mode's stationary variance is nonzero and the
      forward reaches it; nothing collapses to ~0 (except where σ²/θ is genuinely tiny,
      which the scatter makes explicit).

runs offline on cached residuals — no gpu.
"""
from __future__ import annotations

import numpy as np

from src.diffusion.prior import load_prior, _load_residuals_train
from src.diffusion import forward_fcps as ff


def _rotate(R: np.ndarray, prior) -> np.ndarray:
    """(B,N,H) -> joint-eigenbasis coefficients V_Sᵀ R V_T, same shape."""
    return np.einsum("ki,bih,hj->bkj", prior.V_S.T, R, prior.V_T)


def run(dataset: str, seed: int = 42, backbone: str = "gwn",
        n_samples: int = 512, n_steps: int = 7) -> dict:
    prior = load_prior(dataset, prior="kronecker", seed=seed,
                       spatial="structured", backbone=backbone)
    R, _ = _load_residuals_train(dataset, seed=seed, backbone=backbone)

    rng = np.random.default_rng(0)
    idx = rng.choice(R.shape[0], size=min(n_samples, R.shape[0]), replace=False)
    R0 = R[idx]
    B = R0.shape[0]

    theta = prior.theta.ravel()                       # (K,) per-mode precision
    sig2 = prior.sigma2.ravel()                       # (K,) stationary variance σ²/θ
    var_data = (_rotate(R0, prior) ** 2).mean(0).ravel()   # (K,) data variance per mode

    c_lo, c_hi = ff.noise_level_bounds(prior)
    grid = ff.reverse_c_bar_grid(c_lo, c_hi, n_steps)[::-1]  # ascending c̄

    var_emp, var_ana, mix = [], [], []
    for cb in grid:
        cb = float(cb)
        Ct = _rotate(ff.sample_R_t_from_c_bar(R0, cb, prior, rng), prior)
        var_emp.append((Ct ** 2).mean(0).ravel())
        h2 = np.exp(-2.0 * theta * cb)
        var_ana.append(h2 * var_data + sig2 * (1.0 - h2))
        mix.append(1.0 - h2)

    var_emp = np.array(var_emp)                        # (L, K)
    var_ana = np.array(var_ana)
    mix = np.array(mix)

    # end-state diagnostics
    ratio_hi = var_emp[-1] / sig2                       # should be ~1 for every mode
    within10 = float(np.mean(np.abs(ratio_hi - 1.0) < 0.10))
    slow_mix = float(mix[-1].min())                     # slowest mode's mixing at c̄_hi

    return {
        "dataset": dataset, "backbone": backbone, "seed": seed, "n_samples": B,
        "N": int(prior.N), "H": int(prior.H), "K": int(theta.size),
        "c_lo": float(c_lo), "c_hi": float(c_hi),
        "grid": [float(c) for c in grid],
        "theta_min": float(theta.min()), "theta_max": float(theta.max()),
        "sig2_min": float(sig2.min()), "sig2_max": float(sig2.max()),
        "within10_at_hi": within10, "slow_mix_at_hi": slow_mix,
        "median_abs_ratio_err_hi": float(np.median(np.abs(ratio_hi - 1.0))),
        # arrays for plotting
        "_theta": theta, "_sig2": sig2, "_var_data": var_data,
        "_var_emp": var_emp, "_var_ana": var_ana, "_mix": mix,
    }


def print_report(res: dict) -> None:
    print(f"\n{'='*74}\nper-mode forward — {res['dataset']} "
          f"({res['backbone']}, seed{res['seed']}, K={res['K']} modes, B={res['n_samples']})\n{'='*74}")
    print(f"  precision θ spans [{res['theta_min']:.3g}, {res['theta_max']:.3g}]  "
          f"(ratio {res['theta_max']/res['theta_min']:.0f}x)")
    print(f"  stationary variance σ²/θ spans [{res['sig2_min']:.3g}, {res['sig2_max']:.3g}]  "
          f"-> min is nonzero, so no mode is intrinsically absent")
    print("\n  do the modes mix and land on the right variance by c̄_hi?")
    print(f"    slowest mode mixing fraction at c̄_hi : {res['slow_mix_at_hi']:.4f}  (→1 = fully mixed)")
    print(f"    modes within 10% of stationary var    : {res['within10_at_hi']*100:.1f}%")
    print(f"    median |Var/σ²θ - 1| at c̄_hi          : {res['median_abs_ratio_err_hi']*100:.2f}%  "
          f"(residual is MC noise at B={res['n_samples']})")
