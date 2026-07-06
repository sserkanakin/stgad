"""joint spatio-temporal spectrum check for the kronecker forward process.

the spatial-only forward_analysis.py reduces each (N,H) residual to graph signal
metrics and averages the H horizons away — it never looks at the temporal factor
Σ_T or the joint spectrum. this module answers the two questions that leaves open:

  1. is the residual field "mostly low frequency" in the FULL kronecker sense —
     energy concentrated in low spatial-graph-frequency AND low temporal-frequency
     modes, the same low-low corner where Σ = Σ_T ⊗ Σ_S puts its variance?
  2. does the forward drive that joint spectrum onto the stationary distribution —
     does the per-mode energy E[|C|²] converge to the stationary variance σ²/θ?

everything lives in the joint eigenbasis the sde is already defined in. for a
residual field R (N,H) the joint-mode coefficients are C = V_Sᵀ R V_T (see
forward_fcps._rotate_to_eigenbasis). mode (i,j) has spatial frequency lam_S[i]
(laplacian eigenvalue) and temporal frequency = oscillations of V_T[:,j]. its
energy is E[i,j] = mean_b |C_b[i,j]|².

the forward marginal is closed-form per mode (forward_fcps):
    C_t = exp(-θ c̄)·C_0 + N(0, σ²(1-exp(-2θ c̄))/θ)
so the EXPECTED per-mode energy at level c̄ is analytic:
    E_t = exp(-2θ c̄)·E_0 + (σ²/θ)·(1-exp(-2θ c̄))
which starts at the data energy E_0 and relaxes to the stationary variance σ²/θ.
we report empirical E_t (from actual forward samples) next to this analytic curve —
they must agree — and the distance of E_t to σ²/θ, which must go to 0.

runs offline on cached residuals — no gpu / daic.
"""
from __future__ import annotations

import numpy as np

from src.diffusion.prior import load_prior, _load_residuals_train
from src.diffusion import forward_fcps as ff


def mode_energy(R: np.ndarray, prior) -> np.ndarray:
    """per-mode energy E[i,j] = mean_b |C_b[i,j]|² for C = V_Sᵀ R V_T. (N,H)."""
    C = np.einsum("ki,bih,hj->bkj", prior.V_S.T, R, prior.V_T)
    return (C ** 2).mean(axis=0)


def _temporal_freq(V_T: np.ndarray) -> np.ndarray:
    """oscillation proxy per temporal eigenvector: sign changes down the column.
    canonical temporal-frequency ordering, decoupled from Σ_T's variance."""
    sign = np.sign(V_T)
    sign[sign == 0] = 1.0
    return (np.abs(np.diff(sign, axis=0)) > 0).sum(axis=0).astype(float)


def _low_low_fraction(E: np.ndarray, s_order: np.ndarray, t_order: np.ndarray) -> float:
    """fraction of energy in the lowest-frequency quarter of modes (lower half of
    spatial modes × lower half of temporal modes). a white field gives ~0.25."""
    N, H = E.shape
    lo_s = s_order[: N // 2]
    lo_t = t_order[: H // 2]
    block = E[np.ix_(lo_s, lo_t)].sum()
    return float(block / E.sum())


def _centroids(E: np.ndarray, lam_s: np.ndarray, tfreq: np.ndarray) -> tuple[float, float]:
    """normalized spectral centroids (0=lowest freq, 1=highest) on each axis."""
    lam_n = lam_s / (lam_s.max() + 1e-12)
    tf_n = tfreq / (tfreq.max() + 1e-12)
    tot = E.sum()
    sc_s = float((E.sum(1) * lam_n).sum() / tot)
    sc_t = float((E.sum(0) * tf_n).sum() / tot)
    return sc_s, sc_t


def run(dataset: str, seed: int = 42, backbone: str = "gwn",
        n_samples: int = 512, n_steps: int = 7) -> dict:
    prior = load_prior(dataset, prior="kronecker", seed=seed,
                       spatial="structured", backbone=backbone)
    R, _ = _load_residuals_train(dataset, seed=seed, backbone=backbone)  # (B,N,H) unit space

    rng = np.random.default_rng(0)
    idx = rng.choice(R.shape[0], size=min(n_samples, R.shape[0]), replace=False)
    R0 = R[idx]
    B = R0.shape[0]

    lam_s = prior.lam_S
    tfreq = _temporal_freq(prior.V_T)
    s_order = np.argsort(lam_s)                 # low → high spatial frequency
    t_order = np.argsort(tfreq)                 # low → high temporal frequency

    E0 = mode_energy(R0, prior)                 # data energy per mode (N,H)
    sig2 = prior.sigma2                         # stationary variance per mode σ²/θ

    # --- q1: is the data (and the prior) low-frequency in the joint sense? ---
    data_lowlow = _low_low_fraction(E0, s_order, t_order)
    prior_lowlow = _low_low_fraction(sig2, s_order, t_order)
    white_lowlow = (E0.shape[0] // 2) * (E0.shape[1] // 2) / E0.size
    data_cs, data_ct = _centroids(E0, lam_s, tfreq)
    prior_cs, prior_ct = _centroids(sig2, lam_s, tfreq)

    # --- q2: does the forward drive E_t onto the stationary variance σ²/θ? ---
    # everything below works with the FREQUENCY-ORDERED energy map (low→high on each
    # axis) and its two marginal spectra: energy summed over temporal modes (a curve
    # over spatial frequency) and over spatial modes (a curve over temporal frequency).
    c_lo, c_hi = ff.noise_level_bounds(prior)
    grid = ff.reverse_c_bar_grid(c_lo, c_hi, n_steps)[::-1]   # ascending c̄
    stat_l1 = sig2.sum()

    def _ordered(E):
        return E[np.ix_(s_order, t_order)]

    E0_o = _ordered(E0)
    sig2_o = _ordered(sig2)
    # per-level marginal spectra (curves), keyed by level label for plotting the morph
    spec_spatial = {"data": E0_o.sum(1), "prior": sig2_o.sum(1)}
    spec_temporal = {"data": E0_o.sum(0), "prior": sig2_o.sum(0)}

    traj = []
    for cb in grid:
        cb = float(cb)
        Rt = ff.sample_R_t_from_c_bar(R0, cb, prior, rng)     # actual forward samples
        Et = _ordered(mode_energy(Rt, prior))
        decay = np.exp(-2.0 * prior.theta * cb)
        Et_ana = _ordered(decay * E0 + sig2 * (1.0 - decay))  # closed-form expectation
        spec_spatial[f"{cb:.6g}"] = Et.sum(1)
        spec_temporal[f"{cb:.6g}"] = Et.sum(0)
        traj.append({
            "c_bar": cb,
            # rel-L1 distance of the energy spectrum to the stationary target σ²/θ,
            # for the full 2d map and each marginal (all → 0 as c̄ → c̄_hi)
            "dist_joint": float(np.abs(Et - sig2_o).sum() / stat_l1),
            "dist_spatial": float(np.abs(Et.sum(1) - sig2_o.sum(1)).sum() / stat_l1),
            "dist_temporal": float(np.abs(Et.sum(0) - sig2_o.sum(0)).sum() / stat_l1),
            "dist_ana": float(np.abs(Et_ana - sig2_o).sum() / stat_l1),
            "emp_vs_ana": float(np.abs(Et - Et_ana).sum() / stat_l1),  # sampler sanity
            "lowlow": _low_low_fraction(Et, np.arange(prior.N), np.arange(prior.H)),
        })

    return {
        "dataset": dataset, "backbone": backbone, "seed": seed, "n_samples": B,
        "N": int(prior.N), "H": int(prior.H), "c_lo": float(c_lo), "c_hi": float(c_hi),
        "lowlow": {"data": data_lowlow, "prior": prior_lowlow, "white": float(white_lowlow)},
        "centroid_spatial": {"data": data_cs, "prior": prior_cs},
        "centroid_temporal": {"data": data_ct, "prior": prior_ct},
        "grid": [float(c) for c in grid],
        "trajectory": traj,
        # freq-ordered arrays + marginal spectra kept for plotting
        "_E0": E0_o, "_sig2": sig2_o,
        "_lam_s_sorted": lam_s[s_order],
        "_tfreq_sorted": tfreq[t_order],
        "_spec_spatial": spec_spatial,       # level -> (N,) energy vs spatial freq
        "_spec_temporal": spec_temporal,     # level -> (H,) energy vs temporal freq
    }


def print_report(res: dict) -> None:
    ll, cs, ct = res["lowlow"], res["centroid_spatial"], res["centroid_temporal"]
    print(f"\n{'='*74}\njoint spectrum — {res['dataset']} "
          f"({res['backbone']}, seed{res['seed']}, N={res['N']} H={res['H']}, B={res['n_samples']})\n{'='*74}")
    print("\nq1: is the residual mostly low-frequency in the kronecker (space×time) sense?")
    print(f"  energy in lowest-freq quarter of modes:  data={ll['data']:.3f}   "
          f"prior={ll['prior']:.3f}   white-noise baseline={ll['white']:.3f}")
    print(f"  spatial spectral centroid  (0=low,1=high): data={cs['data']:.3f}   prior={cs['prior']:.3f}")
    print(f"  temporal spectral centroid (0=low,1=high): data={ct['data']:.3f}   prior={ct['prior']:.3f}")
    verdict = "LOW-FREQUENCY" if ll["data"] > 2 * ll["white"] else "NOT clearly low-frequency"
    print(f"  -> data energy is {ll['data']/ll['white']:.1f}x the white baseline in the low corner  ({verdict})")

    print("\nq2: does the forward drive the joint spectrum onto the stationary variance σ²/θ?")
    print(f"  {'c̄':>10} | {'dist joint':>10} | {'dist spatial':>12} | {'dist temporal':>13} | "
          f"{'emp vs ana':>11}")
    for r in res["trajectory"]:
        print(f"  {r['c_bar']:10.3g} | {r['dist_joint']:10.4f} | {r['dist_spatial']:12.4f} | "
              f"{r['dist_temporal']:13.4f} | {r['emp_vs_ana']:11.4f}")
    last = res["trajectory"][-1]
    print(f"  -> at c̄_hi the joint spectrum is within {last['dist_joint']*100:.2f}% of stationary; "
          f"empirical matches the closed-form to {last['emp_vs_ana']*100:.2f}% (the residual is MC noise)")
