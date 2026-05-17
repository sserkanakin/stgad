"""sde-sanity sanity checks for the structured-prior SDE.

Two checks per dataset (METR-LA, PEMS-BAY) using the fitted prior parameters
from prior-construction:

  Check 1 — Stationary distribution (forward at τ=1).
      Sample 5000 r_0 from training residuals (per-horizon standardised by the
      target_scale stored in the prior YAML), push through forward_sample(τ=1),
      compute empirical covariance on a 20-sensor subset, compare to
      Σ_prior = Q_H ⊗ Σ_S (restricted to the same subset).
      Pass: rel.Frob < 0.10.

  Check 2 — Oracle-score round-trip.
      Pick 32 r_0 from training residuals, forward to τ=1, then run reverse_oracle
      with 100 Euler-Maruyama steps. Recovered r should match r_0.
      Pass: mean rel.recovery < 0.10 strict, < 0.20 acceptable.

CPU + NumPy throughout (float64). Sizes are modest (≤ 5000 × 12 × 325 ≈ 156 MB);
LAPACK on CPU is faster than GPU at these sizes and gives full precision. PyTorch
+ MPS is reserved for the network-training experiment (score-network).

Usage:
    python experiments/sde-sanity/run_sanity_checks.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.diffusion.prior_fit import build_Q_H, build_Sigma_S
from src.diffusion.sde import (
    beta_bar,
    eigendecompose_prior,
    forward_sample,
    reverse_oracle,
)


# ── Settings (per the spec) ──────────────────────────────────────────────────
#
# β_max is sized so β̄(1) = β_min + 0.5·(β_max − β_min) is many multiples of
# the largest Σ_prior eigenvalue: with the standardised prior (prior-construction v2),
# max(s) ≲ 10, so β_max = 50 → β̄(1) ≈ 25, β̄/max(s) ≳ 2.5, exp(-2.5) ≈ 0.08.
# When the legacy raw-mph² prior is in use, max(s) is O(10⁴) and no realistic
# β_max would reach stationary at τ=1 — see sde-sanity SUMMARY §3.
BETA_MIN = 0.01
BETA_MAX = 50.0
N_REVERSE_STEPS = 100
N_STAT_SAMPLES = 5000
N_OR_SAMPLES = 32
SUBSET_SIZE = 20                 # for the (H·subset, H·subset) covariance check
SEED = 42
TAU_EPS = 1e-3                   # stop reverse SDE here to avoid Σ_τ → 0

DATASETS = ["metrla", "pemsbay"]

OUT_DIR = ROOT / "experiments" / "sde-sanity"
FIG_DIR = OUT_DIR / "figures"


# ── Setup loading ────────────────────────────────────────────────────────────

def load_setup(dataset: str) -> dict:
    """Build Q_H, Σ_S, and the joint eigendecomposition for one dataset."""
    yaml_path = ROOT / "results" / "prior_params" / f"gwn_{dataset}.yaml"
    with open(yaml_path) as f:
        params = yaml.safe_load(f)

    eigs = np.load(ROOT / "data" / "processed" / f"{dataset}_laplacian_eigs.npz")
    eigvals_L = eigs["eigenvalues"].astype(np.float64)
    eigvecs_L = eigs["eigenvectors"].astype(np.float64)

    delta_h = np.array(params["delta_h"], dtype=np.float64)
    Q_H = build_Q_H(delta_h, params["alpha"], params["rho1"], params["rho2"])
    Sigma_S = build_Sigma_S(params["beta"], params["epsilon_s"],
                            eigvals_L, eigvecs_L)

    eig = eigendecompose_prior(Q_H, Sigma_S)

    # target_scale[h] (mph) — sqrt of per-horizon variance of the raw residual.
    # Used to standardise r_train so the SDE acts in the same units as the prior.
    # Older YAMLs without this field are treated as unstandardised (target_scale = 1).
    target_scale = np.array(
        params.get("target_scale", np.ones(Q_H.shape[0])),
        dtype=np.float64,
    )

    return {
        "dataset": dataset,
        "Q_H": Q_H,
        "Sigma_S": Sigma_S,
        "eig": eig,
        "params": params,
        "target_scale": target_scale,
    }


def load_residuals(dataset: str, target_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load training residuals and standardise to the modelled-variable scale.

    The prior (Q_H ⊗ Σ_S) was fit on residuals divided by ``target_scale[h]``,
    so we apply the same per-horizon division here before any SDE operation.
    """
    path = ROOT / "results" / "residuals" / f"gwn_{dataset}_train_seed0.npz"
    data = np.load(path)
    r = data["residuals"].squeeze(-1).astype(np.float64)         # (N_win, 12, N)
    mask = data["mask"].squeeze(-1)                               # (N_win, 12, N)
    r = r / target_scale[None, :, None]                          # standardise
    return r, mask


# ── Check 1: stationary distribution ────────────────────────────────────────

def check_stationary(setup: dict, r_train: np.ndarray,
                     rng: np.random.Generator) -> dict:
    """Forward 5000 random training r_0 to τ=1, compare empirical cov to Σ_prior."""
    H = setup["Q_H"].shape[0]
    N = setup["Sigma_S"].shape[0]
    eig = setup["eig"]

    n_avail = r_train.shape[0]
    idx = rng.choice(n_avail, size=N_STAT_SAMPLES, replace=False)
    r_0 = r_train[idx]                                            # (5000, H, N)

    t0 = time.time()
    r_tau = forward_sample(r_0, tau=1.0, eig=eig,
                           beta_min=BETA_MIN, beta_max=BETA_MAX, rng=rng)
    fwd_time = time.time() - t0

    # Subset SUBSET_SIZE evenly-spaced sensors → tractable covariance
    subset_idx = np.round(np.linspace(0, N - 1, SUBSET_SIZE)).astype(int)
    r_tau_sub = r_tau[:, :, subset_idx]                           # (5000, H, S)
    flat = r_tau_sub.reshape(N_STAT_SAMPLES, H * SUBSET_SIZE)

    # Empirical covariance (samples are not centred — Σ_emp = (X.T @ X) / n
    # is the second-moment estimator, which equals the variance estimator for
    # zero-mean processes; forward samples should have ≈ zero mean).
    cov_emp = (flat.T @ flat) / N_STAT_SAMPLES                    # (H·S, H·S)

    # Target: Q_H ⊗ Σ_S restricted to the subset
    target = np.kron(setup["Q_H"], setup["Sigma_S"][np.ix_(subset_idx, subset_idx)])

    diff = cov_emp - target
    frob_diff = float(np.linalg.norm(diff, "fro"))
    frob_target = float(np.linalg.norm(target, "fro"))
    rel_frob = frob_diff / max(frob_target, 1e-12)

    # Also compute per-mode variance ratio in the joint eigenbasis (full-rank
    # diagnostic, not restricted to a subset)
    from src.diffusion.sde import to_spectral
    r_tau_tilde = to_spectral(r_tau, eig["U_T"], eig["U_S"])
    var_per_mode_emp = (r_tau_tilde ** 2).mean(0)                  # (H, N)
    ratio = var_per_mode_emp / eig["s"]
    ratio_stats = {
        "min": float(ratio.min()),
        "max": float(ratio.max()),
        "median": float(np.median(ratio)),
        "frac_within_10pct": float(np.mean(np.abs(ratio - 1.0) < 0.10)),
        "frac_within_20pct": float(np.mean(np.abs(ratio - 1.0) < 0.20)),
    }

    return {
        "rel_frob": rel_frob,
        "frob_diff": frob_diff,
        "frob_target": frob_target,
        "pass": bool(rel_frob < 0.10),
        "n_samples": N_STAT_SAMPLES,
        "subset_size": SUBSET_SIZE,
        "beta_bar_at_1": float(beta_bar(1.0, BETA_MIN, BETA_MAX)),
        "max_eigval_prior_subset": float(np.max(np.linalg.eigvalsh(target))),
        "min_eigval_prior_subset": float(np.min(np.linalg.eigvalsh(target))),
        "per_mode_variance_ratio": ratio_stats,
        "wall_time_forward_sec": fwd_time,
        # for the optional figure
        "_cov_emp": cov_emp,
        "_cov_target": target,
        "_subset_idx": subset_idx.tolist(),
    }


# ── Check 2: oracle-score round-trip ────────────────────────────────────────

def check_oracle_round_trip(setup: dict, r_train: np.ndarray,
                            rng: np.random.Generator) -> dict:
    """Forward 32 r_0 to τ=1, then reverse with oracle score; compare to r_0."""
    eig = setup["eig"]
    n_avail = r_train.shape[0]
    idx = rng.choice(n_avail, size=N_OR_SAMPLES, replace=False)
    r_0 = r_train[idx]                                            # (32, H, N)

    t0 = time.time()
    r_tau = forward_sample(r_0, tau=1.0, eig=eig,
                           beta_min=BETA_MIN, beta_max=BETA_MAX, rng=rng)
    fwd_time = time.time() - t0

    t0 = time.time()
    r_recovered = reverse_oracle(r_tau, r_0, n_steps=N_REVERSE_STEPS, eig=eig,
                                 beta_min=BETA_MIN, beta_max=BETA_MAX,
                                 rng=rng, tau_start=1.0, tau_eps=TAU_EPS)
    rev_time = time.time() - t0

    # Per-sample relative recovery error
    err_norms = np.linalg.norm(
        (r_recovered - r_0).reshape(N_OR_SAMPLES, -1), axis=1
    )
    r_0_norms = np.linalg.norm(r_0.reshape(N_OR_SAMPLES, -1), axis=1)
    rel = err_norms / r_0_norms                                   # (32,)

    return {
        "rel_recovery_mean": float(rel.mean()),
        "rel_recovery_max": float(rel.max()),
        "rel_recovery_p50": float(np.median(rel)),
        "rel_recovery_p95": float(np.percentile(rel, 95)),
        "rel_recovery_min": float(rel.min()),
        "pass_strict": bool(rel.mean() < 0.10),
        "pass_acceptable": bool(rel.mean() < 0.20),
        "n_samples": N_OR_SAMPLES,
        "n_reverse_steps": N_REVERSE_STEPS,
        "tau_eps": TAU_EPS,
        "wall_time_forward_sec": fwd_time,
        "wall_time_reverse_sec": rev_time,
    }


# ── Optional figure: target Σ vs SDE-empirical Σ on the subset ──────────────

def make_covariance_figure(c1_plot: dict, c1_summary: dict,
                           dataset: str, out_path: Path) -> None:
    """3-panel comparison: target, empirical, diff (on the 20-sensor subset)."""
    target = c1_plot["_cov_target"]
    emp = c1_plot["_cov_emp"]
    diff = emp - target
    vmax = float(np.abs(target).max())
    diff_max = float(np.abs(diff).max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].imshow(target, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
    axes[0].set_title("Target Σ_prior on subset (Q_H ⊗ Σ_S[subset])")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(emp, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
    axes[1].set_title(f"Empirical Σ at τ=1 (n={N_STAT_SAMPLES})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff, vmin=-diff_max, vmax=diff_max, cmap="RdBu_r", aspect="auto")
    axes[2].set_title(f"Δ = empirical − target  (rel.Frob = {c1_summary['rel_frob']:.4f})")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("h·n index")
        ax.set_ylabel("h·n index")

    fig.suptitle(f"sde-sanity Check 1 — stationary covariance — {dataset.upper()}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print("sde-sanity — structured-SDE sanity checks")
    print("=" * 78)
    print(f"  schedule:  β_min={BETA_MIN}  β_max={BETA_MAX}  "
          f"⇒  β̄(1) = {beta_bar(1.0, BETA_MIN, BETA_MAX):.4f}")
    print(f"  Check 1:   forward sample × {N_STAT_SAMPLES} r_0 from training, "
          f"compare empirical Σ to Σ_prior on a {SUBSET_SIZE}-sensor subset")
    print(f"  Check 2:   oracle round-trip on {N_OR_SAMPLES} r_0, "
          f"{N_REVERSE_STEPS} Euler-Maruyama steps from τ=1 to τ_eps={TAU_EPS}")
    print(f"  device:    CPU + NumPy (float64); MPS reserved for score-network")
    print(f"  seed:      {SEED}")

    rng = np.random.default_rng(SEED)
    output = {
        "metadata": {
            "beta_min": BETA_MIN,
            "beta_max": BETA_MAX,
            "beta_bar_at_1": float(beta_bar(1.0, BETA_MIN, BETA_MAX)),
            "n_reverse_steps": N_REVERSE_STEPS,
            "n_stationary_samples": N_STAT_SAMPLES,
            "n_oracle_samples": N_OR_SAMPLES,
            "subset_size": SUBSET_SIZE,
            "tau_eps": TAU_EPS,
            "seed": SEED,
        },
        "results": {},
    }

    overall_pass = True

    for ds in DATASETS:
        print(f"\n{'─' * 78}\n  {ds.upper()}\n{'─' * 78}")
        setup = load_setup(ds)
        r_train, _mask = load_residuals(ds, setup["target_scale"])
        print(f"  target_scale (mph): {np.array2string(setup['target_scale'], precision=2)}")
        print(f"  r_train (standardised) shape: {r_train.shape}  "
              f"per-h std: [{r_train.std(axis=(0,2)).min():.3f}, {r_train.std(axis=(0,2)).max():.3f}]")

        # Diagnostic: prior eigenvalue range (in the joint eigenbasis)
        s = setup["eig"]["s"]
        bb = beta_bar(1.0, BETA_MIN, BETA_MAX)
        ratio = bb / s
        print(f"  s (Σ_prior eigvals): min={s.min():.3e}  max={s.max():.3e}  "
              f"median={np.median(s):.3e}")
        print(f"  β̄/s:                  min={ratio.min():.3e}  max={ratio.max():.3e}  "
              f"median={np.median(ratio):.3e}")
        # exp(-β̄/s) close to 0 means the mode reached stationarity; close to 1 means
        # the forward process barely diffused that mode.
        decay = np.exp(-ratio)
        print(f"  exp(-β̄/s):            min={decay.min():.3e}  max={decay.max():.3e}  "
              f"frac<0.01 (≈ stationary)={float(np.mean(decay < 0.01)):.3f}")

        c1 = check_stationary(setup, r_train, rng)
        c2 = check_oracle_round_trip(setup, r_train, rng)

        # Pop the figure-only fields out of c1 before serialising
        c1_plot = {k: c1.pop(k) for k in ["_cov_emp", "_cov_target", "_subset_idx"]}

        print(f"\n  Check 1 — stationary  (target rel.Frob < 0.10)")
        print(f"    rel.Frob:        {c1['rel_frob']:.4f}   "
              f"[{'PASS' if c1['pass'] else 'FAIL'}]")
        print(f"    per-mode var ratio  (empirical / s):")
        rs = c1["per_mode_variance_ratio"]
        print(f"      min={rs['min']:.3f}  median={rs['median']:.3f}  max={rs['max']:.3f}")
        print(f"      frac within ±10%: {rs['frac_within_10pct']:.3f}    "
              f"frac within ±20%: {rs['frac_within_20pct']:.3f}")
        print(f"    forward wallclock: {c1['wall_time_forward_sec']:.2f}s  "
              f"(n={N_STAT_SAMPLES} on CPU)")

        print(f"\n  Check 2 — oracle round-trip  (target rel.recovery mean < 0.10 strict)")
        print(f"    rel.recovery: mean={c2['rel_recovery_mean']:.4f}  "
              f"median={c2['rel_recovery_p50']:.4f}  "
              f"p95={c2['rel_recovery_p95']:.4f}  "
              f"max={c2['rel_recovery_max']:.4f}")
        verdict = ("PASS-strict"   if c2["pass_strict"]
                   else "PASS-acceptable" if c2["pass_acceptable"]
                   else "FAIL")
        print(f"    verdict:          [{verdict}]")
        print(f"    reverse wallclock: {c2['wall_time_reverse_sec']:.2f}s  "
              f"(n={N_OR_SAMPLES}, {N_REVERSE_STEPS} steps on CPU)")

        # Figure
        fig_path = FIG_DIR / f"fig_check1_covariance_{ds}.png"
        make_covariance_figure(c1_plot, c1, ds, fig_path)
        print(f"    figure saved:     {fig_path.relative_to(ROOT)}")

        output["results"][ds] = {
            "check1_stationary": c1,
            "check2_oracle_round_trip": c2,
        }
        # Acceptable for the round-trip; strict for stationary
        ds_pass = c1["pass"] and c2["pass_acceptable"]
        if not ds_pass:
            overall_pass = False

    out_path = OUT_DIR / "sanity_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\n{'=' * 78}")
    print(f"  Saved {out_path.relative_to(ROOT)}")

    if overall_pass:
        print("  *** BOTH CHECKS PASS ON BOTH DATASETS ***")
    else:
        print("  *** AT LEAST ONE CHECK FAILED — see numbers above; do not patch ***")
    print("=" * 78)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
