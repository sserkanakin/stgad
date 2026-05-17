"""Fit Kronecker prior parameters from GWN training-set residuals.

Usage:
    python scripts/fit_prior.py --dataset metrla
    python scripts/fit_prior.py --dataset pemsbay

Outputs
-------
results/prior_params/gwn_{dataset}.yaml
    Prior parameters: delta_h, alpha, rho1, rho2, beta, epsilon_s, sigma.

experiments/prior-construction/figures/fig_prior_samples_covariance_check_gwn_{dataset}.png
    Six-panel diagnostic figure.

Exit codes: 0 = GO (all checks pass), 1 = NO-GO (at least one check failed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── matplotlib must be imported before any fork (no display on headless machines)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.diffusion.prior_fit import (
    build_Q_H,
    build_Sigma_S,
    empirical_spatial_cov,
    empirical_temporal_cov,
    empirical_temporal_corr,
    fit_prior,
    lag_averaged_acf,
    sample_prior,
)
from src.diffusion.sde import eigendecompose_prior
from src.diffusion.sde_torch import (
    beta_bar as torch_beta_bar,
    eig_to_torch,
    forward_sample as torch_forward_sample,
    to_spectral as torch_to_spectral,
)

_BACKBONE = "gwn"


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_residuals(dataset: str, seed: int):
    """Load GWN training residuals from results/residuals/.

    Returns
    -------
    r    : (N_win, 12, N) float32 — residuals in raw mph
    mask : (N_win, 12, N) bool   — True where reading is valid
    """
    path = (
        ROOT / "results" / "residuals"
        / f"{_BACKBONE}_{dataset}_train_seed{seed}.npz"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Residuals not found: {path}\n"
            "Run: python scripts/extract_residuals.py "
            f"--dataset {dataset} --seed {seed}"
        )
    data = np.load(path)
    r    = data["residuals"].squeeze(-1).astype(np.float32)  # (N_win, 12, N)
    mask = data["mask"].squeeze(-1)                          # (N_win, 12, N)
    return r, mask


def _load_laplacian(dataset: str):
    """Load pre-computed normalised Laplacian eigendecomposition.

    Returns
    -------
    eigvals_L : (N,) float64 — sorted ascending eigenvalues in [0, 2]
    eigvecs_L : (N, N) float64 — eigenvectors as columns
    """
    path = ROOT / "data" / "processed" / f"{dataset}_laplacian_eigs.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Laplacian eigendecomposition not found: {path}\n"
            "Run: python scripts/preprocess.py --dataset {dataset}"
        )
    data = np.load(path)
    return data["eigenvalues"].astype(np.float64), data["eigenvectors"].astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def _covariance_check(
    Q_H: np.ndarray,
    Sigma_S: np.ndarray,
    n_samples: int = 2000,
    subset_size: int = 20,
    seed: int = 42,
) -> dict:
    """Verify sample_prior reproduces the correct Kronecker covariance.

    Draws *n_samples* from N(0, Q_H ⊗ Σ_S), restricts to *subset_size* evenly-
    spaced sensors, and compares the empirical covariance to the analytic target.

    PASS criterion: relative Frobenius error < 0.15.

    Returns
    -------
    dict with keys
        rel_frob   : float  — ||Ŝ - S||_F / ||S||_F
        pass       : bool
        emp_cov    : (H*subset, H*subset) empirical covariance
        target_cov : (H*subset, H*subset) analytic Kronecker covariance
        idx        : (subset_size,) int   — sensor indices used
    """
    rng = np.random.default_rng(seed)
    samples = sample_prior(Q_H, Sigma_S, n_samples, rng=rng)  # (n, H, N)

    N = Sigma_S.shape[0]
    H = Q_H.shape[0]
    idx = np.round(np.linspace(0, N - 1, subset_size)).astype(int)

    s_sub  = samples[:, :, idx]                    # (n, H, subset_size)
    s_flat = s_sub.reshape(n_samples, H * subset_size)  # (n, H*subset)

    # Empirical covariance (samples are zero-mean by construction)
    emp_cov = (s_flat.T @ s_flat) / n_samples      # (H*sub, H*sub)

    # Target: kron(Q_H, Sigma_S[subset, subset])
    S_sub      = Sigma_S[np.ix_(idx, idx)]
    target_cov = np.kron(Q_H, S_sub)

    frob_diff  = float(np.linalg.norm(emp_cov - target_cov, "fro"))
    frob_tgt   = float(np.linalg.norm(target_cov, "fro"))
    rel_frob   = frob_diff / max(frob_tgt, 1e-12)

    return {
        "rel_frob":   rel_frob,
        "pass":       rel_frob < 0.15,
        "emp_cov":    emp_cov,
        "target_cov": target_cov,
        "idx":        idx,
    }


def _vp_forward_marginal_check(
    Q_H: np.ndarray,
    Sigma_S: np.ndarray,
    taus: list[float] | None = None,
    n_samples: int = 2000,
    beta_min: float = 0.01,
    beta_max: float = 50.0,
    seed: int = 99,
    mean_threshold: float = 0.05,
    max_threshold: float = 0.20,
) -> dict:
    """VP-SDE forward marginal check — exercises the actual `forward_sample`.

    Training uses the VP-SDE forward marginal implemented in
    ``src.diffusion.sde_torch.forward_sample``::

        r_τ | r_0 ~ N(decay[i,j] · r_0_tilde[i,j],  s[i,j]·(1 − exp(−β̄(τ)/s[i,j])))

    in the joint Q_H ⊗ Σ_S eigenbasis, where ``decay = exp(−½·β̄/s)``.

    This check generates ``r_0 ~ N(0, Σ_prior)`` via ``sample_prior``, calls
    the torch ``forward_sample`` (the same routine training uses), rotates
    the result into the joint eigenbasis, removes the analytic mean, and
    compares the empirical per-mode variance to the analytic conditional
    variance. Done at multiple τ to catch bugs that show up only at small
    or large noise levels.

    Replaces the prior VE-SDE Tweedie check, which verified a different SDE.

    Args:
        Q_H, Sigma_S: fitted Kronecker prior matrices.
        taus: list of τ values in (0, 1]. Default [0.1, 0.5, 1.0].
        n_samples: number of r_0 draws per τ.
        beta_min, beta_max: VP-SDE schedule. Defaults match the locked
            sde-sanity values (β_max=50 reaches stationary at τ=1).
        seed: RNG seed for the r_0 draw + the torch forward-sample noise.
        mean_threshold, max_threshold: PASS criteria on relative error
            (mean across modes / worst-case mode at each τ).

    Returns:
        dict with keys
            taus         : list[float]
            mean_rel_err : list[float] — average across H·N modes per τ
            max_rel_err  : list[float] — worst mode per τ
            pass         : bool — all τ pass both mean + max thresholds
            beta_min, beta_max, mean_threshold, max_threshold (echoes)
    """
    if taus is None:
        taus = [0.1, 0.5, 1.0]

    # Build the joint eigendecomposition (the same one training uses) and
    # convert to torch float64 for diagnostic precision.
    eig_np = eigendecompose_prior(Q_H, Sigma_S)
    eig_t = eig_to_torch(eig_np, device="cpu", dtype=torch.float64)

    rng = np.random.default_rng(seed)
    r_0_np = sample_prior(Q_H, Sigma_S, n_samples, rng=rng)        # (n, H, N) float64
    r_0 = torch.from_numpy(r_0_np)                                  # already float64
    r_0_tilde = torch_to_spectral(r_0, eig_t["U_T"], eig_t["U_S"])  # (n, H, N)

    # Torch RNG seeded for forward_sample's ε noise.
    g = torch.Generator(device="cpu").manual_seed(seed)

    mean_rel_err: list[float] = []
    max_rel_err: list[float] = []

    for tau in taus:
        # Conditional mean + variance per mode in the joint eigenbasis.
        bb = torch_beta_bar(torch.tensor(tau, dtype=torch.float64),
                            beta_min, beta_max)
        ratio = bb / eig_t["s"]                                     # (H, N)
        decay = torch.exp(-0.5 * ratio)                             # (H, N)
        var_analytic = -eig_t["s"] * torch.expm1(-ratio)            # (H, N)

        # Run the actual training forward_sample, then rotate r_τ to eigenbasis.
        r_tau = torch_forward_sample(r_0, tau, eig_t, beta_min, beta_max,
                                     generator=g)
        r_tau_tilde = torch_to_spectral(r_tau, eig_t["U_T"], eig_t["U_S"])

        # Per-mode conditional residual; empirical variance across the n draws.
        u = r_tau_tilde - decay * r_0_tilde                         # (n, H, N)
        var_empirical = u.var(dim=0, unbiased=True)                 # (H, N)

        rel = (var_empirical - var_analytic).abs() / var_analytic.clamp(min=1e-12)
        mean_rel_err.append(float(rel.mean()))
        max_rel_err.append(float(rel.max()))

    all_pass = all(m < mean_threshold for m in mean_rel_err) and \
               all(x < max_threshold for x in max_rel_err)

    return {
        "taus":           taus,
        "mean_rel_err":   mean_rel_err,
        "max_rel_err":    max_rel_err,
        "pass":           all_pass,
        "beta_min":       beta_min,
        "beta_max":       beta_max,
        "mean_threshold": mean_threshold,
        "max_threshold":  max_threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic figure
# ─────────────────────────────────────────────────────────────────────────────

def _make_figure(
    dataset: str,
    params: dict,
    Q_H: np.ndarray,
    eigvals_L: np.ndarray,
    eigvecs_L: np.ndarray,
    sig_S_emp: np.ndarray,
    Q_corr: np.ndarray,
    cov_result: dict,
    vp_result: dict,
    out_path: Path,
) -> None:
    """Save 6-panel diagnostic figure to *out_path*."""

    alpha = params["alpha"]
    rho1  = params["rho1"]
    rho2  = params["rho2"]
    beta  = params["beta"]
    eps_s = params["epsilon_s"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Prior fitting diagnostics — {_BACKBONE.upper()} / {dataset.upper()}",
        fontsize=13, y=0.98,
    )

    # ── Panel 1: Q_H heatmap ──────────────────────────────────────────────────
    ax = axes[0, 0]
    im = ax.imshow(Q_H, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Q_H — temporal covariance (modelled variable)")
    ax.set_xlabel("horizon")
    ax.set_ylabel("horizon")
    ax.set_xticks(range(0, 12, 2))
    ax.set_yticks(range(0, 12, 2))

    # ── Panel 2: ACF fit ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    acf_emp = lag_averaged_acf(Q_corr)
    lags    = np.arange(1, 12)
    acf_fit = alpha * rho1 ** lags + (1.0 - alpha) * rho2 ** lags

    ax.plot(lags, acf_emp, "o-", color="steelblue", label="Empirical lag-avg ACF")
    ax.plot(lags, acf_fit, "x--", color="tomato",  label=f"Fit (α={alpha:.3f}, ρ₁={rho1:.3f}, ρ₂={rho2:.3f})")
    rmse = float(np.sqrt(np.mean((acf_emp - acf_fit) ** 2)))
    ax.set_title(f"Temporal kernel fit (RMSE={rmse:.4f})")
    ax.set_xlabel("lag k")
    ax.set_ylabel("average correlation")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Spatial GMRF spectral fit ────────────────────────────────────
    ax = axes[0, 2]
    diag_emp   = np.diag(eigvecs_L.T @ sig_S_emp.astype(np.float64) @ eigvecs_L)
    gmrf_diag  = 1.0 / (beta * eigvals_L + eps_s)
    n_modes    = min(len(eigvals_L), 80)   # show first 80 modes for readability

    ax.plot(range(n_modes), diag_emp[:n_modes],  "o",  ms=3, label="Empirical spectral diag",  alpha=0.6)
    ax.plot(range(n_modes), gmrf_diag[:n_modes], "x-", ms=3, label=f"GMRF fit (β={beta:.2f})", alpha=0.8)
    ax.set_title("Spatial spectral fit (first 80 modes)")
    ax.set_xlabel("Laplacian eigenvalue index")
    ax.set_ylabel("spectral energy (modelled var)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Target Kronecker covariance (subset) ─────────────────────────
    ax = axes[1, 0]
    vmax = float(np.abs(cov_result["target_cov"]).max())
    im = ax.imshow(cov_result["target_cov"], aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Target Q_H ⊗ Σ_S[subset]")
    ax.set_xlabel("h·n index")
    ax.set_ylabel("h·n index")

    # ── Panel 5: Empirical covariance from samples ────────────────────────────
    ax = axes[1, 1]
    im = ax.imshow(cov_result["emp_cov"], aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046)
    rel_frob = cov_result["rel_frob"]
    pass_str = "PASS" if cov_result["pass"] else "FAIL"
    ax.set_title(f"Sample empirical cov — rel.Frob={rel_frob:.3f} [{pass_str}]")
    ax.set_xlabel("h·n index")
    ax.set_ylabel("h·n index")

    # ── Panel 6: VP-SDE forward marginal — per-mode variance match ────────────
    ax = axes[1, 2]
    taus    = vp_result["taus"]
    mean_re = vp_result["mean_rel_err"]
    max_re  = vp_result["max_rel_err"]
    mean_th = vp_result["mean_threshold"]
    max_th  = vp_result["max_threshold"]
    x = np.arange(len(taus))
    w = 0.38
    mean_colors = ["green" if m < mean_th else "red" for m in mean_re]
    max_colors  = ["green" if m < max_th  else "red" for m in max_re]
    ax.bar(x - w / 2, mean_re, width=w, color=mean_colors, alpha=0.75, label="mean rel err")
    ax.bar(x + w / 2, max_re,  width=w, color=max_colors,  alpha=0.45, label="max rel err")
    ax.axhline(mean_th, color="black", lw=0.8, ls="--",
               label=f"mean threshold = {mean_th:.2f}")
    ax.axhline(max_th,  color="grey",  lw=0.8, ls=":",
               label=f"max threshold = {max_th:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"τ={τ:.2g}" for τ in taus])
    vp_pass = "PASS" if vp_result["pass"] else "FAIL"
    ax.set_title(f"VP-SDE forward marginal — per-mode var [{vp_pass}]")
    ax.set_ylabel("rel err |emp - analytic| / analytic")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# YAML serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_yaml_native(obj):
    """Recursively convert NumPy scalars/arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_yaml_native(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [float(x) for x in obj.ravel()]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit Kronecker prior parameters from GWN training residuals."
    )
    parser.add_argument(
        "--dataset", choices=["metrla", "pemsbay", "pems08f"], required=True,
        help="Dataset to fit prior for",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="GWN training seed used when extracting residuals (default: 0)",
    )
    parser.add_argument(
        "--epsilon_s", type=float, default=None,
        help="Laplacian regularisation ε_s (default: 1.0 when standardised, 0.01 raw).",
    )
    parser.add_argument(
        "--eps_target", type=float, default=1e-2,
        help="Worst-case SNR target for σ derivation (default: 0.01)",
    )
    parser.add_argument(
        "--no-standardize", action="store_true",
        help="Disable per-horizon standardisation (fit on raw mph² residuals).",
    )
    args = parser.parse_args()
    standardize = not args.no_standardize

    print(f"\n{'='*60}")
    print(f"  Phase 4 — Prior fitting")
    print(f"  backbone={_BACKBONE}  dataset={args.dataset}  seed={args.seed}")
    print(f"  standardize={standardize}  ε_s={args.epsilon_s}  ε_target={args.eps_target}")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[1/5] Loading residuals …")
    r, mask = _load_residuals(args.dataset, args.seed)
    print(f"      residuals: {r.shape}  mask: {mask.shape}")
    print(f"      valid fraction: {mask.mean():.3f}")

    print("[1/5] Loading Laplacian eigendecomposition …")
    eigvals_L, eigvecs_L = _load_laplacian(args.dataset)
    print(f"      eigvals: {eigvals_L.shape}  range [{eigvals_L.min():.4f}, {eigvals_L.max():.4f}]")

    # ── 2. Fit prior parameters ───────────────────────────────────────────────
    print("\n[2/5] Fitting prior parameters …")
    params = fit_prior(
        r, mask, eigvals_L, eigvecs_L,
        epsilon_s=args.epsilon_s,
        eps_target=args.eps_target,
        standardize=standardize,
    )

    print(f"\n  target_scale (mph): {np.array2string(params['target_scale'], precision=3, suppress_small=True)}")
    print(f"  δ_h (modelled var): {np.array2string(params['delta_h'], precision=3, suppress_small=True)}")
    print(f"  α  = {params['alpha']:.4f}   (slow-component weight)")
    print(f"  ρ₁ = {params['rho1']:.4f}   (slow decay)")
    print(f"  ρ₂ = {params['rho2']:.4f}   (fast decay)")
    print(f"  β  = {params['beta']:.4f}   (spatial precision)")
    print(f"  ε_s = {params['epsilon_s']:.4f}")
    print(f"  σ  = {params['sigma']:.4f}   (VP-SDE diffusion coefficient)")

    # ── 3. Build matrices for checks ──────────────────────────────────────────
    print("\n[3/5] Building Q_H and Σ_S …")
    Q_H     = build_Q_H(params["delta_h"], params["alpha"], params["rho1"], params["rho2"])
    Sigma_S = build_Sigma_S(params["beta"], params["epsilon_s"], eigvals_L, eigvecs_L)

    ev_Q = np.linalg.eigvalsh(Q_H)
    ev_S = np.linalg.eigvalsh(Sigma_S)
    print(f"  Q_H  : shape {Q_H.shape}  eigval range [{ev_Q.min():.3f}, {ev_Q.max():.3f}]  cond = {ev_Q.max()/ev_Q.min().clip(1e-12):.1f}")
    print(f"  Σ_S  : shape {Sigma_S.shape}  eigval range [{ev_S.min():.4f}, {ev_S.max():.4f}]  cond = {ev_S.max()/ev_S.min().clip(1e-12):.1f}")
    print(f"  σ (VP-SDE) = {params['sigma']:.4f}  →  ᾱ_T = exp(-σ²) = {np.exp(-params['sigma']**2):.4e}")

    # ── 4. Sanity checks ──────────────────────────────────────────────────────
    print("\n[4/5] Running sanity checks …")

    # 20k samples instead of 5k: the relative-Frobenius estimator has Monte-
    # Carlo variance ∝ 1/√n. With 5k samples PEMS08F's tightly-clustered Σ_S
    # spectrum (cond ≈ 1.24) landed just above the 0.15 threshold even though
    # the sampler is mathematically correct. 20k samples give all three
    # datasets a comfortable margin (rel_frob < 0.10) and the threshold
    # becomes a stricter test for a real sampler bug.
    print("  (a) Covariance check (20000 samples, 20-sensor subset) …")
    cov_result = _covariance_check(Q_H, Sigma_S, n_samples=20000, subset_size=20)
    cov_str = "PASS" if cov_result["pass"] else "FAIL"
    print(f"       relative Frobenius error = {cov_result['rel_frob']:.4f}  [{cov_str}]")

    print("  (b) VP-SDE forward marginal check (τ = 0.1, 0.5, 1.0; 2000 samples) …")
    vp_result = _vp_forward_marginal_check(
        Q_H, Sigma_S,
        taus=[0.1, 0.5, 1.0], n_samples=2000,
        beta_min=0.01, beta_max=50.0,
    )
    for tau, mean_re, max_re in zip(
        vp_result["taus"], vp_result["mean_rel_err"], vp_result["max_rel_err"]
    ):
        mean_ok = mean_re < vp_result["mean_threshold"]
        max_ok  = max_re  < vp_result["max_threshold"]
        flag = "PASS" if (mean_ok and max_ok) else "FAIL"
        print(f"       τ={tau:.2g}: mean rel err = {mean_re:.4f}   "
              f"max rel err = {max_re:.4f}  [{flag}]")
    vp_str = "PASS" if vp_result["pass"] else "FAIL"

    # ── 5. Save results ───────────────────────────────────────────────────────
    print("\n[5/5] Saving results …")

    out_dir = ROOT / "results" / "prior_params"
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = out_dir / f"{_BACKBONE}_{args.dataset}.yaml"
    yaml_data = {
        "backbone":     _BACKBONE,
        "dataset":      args.dataset,
        "seed":         args.seed,
        "standardized": bool(params["standardized"]),
        "epsilon_s":    float(params["epsilon_s"]),
        "eps_target":   float(args.eps_target),
        # Per-horizon de-standardisation factor (mph) — multiply standardised
        # samples by this to recover raw-mph residuals.
        "target_scale": [float(v) for v in params["target_scale"]],
        # Per-horizon variance of the modelled variable (≈ 1.0 when standardised).
        "delta_h":      [float(v) for v in params["delta_h"]],
        # Temporal kernel
        "alpha":  float(params["alpha"]),
        "rho1":   float(params["rho1"]),
        "rho2":   float(params["rho2"]),
        # Spatial GMRF
        "beta":   float(params["beta"]),
        # SDE
        "sigma":  float(params["sigma"]),
        # Sanity check results
        "checks": {
            "covariance_rel_frob": float(cov_result["rel_frob"]),
            "covariance_pass":     bool(cov_result["pass"]),
            # VP-SDE forward marginal: per-mode variance match against
            # `forward_sample` (the routine training uses).
            "vp_marginal_taus":         [float(t) for t in vp_result["taus"]],
            "vp_marginal_mean_rel_err": [float(r) for r in vp_result["mean_rel_err"]],
            "vp_marginal_max_rel_err":  [float(r) for r in vp_result["max_rel_err"]],
            "vp_marginal_beta_min":     float(vp_result["beta_min"]),
            "vp_marginal_beta_max":     float(vp_result["beta_max"]),
            "vp_marginal_pass":         bool(vp_result["pass"]),
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    print(f"  YAML saved: {yaml_path}")

    # ── Figure ────────────────────────────────────────────────────────────────
    # Use the same standardisation as the fit so empirical/fit panels are on
    # the same scale.
    if standardize:
        r_for_fig = r.astype(np.float64) / params["target_scale"][None, :, None]
    else:
        r_for_fig = r
    sig_S_emp = empirical_spatial_cov(r_for_fig, mask)
    Q_emp     = empirical_temporal_cov(r_for_fig, mask)
    Q_corr    = empirical_temporal_corr(Q_emp)

    fig_path = (
        ROOT / "experiments" / "prior-construction" / "figures"
        / f"fig_prior_samples_covariance_check_{_BACKBONE}_{args.dataset}.png"
    )
    _make_figure(
        dataset       = args.dataset,
        params        = params,
        Q_H           = Q_H,
        eigvals_L     = eigvals_L,
        eigvecs_L     = eigvecs_L,
        sig_S_emp     = sig_S_emp,
        Q_corr        = Q_corr,
        cov_result    = cov_result,
        vp_result     = vp_result,
        out_path      = fig_path,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    all_pass = cov_result["pass"] and vp_result["pass"]
    go_str   = "GO" if all_pass else "NO-GO"

    print(f"\n{'='*60}")
    print(f"  Covariance check       : {cov_str}")
    print(f"  VP forward-marginal    : {vp_str}")
    print(f"\n  *** {go_str} ***")
    print(f"{'='*60}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
