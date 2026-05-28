"""Section 3 — Experiment 3.2: Fit γ by Dirichlet-energy matching (Approach A).

For each dataset, choose γ such that the prior r ~ N(0, (L + γI)^{-1}) reproduces
the empirical mean Dirichlet energy from H2.

Equation:
    g(γ) := sum_k λ_k μ_k(γ) / sum_k μ_k(γ),    μ_k(γ) = 1 / (λ_k + γ).

This is monotone-decreasing in γ:
    g(0+)   → 0       (lowest-eigenvalue mode dominates)
    g(+∞)  → λ̄ = 1   (Cesàro limit; all modes weighted equally)

So for any target Ē_emp ∈ (0, λ̄) there is a unique γ*. We solve with brentq.

Per-horizon γ_h is also fit as a diagnostic to see how stable the prior smoothness
parameter is across horizon, but the locked γ* uses the average target.

Outputs to results/section3/<dataset>/exp32_energy_match/:
    summary.json — γ*_A, target, per-horizon γ_h, all eigenvalue sums.
    gamma_curve.png — g(γ) vs log γ with target line + fitted γ*.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from src.data.adjacency import (
    compute_normalized_laplacian,
    laplacian_eigendecomposition,
    load_adjacency,
)

DATASETS = ("metrla", "pemsbay", "pems08f")
OUT_ROOT = Path("results/section3")
H2_PATH_TMPL = "results/hypotheses/{ds}_seed0/summary.json"

GAMMA_LO, GAMMA_HI = 1e-6, 1e4


def g_of_gamma(gamma: float, eigvals: np.ndarray) -> float:
    """g(γ) = Σ λ_k/(λ_k+γ)  /  Σ 1/(λ_k+γ)."""
    denom = eigvals + gamma
    num = (eigvals / denom).sum()
    den = (1.0 / denom).sum()
    return float(num / den)


def fit_gamma(eigvals: np.ndarray, target: float) -> float:
    f = lambda g: g_of_gamma(g, eigvals) - target
    # g is monotone decreasing; verify bracket signs.
    f_lo = f(GAMMA_LO)
    f_hi = f(GAMMA_HI)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"No root in [{GAMMA_LO}, {GAMMA_HI}] for target={target}: "
            f"f(lo)={f_lo:.4f}, f(hi)={f_hi:.4f}"
        )
    return float(brentq(f, GAMMA_LO, GAMMA_HI, xtol=1e-10, rtol=1e-12))


def plot_gamma_curve(
    dataset: str,
    eigvals: np.ndarray,
    target: float,
    gamma_star: float,
    per_horizon_targets: list[float],
    per_horizon_gammas: list[float],
    out_dir: Path,
) -> None:
    gammas = np.logspace(-4, 3, 400)
    g_vals = np.array([g_of_gamma(g, eigvals) for g in gammas])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.semilogx(gammas, g_vals, label=r"$g(\gamma)$")
    ax1.axhline(target, color="red", lw=0.8, linestyle="--",
                label=fr"$\bar E_{{emp}} = {target:.4f}$")
    ax1.axvline(gamma_star, color="green", lw=0.8, linestyle=":",
                label=fr"$\gamma^\star_A = {gamma_star:.4f}$")
    ax1.set_xlabel(r"$\gamma$")
    ax1.set_ylabel(r"$g(\gamma)$ — mean normalised energy of prior")
    ax1.set_title(f"{dataset} — Approach A energy-match curve")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    horizons = np.arange(1, len(per_horizon_targets) + 1)
    ax2.plot(horizons, per_horizon_gammas, marker="o", color="C0", label=r"$\gamma_h$")
    ax2.axhline(gamma_star, color="green", lw=0.8, linestyle=":",
                label=fr"single-fit $\gamma^\star_A = {gamma_star:.4f}$")
    ax2.set_xlabel("horizon h")
    ax2.set_ylabel(r"$\gamma_h$")
    ax2.set_title("Per-horizon γ_h (diagnostic)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "gamma_curve.png", dpi=140)
    plt.close(fig)


def analyse(dataset: str) -> dict:
    A = load_adjacency(dataset)
    L = compute_normalized_laplacian(A)
    eigvals, _ = laplacian_eigendecomposition(L, dataset, cache=True)
    lambda_bar = float(eigvals.mean())

    with open(H2_PATH_TMPL.format(ds=dataset)) as f:
        h2 = json.load(f).get("h2") or json.load(open(H2_PATH_TMPL.format(ds=dataset))).get("H2")

    target = float(h2["energy_mean"])
    per_horizon_targets = [float(x) for x in h2["per_horizon_mean"]]

    gamma_star = fit_gamma(eigvals, target)
    per_horizon_gammas = [fit_gamma(eigvals, t) for t in per_horizon_targets]

    # Verification: g(γ*) should equal target.
    g_at_star = g_of_gamma(gamma_star, eigvals)

    return {
        "dataset": dataset,
        "N": int(eigvals.shape[0]),
        "lambda_bar": lambda_bar,
        "lambda_2": float(eigvals[1]),
        "lambda_N": float(eigvals[-1]),
        "target_E_emp": target,
        "target_per_horizon": per_horizon_targets,
        "gamma_star_A": gamma_star,
        "gamma_star_per_horizon": per_horizon_gammas,
        "gamma_h_min": min(per_horizon_gammas),
        "gamma_h_max": max(per_horizon_gammas),
        "gamma_h_range_ratio": max(per_horizon_gammas) / min(per_horizon_gammas),
        "g_at_gamma_star": g_at_star,
        "residual": g_at_star - target,
        "ratio_gamma_star_over_lambda_2": gamma_star / float(eigvals[1]),
    }


def main() -> None:
    rows = []
    for dataset in DATASETS:
        out_dir = OUT_ROOT / dataset / "exp32_energy_match"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = analyse(dataset)

        eigvals = np.load(f"data/processed/{dataset}_laplacian_eigs.npz")["eigenvalues"]
        plot_gamma_curve(
            dataset,
            eigvals,
            result["target_E_emp"],
            result["gamma_star_A"],
            result["target_per_horizon"],
            result["gamma_star_per_horizon"],
            out_dir,
        )
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
        rows.append(result)
        print(
            f"[{dataset}]  target={result['target_E_emp']:.4f}  "
            f"γ*_A={result['gamma_star_A']:.4f}  "
            f"(γ_h: {result['gamma_h_min']:.4f}→{result['gamma_h_max']:.4f}, "
            f"ratio={result['gamma_h_range_ratio']:.2f})  "
            f"γ*/λ₂={result['ratio_gamma_star_over_lambda_2']:.2f}"
        )

    summary_path = OUT_ROOT / "exp32_energy_match_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote cross-dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
