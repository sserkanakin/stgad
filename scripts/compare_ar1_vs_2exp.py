"""AR(1) vs two-exp temporal-kernel comparison on GWN training residuals.

Both kernels are fit by least-squares to the **same** lag-averaged ACF that
`src/diffusion/prior_fit.py::fit_temporal_kernel` already uses for the two-exp
fit, so the comparison is apples-to-apples. The two-exp fit is reused from the
existing fit_temporal_kernel; we additionally run a single-parameter AR(1) fit
with the same Nelder-Mead/sigmoid setup. The empirical lag curve is computed on
training residuals (consistent with the existing prior fit; documented in
results_ar1_vs_2exp.json metadata).

Outputs (relative to repo root):
    experiments/exp-002/fig_ar1_vs_2exp_lag_curve.pdf
    experiments/exp-002/fig_ar1_vs_2exp_implied_R.pdf
    experiments/exp-002/results_ar1_vs_2exp.json

Usage:
    python scripts/compare_ar1_vs_2exp.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.diffusion.prior_fit import (
    empirical_temporal_cov,
    empirical_temporal_corr,
    lag_averaged_acf,
    fit_temporal_kernel,
)


DATASETS = ["metrla", "pemsbay"]
DATASET_LABELS = {"metrla": "METR-LA", "pemsbay": "PEMS-BAY"}
H = 12
LAGS_FULL = np.arange(0, H, dtype=np.float64)   # k = 0..11, 12 points
LAGS_FIT  = np.arange(1, H, dtype=np.float64)   # k = 1..11, used for fitting


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def fit_ar1(acf_lags1to11: np.ndarray) -> float:
    """Fit a single AR(1) ρ to the lag-averaged ACF on lags 1..11.

    Same loss function and optimiser family as the two-exp fit
    (NLLS / Nelder-Mead / sigmoid parameterisation), so the two fits are
    directly comparable. ρ ∈ (0, 1) by sigmoid parameterisation.
    """
    def loss(p):
        rho = _sigmoid(p[0])
        pred = rho ** LAGS_FIT
        return float(np.sum((pred - acf_lags1to11) ** 2))

    starts = [-2.0, -1.0, 0.0, 1.0, 2.0, 2.94]   # cover ρ ∈ {.12, .27, .5, .73, .88, .95}
    best = None
    for x0 in starts:
        res = minimize(loss, x0=[x0], method="Nelder-Mead",
                       options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 200_000})
        if best is None or res.fun < best.fun:
            best = res
    return float(_sigmoid(best.x[0]))


# ─────────────────────────────────────────────────────────────────────────────
# Goodness-of-fit
# ─────────────────────────────────────────────────────────────────────────────

def gof_metrics(empirical_full: np.ndarray, fit_full: np.ndarray, k_params: int) -> dict:
    """SSR, RMSE, max-abs residual, AIC, BIC over the full 12-point lag curve.

    `empirical_full[0]` and `fit_full[0]` are both 1.0 by construction (lag-0
    autocorrelation), so they contribute zero to the SSR. n=12 is used for the
    table per Phase 1 spec.

    AIC / BIC use the standard Gaussian-MLE form with n=12:
        AIC = n * log(SSR / n) + 2 * k
        BIC = n * log(SSR / n) + k * log(n)
    Constants that depend only on n are dropped (cancel in differences).
    """
    n = len(empirical_full)
    resid = empirical_full - fit_full
    ssr = float(np.sum(resid ** 2))
    rmse = float(np.sqrt(ssr / n))
    max_abs = float(np.max(np.abs(resid)))
    # log(SSR/n) — guard against SSR=0 by clamping
    ll_term = n * float(np.log(max(ssr / n, 1e-30)))
    aic = ll_term + 2.0 * k_params
    bic = ll_term + k_params * float(np.log(n))
    return {"ssr": ssr, "rmse": rmse, "max_abs_lag_resid": max_abs,
            "aic": aic, "bic": bic}


def implied_R(rho_func) -> np.ndarray:
    """Build an H×H correlation matrix from a function f(|h-h'|) -> ρ."""
    lags2d = np.abs(np.arange(H)[:, None] - np.arange(H)[None, :])
    return rho_func(lags2d)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset run
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(dataset: str) -> dict:
    """Compute everything we need for one dataset; return a result dict."""
    res_path = ROOT / "results" / "residuals" / f"gwn_{dataset}_train_seed0.npz"
    data = np.load(res_path)
    r = data["residuals"].squeeze(-1).astype(np.float32)   # (N_win, 12, N)
    mask = data["mask"].squeeze(-1)                        # (N_win, 12, N)
    n_windows, _, n_sensors = r.shape

    # Empirical lag curve — same path as fit_temporal_kernel
    Q_emp = empirical_temporal_cov(r, mask)                # (12, 12)
    Q_corr = empirical_temporal_corr(Q_emp)                # (12, 12) correlation
    acf_1to11 = lag_averaged_acf(Q_corr)                   # (11,) lags 1..11

    # Empirical full curve including k=0 = 1.0
    acf_full = np.concatenate([[1.0], acf_1to11])          # (12,)

    # AR(1) fit
    rho_ar1 = fit_ar1(acf_1to11)
    fit_ar1_full = rho_ar1 ** LAGS_FULL                    # (12,)

    # Two-exp fit — reuse existing fitter for exact parity with the prior
    alpha, rho1, rho2 = fit_temporal_kernel(Q_corr)
    fit_2exp_full = alpha * rho1 ** LAGS_FULL + (1.0 - alpha) * rho2 ** LAGS_FULL

    # Confirm against stored YAML
    yaml_path = ROOT / "results" / "prior_params" / f"gwn_{dataset}.yaml"
    stored = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else {}
    yaml_match = (
        np.isclose(stored.get("alpha", -1), alpha, atol=1e-6)
        and np.isclose(stored.get("rho1", -1), rho1, atol=1e-6)
        and np.isclose(stored.get("rho2", -1), rho2, atol=1e-6)
    )

    # Goodness-of-fit
    gof_ar1  = gof_metrics(acf_full, fit_ar1_full,  k_params=1)
    gof_2exp = gof_metrics(acf_full, fit_2exp_full, k_params=3)

    # Implied 12×12 R
    R_emp  = Q_corr.astype(np.float64)
    R_ar1  = implied_R(lambda L, r=rho_ar1: r ** L)
    R_2exp = implied_R(
        lambda L, a=alpha, r1=rho1, r2=rho2: a * r1 ** L + (1.0 - a) * r2 ** L
    )
    frob_ar1  = float(np.linalg.norm(R_ar1  - R_emp, "fro"))
    frob_2exp = float(np.linalg.norm(R_2exp - R_emp, "fro"))

    return {
        "dataset": dataset,
        "n_windows": int(n_windows),
        "n_sensors": int(n_sensors),
        "acf_full": acf_full.tolist(),                # length 12 (k=0..11)
        "ar1": {
            "params": {"rho": rho_ar1},
            "fit_full": fit_ar1_full.tolist(),
            **gof_ar1,
            "frob_R_minus_emp": frob_ar1,
        },
        "two_exp": {
            "params": {"alpha": alpha, "rho1": rho1, "rho2": rho2},
            "fit_full": fit_2exp_full.tolist(),
            **gof_2exp,
            "frob_R_minus_emp": frob_2exp,
            "yaml_params_match": bool(yaml_match),
        },
        "matrices": {
            "R_emp":  R_emp.tolist(),
            "R_ar1":  R_ar1.tolist(),
            "R_2exp": R_2exp.tolist(),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _plot_lag_curves(results: dict, out_path: Path) -> None:
    """Two-panel plot: empirical scatter + AR(1) line + two-exp line per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        d = results[dataset]
        emp = np.array(d["acf_full"])
        ax.scatter(LAGS_FULL, emp, s=40, color="black", zorder=3,
                   label="Empirical lag-avg ACF")
        rho_ar1 = d["ar1"]["params"]["rho"]
        a, r1, r2 = (d["two_exp"]["params"]["alpha"],
                     d["two_exp"]["params"]["rho1"],
                     d["two_exp"]["params"]["rho2"])
        x_smooth = np.linspace(0, H - 1, 300)
        ax.plot(x_smooth, rho_ar1 ** x_smooth, color="tab:red",
                lw=1.7, label=f"AR(1)  ρ={rho_ar1:.3f}")
        ax.plot(x_smooth, a * r1 ** x_smooth + (1.0 - a) * r2 ** x_smooth,
                color="tab:blue", lw=1.7,
                label=f"2-exp  α={a:.2f}, ρ₁={r1:.3f}, ρ₂={r2:.3f}")
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("lag k")
        ax.set_xticks(np.arange(0, 12, 1))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("lag-averaged correlation")
    fig.suptitle("Empirical lag-averaged residual ACF — AR(1) vs two-exp fit",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_implied_R(results: dict, out_path: Path) -> None:
    """5-column heatmap grid: R_emp, R_AR1, R_2exp, R_AR1−R_emp, R_2exp−R_emp.

    Rows = datasets. The two diff columns are the persuasive part — AR(1)'s
    diff should show structure, two-exp's diff should look like noise.
    """
    n_rows = len(DATASETS)
    fig, axes = plt.subplots(n_rows, 5, figsize=(16, 3.4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = [
        "R_emp", "R_AR(1)", "R_2-exp",
        "R_AR(1) − R_emp", "R_2-exp − R_emp",
    ]
    for row, dataset in enumerate(DATASETS):
        d = results[dataset]
        R_emp  = np.array(d["matrices"]["R_emp"])
        R_ar1  = np.array(d["matrices"]["R_ar1"])
        R_2exp = np.array(d["matrices"]["R_2exp"])
        diff_ar1  = R_ar1  - R_emp
        diff_2exp = R_2exp - R_emp

        # Common scale for first 3 columns (correlation matrices)
        mats_corr = [R_emp, R_ar1, R_2exp]
        vmin_c, vmax_c = 0.0, 1.0

        # Common diverging scale for the two diff columns
        diff_max = float(max(np.abs(diff_ar1).max(), np.abs(diff_2exp).max()))
        diff_max = max(diff_max, 1e-3)

        mats_all = mats_corr + [diff_ar1, diff_2exp]
        for col, mat in enumerate(mats_all):
            ax = axes[row, col]
            if col < 3:
                im = ax.imshow(mat, vmin=vmin_c, vmax=vmax_c, cmap="viridis",
                               aspect="equal")
            else:
                im = ax.imshow(mat, vmin=-diff_max, vmax=diff_max,
                               cmap="RdBu_r", aspect="equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(DATASET_LABELS[dataset], fontsize=11,
                              rotation=90, labelpad=8)
            ax.set_xticks([0, 5, 11])
            ax.set_yticks([0, 5, 11])
    fig.suptitle("Implied 12×12 horizon correlation: empirical vs AR(1) vs two-exp",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown table
# ─────────────────────────────────────────────────────────────────────────────

def make_markdown_table(results: dict) -> str:
    """Single markdown table covering both datasets × both models."""
    header = (
        "| Dataset  | Model  | Params | SSR        | RMSE       | "
        "Max abs lag-resid | ‖R−R_emp‖_F | AIC      | BIC      |\n"
        "|----------|--------|-------:|-----------:|-----------:|"
        "------------------:|------------:|---------:|---------:|"
    )
    rows = []
    for dataset in DATASETS:
        d = results[dataset]
        for mname, mkey, kp in [("AR(1)", "ar1", 1), ("two-exp", "two_exp", 3)]:
            m = d[mkey]
            rows.append(
                f"| {DATASET_LABELS[dataset]:<8s} | {mname:<6s} | {kp:>6d} | "
                f"{m['ssr']:>10.6f} | {m['rmse']:>10.6f} | "
                f"{m['max_abs_lag_resid']:>17.6f} | {m['frob_R_minus_emp']:>11.6f} | "
                f"{m['aic']:>8.3f} | {m['bic']:>8.3f} |"
            )
    return header + "\n" + "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Stop-condition check
# ─────────────────────────────────────────────────────────────────────────────

def check_stop_conditions(results: dict) -> list[str]:
    """Return a list of human-readable stop-condition flags (empty if none)."""
    flags = []
    for dataset in DATASETS:
        d = results[dataset]
        a = d["two_exp"]["params"]["alpha"]
        r1 = d["two_exp"]["params"]["rho1"]
        r2 = d["two_exp"]["params"]["rho2"]
        ar1_rmse = d["ar1"]["rmse"]
        x2_rmse = d["two_exp"]["rmse"]
        ar1_aic = d["ar1"]["aic"]
        x2_aic = d["two_exp"]["aic"]
        ar1_bic = d["ar1"]["bic"]
        x2_bic = d["two_exp"]["bic"]

        # 1. Degenerate two-exp
        if a < 0.05 or a > 0.95 or abs(r1 - r2) < 0.03:
            flags.append(
                f"[{DATASET_LABELS[dataset]}] two-exp may be DEGENERATE: "
                f"α={a:.4f}, ρ₁={r1:.4f}, ρ₂={r2:.4f}  "
                f"(α∉[0.05,0.95] or |ρ₁−ρ₂|<0.03)"
            )
        # 2. Two-exp wins SSR but loses AIC or BIC
        if x2_rmse < ar1_rmse and (x2_aic > ar1_aic or x2_bic > ar1_bic):
            flags.append(
                f"[{DATASET_LABELS[dataset]}] two-exp lower SSR but loses on "
                f"AIC ({x2_aic:.3f} vs {ar1_aic:.3f}) or BIC "
                f"({x2_bic:.3f} vs {ar1_bic:.3f})"
            )
        # 3. Both fits within numerical noise
        if ar1_rmse < 0.005 and x2_rmse < 0.005:
            flags.append(
                f"[{DATASET_LABELS[dataset]}] both RMSE < 0.005 — upgrade may "
                f"be cosmetic (AR(1) RMSE={ar1_rmse:.5f}, "
                f"two-exp RMSE={x2_rmse:.5f})"
            )
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("AR(1) vs two-exp temporal-kernel comparison (GWN train residuals)")
    print("=" * 72)

    results = {}
    for ds in DATASETS:
        print(f"\n--- {DATASET_LABELS[ds]} ---")
        results[ds] = process_dataset(ds)
        d = results[ds]
        print(f"  n_windows={d['n_windows']}  n_sensors={d['n_sensors']}")
        print(f"  AR(1):    ρ={d['ar1']['params']['rho']:.5f}")
        print(f"  two-exp:  α={d['two_exp']['params']['alpha']:.4f}  "
              f"ρ₁={d['two_exp']['params']['rho1']:.5f}  "
              f"ρ₂={d['two_exp']['params']['rho2']:.5f}  "
              f"(yaml match: {d['two_exp']['yaml_params_match']})")
        print(f"  AR(1)   : SSR={d['ar1']['ssr']:.6f}  RMSE={d['ar1']['rmse']:.6f}  "
              f"max|resid|={d['ar1']['max_abs_lag_resid']:.6f}  "
              f"‖R−R_emp‖_F={d['ar1']['frob_R_minus_emp']:.6f}  "
              f"AIC={d['ar1']['aic']:.3f}  BIC={d['ar1']['bic']:.3f}")
        print(f"  two-exp : SSR={d['two_exp']['ssr']:.6f}  RMSE={d['two_exp']['rmse']:.6f}  "
              f"max|resid|={d['two_exp']['max_abs_lag_resid']:.6f}  "
              f"‖R−R_emp‖_F={d['two_exp']['frob_R_minus_emp']:.6f}  "
              f"AIC={d['two_exp']['aic']:.3f}  BIC={d['two_exp']['bic']:.3f}")

    out_dir = ROOT / "experiments" / "exp-002"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figures
    fig1 = out_dir / "fig_ar1_vs_2exp_lag_curve.pdf"
    fig2 = out_dir / "fig_ar1_vs_2exp_implied_R.pdf"
    _plot_lag_curves(results, fig1)
    _plot_implied_R(results, fig2)
    print(f"\nFigures saved:")
    print(f"  {fig1}")
    print(f"  {fig2}")

    # JSON
    metadata = {
        "split": "train",
        "backbone": "gwn",
        "seed": 0,
        "n_lag_points": int(H),
        "lag_points_for_table": "k=0..11 (12 points; k=0 contributes 0 to SSR)",
        "fit_method": ("Nelder-Mead least-squares on lag-averaged ACF (lags 1..11), "
                       "sigmoid parameterisation, multiple starts. AR(1) and two-exp "
                       "use the same loss family; two-exp fit is reused from "
                       "src/diffusion/prior_fit.py::fit_temporal_kernel."),
        "aic_bic_form": "n*log(SSR/n) + {2*k | k*log(n)}; constants in n dropped.",
    }
    json_path = out_dir / "results_ar1_vs_2exp.json"
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)
    print(f"  {json_path}")

    # Markdown table
    print("\n" + "=" * 72)
    print("Markdown table (paste into exp-002 Results section):")
    print("=" * 72)
    print(make_markdown_table(results))

    # Stop conditions
    flags = check_stop_conditions(results)
    print("\n" + "=" * 72)
    print("Stop-condition check:")
    print("=" * 72)
    if not flags:
        print("  No stop conditions triggered.")
    else:
        for f in flags:
            print(f"  ⚠ {f}")


if __name__ == "__main__":
    main()
