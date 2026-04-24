"""
Diagnostic plots for temporal correlation structure of de-meaned traffic speed.

Evaluates three candidate generative stories for the temporal factor R_H
of the Kronecker prior:
  (a) AR(1) + measurement noise / ARMA(1,1)
  (b) Sum of two AR(1) processes at different timescales
  (c) Long-memory / fractional process with power-law ACF

Five plots saved to results/figures/eda/:
  fig_acf_loglinear.png      — log(ACF) vs lag, AR(1) line fit
  fig_acf_loglog.png         — log(ACF) vs log(lag), power-law line fit
  fig_acf_two_ar_fit.png     — 1-AR(1) vs 2-AR(1) mixture
  fig_acf_fit_residuals.png  — residuals of all three candidate models
  fig_pacf.png               — PACF to lag 30 with 95% CI

Usage:
    python scripts/diagnose_temporal_acf.py
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

COLORS = {"metrla": "#4C72B0", "pemsbay": "#55A868"}
LABELS = {"metrla": "METR-LA", "pemsbay": "PEMS-BAY"}
FIG_DIR = Path("results/figures/eda")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAX_LAG = 72
LAGS = np.arange(1, MAX_LAG + 1, dtype=float)   # 1 … 72

_RAW_PATHS = {
    "metrla":  "data/raw/metr-la.h5",
    "pemsbay": "data/raw/pems-bay.h5",
}

# Clean white-background style
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#444444",
    "axes.linewidth":    0.9,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "grid.color":        "#E2E2E2",
    "grid.linewidth":    0.6,
    "font.size":         10,
    "font.family":       "sans-serif",
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.facecolor": "white",
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "legend.framealpha": 0.93,
    "legend.edgecolor":  "#CCCCCC",
    "legend.fontsize":   9,
})

# ---------------------------------------------------------------------------
# Data loading — de-meaned ACF and PACF
# ---------------------------------------------------------------------------

def _n_train(T: int) -> int:
    return round(T * 0.7)


def load_acf_and_pacf(dataset: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load raw .h5, compute de-meaned training speed, return:
      acf_full  : (MAX_LAG+1,) sensor-averaged de-meaned ACF, lags 0..72
      pacf_vals : (31,) PACF of spatially-averaged series, lags 0..30
      n_train   : number of training time steps (used for CI width)
    """
    df = pd.read_hdf(_RAW_PATHS[dataset])
    T, N = df.shape
    n_tr = _n_train(T)

    speed = df.values[:n_tr].astype(float)           # (n_tr, N)
    missing_rate = (df.values == 0.0).mean(axis=0)   # (N,)
    good = np.where(missing_rate < 0.5)[0]           # sensors with <50% missing

    # Time-of-day slot index (0..287, 5-min resolution)
    tod = ((df.index[:n_tr].values
            - df.index[:n_tr].values.astype("datetime64[D]"))
           / np.timedelta64(5, "m")).astype(int)

    # Per-sensor per-slot mean (NaN for sensors with no valid readings in slot)
    slot_means = np.zeros((288, N))
    for slot in range(288):
        mask = tod == slot
        if mask.any():
            v = np.where(speed[mask] == 0, np.nan, speed[mask])
            slot_means[slot] = np.nanmean(v, axis=0)

    # Subtract diurnal mean to get de-meaned speed
    demeaned = speed.copy()
    for t in range(n_tr):
        demeaned[t] -= slot_means[tod[t]]

    # ---- sensor-level ACF, then average ----
    acf_mat = np.zeros((len(good), MAX_LAG + 1))
    for i, s in enumerate(good):
        raw_s = speed[:, s].copy()
        raw_s[raw_s == 0] = np.nan
        valid = ~np.isnan(raw_s)
        if valid.sum() > MAX_LAG * 3:
            dem_s = demeaned[:, s][valid]
            if dem_s.std() > 1e-4:
                acf_mat[i] = sm_acf(dem_s, nlags=MAX_LAG, fft=True)
    mean_acf = acf_mat.mean(axis=0)   # (MAX_LAG+1,) — lag 0 is 1.0

    # ---- spatially-averaged de-meaned series for PACF ----
    # Average over non-missing sensors at each time step (no gaps in result)
    avg = np.zeros(n_tr)
    cnt = np.zeros(n_tr, dtype=float)
    for s in good:
        valid_mask = speed[:, s] != 0.0
        avg[valid_mask] += demeaned[valid_mask, s]
        cnt[valid_mask] += 1.0
    cnt = np.maximum(cnt, 1.0)
    avg /= cnt    # (n_tr,) — no NaNs

    pacf_vals = sm_pacf(avg, nlags=30, method="ywadjusted")

    return mean_acf, pacf_vals, n_tr


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _fit_single_ar1(acf_1to72: np.ndarray) -> tuple[float, np.ndarray, float]:
    """OLS fit of log(ACF) vs lag.
    Returns (rho, acf_pred, rmse)  where  ACF(h) ≈ exp(b) · rho^h.
    """
    log_a = np.log(np.maximum(acf_1to72, 1e-10))
    A = np.column_stack([LAGS, np.ones(MAX_LAG)])
    c, *_ = np.linalg.lstsq(A, log_a, rcond=None)
    slope, intercept = float(c[0]), float(c[1])
    rho = float(np.exp(slope))
    pred = float(np.exp(intercept)) * rho ** LAGS
    rmse = float(np.sqrt(np.mean((acf_1to72 - pred) ** 2)))
    return rho, pred, rmse


def _fit_two_ar1(acf_1to72: np.ndarray) -> tuple:
    """Nonlinear LS: ACF(h) = alpha·ρ₁^h + (1-alpha)·ρ₂^h, ρ₁ ≤ ρ₂.
    Returns (alpha, rho1, rho2, acf_pred, rmse, is_degenerate).

    Strategy: first try with min-separation 0.10; if nothing converges, retry with 0.02.
    is_degenerate=True means the best solution has |ρ₂−ρ₁| < 0.05 (collapses to ~single AR1).
    ρ capped at 0.9995 to prevent unit-root.
    """
    RHO_MAX = 0.9995

    def _model(h, alpha, rho1, rho2):
        return alpha * rho1 ** h + (1 - alpha) * rho2 ** h

    def _search(min_sep: float):
        best_rmse = np.inf
        best = None
        for a0 in (0.2, 0.4, 0.5, 0.6, 0.8):
            for r1_0 in (0.05, 0.15, 0.30, 0.50, 0.70):
                for r2_0 in (0.60, 0.75, 0.88, 0.95, 0.9990):
                    if r2_0 - r1_0 < min_sep:
                        continue
                    try:
                        popt, _ = curve_fit(
                            _model, LAGS, acf_1to72,
                            p0=[a0, r1_0, r2_0],
                            bounds=([0.0, 1e-6, 1e-6], [1.0, RHO_MAX, RHO_MAX]),
                            maxfev=10_000, method="trf",
                        )
                        alpha, rho1, rho2 = popt
                        if rho1 > rho2:
                            alpha, rho1, rho2 = 1 - alpha, rho2, rho1
                        pred = _model(LAGS, alpha, rho1, rho2)
                        rmse = float(np.sqrt(np.mean((acf_1to72 - pred) ** 2)))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best = (float(alpha), float(rho1), float(rho2), pred.copy(), rmse)
                    except Exception:
                        pass
        return best

    best = _search(min_sep=0.10)
    if best is None:
        best = _search(min_sep=0.02)

    if best is None:
        # Complete fallback: single-component guess
        rho_single, pred_single, rmse_single = _fit_single_ar1(acf_1to72)
        best = (1.0, float(rho_single), float(rho_single), pred_single, rmse_single)

    alpha, rho1, rho2, pred, rmse = best
    is_degenerate = abs(rho2 - rho1) < 0.05
    return alpha, rho1, rho2, pred, rmse, is_degenerate


def _fit_power_law(acf_1to72: np.ndarray) -> tuple:
    """OLS in log-log space: ACF(h) = c · h^(−β).
    Long-memory parameter d = (1 − β) / 2 (from ACF(h) ~ h^{2d−1}).
    Returns (beta, c, d, acf_pred, rmse).
    """
    log_a = np.log(np.maximum(acf_1to72, 1e-10))
    log_l = np.log(LAGS)
    A = np.column_stack([log_l, np.ones(MAX_LAG)])
    c, *_ = np.linalg.lstsq(A, log_a, rcond=None)
    slope, intercept = float(c[0]), float(c[1])
    beta = float(-slope)              # ACF = coef_c · h^slope = coef_c · h^{-beta}
    coef_c = float(np.exp(intercept))
    d = float((1.0 - beta) / 2.0)    # slope = 2d−1 = −beta → d = (1−beta)/2
    pred = coef_c * LAGS ** (-beta)
    rmse = float(np.sqrt(np.mean((acf_1to72 - pred) ** 2)))
    return beta, coef_c, d, pred, rmse


# ---------------------------------------------------------------------------
# Plot 1 — Log-linear
# ---------------------------------------------------------------------------

def plot1_loglinear(acfs: dict) -> dict:
    """log(ACF) vs lag. Straight line ↔ pure AR(1)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    rho_fits = {}
    for ds, acf_full in acfs.items():
        acf_dem = acf_full[1:]   # lags 1..72
        rho, pred, rmse = _fit_single_ar1(acf_dem)
        rho_fits[ds] = rho

        valid = acf_dem > 1e-10
        ax.plot(LAGS[valid], np.log(acf_dem[valid]),
                color=COLORS[ds], linewidth=2.0, zorder=3,
                label=f"{LABELS[ds]}  (empirical)")
        ax.plot(LAGS, np.log(np.maximum(pred, 1e-12)),
                color=COLORS[ds], linewidth=1.3, linestyle="--", zorder=2,
                label=f"{LABELS[ds]}  OLS AR(1)  ρ = {rho:.4f}  RMSE = {rmse:.4f}")

    ax.axhline(0, color="#AAAAAA", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Lag  (5-min steps)")
    ax.set_ylabel("log(ACF)")
    ax.set_title(
        "Log-linear ACF  ·  straight line ↔ pure AR(1)\n"
        "Empirical curve above the fit at large lags → slower decay than AR(1) predicts"
    )
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_acf_loglinear.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved  fig_acf_loglinear.png")
    return rho_fits


# ---------------------------------------------------------------------------
# Plot 2 — Log-log
# ---------------------------------------------------------------------------

def plot2_loglog(acfs: dict) -> dict:
    """log(ACF) vs log(lag). Straight line ↔ long-memory power-law process."""
    fig, ax = plt.subplots(figsize=(8, 5))

    loglog_fits = {}
    for ds, acf_full in acfs.items():
        acf_dem = acf_full[1:]
        beta, c, d, pred, rmse = _fit_power_law(acf_dem)
        loglog_fits[ds] = (beta, c, d, pred, rmse)

        valid = acf_dem > 1e-10
        ax.plot(np.log(LAGS[valid]), np.log(acf_dem[valid]),
                color=COLORS[ds], linewidth=2.0, zorder=3,
                label=f"{LABELS[ds]}  (empirical)")
        ax.plot(np.log(LAGS), np.log(np.maximum(pred, 1e-12)),
                color=COLORS[ds], linewidth=1.3, linestyle="--", zorder=2,
                label=f"{LABELS[ds]}  power-law  β={beta:.3f}  d={d:.3f}  RMSE={rmse:.4f}")

    ax.set_xlabel("log(lag)")
    ax.set_ylabel("log(ACF)")
    ax.set_title(
        "Log-log ACF  ·  straight line ↔ long-memory (power-law)\n"
        "Convex curve bending toward −∞ at large lags → AR(1) / mixture, not long memory"
    )
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_acf_loglog.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved  fig_acf_loglog.png")
    return loglog_fits


# ---------------------------------------------------------------------------
# Plot 3 — Two-AR(1) mixture
# ---------------------------------------------------------------------------

def plot3_two_ar_fit(acfs: dict) -> dict:
    """Per-dataset: empirical ACF, single AR(1) fit, and 2-AR(1) mixture fit."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    two_ar_fits = {}
    for ax, (ds, acf_full) in zip(axes, acfs.items()):
        acf_dem = acf_full[1:]

        rho, pred_1ar, rmse_1ar = _fit_single_ar1(acf_dem)
        alpha, rho1, rho2, pred_2ar, rmse_2ar, degen = _fit_two_ar1(acf_dem)
        two_ar_fits[ds] = (alpha, rho1, rho2, pred_2ar, rmse_2ar, degen, rho, pred_1ar, rmse_1ar)

        degen_note = "  [degenerate ≈ 1-AR1]" if degen else ""
        ax.plot(LAGS, acf_dem,
                color=COLORS[ds], linewidth=2.2, zorder=4,
                label="Empirical ACF")
        ax.plot(LAGS, pred_1ar,
                color="#777777", linewidth=1.5, linestyle="--", zorder=3,
                label=f"1-AR(1)   ρ={rho:.4f}   RMSE={rmse_1ar:.5f}")
        ax.plot(LAGS, pred_2ar,
                color="#CC3322", linewidth=1.5, linestyle="-.", zorder=3,
                label=(f"2-AR(1)   α={alpha:.3f}   ρ₁={rho1:.4f}   ρ₂={rho2:.4f}{degen_note}\n"
                       f"           RMSE={rmse_2ar:.5f}"))

        ax.set_xlabel("Lag  (5-min steps)")
        ax.set_ylabel("ACF")
        ax.set_title(f"{LABELS[ds]}  —  Two-AR(1) mixture fit")
        ax.legend()
        ax.grid(True)
        ax.set_ylim(bottom=-0.05)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_acf_two_ar_fit.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved  fig_acf_two_ar_fit.png")
    return two_ar_fits


# ---------------------------------------------------------------------------
# Plot 4 — Fit residuals
# ---------------------------------------------------------------------------

def plot4_residuals(acfs: dict) -> dict:
    """Per-dataset residuals: empirical ACF − model ACF for all three candidates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    rmse_table = {}
    all_fits = {}   # store for summary use
    for ax, (ds, acf_full) in zip(axes, acfs.items()):
        acf_dem = acf_full[1:]

        rho,  pred_1ar, rmse_1ar = _fit_single_ar1(acf_dem)
        _, _, _, pred_2ar, rmse_2ar, _ = _fit_two_ar1(acf_dem)
        beta, c, d, pred_pl, rmse_pl = _fit_power_law(acf_dem)

        rmse_table[ds] = {"1-AR(1)": rmse_1ar, "2-AR(1)": rmse_2ar, "Power-law": rmse_pl}
        all_fits[ds]   = (rho, beta, c, d)

        resid_1ar = acf_dem - pred_1ar
        resid_2ar = acf_dem - pred_2ar
        resid_pl  = acf_dem - pred_pl

        ax.plot(LAGS, resid_1ar,
                color="#7777BB", linewidth=1.6, linestyle="--",
                label=f"1-AR(1)    RMSE = {rmse_1ar:.5f}")
        ax.plot(LAGS, resid_2ar,
                color="#CC3322", linewidth=1.6, linestyle="-.",
                label=f"2-AR(1)    RMSE = {rmse_2ar:.5f}")
        ax.plot(LAGS, resid_pl,
                color="#9B59B6", linewidth=1.6, linestyle=":",
                label=f"Power-law  RMSE = {rmse_pl:.5f}")

        ax.axhline(0, color="#333333", linewidth=0.9, zorder=5)
        ax.set_xlabel("Lag  (5-min steps)")
        ax.set_ylabel("Empirical ACF  −  model ACF")
        ax.set_title(f"{LABELS[ds]}  —  Fit residuals  (flatter = better fit)")
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_acf_fit_residuals.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved  fig_acf_fit_residuals.png")

    # ---- RMSE table to stdout ----
    w = 13
    sep = "─" * (w + 3 * (w + 3) + 4)
    print(f"\n{'':>{w}} {'RMSE summary  (lags 1–72)':^{3*(w+3)}}")
    print(sep)
    header = f"{'Dataset':<{w}} | {'1-AR(1)':>{w}} | {'2-AR(1)':>{w}} | {'Power-law':>{w}}"
    print(header)
    print(sep)
    for ds, r in rmse_table.items():
        best = min(r, key=r.get)
        row = f"{LABELS[ds]:<{w}}"
        for key in ("1-AR(1)", "2-AR(1)", "Power-law"):
            tag = "← best" if key == best else ""
            row += f" | {r[key]:>{w-7}.5f} {tag:<6}"
        print(row)
    print(sep)

    return rmse_table


# ---------------------------------------------------------------------------
# Plot 5 — PACF
# ---------------------------------------------------------------------------

def plot5_pacf(pacf_info: dict) -> None:
    """PACF bar chart with 95% CI. AR(1) → spike at lag 1 only; long memory → slow decay."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (ds, (pacf_vals, N)) in zip(axes, pacf_info.items()):
        lags_p = np.arange(len(pacf_vals))   # 0..30
        conf = 1.96 / np.sqrt(N)

        ax.bar(lags_p, pacf_vals,
               color=COLORS[ds], alpha=0.75, width=0.65, zorder=2,
               label=LABELS[ds])
        ax.axhline( conf, color="tomato", linewidth=1.4, linestyle="--",
                    label=f"95% CI  ±{conf:.4f}  (N = {N:,})")
        ax.axhline(-conf, color="tomato", linewidth=1.4, linestyle="--")
        ax.axhline(0, color="#444444", linewidth=0.7, zorder=3)

        # Annotate number of significant lags
        n_sig = int(np.sum(np.abs(pacf_vals[1:]) > conf))
        ax.text(0.98, 0.95, f"{n_sig} lags > CI\n(excl. lag 0)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"))

        ax.set_xlabel("Lag  (5-min steps)")
        ax.set_ylabel("Partial ACF")
        ax.set_title(
            f"{LABELS[ds]}  —  PACF\n"
            "(spatially-averaged de-meaned speed, training set)"
        )
        ax.legend()
        ax.grid(True, axis="y")
        ax.set_xlim(-0.5, 30.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_pacf.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved  fig_pacf.png")


# ---------------------------------------------------------------------------
# Plain-language summary
# ---------------------------------------------------------------------------

def print_summary(
    acfs: dict,
    rmse_table: dict,
    rho_1ar: dict,
    loglog_fits: dict,
    two_ar_fits: dict,
    pacf_info: dict,
) -> None:
    """Print findings for (i) best model, (ii) recommended parameters, (iii) cross-dataset."""

    line = "=" * 70
    print(f"\n{line}")
    print("  TEMPORAL CORRELATION STRUCTURE  —  DIAGNOSTIC SUMMARY")
    print(line)

    preferred = {}
    param_summaries = {}

    for ds in ("metrla", "pemsbay"):
        r = rmse_table[ds]
        best_key = min(r, key=r.get)
        preferred[ds] = best_key

        rho         = rho_1ar[ds]
        beta, c, d, _, rmse_pl  = loglog_fits[ds]
        alpha, rho1, rho2, _, rmse_2ar, degen, rho_single, _, rmse_1ar = two_ar_fits[ds]
        pacf_vals, N = pacf_info[ds]
        conf        = 1.96 / np.sqrt(N)
        n_sig_pacf  = int(np.sum(np.abs(pacf_vals[1:]) > conf))

        imp_2ar = (rmse_1ar - rmse_2ar) / rmse_1ar * 100 if not degen else 0.0
        imp_pl  = (rmse_1ar - rmse_pl)  / rmse_1ar * 100

        print(f"\n{'─'*60}")
        print(f"  {LABELS[ds]}")
        print(f"{'─'*60}")
        print(f"  Single AR(1)  ρ = {rho:.4f}                    RMSE = {rmse_1ar:.5f}")
        degen_tag = "  [degenerate]" if degen else ""
        print(f"  Two-AR(1)     α={alpha:.3f}  ρ₁={rho1:.4f}  ρ₂={rho2:.4f}   RMSE = {rmse_2ar:.5f}  ({imp_2ar:+.1f}% vs 1-AR1){degen_tag}")
        print(f"  Power-law     β={beta:.3f}  c={c:.4f}  d={d:.3f}          RMSE = {rmse_pl:.5f}  ({imp_pl:+.1f}% vs 1-AR1)")
        print(f"  PACF significant lags (excl. 0): {n_sig_pacf} of 30")
        print(f"  → Best fit by RMSE: {best_key}")
        print()

        # --- (i) Generative story ---
        if best_key == "2-AR(1)" and imp_2ar > 15 and not degen:
            story = "two-timescale AR(1) mixture  (story b)"
        elif best_key == "Power-law" and d > 0.05 and imp_pl > 5:
            story = "long-memory / fractional process  (story c)"
        else:
            story = "approximately AR(1) with measurement noise / ARMA(1,1)  (story a)"

        # --- (ii) Recommended R_H parameters ---
        if "two-timescale" in story:
            params = (
                f"Mixture prior: α={alpha:.3f} weight on fast component (ρ₁={rho1:.4f}) "
                f"+ (1−α)={(1-alpha):.3f} weight on slow component (ρ₂={rho2:.4f}).\n"
                f"     Alternatively model R_H as an ARMA(2,1) kernel with the same poles."
            )
        elif "fractional" in story:
            params = (
                f"Fractional-decay prior with d≈{d:.3f}; horizon-dependent bandwidth "
                f"scaling h^{{2d−1}} ≈ h^{{{2*d-1:.3f}}}.\n"
                f"     Effective single-timescale approximation: ρ_eff ≈ {rho:.4f}."
            )
        else:
            params = (
                f"Single-timescale exponential prior with ρ={rho:.4f}.\n"
                f"     If PACF shows a lag-2 spike, add an MA(1) correction term."
            )

        param_summaries[ds] = params
        preferred[ds] = story          # overwrite best_key with the interpreted story
        print(f"  (i)  Generative story: {story}")
        print(f"  (ii) Recommended R_H: {params}")

    # --- (iii) Cross-dataset ---
    print(f"\n{'─'*60}")
    la_model  = preferred["metrla"]
    bay_model = preferred["pemsbay"]
    if la_model == bay_model:
        print(
            f"  (iii) Both datasets prefer the same structure ({la_model}).\n"
            f"        The Kronecker prior architecture is consistent across datasets;\n"
            f"        fitted parameters differ (METR-LA has higher variance and\n"
            f"        likely a slower-decaying slow component due to heavier congestion)."
        )
    else:
        print(
            f"  (iii) Datasets prefer different structures:\n"
            f"        METR-LA  → {la_model}\n"
            f"        PEMS-BAY → {bay_model}\n"
            f"        Consider fitting the prior family independently per (backbone, dataset).\n"
            f"        Do NOT share temporal hyperparameters across benchmarks."
        )
    print(f"{line}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading raw data and computing de-meaned ACF / PACF …")
    acfs: dict[str, np.ndarray] = {}
    pacf_info: dict[str, tuple]  = {}

    for ds in ("metrla", "pemsbay"):
        print(f"  {LABELS[ds]} …")
        acf_full, pacf_vals, n_tr = load_acf_and_pacf(ds)
        acfs[ds]      = acf_full
        pacf_info[ds] = (pacf_vals, n_tr)

    print("\nFitting models and generating plots …")
    rho_1ar    = plot1_loglinear(acfs)
    loglog     = plot2_loglog(acfs)
    two_ar     = plot3_two_ar_fit(acfs)
    rmse_table = plot4_residuals(acfs)
    plot5_pacf(pacf_info)

    print_summary(acfs, rmse_table, rho_1ar, loglog, two_ar, pacf_info)


if __name__ == "__main__":
    main()
