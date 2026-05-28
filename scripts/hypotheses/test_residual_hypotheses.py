"""H1–H5 hypothesis tests on extracted residuals.

Inputs (per dataset):
    results/residuals/gwn_{dataset}_train_seed{seed}.npz       (B, N, H) raw-unit residuals + mask
    results/residuals/gwn_{dataset}_seed{seed}_diagnostic.json §1.10 diagnostic (if present)

Hypotheses:
    H1  marginal Gaussianity (per sensor × horizon)
    H2  graph smoothness via Dirichlet energy vs isotropic baseline
    H3  cross-horizon ACF: biexponential vs AR(1) fit
    H4  Kronecker separability via NKP decomposition of empirical covariance
    H5  per-horizon residual variance monotone in h

Usage:
    python scripts/test_residual_hypotheses.py --dataset metrla  --seed 0
    python scripts/test_residual_hypotheses.py --dataset pemsbay --seed 0
    python scripts/test_residual_hypotheses.py --dataset pems08f --seed 0

Outputs — one self-contained folder per (dataset, seed):
    results/hypotheses/{dataset}_seed{seed}/
        README.md          human-readable summary with pass/fail per H + §1.10
        summary.json       full numeric output (all five hypotheses)
        h1_qq.png          representative Q-Q plots
        h2_energy.png      Dirichlet-energy distribution per horizon
        h3_acf.png         mean ACF + biexp/AR(1) fits
        h4_singular.png    top singular values of the Van Loan rearrangement
        h5_variance.png    per-horizon residual variance δ_h
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

from src.data.adjacency import compute_normalized_laplacian, load_adjacency


# ---------------------------------------------------------------------------
# H1 — marginal Gaussianity per (sensor, horizon)
# ---------------------------------------------------------------------------

def h1_marginal_gaussianity(
    R: np.ndarray, mask: np.ndarray, n_sample_per_cell: int = 4000,
) -> dict:
    """Per (sensor i, horizon h), test Gaussianity via D'Agostino K² and report
    summary statistics on the {p-value, skew, excess-kurtosis} distributions.

    For each cell with at least 50 valid samples (subsampled to
    ``n_sample_per_cell`` for speed), runs ``scipy.stats.normaltest`` which is
    appropriate for large samples (Shapiro-Wilk is capped at n=5000 and
    becomes overpowered well before that).
    """
    rng = np.random.default_rng(0)
    B, N, H = R.shape
    pvalues, skews, excess_kurts = [], [], []

    for i in range(N):
        for h in range(H):
            cell = R[:, i, h]
            m = mask[:, i, h]
            valid = cell[m]
            if valid.size < 50:
                continue
            if valid.size > n_sample_per_cell:
                valid = rng.choice(valid, n_sample_per_cell, replace=False)
            try:
                _stat, p = stats.normaltest(valid)
            except ValueError:
                continue
            pvalues.append(p)
            skews.append(float(stats.skew(valid)))
            excess_kurts.append(float(stats.kurtosis(valid)))  # SciPy returns excess

    pvalues = np.asarray(pvalues)
    skews = np.asarray(skews)
    excess_kurts = np.asarray(excess_kurts)

    return {
        "cells_tested":              int(pvalues.size),
        "normality_pass_rate_p05":   float((pvalues > 0.05).mean()),
        "normality_pass_rate_p01":   float((pvalues > 0.01).mean()),
        "median_p":                  float(np.median(pvalues)),
        "skew_mean":                 float(skews.mean()),
        "skew_p95_abs":              float(np.percentile(np.abs(skews), 95)),
        "kurtosis_excess_mean":      float(excess_kurts.mean()),
        "kurtosis_excess_p95_abs":   float(np.percentile(np.abs(excess_kurts), 95)),
    }


# ---------------------------------------------------------------------------
# H2 — Dirichlet energy R^T L R / R^T R, per (window, horizon)
# ---------------------------------------------------------------------------

def h2_dirichlet_energy(R: np.ndarray, mask: np.ndarray, L: np.ndarray) -> dict:
    """Normalised graph Dirichlet energy per window-horizon, vs isotropic
    baseline mean(eig(L)) = trace(L) / N.

    Only fully-observed windows are used (avoids ambiguity in how missing
    entries should affect the quadratic form).
    """
    B, N, H = R.shape
    full = mask.all(axis=(1, 2))      # (B,)
    Rf = R[full].astype(np.float64)    # (B', N, H)

    baseline = float(np.trace(L)) / N

    if Rf.shape[0] == 0:
        return {"error": "no fully-observed windows", "isotropic_baseline": baseline}

    energies = np.empty((Rf.shape[0], H), dtype=np.float64)
    for h in range(H):
        Rh = Rf[:, :, h]                     # (B', N)
        denom = (Rh * Rh).sum(axis=1)        # (B',)
        LRh = Rh @ L.T                       # (B', N)  ;  L symmetric so L.T = L
        numer = (Rh * LRh).sum(axis=1)        # (B',)
        with np.errstate(invalid="ignore", divide="ignore"):
            energies[:, h] = np.where(denom > 1e-12, numer / denom, np.nan)

    return {
        "windows_used":         int(Rf.shape[0]),
        "isotropic_baseline":   baseline,
        "energy_mean":          float(np.nanmean(energies)),
        "energy_median":        float(np.nanmedian(energies)),
        "energy_p05":           float(np.nanpercentile(energies, 5)),
        "energy_p95":           float(np.nanpercentile(energies, 95)),
        "ratio_to_baseline":    float(np.nanmean(energies) / baseline),
        "per_horizon_mean":     [float(np.nanmean(energies[:, h])) for h in range(H)],
    }


# ---------------------------------------------------------------------------
# H3 — cross-horizon ACF + biexponential vs AR(1)
# ---------------------------------------------------------------------------

def _biexp(k: np.ndarray, alpha: float, rho1: float, rho2: float) -> np.ndarray:
    return alpha * rho1 ** np.abs(k) + (1 - alpha) * rho2 ** np.abs(k)


def _ar1(k: np.ndarray, rho: float) -> np.ndarray:
    return rho ** np.abs(k)


def h3_temporal_acf(R: np.ndarray, mask: np.ndarray) -> tuple[dict, np.ndarray]:
    """Average horizon correlation Corr(R_{i,h}, R_{i,h+k}) across windows then
    sensors. Fit biexponential mixture and AR(1) to that mean ACF.

    Returns:
        summary dict, mean_acf array (for plotting).
    """
    B, N, H = R.shape

    acf_by_lag: dict[int, list[float]] = {k: [] for k in range(H)}

    for i in range(N):
        full = mask[:, i, :].all(axis=1)
        if full.sum() < 50:
            continue
        Ri = R[full, i, :].astype(np.float64)           # (B', H)
        mu = Ri.mean(axis=0, keepdims=True)
        sd = Ri.std(axis=0, keepdims=True) + 1e-12
        Z = (Ri - mu) / sd
        C = (Z.T @ Z) / Z.shape[0]                       # (H, H)
        for h in range(H):
            for hp in range(H):
                acf_by_lag[abs(h - hp)].append(C[h, hp])

    mean_acf = np.array([
        np.mean(acf_by_lag[k]) if acf_by_lag[k] else np.nan
        for k in range(H)
    ])
    ks = np.arange(H)

    # Biexponential fit
    biexp_params = biexp_rmse = None
    try:
        popt, _ = curve_fit(
            _biexp, ks, mean_acf,
            p0=[0.5, 0.95, 0.5],
            bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            maxfev=20000,
        )
        biexp_params = [float(x) for x in popt]
        biexp_rmse = float(np.sqrt(np.mean((_biexp(ks, *popt) - mean_acf) ** 2)))
    except (RuntimeError, ValueError):
        pass

    # AR(1) fit
    ar1_param = ar1_rmse = None
    try:
        popt, _ = curve_fit(
            _ar1, ks, mean_acf,
            p0=[0.9],
            bounds=([0.0], [1.0]),
            maxfev=20000,
        )
        ar1_param = float(popt[0])
        ar1_rmse = float(np.sqrt(np.mean((_ar1(ks, *popt) - mean_acf) ** 2)))
    except (RuntimeError, ValueError):
        pass

    improvement = None
    if biexp_rmse and ar1_rmse and biexp_rmse > 1e-12:
        improvement = float(ar1_rmse / biexp_rmse)

    return {
        "mean_acf":                  mean_acf.tolist(),
        "biexp_alpha":               biexp_params[0] if biexp_params else None,
        "biexp_rho1":                biexp_params[1] if biexp_params else None,
        "biexp_rho2":                biexp_params[2] if biexp_params else None,
        "biexp_rmse":                biexp_rmse,
        "ar1_rho":                   ar1_param,
        "ar1_rmse":                  ar1_rmse,
        "biexp_improvement_ratio":   improvement,
    }, mean_acf


# ---------------------------------------------------------------------------
# H4 — Kronecker separability via Nearest-Kronecker-Product (Van Loan)
# ---------------------------------------------------------------------------

def h4_kronecker_separability(R: np.ndarray, mask: np.ndarray) -> dict:
    """Mask-aware empirical covariance Σ = Cov(vec(R)) with vec column-major
    over N (matches Section 1.9), then NKP: rank-1 (Σ_T ⊗ Σ_S) best in
    Frobenius. Returns the relative residual ||Σ - Σ_T ⊗ Σ_S||_F / ||Σ||_F.

    The pairwise covariance Σ[a, b] is estimated from windows where *both*
    positions a = (h, i) and b = (h', i') are observed:
        Σ[a, b] = Σ_s M[s,a] M[s,b] Rc[s,a] Rc[s,b] / Σ_s M[s,a] M[s,b]
    where Rc is mean-centred at every (i, h) cell. This uses all training
    windows; the previous "fully-observed-windows-only" approach threw away
    >90% of METR-LA's data, leaving fewer samples than dimensions.
    """
    B, N, H = R.shape

    R_f = R.astype(np.float64)
    M_f = mask.astype(np.float64)

    # Per-(i, h) cell mean using only observed values.
    count_cell = M_f.sum(axis=0)                                  # (N, H)
    mean_cell = np.where(
        count_cell > 0,
        (R_f * M_f).sum(axis=0) / np.where(count_cell > 0, count_cell, 1.0),
        0.0,
    )                                                              # (N, H)

    # Centered residuals; we zero out masked positions so they contribute
    # nothing to the cross-product matmul below.
    Rc = (R_f - mean_cell) * M_f                                   # (B, N, H)

    # vec convention: flat index (h * N + i)  ⇒  shape (B, H*N).
    Rflat = Rc.transpose(0, 2, 1).reshape(B, H * N)                # (B, HN)
    Mflat = M_f.transpose(0, 2, 1).reshape(B, H * N)               # (B, HN)

    Cnum = Rflat.T @ Rflat                                          # (HN, HN)
    Cden = Mflat.T @ Mflat                                          # (HN, HN)

    # Σ[a, b] = Σ_s [M*Rc][s,a] [M*Rc][s,b] / Σ_s M[s,a] M[s,b]
    with np.errstate(divide="ignore", invalid="ignore"):
        Sigma = np.where(Cden > 0, Cnum / Cden, 0.0)

    # Diagnostic: how many windows backed each pair, on average?
    pair_count_min = float(Cden[Cden > 0].min()) if (Cden > 0).any() else 0.0
    pair_count_median = float(np.median(Cden[Cden > 0])) if (Cden > 0).any() else 0.0

    # Van Loan rearrangement:  M[h*H+h', i*N+i'] = Σ[h*N+i, h'*N+i']
    M_vl = (
        Sigma.reshape(H, N, H, N)
             .transpose(0, 2, 1, 3)
             .reshape(H * H, N * N)
    )

    s = np.linalg.svd(M_vl, full_matrices=False, compute_uv=False)
    sigma1 = float(s[0])
    total = float(np.sqrt((s ** 2).sum()))
    residual = float(np.sqrt((s[1:] ** 2).sum()))

    return {
        "windows_total":           int(B),
        "pair_count_min":          pair_count_min,
        "pair_count_median":       pair_count_median,
        "sigma1":                  sigma1,
        "sigma_total_frobenius":   total,
        "residual_frobenius":      residual,
        "separability_ratio":      residual / total if total > 0 else None,
        "rank1_explained":         sigma1 / total if total > 0 else None,
        "top_singular_values":     s[:5].tolist(),
    }


# ---------------------------------------------------------------------------
# H5 — per-horizon variance monotone in h
# ---------------------------------------------------------------------------

def h5_horizon_variance_monotone(R: np.ndarray, mask: np.ndarray) -> dict:
    """Per-horizon scale δ_h, two flavours:

    - δ_h^(var) = masked second moment about zero (matches §1.10).
    - δ_h^(MAD) = 1.4826 × median(|r − median(r)|) over masked entries; this
      is a Gaussian-consistent robust scale that ignores the H1 heavy tails.

    Reports the variance-vs-MAD ratio per horizon — large ratios (> ~1.5) say
    the variance estimate is being inflated by tail outliers. Checks
    δ_1 ≤ δ_2 ≤ ... ≤ δ_H monotonicity for both estimators.
    """
    H = R.shape[-1]
    R_f = R.astype(np.float64)

    deltas_var = np.empty(H, dtype=np.float64)
    sigma_mad = np.empty(H, dtype=np.float64)
    n_valid = np.empty(H, dtype=np.int64)

    sq = R_f ** 2
    for h in range(H):
        m = mask[..., h]
        denom = int(m.sum())
        n_valid[h] = denom
        if denom == 0:
            deltas_var[h] = np.nan
            sigma_mad[h] = np.nan
            continue
        deltas_var[h] = float((sq[..., h] * m).sum() / denom)
        valid = R_f[..., h][m]
        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        sigma_mad[h] = float(1.4826 * mad)

    # Robust variance equivalent for fair comparison with deltas_var:
    deltas_mad = sigma_mad ** 2
    ratio = deltas_var / np.where(deltas_mad > 0, deltas_mad, np.nan)

    diffs_var = np.diff(deltas_var)
    diffs_mad = np.diff(deltas_mad)

    return {
        "n_valid_per_horizon":             n_valid.tolist(),
        "delta_h":                         deltas_var.tolist(),
        "delta_h_mad_equivalent":          deltas_mad.tolist(),
        "sigma_mad_per_horizon":           sigma_mad.tolist(),
        "var_over_mad_eq_ratio":           ratio.tolist(),
        "monotone_increasing_var":         bool((diffs_var >= 0).all()),
        "monotone_increasing_mad":         bool((diffs_mad >= 0).all()),
        "n_violations_var":                int((diffs_var < 0).sum()),
        "n_violations_mad":                int((diffs_mad < 0).sum()),
        # Backwards-compat key (also kept in tooling that reads earlier outputs):
        "monotone_increasing":             bool((diffs_var >= 0).all()),
        "n_violations":                    int((diffs_var < 0).sum()),
        "delta_ratio_h12_over_h1":         float(deltas_var[-1] / deltas_var[0]) if deltas_var[0] > 0 else None,
        "delta_mad_ratio_h12_over_h1":     float(deltas_mad[-1] / deltas_mad[0]) if deltas_mad[0] > 0 else None,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_acf(mean_acf: np.ndarray, h3: dict, out_path: Path) -> None:
    ks = np.arange(len(mean_acf))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(ks, mean_acf, "o", label="empirical mean ACF", color="C0")
    if h3.get("biexp_alpha") is not None:
        ax.plot(
            ks,
            _biexp(ks, h3["biexp_alpha"], h3["biexp_rho1"], h3["biexp_rho2"]),
            "-", label=f"biexp (RMSE={h3['biexp_rmse']:.4f})", color="C1",
        )
    if h3.get("ar1_rho") is not None:
        ax.plot(
            ks, _ar1(ks, h3["ar1_rho"]),
            "--", label=f"AR(1) (RMSE={h3['ar1_rmse']:.4f})", color="C2",
        )
    ax.set_xlabel("horizon lag |k|")
    ax.set_ylabel("Corr(R_{i,h}, R_{i,h+k})")
    ax.set_title("H3: horizon ACF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_h2_energy(h2: dict, out_path: Path) -> None:
    if "per_horizon_mean" not in h2:
        return
    ks = np.arange(1, len(h2["per_horizon_mean"]) + 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(ks, h2["per_horizon_mean"], "o-", color="C0", label="mean energy / baseline (per h)")
    ax.axhline(1.0, color="grey", linestyle="--", label="isotropic baseline (=1)")
    ax.axhline(h2["ratio_to_baseline"], color="C3", linestyle=":",
               label=f"overall mean = {h2['ratio_to_baseline']:.3f}")
    ax.set_xlabel("horizon h")
    ax.set_ylabel("E(R) / mean(eig(L))")
    ax.set_title("H2: Dirichlet energy vs isotropic baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_h4_singular(h4: dict, out_path: Path) -> None:
    if "top_singular_values" not in h4:
        return
    s = np.asarray(h4["top_singular_values"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(np.arange(1, len(s) + 1), s, color="C0")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σ_k")
    ax.set_title(
        "H4: top SVs of Van Loan rearrangement   "
        f"(rank-1 explained = {h4.get('rank1_explained', 0):.3f},  "
        f"residual/total = {h4.get('separability_ratio', 0):.3f})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_h5_variance(h5: dict, out_path: Path) -> None:
    deltas_var = np.asarray(h5["delta_h"], dtype=float)
    deltas_mad = np.asarray(h5.get("delta_h_mad_equivalent", [np.nan] * len(deltas_var)), dtype=float)
    ks = np.arange(1, len(deltas_var) + 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(ks, deltas_var, "o-", color="C0", label=r"$\widehat{\delta}_h^{(\mathrm{var})}$ (second moment)")
    ax.plot(ks, deltas_mad, "s--", color="C1",
            label=r"$\widehat{\delta}_h^{(\mathrm{MAD})} = (1.4826\,\mathrm{MAD})^2$ (robust)")
    monotone = "monotone ✓" if h5["monotone_increasing_var"] else f"{h5['n_violations_var']} violations"
    ax.set_xlabel("horizon h")
    ax.set_ylabel(r"per-horizon scale (raw-unit squared)")
    ax.set_title(
        f"H5: per-horizon residual scale   ({monotone},  "
        f"δ12/δ1 (var)={h5['delta_ratio_h12_over_h1']:.2f},  "
        f"(MAD)={h5.get('delta_mad_ratio_h12_over_h1', float('nan')):.2f})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_qq(R: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    rng = np.random.default_rng(0)
    B, N, H = R.shape
    sensors = rng.choice(N, size=3, replace=False)
    horizons = [0, 3, 6, 11] if H >= 12 else list(range(min(4, H)))

    fig, axes = plt.subplots(len(sensors), len(horizons),
                              figsize=(2.4 * len(horizons), 2.4 * len(sensors)),
                              squeeze=False)
    for r, i in enumerate(sensors):
        for c, h in enumerate(horizons):
            ax = axes[r, c]
            valid = R[:, i, h][mask[:, i, h]]
            if valid.size < 30:
                ax.set_title(f"sensor {i}, h={h+1}: n={valid.size}", fontsize=7)
                continue
            if valid.size > 2000:
                valid = rng.choice(valid, 2000, replace=False)
            stats.probplot(valid, dist="norm", plot=ax)
            ax.set_title(f"sensor {i}, h={h+1}", fontsize=8)
            ax.tick_params(labelsize=6)
            for txt in ax.texts:
                txt.set_visible(False)
    fig.suptitle("H1: representative Q-Q plots", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render_readme(out: dict, diag: dict | None, out_path: Path) -> None:
    """Write a human-readable Markdown summary of the run.

    Pulls headline numbers from each hypothesis's dict so readers don't have
    to crack open summary.json. Treats criteria as conventions, not hard tests:
    the user inspects numbers + plots to decide whether to act.
    """
    h1, h2, h3, h4, h5 = out["H1"], out["H2"], out["H3"], out["H4"], out["H5"]

    if diag:
        diag_block = (
            f"- **§1.10 (train vs val per-horizon variance):** "
            f"{'✅ PASS' if diag['pass_overall'] else '❌ FAIL'}\n"
            f"  - Criterion: `|δ_h^(vl)/δ_h^(tr) - 1| < 0.10  ∀ h`\n"
            f"  - Per-horizon ratios: "
            + " ".join(f"{r:+.3f}" for r in diag["ratio_vl_over_tr_minus_one"])
            + "\n"
        )
    else:
        diag_block = "- §1.10 diagnostic not found (extract_residuals.py needs to be rerun).\n"

    h1_verdict = (
        "❌ heavy tails / strong skew" if (h1["kurtosis_excess_p95_abs"] > 5 or h1["skew_p95_abs"] > 1)
        else "⚠ marginal" if (h1["normality_pass_rate_p05"] < 0.5)
        else "✅ approximately Gaussian"
    )
    h2_verdict = "✅ graph-smooth" if h2.get("ratio_to_baseline", 1.0) < 0.95 else "⚠ near-isotropic (graph adds little)"
    h3_verdict = (
        f"✅ biexp clearly better ({h3['biexp_improvement_ratio']:.2f}× lower RMSE)"
        if h3.get("biexp_improvement_ratio") and h3["biexp_improvement_ratio"] > 1.5
        else "⚠ biexp ≈ AR(1) (single-exponential is enough)"
    )
    h4_sep = h4.get("separability_ratio", 1.0)
    h4_verdict = (
        f"✅ strongly separable (residual/total = {h4_sep:.3f})" if h4_sep < 0.2
        else f"⚠ partial ({h4_sep:.3f}) — non-separable interactions present" if h4_sep < 0.4
        else f"❌ poorly separable ({h4_sep:.3f})"
    )
    h5_verdict_var = (
        f"✅ monotone increasing (variance δ12/δ1 = {h5['delta_ratio_h12_over_h1']:.2f})"
        if h5["monotone_increasing_var"]
        else f"❌ {h5['n_violations_var']} violations (variance)"
    )
    h5_verdict_mad = (
        f"✅ monotone increasing (MAD δ12/δ1 = {h5.get('delta_mad_ratio_h12_over_h1', float('nan')):.2f})"
        if h5.get("monotone_increasing_mad", False)
        else f"❌ {h5.get('n_violations_mad', 0)} violations (MAD)"
    )
    h5_verdict = f"{h5_verdict_var}; {h5_verdict_mad}"

    md = f"""# Hypothesis tests — {out['dataset']} (seed {out['seed']})

| | |
|---|---|
| Source | `results/residuals/gwn_{out['dataset']}_train_seed{out['seed']}.npz` |
| Windows | {out['n_windows']:,} |
| Sensors | {out['n_sensors']} |
| Horizon | {out['horizon']} |

## In-sample diagnostic
{diag_block}

## H1 — Marginal Gaussianity (per sensor × horizon)
**{h1_verdict}**

- Cells tested: {h1['cells_tested']:,}
- Median p (D'Agostino K²): {h1['median_p']:.3g}
- Pass rate at p > 0.05: {h1['normality_pass_rate_p05']:.3f}
- Skew (mean / p95 |·|): {h1['skew_mean']:+.3f} / {h1['skew_p95_abs']:.3f}
- Excess kurtosis (mean / p95 |·|): {h1['kurtosis_excess_mean']:+.3f} / {h1['kurtosis_excess_p95_abs']:.3f}

See `h1_qq.png` for representative Q-Q plots.

## H2 — Graph smoothness (Dirichlet energy)
**{h2_verdict}**

- Energy mean / baseline = {h2.get('ratio_to_baseline', float('nan')):.4f}  (< 1 = graph-smoother than isotropic)
- Energy mean / median / p5 / p95: {h2.get('energy_mean', float('nan')):.4f} / {h2.get('energy_median', float('nan')):.4f} / {h2.get('energy_p05', float('nan')):.4f} / {h2.get('energy_p95', float('nan')):.4f}
- Isotropic baseline = trace(L)/N = {h2.get('isotropic_baseline', float('nan')):.4f}
- Windows used (fully observed): {h2.get('windows_used', 0):,}

See `h2_energy.png` for per-horizon means.

## H3 — Cross-horizon ACF: biexponential vs AR(1)
**{h3_verdict}**

- Biexp:  α={h3.get('biexp_alpha')}, ρ1={h3.get('biexp_rho1')}, ρ2={h3.get('biexp_rho2')}, RMSE={h3.get('biexp_rmse')}
- AR(1):  ρ={h3.get('ar1_rho')}, RMSE={h3.get('ar1_rmse')}
- AR(1) / biexp RMSE ratio: {h3.get('biexp_improvement_ratio')}

See `h3_acf.png` for empirical ACF + both fits.

## H4 — Kronecker separability (NKP, mask-aware pairwise covariance)
**{h4_verdict}**

- Rank-1 explained: {h4.get('rank1_explained', float('nan')):.4f}
- Separability ratio (||Σ - Σ_T⊗Σ_S||_F / ||Σ||_F): {h4.get('separability_ratio', float('nan')):.4f}
- σ_1: {h4.get('sigma1', float('nan')):.4g}
- Windows: {h4.get('windows_total', 0):,}  (mask-aware pair counts: min={h4.get('pair_count_min', 0):.0f}, median={h4.get('pair_count_median', 0):.0f})

See `h4_singular.png` for top singular values.

## H5 — Horizon-monotone scale (variance + MAD)
**{h5_verdict}**

- δ_h (variance, raw²): {' '.join(f'{d:.3f}' for d in h5['delta_h'])}
- δ_h (MAD-equiv = (1.4826·MAD)², raw²): {' '.join(f'{d:.3f}' for d in h5.get('delta_h_mad_equivalent', []))}
- var / MAD-equiv per horizon (large ⇒ tail-inflated): {' '.join(f'{r:.2f}' for r in h5.get('var_over_mad_eq_ratio', []))}
- δ12 / δ1 — variance: {h5['delta_ratio_h12_over_h1']:.3f}; MAD-eq: {h5.get('delta_mad_ratio_h12_over_h1', float('nan')):.3f}

See `h5_variance.png` for both curves.

---
*Full numeric output: `summary.json`.*
"""
    out_path.write_text(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1–H5 on extracted residuals.")
    parser.add_argument("--dataset", required=True,
                        choices=["metrla", "pemsbay", "pems08f"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip QQ + ACF plots (numbers only).")
    args = parser.parse_args()

    res_path = Path(f"results/residuals/gwn_{args.dataset}_train_seed{args.seed}.npz")
    if not res_path.exists():
        raise FileNotFoundError(
            f"Residuals not found: {res_path}\n"
            f"Run  python scripts/extract_residuals.py --dataset {args.dataset} "
            f"--seed {args.seed}  first."
        )

    print(f"Loading {res_path} ...")
    data = np.load(res_path)
    R = data["residuals"].astype(np.float32)
    mask = data["mask"].astype(bool)
    print(f"  R.shape = {R.shape}   mask coverage = {mask.mean():.4f}")

    print("Computing symmetric normalised Laplacian ...")
    adj = load_adjacency(args.dataset)
    L = compute_normalized_laplacian(adj)
    print(f"  L.shape = {L.shape}  trace(L)/N = {np.trace(L) / L.shape[0]:.4f}")

    out: dict = {
        "dataset":   args.dataset,
        "seed":      args.seed,
        "n_windows": int(R.shape[0]),
        "n_sensors": int(R.shape[1]),
        "horizon":   int(R.shape[2]),
    }

    print("\n[H1] marginal Gaussianity ...")
    out["H1"] = h1_marginal_gaussianity(R, mask)
    print(f"  cells={out['H1']['cells_tested']}  "
          f"median p={out['H1']['median_p']:.3g}  "
          f"|skew| p95={out['H1']['skew_p95_abs']:.3f}  "
          f"|excess kurt| p95={out['H1']['kurtosis_excess_p95_abs']:.3f}")

    print("[H2] Dirichlet energy ...")
    out["H2"] = h2_dirichlet_energy(R, mask, L)
    if "ratio_to_baseline" in out["H2"]:
        print(f"  energy mean={out['H2']['energy_mean']:.4f}  "
              f"baseline={out['H2']['isotropic_baseline']:.4f}  "
              f"ratio={out['H2']['ratio_to_baseline']:.4f}  (<1 supports H2)")
    else:
        print(f"  {out['H2'].get('error', 'no output')}")

    print("[H3] cross-horizon ACF, biexp vs AR(1) ...")
    h3, mean_acf = h3_temporal_acf(R, mask)
    out["H3"] = h3
    if h3["biexp_rmse"] is not None and h3["ar1_rmse"] is not None:
        print(f"  biexp RMSE={h3['biexp_rmse']:.4f}  "
              f"AR(1) RMSE={h3['ar1_rmse']:.4f}  "
              f"ratio AR1/biexp={h3['biexp_improvement_ratio']}")
    else:
        print("  fit failed (see JSON)")

    print("[H4] Kronecker separability (NKP, mask-aware) ...")
    out["H4"] = h4_kronecker_separability(R, mask)
    if "separability_ratio" in out["H4"]:
        print(f"  windows={out['H4']['windows_total']}  "
              f"pair counts (min/median)={out['H4']['pair_count_min']:.0f}"
              f"/{out['H4']['pair_count_median']:.0f}  "
              f"separability ratio={out['H4']['separability_ratio']:.4f}  "
              f"rank-1={out['H4']['rank1_explained']:.4f}")
    else:
        print(f"  {out['H4'].get('error', 'no output')}")

    print("[H5] horizon-monotone scale (variance + MAD) ...")
    out["H5"] = h5_horizon_variance_monotone(R, mask)
    print(f"  δ_h (var)        = " + " ".join(f"{d:.3f}" for d in out["H5"]["delta_h"]))
    print(f"  δ_h (MAD-equiv)  = " + " ".join(f"{d:.3f}" for d in out["H5"]["delta_h_mad_equivalent"]))
    print(f"  var/MAD ratio    = " + " ".join(f"{r:.2f}" for r in out["H5"]["var_over_mad_eq_ratio"]))
    print(f"  monotone var={out['H5']['monotone_increasing_var']}  "
          f"MAD={out['H5']['monotone_increasing_mad']}  "
          f"δ12/δ1: var={out['H5']['delta_ratio_h12_over_h1']:.2f}, "
          f"MAD={out['H5']['delta_mad_ratio_h12_over_h1']:.2f}")

    # One self-contained folder per (dataset, seed): summary.json + README.md
    # + one plot per hypothesis. Older "flat" outputs are not migrated; just
    # delete the old top-level files manually if they bother you.
    run_dir = Path(f"results/hypotheses/{args.dataset}_seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "summary.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {json_path}")

    diag_path = Path(
        f"results/residuals/gwn_{args.dataset}_seed{args.seed}_diagnostic.json"
    )
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else None

    if not args.no_plots:
        _plot_qq(R, mask, run_dir / "h1_qq.png")
        _plot_h2_energy(out["H2"], run_dir / "h2_energy.png")
        _plot_acf(np.asarray(mean_acf), h3, run_dir / "h3_acf.png")
        _plot_h4_singular(out["H4"], run_dir / "h4_singular.png")
        _plot_h5_variance(out["H5"], run_dir / "h5_variance.png")
        print(f"Saved plots in {run_dir}/")

    _render_readme(out, diag, run_dir / "README.md")
    print(f"Saved {run_dir / 'README.md'}")


if __name__ == "__main__":
    main()
