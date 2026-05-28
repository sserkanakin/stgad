"""Section 3 — Experiments 3.4, 3.5, 3.6.

3.4 — Reconcile γ*_A vs γ*_B. All three datasets are in the 2–10× range
      per the spec → take geometric mean.

3.5 — Sample 200 fields from N(0, (L + γ*I)^{-1}) via the eigenbasis. Compare
      against the empirical residual distribution:
        - Fiedler-ordered sample heatmaps (3 random draws per dataset)
        - Dirichlet-energy histogram overlay (samples vs empirical per-window
          normalised r'Lr/r'r; an empirical 1d distribution from the H2 setup)
        - Correlation-vs-graph-distance curve (samples vs empirical),
          using shortest-path hop distance on the symmetrised binary adjacency.

3.6 — PSD verification: L symmetric, λ_min(L+γ*I) > 0, condition number.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.csgraph

from src.data.adjacency import (
    compute_normalized_laplacian,
    laplacian_eigendecomposition,
    load_adjacency,
)

DATASETS = ("metrla", "pemsbay", "pems08f")
OUT_ROOT = Path("results/section3")
RES_TMPL = "results/residuals/gwn_{ds}_train_seed0.npz"
SEED = 0
N_SAMPLES = 200
N_PANEL_SAMPLES = 3


def sample_prior(eigvals: np.ndarray, V: np.ndarray, gamma: float,
                 n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n_samples from N(0, (L + γI)^{-1}) via eigenbasis.
    samples shape: (n_samples, N)."""
    N = eigvals.shape[0]
    z = rng.standard_normal((n_samples, N))
    scale = 1.0 / np.sqrt(eigvals + gamma)               # (N,)
    return z * scale @ V.T                                # (n_samples, N)


def empirical_dirichlet_energies(R: np.ndarray, mask: np.ndarray,
                                 L: np.ndarray) -> np.ndarray:
    """Per-window-per-horizon r'Lr/r'r for fully-observed (window, horizon) pairs.
    Returns 1-D array of energies. Matches H2 convention."""
    B, N, H = R.shape
    R_f = R.astype(np.float64)
    M_f = mask.astype(bool)
    out: list[float] = []
    # full-obs windows per horizon: mask all-True along sensors for that horizon
    for h in range(H):
        ok = M_f[..., h].all(axis=1)                      # (B,)
        if not ok.any():
            continue
        Rh = R_f[ok, :, h]                                # (B', N)
        num = (Rh @ L * Rh).sum(axis=1)
        den = (Rh ** 2).sum(axis=1)
        good = den > 0
        out.extend((num[good] / den[good]).tolist())
    return np.asarray(out)


def hop_distances(A: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path (BFS hop count) on the symmetric binary adjacency.
    A is symmetrised + self-loop-stripped first. Returns (N, N) int.
    Disconnected pairs get inf — set to N for binning purposes."""
    Asym = A.copy()
    np.fill_diagonal(Asym, 0)
    Asym = (Asym + Asym.T) / 2
    binary = (Asym > 0).astype(np.int8)
    D = scipy.sparse.csgraph.shortest_path(binary, method="D", unweighted=True)
    D[~np.isfinite(D)] = binary.shape[0]
    return D.astype(int)


def corr_vs_hop(samples: np.ndarray, D: np.ndarray, max_hop: int = 8) -> np.ndarray:
    """Mean pairwise correlation as a function of hop distance.
    samples: (n_samples, N); returns array of length max_hop+1 (hop 0 dropped)."""
    Ns = samples.shape[1]
    # Pearson correlation over samples (axis=0) per sensor pair: equivalent to
    # cov(samples) normalised by std outer-product.
    centred = samples - samples.mean(axis=0, keepdims=True)
    cov = (centred.T @ centred) / max(samples.shape[0] - 1, 1)
    std = np.sqrt(np.diag(cov))
    corr = cov / (std[:, None] * std[None, :] + 1e-12)
    out = np.full(max_hop + 1, np.nan)
    for h in range(1, max_hop + 1):
        mask = D == h
        if mask.any():
            out[h] = float(corr[mask].mean())
    return out


def empirical_corr_vs_hop(R: np.ndarray, mask: np.ndarray, D: np.ndarray,
                          max_hop: int = 8, sample_n_windows: int = 2000,
                          rng: np.random.Generator = None) -> np.ndarray:
    """Same metric on empirical residuals: sensor-pair correlation over windows.
    Uses full-obs windows for horizon h=0; subsamples to keep memory in check."""
    if rng is None:
        rng = np.random.default_rng(0)
    B, N, H = R.shape
    ok = mask.astype(bool).all(axis=(1, 2))               # all sensors + horizons
    if not ok.any():
        ok = mask.astype(bool)[..., 0].all(axis=1)        # fall back to h=0
        Rh = R[ok, :, 0]
    else:
        Rh = R[ok, :, 0]                                  # use h=0 to match prior shape
    if Rh.shape[0] > sample_n_windows:
        idx = rng.choice(Rh.shape[0], size=sample_n_windows, replace=False)
        Rh = Rh[idx]
    Rh = Rh.astype(np.float64)
    centred = Rh - Rh.mean(axis=0, keepdims=True)
    cov = (centred.T @ centred) / max(Rh.shape[0] - 1, 1)
    std = np.sqrt(np.diag(cov))
    corr = cov / (std[:, None] * std[None, :] + 1e-12)
    out = np.full(max_hop + 1, np.nan)
    for h in range(1, max_hop + 1):
        m = D == h
        if m.any():
            out[h] = float(corr[m].mean())
    return out


def run_per_dataset(dataset: str, gamma_lookup: dict) -> dict:
    A = load_adjacency(dataset)
    L = compute_normalized_laplacian(A)
    eigvals, V = laplacian_eigendecomposition(L, dataset, cache=True)
    N = eigvals.shape[0]

    gamma_A = gamma_lookup[dataset]["A"]
    gamma_B = gamma_lookup[dataset]["B"]
    gamma_star = float(np.sqrt(gamma_A * gamma_B))        # geometric mean
    rng = np.random.default_rng(SEED)

    # ---- 3.6 PSD checks ----
    sym_err = float(np.abs(L - L.T).max())
    L_gI_eigs = eigvals + gamma_star
    psd_ok = bool((L_gI_eigs > 0).all())
    cond_number = float(L_gI_eigs[-1] / L_gI_eigs[0])

    # ---- 3.5 sampling ----
    print(f"  [{dataset}] sampling {N_SAMPLES} prior fields at γ* = {gamma_star:.4f} ...")
    samples = sample_prior(eigvals, V, gamma_star, N_SAMPLES, rng)
    sample_energies = ((samples @ L) * samples).sum(axis=1) / (samples ** 2).sum(axis=1)

    # ---- Empirical baseline ----
    print(f"  [{dataset}] loading residuals + computing empirical comparators ...")
    data = np.load(RES_TMPL.format(ds=dataset))
    R = data["residuals"].astype(np.float32)
    mask = data["mask"].astype(bool)
    emp_energies = empirical_dirichlet_energies(R, mask, L)

    print(f"  [{dataset}] hop distances + correlation-vs-distance ...")
    D = hop_distances(A)
    corr_samples = corr_vs_hop(samples, D)
    corr_emp = empirical_corr_vs_hop(R, mask, D, rng=rng)

    # ---- Plot 6-panel figure ----
    out_dir = OUT_ROOT / dataset / "exp35_prior_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fiedler order for visualisation.
    fiedler = V[:, 1]                                     # eigenvector of λ_2
    order = np.argsort(fiedler)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    sample_ix = rng.choice(N_SAMPLES, size=N_PANEL_SAMPLES, replace=False)
    for j, ix in enumerate(sample_ix):
        ax = axes[0, j]
        s = samples[ix][order]
        vmax = np.abs(samples[sample_ix]).max()
        ax.imshow(s[None, :], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks([])
        ax.set_xlabel("sensor (Fiedler-sorted)")
        ax.set_title(f"prior sample #{ix}")
        ax.set_xlim(0, N)

    # Energy histogram overlap.
    ax = axes[1, 0]
    bins = np.linspace(0, 2, 60)
    ax.hist(emp_energies, bins=bins, alpha=0.5, density=True, label=f"empirical (n={len(emp_energies)})")
    ax.hist(sample_energies, bins=bins, alpha=0.5, density=True, label=f"prior samples (n={N_SAMPLES})")
    ax.axvline(emp_energies.mean(), color="C0", lw=0.8, linestyle="--")
    ax.axvline(sample_energies.mean(), color="C1", lw=0.8, linestyle="--")
    ax.set_xlabel(r"normalised Dirichlet energy $r^\top L r / r^\top r$")
    ax.set_title("energy distribution overlap")
    ax.legend()
    ax.grid(alpha=0.3)

    # Corr vs hop distance.
    ax = axes[1, 1]
    hops = np.arange(len(corr_samples))
    ax.plot(hops, corr_samples, marker="o", label="prior samples")
    ax.plot(hops, corr_emp, marker="s", label="empirical residuals (h=0)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("graph hop distance")
    ax.set_ylabel("mean pairwise correlation")
    ax.set_title("corr-vs-hop")
    ax.legend()
    ax.grid(alpha=0.3)

    # Verdict.
    ax = axes[1, 2]
    ax.axis("off")
    verdict_lines = [
        f"Dataset: {dataset}",
        f"γ⋆ (geomean) = {gamma_star:.4f}",
        f"  γ⋆_A = {gamma_A:.4f}",
        f"  γ⋆_B = {gamma_B:.4f}",
        "",
        f"sample E mean = {sample_energies.mean():.4f}",
        f"empirical E mean = {emp_energies.mean():.4f}",
        f"diff = {sample_energies.mean() - emp_energies.mean():+.4f}",
        "",
        f"corr(h=1) sample = {corr_samples[1]:+.3f}",
        f"corr(h=1) empirical = {corr_emp[1]:+.3f}",
        "",
        f"L+γI cond. number = {cond_number:.1f}",
        f"L+γI PSD: {'YES' if psd_ok else 'NO'}",
    ]
    ax.text(0.0, 1.0, "\n".join(verdict_lines), va="top", ha="left",
            family="monospace", fontsize=10)

    fig.suptitle(f"Exp 3.5 — Prior-sample diagnostics ({dataset})", y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "prior_samples.png", dpi=140)
    plt.close(fig)

    return {
        "dataset": dataset,
        "N": int(N),
        "gamma_A": gamma_A,
        "gamma_B": gamma_B,
        "gamma_star_geomean": gamma_star,
        "reconcile_rule": "geomean (γ*_A and γ*_B differed by 2–10×)",
        "sample_energy_mean": float(sample_energies.mean()),
        "sample_energy_std": float(sample_energies.std()),
        "empirical_energy_mean": float(emp_energies.mean()),
        "empirical_energy_std": float(emp_energies.std()),
        "energy_mean_diff": float(sample_energies.mean() - emp_energies.mean()),
        "corr_samples_vs_hop": corr_samples.tolist(),
        "corr_empirical_vs_hop": corr_emp.tolist(),
        "L_symmetry_max_abs_err": sym_err,
        "lambda_min_L_plus_gI": float(L_gI_eigs[0]),
        "lambda_max_L_plus_gI": float(L_gI_eigs[-1]),
        "cond_number": cond_number,
        "psd_ok": psd_ok,
    }


def main() -> None:
    gamma_A_lookup = {
        r["dataset"]: r["gamma_star_A"]
        for r in json.loads((OUT_ROOT / "exp32_energy_match_summary.json").read_text())
    }
    gamma_B_lookup = {
        r["dataset"]: r["gamma_star_B"]
        for r in json.loads((OUT_ROOT / "exp33_spectral_match_summary.json").read_text())
    }
    gamma_lookup = {
        ds: {"A": gamma_A_lookup[ds], "B": gamma_B_lookup[ds]} for ds in DATASETS
    }

    rows = []
    for dataset in DATASETS:
        r = run_per_dataset(dataset, gamma_lookup)
        out_dir = OUT_ROOT / dataset / "exp35_prior_samples"
        (out_dir / "summary.json").write_text(json.dumps(r, indent=2))
        rows.append(r)
        print(
            f"[{dataset}]  γ⋆={r['gamma_star_geomean']:.4f}  "
            f"E_sample={r['sample_energy_mean']:.4f} vs E_emp={r['empirical_energy_mean']:.4f}  "
            f"(Δ={r['energy_mean_diff']:+.4f})  "
            f"cond={r['cond_number']:.1f}  PSD={r['psd_ok']}"
        )

    summary_path = OUT_ROOT / "exp34_36_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote cross-dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
