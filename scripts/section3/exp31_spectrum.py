"""Section 3 — Experiment 3.1: Spectrum of the normalised Laplacian.

For each dataset, loads the cached eigendecomposition of L = I - D^{-1/2} A D^{-1/2}
(adjacency symmetrised and self-loops stripped — see src/data/adjacency.py),
and reports:

  - lambda_1 (should be ~0; numerical floor)
  - lambda_2 (algebraic connectivity / spectral gap)
  - lambda_N (should be <= 2)
  - number of "zero" modes (lambda_k < 1e-6) — connected-component count
  - number of "near-zero" modes (lambda_k < 0.01) — cluster modes
  - bulk mean/std of {lambda_2, ..., lambda_{N-1}}
  - average eigenvalue lambda_bar = trace(L)/N

Writes per-dataset JSON + a spectrum plot to:
  results/section3/<dataset>/exp31_spectrum/{summary.json, spectrum.png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.adjacency import (
    compute_normalized_laplacian,
    laplacian_eigendecomposition,
    load_adjacency,
)

DATASETS = ("metrla", "pemsbay", "pems08f")
OUT_ROOT = Path("results/section3")


def analyse(dataset: str) -> dict:
    A = load_adjacency(dataset)
    L = compute_normalized_laplacian(A)
    eigvals, eigvecs = laplacian_eigendecomposition(L, dataset, cache=True)
    N = eigvals.shape[0]

    sym_err = float(np.abs(L - L.T).max())
    lambda_bar = float(eigvals.mean())
    bulk = eigvals[1:-1]

    n_zero = int((eigvals < 1e-6).sum())
    n_near_zero = int((eigvals < 0.01).sum())

    return {
        "dataset": dataset,
        "N": int(N),
        "L_symmetry_max_abs_err": sym_err,
        "lambda_1": float(eigvals[0]),
        "lambda_2": float(eigvals[1]),
        "lambda_N": float(eigvals[-1]),
        "lambda_bar": lambda_bar,
        "n_modes_below_1e-6": n_zero,
        "n_modes_below_0.01": n_near_zero,
        "bulk_mean": float(bulk.mean()),
        "bulk_std": float(bulk.std()),
        "eigvals_path": str(Path(f"data/processed/{dataset}_laplacian_eigs.npz")),
    }


def plot_spectrum(dataset: str, eigvals: np.ndarray, out_dir: Path) -> None:
    N = eigvals.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(np.arange(1, N + 1), eigvals, marker=".", linestyle="-", lw=0.8)
    ax1.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax1.axhline(2, color="r", lw=0.5, alpha=0.4, linestyle="--")
    ax1.set_xlabel("k")
    ax1.set_ylabel(r"$\lambda_k$")
    ax1.set_title(f"{dataset} — Laplacian spectrum (linear)")
    ax1.grid(alpha=0.3)

    # Log-y of strictly positive eigenvalues (skip the structural zero at k=1)
    positive = np.where(eigvals > 1e-15, eigvals, np.nan)
    ax2.plot(np.arange(1, N + 1), positive, marker=".", linestyle="-", lw=0.8)
    ax2.axhline(0.01, color="orange", lw=0.5, linestyle=":", label=r"$0.01$")
    ax2.set_yscale("log")
    ax2.set_xlabel("k")
    ax2.set_ylabel(r"$\log \lambda_k$")
    ax2.set_title(f"{dataset} — log scale (zoom on small modes)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "spectrum.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        out_dir = OUT_ROOT / dataset / "exp31_spectrum"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = analyse(dataset)
        eigvals = np.load(result["eigvals_path"])["eigenvalues"]
        plot_spectrum(dataset, eigvals, out_dir)
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
        rows.append(result)
        print(
            f"[{dataset}]  N={result['N']}  λ1={result['lambda_1']:.3e}  "
            f"λ2={result['lambda_2']:.4f}  λN={result['lambda_N']:.4f}  "
            f"λ̄={result['lambda_bar']:.4f}  "
            f"#<1e-6={result['n_modes_below_1e-6']}  "
            f"#<0.01={result['n_modes_below_0.01']}  "
            f"bulk={result['bulk_mean']:.3f}±{result['bulk_std']:.3f}"
        )

    # Cross-dataset summary
    summary_path = OUT_ROOT / "exp31_spectrum_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote cross-dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
