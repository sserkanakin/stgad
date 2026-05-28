"""Temporal stationarity diagnostic — empirical R̂_H per dataset.

Tests whether ρ(h, h') depends only on |h − h'| (Toeplitz assumption) by
computing the full 12×12 mask-aware empirical correlation matrix without any
diagonal pooling, then inspecting the spread of entries along each diagonal.

Mask-aware pairwise correlation (consistent with H4):
    Σ[h, h'] = Σ_(b,n) m_h m_h' rc_h rc_h' / Σ_(b,n) m_h m_h'
where rc = (r - mean_cell) and mean_cell is the per-(n, h) cell mean.
Then ρ[h, h'] = Σ[h, h'] / sqrt(Σ[h, h] · Σ[h', h']).

Output per dataset (results/section3_extras/<ds>/temporal_stationarity/):
    summary.json — per-lag mean / std / min / max / spread
    heatmap.png — 12×12 R̂_H heatmap
    per_lag.png — for each lag, scatter of entries vs starting horizon h
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS = ("metrla", "pemsbay", "pems08f")
OUT_ROOT = Path("results/section3_extras")
RES_TMPL = "results/residuals/gwn_{ds}_train_seed0.npz"

# Spread thresholds for the verdict.
SPREAD_STATIONARY_OK = 0.05      # range over a diagonal ≤ this → OK
SPREAD_MILD_VIOLATION = 0.15     # ≤ this → mild violation, still usable
# > 0.15 → meaningful violation, consider full empirical R̂_H


def empirical_R_H(R: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask-aware empirical correlation matrix R̂_H (H×H), plus the per-pair
    sample-count matrix (informative when masks are heavy) and the per-horizon
    standard deviations.

    Returns:
        R_H        — (H, H) correlation, ones on the diagonal
        sigma_h    — (H,) per-horizon std (= √diag(Σ))
        pair_count — (H, H) number of (b, n) cells contributing to each entry
    """
    B, N, H = R.shape
    R_f = R.astype(np.float64)
    M_f = mask.astype(np.float64)

    # Per-(n, h) cell mean.
    cnt_nh = M_f.sum(axis=0)                                    # (N, H)
    mu_nh = np.where(
        cnt_nh > 0,
        (R_f * M_f).sum(axis=0) / np.where(cnt_nh > 0, cnt_nh, 1.0),
        0.0,
    )                                                            # (N, H)
    Rc = (R_f - mu_nh) * M_f                                     # (B, N, H)

    # Flatten (B, N) into a single sample axis of length B*N for matmul.
    Rc_flat = Rc.reshape(B * N, H)                               # (BN, H)
    M_flat = M_f.reshape(B * N, H)                               # (BN, H)

    Cov_num = Rc_flat.T @ Rc_flat                                # (H, H)
    Cov_den = M_flat.T @ M_flat                                  # (H, H)
    with np.errstate(divide="ignore", invalid="ignore"):
        Sigma = np.where(Cov_den > 0, Cov_num / Cov_den, 0.0)

    sigma_h = np.sqrt(np.diag(Sigma))                             # (H,)
    denom = np.outer(sigma_h, sigma_h)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_H = np.where(denom > 0, Sigma / denom, 0.0)
    np.fill_diagonal(R_H, 1.0)
    return R_H, sigma_h, Cov_den


def per_lag_stats(R_H: np.ndarray) -> dict[int, dict]:
    H = R_H.shape[0]
    out: dict[int, dict] = {}
    for k in range(1, H):
        diag = np.array([R_H[i, i + k] for i in range(H - k)])
        out[k] = {
            "n_entries": int(diag.size),
            "mean": float(diag.mean()),
            "std": float(diag.std(ddof=0)),
            "min": float(diag.min()),
            "max": float(diag.max()),
            "spread": float(diag.max() - diag.min()),
            "entries": diag.tolist(),
        }
    return out


def verdict(per_lag: dict[int, dict]) -> str:
    max_spread = max(v["spread"] for v in per_lag.values())
    if max_spread <= SPREAD_STATIONARY_OK:
        return f"STATIONARY (max diag spread {max_spread:.3f} ≤ {SPREAD_STATIONARY_OK})"
    if max_spread <= SPREAD_MILD_VIOLATION:
        return f"MILD VIOLATION (max diag spread {max_spread:.3f} ≤ {SPREAD_MILD_VIOLATION}) — Toeplitz still defensible"
    return f"MEANINGFUL VIOLATION (max diag spread {max_spread:.3f}) — consider full empirical R̂_H"


def plot_outputs(dataset: str, R_H: np.ndarray, sigma_h: np.ndarray,
                 per_lag: dict[int, dict], out_dir: Path) -> None:
    H = R_H.shape[0]

    # --- Heatmap of R̂_H ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(R_H, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(H))
    ax.set_yticks(range(H))
    ax.set_xticklabels([f"h{h+1}" for h in range(H)], fontsize=7)
    ax.set_yticklabels([f"h{h+1}" for h in range(H)], fontsize=7)
    ax.set_title(f"{dataset} — empirical $\\widehat{{R}}_H$ (mask-aware, no pooling)")
    for i in range(H):
        for j in range(H):
            ax.text(j, i, f"{R_H[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(R_H[i, j]) < 0.6 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=140)
    plt.close(fig)

    # --- Per-lag scatter: entries on each diagonal vs starting position ---
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharey=True)
    axes = axes.flatten()
    for k in range(1, H):
        ax = axes[k - 1]
        vals = per_lag[k]["entries"]
        xs = np.arange(len(vals))
        ax.plot(xs, vals, marker="o", linestyle="-", lw=0.8)
        ax.axhline(per_lag[k]["mean"], color="red", lw=0.8, linestyle="--",
                   label=f"mean={per_lag[k]['mean']:.3f}")
        ax.set_title(f"lag {k}: range={per_lag[k]['spread']:.3f}")
        ax.set_xlabel("starting horizon i (entry R̂[i, i+k])")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("correlation")
    axes[4].set_ylabel("correlation")
    axes[8].set_ylabel("correlation")
    fig.suptitle(f"{dataset} — entries on each diagonal of $\\widehat{{R}}_H$", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "per_lag.png", dpi=140)
    plt.close(fig)


def analyse(dataset: str) -> dict:
    print(f"  [{dataset}] loading residuals ...")
    data = np.load(RES_TMPL.format(ds=dataset))
    R = data["residuals"].astype(np.float32)
    mask = data["mask"].astype(bool)

    print(f"    computing empirical R̂_H ...")
    R_H, sigma_h, pair_count = empirical_R_H(R, mask)

    pl = per_lag_stats(R_H)

    out_dir = OUT_ROOT / dataset / "temporal_stationarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_outputs(dataset, R_H, sigma_h, pl, out_dir)

    # Pooled (Toeplitz) lag profile for comparison.
    pooled = {k: pl[k]["mean"] for k in pl}

    v = verdict(pl)
    print(f"    {v}")

    summary = {
        "dataset": dataset,
        "n_samples_BN": int(R.shape[0] * R.shape[1]),
        "H": int(R.shape[2]),
        "sigma_h": sigma_h.tolist(),
        "R_H": R_H.tolist(),
        "pair_count_min": int(pair_count[pair_count > 0].min()) if (pair_count > 0).any() else 0,
        "pair_count_median": float(np.median(pair_count[pair_count > 0])) if (pair_count > 0).any() else 0,
        "per_lag": pl,
        "pooled_toeplitz_profile": pooled,
        "max_diagonal_spread": max(v["spread"] for v in pl.values()),
        "verdict": v,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"\n{'dataset':<10} {'verdict':<70} {'max_spread':>10}\n"
        + "-" * 95
    )
    rows = []
    for dataset in DATASETS:
        r = analyse(dataset)
        rows.append(r)
        print(f"{dataset:<10} {r['verdict']:<70} {r['max_diagonal_spread']:>10.4f}")

    print()
    print("Lag-1 entries per dataset (the headline diagonal):")
    print(f"  {'dataset':<10}  {'lag-1 entries (i=0..10)':<80}  {'mean':<6}  {'spread':<6}")
    for r in rows:
        entries = r["per_lag"]["1"] if "1" in r["per_lag"] else r["per_lag"][1]
        formatted = "  ".join(f"{x:.3f}" for x in entries["entries"])
        print(f"  {r['dataset']:<10}  {formatted:<80}  {entries['mean']:<6.3f}  {entries['spread']:<6.3f}")

    (OUT_ROOT / "temporal_stationarity_summary.json").write_text(
        json.dumps(rows, indent=2)
    )
    print(f"\nWrote summary: {OUT_ROOT / 'temporal_stationarity_summary.json'}")


if __name__ == "__main__":
    main()
