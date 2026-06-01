"""Render the evaluation figures from the cached metrics + sample bank.

Consumes (under ``results/eval/{dataset}/``):
  * ``metrics_all_{split}.json`` — all methods (reliability, CRPS/horizon, sharpness)
  * ``metrics_{split}.json``     — diffusion PIT histogram
  * ``samples_{split}.npz``      — sample bank + reverse-sampling trace
  * ``diagnostics.json``         — generated-sample quality (optional)

Writes PNGs to ``results/eval/{dataset}/figures/``. Each figure is guarded: if
its input is missing it is skipped with a note, so the script is usable at any
stage of the pipeline.

Usage:
    python scripts/eval/make_figures.py --dataset metrla --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

METHOD_LABELS = {
    "gwn_point": "GWN (point)",
    "per_horizon_gaussian": "Per-horizon Gaussian",
    "prior_only": "Prior-only",
    "diffusion": "Diffusion",
}
METHOD_COLORS = {
    "gwn_point": "#777777",
    "per_horizon_gaussian": "#1f77b4",
    "prior_only": "#2ca02c",
    "diffusion": "#d62728",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path.name}", flush=True)


# ─── Figures from metrics_all ─────────────────────────────────────────────────


def fig_reliability(methods: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    for m, d in methods.items():
        if "reliability" not in d:
            continue
        nom = np.array(d["reliability"]["nominal"])
        emp = np.array(d["reliability"]["empirical"])
        ax.plot(nom, emp, "o-", color=METHOD_COLORS.get(m), label=METHOD_LABELS.get(m, m))
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability — central prediction intervals")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    _save(fig, out / "reliability.png")


def fig_crps_vs_horizon(methods: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, d in methods.items():
        crps = d.get("per_horizon", {}).get("crps")
        if crps is None:
            continue
        h = np.arange(1, len(crps) + 1)
        ax.plot(h, crps, "o-", color=METHOD_COLORS.get(m), label=METHOD_LABELS.get(m, m))
    ax.set_xlabel("horizon step")
    ax.set_ylabel("CRPS (lower is better)")
    ax.set_title("CRPS vs forecast horizon")
    ax.legend(fontsize=8)
    _save(fig, out / "crps_vs_horizon.png")


def fig_sharpness(methods: dict, out: Path) -> None:
    """Mean interval width vs nominal coverage — sharpness profile per method."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, d in methods.items():
        rel = d.get("reliability")
        if rel is None or "mean_width" not in rel:
            continue
        ax.plot(rel["nominal"], rel["mean_width"], "o-",
                color=METHOD_COLORS.get(m), label=METHOD_LABELS.get(m, m))
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("mean interval width")
    ax.set_title("Sharpness — narrower at a given coverage is better")
    ax.legend(fontsize=8)
    _save(fig, out / "sharpness.png")


def fig_pit(diff_metrics: dict, out: Path) -> None:
    pit = diff_metrics.get("pit")
    if pit is None:
        return
    hist = np.array(pit["hist"])
    nb = pit["bins"]
    fig, ax = plt.subplots(figsize=(6, 4))
    edges = np.linspace(0, 1, nb + 1)
    ax.bar(edges[:-1], hist, width=1.0 / nb, align="edge",
           color=METHOD_COLORS["diffusion"], alpha=0.8, edgecolor="white")
    ax.axhline(1.0 / nb, color="k", ls="--", lw=1, label="uniform (calibrated)")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("frequency")
    ax.set_title("PIT histogram (diffusion) — uniform = calibrated")
    ax.legend(fontsize=8)
    _save(fig, out / "pit_histogram.png")


# ─── Figures from the sample bank ─────────────────────────────────────────────


def fig_fan_chart(bank: dict, out: Path) -> None:
    """Predictive fan for one (window, sensor): truth, GWN point, diffusion bands."""
    Y = bank["Y_sample"]          # (K, n_bank, N, H)
    y_true = bank["y_true"]       # (n_bank, N, H)
    y_hat = bank["y_hat"]
    mask = bank["mask"]
    # Pick the window+sensor with the most predictive spread among valid cells.
    std = Y.std(0)                # (n_bank, N, H)
    valid = mask.all(-1)          # (n_bank, N)
    score = np.where(valid, std.mean(-1), -np.inf)
    w, s = np.unravel_index(int(np.argmax(score)), score.shape)

    H = Y.shape[-1]
    h = np.arange(1, H + 1)
    samp = Y[:, w, s, :]          # (K, H)
    med = np.median(samp, 0)
    q05, q25, q75, q95 = np.quantile(samp, [0.05, 0.25, 0.75, 0.95], axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(h, q05, q95, color=METHOD_COLORS["diffusion"], alpha=0.18, label="90% band")
    ax.fill_between(h, q25, q75, color=METHOD_COLORS["diffusion"], alpha=0.35, label="50% band")
    ax.plot(h, med, color=METHOD_COLORS["diffusion"], lw=2, label="diffusion median")
    ax.plot(h, y_hat[w, s], "s--", color=METHOD_COLORS["gwn_point"], label="GWN point")
    ax.plot(h, y_true[w, s], "o-", color="k", label="truth")
    ax.set_xlabel("horizon step")
    ax.set_ylabel("value")
    ax.set_title(f"Predictive fan — window {int(bank['bank_idx'][w])}, sensor {s}")
    ax.legend(fontsize=8)
    _save(fig, out / "fan_chart.png")


def fig_spatial_std(bank: dict, out: Path) -> None:
    """Predictive std per sensor at the last horizon, one window (sorted)."""
    Y = bank["Y_sample"]
    h = Y.shape[-1] - 1
    w = 0
    std = Y[:, w, :, h].std(0)    # (N,)
    order = np.argsort(std)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(std)), std[order], color=METHOD_COLORS["diffusion"])
    ax.set_xlabel("sensor (sorted by predictive std)")
    ax.set_ylabel("predictive std")
    ax.set_title(f"Per-sensor predictive uncertainty — window {int(bank['bank_idx'][w])}, horizon {h + 1}")
    _save(fig, out / "spatial_std.png")


def fig_reverse_trajectory(bank: dict, out: Path) -> None:
    t = bank.get("trace_t")
    if t is None or len(t) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, bank["trace_Rt_norm"], "o-", label="‖R_t‖ (state)")
    ax.plot(t, bank["trace_R0_norm"], "s-", label="‖R̂_0‖ (denoiser output)")
    ax.set_xlabel("diffusion time t  (T → 0)")
    ax.set_ylabel("mean L2 norm")
    ax.set_title("Reverse-sampling trajectory")
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    _save(fig, out / "reverse_trajectory.png")


# ─── Figures from diagnostics ─────────────────────────────────────────────────


def fig_generated_quality(diag: dict, out: Path) -> None:
    gq = diag.get("generated_sample_quality")
    if gq is None:
        return
    var = gq["per_horizon_variance"]
    H = len(var["sample"])
    h = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(h, var["real"], "o-", color="k", label="real residuals")
    axes[0].plot(h, var["sample"], "s-", color=METHOD_COLORS["diffusion"], label="generated")
    axes[0].set_xlabel("horizon step")
    axes[0].set_ylabel("variance")
    axes[0].set_title("Per-horizon residual variance")
    axes[0].legend(fontsize=8)

    for ax, key, title in (
        (axes[1], "RH_real", "Horizon corr R_H (real)"),
        (axes[2], "RH_sample", "Horizon corr R_H (generated)"),
    ):
        im = ax.imshow(np.array(gq[key]), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out / "generated_quality.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("metrla", "pemsbay", "pems08f"), required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = p.parse_args()

    base = ROOT / "results" / "eval" / args.dataset
    out = base / "figures"
    print(f"[figures] {args.dataset} / {args.split} -> {out}", flush=True)

    all_path = base / f"metrics_all_{args.split}.json"
    if all_path.exists():
        methods = json.loads(all_path.read_text())
        fig_reliability(methods, out)
        fig_crps_vs_horizon(methods, out)
        fig_sharpness(methods, out)
    else:
        print(f"  (skip metric figures: {all_path.name} missing — run evaluate_forecasts)")

    diff_path = base / f"metrics_{args.split}.json"
    if diff_path.exists():
        fig_pit(json.loads(diff_path.read_text()), out)

    bank_path = base / f"samples_{args.split}.npz"
    if bank_path.exists():
        bank = dict(np.load(bank_path))
        fig_fan_chart(bank, out)
        fig_spatial_std(bank, out)
        fig_reverse_trajectory(bank, out)
    else:
        print(f"  (skip sample-bank figures: {bank_path.name} missing — run generate_samples)")

    diag_path = base / "diagnostics.json"
    if diag_path.exists():
        fig_generated_quality(json.loads(diag_path.read_text()), out)

    print("[figures] done", flush=True)


if __name__ == "__main__":
    main()
