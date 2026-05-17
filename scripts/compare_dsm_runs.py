"""Compare evaluation outputs across multiple trained DSM checkpoints.

Loads the eval JSONs produced by ``scripts/eval_diffusion.py`` for a set
of run_ids (typically 6 for the Stage-2 mini-sweep: 3 datasets × 2 priors)
and emits four tables + summary plots for the meeting plot deck.

Tables produced (printed as markdown to stdout, also saved as CSV):

  Table 1  PEMS-BAY vs USTD published baseline (normalised CRPS per horizon).
  Table 2  PEMS08F vs DiffSTG / SpecSTG / HySRD-Net published baselines.
  Table 3  Internal ablation: structured vs isotropic on each dataset.
  Table 4  Fitted prior parameters across datasets.

Plots produced (PNG, saved to ``experiments/score-network/comparison_seed0/``):

  fig_crps_per_horizon.png   3 panels (one per dataset), structured vs isotropic
  fig_pit_grid.png           3×2 PIT histograms

Usage:
    # Auto-discover all eval JSONs in the default out_dir:
    python scripts/compare_dsm_runs.py

    # Specify explicit run_ids:
    python scripts/compare_dsm_runs.py --run-ids dsm_structured_metrla_seed0_<h> \\
                                                 dsm_isotropic_metrla_seed0_<h> ...

    # Different eval root:
    python scripts/compare_dsm_runs.py --eval-dir experiments/score-network
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Published baselines (from named papers; numbers extracted by hand — verify
# against the source before quoting in the thesis).
# ─────────────────────────────────────────────────────────────────────────────

# Normalised CRPS at horizons 3 / 6 / 12 (15min / 30min / 60min ahead).
# Reported papers use slightly different normalisation conventions; here we
# assume "CRPS / mean|y|" (Variant 2) for all. Mark NA for cells where the
# source paper does not report that horizon.
PUBLISHED = {
    "pemsbay": {
        # USTD (Hu et al., 2024) — diffusion-based probabilistic forecasting.
        "USTD": {3: None, 6: None, 12: None, "avg": None},
    },
    "pems08f": {
        # DiffSTG (Wen et al., KDD 2024)
        "DiffSTG":   {3: None, 6: None, 12: None, "avg": None},
        # SpecSTG (Lin et al., 2024)
        "SpecSTG":   {3: None, 6: None, 12: None, "avg": None},
        # HySRD-Net (placeholder name; fill from paper)
        "HySRD-Net": {3: None, 6: None, 12: None, "avg": None},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Discovery + loading
# ─────────────────────────────────────────────────────────────────────────────

_RUN_RE = re.compile(r"^dsm_(?P<prior>structured|isotropic)_(?P<dataset>metrla|pemsbay|pems08f)_seed(?P<seed>\d+)_(?P<hash>\w+)$")


def parse_run_id(run_id: str) -> dict | None:
    """Extract (prior, dataset, seed, hash) from a run_id string."""
    m = _RUN_RE.match(run_id)
    if not m:
        return None
    return {
        "run_id":  run_id,
        "prior":   m.group("prior"),
        "dataset": m.group("dataset"),
        "seed":    int(m.group("seed")),
        "hash":    m.group("hash"),
    }


def discover_evals(eval_dir: Path) -> list[dict]:
    """Find every eval_<run_id>.json under eval_dir and parse the run_id."""
    out: list[dict] = []
    for p in sorted(eval_dir.glob("eval_dsm_*.json")):
        run_id = p.stem[len("eval_"):]
        meta = parse_run_id(run_id)
        if meta is None:
            print(f"  [warn] skipping malformed run_id: {run_id}")
            continue
        meta["json_path"] = p
        out.append(meta)
    return out


def load_eval(meta: dict) -> dict:
    """Load an eval JSON dict; add the parsed metadata + computed shortcuts."""
    with open(meta["json_path"]) as f:
        data = json.load(f)
    return {**meta, **data}


# ─────────────────────────────────────────────────────────────────────────────
# Table builders
# ─────────────────────────────────────────────────────────────────────────────

def _crps_at(record: dict, h: int, normalised: bool = True) -> float | None:
    """Look up CRPS at horizon h (1-indexed) from one eval record."""
    key = "crps_per_horizon_normalized" if normalised else "crps_per_horizon"
    arr = record.get(key)
    if arr is None or h - 1 >= len(arr):
        return None
    return float(arr[h - 1])


def _crps_avg(record: dict, normalised: bool = True) -> float | None:
    key = "crps_average_normalized" if normalised else "crps_average"
    v = record.get(key)
    return float(v) if v is not None else None


def table1_pemsbay_vs_ustd(records: dict) -> pd.DataFrame:
    """Normalised CRPS, PEMS-BAY only: ours (structured) vs USTD baseline.

    Rows: methods. Columns: h=3, h=6, h=12, mean over those.
    """
    rows = []
    rec_struct = records.get(("pemsbay", "structured"))
    if rec_struct is not None:
        ours = {h: _crps_at(rec_struct, h) for h in (3, 6, 12)}
        ours["avg"] = float(np.mean([v for v in ours.values() if v is not None])) if any(v is not None for v in ours.values()) else None
        rows.append({"method": "ours (structured)", **ours})
    rec_iso = records.get(("pemsbay", "isotropic"))
    if rec_iso is not None:
        ours_iso = {h: _crps_at(rec_iso, h) for h in (3, 6, 12)}
        ours_iso["avg"] = float(np.mean([v for v in ours_iso.values() if v is not None])) if any(v is not None for v in ours_iso.values()) else None
        rows.append({"method": "ours (isotropic)", **ours_iso})
    for name, baseline in PUBLISHED.get("pemsbay", {}).items():
        rows.append({"method": name, **baseline})
    return pd.DataFrame(rows).set_index("method")


def table2_pems08f_vs_published(records: dict) -> pd.DataFrame:
    """Normalised CRPS, PEMS08F only: ours vs DiffSTG / SpecSTG / HySRD-Net."""
    rows = []
    for prior in ("structured", "isotropic"):
        rec = records.get(("pems08f", prior))
        if rec is None:
            continue
        ours = {h: _crps_at(rec, h) for h in (3, 6, 12)}
        ours["avg"] = float(np.mean([v for v in ours.values() if v is not None])) if any(v is not None for v in ours.values()) else None
        rows.append({"method": f"ours ({prior})", **ours})
    for name, baseline in PUBLISHED.get("pems08f", {}).items():
        rows.append({"method": name, **baseline})
    return pd.DataFrame(rows).set_index("method")


def table3_structured_vs_isotropic(records: dict) -> pd.DataFrame:
    """All-dataset internal ablation: normalised CRPS at h=3, 6, 12, avg."""
    rows = []
    for ds in ("metrla", "pemsbay", "pems08f"):
        rec_struct = records.get((ds, "structured"))
        rec_iso = records.get((ds, "isotropic"))
        if rec_struct is None and rec_iso is None:
            continue
        row = {"dataset": ds}
        for h in (3, 6, 12):
            row[f"struct h{h}"] = _crps_at(rec_struct, h) if rec_struct else None
            row[f"iso h{h}"] = _crps_at(rec_iso, h) if rec_iso else None
            if rec_struct and rec_iso:
                s = _crps_at(rec_struct, h)
                i = _crps_at(rec_iso, h)
                row[f"Δ h{h}"] = i - s if (s is not None and i is not None) else None
        row["struct avg"] = _crps_avg(rec_struct) if rec_struct else None
        row["iso avg"] = _crps_avg(rec_iso) if rec_iso else None
        if row["struct avg"] is not None and row["iso avg"] is not None:
            row["Δ avg"] = row["iso avg"] - row["struct avg"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("dataset")


def table4_fitted_priors(prior_yaml_dir: Path) -> pd.DataFrame:
    """Fitted prior parameter table from results/prior_params/."""
    rows = []
    for ds in ("metrla", "pemsbay", "pems08f"):
        path = prior_yaml_dir / f"gwn_{ds}.yaml"
        if not path.exists():
            print(f"  [warn] {path} missing")
            continue
        with open(path) as f:
            p = yaml.safe_load(f)
        rows.append({
            "dataset":   ds,
            "α":         p.get("alpha"),
            "ρ₁ slow":   p.get("rho1"),
            "ρ₂ fast":   p.get("rho2"),
            "β spatial": p.get("beta"),
            "ε_s":       p.get("epsilon_s"),
            "σ":         p.get("sigma"),
            "δ_h mean":  float(np.mean(p.get("delta_h", [np.nan]))),
            "tgt_scale[h12]": float(p.get("target_scale", [None]*12)[-1]) if p.get("target_scale") else None,
        })
    return pd.DataFrame(rows).set_index("dataset")


# ─────────────────────────────────────────────────────────────────────────────
# Plot builders
# ─────────────────────────────────────────────────────────────────────────────

def fig_crps_per_horizon(records: dict, out_path: Path) -> None:
    """3 panels (one per dataset). Each: structured + isotropic per-horizon CRPS."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=False)
    for ax, ds in zip(axes, ("metrla", "pemsbay", "pems08f")):
        for prior, marker, ls in (("structured", "o", "-"), ("isotropic", "x", "--")):
            rec = records.get((ds, prior))
            if rec is None:
                continue
            crps_norm = rec.get("crps_per_horizon_normalized")
            if crps_norm is None:
                continue
            h = np.arange(1, len(crps_norm) + 1)
            ax.plot(h, crps_norm, marker + ls, label=prior, alpha=0.85)
        ax.set_xlabel("horizon h (5-min steps)")
        ax.set_title(f"{ds}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("normalised CRPS")
    fig.suptitle("Per-horizon CRPS (Variant 2 normalised) — structured vs isotropic", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_pit_grid(records: dict, out_path: Path) -> None:
    """3×2 grid (datasets × priors) of mean-over-horizons PIT histograms."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(8, 7.5), sharex=True, sharey=True)
    datasets = ("metrla", "pemsbay", "pems08f")
    priors = ("structured", "isotropic")
    for i, ds in enumerate(datasets):
        for j, prior in enumerate(priors):
            ax = axes[i, j]
            rec = records.get((ds, prior))
            if rec is None or rec.get("pit_histogram_per_horizon") is None:
                ax.set_visible(False)
                continue
            pit = np.array(rec["pit_histogram_per_horizon"])  # (H, 20)
            # Average across horizons → 20-bin avg histogram
            avg = pit.mean(axis=0)
            bins = np.linspace(0, 1, 21)
            centres = (bins[:-1] + bins[1:]) / 2
            ax.bar(centres, avg / avg.sum() if avg.sum() > 0 else avg,
                   width=1/20 * 0.9, color="steelblue", alpha=0.8)
            ax.axhline(1/20, color="red", lw=0.8, ls="--", label="uniform (calibrated)")
            ax.set_title(f"{ds} / {prior}", fontsize=9)
            if i == 2:
                ax.set_xlabel("PIT value")
            if j == 0:
                ax.set_ylabel("density")
    fig.suptitle("Average PIT histogram per (dataset, prior). Uniform ↔ calibrated.", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval-dir", default="experiments/score-network",
                   help="Directory containing eval_<run_id>.json files.")
    p.add_argument("--out-dir", default="experiments/score-network/comparison_seed0",
                   help="Where to save tables + plots.")
    p.add_argument("--prior-yaml-dir", default="results/prior_params",
                   help="Directory containing gwn_<dataset>.yaml prior files.")
    p.add_argument("--run-ids", nargs="*", default=None,
                   help="If given, restrict to these run_ids (basename without .json).")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = discover_evals(eval_dir)
    if args.run_ids:
        wanted = set(args.run_ids)
        metas = [m for m in metas if m["run_id"] in wanted]

    if not metas:
        print(f"[compare_dsm_runs] No eval JSONs found under {eval_dir}.")
        print(f"  Expected files like eval_dsm_<prior>_<dataset>_seed<S>_<hash>.json")
        return 1

    print(f"[compare_dsm_runs] found {len(metas)} eval JSONs:")
    for m in metas:
        print(f"  - {m['run_id']}")
    print()

    # Index by (dataset, prior) — the comparison key
    records: dict[tuple[str, str], dict] = {}
    for m in metas:
        rec = load_eval(m)
        key = (m["dataset"], m["prior"])
        if key in records:
            print(f"  [warn] duplicate ({m['dataset']}, {m['prior']}); keeping latest seed={m['seed']}")
        records[key] = rec

    # ── Tables ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Table 1 — PEMS-BAY: ours vs USTD baseline (normalised CRPS)")
    print("=" * 70)
    t1 = table1_pemsbay_vs_ustd(records)
    print(t1.to_markdown(floatfmt=".4f"))
    t1.to_csv(out_dir / "table1_pemsbay_vs_ustd.csv")
    print()

    print("=" * 70)
    print("Table 2 — PEMS08F: ours vs DiffSTG / SpecSTG / HySRD-Net (norm. CRPS)")
    print("=" * 70)
    t2 = table2_pems08f_vs_published(records)
    print(t2.to_markdown(floatfmt=".4f"))
    t2.to_csv(out_dir / "table2_pems08f_vs_published.csv")
    print()

    print("=" * 70)
    print("Table 3 — Internal ablation: structured vs isotropic on each dataset")
    print("=" * 70)
    t3 = table3_structured_vs_isotropic(records)
    print(t3.to_markdown(floatfmt=".4f"))
    t3.to_csv(out_dir / "table3_struct_vs_iso.csv")
    print()

    print("=" * 70)
    print("Table 4 — Fitted prior parameters across datasets")
    print("=" * 70)
    t4 = table4_fitted_priors(Path(args.prior_yaml_dir))
    print(t4.to_markdown(floatfmt=".4f"))
    t4.to_csv(out_dir / "table4_fitted_priors.csv")
    print()

    # ── Plots ────────────────────────────────────────────────────────────────
    print(f"Saving plots to {out_dir} …")
    fig_crps_per_horizon(records, out_dir / "fig_crps_per_horizon.png")
    fig_pit_grid(records, out_dir / "fig_pit_grid.png")
    print("  fig_crps_per_horizon.png")
    print("  fig_pit_grid.png")

    print()
    print("Done. CSVs + PNGs saved under", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
