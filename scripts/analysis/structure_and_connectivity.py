"""does the residual actually carry spatial + temporal structure, and is the graph
connected so the heat-equation prior can mix every node?

prints, per dataset:
  1. CONNECTIVITY — components of the raw vs repaired graph (BFS/DFS), Fiedler value,
     and the slowest-mode mixing H_T (is anything left un-noised at t=T?).
  2. SPATIAL    — residual correlation vs graph hop distance. if neighbours (1 hop)
     correlate more than far nodes, close sensors are related.
  3. TEMPORAL   — population horizon correlation R_H by lag. if adjacent horizons
     correlate, close horizons are related.

usage:  python scripts/analysis/structure_and_connectivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import stage0_lib as s0                                  # noqa: E402
from src.data.adjacency import connectivity  # noqa: E402
from src.diffusion.prior import load_prior  # noqa: E402

DATASETS = ["metrla", "pems08f", "pemsbay"]


def main() -> None:
    for ds in DATASETS:
        print(f"\n{'='*70}\n{ds}\n{'='*70}")

        # 1. connectivity ----------------------------------------------------
        raw = connectivity(ds, repair=False)
        rep = connectivity(ds, repair=True)
        print("CONNECTIVITY (BFS/DFS graph search + Laplacian spectrum)")
        print(f"  raw graph     : {raw['n_components']} component(s), "
              f"sizes {raw['component_sizes'][:6]}{'...' if len(raw['component_sizes'])>6 else ''}")
        print(f"  repaired graph: {rep['n_components']} component(s), "
              f"Fiedler λ₂ = {rep['fiedler']:.4f}, zero-eig mult = {rep['zero_eig_multiplicity']}")
        print(f"  -> single connected component: {'YES ✓' if rep['ok'] else 'NO ✗'}  "
              f"(heat equation can mix every node)")

        prior = load_prior(ds, prior="kronecker", spatial="structured", seed=0)
        mix = s0.mixing_check(prior)
        print(f"  slowest-mode signal left at t=T : H_T = e^(-c₀θ_min) = {mix['H_T_slowest']:.4f}  "
              f"({'mixed ✓' if mix['H_T_slowest'] < 0.01 else 'NOT mixed ✗'})")
        print(f"  fastest-mode signal left at t=T : H_T = {mix['H_T_fastest']:.1e}")

        val = s0.load_val_fields(ds)

        # 2. spatial ---------------------------------------------------------
        sp = s0.spatial_corr_by_hop(ds, val)
        print("\nSPATIAL — residual correlation vs graph hop distance")
        for d in range(1, 6):
            c, n = sp[d]
            print(f"  {d} hop{'s' if d>1 else ' '}: corr = {c:+.3f}  ({n} pairs)")
        c_all, _ = sp["all"]
        c1, _ = sp[1]
        print(f"  all pairs : corr = {c_all:+.3f}")
        print(f"  -> neighbours vs far: {c1:+.3f} vs {c_all:+.3f}  "
              f"({'close nodes ARE related ✓' if c1 > c_all + 0.01 else 'weak/no spatial signal'})")

        # 3. temporal --------------------------------------------------------
        tp = s0.temporal_corr_by_lag(prior)
        print("\nTEMPORAL — population horizon correlation R_H by lag")
        print("  " + "  ".join(f"lag{k}={tp[k]:+.2f}" for k in range(0, 6)))
        print(f"  -> adjacent horizons (lag 1) corr = {tp[1]:+.3f}  "
              f"({'close horizons ARE related ✓' if tp[1] > 0.2 else 'weak temporal signal'})")


if __name__ == "__main__":
    main()
