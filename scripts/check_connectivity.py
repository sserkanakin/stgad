"""connectivity check for all dataset graphs, BEFORE and AFTER repair.

the heat-equation prior must run on a SINGLE connected component (supervisor
requirement): no isolated sensors, no multi-node clusters. this script verifies
it two independent ways per graph (BFS component count + zero-eigenvalue
multiplicity / Fiedler value of the normalized laplacian), lists every edge the
repair added (node pair, train-signal correlation, weight), and asserts all
post-repair graphs are single-component.

usage:
    python scripts/check_connectivity.py                       # print table
    python scripts/check_connectivity.py --out results/exp1    # + write connectivity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.adjacency import connectivity, load_adjacency, _train_signal_corr

DATASETS = ("metrla", "pems04s", "pems04f", "pems08s", "pems08f", "pemsbay")


def added_edges(dataset: str) -> list[dict]:
    """edges present in the repaired graph but not the raw one (upper triangle)."""
    A0 = np.asarray(load_adjacency(dataset, repair=False), dtype=float)
    A1 = np.asarray(load_adjacency(dataset, repair=True), dtype=float)
    D = (A1 > 0) & ~(A0 > 0)
    ij = np.argwhere(np.triu(D | D.T, 1))
    if len(ij) == 0:
        return []
    C = _train_signal_corr(dataset)
    return [{"i": int(i), "j": int(j), "corr": round(float(C[i, j]), 4),
             "weight": float(A1[i, j])} for i, j in ij]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=None,
                   help="directory to write connectivity.json into (e.g. the experiment root)")
    args = p.parse_args()

    report = {}
    print(f"{'dataset':9s} | {'raw comp':>8s} {'sizes (top4)':>18s} {'zero':>4s} | "
          f"{'rep comp':>8s} {'zero':>4s} {'fiedler':>9s} {'ok':>3s} | {'edges added':>11s}")
    print("-" * 92)
    for ds in DATASETS:
        raw = connectivity(ds, repair=False)
        rep = connectivity(ds, repair=True)
        edges = added_edges(ds)
        sizes = ",".join(str(s) for s in raw["component_sizes"][:4])
        print(f"{ds:9s} | {raw['n_components']:>8d} {sizes:>18s} "
              f"{raw['zero_eig_multiplicity']:>4d} | {rep['n_components']:>8d} "
              f"{rep['zero_eig_multiplicity']:>4d} {rep['fiedler']:>9.5f} "
              f"{'yes' if rep['ok'] else 'NO':>3s} | {len(edges):>11d}")
        report[ds] = {"raw": raw, "repaired": rep, "added_edges": edges}
        assert rep["ok"], f"{ds}: repaired graph is NOT a single component!"

    print("-" * 92)
    print("all repaired graphs are single connected components "
          "(components=1, zero-eig multiplicity=1, fiedler>0)")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "connectivity.json").write_text(json.dumps(report, indent=2))
        print(f"wrote {out / 'connectivity.json'}")


if __name__ == "__main__":
    main()
