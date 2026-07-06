"""cross-dataset summary of the forward-analysis jsons.

one compact validation table: for each dataset, does the forward drive the graph
signal metrics to the stationary prior? we report, per operator, the TV mean at
(data, c̄_hi, prior) and the wasserstein-1 distance to the prior at the loosest
and stiffest noise level — the headline `→0` convergence.
"""
from __future__ import annotations

import glob
import json
import os

from src.utils import paths

METRICS = ("tv", "sc", "degcorr")


def _rel(a: float, b: float) -> float:
    return abs(a - b) / (abs(b) + 1e-12) * 100.0


def main() -> None:
    files = sorted(glob.glob(str(paths.RESULTS / "forward_analysis" / "*.json")))
    if not files:
        print("no forward_analysis jsons found — run scripts/forward_analysis.py first")
        return

    for op_name in ("combinatorial", "normalized"):
        print(f"\n{'='*96}\noperator: {op_name}\n{'='*96}")
        print(f"{'dataset':>9} | {'TV data':>9} {'TV c̄_hi':>9} {'TV prior':>9} {'relerr%':>8}"
              f" | {'W1(TV) lo→hi':>18} | {'W1(SC) lo→hi':>16} | {'W1(dc) hi':>9}")
        for f in files:
            res = json.load(open(f))
            op = res["operators"][op_name]
            grid = res["grid"]
            c_lo, c_hi = str(float(grid[0])), str(float(grid[-1]))
            m = op["means"]
            tv_d, tv_hi, tv_p = m["data"]["tv"], m[str(grid[-1])]["tv"], m["prior"]["tv"]
            d = op["distances"]
            w_tv = (d[c_lo]["tv"]["wasserstein"], d[c_hi]["tv"]["wasserstein"])
            w_sc = (d[c_lo]["sc"]["wasserstein"], d[c_hi]["sc"]["wasserstein"])
            w_dc = d[c_hi]["degcorr"]["wasserstein"]
            print(f"{res['dataset']:>9} | {tv_d:9.3g} {tv_hi:9.3g} {tv_p:9.3g} {_rel(tv_hi, tv_p):8.2f}"
                  f" | {w_tv[0]:8.3g} → {w_tv[1]:<7.3g} | {w_sc[0]:7.3g} → {w_sc[1]:<6.3g} | {w_dc:9.3g}")


if __name__ == "__main__":
    main()
