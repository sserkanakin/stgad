"""thin CLI for the forward-process analysis (logic lives in src/eval/forward_analysis.py).

runs our forward sde on cached residuals across a c̄ sweep and measures gad's graph
signal metrics (TV/SC/DegCorr) vs samples from the stationary prior. see the module
docstring for the method.
"""
from __future__ import annotations

import argparse
import json
import os

# re-exported so `from scripts.forward_analysis import run, build_operators, ...`
# keeps working (the logic moved to src/eval/forward_analysis.py so the plot script
# and notebooks can import it from the installed `src` package).
from src.eval.forward_analysis import (  # noqa: F401
    run, print_report, build_operators, _features, METRICS,
)
from src.utils import paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="metrla")
    ap.add_argument("--backbone", default="gwn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--n-steps", type=int, default=7)
    ap.add_argument("--out", default=None, help="json path (default results/forward_analysis/<ds>.json)")
    args = ap.parse_args()

    res = run(args.dataset, args.seed, args.backbone, args.n_samples, args.n_steps)
    print_report(res)

    out = args.out or str(paths.RESULTS / "forward_analysis" / f"{args.dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
