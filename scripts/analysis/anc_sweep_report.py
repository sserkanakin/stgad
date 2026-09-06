r"""Read the ancestral-sampler resolution sweep and answer one question.

    Is the structured prior's under-dispersion a property of the prior, or of
    where the sampler stops?

The ancestral sampler walks a geometric grid from c_0 down to a terminal floor
c_lo and returns the denoiser's R0_hat there. A mode of precision theta is only
resolved once c_bar << 1/theta, so a floor above 1/theta_max leaves the fastest
modes pinned at their conditional mean with no sampled variability. The isotropic
control has theta = 0.8 flat (1/theta = 1.25, far above any floor used here); the
structured prior has theta_max 16-222, i.e. 1/theta_max = 0.004-0.06, at or below
the locked protocol's c_lo = 0.02.

The prediction, if the floor is what matters:
  * the isotropic arm should barely move when c_lo is lowered,
  * the structured arm should gain, and
  * the gain should scale with stiffness -- large on mlp/gcn, small on gwn.

If instead the arms move together, or the structured arm does not gain, the
prior explanation stands and this rules out the obvious objection to it.

usage: python scripts/analysis/anc_sweep_report.py [--campaign 20260728_histcond]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results"
FNAME = re.compile(r"test_metrics_bestcrps_ancestral(\d+)(?:_clo([0-9.eE+-]+))?\.json$")
DEFAULT_C_LO = 0.02          # reverse_sample_ancestral's default


def collect(campaign: str) -> dict:
    """rows[(ds, bb, arm, seed)][(steps, c_lo)] = metrics."""
    rows = defaultdict(dict)
    for p in (ROOT / campaign).glob("*/*/*/*/seed*/eval/test_metrics_bestcrps_ancestral*.json"):
        m = FNAME.search(p.name)
        if not m:
            continue
        steps = int(m.group(1))
        c_lo = float(m.group(2)) if m.group(2) else DEFAULT_C_LO
        try:
            d = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        armdir = p.parts[-7]                       # e.g. "iso-hist"
        arm = armdir.split("-")[0]
        rows[(d["dataset"], d["backbone"], arm, d["seed"])][(steps, c_lo)] = d
    return rows


def theta_max(ds: str, bb: str, arm: str) -> float:
    """1/theta_max is the transition scale of the fastest mode -- the quantity the
    floor has to sit below. cheap enough to recompute; None if inputs are absent."""
    try:
        import sys
        sys.path.insert(0, str(ROOT.parent))
        from src.diffusion.prior import load_prior
        pr = "isotropic" if arm == "iso" else "kronecker"
        kw = {}
        return float(load_prior(ds, pr, 42, backbone=bb, norm="pernode", **kw).theta.max())
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="20260728_histcond")
    ap.add_argument("--metrics", default="coverage,crps,winkler,energy")
    a = ap.parse_args()
    metrics = a.metrics.split(",")

    rows = collect(a.campaign)
    if not rows:
        raise SystemExit(f"no ancestral evals found under results/{a.campaign}")

    # every (steps, c_lo) combination present, default first
    combos = sorted({k for v in rows.values() for k in v},
                    key=lambda t: (t[0], -t[1]))
    print(f"campaign {a.campaign}: {len(rows)} (cell, arm, seed) groups, "
          f"combinations {combos}\n")

    base = (50, DEFAULT_C_LO)
    cells = sorted({(ds, bb) for ds, bb, _, _ in rows})
    for combo in combos:
        if combo == base:
            continue
        print(f"===== {combo[0]} steps, c_lo = {combo[1]:g}   vs   "
              f"locked protocol ({base[0]} steps, c_lo = {base[1]:g}) =====")
        hdr = f"{'ds':9s} {'bb':4s} {'arm':4s} {'1/th_max':>9s}  " + \
              "  ".join(f"{m[:9]:>19s}" for m in metrics)
        print(hdr)
        for ds, bb in cells:
            for arm in ("iso", "emp", "rsh"):
                seeds = [s for (d, b, ar, s) in rows if (d, b, ar) == (ds, bb, arm)]
                pairs = [(rows[(ds, bb, arm, s)].get(base), rows[(ds, bb, arm, s)].get(combo))
                         for s in seeds]
                pairs = [(x, y) for x, y in pairs if x and y]
                if not pairs:
                    continue
                tm = theta_max(ds, bb, arm)
                out = []
                for m in metrics:
                    b0 = np.mean([x[m] for x, _ in pairs])
                    b1 = np.mean([y[m] for _, y in pairs])
                    out.append(f"{b0:8.4f}->{b1:8.4f}")
                print(f"{ds:9s} {bb:4s} {arm:4s} {1 / tm:9.4f}  " + "  ".join(out))
        print()

    # the falsification test: does the structured arm's coverage gain track stiffness?
    print("===== does the coverage gain track 1/theta_max? =====")
    print("if the terminal floor is the cause, the gain should be large where")
    print("1/theta_max << c_lo and near zero where 1/theta_max > c_lo.\n")
    for combo in combos:
        if combo == base or combo[1] >= DEFAULT_C_LO:
            continue
        xs, ys, labels = [], [], []
        for ds, bb in cells:
            for arm in ("emp", "rsh"):
                seeds = [s for (d, b, ar, s) in rows if (d, b, ar) == (ds, bb, arm)]
                pairs = [(rows[(ds, bb, arm, s)].get(base), rows[(ds, bb, arm, s)].get(combo))
                         for s in seeds]
                pairs = [(x, y) for x, y in pairs if x and y]
                if not pairs:
                    continue
                tm = theta_max(ds, bb, arm)
                if not np.isfinite(tm):
                    continue
                gain = np.mean([y["coverage"] - x["coverage"] for x, y in pairs])
                xs.append(1 / tm); ys.append(gain); labels.append(f"{ds}/{bb}/{arm}")
        if len(xs) < 3:
            print(f"  {combo}: only {len(xs)} points, not enough to correlate")
            continue
        r = float(np.corrcoef(np.log(xs), ys)[0, 1])
        print(f"  {combo}: corr(log 1/theta_max, coverage gain) = {r:+.3f}  over {len(xs)} cells")
        for lbl, x, y in sorted(zip(labels, xs, ys), key=lambda t: t[1]):
            print(f"    {lbl:22s} 1/th_max={x:.4f}  coverage {y:+.4f}")


if __name__ == "__main__":
    main()
