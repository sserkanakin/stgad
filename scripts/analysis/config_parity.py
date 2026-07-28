r"""Assert a new campaign's configs differ from a reference campaign ONLY where intended.

Written after the history-conditioning campaign launched with --val-windows at its
default 64 instead of the ladder's 512. That is not a diagnostic knob: _fast_val scores
that clean-window subset and the resulting CRPS drives best_crps.pt, early stopping and
the LR plateau -- so the new cells would have selected their checkpoint on an 8x smaller
validation set than the forecast-only cells they are compared against. Nothing in the
launch flags looked wrong; only a field-by-field diff of the WRITTEN configs caught it.

Run this after the first few cells of any new campaign write their config.json, before
letting the rest of the queue drain.

usage:
  .venv/bin/python scripts/analysis/config_parity.py \
      --new results/20260728_histcond --ref results/20260721_ladder \
      --ref-arms iso emp --expect cond_past exp
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# recorded but never read unless --eval-after is passed; they carry argparse defaults
# rather than the run's behaviour, so a difference here means nothing.
INERT = {"eval_anc_steps", "eval_ckpt", "eval_steps", "eval_k", "eval_max_windows",
         "eval_gen_batch", "eval_after", "final_val", "final_anc_steps",
         "final_steps_list", "smoke", "fresh", "residual_seed"}


def load(root, arms=None):
    """{(dataset, backbone, prior, seed): config} over a results tree."""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "train", "config.json"), recursive=True):
        rel = os.path.relpath(p, root)
        if arms and rel.split(os.sep)[0] not in arms:
            continue
        try:
            c = json.load(open(p))
        except Exception:
            continue
        out[(c["dataset"], c["backbone"], c["prior"], c["seed"])] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="results tree of the new campaign")
    ap.add_argument("--ref", required=True, help="results tree to match")
    ap.add_argument("--new-arms", nargs="*", default=None)
    ap.add_argument("--ref-arms", nargs="*", default=None,
                    help="restrict the reference to these top-level arm dirs")
    ap.add_argument("--expect", nargs="*", default=["cond_past", "exp"],
                    help="keys that SHOULD differ")
    args = ap.parse_args()

    new, ref = load(args.new, args.new_arms), load(args.ref, args.ref_arms)
    if not new:
        print(f"no configs under {args.new} — nothing to check yet")
        return 0
    expect = set(args.expect)
    print(f"{len(new)} new configs vs {len(ref)} reference configs")
    print(f"expected to differ: {sorted(expect)}\n")

    bad = 0
    for k, c in sorted(new.items(), key=lambda kv: str(kv[0])):
        r = ref.get(k)
        name = "/".join(map(str, k))
        if r is None:
            print(f"  {name:<34} (no reference twin — new cell)")
            continue
        diffs = {kk for kk in set(c) | set(r)
                 if kk not in INERT and c.get(kk) != r.get(kk)}
        unexpected = sorted(diffs - expect)
        unchanged = sorted(expect - diffs)
        if not unexpected and not unchanged:
            print(f"  {name:<34} OK")
            continue
        bad += 1
        print(f"  {name:<34} MISMATCH")
        for kk in unexpected:
            print(f"       unexpected  {kk}: ref={r.get(kk)!r}  new={c.get(kk)!r}")
        for kk in unchanged:
            print(f"       did not change  {kk}")
    print(f"\n{'PARITY OK' if not bad else f'{bad} CELL(S) NEED REVIEW'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
