"""quick reduced-setting run of the two val evals for a supervisor update.
forces CPU (MPS OOMs on the graph-conv here). prints rows as they land.

phase 1 (cheap): reconstruction MAE/MSE for all 3 datasets x kron/iso.
phase 2 (heavier): generative gen-MAE/MSE + W1/energy for the headline datasets.

NOT publication settings — directional numbers only. seed 42.
"""
from __future__ import annotations

import json
import time

import os
import sys

import torch

# run as `python notebooks/_quick_eval_for_supervisor.py` from repo root: sys.path[0]
# is notebooks/, so import the lib directly (it adds the repo root for src/scripts).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import val_eval_lib as ve

DEV = torch.device("cpu")          # MPS OOMs on the DiffSTG spatial conv
SEED = 42

RECON = dict(n_steps=25, max_windows=128, chunk=64, log_every=12)
GEN = dict(k=12, n_steps=25, eta=1.0, alpha=0.1, max_windows=96,
           gen_batch=192, dist_windows=96, dist_samples=4, w1_proj=64)
GEN_DATASETS = ("metrla", "pems08f", "pemsbay")   # order: headline first

results = {}


def main():
    print(f"device={DEV} seed={SEED}\nRECON={RECON}\nGEN={GEN}\n", flush=True)

    print("── phase 1: reconstruction MAE/MSE (all datasets) ──", flush=True)
    for ds in ve.DATASETS:
        runs = ve.load_runs(ds, device=DEV, seed=SEED)
        for prior in ("kronecker", "isotropic"):
            if prior not in runs:
                continue
            t0 = time.time()
            r = ve.reconstruction_eval(runs[prior], device=DEV,
                                       generator=ve.make_generator(DEV, SEED),
                                       verbose=False, **RECON)
            results[(ds, prior, "recon")] = r
            print(f"  {ds:8s} {prior:9s}  recon MAE={r['mae']:.3f}  MSE={r['mse']:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print("\n── phase 2: generative gen-MAE/MSE + W1/energy ──", flush=True)
    for ds in GEN_DATASETS:
        runs = ve.load_runs(ds, device=DEV, seed=SEED)
        for prior in ("kronecker", "isotropic"):
            if prior not in runs:
                continue
            t0 = time.time()
            r = ve.generative_eval(runs[prior], device=DEV,
                                   generator=ve.make_generator(DEV, SEED),
                                   verbose=False, **GEN)
            results[(ds, prior, "gen")] = r
            print(f"  {ds:8s} {prior:9s}  gen MAE={r['mae']:.3f}  MSE={r['mse']:.3f}  "
                  f"W1={r['w1']:.3f}  energy={r['energy']:.3f}  cov={r['coverage']:.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        _dump()

    _dump()
    print("\nDONE", flush=True)


def _dump():
    out = []
    for (ds, prior, kind), r in results.items():
        row = {"dataset": ds, "prior": prior, "kind": kind,
               "mae": r["mae"], "mse": r["mse"]}
        if kind == "gen":
            row.update(w1=r["w1"], energy=r["energy"], coverage=r["coverage"],
                       winkler=r["winkler"])
        out.append(row)
    with open("notebooks/_supervisor_eval_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
