#!/usr/bin/env python
"""
Eval-only recovery for the DiffSTG baseline. Loads a saved best-val checkpoint and
runs DiffSTG's OWN final test eval (ddim_multi, 40 steps + their Metric) to produce
the MAE/RMSE/MAPE/CRPS/MIS row -- WITHOUT retraining.

Use it when:
  - a training job hits the walltime before early-stopping (the best-val model is
    saved on every improvement -- torch.save(model, ...) -- so it survives the kill), or
  - you want to draw more samples (e.g. --n-samples 100) from any finished checkpoint.

DiffSTG's train.py has no resume, and a plain resubmit retrains from scratch AND
overwrites the checkpoint on its first improvement -- so run this BEFORE any resubmit.

Runs the eval from inside third_party/DiffSTG (like train.py) so `train`/`utils`/
`algorithm` import and the ws-relative output paths resolve:
    python scripts/baselines/eval_diffstg_ckpt.py --data pemsbay
    python scripts/baselines/eval_diffstg_ckpt.py --data pemsbay --n-samples 100
    python scripts/baselines/eval_diffstg_ckpt.py --data pemsbay --ckpt /abs/path.dm4stg
"""
import os
import sys
import glob
import csv
import argparse

import torch

# run from the DiffSTG repo root so its imports + ws paths resolve.
HERE = os.path.dirname(os.path.abspath(__file__))
DIFFSTG = os.path.abspath(os.path.join(HERE, "..", "..", "third_party", "DiffSTG"))
sys.path.insert(0, DIFFSTG)
os.chdir(DIFFSTG)

from train import default_config, evals          # noqa: E402  (their code = faithful eval)
from algorithm.dataset import CleanDataset, TrafficDataset  # noqa: E402
from utils.eval import Metric                     # noqa: E402
from utils.common_utils import dir_check          # noqa: E402


def find_ckpt(config, data, seed):
    """newest checkpoint matching this dataset + seed under output/model/."""
    pat = os.path.join(config.PATH_MOD, f"*+{data}+*+{seed}*.dm4stg")
    hits = sorted(glob.glob(pat), key=os.path.getmtime)
    if not hits:
        raise SystemExit(f"no checkpoint matching {pat}\n"
                         f"(pass --ckpt with the 'model path:' from the training log)")
    return hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="path to .dm4stg (default: newest match for --data/--seed)")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--sample-steps", type=int, default=40)
    ap.add_argument("--strategy", default="ddim_multi")
    args = ap.parse_args()

    config = default_config(args.data)
    config.is_test = False      # full test set, not the 1-batch smoke path
    config.nni = False
    config.n_samples = args.n_samples
    device = config.device

    clean_data = CleanDataset(config)
    config.model.A = clean_data.adj

    T_h, T_p = config.model.T_h, config.model.T_p
    test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + T_p, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, 64, shuffle=False)

    ckpt = args.ckpt or find_ckpt(config, args.data, args.seed)
    print(f"loading checkpoint: {ckpt}")
    model = torch.load(ckpt, map_location=device, weights_only=False).to(device)
    model.set_ddim_sample_steps(args.sample_steps)  # test sampler = ddim_multi/40, as train.py's final eval
    model.set_sample_strategy(args.strategy)

    config.forecast_path = os.path.join(
        config.PATH_FORECAST, f"{args.data}_seed{args.seed}_recovered.pkl")
    dir_check(config.forecast_path)

    metric = Metric(T_p=T_h + T_p)
    evals(model, test_loader, -1, metric, config, clean_data, mode="test")

    m = metric.metrics
    print(f"\n[recovered] {args.data} seed{args.seed} "
          f"(n_samples={args.n_samples}, steps={args.sample_steps})  "
          f"MAE {m['mae']:.4f}  RMSE {m['rmse']:.4f}  MAPE {m['mape']:.4f}  "
          f"CRPS {m['crps']:.4f}  MIS {m['mis']:.4f}")

    diffstg_home = os.environ.get("DIFFSTG_HOME")
    out_dir = (os.path.join(diffstg_home, "results", "recovered") if diffstg_home
               else os.path.join(os.path.dirname(config.PATH_FORECAST.rstrip("/")), "recovered"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.data}_seed{args.seed}_recovered.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["data.name", "mae", "rmse", "mape", "crps", "mis",
                    "n_samples", "sample_steps", "ckpt"])
        w.writerow([args.data, m["mae"], m["rmse"], m["mape"], m["crps"], m["mis"],
                    args.n_samples, args.sample_steps, ckpt])
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
