# Running on DAIC — quickstart

One entry point, `scripts/run.sh`, hides the env-var soup behind three verbs:

```bash
scripts/run.sh profile <dataset> [arm]        # short profiled run (size resources)
scripts/run.sh train   <dataset> <seed> [arm] # one full diffusion run
scripts/run.sh all                            # submit every missing stage (idempotent)
```

`arm` = **kron** (structured kronecker, the contribution) · **flat** (temporal-only) ·
**iso** (isotropic baseline). Default `kron`. Set `EXP=<tag>` to namespace a run,
`LOSS=<snr|mse|mahalanobis>` to pick the loss (default `snr`).

The verbs wrap the tested `submit.sh` / `resume_pipeline.sh` — those still work directly
if you need the low-level knobs.

---

## The flow: sync → profile → size → submit

### 1. Sync code + data to DAIC
```bash
bash scripts/daic/sync_to_daic.sh           # code
# residuals must exist on DAIC before diffusion training. either:
#   - run the backbone + extract stages (scripts/run.sh on the backbone/residuals
#     stages, or submit_pipeline.sh which chains them), or
#   - sync the local bootstrapped residuals up (results/residuals/<ds>/seed0/).
```

### 2. Profile each dataset × arm (structured + iso)
Run a short **production-config** job (real batch size + the val diagnostic, but only
~40 train batches) so the resource numbers reflect a real run:
```bash
for ds in metrla pems08f pemsbay; do
  for arm in kron iso; do
    scripts/run.sh profile $ds $arm
  done
done
```
Each job prints a resource block at the end of its `.out` log:
```
===================== resource profile =====================
GPU: NVIDIA A40, 46068 MiB
peak GPU mem used: 12345 MiB | peak GPU util: 88%
CPU/RAM/walltime (run AFTER the job ends):
  seff <jobid>
  sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,State%20
============================================================
```

### 3. Read the numbers and size production resources
| resource | from | sbatch knob |
|---|---|---|
| GPU memory | `peak GPU mem used` | `--gres` (pick a GPU with headroom) |
| CPU RAM | `sacct … MaxRSS` | `--mem` (set ~1.3× MaxRSS) |
| walltime | per-epoch time × epochs (`epoch NNN … (Xs)` lines) | `--time` |
| CPU cores | `seff` CPU efficiency | `--cpus-per-task` |

The defaults in `submit_train_diffusion.sbatch` are `--cpus-per-task=4 --mem=32G
--time=06:00:00`. Bump them from the profile if needed — e.g. heavier datasets
(pemsbay, N=325) drive the val-diagnostic reverse sampler hardest.

### 4. Submit the production jobs
```bash
EXP=<tag> scripts/run.sh all          # all datasets × seeds × {kron, iso}, idempotent
# or one at a time:
EXP=<tag> scripts/run.sh train metrla 42 kron --time=02:00:00
```
`all` chains backbone → residuals → diffusion → eval with slurm `afterok` deps, and
skips any cell already on disk. Re-run it anytime; it only submits what's missing.

---

## Notes
- `profile` and `train` set `PROFILE=1`, so every diffusion job writes
  `logs/diffusion/profile_*.csv` and the resource block above.
- The **flat** arm shares the `kronecker` results dir with **kron** — only run one of
  them per `EXP`, or give flat its own `EXP` tag, until the path key is split.
- Watch jobs: `squeue --me` · pull results back: `bash scripts/daic/sync_from_daic.sh`.
