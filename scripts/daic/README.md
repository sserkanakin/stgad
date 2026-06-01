# Running on DAIC (TU Delft AI Cluster)

End-to-end runbook for executing the thesis-diffusion pipeline on DAIC.
Official docs: <https://daic.tudelft.nl/docs/>.

Six-phase project plan mirrors `/Users/serkan/.claude/plans/lets-plan-from-this-fuzzy-hopcroft.md`:

```
   Phase 1  Backbone (GWN) + residual extraction     ✅ DONE
   Phase 2  Kronecker prior fit + diagnostics        ✅ DONE
   Phase 3  Forward sampler (FCPS, eigenbasis)       Stage B
   Phase 4  Denoiser (UGnet) + training harness      Stage C
   Phase 5  Reverse sampling + headline experiments  Stage D
   Phase 6  Writeup                                   Stage E
```

The pre-rewrite diffusion / DSM / eval submission scripts were removed in the
2026-05-28 cleanup. New sbatch templates land alongside the corresponding stage.

## One-time setup (per machine)

### On your Mac

```bash
cp scripts/daic/env.example.sh scripts/daic/env.sh
$EDITOR scripts/daic/env.sh                   # set DAIC_NETID, group dir
source scripts/daic/env.sh
```

Add to `~/.ssh/config` so `ssh daic` works:

```
Host daic
  HostName login.daic.tudelft.nl
  User <your-netid>
```

You must be on TU Delft network or **eduVPN** to reach DAIC.
SSH public keys are NOT supported — every login is a Kerberos password prompt.

### On DAIC (first login)

```bash
ssh daic
git clone <repo-url> ~/research-tud
cd ~/research-tud
cp scripts/daic/env.example.sh scripts/daic/env.sh
$EDITOR scripts/daic/env.sh                   # same NetID + paths
source scripts/daic/env.sh

bash scripts/daic/setup_env.sh                # conda env, torch+cu121
wandb login                                   # paste key from wandb.ai/authorize
install_keytab                                # Slurm Kerberos auto-renewal
```

`setup_env.sh` is idempotent.

## Pushing artefacts (Mac → DAIC)

Code travels via `git push` / `git pull` on DAIC. `sync_to_daic.sh` pushes
the large binary artefacts that aren't in git.

```bash
bash scripts/daic/sync_to_daic.sh                  # default: raw + processed + checkpoints
```

Flags:

| Flag | Effect |
|---|---|
| (no flag) | raw + processed + checkpoints |
| `--all` | + residuals + priors |
| `--raw` | raw .h5 / PEMS08.npz + adjacency only |
| `--processed` | processed .npz + scaler YAMLs only |
| `--checkpoints` | backbone checkpoints only |
| `--residuals` | extracted residual .npz only |
| `--priors` | fitted prior outputs only |

Specific-target flags are exclusive.

## Pre-flight (login node)

```bash
ssh daic
cd ~/research-tud && source scripts/daic/env.sh
bash scripts/daic/verify_setup.sh
```

Checks env, modules, conda env, imports, symlinks, data files, wandb auth.
No Slurm, no GPU.

## GPU smoke test

```bash
sbatch --export=DATASET=metrla scripts/daic/submit_smoke.sbatch
```

Confirms CUDA + wandb + repo imports. Once green, you're cleared.

## Phase 1 — Backbone + residuals ✅ DONE

Outputs already cached under `results/checkpoints/` and `results/residuals/`.
**Do not rerun** unless intentionally retraining a backbone or invalidating a
residual cache.

If you do need to rerun:

```bash
# Backbone (A40, medium, ~12h)
sbatch --export=DATASET=metrla,SEED=0  scripts/daic/submit_train_backbone.sbatch
sbatch --export=DATASET=pemsbay,SEED=0 scripts/daic/submit_train_backbone.sbatch
sbatch --export=DATASET=pems08f,SEED=0 scripts/daic/submit_train_backbone.sbatch

# Residual extraction (A40, short, ~1h)
sbatch --export=DATASET=metrla,SEED=0  scripts/daic/submit_extract_residuals.sbatch
sbatch --export=DATASET=pemsbay,SEED=0 scripts/daic/submit_extract_residuals.sbatch
sbatch --export=DATASET=pems08f,SEED=0 scripts/daic/submit_extract_residuals.sbatch
```

## Phase 2 — Prior + diagnostics ✅ DONE

γ* locked per dataset (geomean of energy + spectral fits):
METR-LA 0.158, PEMS-BAY 0.213, PEMS08F 0.924.
Acceptance suite (D1/D2/D3) ran on all three datasets.

Outputs under `results/prior_params/` and `results/section3/`.
**Do not rerun** unless changing the prior spec. Driver scripts live under
`scripts/section3/` (exp31–36, diag_prior_kron, exp_temporal_stationarity).

## Phase 3 — Forward sampler (Stage B, coming)

To be added: `scripts/daic/submit_forward_sanity.sbatch` once
`src/diffusion/forward_fcps.py` lands. Will run the three forward-sampler sanity
checks + per-dataset mixing diagnostic on CPU.

## Phase 4 — Denoiser training (Stage C) 🚧

Trains the R0Denoiser (UGnet-based) via the GAD forward FCPS + DDIM reverse
plumbing. A40, medium QoS, 24 h budget.

```bash
# Single run
sbatch --job-name="diff-metrla-s0" \
       --export=DATASET=metrla,SEED=0 \
       scripts/daic/submit_train_diffusion.sbatch

# 10-job sweep (5 seeds × 2 headline datasets)
for ds in metrla pems08f; do
  for seed in 0 1 2 3 4; do
    sbatch --job-name="diff-${ds}-s${seed}" \
           --export=DATASET=$ds,SEED=$seed \
           scripts/daic/submit_train_diffusion.sbatch
  done
done
```

Optional overrides via `--export`: `EPOCHS`, `BATCH_SIZE`, `LR`, `D_H`, `C_0`,
`ALPHA`, `EXTRA_FLAGS` (passed verbatim to the python script). The sbatch
auto-creates `logs/train-diffusion/` on first invocation.

## Phase 5 — Evaluation (Stage D) 🚧

Generate K reverse samples per window, score them, and build the comparison
tables + figures in one job (auto-discovers the trained checkpoint):

```bash
sbatch --export=DATASET=metrla,SEED=0,SPLIT=test scripts/daic/submit_generate_samples.sbatch
```

Optional `--export` knobs: `K` (default 100), `N_STEPS` (100), `ETA` (1.0),
`WHICH` (best|last), `CKPT_DIR` (override auto-discovery), `RUN_EVAL` (1|0 — also
runs the CPU evaluate/diagnostics/figures chain), `EXTRA_FLAGS`. Outputs land in
`results/eval/${DATASET}/` (metrics JSON, `comparison_*.md/.csv`, figures). What
each metric and figure means is documented in `scripts/eval/README.md`. The log
dir `logs/eval-samples/` is created by the sbatch on first submit.

## Monitoring (any phase)

```bash
bash scripts/daic/watch_jobs.sh           # one-shot squeue + wandb link + newest log
bash scripts/daic/watch_jobs.sh --follow  # refreshes every 10 s
bash scripts/daic/watch_jobs.sh --tail    # tail -F the newest slurm log
```

Wandb: <https://wandb.ai/tud-thesis/thesis-diffusion>.
`init_wandb` uses run IDs `{backbone}_{dataset}_seed{seed}_{cfg_hash}` suffixed
with `SLURM_JOB_ID` so manual resubmits get fresh dashboard entries.

Direct Slurm:

```bash
squeue --me                                            # your queue
squeue --me -t RUNNING -o "%i %T %M %j %R"             # running with elapsed
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State   # post-mortem
scancel <jobid>                                        # kill
sinteractive --cpus-per-task=2 --mem=8G --gres=gpu:a40:1 --time=01:00:00
```

## Pulling results (DAIC → Mac)

```bash
bash scripts/daic/sync_from_daic.sh                       # everything
bash scripts/daic/sync_from_daic.sh --no-checkpoints      # skip GB of .pt
bash scripts/daic/sync_from_daic.sh --no-residuals        # skip GB of .npz
bash scripts/daic/sync_from_daic.sh --no-priors           # skip prior YAMLs
```

Always grabs `results/**/*.{md,json,png,yaml}` so SUMMARY.md updates and
figures come back regardless of flags.

## File reference

| File | Phase | Purpose |
|---|---|---|
| `env.example.sh` | setup | Per-user config template. Copy → `env.sh` (gitignored). |
| `setup_env.sh` | setup | One-time conda env + CUDA torch install. |
| `verify_setup.sh` | setup | Login-node pre-flight. No Slurm, no GPU. |
| `_common.sh` | shared | Sourced by every sbatch. Modules + conda env + symlinks. |
| `sync_to_daic.sh` | sync | rsync Mac → DAIC. |
| `sync_from_daic.sh` | sync | rsync DAIC → Mac. |
| `submit_smoke.sbatch` | smoke | 5-min A40 sanity. |
| `smoke_test.py` | smoke | Python driver behind submit_smoke. |
| `submit_train_backbone.sbatch` | 1 | GWN training. A40, medium, 12h. ✅ done. |
| `submit_extract_residuals.sbatch` | 1 | Inference + residual dump. A40, short, 1h. ✅ done. |
| `submit_train_diffusion.sbatch` | 4 | Diffusion denoiser training. A40, medium, 24h. |
| `submit_generate_samples.sbatch` | 5 | Sampling + scoring + figures. A40, short, 4h. |
| `watch_jobs.sh` | monitor | Local live monitor. |

## Hardware crib sheet

| GPU | Mem | Nodes | Use for | Slurm flag |
|---|---|---|---|---|
| A40 | 46 GB | 84 | default workhorse | `--gres=gpu:a40:1` |
| L40 | 49 GB | 18 | larger batches / faster | `--gres=gpu:l40:1` |
| V100 | 32 GB | 11 | older, FP64 ok | `--gres=gpu:v100:1` |
| RTX 2080 Ti | 11 GB | 24 | small models, plentiful | `--gres=gpu:turing:1` |

QoS on the `general` partition:

| QoS | Max time | Use for |
|---|---|---|
| `interactive` | 1 h | `sinteractive` only — do not `sbatch` with this |
| `short` | 4 h | residual extraction, prior fit, sanity checks, smoke |
| `medium` | 1.5 d | backbone, single diffusion training run |
| `long` | 7 d | only if needed; lowest priority |

CPUs allocate in multiples of 2; default `--cpus-per-task=4`.

## Storage tiers

- `$HOME` — small quota, slow. Code + logs. Conda env under `$HOME/.conda/envs/` (~3 GB).
- `staff-umbrella/<group>/<netid>/research-tud-data` — research-data tier,
  larger quota, snapshotted. `DAIC_DATA_DIR` points here; `_common.sh` symlinks
  `data/`, `results/checkpoints`, `results/residuals`, `results/prior_params`
  into the project tree per job.
- `/tmp` on compute nodes — local NVMe scratch, evaporates at job end.

Never `sshfs` against login nodes. Use `rsync -avz --no-perms` (the helpers do).

## Known DAIC quirks

- **No public-key SSH** — Kerberos password every session. Use `eduVPN` off-campus.
- **CPUs in pairs** — Slurm rounds `--cpus-per-task` up to an even number.
- **Lmod at `/opt/insy/modulefiles`** — `_common.sh` does `module use` for you.
- **Kerberos tickets expire after 10 h** — `install_keytab` enables auto-renewal.
- **Multi-partition submission** — `--partition=insy,general` takes whichever opens first.
- **wandb egress allowed by default** — if restricted, set `WANDB_MODE=offline` in `env.sh`.
