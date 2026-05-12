# Running on DAIC (TU Delft AI Cluster)

Runbook for executing the thesis-diffusion pipeline on DAIC. Docs:
<https://daic.tudelft.nl/docs/>.

## What's in this folder

| File | Purpose |
|---|---|
| `env.example.sh` | Copy → `env.sh`, edit NetID + paths + group dir. `env.sh` is gitignored. |
| `setup_env.sh` | One-time bootstrap: loads modules, builds the project venv with CUDA torch. |
| `_common.sh` | Sourced by every sbatch script: loads modules, activates venv, sets up artefact symlinks, prints diagnostics. |
| `sync_to_daic.sh` | Push code + data + checkpoints from local Mac → DAIC. |
| `sync_from_daic.sh` | Pull results back. |
| `submit_train_backbone.sbatch` | GWN backbone training. A40 GPU, medium QoS. |
| `submit_extract_residuals.sbatch` | Inference + residual dump. A40, short QoS. |
| `submit_fit_prior.sbatch` | NumPy prior fit. CPU only, short QoS. |
| `submit_sanity_checks.sbatch` | Re-run exp-004 sanity checks. CPU only. |
| `submit_diffusion.sbatch` | exp-003 score-net training. A40, medium QoS. **Driver script not written yet** — see PLAN.md step §14.4. |

## First-time setup (local Mac)

```bash
cp scripts/daic/env.example.sh scripts/daic/env.sh
$EDITOR scripts/daic/env.sh   # set NetID, group name in DAIC_DATA_DIR
source scripts/daic/env.sh
```

Add to `~/.ssh/config` (locally) so `ssh daic` works:
```
Host daic
  HostName login.daic.tudelft.nl
  User <your-netid>
```

You must be on the TU Delft network or connected to **eduVPN** to reach DAIC.
DAIC does not accept SSH keys — every login is a Kerberos password prompt.

## First-time setup (on DAIC)

```bash
ssh daic
git clone <repo-url> ~/research-tud      # or use staff-umbrella path
cd ~/research-tud
cp scripts/daic/env.example.sh scripts/daic/env.sh
$EDITOR scripts/daic/env.sh
source scripts/daic/env.sh
bash scripts/daic/setup_env.sh

# One-time Kerberos: enables automatic ticket renewal for Slurm jobs.
install_keytab
```

`setup_env.sh` creates `${DAIC_VENV}` (default `~/research-tud/.venv`) and
installs torch+cu121 wheels matching the `cuda/12.1` module.

## Pushing data from local

After running the pipeline locally to produce processed npz / checkpoints /
residuals, push them up:

```bash
# locally, repo root, env.sh sourced
bash scripts/daic/sync_to_daic.sh                 # code + data + checkpoints + residuals
bash scripts/daic/sync_to_daic.sh --code-only     # iterate on a code change
bash scripts/daic/sync_to_daic.sh --data-only     # refresh just the artefacts
```

Then on DAIC, symlink the data dir into the project tree (the sync helper
prints the exact commands, and `_common.sh` makes them idempotent on each
job, so this is automatic the first time any sbatch script runs):

```bash
cd ${DAIC_PROJECT_DIR}
ln -sfn ${DAIC_DATA_DIR}/raw          data/raw
ln -sfn ${DAIC_DATA_DIR}/processed    data/processed
ln -sfn ${DAIC_DATA_DIR}/checkpoints  results/checkpoints
ln -sfn ${DAIC_DATA_DIR}/residuals    results/residuals
ln -sfn ${DAIC_DATA_DIR}/prior_params results/prior_params
```

## Submitting jobs

Renew the Slurm Kerberos cache before submitting (cheap; do it once per
session unless you set up `install_keytab` permanent renewal):

```bash
auks -a
```

Then:

```bash
# Backbone training (already done on the Mac; only re-run if you change architecture)
sbatch --export=DATASET=metrla,SEED=0 scripts/daic/submit_train_backbone.sbatch
sbatch --export=DATASET=pemsbay,SEED=0 scripts/daic/submit_train_backbone.sbatch

# Residual extraction (after backbone training finishes)
sbatch --export=DATASET=metrla,SEED=0 scripts/daic/submit_extract_residuals.sbatch
sbatch --export=DATASET=pemsbay,SEED=0 scripts/daic/submit_extract_residuals.sbatch

# Prior fit (CPU only — fast)
sbatch --export=DATASET=metrla scripts/daic/submit_fit_prior.sbatch
sbatch --export=DATASET=pemsbay scripts/daic/submit_fit_prior.sbatch

# exp-004 sanity checks (CPU only)
sbatch scripts/daic/submit_sanity_checks.sbatch

# exp-003 score net (after step §14.4 of PLAN.md adds train_diffusion.py)
# Full sweep: 5 seeds × 2 priors × 2 datasets = 20 jobs.
for ds in metrla pemsbay; do
  for prior in structured isotropic; do
    for seed in 0 1 2 3 4; do
      sbatch --export=DATASET=$ds,SEED=$seed,PRIOR=$prior \
             scripts/daic/submit_diffusion.sbatch
    done
  done
done
```

## Monitoring

```bash
squeue --me                                          # your jobs
squeue --me -t RUNNING -o "%i %T %M %j %R"           # what's running, time used
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State # post-mortem
tail -f logs/slurm-exp003-dsm-<jobid>.out            # live log
scancel <jobid>                                      # kill a job
```

Interactive GPU shell for debugging:

```bash
sinteractive --cpus-per-task=2 --mem=8G --time=01:00:00 --gres=gpu:a40:1
```

## Pulling results back

```bash
# locally, repo root, env.sh sourced
bash scripts/daic/sync_from_daic.sh                       # everything
bash scripts/daic/sync_from_daic.sh --no-checkpoints      # skip ~GB of .pt
bash scripts/daic/sync_from_daic.sh --no-residuals        # skip ~GB of .npz
```

## Hardware crib sheet

| GPU | Mem | Nodes | Use for | Slurm flag |
|---|---|---|---|---|
| A40 | 46 GB | 84 | default workhorse | `--gres=gpu:a40:1` |
| L40 | 49 GB | 18 | larger batches / faster | `--gres=gpu:l40:1` |
| V100 | 32 GB | 11 | older, FP64 ok | `--gres=gpu:v100:1` |
| RTX 2080 Ti | 11 GB | 24 | small models, plentiful | `--gres=gpu:turing:1` |

QoS levels on the `general` partition (DAIC priorities page):

| QoS | Max time | Use for |
|---|---|---|
| `interactive` | 1 h | sinteractive; do not sbatch with this |
| `short` | 4 h | residual extraction, prior fit, sanity checks |
| `medium` | 1.5 d | backbone training, single exp-003 run |
| `long` | 7 d | only if needed; lowest priority |

CPUs are always allocated in multiples of 2 — `--cpus-per-task=4` is the
project default, ample for our dataloader + GPU pipeline.

## Storage tiers

- `$HOME` — small quota, slow; ok for code + venv.
- `staff-umbrella/<group>/...` — DAIC-recommended for research data, larger
  quota, snapshotted. Default location for `${DAIC_DATA_DIR}`.
- Compute-node `/tmp` — local NVMe; useful for scratch but evaporates at
  job end.

Never use `sshfs` against the login nodes — overloads the shared filesystem.
`rsync -avz --no-perms` (used by the helpers) is the supported path.

## Known DAIC quirks

- **No public-key SSH** — Kerberos password every session.
- **CPUs in pairs** — Slurm rounds `--cpus-per-task` up to the next even number.
- **Lmod uses `/opt/insy/modulefiles`** — `_common.sh` adds it; you only need
  `module load …` once you've done `module use`.
- **Time tickets expire after 10 h** — `install_keytab` enables auto-renewal,
  otherwise long jobs need a `screen` session or periodic `auks -a`.
- **Multi-partition submission** — combine partitions (e.g.
  `--partition=insy,general`) to take whichever opens first. Defaults to
  `general` here for portability.
