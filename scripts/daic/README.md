# Running on DAIC (TU Delft AI Cluster)

End-to-end runbook for executing the thesis-diffusion pipeline on DAIC.
Official docs: <https://daic.tudelft.nl/docs/>.

The workflow has four phases. Once Phase 1+2 are done (one-time setup),
every subsequent training run is a single `sbatch` command.

```
   Phase 1  Setup     ── env.sh, setup_env.sh, install_keytab           [~10 min]
   Phase 2  Verify    ── sync_to_daic.sh --hybrid, verify, smoke job    [~15 min]
   Phase 3  Run       ── extract residuals + fit prior + score net      [~hours]
   Phase 4  Monitor   ── wandb dashboard, watch_jobs.sh, sync_from       [live]
```

---

## Phase 1 — One-time setup

### 1.1  Local Mac

```bash
cp scripts/daic/env.example.sh scripts/daic/env.sh
$EDITOR scripts/daic/env.sh                   # set DAIC_NETID, group dir
source scripts/daic/env.sh
```

Add to your local `~/.ssh/config` so `ssh daic` works:

```
Host daic
  HostName login.daic.tudelft.nl
  User <your-netid>
```

You must be on the TU Delft network or connected to **eduVPN** to reach
DAIC. DAIC does not accept SSH public keys — every login is a Kerberos
password prompt.

### 1.2  On DAIC (first login)

```bash
ssh daic
git clone <repo-url> ~/research-tud
cd ~/research-tud
cp scripts/daic/env.example.sh scripts/daic/env.sh
$EDITOR scripts/daic/env.sh                   # same NetID + paths
source scripts/daic/env.sh

bash scripts/daic/setup_env.sh                # creates venv, installs torch+cu121
wandb login                                   # one-time; paste key from wandb.ai/authorize
install_keytab                                # enables Slurm Kerberos auto-renewal
```

`setup_env.sh` is idempotent — re-running it just verifies the venv and
the `staff-umbrella` artefact tree.

---

## Phase 2 — Push data and verify

### 2.1  Push data (Mac → DAIC), hybrid mode

The recommended bootstrap: push raw + processed data + checkpoints, but
**not** residuals or prior_params — those are cheap to rebuild on DAIC
and the rebuild exercises the pipeline end-to-end before any expensive
training.

```bash
# Locally, repo root, env.sh sourced
bash scripts/daic/sync_to_daic.sh --hybrid
```

Alternative flags:

| Flag | Effect |
|---|---|
| (no flag) | code + all data + checkpoints + residuals + priors |
| `--hybrid` | code + data + checkpoints, skip residuals + priors (recommended) |
| `--code-only` | only code (use this after every code edit) |
| `--data-only` | everything except code |
| `--no-residuals` / `--no-priors` | fine-grained omits |

### 2.2  Pre-flight verifier (login node)

```bash
ssh daic
cd ~/research-tud && source scripts/daic/env.sh
bash scripts/daic/verify_setup.sh
```

Checks env vars, modules, venv, project imports, data symlinks, data
files, wandb auth, and runs a subset of unit tests. No Slurm, no GPU.

### 2.3  GPU smoke test (5 min on an A40)

```bash
sbatch --export=DATASET=metrla scripts/daic/submit_smoke.sbatch
```

Confirms CUDA works, the SDE forward marginal runs against the real
prior YAML and residual npz, and wandb logs a run. Once this passes
you're cleared for the full pipeline.

---

## Phase 3 — Run the pipeline

### 3.1  Rebuild residuals and prior on DAIC

```bash
# Residual extraction (uses pushed checkpoints; ~10 min per dataset on A40)
sbatch --export=DATASET=metrla,SEED=0  scripts/daic/submit_extract_residuals.sbatch
sbatch --export=DATASET=pemsbay,SEED=0 scripts/daic/submit_extract_residuals.sbatch

# Prior fit (CPU only, ~2 min)
sbatch --export=DATASET=metrla  scripts/daic/submit_fit_prior.sbatch
sbatch --export=DATASET=pemsbay scripts/daic/submit_fit_prior.sbatch

# exp-004 SDE sanity checks (CPU only, ~10 min)
sbatch scripts/daic/submit_sanity_checks.sbatch
```

If you want to retrain the backbone from scratch on DAIC instead:

```bash
sbatch --export=DATASET=metrla,SEED=0  scripts/daic/submit_train_backbone.sbatch
sbatch --export=DATASET=pemsbay,SEED=0 scripts/daic/submit_train_backbone.sbatch
```

### 3.2  exp-003 score network

After PLAN §14.4 lands `scripts/train_diffusion.py`:

```bash
# Single run
sbatch --export=DATASET=metrla,SEED=0,PRIOR=structured \
       scripts/daic/submit_diffusion.sbatch

# Full 20-job sweep (5 seeds × 2 priors × 2 datasets)
for ds in metrla pemsbay; do
  for prior in structured isotropic; do
    for seed in 0 1 2 3 4; do
      sbatch --export=DATASET=$ds,SEED=$seed,PRIOR=$prior \
             scripts/daic/submit_diffusion.sbatch
    done
  done
done
```

---

## Phase 4 — Monitor and pull results

### 4.1  Live monitoring (Mac)

```bash
bash scripts/daic/watch_jobs.sh           # one-shot squeue + wandb link + newest log
bash scripts/daic/watch_jobs.sh --follow  # refreshes every 10 s
bash scripts/daic/watch_jobs.sh --tail    # tail -F the newest slurm log
bash scripts/daic/watch_jobs.sh --tail 12345  # tail a specific job
```

### 4.2  Wandb dashboard

All training runs log to <https://wandb.ai/tud-thesis/thesis-diffusion>.
`init_wandb` uses deterministic run IDs of the form
`{backbone}_{dataset}_seed{seed}_{cfg_hash}`, so re-submitting an sbatch
after a crash resumes the same run (same URL, same loss curves).

### 4.3  Direct Slurm commands (on DAIC)

```bash
squeue --me                                            # your queue
squeue --me -t RUNNING -o "%i %T %M %j %R"             # running with elapsed
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State   # post-mortem
tail -F logs/slurm-exp003-dsm-<jobid>.out              # live log
scancel <jobid>                                        # kill
sinteractive --cpus-per-task=2 --mem=8G --gres=gpu:a40:1 --time=01:00:00
```

### 4.4  Pull results back to Mac

```bash
bash scripts/daic/sync_from_daic.sh                       # everything (incl. ckpts)
bash scripts/daic/sync_from_daic.sh --no-checkpoints      # skip GB of .pt
bash scripts/daic/sync_from_daic.sh --no-residuals        # skip GB of .npz
```

The puller always grabs `experiments/**/*.{md,json,png,yaml}` so you get
SUMMARY.md updates and figures back regardless of flags.

---

## File reference

| File | Phase | Purpose |
|---|---|---|
| `env.example.sh` | 1 | Per-user config template. Copy → `env.sh` (gitignored). |
| `setup_env.sh` | 1 | One-time venv + CUDA torch install. |
| `verify_setup.sh` | 2 | Login-node pre-flight. No Slurm, no GPU. |
| `sync_to_daic.sh` | 2 | rsync Mac → DAIC with `--hybrid` flag. |
| `submit_smoke.sbatch` | 2 | 5-min A40 sanity (CUDA + SDE + wandb). |
| `smoke_test.py` | 2 | Python driver behind `submit_smoke.sbatch`. |
| `_common.sh` | 3 | Sourced by every sbatch. Modules + venv + symlinks + diagnostics. |
| `submit_train_backbone.sbatch` | 3 | GWN training. A40, medium, 12h. |
| `submit_extract_residuals.sbatch` | 3 | Inference + residual dump. A40, short, 1h. |
| `submit_fit_prior.sbatch` | 3 | Prior fit. CPU, short. |
| `submit_sanity_checks.sbatch` | 3 | exp-004 sanity. CPU, short. |
| `submit_diffusion.sbatch` | 3 | exp-003 score net. A40, medium, 24h. |
| `watch_jobs.sh` | 4 | Local live monitor. squeue + wandb + tail. |
| `sync_from_daic.sh` | 4 | rsync DAIC → Mac. |

---

## Hardware crib sheet

| GPU | Mem | Nodes | Use for | Slurm flag |
|---|---|---|---|---|
| A40 | 46 GB | 84 | default workhorse | `--gres=gpu:a40:1` |
| L40 | 49 GB | 18 | larger batches / faster | `--gres=gpu:l40:1` |
| V100 | 32 GB | 11 | older, FP64 ok | `--gres=gpu:v100:1` |
| RTX 2080 Ti | 11 GB | 24 | small models, plentiful | `--gres=gpu:turing:1` |

QoS levels on the `general` partition:

| QoS | Max time | Use for |
|---|---|---|
| `interactive` | 1 h | `sinteractive` only — do not `sbatch` with this |
| `short` | 4 h | residual extraction, prior fit, sanity checks, smoke |
| `medium` | 1.5 d | backbone training, single exp-003 run |
| `long` | 7 d | only if needed; lowest priority |

CPUs are always allocated in multiples of 2; project default is
`--cpus-per-task=4`. Adjust upward only if your dataloader is the bottleneck.

---

## Storage tiers

- `$HOME` — small quota, slow; OK for code + venv + logs.
- `staff-umbrella/<group>/<netid>/research-tud-data` — DAIC-recommended
  research-data location, larger quota, snapshotted. This is what
  `DAIC_DATA_DIR` points at; the `_common.sh` preamble symlinks
  `data/`, `results/checkpoints`, `results/residuals`, `results/prior_params`
  into the project tree automatically on each job.
- `/tmp` on compute nodes — local NVMe scratch, evaporates at job end.
  Useful for short-lived intermediates only.

Never `sshfs` against the login nodes — overloads the shared filesystem.
`rsync -avz --no-perms` (used by the helpers) is the supported path.

---

## Known DAIC quirks

- **No public-key SSH.** Kerberos password every session. Use `eduVPN`
  if you're off-campus.
- **CPUs in pairs.** Slurm rounds `--cpus-per-task` up to an even number.
- **Lmod uses `/opt/insy/modulefiles`.** `_common.sh` does `module use` for you.
- **Kerberos tickets expire after 10 h.** `install_keytab` (one-time)
  enables auto-renewal for unattended jobs.
- **Multi-partition submission.** `--partition=insy,general` takes
  whichever opens first; we default to `general` for portability.
- **wandb network egress** is allowed by default. If your group restricts
  it, flip `WANDB_MODE=offline` in `env.sh` and `wandb sync` from the
  login node after each run.
