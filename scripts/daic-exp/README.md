# Running on experimental DAIC

End-to-end runbook for someone with an account on the **experimental** TU Delft AI cluster (`daic01.hpc.tudelft.nl` / `daic02.hpc.tudelft.nl`, docs at <https://doc.daic.tudelft.nl/>).

For the original DAIC (`login.daic.tudelft.nl`) use [`scripts/daic/README.md`](../daic/README.md) — the cluster is different enough that the two sets of scripts can't share boilerplate.

```
   Phase 1  Setup     ── env.sh, setup_env.sh, kinit                  [~15 min]
   Phase 2  Verify    ── sync_to_daic_exp.sh, verify, smoke job       [~15 min]
   Phase 3  Run       ── extract residuals + fit prior + score net    [~hours]
   Phase 4  Monitor   ── wandb dashboard, watch_jobs.sh                [live]
```

The owner of the project monitors live via wandb (<https://wandb.ai/.../thesis-diffusion>) and pulls raw artefacts back with `sync_from_daic_exp.sh` when the runs finish.

---

## Phase 1 — One-time setup

### 1.1 Your local machine

```bash
cp scripts/daic-exp/env.example.sh scripts/daic-exp/env.sh
$EDITOR scripts/daic-exp/env.sh    # set DAIC_EXP_NETID + DAIC_EXP_ACCOUNT
source scripts/daic-exp/env.sh
```

Optional — add a local `~/.ssh/config` shortcut so the helpers work without
typing the long hostname:

```
Host daic-exp
  HostName daic01.hpc.tudelft.nl
  User <your-netid>
```

Off-campus you need TU Delft network or **eduVPN** to reach the cluster.

### 1.2 First login on experimental DAIC

```bash
ssh <your-netid>@daic01.hpc.tudelft.nl
# enter TU Delft password
# the system runs a one-time first-login setup, creating symlinks to
# your linuxhome etc. Verify:
hostname    # → daic01.hpc.tudelft.nl
pwd         # → /home/<netid>
```

Then set up passwordless SSH from your laptop:

```bash
# from your laptop
ssh-copy-id <your-netid>@daic01.hpc.tudelft.nl
```

Important: **SSH-key logins do not create a Kerberos ticket**, so the first
time you connect each day (or after a 10-hour ticket expiry) you must
`kinit` to mount network storage (`~/linuxhome`, staff-umbrella):

```bash
ssh daic-exp
kinit         # enter TU Delft password once per day
klist         # confirm a ticket is present
```

If `~/linuxhome` shows "Stale file handle" right after login, wait ~30s for
the mount to settle.

### 1.3 Identify your Slurm account

Every sbatch on experimental DAIC requires `--account=<your-account>`. Find
yours with:

```bash
sacctmgr show user $USER withassoc format=account%30
```

If you see multiple, pick the one for the group/project that should be
charged for this work. Put it into `scripts/daic-exp/env.sh` as
`DAIC_EXP_ACCOUNT`. (Once env.sh is sourced, `SBATCH_ACCOUNT` is
auto-exported, so subsequent `sbatch` calls don't need `-A`.)

### 1.4 Clone + bootstrap

```bash
cd ~/linuxhome    # /trinity/home is 5 MB; do NOT clone there
git clone <repo-url> research-tud
cd research-tud

cp scripts/daic-exp/env.example.sh scripts/daic-exp/env.sh
$EDITOR scripts/daic-exp/env.sh    # same as locally — NetID, account, paths
source scripts/daic-exp/env.sh

bash scripts/daic-exp/setup_env.sh   # installs Miniforge + creates conda env
wandb login                          # paste key from https://wandb.ai/authorize
```

`setup_env.sh` is idempotent: re-run to repair or update. It installs
Miniforge under `${HOME}/linuxhome/miniforge3` (experimental DAIC has no
miniconda module) and creates `thesis-diffusion` with Python 3.11 +
PyTorch cu121 wheels.

---

## Phase 2 — Push data and verify

### 2.1 Push data (your laptop → cluster)

```bash
# From your laptop, repo root, with env.sh sourced
bash scripts/daic-exp/sync_to_daic_exp.sh           # raw + processed + checkpoints
```

Flags:

| Flag | Effect |
|---|---|
| (no flag) | raw + processed + checkpoints (residuals + priors skipped — cheap to rebuild) |
| `--all` | also push residuals + priors |
| `--raw` / `--processed` / `--checkpoints` / `--residuals` / `--priors` | only that target |

### 2.2 Pre-flight check (login node)

```bash
ssh daic-exp
cd ~/linuxhome/research-tud && source scripts/daic-exp/env.sh
bash scripts/daic-exp/verify_setup.sh
```

Checks env vars, your Slurm account association, module loads, conda env,
project imports, data symlinks, data files, wandb auth, and runs a subset
of unit tests. No Slurm, no GPU.

### 2.3 GPU smoke test (~5 min on a nvidia_a40)

```bash
sbatch --export=DATASET=metrla scripts/daic-exp/submit_smoke.sbatch
squeue --me                                # watch it run
tail -F logs/smoke/daic-exp-smoke-*.out    # live log
```

Confirms CUDA works, the SDE forward marginal evaluates against the real
prior YAML + residual npz, and wandb logs a run. Once this passes you're
cleared for the full pipeline.

---

## Phase 3 — Run the pipeline

### 3.1 Rebuild residuals and prior on the cluster

```bash
# Residual extraction (uses pushed checkpoints; ~10 min on nvidia_a40)
sbatch --export=DATASET=metrla,SEED=0  scripts/daic-exp/submit_extract_residuals.sbatch
sbatch --export=DATASET=pemsbay,SEED=0 scripts/daic-exp/submit_extract_residuals.sbatch

# Prior fit (CPU only, ~2 min)
sbatch --export=DATASET=metrla  scripts/daic-exp/submit_fit_prior.sbatch
sbatch --export=DATASET=pemsbay scripts/daic-exp/submit_fit_prior.sbatch

# SDE sanity checks (CPU only, ~10 min)
sbatch scripts/daic-exp/submit_sanity_checks.sbatch
```

Or if you want to retrain the backbone from scratch:

```bash
sbatch --export=DATASET=metrla,SEED=0  scripts/daic-exp/submit_train_backbone.sbatch
sbatch --export=DATASET=pemsbay,SEED=0 scripts/daic-exp/submit_train_backbone.sbatch
```

### 3.2 Score network

```bash
# Single run
sbatch --job-name="dsm-metrla-structured-s0" \
       --export=DATASET=metrla,SEED=0,PRIOR=structured \
       scripts/daic-exp/submit_diffusion.sbatch

# Full 20-job sweep (5 seeds × 2 priors × 2 datasets)
for ds in metrla pemsbay; do
  for prior in structured isotropic; do
    for seed in 0 1 2 3 4; do
      sbatch --job-name="dsm-${ds}-${prior}-s${seed}" \
             --export=DATASET=$ds,SEED=$seed,PRIOR=$prior \
             scripts/daic-exp/submit_diffusion.sbatch
    done
  done
done
```

---

## Phase 4 — Monitor and pull results

### 4.1 Wandb (the project owner monitors here)

All training runs log to <https://wandb.ai/.../thesis-diffusion>.
`init_wandb` derives deterministic run IDs from `{backbone}_{dataset}_seed{seed}_{cfg_hash}`,
so resubmitting after a crash resumes the same run (same URL, same curves).

### 4.2 Live monitoring (the friend, from cluster)

```bash
squeue --me                                          # your queue
tail -F logs/train-diffusion/*-<jobid>.out           # live log
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State # post-mortem
scancel <jobid>                                      # kill
```

Interactive GPU session for debugging:

```bash
salloc --partition=all --gres=gpu:nvidia_a40:1 --cpus-per-task=2 --mem=8G --time=01:00:00
```

### 4.3 Pulling raw artefacts back (project owner, from laptop)

```bash
source scripts/daic-exp/env.sh
bash scripts/daic-exp/sync_from_daic_exp.sh              # everything
bash scripts/daic-exp/sync_from_daic_exp.sh --no-checkpoints
bash scripts/daic-exp/sync_from_daic_exp.sh --no-residuals
```

The puller always grabs `experiments/**/*.{md,json,png,yaml}` so figures
and SUMMARY.md updates come back regardless of flags.

You can also live-watch jobs without SSH'ing manually:

```bash
bash scripts/daic-exp/watch_jobs.sh           # one-shot status
bash scripts/daic-exp/watch_jobs.sh --follow  # squeue every 10s
bash scripts/daic-exp/watch_jobs.sh --tail    # newest log
```

---

## File reference

| File | Phase | Purpose |
|---|---|---|
| `env.example.sh` | 1 | Per-user config template. Copy → `env.sh` (gitignored). |
| `setup_env.sh` | 1 | One-time Miniforge install + conda env + project install. |
| `verify_setup.sh` | 2 | Login-node pre-flight. No Slurm, no GPU. |
| `sync_to_daic_exp.sh` | 2 | rsync laptop → cluster. |
| `submit_smoke.sbatch` | 2 | 5-min GPU sanity. |
| `_common.sh` | 3 | Sourced by every sbatch. Modules + conda + symlinks + diagnostics. |
| `submit_train_backbone.sbatch` | 3 | GWN training. |
| `submit_extract_residuals.sbatch` | 3 | Inference + residual dump. |
| `submit_fit_prior.sbatch` | 3 | Prior fit. CPU. |
| `submit_sanity_checks.sbatch` | 3 | SDE sanity. CPU. |
| `submit_diffusion.sbatch` | 3 | Score net. |
| `submit_eval.sbatch` | 3 | CRPS / PIT / pinball. |
| `watch_jobs.sh` | 4 | Local live monitor. |
| `sync_from_daic_exp.sh` | 4 | rsync cluster → laptop. |

---

## Hardware crib sheet (experimental DAIC)

| GPU | Mem | Nodes × GPUs | gres flag |
|---|---|---|---|
| NVIDIA A40 | 48 GB | 4 × 3 | `--gres=gpu:nvidia_a40:1` |
| NVIDIA L40 | 48 GB | 2 × 3 | `--gres=gpu:nvidia_l40:1` |
| NVIDIA RTX Pro 6000 | 96 GB | 10 × 2 | `--gres=gpu:nvidia_rtx_pro_6000:1` |

Partitions:

| Partition | Use for |
|---|---|
| `all` | open default — all GPU nodes |
| `test` | CPU-only quick tests |
| `ewi-insy`, `ewi-me`, `ewi-st`, `me-cor` | dept partitions (higher priority if your account has access) |

If you have access to a dept partition, prepend it in `env.sh`:
`DAIC_EXP_PARTITION="ewi-insy,all"`.

---

## Storage tiers

| Tier | Path | Quota | Use for |
|---|---|---|---|
| Cluster home | `/trinity/home/<netid>` | **5 MB** | shell config only |
| Linux home | `~/linuxhome` | ~30 GB | code, conda env, small artefacts |
| Project | `/tudelft.net/staff-umbrella/<project>` | by request | research data — recommended |
| Bulk / Group (legacy) | `/tudelft.net/staff-{bulk,groups}/...` | fair use | deprecated |
| Local scratch | `/tmp/<netid>` | host-local | job-time intermediates only |

`DAIC_EXP_DATA_DIR` in `env.sh` defaults to `~/linuxhome/research-tud-data`.
If your group grants you staff-umbrella project space, point it there.

---

## Known experimental-DAIC quirks vs the regular DAIC

- **Hostname**: `daic01.hpc.tudelft.nl` (not `login.daic.tudelft.nl`).
- **SSH keys allowed**: `ssh-copy-id` works (regular DAIC required Kerberos every session).
- **Kerberos still needed** for storage: run `kinit` once per 10h or after key-only logins.
- **No miniconda module** — we install Miniforge into `~/linuxhome/miniforge3`.
- **Module tree changed** — `module use /opt/insy/modulefiles` is gone; start with `module load 2025/gpu`.
- **`--account=` is mandatory**. Auto-exported as `SBATCH_ACCOUNT` from env.sh.
- **GPU gres prefix is `nvidia_*`**, not bare `a40`.
- **`/trinity/home` is 5 MB** — clone the repo to `~/linuxhome` instead.
- **No `/winhome`**.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `sbatch: error: Job submit/allocate failed: Invalid account` | `SBATCH_ACCOUNT` not exported | `source scripts/daic-exp/env.sh` |
| `Permission denied` on `~/linuxhome` | no Kerberos ticket | `kinit` |
| `Stale file handle` right after login | NFS mount catching up | wait 30s, retry |
| `module: command not found` | Lmod not initialised on this shell | log out and back in |
| `setup_env.sh` hangs at Miniforge download | network egress blocked | use offline installer; copy from another machine |
| wandb run rejects auth | API key missing | `wandb login` |
| `module load 2025/gpu` not found | wrong base env name | `module avail 2025` to list variants |
