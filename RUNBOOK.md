# Runbook

User-facing execution instructions. Each phase section is added when that phase is
scaffolded. Commands are copy-pasteable and have been verified to work on Mac
(Apple Silicon, MPS/CPU).

---

## Phase 0 — Environment setup

```bash
# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install the project in editable mode with dev extras
pip install -e ".[dev]"

# Verify tests pass
pytest tests/ -v
```

---

## Phase 1 — Data download and preprocessing

```bash
# Download raw data (adjacency matrices from DCRNN GitHub, h5 files from Zenodo)
# Also clones and pins third_party backbone repos
bash scripts/download_data.sh

# Preprocess METR-LA (saves train/val/test .npz + scaler.yaml)
python scripts/preprocess.py --dataset metrla

# Preprocess PEMS-BAY
python scripts/preprocess.py --dataset pemsbay

# Verify preprocessing sanity
python -c "
import numpy as np, yaml
raw = np.load('data/processed/metrla_train.npz')
scaler = yaml.safe_load(open('data/processed/metrla_scaler.yaml'))
print('METR-LA y_mean (raw mph):', raw['y'].mean().round(2))   # target: ~54
print('x_tod sample:', raw['x'][0, :5, 0, 1])                  # target: floats in [0,1)
print('scaler mean/std:', scaler['mean'], scaler['std'])        # target: ~54 / ~19
"
```

---

## Phase 2 — Run backbone training on Mac

### One-time setup

```bash
brew install tmux
# Either:
#   Install Amphetamine from the App Store and add "python" to allowed processes, OR:
sudo pmset -c disablesleep 1   # prevents sleep while on charger

# Login to wandb (paste API key from wandb.ai/settings)
wandb login
```

### Verify backbone wiring (run once before training)

```bash
python scripts/verify_backbone.py --backbone dcrnn --dataset metrla
python scripts/verify_backbone.py --backbone dcrnn --dataset pemsbay
python scripts/verify_backbone.py --backbone gwn   --dataset metrla
python scripts/verify_backbone.py --backbone gwn   --dataset pemsbay
```

All four should exit 0 with "PASS" printed.

### Start DCRNN on METR-LA

```bash
tmux new -s dcrnn_metrla
# Inside the session:
caffeinate -i python scripts/train_backbone.py --backbone dcrnn --dataset metrla --seed 0 --epochs 100
# Detach: Ctrl+b then d
# Expected wallclock: 3–6 h on M2/M3 MPS (DCRNN runs on CPU — see DECISIONS.md)
```

### Start DCRNN on PEMS-BAY

```bash
tmux new -s dcrnn_pemsbay
caffeinate -i python scripts/train_backbone.py --backbone dcrnn --dataset pemsbay --seed 0 --epochs 100
```

### Start GWN on METR-LA

```bash
tmux new -s gwn_metrla
caffeinate -i python scripts/train_backbone.py --backbone gwn --dataset metrla --seed 0 --epochs 100
# Expected wallclock: 1–2 h on M2/M3 MPS
```

### Start GWN on PEMS-BAY

```bash
tmux new -s gwn_pemsbay
caffeinate -i python scripts/train_backbone.py --backbone gwn --dataset pemsbay --seed 0 --epochs 100
```

### Monitor training

```bash
# List active tmux sessions
tmux ls

# Reattach to a session
tmux attach -t dcrnn_metrla

# wandb project (replace <user> with your wandb username)
# https://wandb.ai/<user>/thesis-diffusion
```

Key metric to watch: `val/mae_h12` — should reach:

| Backbone | Dataset | Target val_h12 (±5%) |
|----------|---------|----------------------|
| DCRNN    | METR-LA  | ≈ 3.10 |
| DCRNN    | PEMS-BAY | ≈ 1.59 |
| GWN      | METR-LA  | ≈ 3.53 |
| GWN      | PEMS-BAY | ≈ 1.95 |

### If a run is interrupted

Re-run the exact same command — resume is automatic from the last saved checkpoint.

To force a fresh start (deletes checkpoint and creates a new wandb run):

```bash
python scripts/train_backbone.py --backbone dcrnn --dataset metrla --seed 0 --fresh
```

### Extract residuals (run after all four training runs finish)

```bash
python scripts/extract_residuals.py --backbone dcrnn --dataset metrla --seed 0
python scripts/extract_residuals.py --backbone dcrnn --dataset pemsbay --seed 0
python scripts/extract_residuals.py --backbone gwn   --dataset metrla --seed 0
python scripts/extract_residuals.py --backbone gwn   --dataset pemsbay --seed 0
```

Output: `results/residuals/{backbone}_{dataset}_{split}_seed0.npz`

### Sanity-check residuals

Open `notebooks/02_residual_analysis.ipynb` and run the first three cells.

Confirm:
- Mean residual ≈ 0 globally (backbone is unbiased on average)
- Per-horizon std grows monotonically h1 → h12 (uncertainty increases with horizon)

**Report back when all four residual files are on disk.  Phase 3 starts from there.**

---

## tmux quick reference

```bash
# Detach from current session (leaves it running)
Ctrl+b then d

# List sessions
tmux ls

# Reattach
tmux attach -t <session-name>

# Kill a dead session
tmux kill-session -t <session-name>
```

---

## Phase 3 — Residual analysis

Open `notebooks/02_residual_analysis.ipynb` and run all cells.

Figures are saved to `results/figures/`. All four go/no-go checks should print PASS:
- Variance vs horizon (monotone growth)
- Σ_Q temporal structure (AR-type decay)
- Σ_S spectral concentration (low-frequency energy)
- Kronecker separability error (< 40%)

---

## Phase 4 — Prior fitting

Run after GWN residuals are extracted.  Each command takes < 2 min on M-series Mac.

```bash
# Fit prior on METR-LA (GWN backbone)
python scripts/fit_prior.py --backbone gwn --dataset metrla

# Fit prior on PEMS-BAY (GWN backbone)
python scripts/fit_prior.py --backbone gwn --dataset pemsbay
```

Expected output per run:
- Fitted parameters printed to stdout: δ_h, α, ρ₁, ρ₂, β, σ
- Covariance check: relative Frobenius error < 0.15 → `[PASS]`
- Tweedie check: MSE ratio ≈ 1.0 ± 10% for τ=0.5, 1.0, 2.0 → `[PASS]`
- `*** GO ***` printed at the end

Outputs:
```
results/prior_params/gwn_metrla.yaml
results/prior_params/gwn_pemsbay.yaml
results/figures/fig_prior_samples_covariance_check_gwn_metrla.png
results/figures/fig_prior_samples_covariance_check_gwn_pemsbay.png
```

If a check fails (`NO-GO`), the script exits with code 1.  Investigate the figure
to identify which component is misfitting.

### Optional: DCRNN backbone (after DCRNN training converges)

```bash
python scripts/fit_prior.py --backbone dcrnn --dataset metrla
python scripts/fit_prior.py --backbone dcrnn --dataset pemsbay
```

---

## wandb offline workflow (no wifi)

```bash
# Before starting, switch to offline mode
export WANDB_MODE=offline

# Run training as usual
python scripts/train_backbone.py --backbone gwn --dataset metrla --seed 0

# After reconnecting, sync the offline run
wandb sync ./wandb/offline-run-<timestamp>-<id>
```
