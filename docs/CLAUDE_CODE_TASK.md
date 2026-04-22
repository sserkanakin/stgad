# Claude Code Task Brief

## Context

You are implementing the first phases of a master's thesis project on structured residual diffusion for probabilistic spatiotemporal traffic forecasting. The full project specification is in `PROJECT_SPEC.md` in the same directory — read it fully before starting any work.

The thesis contribution is a Kronecker-structured (graph-Laplacian × horizon-dependent AR(1)) prior for the residual diffusion model. Everything else — backbone, data protocol, evaluation — follows community standards exactly. Use existing code wherever it exists. Write new code only for the contribution itself.

---

## What to do in this session

Execute Phases 0, 1, and 2 of the plan in `PROJECT_SPEC.md`.

---

### Phase 0 — Scaffolding (½ day)

Create the full repository structure from §9 of the spec. The layout is:

```
thesis_diffusion/
├── README.md
├── pyproject.toml
├── .gitignore
├── DECISIONS.md
├── configs/
│   ├── data/
│   │   ├── metrla.yaml
│   │   └── pemsbay.yaml
│   ├── backbone/
│   │   └── stid.yaml
│   └── experiment/
│       ├── smoke_mac.yaml
│       ├── main_structured.yaml
│       ├── ablation_iso.yaml
│       ├── ablation_spatial_only.yaml
│       ├── ablation_temporal_only.yaml
│       └── ablation_const_rho.yaml
├── data/
│   ├── raw/                     # gitignored
│   └── processed/               # gitignored
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── dataset.py
│   │   └── adjacency.py
│   ├── backbones/
│   │   └── __init__.py          # thin wrapper only — see Phase 2
│   ├── diffusion/               # empty stubs for now
│   │   ├── __init__.py
│   │   ├── sde.py
│   │   ├── prior_fit.py
│   │   ├── denoiser.py
│   │   ├── precond.py
│   │   └── sampler.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics_det.py
│   │   ├── metrics_prob.py
│   │   └── stratified.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── backbone_train.py
│   │   └── diffusion_train.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── logging.py
│       └── config.py
├── scripts/
│   ├── download_data.sh
│   ├── preprocess.py
│   ├── extract_residuals.py
│   ├── fit_prior.py
│   ├── train_diffusion.py
│   ├── evaluate.py
│   └── submit_delft.sh
├── notebooks/
│   ├── 01_data_eda.ipynb
│   ├── 02_residual_analysis.ipynb
│   └── 03_results_analysis.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_metrics.py
│   └── test_sde.py
└── results/
    ├── checkpoints/             # gitignored
    ├── residuals/               # gitignored
    ├── figures/                 # not gitignored
    └── prior_params/            # yaml files, not gitignored
```

**`pyproject.toml`**:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "thesis-diffusion"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "numpy",
    "scipy",
    "pandas",
    "h5py",
    "pyyaml",
    "wandb",
    "properscoring",
    "tqdm",
    "matplotlib",
    "einops",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "jupyter",
    "ipykernel",
    "black",
    "ruff",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

**`.gitignore`** must include:
```
data/raw/
data/processed/
results/checkpoints/
results/residuals/
third_party/
__pycache__/
*.pyc
.venv/
wandb/
*.pt
*.h5
*.pkl
*.npz
```

**`src/utils/seed.py`**: write `set_seed(seed: int)` that seeds `random`, `numpy`, and `torch` (handles both CPU and GPU/MPS). Called at the top of every training script.

**`src/utils/config.py`**: write `load_config(path: str) -> dict` that reads a YAML file and returns a dict.

**`DECISIONS.md`**: create with a header and one example entry template. Append any non-obvious design decision here as work progresses.

**Deliverable**: `pip install -e ".[dev]"` succeeds. `pytest tests/` runs and passes (trivially — empty test files at this stage). Repo structure matches exactly.

---

### Phase 1 — Data download and preprocessing

#### 1a. Download script

Write `scripts/download_data.sh`. It downloads:

- METR-LA traffic data: `https://zenodo.org/records/5146275/files/METR-LA.h5` → `data/raw/METR-LA.h5`
- PEMS-BAY traffic data: `https://zenodo.org/records/5146275/files/PEMS-BAY.h5` → `data/raw/PEMS-BAY.h5`
- METR-LA adjacency: `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl` → `data/raw/adj_mx_metrla.pkl`
- PEMS-BAY adjacency: `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx_bay.pkl` → `data/raw/adj_mx_pemsbay.pkl`

Use `wget -c` (resume-able) with a check at the top: skip download if file already exists.

#### 1b. Preprocessing script

Write `scripts/preprocess.py --dataset {metrla|pemsbay}`.

Applies the **fixed protocol from §3 of the spec** — do not deviate:

| Step | Detail |
|---|---|
| Split | 70% train / 10% val / 20% test, chronological. `n_train = int(0.7 * T)`, `n_val = int(0.8 * T)`. |
| Windows | Sliding window T_in=12 input + T_out=12 output steps, stride 1. No shuffling. |
| Normalization | Global z-score (one mean, one std across all sensors and training timestamps). Compute on train only. Apply to all splits. |
| Missing mask | `mask = (raw_value != 0.0)` computed on raw values before normalization. |
| Time features | `time_of_day` = step index within day (0–287). `day_of_week` = day index (0–6). One per timestep. |

Output files saved to `data/processed/{dataset}_{split}.npz` with arrays:

```
x        float32  (N_windows, 12, N_sensors, 1)   # normalized input
y        float32  (N_windows, 12, N_sensors, 1)   # normalized target
x_mask   bool     (N_windows, 12, N_sensors, 1)   # False = missing
y_mask   bool     (N_windows, 12, N_sensors, 1)
x_tod    int32    (N_windows, 12)                  # time-of-day index
x_dow    int32    (N_windows, 12)                  # day-of-week index
y_tod    int32    (N_windows, 12)
y_dow    int32    (N_windows, 12)
```

Scaler saved to `data/processed/{dataset}_scaler.yaml`:
```yaml
mean: <float>
std: <float>
```

Expected window counts (approximate, for sanity check printed by the script):
- METR-LA: ~24,000 train / ~3,400 val / ~6,850 test
- PEMS-BAY: ~36,500 train / ~5,200 val / ~10,500 test

#### 1c. Source modules

Write the core logic in `src/data/`:

**`preprocessing.py`**: functions `split_data`, `make_windows`, `compute_scaler`, `normalize`, `compute_mask`, `compute_time_features`. Each is a pure function. No side effects.

**`dataset.py`**: `TrafficDataset(torch.utils.data.Dataset)` that wraps an `.npz` file. Returns a dict with keys `x, y, x_mask, y_mask, x_tod, x_dow, y_tod, y_dow` as tensors.

**`adjacency.py`**: 
- `load_adjacency(dataset: str) -> np.ndarray`: loads the pickle, returns N×N adjacency.
- `compute_normalized_laplacian(adj: np.ndarray) -> np.ndarray`: returns D^{-1/2} (D - A) D^{-1/2}, the symmetric normalized Laplacian. This is the matrix used in ChebConv and in the prior.
- `laplacian_eigendecomposition(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: returns (eigenvalues, eigenvectors). Cached to disk at `data/processed/{dataset}_laplacian_eigs.npz`.

#### 1d. Tests

Write `tests/test_preprocessing.py`:
- Test split sizes sum to total windows.
- Test that scaler is computed on train only (mean/std of train normalizes to ~0 mean, ~1 std; val and test will not).
- Test mask is False where raw value is 0.
- Test window time ordering — last timestamp of window i < first timestamp of window i+1.
- Test adjacency loads to correct shape: (207, 207) for METR-LA, (325, 325) for PEMS-BAY.
- Test normalized Laplacian is symmetric and has correct spectral properties (eigenvalues in [0, 2]).

#### 1e. EDA notebook

Write `notebooks/01_data_eda.ipynb`. All cells run top-to-bottom. Figures saved to `results/figures/eda/`.

**Cell group 1 — Load data**: load `data/processed/metrla_train.npz`, load adjacency, load Laplacian eigenvectors.

**Figure 1** `fig_weekly_cycles.png`: 7 days of speed for sensors 0, 40, 80, 120, 160, 200. Raw values (de-normalize with scaler). Show missing gaps as NaN gaps in the plot line.

**Figure 2** `fig_missingness_heatmap.png`: per-sensor missing rate (fraction of training timestamps where mask=False). Bar chart sorted by rate. Should show most sensors near 0%, a few above 50%.

**Figure 3** `fig_acf_raw_vs_demeaned.png`: compute ACF out to lag 72 (6 hours), averaged over sensors:
  - Raw speed ACF.
  - De-meaned speed ACF (subtract per-sensor per-time-of-day mean computed on train set).
  Plot both on same axes with different colors and legend.

**Figure 4** `fig_dirichlet_energy_distribution.png`: for each training timestamp compute `x_t^T L x_t / (x_t^T x_t + 1e-8)`. Histogram over all timestamps. Print and record the mean value — this is what gets reported in the deliverables.

**Output file** `data/processed/metrla_regimes_test.npz`: compute mean speed of input window for each test window. Assign to `free_flow` (top tercile), `transition` (middle), `congested` (bottom). Save as string array `regime` shape `(N_test_windows,)`.

---

### Phase 2 — STID backbone (official code)

**Do not reimplement STID.** Use the official code from `github.com/GestaltCogTeam/STID`.

#### 2a. Set up external code

Add to `scripts/download_data.sh` (or a separate `scripts/setup_third_party.sh`):

```bash
mkdir -p third_party
git clone https://github.com/GestaltCogTeam/STID.git third_party/STID
cd third_party/STID
git checkout <pin to latest commit on main — record the hash>
cd ../..
```

Record the exact commit hash in `DECISIONS.md` under the heading "External code pins".

`third_party/` is gitignored — it does not go into our repo.

#### 2b. Thin wrapper

Write `src/backbones/__init__.py` with a function:

```python
def load_stid(dataset: str, device: torch.device) -> torch.nn.Module:
    """
    Load the STID model class from third_party/STID and instantiate
    it with the correct config for the given dataset.
    Returns the model on the specified device.
    """
```

This function adds `third_party/STID` to `sys.path`, imports the STID arch module, instantiates with the correct hyperparameters for the dataset (see below), and returns the model. All downstream code uses this wrapper — nothing else should touch `third_party/` directly.

STID config for METR-LA:
```python
{
    "num_nodes": 207,
    "input_len": 12,
    "output_len": 12,
    "input_dim": 1,
    "embed_dim": 32,
    "output_dim": 1,
    "num_layer": 3,
    "if_spatial": True,
    "if_time_in_day": True,
    "if_day_in_week": True,
    "time_of_day_size": 288,
    "day_of_week_size": 7,
}
```

STID config for PEMS-BAY: same but `num_nodes: 325`.

#### 2c. Verification script

Write `scripts/verify_stid.py --dataset metrla`. It:
1. Calls `load_stid("metrla", device)`.
2. Runs one forward pass on a dummy batch: `x` shape `(2, 12, 207, 1)`, `x_tod` shape `(2, 12)`, `x_dow` shape `(2, 12)`.
3. Verifies output shape is `(2, 12, 207, 1)`.
4. Prints total parameter count.
5. Exits 0. Crashes with a clear error message if anything fails.

#### 2d. Training script

Write `scripts/train_backbone.py --dataset {metrla|pemsbay} --seed 0 [--epochs N]`.

The script:
1. Calls `set_seed(seed)`.
2. Loads data via `TrafficDataset` and `DataLoader`.
3. Loads STID via `load_stid`.
4. Trains with Adam (lr=1e-3), masked MAE loss, for `epochs` steps (default 100, `--epochs 1` for smoke test).
5. Validates every epoch, saves best checkpoint by val MAE (averaged over all 12 horizons) to `results/checkpoints/stid_{dataset}_seed{seed}.pt`.
6. Logs to wandb: per-epoch train loss, val MAE per horizon, total val MAE.
7. After training: evaluates on test set, reports MAE per horizon 3/6/12. Prints a WARNING if any horizon deviates more than 5% from the targets in §5 of the spec.

The masked MAE loss:
```python
def masked_mae_loss(pred, true, mask):
    # pred, true, mask: (batch, T_out, N, 1)
    loss = torch.abs(pred - true) * mask.float()
    return loss.sum() / (mask.float().sum() + 1e-8)
```

For the smoke test (`--epochs 1`): must finish in under 10 minutes on MacBook CPU/MPS. If it's going to take longer, reduce batch_size to 16 for the smoke test only (add a `--smoke` flag that sets batch_size=16 and skips wandb logging).

#### 2e. Residual extraction script

Write `scripts/extract_residuals.py --dataset {metrla|pemsbay} --seed 0`.

1. Loads the best checkpoint from `results/checkpoints/stid_{dataset}_seed{seed}.pt`.
2. Runs inference on train, val, test sets with `torch.no_grad()`.
3. Computes `residuals = y_true - y_pred` in normalized space.
4. Saves to `results/residuals/{dataset}_{split}_seed{seed}.npz`:
   ```
   residuals   float32  (N_windows, 12, N_sensors, 1)
   y_pred      float32  (N_windows, 12, N_sensors, 1)
   y_true      float32  (N_windows, 12, N_sensors, 1)
   mask        bool     (N_windows, 12, N_sensors, 1)
   ```
5. Prints summary stats: mean residual (should be near 0), std per horizon (should grow with horizon), fraction missing.

---

## What NOT to do in this session

- Do not write any diffusion model code. The stub files in `src/diffusion/` stay empty (just `pass` in each function placeholder).
- Do not implement baselines (DiffSTG, CSDI, etc.).
- Do not modify STID's source code in any way.
- Do not train for more than 1 epoch on Mac unless explicitly testing with `--epochs 1`.
- Do not commit data files, checkpoints, residuals, or `third_party/` to git.

---

## Key constraints

**Device handling** — never use `.cuda()`. Always:
```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model = model.to(device)
```

**Protocol fidelity** — the data protocol in §3 of the spec is fixed. Any ambiguity resolves in favour of DCRNN/STID convention. Record the decision in `DECISIONS.md`.

**Code quality** — every function has a docstring. Every module has a one-line header. Tests are written alongside modules.

**Git discipline** — commit format `phase<N>: <description>`. Commit after each logical unit, not after each file.

**wandb** — wrap all `wandb.init` calls to fall back to `mode="offline"` if `WANDB_API_KEY` is not set, so the script never crashes on login failure. The user runs `wandb login` manually.

---

## Deliverables at end of session

Report back with all of the following confirmed or pasted:

1. `pip install -e ".[dev]"` succeeds.
2. `pytest tests/` output — must pass.
3. Data files present at `data/raw/` after running `download_data.sh`.
4. `python scripts/preprocess.py --dataset metrla` output — paste the reported window counts and scaler mean/std.
5. `python scripts/verify_stid.py --dataset metrla` output — paste parameter count.
6. `python scripts/train_backbone.py --dataset metrla --seed 0 --epochs 1 --smoke` output — paste the epoch 1 train loss and val MAE. Must not crash.
7. Dirichlet energy histogram mean value from the EDA notebook.
8. Contents of `DECISIONS.md`.
9. Any blockers or deviations from spec.

---

## Proposed next session (Phase 3)

After this session, Phase 3 is the residual analysis notebook `notebooks/02_residual_analysis.ipynb`, which loads `results/residuals/metrla_train_seed0.npz` and produces:
- `fig_variance_vs_horizon.png` — empirical Var(R_h) per horizon. Justifies horizon-dependent prior.
- `fig_sigma_Q.png` — H×H temporal residual covariance. Justifies AR(1) structure.
- `fig_sigma_R_spectral.png` — spatial residual covariance in Laplacian eigenbasis. Justifies Laplacian prior.
- Separability Frobenius error scalar. Justifies (or challenges) Kronecker assumption.
- `fig_variance_by_regime.png` — variance by traffic regime. Motivates future work on non-stationary priors.

These figures drive all prior parameter choices in Phase 4. Phase 3 only begins after Phase 2 residuals are extracted and look sensible.
