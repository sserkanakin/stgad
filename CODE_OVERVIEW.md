# Code overview — module-by-module tour

Read this alongside [`THESIS_OVERVIEW.md`](THESIS_OVERVIEW.md) (math + rationale) and [`DIVERGENCES.md`](DIVERGENCES.md) (where we deviate from standard practice).

---

## Repository layout

```
src/                        installable Python package (thesis-diffusion)
  data/                     loading, splits, adjacency, Laplacian eigendecomp
  backbones/                Graph WaveNet wrapper around third_party clone
  diffusion/                Kronecker prior + SDE + DSM loss + score net + sampler
  eval/                     CRPS / PIT / pinball probabilistic metrics
  utils/                    config, seed, checkpoints, wandb init

scripts/                    runnable entry points
  daic/                     DAIC-specific sbatch + env helpers

tests/                      pytest suite (162 tests, all green)
configs/
  backbone/gwn.yaml         GWN hyperparameters
  data/                     dataset-specific paths and node counts
experiments/                per-phase SUMMARY.md + figures + JSON results
results/
  checkpoints/              backbone + score-net .pt files
  residuals/                gwn_{dataset}_{split}_seed{S}.npz
  prior_params/             gwn_{dataset}.yaml — fitted prior parameters
data/
  raw/                      h5 + adj pkl downloads
  processed/                {dataset}_{split}.npz, scaler.yaml, laplacian_eigs.npz
third_party/                Graph-WaveNet @ 6b162e8 (pinned, unmodified)
notebooks/                  EDA / residual / results Jupyter notebooks
a-thesis-vault/             gitignored thesis-writing scratchpad
```

---

## End-to-end pipeline (data → forecast)

```
data/raw/{dataset}.h5 + adj_mx_{dataset}.pkl
         │
         │ scripts/preprocess.py
         ▼
data/processed/{ds}_{split}.npz  +  {ds}_scaler.yaml  +  {ds}_laplacian_eigs.npz
         │
         │ scripts/train_backbone.py
         ▼
results/checkpoints/gwn_{ds}_seed0_<hash>/best.pt
         │
         │ scripts/extract_residuals.py
         ▼
results/residuals/gwn_{ds}_{split}_seed0.npz
         │
         │ scripts/fit_prior.py
         ▼
results/prior_params/gwn_{ds}.yaml      ← Q_H + Σ_S parameters
         │
         │ scripts/train_diffusion.py
         ▼
results/checkpoints/dsm_{prior}_{ds}_seed0_<hash>/best.pt
         │
         │ scripts/eval_diffusion.py
         ▼
experiments/score-network/eval_<run_id>.json
                                                       ← per-horizon CRPS/PIT/pinball
```

---

## `src/data/` — dataset and graph utilities

| File | Responsibility |
|---|---|
| `dataset.py::TrafficDataset` | Wraps a preprocessed `.npz`. Returns `x` (normalised speed + time-of-day) of shape `(12, N, 2)`, `y` (raw mph), and masks. Speed channel z-scored using training-set scaler. |
| `adjacency.py::load_adjacency` | Loads the pickled adjacency matrix (handles Python-2 pickle quirk). |
| `adjacency.py::compute_normalized_laplacian` | Symmetric normalised Laplacian `L = I − D⁻¹ᐟ²ÂD⁻¹ᐟ²`, with self-loops removed and asymmetric raw matrix symmetrised first. |
| `adjacency.py::laplacian_eigendecomposition` | Caches full eigendecomp of L to `data/processed/{dataset}_laplacian_eigs.npz`. |
| `preprocessing.py` | Helpers shared by `scripts/preprocess.py` (windowing, train/val/test split, scaler computation). |

---

## `src/backbones/` — deterministic forecaster

| File | Responsibility |
|---|---|
| `__init__.py::load_backbone` | Thin wrapper around the pinned Graph WaveNet repo at `third_party/Graph-WaveNet`. Returns a `_GWNWrapper` that adapts shape conventions: `(B, T, N, C) → GWN → (B, T, N, 1)`. Patches GWN's `Conv1d` → `Conv2d` for PyTorch ≥ 2.x compatibility. DCRNN was deprecated in the GWN-only refactor. |

---

## `src/diffusion/` — the meat

| File | Responsibility |
|---|---|
| `prior_fit.py` | Fit `Q_H` (two-exponential temporal kernel + per-horizon `δ_h`) and `Σ_S` (graph-Laplacian GMRF with `β, ε_s`). Both fit by Nelder-Mead / scipy.optimize on the standardised residuals. Produces the canonical `results/prior_params/gwn_{ds}.yaml`. |
| `sde.py` | NumPy / float64 / CPU forward + reverse SDE. β schedule, eigendecompose_prior, `to_spectral`/`from_spectral`, apply_Q, forward_sample, sample_stationary, oracle_score, reverse_oracle. Used by `experiments/sde-sanity/run_sanity_checks.py`. |
| `sde_torch.py` | PyTorch adapter. Same API as `sde.py` but tensor-native and differentiable. `eig_to_torch` bridges NumPy → torch. forward_sample handles per-batch τ via broadcasting. Used by training + inference. |
| `score_net.py` | `ScoreNet` — GWN-style score network. Components: `SinusoidalEmbedding` (DDPM-style τ encoding), `FiLM` (per-channel affine modulation, identity init), `GraphConv` (single-support K=1), `ScoreNetBlock` (gated TCN + GCN + FiLM + GroupNorm + residual), `ScoreNet` (4 blocks, ~232k params at default config). |
| `dsm.py` | `sample_tau` (log-uniform), `dsm_x0_loss` (eigenbasis-weighted MSE or plain uniform MSE), `per_mode_squared_error` for diagnostics. |
| `sampling.py` | `reverse_learned` — Euler-Maruyama reverse SDE with the learned score net. `sample_per_window` — fuses K trajectories per window into a single batched reverse pass. |

---

## `src/eval/` — probabilistic metrics

| File | Responsibility |
|---|---|
| `probabilistic.py` | `crps_ensemble` (mean over absolute errors minus half-pairwise-mean), `pit_values` (empirical CDF), `pinball_loss` at any quantile. Self-contained (no scipy/properscoring dependency). |

---

## `src/utils/` — plumbing

| File | Responsibility |
|---|---|
| `config.py` | `load_config` (YAML) and `hash_config` (stable 8-char hash of a dict). |
| `seed.py` | `set_seed(s)` — seeds Python, NumPy, torch (incl. CUDA if available). |
| `checkpoint.py` | Save/load checkpoints with optimizer state + epoch + config. |
| `wandb_setup.py` | `init_wandb` — handles `.env` loading, ~/.netrc credentials, online/offline fallback, deterministic run IDs for resumability. |

---

## `scripts/` — runnable entry points

| Script | What it does | Time on A40 |
|---|---|---|
| `download_data.sh` | Pulls METR-LA + PEMS-BAY .h5 files and GWN + DCRNN repos to `third_party/`. | ~1 min |
| `preprocess.py` | Raw .h5 → windowed train/val/test .npz + scaler + Laplacian eigendecomp cache. | ~30 s |
| `train_backbone.py` | Trains Graph WaveNet on one (dataset, seed). Saves best.pt by val MAE. wandb-logged. | ~30 min |
| `extract_residuals.py` | Loads best.pt, runs inference, saves (`residuals`, `y_pred`, `y_true`, `mask`) per split. | ~1 min |
| `fit_prior.py` | Fits Q_H (two-exp temporal) + Σ_S (graph-Laplacian GMRF) on training residuals. Produces standardised YAML with `target_scale`. | ~2 min (CPU) |
| `train_diffusion.py` | Trains the score network. Composes DiffusionDataset + eig + adjacency + ScoreNet + DSM loss + wandb. Supports `--prior {structured,isotropic}`, `--weighting {eigenbasis,uniform}`, `--smoke`. | ~30 min METR-LA, ~60 min PEMS-BAY |
| `eval_diffusion.py` | Loads a trained checkpoint, runs reverse-SDE sampling on test residuals, computes per-horizon CRPS / PIT / pinball, writes JSON. | ~30-90 min depending on K and N |
| `compute_test_metrics.py` | Computes backbone test-set MAE/RMSE/MAPE from extracted residuals. Used by residual-diagnostics. | ~30 s |
| `compare_ar1_vs_2exp.py` | Diagnostic: AR(1) vs two-exp temporal kernel fit comparison. Used by prior-construction. | ~1 min |
| `diagnose_temporal_acf.py` | Diagnostic: empirical autocorrelation + fitted curves. Used by prior-construction. | ~30 s |
| `verify_backbone.py` | Diagnostic: sanity-check a trained backbone produces sensible numbers. | ~30 s |

---

## `scripts/daic/` — DAIC cluster runners

See [`scripts/daic/README.md`](scripts/daic/README.md) for the full runbook. Quick map:

| File | Role |
|---|---|
| `env.example.sh` | Per-user config template. Copy → `env.sh` (gitignored), edit NetID + paths. |
| `setup_env.sh` | One-time bootstrap: builds Python 3.11 conda env at `~/.conda/envs/thesis-diffusion` with cu121 torch wheels. |
| `_common.sh` | Sourced by every sbatch: loads modules, activates conda env, sets up symlinks, prints diagnostics. |
| `verify_setup.sh` | Login-node pre-flight check — no Slurm, no GPU. |
| `smoke_test.py` | GPU sanity test driver. |
| `sync_to_daic.sh` / `sync_from_daic.sh` | rsync helpers for data only (code goes via git). |
| `submit_*.sbatch` | One per pipeline step: train_backbone, extract_residuals, fit_prior, sanity_checks, train_diffusion, eval_diffusion, smoke. |
| `watch_jobs.sh` | Local-Mac helper to monitor `squeue` + newest log over SSH. |

---

## `tests/` — 162 pytest tests, all green

| File | Coverage |
|---|---|
| `test_preprocessing.py` | Windowing, splits, adjacency loading + Laplacian properties. |
| `test_prior_fit.py` | Q_H + Σ_S builders, fit invariants, covariance Frobenius checks. |
| `test_sde.py` | NumPy SDE: β schedule, eigendecompose_prior, basis round-trip, forward marginal, oracle round-trip on a 4×5 synthetic prior. |
| `test_sde_torch.py` | PyTorch SDE: same fixtures, bit-for-bit comparison with NumPy version (float64), shape + autograd checks (float32). |
| `test_dsm.py` | sample_tau distribution properties, DSM loss zero/positive/finite/differentiable, eigenbasis vs uniform weighting, end-to-end integration through a ScoreNet. |
| `test_score_net.py` | SinusoidalEmbedding, FiLM identity init + learning, GraphConv with/without adj, ScoreNetBlock shape + residual, full ScoreNet + autograd + param count. |
| `test_sampling.py` | reverse_learned shape + autograd, oracle round-trip (mirror of sde-sanity Check 2 in torch), sample_per_window stacks K trajectories. |
| `test_probabilistic.py` | CRPS = 0 at perfect prediction, matches Gaussian analytic value, PIT uniform on calibrated ensembles, pinball zero at perfect quantile, all reject bad inputs. |

Run: `pytest tests/ -q`. Local time: ~8 s.

---

## Configuration

| File | Purpose |
|---|---|
| `pyproject.toml` | Python ≥ 3.11. Dependencies pinned only where necessary (pandas optional via `[hdf]` extra to skip pytables on DAIC). |
| `configs/backbone/gwn.yaml` | GWN hyperparameters (residual_channels=32, blocks=4, etc.). |
| `configs/data/{metrla,pemsbay}.yaml` | Dataset metadata (paths, node count, horizon). |
| `results/prior_params/gwn_{ds}.yaml` | Output of `fit_prior.py` — fitted α, ρ₁, ρ₂, β, ε_s, σ, target_scale[12], δ_h[12], `standardized: true`. |

---

## Where things are *not*

- **No `src/diffusion/{denoiser,sampler,precond}.py`** — those were Phase-0 placeholders, deleted in the GWN-only cleanup. The current sampler is `src/diffusion/sampling.py`; the score net is `src/diffusion/score_net.py`. No EDM-style preconditioner because Tweedie + eigenbasis-weighted loss play that role.
- **No `src/eval/{metrics_det,stratified}.py`** — Phase-0 placeholders, deleted. Backbone deterministic metrics live in `scripts/compute_test_metrics.py`. Diffusion probabilistic metrics live in `src/eval/probabilistic.py`.
- **No `src/training/`** — Phase-0 placeholder, deleted. Training entry points live as scripts: `train_backbone.py`, `train_diffusion.py`.
- **No DCRNN active code** — the GWN-only refactor removed all DCRNN-specific paths. `third_party/DCRNN_PyTorch/` is left in place but unused.
- **`a-thesis-vault/`** is gitignored. Thesis-writing scratchpad lives there.
