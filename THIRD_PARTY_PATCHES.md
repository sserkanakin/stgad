# Third-party source patches

`third_party/` is **gitignored**, so edits to vendored repos are NOT version-controlled
and are lost on a fresh clone. This file is the durable record: every change we make to
vendored source is logged here with enough detail to reapply it from a clean clone.

**Policy (clean-rebuild, 2026-07-16):** vendored repos are used as *libraries only* — we
import their model classes and run them under our own training/eval code. We do not use
their training loops, losses, masking, or checkpointing. That keeps the patch surface to
the minimum required to make the code run at all.

Re-clone from scratch:

```sh
rm -rf third_party && mkdir -p third_party && cd third_party
git clone https://github.com/wenhaomin/DiffSTG.git DiffSTG
git -C DiffSTG checkout ed2c523022a2490ddb9b78aab641eac53b0274d3
git clone https://github.com/nnzhan/Graph-WaveNet.git Graph-WaveNet
git -C Graph-WaveNet checkout 6b162e80c59a1d494809252eca055cff93dc66b1
# then reapply PATCH 1 below (Graph-WaveNet only)
```

---

## third_party/DiffSTG (commit ed2c523)

**Two roles.** (a) In the main pipeline we import `algorithm.diffstg.ugnet.UGnet` as a
denoiser (library-only, no edits there). (b) For the **cited DiffSTG baseline** we run
their *own* `train.py` end-to-end (faithful protocol: their loader, 60/20/20 split,
global z-score, DDPM-200 training, DDIM-multi 40-step test sampling, their
MAE/RMSE/MAPE/CRPS/MIS). Role (b) needs the six patches below — all confined to
`train.py`; the model, loss, training loop, sampler, and metrics are untouched.

Runner: `scripts/daic/submit_diffstg.sh` (→ `submit_diffstg_baseline.sbatch`), seed 43.
Data files (below) are built by `scripts/baselines/build_diffstg_data.py`.

### Data files (not source, but required) — `data/dataset/<NAME>/{flow.npy,adj.npy}`

DiffSTG's `CleanDataset` reads a raw series `flow.npy` (T,V,F) + `adj.npy` (V,V) per
dataset and does its own z-score/window/split. We generate these for our six variants
with `python scripts/baselines/build_diffstg_data.py` (writes single-channel (T,V,1)
flow from our raw sources + our `data/processed/<v>/adj.npz['A']`). Verified anchor:
`pems08f` flow **and** adj come out byte-identical to DiffSTG's shipped `PEMS08/`.
These live under the gitignored `third_party/`, so rebuild (or rsync) them per clone.

### PATCH 1 — optional `nni` import (`train.py`, module top, after the model import)

`train.py` imports `nni` only inside `__main__` and calls `nni.report_final_result(...)`
unconditionally at the end of `main()`. To run standalone (no NNI experiment / package),
add a module-level `try: import nni / except: nni = _NniStub()` where the stub's
`get_next_parameter()`→`{}` and the `report_*` are no-ops. Also drop the duplicate
`import nni` in `__main__` and guard `print('GPU:', torch.cuda.current_device())` with
`if torch.cuda.is_available()` (it raises on CPU). Faithful: nni only reports to a tuner.

### PATCH 2 — `--seed` arg (`get_params`) + use it (`main`)

Add `parser.add_argument("--seed", type=int, default=43)`. In `main`, change the hardcoded
`setup_seed(2022)` → `setup_seed(params.get('seed', 2022))`. Our run seed is 43.

### PATCH 3 — dataset config blocks for our six variants (`default_config`, after AIR_GZ)

`default_config` only knows PEMS08/AIR_BJ/AIR_GZ. Add a block: for
`name in {metrla:207, pemsbay:325, pems04f:307, pems04s:307, pems08f:170, pems08s:170}`
set `num_features=1`, `num_vertices=V`, `points_per_hour=12`, and
`val_start_idx=int(T*0.6)`, `test_start_idx=int(T*0.8)` where `T = np.load(feature_file,
mmap_mode='r').shape[0]`. This is DiffSTG's native 60/20/20-on-time split (their PEMS08 uses
the identical formula). Names are lowercase, so `CleanDataset.read_data`'s `'PEMS'`/`'AIR'`
substring checks miss → its generic `np.load` branch loads our (T,V,1) file as-is (correct).

### PATCH 4 — neutralize GPU auto-dispatch (`default_config`)

Replace the `GPU().get_usefuel_gpu(...)` + `torch.cuda.set_device` block with: if
`os.environ.get('DIFFSTG_AUTO_GPU','0')=='1'` keep their nvidia-smi scan, else
`config.gpu_id = 0 if torch.cuda.is_available() else None`. Their scanner picks a GPU by
**absolute** id, which mis-selects under a single-GPU SLURM allocation (only the allocated
card is visible) and errors off-CUDA (Mac/CPU). Default now trusts `CUDA_VISIBLE_DEVICES`.

### PATCH 5 — `is_test` (smoke) via env, not the broken bool arg (`main`)

Replace `config.is_test = params['is_test']` with
`config.is_test = os.environ.get('DIFFSTG_SMOKE','0') == '1'`. Their `--is_test type=bool
default=True` can't be set False from the CLI (`bool("False")` is True), and `is_test=True`
caps training to a few batches / 2 epochs. So a full run needs no flag; `DIFFSTG_SMOKE=1`
opts into the fast plumbing check.

### PATCH 6 — `torch.load(..., weights_only=False)` (`main`, best-model reload)

torch ≥ 2.6 flipped `weights_only` default to True, which can't unpickle the whole `DiffSTG`
module saved by their `torch.save(model, path)`. Without this the reload throws and the final
**test** eval silently falls back to the last-epoch model instead of the best checkpoint. Add
`weights_only=False` to the `torch.load(model_path, ...)` in the `try:` before the test loop.

---

## third_party/Graph-WaveNet (commit 6b162e8)

Used as a **model class only** (`model.py::gwnet`), as the strong backbone on the
conditioning-strength ladder. `model.py` imports nothing but torch, so their
`engine.py` / `train.py` / `util.py` are unused and unpatched.

### PATCH 1 — model.py: Conv1d → Conv2d (upstream typo; required for torch >= 2.x)

**File:** `model.py`, in `gwnet.__init__` (lines ~105, ~110, ~115).
**Change:** `nn.Conv1d(` → `nn.Conv2d(` on `gate_convs`, `residual_convs`, `skip_convs`.

**Why.** Upstream declares these three as `nn.Conv1d` but passes a 2-D `(1, k)` kernel and
feeds them a 4-D `(B, C, N, T)` tensor. torch >= 2.x rejects this outright:

```
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d,
              but got input of size: [2, 32, 207, 13]
```

This is an upstream **typo**, not a design choice — `filter_convs` (line 101) is already
`nn.Conv2d` with a character-for-character identical argument list, and `filter_convs` /
`gate_convs` are the two halves of one gated activation unit (`tanh(filter) * sigmoid(gate)`),
so they must produce the same shape or the multiply is invalid. Weight shapes are identical
between the two classes (verified); the swap changes no math, no dilation, no kernel size.

**Verify after reapplying:** `grep -c "nn.Conv1d" third_party/Graph-WaveNet/model.py` → `0`.

---

## Dropped patches (previously applied, deliberately not carried over)

Kept here so the reasoning survives; recover the old text from git history
(`git show raw-signal-gad:THIRD_PARTY_PATCHES.md`).

| Old patch | Why dropped |
|---|---|
| PATCH 2 — `gcn_order` knob | Only existed to coerce GWN into a 1-hop "GCN" arm. We now build our own GCN backbone, so we never need to reshape theirs. |
| PATCH 3 — configurable `null_val` | Lived in `engine.py`/`util.py`/`train.py`. Our training loop owns masking, so the PEMS-flow convention (`null_val=nan`; flow zeros are real low-traffic readings, not missing) is enforced in our code for every backbone at once. |
| PATCH 4 — best-only checkpointing | Lived in `train.py`. Our loop owns checkpointing. |
