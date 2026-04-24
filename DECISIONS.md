# Design Decisions

Log of non-obvious design choices and their rationale. Append entries as work progresses.
Format: `## YYYY-MM-DD — <short title>` followed by the decision and reason.

---

## External code pins

| Component | Repo | Commit hash |
|-----------|------|-------------|
| DCRNN_PyTorch | github.com/chnsh/DCRNN_PyTorch | d92490b |
| Graph-WaveNet | github.com/nnzhan/Graph-WaveNet | 6b162e8 |

---

## 2026-04-22 — Project root is repo root

**Decision**: scaffold directly in the repo root (`research-tud/`) rather than a `thesis_diffusion/` subdirectory.

**Reason**: avoids an extra nesting level with no benefit; project context files moved to `docs/`.

---

## 2026-04-22 — Backbones: DCRNN + Graph WaveNet, not STID

**Decision**: use DCRNN (Li et al., ICLR 2018) and Graph WaveNet (Wu et al., IJCAI 2019) as the backbone predictors. STID is explicitly excluded as a main backbone.

**Reason**: STID is a pure MLP that ignores the road network graph. The thesis contribution encodes graph structure in the Kronecker prior (graph-Laplacian spatial factor). Using graph-based backbones keeps the story internally consistent — the backbone captures mean graph dynamics, the structured prior captures residual spatial uncertainty. Using both DCRNN and GWN also provides a "framework generality" argument (following CoST).

**Spec reference**: updated CLAUDE_CODE_TASK_1.md, "Why two graph backbones" section.

---

## 2026-04-22 — Normalisation at load time; y never normalised

**Decision**: raw (un-normalised) arrays are saved to .npz. `TrafficDataset` applies z-score normalisation to x's speed channel at instantiation. y (targets) are stored and used in raw mph throughout — never normalised.

**Reason**: confirmed from DCRNN and Graph WaveNet source code (`generate_training_data.py` in both repos). Papers in this space train on normalised x but report metrics in raw units by inverse-transforming predictions. The scaler (mean, std) is saved to `{dataset}_scaler.yaml` and exposed as `dataset.mean` / `dataset.std` for inverse-transform at evaluation time.

**Spec reference**: updated CLAUDE_CODE_TASK_1.md, "Normalization" section.

---

## 2026-04-22 — time_of_day as continuous float [0,1), no day_of_week

**Decision**: x channel 1 is time-of-day as a continuous float in [0, 1): `(t - midnight) / 24h`. Day-of-week is not added to x. This is the DCRNN/GWN convention for METR-LA and PEMS-BAY.

**Alternatives considered**: integer time-of-day index (0–287) fed to learnable embeddings, plus day-of-week, as in STID.

**Reason**: using the same feature encoding as DCRNN/GWN ensures our preprocessing is pixel-identical to the reference implementations, which is what makes our numbers comparable to published baselines. STID uses a different encoding, but STID is not our backbone.

**Spec reference**: updated CLAUDE_CODE_TASK_1.md, "Feature channels in x" section.

---

## 2026-04-23 — h5 filenames are lowercase with hyphens

**Decision**: raw h5 files are named `metr-la.h5` and `pems-bay.h5` (not `METR-LA.h5` / `PEMS-BAY.h5`).

**Reason**: this is how they come from the DCRNN Google Drive. All path references updated to match.

---

## 2026-04-23 — Symmetrise + strip self-loops before Laplacian

**Decision**: `compute_normalized_laplacian` removes diagonal self-loops and symmetrises via `(A + A^T) / 2` before computing L = I - D^{-1/2} A D^{-1/2}.

**Reason**: DCRNN adjacency matrices are directed (max asymmetry ~0.996) and have unit self-loops on every node. Without this fix, the "Laplacian" has 13 negative eigenvalues (range [-0.40, 1.26]), making it non-PSD — invalid for the spatial prior and for ChebConv. After symmetrisation: eigenvalues in [0, 1.71], fully PSD.

**Note**: the backbone models (DCRNN, GWN) use the directed adjacency directly for diffusion convolution — this fix only applies to the Laplacian used in our prior and in `compute_scaled_laplacian` for ChebConv.

**Spec reference**: §5 of PROJECT_SPEC.md (Kronecker prior), §8 (DCRNN reuse policy).

---

## 2026-04-23 — DCRNN falls back to CPU on MPS machines

**Decision**: `load_backbone("dcrnn", ...)` always targets `torch.device("cuda" if cuda else "cpu")`, ignoring MPS even when it is available.

**Reason**: `dcrnn_cell.py` creates `torch.sparse_coo_tensor` objects at module-import time on a device determined by a module-level global (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`). Sparse tensor operations are not supported on MPS; `.to("mps")` on the model would move parameters but leave sparse supports on CPU, causing device mismatch on the first forward pass. Since we cannot modify third-party source, we fall back to CPU when CUDA is unavailable.

**Impact**: DCRNN smoke tests and full training on an MPS Mac run on CPU. This is acceptable — the same code runs on CUDA cluster nodes for real training. GWN is fully MPS-compatible.

---

## 2026-04-23 — GWN Conv1d patched to Conv2d at load time

**Decision**: `_patch_gwn_conv1d()` in `src/backbones/__init__.py` replaces every `nn.Conv1d` in GWN's `gate_convs`, `residual_convs`, and `skip_convs` with an equivalent `nn.Conv2d` immediately after instantiation.  Weights are copied by shape (data layout is identical).

**Reason**: The published GWN source (`nnzhan/Graph-WaveNet`) uses `nn.Conv1d(kernel_size=(1, k))` which produces a 4D weight tensor `(out, in, 1, k)` — identical in memory to `nn.Conv2d(kernel_size=(1, k))`. In PyTorch < ~1.10 this accidentally worked because the internal dispatch used the weight dimensionality to pick the convolution path.  In PyTorch ≥ 2.x, `nn.Conv1d.forward` strictly validates `input.ndim == 3` and raises an error on the 4D feature maps `(B, C, N, T)`.  We cannot modify third-party source, so the patch is applied in our wrapper.

**Verification**: All 4 combinations (DCRNN/GWN × METR-LA/PEMS-BAY) pass `verify_backbone.py` with correct output shapes and decreasing loss over 3 gradient steps.

---

## 2026-04-23 — config key `hidden_state_size` mapped to DCRNN's `rnn_units`

**Decision**: `configs/backbone/dcrnn.yaml` stores the hidden dim as `hidden_state_size: 64`. The wrapper in `src/backbones/__init__.py` passes it to `DCRNNModel` as `rnn_units=64`.

**Reason**: the public DCRNN config convention (and the project spec) uses `hidden_state_size`; the actual `chnsh/DCRNN_PyTorch` code reads `model_kwargs.get('rnn_units')` (not `hidden_state_size`). The mapping is done explicitly in the wrapper so both the user-facing config and the third-party code remain unchanged.

---

## Template entry

**Decision**: <what was chosen>.

**Alternatives considered**: <what else was considered>.

**Reason**: <why this choice>.

**Spec reference**: §<N> of PROJECT_SPEC.md or section of CLAUDE_CODE_TASK.md.
