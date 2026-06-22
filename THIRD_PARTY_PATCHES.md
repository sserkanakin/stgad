# Third-party source patches

`third_party/` is **gitignored**, so any edits we make to vendored repos are NOT
version-controlled and would be lost on a fresh clone. This file is the durable
record: every change we make to vendored source is logged here with enough
detail to reapply it. Keep it updated whenever you touch anything under
`third_party/`.

Convention: log additive helper scripts too (note them as "added file", not a
patch), but the critical entries are edits to existing upstream files.

---

## third_party/Graph-WaveNet  (commit 6b162e80c59a1d494809252eca055cff93dc66b1)

### PATCH 1 — model.py: Conv1d → Conv2d for torch >= 2.x  (2026-06-22)
**File:** `third_party/Graph-WaveNet/model.py`, in `gwnet.__init__` (~lines 105-117).
**Why:** upstream declares `gate_convs`, `residual_convs`, `skip_convs` as
`nn.Conv1d` but passes a 2-D `(1, k)` kernel. That ran on torch <= 1.x but
torch >= 2.x rejects 4-D input to `Conv1d.forward` (`RuntimeError: Expected 2D
or 3D input to conv1d, but got [B, C, N, T]`). `nn.Conv2d` with the same
`(1, k)` kernel is mathematically identical on the 4-D tensor.
**Change:** replace `nn.Conv1d(` with `nn.Conv2d(` on those three layer
definitions (the `filter_convs` line above them is already `Conv2d`). No weight
shapes, dilation, or kernel sizes change — purely the layer class.
**Reapply after re-clone:** edit the three `nn.Conv1d(` occurrences in
`gwnet.__init__` to `nn.Conv2d(`. (The previously-existing in-repo workaround,
`src/backbones/__init__.py::_patch_gwn_conv1d`, does the same swap at runtime;
this source patch makes the repo's own `train.py` run directly without it.)
