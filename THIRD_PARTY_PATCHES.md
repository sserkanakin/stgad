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

### PATCH 2 — gcn_order knob (GCN backbone at order 1)  (2026-06-22)
**Files:** `model.py`, `engine.py`, `train.py`.
**Why:** the residual-backbone grid includes a plain GCN (diffusion order 1)
alongside GWN (order 2). Upstream hardcodes the `gcn(...)` default order (2).
**Change:**
- `model.py` `gwnet.__init__`: add trailing param `gcn_order=2`; in the
  `self.gconv.append(gcn(... support_len=self.supports_len))` call add
  `,order=gcn_order`.
- `engine.py` `trainer.__init__`: add params `gcn_order=2, null_val=0.0`; pass
  `gcn_order=gcn_order` into the `gwnet(...)` call; store `self.null_val`.
- `train.py`: add `--gcn_order` (int, default 2); pass `gcn_order=args.gcn_order`
  into the `trainer(...)` call.
**Reapply:** thread `gcn_order` model->engine->train as above. Defaults to 2, so
GWN/MLP commands are unaffected.

### PATCH 3 — configurable null_val (PEMS flow zeros not masked)  (2026-06-22)
**Files:** `engine.py`, `util.py`, `train.py`.
**Why:** upstream hardcodes `0.0` as the masked-loss missing flag everywhere, so
PEMS flow zeros (real low-traffic readings) get masked out. Speed datasets need
0.0; PEMS flow needs `np.nan` (mask nothing).
**Change:**
- `engine.py`: store `self.null_val` (from the new ctor param); replace the six
  hardcoded `0.0` in `train`/`eval` masked-loss calls with `self.null_val`.
- `util.py` `metric(pred, real)`: add `null_val=0.0` param and pass it to the
  three `masked_*` calls (was hardcoded 0.0).
- `train.py`: add `--null_val` (str, default `'0.0'`); convert
  `NULL_VAL = np.nan if val.lower()=='nan' else float(val)`; pass `null_val=NULL_VAL`
  into `trainer(...)` and `NULL_VAL` into the test-loop `util.metric(pred,real,...)`.
**Reapply:** as above. Defaults keep behaviour identical for speed datasets.

Backbone flag cheatsheet (per the grid):
- GWN: `--gcn_bool --adjtype doubletransition --addaptadj` (gcn_order 2 default)
- GCN: `--gcn_bool --adjtype symnadj --gcn_order 1` (no --addaptadj)
- MLP: (no --gcn_bool)
- null_val: `--null_val 0.0` for METR-LA/PEMS-BAY, `--null_val nan` for PEMS04/08.
