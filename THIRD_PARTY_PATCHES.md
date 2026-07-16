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

**No patches. Pristine.** We import `algorithm.diffstg.ugnet.UGnet` as the denoiser and
nothing else. Verify with `git -C third_party/DiffSTG status --porcelain` (expect empty).

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
