# Dataset pipeline

From-scratch, reproducible build of six windowed spatio-temporal traffic
datasets for the diffusion / Graph-WaveNet-backbone thesis. Built 2026-06-22.

- **Mac data root:** `/Users/serkan/Projects/research-tud/data` (real dirs, no symlinks)
- **DAIC data root:** `/tudelft.net/student-homes/a/sakin/research-tud-data`
  (the repo's `data/` on DAIC is a symlink to this share; we write through it)
- **Build env:** conda env `tud-data`, python 3.10.20 — numpy 2.2.6, pandas 2.3.3,
  scipy 1.15.2, h5py 3.13.0, pytables 3.10.1, scikit-learn 1.7.2, tqdm 4.68.3,
  gdown 6.1.0.
- **Commands:**
  ```
  PY=/Users/serkan/miniconda/envs/tud-data/bin/python
  $PY scripts/build_datasets.py --all      # raw -> processed (uniform + GWN-native)
  $PY scripts/verify_datasets.py           # all spec gate checks + summary table
  ```

## Variants

| variant | N | modality | units | train/val/test | split | source |
|---------|---|----------|-------|----------------|-------|--------|
| metrla  | 207 | speed | mph | 23974 / 3425 / 6850 | 70/10/20 on windows | DCRNN |
| pemsbay | 325 | speed | mph | 36465 / 5209 / 10419 | 70/10/20 on windows | DCRNN |
| pems04f | 307 | flow | veh/5min | 10173 / 3375 / 3375 | 60/20/20 on time axis | ASTGNN |
| pems04s | 307 | speed | mph | 10173 / 3375 / 3375 | 60/20/20 on time axis | ASTGNN |
| pems08f | 170 | flow | veh/5min | 10691 / 3548 / 3548 | 60/20/20 on time axis | ASTGNN |
| pems08s | 170 | speed | mph | 10691 / 3548 / 3548 | 60/20/20 on time axis | ASTGNN |

All variants: 12-in / 12-out windows. Missing-value convention: raw value `== 0`
marks missing (masked loss downstream); METR-LA ~8.7%, PEMS-BAY ~0.01%, PEMS
flow 0.3–1.5%, PEMS speed ~0%.

## Sources & provenance (single-source per family)

Each dataset family is sourced from ONE data repo; baseline repos (GWN, SpecSTG)
contribute only *conventions*, not bytes (GWN's `adj_mx.pkl` is byte-identical to
DCRNN's; it re-hosts via LFS).

### Speed — METR-LA (207), PEMS-BAY (325): DCRNN lineage
- Raw `.h5` + adjacency + distance CSVs from **DCRNN**
  <https://github.com/liyaguang/DCRNN> @ `602afd9d767d3aa1c9b3eac51710d6aeee12c227`.
  - `metr-la.h5`  Drive id `1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC`  sha256 `64784b76d6fb8ec9…`
  - `pems-bay.h5` Drive id `1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq`  sha256 `65d69fb0a2323dba…`
  - `adj_mx.pkl` `a35687c6…`, `adj_mx_bay.pkl` `116275f5…`
  - Drive folder (DCRNN README): `10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX`
- **Windowing (authoritative = Graph-WaveNet / DCRNN `generate_training_data.py`):**
  x_offsets `[-11..0]`, y_offsets `[1..12]`; window the FULL series then split the
  window array 70/10/20 (`round(.7)` train, `round(.2)` test, rest val). This is
  why counts equal the canonical GWN numbers.
- **Normalization:** train-only z-score (per-channel) in `scaler.json`; values
  stored RAW. Time-of-day from the DatetimeIndex (00:00→0.0), kept in a separate key.
- **Adjacency:** DCRNN weighted Gaussian-kernel road graph (`adj_mx.pkl`).

### Flow/speed — PEMS04 (307), PEMS08 (170): ASTGNN + SpecSTG lineage
- Raw `pems0X.npz` (T,N,3) `[flow, occupancy, speed]` + `PEMS0X.csv` distance edges
  from **ASTGNN** <https://github.com/guoshnBJTU/ASTGNN> `data/PEMS04|PEMS08` (main).
  - `PEMS04.npz` `95a3c9b7…`, `PEMS08.npz` `e1d03ce7…` (PEMS08.npz matches prior copy).
  - Channel order verified from the array AND SpecSTG's loader: flow=0, speed=2.
- **Split/windowing (authoritative = SpecSTG `data/dataloader_stg.py` +
  `SpecSTG_exe.py`):** <https://github.com/lequanlin/SpecSTG> @
  `c434bcd27507851ad9f40aa92e7147d3e91e5b20`. `test_pct=0.2`, `val_pct=0.2` →
  split the TIME axis 60/20/20 (first 60% train, middle 20% val, last 20% test),
  then window each segment stride-1 (L−23). 12-in/12-out.
- **Adjacency (SpecSTG rule):** binary connectivity from `PEMS0X.csv` edges,
  symmetrised (`A = A + Aᵀ; A[A≠0]=1`) — NOT distance-weighted.
- **Normalization divergence (noted):** SpecSTG normalises on the FULL series with
  a single global scalar; we instead store RAW + a **train-only per-channel**
  `scaler.json` (the project's hard verify gate). SpecSTG's global mean/std is
  recorded in each `meta.json` (`normalization.specstg_global_*`) for reference.
  Everything else (split, channel, adjacency, windows, masking) matches SpecSTG.
  Time-of-day synthesised from the sample index (288 slots/day, 2018-01-01 start),
  separate key.

## Output layout (per `data/processed/<variant>/`)

- `train.npz` / `val.npz` / `test.npz` — uniform format:
  - `x` (B,12,N,1) float32 value, RAW · `y` (B,12,N,1) target, RAW
  - `x_tod`/`y_tod` (B,12,N,1) time-of-day in [0,1) (separate keys)
  - `x_mask`/`y_mask` (B,12,N,1) uint8, 1=observed (raw value ≠ 0)
- `adj.npz` — `A` (N,N weighted/binary), `A_sym` (N,N symmetric, self-loops removed),
  `L` (N,N symmetric normalised Laplacian `I − D^-½ A_sym D^-½`, PSD, eigs ∈ [0,2]).
- `scaler.json` — train-only per-channel z-score (`mean`,`std`); tod unnormalised.
- `meta.json` — provenance, shapes, split, missing fraction, raw source URLs+SHA256,
  preprocessing repo+commit.
- `gwn_native/` — **GWN-repo-ready view** of the SAME windows (see below).
- `data/manifest.json` — aggregate of all processed files (sha256) + build date + commands.

## Training the Graph-WaveNet backbone directly from the repo

The vendored `third_party/Graph-WaveNet` repo trains directly on the
`gwn_native/` view — no reimplemented wrapper:

- `gwn_native/{train,val,test}.npz`: `x`,`y` both (B,12,N,2) = `[value_raw, tod]`.
  The repo's `util.load_dataset` builds its own train-only scaler from
  `x_train[...,0]` and normalises in place (exactly GWN's `generate_training_data.py`).
- `gwn_native/adj_mx.pkl`: DCRNN-format `[sensor_ids, id→idx, A]` for
  `util.load_adj(..., 'doubletransition')`.

Example (per variant):
```
python third_party/Graph-WaveNet/train.py \
    --data data/processed/<variant>/gwn_native \
    --adjdata data/processed/<variant>/gwn_native/adj_mx.pkl \
    --num_nodes <N> --in_dim 2 --seq_length 12 --gcn_bool --addaptadj --adjtype doubletransition
```
Residual extraction (`r = y − ŷ` in raw units, shape (B,N,H)) loads the trained
checkpoint with the repo's `gwnet` + `util.load_adj`, runs inference no-shuffle
over each split, and `scaler.inverse_transform`s. Verified end-to-end on pems08s.

**Patch required:** torch ≥ 2.x needs the GWN `model.py` Conv1d→Conv2d fix —
see `THIRD_PARTY_PATCHES.md`.

## Mac ↔ DAIC sync

Code travels via git; data is pushed with rsync over ssh (content-verified):
```
rsync -av --checksum --no-links \
    data/raw data/processed data/manifest.json \
    <netid>@<daic-host>:/tudelft.net/student-homes/a/sakin/research-tud-data/
```
DAIC's `data/` symlink target is written through (never replaced). After sync,
re-run `scripts/verify_datasets.py` on DAIC and confirm `manifest.json` hashes
match the Mac's bit-for-bit.
