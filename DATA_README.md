# Dataset pipeline

From-scratch, reproducible build of six windowed spatio-temporal traffic
datasets for the diffusion / Graph-WaveNet-backbone thesis. Built 2026-06-22,
rebuilt from verified sources 2026-07-17 (clean-rebuild branch).

- **Mac data root:** `/Users/serkan/Projects/research-tud/data` (real dirs, no symlinks)
- **DAIC data root:** `/tudelft.net/student-homes/a/sakin/research-tud-data`
  (the repo's `data/` on DAIC is a symlink to this share; we write through it)
- **Build env:** conda env `tud-data`, python 3.10.20 — numpy 2.2.6, pandas 2.3.3,
  scipy 1.15.2, h5py 3.13.0, pytables 3.10.1, scikit-learn 1.7.2, tqdm 4.68.3,
  gdown 6.1.0.
- **Commands:**
  ```
  PY=/Users/serkan/miniconda/envs/tud-data/bin/python
  bash scripts/download_data.sh            # fetch + identity-verify all 10 raw files
  $PY scripts/build_datasets.py --all      # raw -> processed (uniform view)
  $PY scripts/verify_datasets.py           # all spec gate checks + summary table
  $PY scripts/check_connectivity.py        # graph gate: components/isolated/fiedler
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

All variants: 12-in / 12-out windows. Missing-value convention is **per family**:
- **Speed pair (metrla, pemsbay)** — DCRNN convention: raw `== 0` marks a missing
  sensor (masked out of loss & metrics). Measured missing: metrla 8.73%, pemsbay 0.01%.
- **PEMS (pems04/08, flow AND speed)** — `null_val=nan`, nothing masked. Flow zeros
  are real low-traffic readings, not missing (where flow==0, speed reads ~68 mph =
  free-flow); the speed channel has no zeros at all (min 3.0). So 0.00% masked, and
  the "mask y_true==0 for PEMS metrics" rule (SpecSTG/DiffSTG) applies to flow metrics
  only. This matches BasicTS (`null_val: 0.0`, but zeros are genuine so it barely bites).

## Sources & provenance (single-source per family)

Each dataset family is sourced from ONE data repo; baseline repos contribute only
*conventions*, not bytes. Every raw file is fetched by `scripts/download_data.sh`
from its canonical location and identity-checked before the build runs (see that
script for the pinned git blob SHA-1 / published md5 of each file). No project
publishes a checksum for `metr-la.h5`, `PEMS04.npz`, or `PEMS08.npz`, so those are
anchored by git blob SHA-1, verified identical across DCRNN/MTGNN/GTS/DL-Traff and
across ASTGCN↔ASTGNN.

Provenance notes corrected 2026-07-17:
- **Graph-WaveNet ships no data at all** — its README points at DCRNN's Drive
  folder. (The repo that re-hosts these files is MTGNN, same author.) The old
  claim that "GWN re-hosts `adj_mx.pkl` via LFS" was wrong.
- **PEMS04/PEMS08 were introduced by Guo et al., ASTGCN, AAAI 2019** (PeMSD4:
  Bay Area, 307 detectors after 3.5-mile pruning, Jan–Feb 2018; PeMSD8: San
  Bernardino, 170, Jul–Aug 2016). ASTGNN merely re-hosts the identical bytes —
  cite the ASTGCN paper, use the ASTGNN repo for the files.
- The old `download_data.sh` fetched the `.h5` from **Zenodo record 5724362**;
  that record is a third-party CSV conversion (S. Kwak, EPFL) and serves NO `.h5`
  (both URLs 404). METR-LA has no `.h5` mirror on Zenodo — Drive is the only
  source; `pems-bay.h5` is on Zenodo 4263971 (the one file with a published md5).

### Speed — METR-LA (207), PEMS-BAY (325): DCRNN lineage
Li et al., DCRNN, ICLR 2018. <https://github.com/liyaguang/DCRNN> @
`602afd9d767d3aa1c9b3eac51710d6aeee12c227`. Adjacency from the repo; `.h5` series
from the DCRNN Drive folder `10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX` (METR-LA) and
Zenodo 4263971 (PEMS-BAY, checksummed).
  - `metr-la.h5`  Drive id `1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC`  sha256 `64784b76…`  blob `7df00030…`
  - `pems-bay.h5` Zenodo 4263971  sha256 `65d69fb0…`  **md5 `bfa47e6e…`** (the only published checksum)
  - `adj_mx.pkl` sha256 `a35687c6…` blob `a15a5904…`, `adj_mx_bay.pkl` sha256 `116275f5…` blob `6651951a…`
- **Windowing (authoritative = Graph-WaveNet / DCRNN `generate_training_data.py`):**
  x_offsets `[-11..0]`, y_offsets `[1..12]`; window the FULL series then split the
  window array 70/10/20 (`round(.7)` train, `round(.2)` test, rest val). This is
  why counts equal the canonical GWN numbers.
- **Normalization:** train-only global scalar z-score in `scaler.json`; values
  stored RAW. Time-of-day from the DatetimeIndex (00:00→0.0), kept in a separate key.
- **Adjacency:** DCRNN weighted Gaussian-kernel road graph (`adj_mx.pkl`).

### Flow/speed — PEMS04 (307), PEMS08 (170): ASTGCN data, SpecSTG protocol
Data introduced by **Guo et al., ASTGCN, AAAI 2019**; files re-hosted (byte-identical)
by ASTGNN <https://github.com/guoshnBJTU/ASTGNN> `data/PEMS04|PEMS08` (main).
  - `PEMS04.npz` sha256 `95a3c9b7…` blob `a1a39eac…`, `PEMS08.npz` sha256 `e1d03ce7…` blob `f5f3e84a…`.
  - **Channel order `[flow, occupancy, speed]` asserted by value range, not by citation:**
    occupancy is the only channel bounded in [0,1]; speed is min 3.0 (never 0); flow is
    unbounded counts. The ASTGCN *paper's prose* says "flow, speed, occupancy" — that
    ordering is WRONG and is the likely origin of the widespread mis-citation. So
    flow=ch0, speed=ch2. (`verify_datasets.py` re-checks this by range on every build.)
- **Split/windowing (authoritative = SpecSTG `data/dataloader_stg.py` +
  `SpecSTG_exe.py`):** <https://github.com/lequanlin/SpecSTG> @
  `c434bcd27507851ad9f40aa92e7147d3e91e5b20`. `test_pct=0.2`, `val_pct=0.2` →
  split the TIME axis 60/20/20 (first 60% train, middle 20% val, last 20% test),
  then window each segment stride-1 (L−23). 12-in/12-out.
- **Adjacency (SpecSTG rule):** binary connectivity from `PEMS0X.csv` edges,
  symmetrised (`A = A + Aᵀ; A[A≠0]=1`) — NOT distance-weighted.
- **Normalization:** see the dedicated Normalization section below. Everything else
  (split, channel, adjacency, windows, masking) matches SpecSTG. Time-of-day
  synthesised from the sample index (288 slots/day, 2018-01-01 start), separate key.

## Output layout (per `data/processed/<variant>/`)

- `train.npz` / `val.npz` / `test.npz` — uniform format:
  - `x` (B,12,N,1) float32 value, RAW · `y` (B,12,N,1) target, RAW
  - `x_tod`/`y_tod` (B,12,N,1) time-of-day in [0,1) (separate keys)
  - `x_mask`/`y_mask` (B,12,N,1) uint8, 1=observed (raw value ≠ 0)
- `adj.npz` — `A` (N,N weighted/binary), `A_sym` (N,N symmetric, self-loops removed),
  `L` (N,N symmetric normalised Laplacian `I − D^-½ A_sym D^-½`, PSD, eigs ∈ [0,2]).
- `scaler.json` — train-only **global scalar** z-score (`mean`,`std`) over the value
  channel; tod unnormalised. (One value channel, so "per-channel" = one scalar. This
  is the GWN/DCRNN/DiffSTG community convention; see Normalization below.)
- `meta.json` — provenance, shapes, split, missing fraction, raw source URLs+SHA256,
  preprocessing repo+commit.
- `data/manifest.json` — aggregate of all processed files (sha256) + build date + commands.

## Normalization (matches the community convention)

The data-level scaler is a **single global scalar z-score fit on TRAIN only**
(`{mean, std}` over all valid train entries of the value channel) — the same
convention as Graph-WaveNet, DCRNN, and DiffSTG. Values are stored RAW; the scaler
is applied at load. Time-of-day is kept raw in [0,1) in a separate key (never
z-scored), exactly as GWN appends it as a second input channel.

Divergence from SpecSTG (noted, deliberate): SpecSTG fits its scalar on the FULL
series *including test* (a leak). We fit on train only. To reproduce SpecSTG's
published numbers one must replicate the leak; SpecSTG's global mean/std is recorded
in each `meta.json` (`normalization.specstg_global_*`) for that purpose.

Downstream (diffusion-stage) normalization — the residual scale `σ_r` and the raw-
signal centering — is NOT a data-layer concern and is documented with the diffusion
methodology, not here. Those are our own modelling choices, not a paper convention.

## Backbones

Backbones (MLP / GCN / GWN) are trained under our OWN training loop, using each
model as a plain module. The vendored `third_party/Graph-WaveNet` supplies only the
`gwnet` model class (torch ≥ 2.x needs its `model.py` Conv1d→Conv2d fix — see
`THIRD_PARTY_PATCHES.md`); its `train.py`/`util.py`/`engine.py` are unused. One loop
owns masking (`null_val` per family), the scaler, checkpointing, and seeds for every
backbone, so the backbone and the diffusion stage share a single normalization.
Residual extraction runs the trained checkpoint no-shuffle over each split and writes
`r = y − ŷ` in raw units, shape (B, N, H).

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
