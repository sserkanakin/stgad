"""Build all six windowed datasets from raw sources into the uniform format.

This is the single, from-scratch preprocessing entry point. It reads ONLY the
raw files under data/raw/<source>/ and writes the processed tree under
data/processed/<variant>/ plus data/manifest.json.

Six variants, two source families with DIFFERENT (baseline-faithful) protocols:

  metrla, pemsbay   (speed, mph)   -- DCRNN / Graph-WaveNet lineage
      raw .h5 (pandas HDFStore, DatetimeIndex) from DCRNN.
      WINDOW the FULL series (12-in/12-out, x_offsets[-11..0], y_offsets[1..12]),
      THEN split the window array 70/10/20 (round(.7)/round(.2)/rest), exactly as
      DCRNN/GWN generate_training_data.py. time-of-day from the DatetimeIndex.
      adjacency = DCRNN adj_mx.pkl (weighted Gaussian-kernel road graph).

  pems04f, pems04s, pems08f, pems08s  (flow / speed)  -- ASTGNN / SpecSTG lineage
      raw pems0X.npz (T,N,3) [flow, occupancy, speed] + PEMS0X.csv distance edges,
      both from ASTGNN. SELECT channel (flow=0, speed=2).
      SPLIT the time axis 60/20/20 (round(.2) test, round(.2) val, rest train),
      THEN window each segment stride-1 (L-23 windows), matching SpecSTG.
      time-of-day synthesised from the sample index (288 slots/day, 2018-01-01 start).
      adjacency = SpecSTG rule: binary connectivity from the distance csv,
      symmetrised (A=A+A^T; A[A!=0]=1).

Uniform output per variant (data/processed/<variant>/):
  train.npz / val.npz / test.npz:
      x      (B,12,N,1) float32  value channel, RAW units (normalised at load)
      y      (B,12,N,1) float32  target value, RAW units
      x_tod  (B,12,N,1) float32  time-of-day in [0,1)  (separate key, NOT in x)
      y_tod  (B,12,N,1) float32
      x_mask (B,12,N,1) uint8    1 = observed, 0 = missing (raw value != 0)
      y_mask (B,12,N,1) uint8
  adj.npz:
      A      (N,N) float32  weighted (speed) or binary (flow) source adjacency
      A_sym  (N,N) float32  symmetric (A+A^T)/2 with self-loops removed
      L      (N,N) float64  symmetric normalised Laplacian I - D^-1/2 A_sym D^-1/2
  scaler.json : per-channel z-score stats from TRAIN ONLY (value channel; tod raw).
  meta.json   : provenance + shapes + split + missing fraction + source hashes.

Normalisation note: SpecSTG normalises on the FULL series with a single global
scalar; we instead store RAW values and a TRAIN-ONLY per-channel scaler (the
project's hard verify gate). SpecSTG's global stat is recorded in meta.json for
reference. Everything else (split, channel, adjacency, windows, masking) matches
the respective baseline.

Usage:
    PY=/Users/serkan/miniconda/envs/tud-data/bin/python
    $PY scripts/build_datasets.py --all
    $PY scripts/build_datasets.py --variant metrla
"""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

X_OFFSETS = np.arange(-11, 1, 1)   # [-11..0]  12 input steps
Y_OFFSETS = np.arange(1, 13, 1)    # [1..12]   12 target steps
SLOTS_PER_DAY = 288                # 5-min sampling

# variant -> static description. raw files are grouped by SOURCE (shared by F/S).
VARIANTS = {
    "metrla":  dict(family="speed", source="metrla",  n=207, modality="speed",
                    units="mph"),
    "pemsbay": dict(family="speed", source="pemsbay", n=325, modality="speed",
                    units="mph"),
    "pems04f": dict(family="flow",  source="pems04",  n=307, modality="flow",
                    units="veh/5min", channel=0),
    "pems04s": dict(family="flow",  source="pems04",  n=307, modality="speed",
                    units="mph",      channel=2),
    "pems08f": dict(family="flow",  source="pems08",  n=170, modality="flow",
                    units="veh/5min", channel=0),
    "pems08s": dict(family="flow",  source="pems08",  n=170, modality="speed",
                    units="mph",      channel=2),
}

SOURCES = {
    "metrla":  dict(repo="DCRNN", repo_url="https://github.com/liyaguang/DCRNN",
                    commit="602afd9d767d3aa1c9b3eac51710d6aeee12c227",
                    drive="https://drive.google.com/uc?id=1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC"),
    "pemsbay": dict(repo="DCRNN", repo_url="https://github.com/liyaguang/DCRNN",
                    commit="602afd9d767d3aa1c9b3eac51710d6aeee12c227",
                    drive="https://drive.google.com/uc?id=1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq"),
    "pems04":  dict(repo="ASTGNN", repo_url="https://github.com/guoshnBJTU/ASTGNN",
                    commit="main",
                    url="https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data/PEMS04/"),
    "pems08":  dict(repo="ASTGNN", repo_url="https://github.com/guoshnBJTU/ASTGNN",
                    commit="main",
                    url="https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data/PEMS08/"),
}
# processing convention authority per family (for meta provenance).
PROC_AUTH = {
    "speed": dict(repo="Graph-WaveNet / DCRNN",
                  repo_url="https://github.com/nnzhan/Graph-WaveNet",
                  commit="6b162e80c59a1d494809252eca055cff93dc66b1",
                  note="adj_mx.pkl == DCRNN's (GWN re-hosts via LFS); windowing per "
                       "generate_training_data.py"),
    "flow":  dict(repo="SpecSTG",
                  repo_url="https://github.com/lequanlin/SpecSTG",
                  commit="c434bcd27507851ad9f40aa92e7147d3e91e5b20",
                  note="split/channel/adjacency rule from data/dataloader_stg.py + "
                       "SpecSTG_exe.py"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_windows(series: np.ndarray, tod: np.ndarray):
    """sliding 12-in/12-out windows over a (T, N) series and (T,) tod.

    returns x (B,12,N,1), y (B,12,N,1), x_tod (B,12,N,1), y_tod (B,12,N,1).
    anchors t in [11, T-12); B = T - 23.
    """
    T, N = series.shape
    min_t = int(abs(X_OFFSETS.min()))        # 11
    max_t = T - int(Y_OFFSETS.max())         # exclusive
    anchors = np.arange(min_t, max_t)
    xi = anchors[:, None] + X_OFFSETS[None, :]   # (B,12)
    yi = anchors[:, None] + Y_OFFSETS[None, :]   # (B,12)

    x = series[xi][..., None]                # (B,12,N,1)
    y = series[yi][..., None]
    tod_b = np.broadcast_to(tod[:, None], (T, N))
    x_tod = tod_b[xi][..., None]
    y_tod = tod_b[yi][..., None]
    return (x.astype(np.float32), y.astype(np.float32),
            x_tod.astype(np.float32), y_tod.astype(np.float32))


def normalized_laplacian(A: np.ndarray):
    """A_sym = symmetrise(A) with self-loops removed; L = I - D^-1/2 A_sym D^-1/2."""
    A = A.astype(np.float64)
    A_sym = (A + A.T) / 2.0
    np.fill_diagonal(A_sym, 0.0)             # drop self-loops (DCRNN adj has diag=1)
    d = A_sym.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    L = np.eye(A.shape[0]) - (d_inv_sqrt[:, None] * A_sym * d_inv_sqrt[None, :])
    return A_sym.astype(np.float32), L


def adjacency_speed(source: str, n: int):
    """DCRNN .pkl -> (sensor_ids, id2idx, adj_mx); return weighted adj_mx (N,N)."""
    fn = "adj_mx.pkl" if source == "metrla" else "adj_mx_bay.pkl"
    with open(RAW / source / fn, "rb") as f:
        _, _, adj = pickle.load(f, encoding="latin1")
    adj = np.asarray(adj, dtype=np.float32)
    assert adj.shape == (n, n), f"{source} adj {adj.shape} != ({n},{n})"
    return adj


def adjacency_flow(source: str, n: int):
    """SpecSTG rule: binary connectivity from PEMS0X.csv edges, symmetrised."""
    csv_path = RAW / source / f"{source.upper()}.csv"
    A = np.zeros((n, n), dtype=np.float32)
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader, None)                   # header
        for row in reader:
            if len(row) < 2:
                continue
            i, j = int(row[0]), int(row[1])
            A[i, j] = 1.0
    A = A + A.T
    A[A != 0] = 1.0                          # binarise (SpecSTG)
    return A


def load_speed_series(source: str):
    """DCRNN .h5 -> (T,N) speed, (T,) time-of-day.

    read with h5py, not pd.read_hdf: the file is a pandas fixed-format frame written
    years ago, and pandas 3's HDFStore reader fails on its byte-encoded attributes.
    the layout is plain -- <group>/axis0 = sensor ids (the column labels; bytes or
    ints), axis1 = int64 ns timestamps (the DatetimeIndex), block0_values = (T,N)
    floats in block0_items order; the group is 'df' for metr-la and 'speed' for pems-bay -- so we rebuild the same (T,N) array in axis0 order and
    the same index directly. verified byte-identical to the pandas path on the
    archived processed tree (archive/20260902/manifest_data.tsv).
    """
    import h5py
    fn = "metr-la.h5" if source == "metrla" else "pems-bay.h5"
    with h5py.File(RAW / source / fn, "r") as f:
        g = f[list(f.keys())[0]]                               # 'df' (metr-la) / 'speed' (pems-bay)
        axis0 = [c.decode() if isinstance(c, bytes) else str(c) for c in g["axis0"][()]]
        items = [c.decode() if isinstance(c, bytes) else str(c) for c in g["block0_items"][()]]
        values = np.asarray(g["block0_values"][()])            # (T, N) in block0_items order
        ts = np.asarray(g["axis1"][()]).astype("int64")        # ns since epoch
    order = [items.index(c) for c in axis0]                    # back to axis0 (column) order
    speed = values[:, order].astype(np.float32)                # (T,N)
    idx = pd.DatetimeIndex(ts.astype("datetime64[ns]"))
    tod = ((idx - idx.normalize()).total_seconds().values / 86400.0).astype(np.float32)
    return speed, tod


def load_flow_series(source: str, channel: int):
    """ASTGNN .npz (T,N,3) -> (T,N) selected channel, (T,) synthetic tod."""
    arr = np.load(RAW / source / f"{source.upper()}.npz")["data"].astype(np.float32)
    series = arr[:, :, channel]                                # (T,N)
    T = series.shape[0]
    tod = ((np.arange(T) % SLOTS_PER_DAY) / SLOTS_PER_DAY).astype(np.float32)
    return series, tod


def split_speed(x, y, x_tod, y_tod):
    """DCRNN: split the WINDOW array 70/10/20 (round .7 / round .2 / rest)."""
    B = x.shape[0]
    n_test = round(B * 0.2)
    n_train = round(B * 0.7)
    n_val = B - n_train - n_test
    sl = {
        "train": slice(0, n_train),
        "val":   slice(n_train, n_train + n_val),
        "test":  slice(B - n_test, B),
    }
    return {s: (x[i], y[i], x_tod[i], y_tod[i]) for s, i in sl.items()}


def split_flow(series, tod):
    """SpecSTG: split the TIME axis 60/20/20, then window each segment."""
    T = series.shape[0]
    test_len = round(T * 0.2)
    val_len = round(T * 0.2)
    train_len = T - val_len - test_len
    segs = {
        "train": slice(0, train_len),
        "val":   slice(train_len, train_len + val_len),
        "test":  slice(train_len + val_len, T),
    }
    out = {}
    for s, sl in segs.items():
        out[s] = build_windows(series[sl], tod[sl])
    return out, (train_len, val_len, test_len)


def write_variant(variant: str):
    spec = VARIANTS[variant]
    src = spec["source"]
    n = spec["n"]
    out_dir = PROC / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {variant}  (source={src}, family={spec['family']}, N={n}) ===")

    # --- load + window + split ---
    if spec["family"] == "speed":
        series, tod = load_speed_series(src)
        T = series.shape[0]
        x, y, x_tod, y_tod = build_windows(series, tod)
        splits = split_speed(x, y, x_tod, y_tod)
        split_desc = "70/10/20 on windows (DCRNN/GWN: window full series then split)"
        seg_lens = None
        global_mean, global_std = float(series.mean()), float(series.std())
    else:
        series, tod = load_flow_series(src, spec["channel"])
        T = series.shape[0]
        splits, seg_lens = split_flow(series, tod)
        split_desc = "60/20/20 on time axis (SpecSTG: split series then window)"
        global_mean, global_std = float(series.mean()), float(series.std())

    # --- train-only scaler (value channel) ---
    xtr = splits["train"][0]                         # (B,12,N,1) raw value
    scaler = {"mean": float(xtr.mean()), "std": float(xtr.std())}

    # --- masks follow the null_val convention used downstream by GWN training:
    # speed (METR-LA/PEMS-BAY) treat raw 0 as missing (null_val=0.0); PEMS flow
    # AND speed treat every value as valid (null_val=nan) since the PeMS arrays
    # are complete and a 0 flow is a real low-traffic reading, not a gap.
    null_val = "0.0" if spec["family"] == "speed" else "nan"

    # --- write split npz with masks ---
    # single uniform view. we used to ALSO emit a "gwn_native" copy so the vendored
    # third_party/Graph-WaveNet/train.py could run unmodified -- dropped 2026-07-16:
    # GWN is now used as a model class under our own training loop, so their train.py
    # is never invoked. that copy doubled the tree (267M of 437M) and, worse, carried
    # its own scaler (their util.load_dataset z-scores in place from x_train[...,0]),
    # so the backbone and the diffusion stage were normalized by two independent
    # scalers. one loop -> one scaler -> one convention for every backbone.
    counts = {}
    for s, (xv, yv, xt, yt) in splits.items():
        if null_val == "nan":
            x_mask = np.ones_like(xv, dtype=np.uint8)
            y_mask = np.ones_like(yv, dtype=np.uint8)
        else:
            x_mask = (xv != 0.0).astype(np.uint8)
            y_mask = (yv != 0.0).astype(np.uint8)
        np.savez_compressed(
            out_dir / f"{s}.npz",
            x=xv, y=yv, x_tod=xt, y_tod=yt, x_mask=x_mask, y_mask=y_mask,
        )
        counts[s] = int(xv.shape[0])
        print(f"  {s}: x{xv.shape} y{yv.shape}  ({counts[s]} windows)")

    # --- adjacency ---
    if spec["family"] == "speed":
        A = adjacency_speed(src, n)
    else:
        A = adjacency_flow(src, n)
    A_sym, L = normalized_laplacian(A)
    np.savez_compressed(out_dir / "adj.npz", A=A.astype(np.float32),
                        A_sym=A_sym, L=L)
    eig_min = float(np.linalg.eigvalsh(L).min())
    print(f"  adj: A{A.shape} nnz={int((A != 0).sum())}  L min-eig={eig_min:.2e}")

    # --- scaler.json ---
    (out_dir / "scaler.json").write_text(json.dumps({
        "channels": ["value"],
        "mean": scaler["mean"],
        "std": scaler["std"],
        "scope": "train_only",
        "tod_normalized": False,
    }, indent=2))

    # --- missing fraction (whole materialised set, observed via mask) ---
    miss = []
    for s in ("train", "val", "test"):
        d = np.load(out_dir / f"{s}.npz")
        miss.append(1.0 - float(d["x_mask"].mean()))
    missing_frac = float(np.mean(miss))

    # --- meta.json ---
    src_files = sorted((RAW / src).glob("*"))
    meta = {
        "variant": variant,
        "source_dataset": src,
        "N": n,
        "modality": spec["modality"],
        "units": spec["units"],
        "in_steps": 12, "out_steps": 12,
        "n_train": counts["train"], "n_val": counts["val"], "n_test": counts["test"],
        "n_total_windows": sum(counts.values()),
        "raw_timesteps": int(T),
        "null_val": null_val,   # GWN --null_val for this variant (0.0 speed / nan flow)
        "split": split_desc,
        "segment_lengths": (
            {"train": seg_lens[0], "val": seg_lens[1], "test": seg_lens[2]}
            if seg_lens else None),
        "channel_index": spec.get("channel"),
        "missing_convention": (
            "raw value == 0 marks missing (null_val=0.0)" if null_val == "0.0"
            else "no masking; all values valid (null_val=nan, complete PeMS array)"),
        "missing_fraction": round(missing_frac, 6),
        "normalization": {
            "stored": "raw values; train-only per-channel z-score in scaler.json",
            "train_mean": scaler["mean"], "train_std": scaler["std"],
            "specstg_global_mean": global_mean if spec["family"] == "flow" else None,
            "specstg_global_std": global_std if spec["family"] == "flow" else None,
        },
        "adjacency_rule": (
            "DCRNN weighted Gaussian-kernel road graph (adj_mx.pkl); "
            "A_sym=(A+A^T)/2 self-loops removed; L normalised Laplacian"
            if spec["family"] == "speed" else
            "SpecSTG binary connectivity from distance csv, symmetrised; "
            "A_sym self-loops removed; L normalised Laplacian"),
        "raw_source": {
            **SOURCES[src],
            "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                      for p in src_files},
        },
        "processing_authority": PROC_AUTH[spec["family"]],
        "time_of_day": (
            "from DatetimeIndex (00:00->0.0)" if spec["family"] == "speed"
            else "synthetic: (index %% 288)/288, 2018-01-01 midnight start"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote scaler.json, meta.json (missing {missing_frac*100:.2f}%)")
    return meta


def write_manifest(metas):
    files = {}
    for p in sorted(PROC.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(ROOT))
            files[rel] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    manifest = {
        "generated": "2026-06-22",
        "mac_data_root": str(ROOT / "data"),
        "python_env": "conda env 'tud-data' (python 3.10.20)",
        "variants": {m["variant"]: {k: m[k] for k in
                     ("N", "modality", "units", "n_train", "n_val", "n_test",
                      "missing_fraction", "split", "source_dataset")} for m in metas},
        "commands": [
            "python scripts/build_datasets.py --all",
        ],
        "processed_files": files,
    }
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n=== manifest.json: {len(files)} processed files hashed ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    targets = list(VARIANTS) if args.all else [args.variant]
    if not targets or targets == [None]:
        ap.error("pass --all or --variant <name>")
    metas = [write_variant(v) for v in targets]
    if args.all:
        write_manifest(metas)


if __name__ == "__main__":
    main()
