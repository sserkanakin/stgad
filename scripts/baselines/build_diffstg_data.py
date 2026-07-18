#!/usr/bin/env python
"""
build the raw-series data files DiffSTG's own loader expects, for our six traffic
variants, so we can run their code (third_party/DiffSTG/train.py) unmodified in
spirit -- faithful to their paper protocol -- but on the datasets we use.

DiffSTG's CleanDataset reads, per dataset <NAME>:
  data/dataset/<NAME>/flow.npy   raw series, shape (T, V, F)   (it z-scores + windows itself)
  data/dataset/<NAME>/adj.npy    adjacency,  shape (V, V)      (UGnet builds dual random-walk supports)

We write a SINGLE-channel flow.npy (T, V, 1) holding the target modality, because
DiffSTG only branches on the literal substring 'PEMS'/'AIR' (uppercase) in the
dataset name to pick channel 0; our lowercase names fall to its generic
`np.load(feature_file)` branch, which uses the array as-is. A (T, V, 1) file is
therefore unambiguous for every variant.

Sources (same bytes as the rest of the thesis, see DATA_README.md):
  metrla / pemsbay  -- DCRNN .h5 (read via h5py block0_values, no pytables needed)
  pems04* / pems08* -- ASTGNN .npz 'data' (T, V, 3) = [flow, occupancy, speed]
Adjacency is our canonical per-variant `data/processed/<NAME>/adj.npz['A']`
(weighted-directed for the speed pair, binary-symmetric for PEMS -- both drop
straight into UGnet's asym_adj supports).

usage:
    python scripts/baselines/build_diffstg_data.py            # all six
    python scripts/baselines/build_diffstg_data.py pems08f    # one or more
"""
import sys
from pathlib import Path

import numpy as np
import h5py

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
PROCESSED = REPO / "data" / "processed"
OUT_ROOT = REPO / "third_party" / "DiffSTG" / "data" / "dataset"

# our PEMS npz channel order is [flow, occupancy, speed]
FLOW, SPEED = 0, 2

# variant -> how to pull its single target channel out of the raw source
SPEC = {
    "metrla":  dict(kind="h5",   file="metrla/metr-la.h5",   h5key="df"),
    "pemsbay": dict(kind="h5",   file="pemsbay/pems-bay.h5", h5key="speed"),
    "pems04f": dict(kind="npz",  file="pems04/PEMS04.npz",   channel=FLOW),
    "pems04s": dict(kind="npz",  file="pems04/PEMS04.npz",   channel=SPEED),
    "pems08f": dict(kind="npz",  file="pems08/PEMS08.npz",   channel=FLOW),
    "pems08s": dict(kind="npz",  file="pems08/PEMS08.npz",   channel=SPEED),
}


def load_series(name: str) -> np.ndarray:
    """return the target series as (T, V) float64, raw units."""
    spec = SPEC[name]
    if spec["kind"] == "h5":
        with h5py.File(RAW / spec["file"], "r") as h:
            # DCRNN stores the DataFrame as <key>/block0_values -> (T, V)
            series = h[f"{spec['h5key']}/block0_values"][()]
        return np.asarray(series, dtype=np.float64)
    arr = np.load(RAW / spec["file"])["data"].astype(np.float64)  # (T, V, 3)
    return arr[:, :, spec["channel"]]


def build(name: str) -> None:
    series = load_series(name)                       # (T, V)
    flow = series[:, :, None]                         # (T, V, 1) single channel
    adj = np.load(PROCESSED / name / "adj.npz")["A"]  # (V, V) our canonical adjacency

    assert adj.shape[0] == adj.shape[1] == flow.shape[1], \
        f"{name}: adj {adj.shape} vs V={flow.shape[1]}"

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "flow.npy", flow.astype(np.float32))
    np.save(out_dir / "adj.npy", np.asarray(adj, dtype=np.float32))

    T, V, _ = flow.shape
    val_i, test_i = int(T * 0.6), int(T * 0.8)
    print(f"{name:9s} flow {flow.shape} adj {adj.shape}  "
          f"T={T} val@{val_i} test@{test_i}  "
          f"range[{np.nanmin(flow):.1f},{np.nanmax(flow):.1f}] -> {out_dir}")

    # sanity anchor: pems08f is literally DiffSTG's shipped PEMS08 (flow channel)
    if name == "pems08f":
        ship_dir = OUT_ROOT / "PEMS08"
        if (ship_dir / "flow.npy").exists() and (ship_dir / "adj.npy").exists():
            ship_flow = np.load(ship_dir / "flow.npy")[:, :, FLOW]
            ship_adj = np.load(ship_dir / "adj.npy")
            same_flow = np.allclose(ship_flow, flow[:, :, 0])
            same_adj = np.array_equal(ship_adj, adj)
            print(f"  [anchor] pems08f flow == shipped PEMS08 flow: {same_flow} | "
                  f"adj == shipped: {same_adj}")


def main() -> None:
    names = sys.argv[1:] or list(SPEC)
    for name in names:
        if name not in SPEC:
            raise SystemExit(f"unknown variant {name!r}; choose from {list(SPEC)}")
        build(name)


if __name__ == "__main__":
    main()
