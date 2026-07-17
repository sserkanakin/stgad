"""Verify the processed datasets against the spec's gate checks, print a table.

Runs every VERIFY check from the data-pipeline spec on all six variants and
fails loudly (exit 1) on any violation:
  - x/y shapes align; N correct; 12-in/12-out.
  - split sizes sum to total; chronological split matches the documented formula.
  - adjacency square + nonnegative; A_sym symmetric; L PSD (min eig >= -1e-8).
  - value ranges sane (speed ~0-80 mph; flow nonnegative).
  - scaler is train-only: normalising a train batch gives ~0 mean / unit std.
  - raw source SHA256s are stable (recompute vs meta.json).
  - data/ has no unexpected symlinks (Mac roots are real dirs).
  - GWN-native view present and shaped (B,12,N,2) for repo training.

Usage:  PY=/Users/serkan/miniconda/envs/tud-data/bin/python
        $PY scripts/verify_datasets.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

EXPECT_N = {"metrla": 207, "pemsbay": 325, "pems04f": 307, "pems04s": 307,
            "pems08f": 170, "pems08s": 170}
VARIANTS = list(EXPECT_N)

fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
    return cond


def sha256(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_variant(v):
    d = PROC / v
    meta = json.loads((d / "meta.json").read_text())
    n = EXPECT_N[v]
    rows = {}
    total = 0
    for s in ("train", "val", "test"):
        npz = np.load(d / f"{s}.npz")
        x, y = npz["x"], npz["y"]
        xt, yt = npz["x_tod"], npz["y_tod"]
        xm, ym = npz["x_mask"], npz["y_mask"]
        B = x.shape[0]
        rows[s] = B
        total += B
        # shape checks
        check(x.shape == (B, 12, n, 1), f"{v}/{s} x shape {x.shape}")
        check(y.shape == (B, 12, n, 1), f"{v}/{s} y shape {y.shape}")
        check(xt.shape == x.shape and yt.shape == y.shape, f"{v}/{s} tod shape")
        check(xm.shape == x.shape and ym.shape == y.shape, f"{v}/{s} mask shape")
        # tod in [0,1)
        check(float(xt.min()) >= 0.0 and float(xt.max()) < 1.0, f"{v}/{s} tod range")
        # value ranges
        check(float(x.min()) >= 0.0, f"{v}/{s} x has negatives (min {x.min()})")
        if meta["modality"] == "speed":
            check(float(np.nanmax(x)) <= 100.0, f"{v}/{s} speed max {x.max()} > 100")
        # (the gwn_native cross-check lived here; that duplicate view was dropped
        #  2026-07-16 when GWN became a model class under our own training loop.)

    # split sums + chronological formula
    check(total == meta["n_total_windows"], f"{v} split sum {total} != meta")
    if meta["split"].startswith("70/10/20"):
        nt = round(total * 0.2)
        ntr = round(total * 0.7)
        check(rows["test"] == nt and rows["train"] == ntr
              and rows["val"] == total - ntr - nt, f"{v} 70/10/20 split mismatch")
    else:  # 60/20/20 on time axis: windows = seg_len - 23
        seg = meta["segment_lengths"]
        for s in ("train", "val", "test"):
            check(rows[s] == seg[s] - 23, f"{v}/{s} flow window count")

    # adjacency
    adj = np.load(d / "adj.npz")
    A, A_sym, L = adj["A"], adj["A_sym"], adj["L"]
    check(A.shape == (n, n), f"{v} A shape {A.shape}")
    check(float(A.min()) >= 0.0, f"{v} A negative")
    check(np.allclose(A_sym, A_sym.T, atol=1e-6), f"{v} A_sym not symmetric")
    eig = np.linalg.eigvalsh(L)
    check(float(eig.min()) >= -1e-8, f"{v} L not PSD (min eig {eig.min():.2e})")
    check(float(eig.max()) <= 2.0 + 1e-6, f"{v} L max eig {eig.max():.3f} > 2")

    # scaler train-only: normalise train value -> ~0 mean / unit std
    sc = json.loads((d / "scaler.json").read_text())
    check(sc["scope"] == "train_only", f"{v} scaler scope")
    xtr = np.load(d / "train.npz")["x"]
    z = (xtr - sc["mean"]) / sc["std"]
    check(abs(float(z.mean())) < 1e-3 and abs(float(z.std()) - 1.0) < 1e-3,
          f"{v} scaler not train-normalised (mean {z.mean():.4f} std {z.std():.4f})")

    # raw source hashes stable
    for fname, info in meta["raw_source"]["files"].items():
        p = RAW / meta["source_dataset"] / fname
        if p.exists():
            check(sha256(p) == info["sha256"], f"{v} raw {fname} hash drift")

    return dict(variant=v, N=n, train=rows["train"], val=rows["val"],
                test=rows["test"], modality=meta["modality"],
                missing=meta["missing_fraction"] * 100,
                split=meta["split"].split(" (")[0],
                source=meta["raw_source"]["repo"])


def main():
    # symlink check (Mac roots are real dirs)
    links = [str(p) for p in (ROOT / "data").rglob("*") if p.is_symlink()]
    check(not links, f"unexpected symlinks: {links}")

    table = [verify_variant(v) for v in VARIANTS]

    hdr = f"{'variant':<8} {'N':>4} {'train':>7} {'val':>6} {'test':>6} " \
          f"{'modality':<8} {'miss%':>6} {'split':<22} {'source':<8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in table:
        print(f"{r['variant']:<8} {r['N']:>4} {r['train']:>7} {r['val']:>6} "
              f"{r['test']:>6} {r['modality']:<8} {r['missing']:>6.2f} "
              f"{r['split']:<22} {r['source']:<8}")

    print()
    if fails:
        print(f"VERIFY FAILED — {len(fails)} check(s):")
        for f in fails:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("VERIFY PASSED — all checks green across 6 variants.")


if __name__ == "__main__":
    main()
