"""Tests for the rebuilt dataset pipeline (scripts/build_datasets.py output).

Validates the real processed data on disk for all six variants plus the
migrated adjacency loader (reads data/processed/<variant>/adj.npz). Variant
data tests skip cleanly if a variant hasn't been built yet.
"""
import json

import numpy as np
import pytest

from src.utils import paths
from src.data.adjacency import (
    load_adjacency,
    compute_normalized_laplacian,
    _NUM_NODES,
)

VARIANTS = list(_NUM_NODES)


def _built(v):
    return paths.split_npz(v, "train").exists() and paths.adj_npz(v).exists()


BUILT = {v: _built(v) for v in VARIANTS}


# ── pure adjacency math (synthetic) ──────────────────────────────────────────

def test_normalized_laplacian_symmetric():
    rng = np.random.default_rng(0)
    A = rng.random((10, 10)); A = A + A.T
    L = compute_normalized_laplacian(A)
    assert np.allclose(L, L.T, atol=1e-10)


def test_normalized_laplacian_eigs_in_0_2():
    rng = np.random.default_rng(1)
    A = rng.random((12, 12)); A = A + A.T
    eig = np.linalg.eigvalsh(compute_normalized_laplacian(A))
    assert eig.min() >= -1e-8 and eig.max() <= 2.0 + 1e-6


# ── real built data (per variant) ────────────────────────────────────────────

@pytest.mark.parametrize("v", VARIANTS)
def test_variant_shapes_and_keys(v):
    if not BUILT[v]:
        pytest.skip(f"{v} not built")
    n = _NUM_NODES[v]
    for s in ("train", "val", "test"):
        d = np.load(paths.split_npz(v, s))
        B = d["x"].shape[0]
        assert d["x"].shape == (B, 12, n, 1)
        assert d["y"].shape == (B, 12, n, 1)
        for k in ("x_tod", "y_tod", "x_mask", "y_mask"):
            assert d[k].shape == (B, 12, n, 1), f"{v}/{s}/{k}"
        assert d["x"].min() >= 0.0                      # raw values nonnegative
        assert d["x_tod"].min() >= 0.0 and d["x_tod"].max() < 1.0


@pytest.mark.parametrize("v", VARIANTS)
def test_gwn_native_view(v):
    if not BUILT[v]:
        pytest.skip(f"{v} not built")
    n = _NUM_NODES[v]
    for s in ("train", "val", "test"):
        nat = np.load(paths.gwn_native_dir(v) / f"{s}.npz")
        uni = np.load(paths.split_npz(v, s))
        B = uni["x"].shape[0]
        assert nat["x"].shape == (B, 12, n, 2) and nat["y"].shape == (B, 12, n, 2)
        # native ch0 == raw value, ch1 == tod
        assert np.allclose(nat["x"][..., 0:1], uni["x"])
        assert np.allclose(nat["x"][..., 1:2], uni["x_tod"])
    assert (paths.gwn_native_dir(v) / "adj_mx.pkl").exists()


@pytest.mark.parametrize("v", VARIANTS)
def test_scaler_train_only(v):
    if not BUILT[v]:
        pytest.skip(f"{v} not built")
    sc = json.loads(paths.scaler_json(v).read_text())
    assert sc["scope"] == "train_only"
    x = np.load(paths.split_npz(v, "train"))["x"]
    z = (x - sc["mean"]) / sc["std"]
    assert abs(float(z.mean())) < 1e-3
    assert abs(float(z.std()) - 1.0) < 1e-3


@pytest.mark.parametrize("v", VARIANTS)
def test_adjacency_loads_and_psd(v):
    if not BUILT[v]:
        pytest.skip(f"{v} not built")
    n = _NUM_NODES[v]
    A = load_adjacency(v, repair=False)
    assert A.shape == (n, n)
    assert A.min() >= 0.0
    eig = np.linalg.eigvalsh(compute_normalized_laplacian(A))
    assert eig.min() >= -1e-8


@pytest.mark.parametrize("v", VARIANTS)
def test_split_sizes_match_meta(v):
    if not BUILT[v]:
        pytest.skip(f"{v} not built")
    meta = json.loads((paths.processed_dir(v) / "meta.json").read_text())
    got = {s: int(np.load(paths.split_npz(v, s))["x"].shape[0])
           for s in ("train", "val", "test")}
    assert got["train"] == meta["n_train"]
    assert got["val"] == meta["n_val"]
    assert got["test"] == meta["n_test"]
    assert sum(got.values()) == meta["n_total_windows"]
