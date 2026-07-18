"""Build the Molene dataset: the graph-SMOOTH positive control.

Molene is not a forecasting benchmark. It is GAD's own dataset (Rozada et al.
2510.05036) at GAD's own scale, and it plays one role in this thesis: the
positive control for "when does a graph-structured prior help". Traffic
residuals are only 4-10% graph-explained; Molene temperature is ~91% low-graph-
frequency energy, so it is the regime where the structured prior should win.
It anchors the reframed claim -- "the prior helps when the signal is genuinely
graph-smooth, and traffic residuals aren't" -- rather than "our prior never helps".

Task shape: UNCONDITIONAL graph-signal generation. Each of the 744 hourly
snapshots is one graph signal x in R^N; there is no history/horizon and no
trained backbone. Output is (T, N), not (B,12,N,1).

Source (canonical, = the file GAD cites): Girault's repo
  https://github.com/bgirault-usc/Molene-Dataset  Original_Data/
  - aggregated_data.csv : 55 stations x 744 hourly steps, Jan 2014.
    temperature 't' in KELVIN; missing encoded as the string "mq".
  - weather_stations.csv : station coordinates (49 of the 55 listed).

Station selection (verified against both the CSV and the literature):
  - 37 stations have COMPLETE 't' across all 744 steps  = GAD's N=37.
  - only 32 of those have coordinates                    = GSPBox's N=32.
  We use N=32: the geographic graph is the entire point of this arm, and it
  cannot be built for the 5 coordinate-less stations. GAD's graph is therefore
  NOT reproducible on all 37 from the canonical source -- a deviation we state
  explicitly. (GAD also calls these "daily average temperatures"; their own
  670+74=744 counts show they use the hourly snapshots -- daily would give T=31.)

Graph (GSPBox / Perraudin recipe): geographic kNN, k=5, Gaussian weights
  w_ij = exp(-d_ij^2 / sigma^2), sigma = mean kNN distance; distance combines
  horizontal (haversine, km) with elevation amplified x5 (GSP convention);
  symmetrized (union), self-loops removed. Verified single-component.

Centering: per-station mean removed (GSP convention; also removes the static
sensor-mean heterogeneity that would otherwise masquerade as low-graph-frequency
energy). Mean is fit on the TRAIN split only.

Split: 670 train / 74 test (GAD's 90/10 on the 744 snapshots). No val (GAD has none).

Usage:  PY=/Users/serkan/miniconda/envs/tud-data/bin/python
        $PY scripts/build_molene.py
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "molene"
OUT = ROOT / "data" / "processed" / "molene"
N_TRAIN = 670           # GAD's 90/10 on 744 snapshots
K_NN = 5                # GSPBox kNN
ELEV_AMP = 5.0          # GSP elevation amplification
EARTH_R = 6371.0        # km


def _dms_to_deg(s: str) -> float:
    """'48°51'18N' / '3°00'12W' -> signed decimal degrees."""
    m = re.match(r"\s*(\d+)[^\d]+(\d+)[^\d]+(\d+)\s*([NSEW])", s)
    if not m:
        raise ValueError(f"bad DMS: {s!r}")
    d, mm, ss, hemi = int(m[1]), int(m[2]), int(m[3]), m[4]
    val = d + mm / 60.0 + ss / 3600.0
    return -val if hemi in ("S", "W") else val


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_signals():
    """(T, N) temperature (Kelvin), station ids, timestamps -- 32 complete+coord stations."""
    df = pd.read_csv(RAW / "aggregated_data.csv", dtype={"numer_sta": str})
    df["t"] = pd.to_numeric(df["t"], errors="coerce")            # "mq" -> NaN
    piv = df.pivot_table(index="date", columns="numer_sta", values="t", aggfunc="first")
    piv = piv.sort_index()                                       # chronological

    complete = piv.columns[piv.notna().all(axis=0)]             # 37 with full 't'
    st = pd.read_csv(RAW / "weather_stations.csv", dtype={"Numéro": str})
    st.columns = [c.strip() for c in st.columns]
    coords = st.set_index("Numéro")
    keep = [s for s in complete if s in coords.index]           # 32 with coords
    keep = sorted(keep)

    X = piv[keep].to_numpy(dtype=np.float64)                     # (744, 32) Kelvin
    lat = np.array([_dms_to_deg(coords.loc[s, "Latitude"]) for s in keep])
    lon = np.array([_dms_to_deg(coords.loc[s, "Longitude"]) for s in keep])
    alt = coords.loc[keep, "Altitude"].astype(float).to_numpy() / 1000.0   # km
    ts = piv.index.to_numpy()
    return X, keep, lat, lon, alt, ts


def geographic_graph(lat, lon, alt):
    """GSPBox kNN(k=5) gaussian graph on (horizontal haversine + 5x elevation)."""
    n = len(lat)
    la = np.radians(lat)[:, None]
    lo = np.radians(lon)[:, None]
    dlat = la - la.T
    dlon = lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    d_horiz = 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))            # km
    d_elev = ELEV_AMP * np.abs(alt[:, None] - alt[None, :])                # km, amplified
    D = np.sqrt(d_horiz ** 2 + d_elev ** 2)

    sigma = float(D[D > 0].reshape(n, -1).min(axis=1).mean())  # mean nearest-neighbor dist... use mean kNN below
    # kNN adjacency (directed), then union-symmetrize
    A = np.zeros((n, n))
    knn_d = []
    for i in range(n):
        order = np.argsort(D[i])
        nbrs = order[order != i][:K_NN]
        knn_d.extend(D[i, nbrs].tolist())
        A[i, nbrs] = 1.0
    sigma = float(np.mean(knn_d))                              # GSPBox: sigma = mean kNN distance
    W = np.exp(-(D ** 2) / (sigma ** 2))
    A = np.maximum(A, A.T) * W                                 # keep kNN edges, gaussian weight, symmetric
    np.fill_diagonal(A, 0.0)
    return A.astype(np.float64), sigma


def normalized_laplacian(A):
    A = np.asarray(A, dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    A_sym = np.maximum(A, A.T)
    d = A_sym.sum(1)
    inv = np.zeros_like(d)
    nz = d > 1e-12
    inv[nz] = 1.0 / np.sqrt(d[nz])
    L = np.eye(len(d)) - (inv[:, None] * A_sym) * inv[None, :]
    L = 0.5 * (L + L.T)
    return A_sym.astype(np.float32), L.astype(np.float64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, ids, lat, lon, alt, ts = load_signals()
    T, N = X.shape
    print(f"molene: T={T} snapshots x N={N} stations (complete 't' + coords)")
    print(f"  temperature (K): min={X.min():.1f} max={X.max():.1f} mean={X.mean():.1f}")

    # graph
    A, sigma = geographic_graph(lat, lon, alt)
    A_sym, L = normalized_laplacian(A)
    ncomp, _ = connected_components(csr_matrix(A_sym > 0), directed=False)
    lam = np.linalg.eigvalsh(L)
    nzero = int((lam < 1e-8).sum())
    deg = (A_sym > 0).sum(1)
    ok = ncomp == 1 and int((deg == 0).sum()) == 0 and nzero == 1
    print(f"  graph: k={K_NN} sigma={sigma:.1f}km edges={int((A_sym>0).sum()//2)} "
          f"deg[{deg.min()},{deg.max()}] comp={ncomp} zero_eig={nzero} "
          f"lam_max={lam.max():.3f}  {'PASS' if ok else 'FAIL -- graph not single-component'}")
    if not ok:
        raise SystemExit("molene graph failed the connectivity gate")

    # per-station centering, train-mean only (GSP convention)
    mean = X[:N_TRAIN].mean(axis=0)                            # (N,) per-station train mean
    Xc = (X - mean[None, :]).astype(np.float32)

    # graph-smoothness diagnostic (the reason this dataset exists)
    Vs = np.linalg.eigh(L)[1]
    C = Xc @ Vs                                                # (T, N) graph-fourier coeffs
    energy = (C ** 2).mean(0)
    order = np.argsort(lam)
    low = energy[order][: max(1, N // 10)].sum() / energy.sum()
    centroid = float((lam * energy).sum() / energy.sum())
    print(f"  smoothness: energy in lowest 10% of modes = {low:.1%}  "
          f"spectral centroid = {centroid:.3f} (white=1.0)")

    np.savez_compressed(OUT / "signals.npz",
                        X=X.astype(np.float32),                # (T,N) raw Kelvin
                        Xc=Xc,                                 # (T,N) per-station centered
                        station_mean=mean.astype(np.float32),  # (N,) train mean
                        train_idx=np.arange(N_TRAIN),
                        test_idx=np.arange(N_TRAIN, T))
    np.savez_compressed(OUT / "adj.npz", A=A.astype(np.float32), A_sym=A_sym, L=L)

    meta = {
        "variant": "molene", "N": N, "T": T, "modality": "temperature", "units": "kelvin",
        "task": "unconditional graph-signal generation (no backbone, no horizon)",
        "role": "graph-smooth positive control for the structured prior",
        "split": {"train": N_TRAIN, "test": T - N_TRAIN, "rule": "GAD 90/10 on snapshots, no val"},
        "station_ids": ids, "lat": lat.tolist(), "lon": lon.tolist(), "alt_km": alt.tolist(),
        "graph": {"type": "geographic knn gaussian", "k": K_NN, "sigma_km": sigma,
                  "elevation_amplification": ELEV_AMP, "components": int(ncomp),
                  "zero_eig": nzero, "lam_max": float(lam.max())},
        "centering": "per-station mean, train-only (GSP convention)",
        "smoothness": {"low_freq_energy_frac": float(low), "spectral_centroid": centroid},
        "source": {
            "repo": "https://github.com/bgirault-usc/Molene-Dataset",
            "cites": "Girault et al. 2017 (GAD's citation); GSPBox/Perraudin graph recipe",
            "aggregated_data.csv_sha256": _sha256(RAW / "aggregated_data.csv"),
            "weather_stations.csv_sha256": _sha256(RAW / "weather_stations.csv"),
        },
        "deviations": [
            "N=32 (complete-t AND has-coords), not GAD's 37: 5 complete-t stations "
            "have no coordinates so the geographic graph is not buildable on them.",
            "hourly snapshots (T=744), not 'daily averages' -- GAD's 670+74=744 counts "
            "confirm hourly; daily would give T=31.",
        ],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote signals.npz, adj.npz, meta.json -> {OUT}")


if __name__ == "__main__":
    main()
