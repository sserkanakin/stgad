#!/usr/bin/env bash
# Download raw METR-LA and PEMS-BAY data.
# Adjacency matrices come from the DCRNN GitHub repo directly.
# Traffic h5 files come from Zenodo (record 5724362) or manual download.
#
# If the h5 downloads fail, download manually from DCRNN Google Drive:
#   https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX
# and place METR-LA.h5 and PEMS-BAY.h5 in data/raw/.

set -euo pipefail

RAW=data/raw
mkdir -p "$RAW"

download_if_missing() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "  Already exists, skipping: $dest"
    else
        echo "  Downloading $dest ..."
        # --show-progress requires wget >= 1.16; DAIC's wget is older.
        # Plain `wget -c -O` works everywhere and shows enough output.
        wget -c -O "$dest" "$url"
    fi
}

echo "=== Adjacency matrices (DCRNN repo) ==="
download_if_missing \
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl" \
    "$RAW/adj_mx_metrla.pkl"

download_if_missing \
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx_bay.pkl" \
    "$RAW/adj_mx_pemsbay.pkl"

echo "=== Traffic speed data (Zenodo record 5724362) ==="
download_if_missing \
    "https://zenodo.org/records/5724362/files/METR-LA.h5" \
    "$RAW/metr-la.h5"

download_if_missing \
    "https://zenodo.org/records/5724362/files/PEMS-BAY.h5" \
    "$RAW/pems-bay.h5"

echo ""
echo "=== Checking files ==="
for f in "$RAW/metr-la.h5" "$RAW/pems-bay.h5" "$RAW/adj_mx_metrla.pkl" "$RAW/adj_mx_pemsbay.pkl"; do
    if [ -f "$f" ]; then
        echo "  OK  $f  ($(du -h "$f" | cut -f1))"
    else
        echo "  MISSING  $f"
    fi
done

echo ""
echo "=== Backbone repo (Phase 2) ==="
mkdir -p third_party

if [ ! -d "third_party/Graph-WaveNet" ]; then
    echo "  Cloning Graph-WaveNet ..."
    git clone https://github.com/nnzhan/Graph-WaveNet.git third_party/Graph-WaveNet
    cd third_party/Graph-WaveNet && git checkout 6b162e8 && cd ../..
    echo "  Pinned Graph-WaveNet to 6b162e8"
else
    echo "  Already exists, skipping: third_party/Graph-WaveNet"
fi

echo ""
echo "=== All done. Run: python scripts/preprocess.py --dataset metrla ==="
