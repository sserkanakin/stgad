#!/usr/bin/env bash
# Download raw METR-LA and PEMS-BAY data from Zenodo and DCRNN repo.
# Use -c to resume interrupted downloads.

set -euo pipefail

RAW=data/raw
mkdir -p "$RAW"

download_if_missing() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "Already exists, skipping: $dest"
    else
        echo "Downloading $dest ..."
        wget -c -O "$dest" "$url"
    fi
}

# Traffic speed data (Zenodo mirror — same files as original DCRNN repo)
download_if_missing \
    "https://zenodo.org/records/5146275/files/METR-LA.h5" \
    "$RAW/METR-LA.h5"

download_if_missing \
    "https://zenodo.org/records/5146275/files/PEMS-BAY.h5" \
    "$RAW/PEMS-BAY.h5"

# Adjacency matrices (thresholded Gaussian kernel, normalized_k=0.1)
download_if_missing \
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl" \
    "$RAW/adj_mx_metrla.pkl"

download_if_missing \
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx_bay.pkl" \
    "$RAW/adj_mx_pemsbay.pkl"

echo "All raw data files present in $RAW/"
