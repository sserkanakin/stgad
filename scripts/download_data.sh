#!/usr/bin/env bash
# Download every raw source file, from the CANONICAL location, and verify identity.
#
# Provenance re-verified 2026-07-16. Two things the previous version got wrong:
#   * it fetched the .h5 from Zenodo record 5724362 -- that record is
#     "PEMS-BAY and METR-LA in csv" (a third-party CSV conversion by S. Kwak, EPFL)
#     and serves NO .h5 at all. Both URLs 404. There is no metr-la.h5 on Zenodo
#     anywhere; the DCRNN Google Drive folder is the only source.
#   * PEMS04/08 were credited to ASTGNN. The bytes do live in ASTGNN, but the
#     datasets were introduced by Guo et al., ASTGCN, AAAI 2019 -- cite that.
#
# Identity anchors. No project publishes a checksum for metr-la.h5, PEMS04.npz or
# PEMS08.npz, so we pin the git blob SHA-1 (verified identical across DCRNN /
# MTGNN / GTS / DCRNN_PyTorch / DL-Traff-Graph, and across ASTGCN <-> ASTGNN).
# pems-bay.h5 is the one file with a published md5 (Zenodo 4263971).
#
# usage:  bash scripts/download_data.sh
set -euo pipefail

RAW=data/raw
PY=${PY:-/Users/serkan/miniconda/envs/tud-data/bin/python}   # has gdown
mkdir -p "$RAW"/{metrla,pemsbay,pems04,pems08,molene}

pass=0; fail=0

# verify a file against an expected size and (blob sha | md5). empty expected = record only.
verify() {
    local f="$1" want_size="$2" kind="$3" want_hash="$4"
    local got_size got_hash
    got_size=$(wc -c < "$f" | tr -d ' ')
    if [ "$kind" = md5 ]; then got_hash=$(md5 -q "$f"); else got_hash=$(git hash-object "$f"); fi
    if [ -n "$want_size" ] && [ "$got_size" != "$want_size" ]; then
        echo "  FAIL $(basename "$f"): size $got_size != $want_size"; fail=$((fail+1)); return
    fi
    if [ -n "$want_hash" ] && [ "$got_hash" != "$want_hash" ]; then
        echo "  FAIL $(basename "$f"): $kind $got_hash != $want_hash"; fail=$((fail+1)); return
    fi
    echo "  OK   $(basename "$f")  $got_size bytes  $kind=$got_hash"
    pass=$((pass+1))
}

get() {  # url dest
    [ -f "$2" ] && { echo "  have $(basename "$2")"; return 0; }
    curl -fsSL --retry 3 --max-time 900 -o "$2" "$1"
}

echo "=== 1/4  Sensor graphs (DCRNN; verbatim copies everywhere) ==="
# python-2 pickles -> load with encoding='latin1'. (sensor_ids, sensor_id_to_ind, adj_mx)
get "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl" \
    "$RAW/metrla/adj_mx_metrla.pkl"
verify "$RAW/metrla/adj_mx_metrla.pkl" 680459 blob a15a5904a83e7f13fe94d32fcd45c00f163ffd97
get "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx_bay.pkl" \
    "$RAW/pemsbay/adj_mx_pemsbay.pkl"
verify "$RAW/pemsbay/adj_mx_pemsbay.pkl" 1681480 blob 6651951a08ab101c818b1f27ba9a908d3ad56539

echo "=== 2/4  Speed series -- METR-LA (Drive only) / PEMS-BAY (Zenodo, checksummed) ==="
# METR-LA: no Zenodo mirror exists. DCRNN Drive folder 10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX.
# 57,038,056 = uncompressed float64 DataFrame (34272 x 207 x 8 + index). The ~23MB
# figure sometimes quoted is the ZIPPED distribution, not this file.
if [ ! -f "$RAW/metrla/metr-la.h5" ]; then
    # gdown >= 5 dropped --id (positional id now). It also needs its cookie cache to
    # exist; HOME is redirected when ~/.cache is not writable.
    GD_HOME="${GDOWN_HOME:-$HOME}"
    mkdir -p "$GD_HOME/.cache/gdown" 2>/dev/null && : > "$GD_HOME/.cache/gdown/cookies.txt" 2>/dev/null || true
    HOME="$GD_HOME" "$PY" -m gdown 1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC -O "$RAW/metrla/metr-la.h5" || {
        echo "  gdown failed -- fetch manually from"
        echo "  https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
        echo "  and place metr-la.h5 at $RAW/metrla/"; }
fi
# blob verified 2026-07-16: the Drive copy is byte-identical to the copies vendored
# in GCGRNN and DL-Traff-Graph, which confirms this file id is the canonical METR-LA.
verify "$RAW/metrla/metr-la.h5" 57038056 blob 7df0003054d1bccd255d38949b2ef011dcadad3f

# PEMS-BAY: Zenodo 4263971 is the ONLY .h5 with a published checksum.
get "https://zenodo.org/records/4263971/files/pems-bay.h5" "$RAW/pemsbay/pems-bay.h5"
verify "$RAW/pemsbay/pems-bay.h5" 135930936 md5 bfa47e6ee7cc2d665e9b62c0c95c0b41

echo "=== 3/4  Flow/speed series -- PEMS04, PEMS08 (Guo et al., ASTGCN, AAAI 2019) ==="
# bytes re-hosted by ASTGNN; byte-identical to ASTGCN's originals.
# .npz is (T,N,3). Channel order [flow, occupancy, speed] -- ASSERTED, NOT assumed:
# verify_datasets.py checks it by value range (occupancy in [0,1], speed ~[0,70]).
# The ASTGCN paper's prose says "flow, speed, occupancy" -- that sentence is wrong
# and is the likely origin of the widespread mis-citation.
for d in PEMS04 PEMS08; do
    low=$(echo "$d" | tr 'A-Z' 'a-z')
    get "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data/$d/$d.npz" "$RAW/$low/$d.npz"
    get "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data/$d/$d.csv" "$RAW/$low/$d.csv"
done
verify "$RAW/pems04/PEMS04.npz" 32956284 blob a1a39eac4b86963b1ac22874fd1cc693eb642cd2
verify "$RAW/pems04/PEMS04.csv" 4852     blob 2569623e29fa9685ae230246a20f31741ee6cf01
verify "$RAW/pems08/PEMS08.npz" 18525419 blob f5f3e84a00417573aa8487fe1e9e8cd13c6026c8
verify "$RAW/pems08/PEMS08.csv" 4032     blob cf11bea3ff4229d2d900fdcf1085828d25429c88

echo "=== 4/4  Molene (Girault's repo; the source GAD cites) ==="
# 55 stations x 744 hourly steps, Jan 2014. temperature 't' in KELVIN, missing = "mq".
# 37 stations have complete 't' (= GAD's N); only 32 of those have coordinates,
# so a geographic graph is buildable on 32 -- see DATA_README.
get "https://raw.githubusercontent.com/bgirault-usc/Molene-Dataset/master/Original_Data/aggregated_data.csv" \
    "$RAW/molene/aggregated_data.csv"
get "https://raw.githubusercontent.com/bgirault-usc/Molene-Dataset/master/Original_Data/weather_stations.csv" \
    "$RAW/molene/weather_stations.csv"
verify "$RAW/molene/weather_stations.csv" 2820 blob 2d7b2f8682a1f374815ad4bf70a967e48a810598
verify "$RAW/molene/aggregated_data.csv" 7242602 blob c431080c25403f594a54b8437702490b9d108458

echo
echo "=== $pass verified, $fail failed ==="
[ "$fail" -eq 0 ] || { echo "IDENTITY CHECK FAILED -- do not build on these bytes."; exit 1; }
echo "next: \$PY scripts/build_datasets.py --all && \$PY scripts/verify_datasets.py"
