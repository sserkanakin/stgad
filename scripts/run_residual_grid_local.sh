#!/usr/bin/env bash
# Run the backbone -> residuals grid LOCALLY (no slurm), in PARALLEL, with
# per-cell foldered logs. The laptop side of the laptop-vs-DAIC race.
#
#   scripts/run_residual_grid_local.sh                          # full grid
#   JOBS=6 DEVICE=cpu EPOCHS=100 scripts/run_residual_grid_local.sh
#   BACKBONES="mlp gcn gwn" DATASETS="pems08f pems08s" scripts/run_residual_grid_local.sh
#
# JOBS    : concurrent cells (default ~= half the logical cores).
# DEVICE  : cpu (default; parallelises cleanly) or mps (1 GPU -> keep JOBS small).
# EPOCHS  : backbone epochs (default 100). Resumable: a cell whose residuals
#           already exist is skipped, so re-running continues where it left off.
#
# Per-cell logs (mirrors the DAIC layout):
#   logs/local/<dataset>/<backbone>/seed<seed>/{train.log,extract.log}
# A driver summary streams to this script's stdout (logs/local_grid_latest.log).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PY:-.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-100}"
NHID="${NHID:-32}"
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)"
JOBS="${JOBS:-$(( NCPU/2 > 2 ? NCPU/2 : 2 ))}"
THREADS="${THREADS:-$(( NCPU/JOBS > 1 ? NCPU/JOBS : 1 ))}"
# small-N datasets first, so useful cells land before the night runs out.
DATASETS="${DATASETS:-pems08f pems08s metrla pems04f pems04s pemsbay}"
BACKBONES="${BACKBONES:-mlp gcn gwn}"
SEEDS="${SEEDS:-42}"

nodes_of() { case "$1" in metrla) echo 207 ;; pemsbay) echo 325 ;;
  pems04f|pems04s) echo 307 ;; pems08f|pems08s) echo 170 ;; esac; }
nullval_of() { case "$1" in metrla|pemsbay) echo 0.0 ;; *) echo nan ;; esac; }

run_cell() {  # ds bb seed
  local ds=$1 bb=$2 s=$3
  local N NV NAT OUT RES_OUT LOGD
  N=$(nodes_of "$ds"); NV=$(nullval_of "$ds")
  NAT="data/processed/${ds}/gwn_native"
  OUT="results/backbone/${ds}/${bb}/seed${s}"
  RES_OUT="results/residuals/${ds}/${bb}/seed${s}"
  LOGD="logs/local/${ds}/${bb}/seed${s}"; mkdir -p "$LOGD" "$OUT"
  local TR RES
  case "$bb" in
    gwn) TR=(--gcn_bool --addaptadj --adjtype doubletransition --gcn_order 2)
         RES=(--adjtype doubletransition --gcn_order 2) ;;
    gcn) TR=(--gcn_bool --adjtype symnadj --gcn_order 1)
         RES=(--adjtype symnadj --gcn_order 1 --no_addaptadj) ;;
    mlp) TR=(--adjtype identity)
         RES=(--adjtype identity --no_gcn --no_addaptadj) ;;
  esac
  # cap torch/BLAS threads per cell so JOBS cells don't oversubscribe the cores.
  export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
         OPENBLAS_NUM_THREADS="$THREADS" MPLBACKEND=Agg
  echo "[start] ${ds}/${bb}/seed${s} (N=${N} null_val=${NV} dev=${DEVICE} thr=${THREADS}) -> ${LOGD}"
  if ! "${PY}" third_party/Graph-WaveNet/train.py \
        --data "${NAT}" --adjdata "${NAT}/adj_mx.pkl" --num_nodes "${N}" \
        --in_dim 2 --seq_length 12 --device "${DEVICE}" --epochs "${EPOCHS}" \
        --nhid "${NHID}" --save "${OUT}/ckpt" --expid "${s}" --null_val "${NV}" \
        "${TR[@]}" > "${LOGD}/train.log" 2>&1; then
    echo "[FAIL] ${ds}/${bb}/seed${s} train.py — see ${LOGD}/train.log"; return 1
  fi
  local BEST; BEST="$(ls -t "${OUT}"/ckpt_exp"${s}"_best_*.pth 2>/dev/null | head -1 || true)"
  [ -n "${BEST}" ] || { echo "[FAIL] ${ds}/${bb}/seed${s} no best ckpt"; return 1; }
  cp -f "${BEST}" "${OUT}/best.pth"; rm -f "${OUT}"/ckpt_*.pth
  if ! "${PY}" scripts/gwn_extract_residuals.py --variant "${ds}" --seed "${s}" \
        --backbone "${bb}" --checkpoint "${OUT}/best.pth" --nhid "${NHID}" \
        --null_val "${NV}" "${RES[@]}" > "${LOGD}/extract.log" 2>&1; then
    echo "[FAIL] ${ds}/${bb}/seed${s} extract — see ${LOGD}/extract.log"; return 1
  fi
  echo "[done] ${ds}/${bb}/seed${s} -> ${RES_OUT}"
}

echo "=== local grid: JOBS=${JOBS} device=${DEVICE} epochs=${EPOCHS} threads/cell=${THREADS} (ncpu=${NCPU}) ==="
total=0; launched=0
for ds in ${DATASETS}; do for bb in ${BACKBONES}; do for s in ${SEEDS}; do
  total=$((total+1))
  if [ -f "results/residuals/${ds}/${bb}/seed${s}/test.npz" ]; then
    echo "[skip] ${ds}/${bb}/seed${s} (residuals exist)"; continue
  fi
  # rolling pool: wait until a slot frees (portable; no `wait -n`).
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "${JOBS}" ]; do sleep 3; done
  run_cell "$ds" "$bb" "$s" &
  launched=$((launched+1))
done; done; done
wait
echo "=== grid finished: launched ${launched}/${total} cells. residuals: $(ls -d results/residuals/*/*/seed* 2>/dev/null | wc -l | tr -d ' ') dirs ==="
