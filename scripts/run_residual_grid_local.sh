#!/usr/bin/env bash
# Run the backbone -> residuals grid LOCALLY (no slurm) — the laptop side of the
# laptop-vs-DAIC race. Same cells as scripts/daic/submit_residual_grid.sh, run
# sequentially with the project venv. Resumable: a cell whose residuals already
# exist is skipped, so re-running continues where an overnight run left off.
#
#   scripts/run_residual_grid_local.sh                       # full grid
#   DEVICE=mps EPOCHS=100 scripts/run_residual_grid_local.sh
#   DATASETS="pems08f pems08s" scripts/run_residual_grid_local.sh   # subset
#
# DEVICE: cpu (default, reliable) or mps (faster, GWN's adaptive-adj SVD falls
# back to CPU with a harmless warning). EPOCHS: backbone epochs (default 100).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PY:-.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-100}"
NHID="${NHID:-32}"
# small-N datasets first, so useful cells land before the night runs out.
DATASETS="${DATASETS:-pems08f pems08s metrla pems04f pems04s pemsbay}"
BACKBONES="${BACKBONES:-gwn gcn mlp}"
SEEDS="${SEEDS:-42 43 44}"

nodes_of() { case "$1" in metrla) echo 207 ;; pemsbay) echo 325 ;;
  pems04f|pems04s) echo 307 ;; pems08f|pems08s) echo 170 ;; esac; }
nullval_of() { case "$1" in metrla|pemsbay) echo 0.0 ;; *) echo nan ;; esac; }

n=0; done=0
for ds in ${DATASETS}; do
  N=$(nodes_of "$ds"); NV=$(nullval_of "$ds")
  NAT="data/processed/${ds}/gwn_native"
  for bb in ${BACKBONES}; do
    case "$bb" in
      gwn) TR=(--gcn_bool --addaptadj --adjtype doubletransition --gcn_order 2)
           RES=(--adjtype doubletransition --gcn_order 2) ;;
      gcn) TR=(--gcn_bool --adjtype symnadj --gcn_order 1)
           RES=(--adjtype symnadj --gcn_order 1 --no_addaptadj) ;;
      mlp) TR=(--adjtype identity)
           RES=(--adjtype identity --no_gcn --no_addaptadj) ;;
    esac
    for s in ${SEEDS}; do
      n=$((n+1))
      OUT="results/backbone/${ds}/${bb}/seed${s}"
      RES_OUT="results/residuals/${ds}/${bb}/seed${s}"
      if [ -f "${RES_OUT}/test.npz" ]; then
        echo "== [$n] ${ds}/${bb}/seed${s}  SKIP (residuals exist) =="; done=$((done+1)); continue
      fi
      echo "== [$n] ${ds}/${bb}/seed${s}  (N=${N} null_val=${NV} device=${DEVICE}) =="
      mkdir -p "${OUT}"
      MPLBACKEND=Agg "${PY}" third_party/Graph-WaveNet/train.py \
        --data "${NAT}" --adjdata "${NAT}/adj_mx.pkl" --num_nodes "${N}" \
        --in_dim 2 --seq_length 12 --device "${DEVICE}" --epochs "${EPOCHS}" \
        --nhid "${NHID}" --save "${OUT}/ckpt" --expid "${s}" --null_val "${NV}" "${TR[@]}"
      BEST="$(ls -t "${OUT}"/ckpt_exp"${s}"_best_*.pth 2>/dev/null | head -1 || true)"
      [ -n "${BEST}" ] || { echo "ERROR: no best ckpt for ${ds}/${bb}/seed${s}" >&2; exit 1; }
      cp -f "${BEST}" "${OUT}/best.pth"
      rm -f "${OUT}"/ckpt_*.pth        # drop per-epoch ckpts; keep best.pth
      "${PY}" scripts/gwn_extract_residuals.py --variant "${ds}" --seed "${s}" \
        --backbone "${bb}" --checkpoint "${OUT}/best.pth" --nhid "${NHID}" \
        --null_val "${NV}" "${RES[@]}"
      done=$((done+1))
      echo "   done ${done} cells so far"
    done
  done
done
echo "grid complete: ${done}/${n} cells have residuals."
