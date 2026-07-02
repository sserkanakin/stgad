#!/bin/bash
# point the big artefact dirs at the data share (windowshome) and MIGRATE any real
# content that's currently on the home share (linuxhome), so the 8 GB code quota is
# freed and both interactive runs and sbatch jobs resolve to the same place.
#
# the sbatch jobs symlink results/* → $DAIC_DATA_DIR via _common.sh, but that relink
# is a no-op if the dir is already a POPULATED REAL dir (guard: `[ -L ] || [ ! -e ]`).
# so if you ever wrote artefacts before the symlinks existed (e.g. build_signal_cache
# on the login node), they sit on linuxhome and fill the quota. this migrates them.
#
#   source scripts/daic/env.sh        # or: source scripts/daic/activate.sh
#   bash scripts/daic/link_artifacts.sh
#
# idempotent + safe: a symlink is just repointed; a real dir is rsync'd to the data
# share and only removed AFTER the copy succeeds; a missing dir is just linked.
set -euo pipefail

if [ -z "${DAIC_DATA_DIR:-}" ]; then
  # shellcheck source=/dev/null
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
fi
: "${DAIC_DATA_DIR:?set DAIC_DATA_DIR (source scripts/daic/env.sh)}"

echo "[link_artifacts] data share: ${DAIC_DATA_DIR}"
mkdir -p "${DAIC_DATA_DIR}"/{raw,processed,backbone,diffusion,residuals,eval}

_migrate() {
  local link="$1" target="$2"
  mkdir -p "${target}"
  if [ -L "${link}" ]; then
    ln -sfn "${target}" "${link}"                       # already a symlink → repoint
    echo "  linked   ${link} -> ${target}"
  elif [ -d "${link}" ]; then
    echo "  migrating ${link} -> ${target} (moving existing content off linuxhome) ..."
    # windowshome is a windows/CIFS share: it can't set posix perms/owner/times and
    # rejects rsync's temp-file+rename, so skip all of that and write in place.
    rsync -rl --inplace --no-perms --no-owner --no-group --no-times --omit-dir-times \
      "${link}/" "${target}/"
    rm -rf "${link}"
    ln -sfn "${target}" "${link}"
    echo "  migrated ${link} -> ${target}"
  else
    ln -sfn "${target}" "${link}"                       # missing → just link
    echo "  linked   ${link} -> ${target}"
  fi
}

# project-relative dirs (run from the project root)
cd "${PROJECT_DIR:-${DAIC_PROJECT_DIR:-$PWD}}"
mkdir -p data results
_migrate data/raw          "${DAIC_DATA_DIR}/raw"
_migrate data/processed    "${DAIC_DATA_DIR}/processed"
_migrate results/backbone  "${DAIC_DATA_DIR}/backbone"
_migrate results/residuals "${DAIC_DATA_DIR}/residuals"
_migrate results/diffusion "${DAIC_DATA_DIR}/diffusion"
_migrate results/eval      "${DAIC_DATA_DIR}/eval"

echo "[link_artifacts] done. artefact dirs now live on the data share; linuxhome freed."
