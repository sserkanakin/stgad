#!/usr/bin/env bash
# pin the Blackwell (RTX Pro 6000, 96 GB, nodes gpu36-53) GPUs when GPU_PREFER=blackwell
# (the default). the DAIC docs expose the gres TYPE name `nvidia_rtx_pro_6000` and there
# are NO node feature tags, so a soft `--prefer=<feature>` never matched -> jobs fell back
# to a40. a gres-type pin is the reliable way (https://doc.daic.tudelft.nl/docs/hardware/gpus/).
#
# this is a HARD pin: jobs queue (PD, Resources) if all Blackwell are busy rather than
# dropping to a40/l40. set GPU_PREFER=any to disable (cluster default any-gpu gres), or
# GPU_GRES=gpu:<name>:1 to pin a different type. NOT --nodelist (a node list forces ALL
# listed nodes -> AssocMaxNodePerJobLimit; that was the earlier bug).

# echo the gres string for Blackwell (e.g. "gpu:nvidia_rtx_pro_6000:1"), or nothing if
# GPU_PREFER!=blackwell or the cluster doesn't expose that gres.
gpu_blackwell_gres() {
  [ -n "${GPU_GRES:-}" ] && { printf '%s' "${GPU_GRES}"; return 0; }
  [ "${GPU_PREFER:-blackwell}" = "blackwell" ] || return 0
  command -v sinfo >/dev/null 2>&1 || return 0
  sinfo -h -o '%G' 2>/dev/null | grep -qi 'rtx_pro_6000' || return 0
  printf 'gpu:nvidia_rtx_pro_6000:1'
}

# back-compat wrapper: callers that appended a flag get a full `--gres=` override (the last
# --gres on an sbatch line wins, so this overrides the wrapper's default any-gpu gres).
gpu_prefer_flag() {
  local g; g="$(gpu_blackwell_gres)" || return 0
  [ -n "$g" ] && printf -- '--gres=%s' "$g"
}
