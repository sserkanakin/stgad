#!/usr/bin/env bash
# resolve a SOFT gpu preference flag for Blackwell (rtx pro 6000).
#
# echoes "--prefer=<feature>" when ALL of these hold, else nothing (= any gpu):
#   - GPU_PREFER=blackwell (the default; set GPU_PREFER=any to force any-gpu)
#   - this host has sinfo + an sbatch that supports --prefer (SLURM >= 21.08)
#   - the cluster has a Blackwell node exposing a distinguishing feature
#
# --prefer is SOFT: jobs land on Blackwell when free and fall back to a40/l40 when not,
# so this can never block. it is NOT --nodelist (a node list forces ALL those nodes to be
# allocated -> 18-node request -> AssocMaxNodePerJobLimit; that was the earlier bug).
gpu_prefer_flag() {
  [ "${GPU_PREFER:-blackwell}" = "blackwell" ] || return 0
  command -v sinfo >/dev/null 2>&1 || return 0
  sbatch --help 2>&1 | grep -q -- '--prefer' || return 0    # older SLURM: skip, use any gpu
  local bn feat
  bn=$(sinfo -h -N -o '%N %G' 2>/dev/null | awk 'tolower($0) ~ /rtx_pro_6000/{print $1; exit}')
  [ -z "${bn}" ] && return 0
  feat=$(scontrol show node "${bn}" 2>/dev/null | tr ' ' '\n' \
         | sed -n 's/^A[a-z]*Features=//p' | tr ',' '\n' \
         | grep -iE 'rtx|blackwell|6000|sm_?120' | head -1)
  [ -n "${feat}" ] && printf -- '--prefer=%s' "${feat}"
}
