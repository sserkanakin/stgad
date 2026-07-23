#!/usr/bin/env bash
# check_quota.sh — how much room is left on the DAIC home shares.
#
# two 8G per-user shares matter for us:
#   linuxhome   (svm112, student_homes_linux) — code + the cu128 venv
#   windowshome (svm111, student_homes)        — data / residuals
#
# both are per-user NetApp volumes, so the numbers are identical whichever
# cluster you ssh into. two quirks decide how we read them:
#   - linuxhome IS reported by `quota -s` (from either cluster).
#   - windowshome is NOT exposed by the quota protocol at all, and its
#     ~/windowshome symlink only exists on the experimental cluster, so we
#     size it with `du` and compare against its documented 8G cap.
#
# usage: scripts/daic/check_quota.sh [ssh-host ...]
#        default hosts: daic-exp daic-exp-denis   (ssh aliases from ~/.ssh/config)

set -u

if [ "$#" -gt 0 ]; then
  HOSTS=("$@")
else
  HOSTS=(daic-exp daic-exp-denis)
fi

# remote probe: prints one "<share> <used_mb> <hard_mb>" line per share, or
# "<share> not-mounted-here". all the M/G/K -> MB conversion happens here so
# the local formatter just deals in integers.
read -r -d '' REMOTE <<'EOS'
netid=$(whoami)
win_hard_mb=8192   # windowshome is invisible to `quota`; assume its 8G cap.

# --- linuxhome: parsed from `quota -s` --------------------------------------
# the long NetApp filesystem name wraps onto its own line and the numbers land
# on the next line, so we latch on our student_homes_linux/<netid> row and read
# the following line. (quota stderr carries an unrelated stale-handle warning.)
quota -s 2>/dev/null | awk -v nid="$netid" '
  function mb(x,   u,n){ u=x; sub(/\*$/,"",u); n=u+0;
    if (u ~ /[Gg]$/) n=n*1024; else if (u ~ /[Kk]$/) n=n/1024; return int(n) }
  /student_homes_linux\// && $0 ~ ("/" nid "$") { pend=1; next }
  pend==1 { printf "linuxhome %d %d\n", mb($1), mb($3); pend=0 }
'

# --- windowshome: quota protocol is blind to svm111, so measure with du -----
# it's an autofs mount that stays empty until a path under it is accessed, so
# an `ls` first forces it to mount — otherwise du reports a bogus 0.
if [ -d "$HOME/windowshome" ]; then
  ls "$HOME/windowshome/" >/dev/null 2>&1
  # trailing slash: ~/windowshome is a symlink, and du without it measures the
  # link itself (0 MB) instead of following it into the share.
  used_mb=$(du -sm "$HOME/windowshome/" 2>/dev/null | cut -f1)
  printf "windowshome %s %s\n" "${used_mb:-0}" "$win_hard_mb"
else
  printf "windowshome not-mounted-here\n"
fi
EOS

for h in "${HOSTS[@]}"; do
  printf '\n== %s ==\n' "$h"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" "bash -s" <<<"$REMOTE" \
  | awk '
    $2 == "not-mounted-here" { printf "  %-12s (not mounted on this cluster)\n", $1; next }
    NF == 3 {
      used=$2; hard=$3; rem=hard-used; pct = (hard>0 ? used*100/hard : 0);
      flag = (rem <= 0 ? "  OVER QUOTA" : (pct >= 90 ? "  <-- nearly full" : ""));
      printf "  %-12s %5.2f / %4.1f GB used  (%5.1f%%),  %5.2f GB free%s\n", \
             $1, used/1024, hard/1024, pct, rem/1024, flag
    }'
done
echo
