#!/usr/bin/env bash
# Push kernel dirs one after another, each as soon as a batch GPU slot is free
# (the account allows 2 concurrent sessions; other competitions' kernels share them).
# usage: bash tools/push_queue.sh kaggle/eval-base kaggle/eval-audit ...
#        QUOTA_WAIT=1 bash tools/push_queue.sh ...   # also wait out "weekly GPU quota reached"
#                                                    # (retries every 15 min for up to 24 h)
# A push counts as done only when Kaggle answers "successfully pushed". Slot-full, server
# and network errors are retried every 3 min, and every failed attempt is logged.
cd "$(dirname "$0")/.."
for d in "$@"; do
  ok=0
  for i in $(seq 1 480); do
    out=$(kaggle kernels push -p "$d" 2>&1 | tail -2 | tr '\n' ' ')
    if echo "$out" | grep -q "successfully pushed"; then
      echo "$(date -u +%m-%d\ %H:%M) $d: $out"
      ok=1
      break
    fi
    if echo "$out" | grep -q "weekly GPU quota"; then
      [ "${QUOTA_WAIT:-0}" = 1 ] || { echo "$(date -u +%m-%d\ %H:%M) $d: $out"; exit 1; }
      [ "$i" -gt 96 ] && { echo "$(date -u +%m-%d\ %H:%M) $d: quota still exhausted, giving up"; exit 1; }
      echo "$(date -u +%m-%d\ %H:%M) $d: waiting (quota)"
      sleep 900
    else
      echo "$(date -u +%m-%d\ %H:%M) $d: retry: $out"
      sleep 180
    fi
  done
  [ "$ok" = 1 ] || { echo "$(date -u +%m-%d\ %H:%M) $d: not pushed after all retries"; exit 1; }
done
