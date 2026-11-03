#!/usr/bin/env bash
# Wait for the think-close smoke run to finish, check its log, and queue the full eval only
# if the loop guard worked on the GPU.
# Pass = "think-close fired: N recovered: M failed: 0" with N > 0 and M >= 0.8 N.
# usage: bash tools/chain_smoke_eval.sh   (run after the smoke kernel has been queued)
cd "$(dirname "$0")/.."
SMOKE=malikhammadfarooq/arc3-smoke-thinkclose
OUT=runs/smoke-thinkclose
stamp() { date -u +%m-%d\ %H:%M; }
for i in $(seq 1 288); do   # up to 24 h at 5 min
  st=$(kaggle kernels status "$SMOKE" 2>&1 | tail -1)
  case "$st" in
    *COMPLETE*|*ERROR*|*CANCEL*) break ;;
  esac
  sleep 300
done
echo "$(stamp) smoke status: $st"
mkdir -p "$OUT"
kaggle kernels output "$SMOKE" -p "$OUT" --file-pattern "serve\.log" > /dev/null 2>&1
line=$(grep -ho 'think-close fired: [0-9]* recovered: [0-9]* failed: [0-9]*' "$OUT"/*.log | tail -1)
echo "$(stamp) $line"
read -r n m f <<< "$(echo "$line" | grep -o '[0-9]\+' | tr '\n' ' ')"
if [ -n "$n" ] && [ "$n" -gt 0 ] && [ "$f" = 0 ] && [ $((m * 10)) -ge $((n * 8)) ]; then
  echo "$(stamp) smoke PASSED: queueing kaggle/eval-thinkclose"
  QUOTA_WAIT=1 bash tools/push_queue.sh kaggle/eval-thinkclose
else
  echo "$(stamp) smoke FAILED or inconclusive: eval NOT queued; read $OUT"
  exit 1
fi
