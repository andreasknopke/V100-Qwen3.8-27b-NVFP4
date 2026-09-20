#!/usr/bin/env bash
# ============================================================================
#  Start exactly ONE ninfer-serve, with guards.
#
#  Why: a restart race once left TWO servers running, both loading ~20 GiB onto
#  a 32 GiB card. VRAM hit ~32 GiB, both stalled at ~45% and NEITHER served a
#  single request (port LISTENing, zero requests answered - silent failure).
#
#  Guards:
#    1. kill every instance and WAIT until none remain
#    2. wait until VRAM is actually released (< 2 GiB used)
#    3. refuse to start if VRAM is still held
#    4. verify exactly one process afterwards
#
#  Usage: ./start_single.sh [CTX] [PORT] [THINKING] [CONCURRENCY]
#  Logs to $NINFER_HOME/ninfer.log.
# ============================================================================
export PATH="/usr/local/cuda-12.8/bin:$PATH"
NINFER_HOME="${NINFER_HOME:-$HOME/ninfer}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CTX="${1:-212992}"
PORT="${2:-8084}"
THINKING="${3:-1}"
CONCURRENCY="${4:-3}"
LOG="$NINFER_HOME/ninfer.log"

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }

echo "=== phase 1: stop existing instances ==="
for attempt in 1 2 3; do
    pids=$(pgrep -f '[a]pps/ninfer-serve' || true)   # bracket trick: do not match this script
    [ -z "$pids" ] && { echo "  none running"; break; }
    echo "  attempt $attempt: killing $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 4
done
remaining=$(pgrep -f '[a]pps/ninfer-serve' || true)
if [ -n "$remaining" ]; then echo "  SIGKILL: $remaining"; kill -9 $remaining 2>/dev/null || true; sleep 4; fi

echo "=== phase 2: wait for VRAM release ==="
for i in $(seq 1 30); do
    used=$(vram_used); echo "  [$i] VRAM used: ${used} MiB"
    [ "${used:-0}" -lt 2048 ] && { echo "  released"; break; }
    sleep 3
done
remaining=$(pgrep -f '[a]pps/ninfer-serve' || true)
[ -n "$remaining" ] && { echo "ABORT: instances still alive: $remaining"; exit 1; }
used=$(vram_used)
if [ "${used:-0}" -ge 4096 ]; then
    echo "ABORT: VRAM still ${used} MiB used - something else holds the GPU:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null
    exit 1
fi

echo "=== phase 3: start one instance ==="
: > "$LOG"
nohup bash "$HERE/serve.sh" "$CTX" mtp 3 1 "$PORT" "$THINKING" "$CONCURRENCY" > "$LOG" 2>&1 &

echo "=== phase 4: wait for readiness ==="
for i in $(seq 1 80); do
    if curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "  ready after ~$((i*3))s"; break
    fi
    sleep 3
    if [ "$i" -eq 80 ]; then echo "TIMEOUT waiting for readiness"; tail -20 "$LOG"; exit 1; fi
done

echo "=== phase 5: verify single instance ==="
count=$(pgrep -c -f '[a]pps/ninfer-serve' || echo 0)
echo "  ninfer-serve process count: $count"
[ "$count" != "1" ] && { echo "WARNING: expected exactly 1, found $count"; pgrep -af '[a]pps/ninfer-serve' | cut -c1-100; }
echo "  VRAM used: $(vram_used) MiB"
grep -E 'capacity|listening|thinking' "$LOG" | head -5
echo "START_SINGLE_DONE"
