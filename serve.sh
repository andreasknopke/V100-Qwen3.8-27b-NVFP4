#!/usr/bin/env bash
# ============================================================================
#  NInfer server launcher - Qwen3.8-27B NVFP4 on a V100 (sm_70).
#
#  Usage: ./serve.sh [CTXSIZE] [SPEC] [DRAFT] [VISION] [PORT] [THINKING] [CONCURRENCY]
#    CTXSIZE      logical context ceiling   (default 212992)
#    SPEC         none | mtp | dflash2      (default mtp)
#    DRAFT        draft window 1-7 (mtp) / 1-15 (dflash2)   (default 3)
#    VISION       1 | 0                     (default 1)
#    PORT         HTTP port                 (default 8084)
#    THINKING     1 | 0                     (default 1 = thinking ON, low effort)
#    CONCURRENCY  parallel requests         (default 1 - see WHY below)
#
#  WHY 212992 AND NOT 262144:
#    NVFP4 weights occupy ~20.0 GiB on the 32 GiB card (attention/GDN/output
#    head/embedding and the last MLP layers are row-scaled FP8, not FP4). At
#    262144 the engine needs ~12.0 GiB runtime reservation but only ~10.55 GiB
#    is free after weights -> it refuses to load. 212992 is the measured safe
#    ceiling. If you want the full 262144 context, use the groupwise-int
#    artifact instead (16.9 GiB weights), not NVFP4.
#
#  WHY CONCURRENCY 1:
#    Concurrency N reserves N active + N cached device-state slots (2N slots),
#    whose MINIMUM reservation is a fixed cost independent of --max-context
#    (decoder state + GDN state image + CUDA-graph allowance). With only ~9.55
#    GiB of runtime budget left after the 20 GiB weights, 2 slots (~9.0 GiB)
#    fit but 6 slots (~10.0 GiB) do NOT - the engine aborts startup even at a
#    reduced context, because the binding constraint is the fixed minimum, not
#    the per-token KV growth. Lowering the context cannot help.
#
#  THINKING: the bundled chat template exposes low/medium/xhigh (NO "minimal"),
#  and ninfer-serve has no --reasoning-effort flag, so the effort default was
#  patched to Low in source (patch_default_low_effort.sh). THINKING=1 therefore
#  means "thinking on, low effort". No thinking budget cap is applied on
#  purpose: a cap can truncate reasoning and leave an EMPTY completion.
#  IMPORTANT for clients: reasoning shares the output budget, so the client's
#  max_tokens (or --default-max-tokens below) must comfortably exceed the
#  reasoning length, or thinking-on requests return empty content.
#
#  Env overrides: NINFER_HOME, NINFER_MODEL, EXTRA_ARGS, DEFAULT_MAX_TOKENS,
#                 CONCURRENCY, MAX_PENDING_REQUESTS, PENDING_TIMEOUT_MS,
#                 NINFER_TCP_USER_TIMEOUT_MS (see the TCP patch).
# ============================================================================
set -uo pipefail

CTX="${1:-212992}"
SPEC="${2:-mtp}"
DRAFT="${3:-3}"
VISION="${4:-1}"
PORT="${5:-8084}"
THINKING="${6:-1}"
# NVFP4 runs SINGLE-REQUEST by default: its 20.0 GiB weights leave only ~9.55 GiB
# of runtime budget, and concurrency 3 needs ~10.0 GiB minimum reservation
# (6 device-state slots) -> the engine refuses to start even at reduced context.
# Only concurrency 1 fits above ~196k. Override with arg 7 or CONCURRENCY env.
CONCURRENCY="${7:-${CONCURRENCY:-1}}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-32768}"
MAX_PENDING_REQUESTS="${MAX_PENDING_REQUESTS:-16}"
PENDING_TIMEOUT_MS="${PENDING_TIMEOUT_MS:-600000}"

NINFER_HOME="${NINFER_HOME:-$HOME/ninfer}"
MODEL="${NINFER_MODEL:-$NINFER_HOME/models/qwen3_8_27b_nvfp4.ninfer}"
BIN="$NINFER_HOME/build-v100/apps/ninfer-serve"

if [ ! -x "$BIN" ]; then
    echo "FATAL: $BIN not found or not executable."
    echo "Build it first:  bash scripts/build_ninfer.sh"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "FATAL: artifact not found: $MODEL"
    echo "Download it first:  bash scripts/download_model.sh"
    exit 1
fi

ARGS=(
    --host 0.0.0.0
    --port "$PORT"
    --max-context "$CTX"
    --prefill-chunk 2048
    --kv-capacity auto
    --max-concurrency "$CONCURRENCY"
    --device-state-slots "$CONCURRENCY"
    --max-pending-requests "$MAX_PENDING_REQUESTS"
    --pending-timeout-ms "$PENDING_TIMEOUT_MS"
    --kv-dtype int8
    --host-state-slots 8
    --host-kv-mib 8192
    --default-max-tokens "$DEFAULT_MAX_TOKENS"
)

case "${SPEC,,}" in
    mtp)     ARGS+=(--spec mtp --draft-tokens "$DRAFT" --lm-head-draft) ;;
    dflash2) ARGS+=(--spec dflash2 --draft-tokens "$DRAFT") ;;
    none|"") : ;;   # "--spec none" is NOT valid; baseline = omit --spec entirely
    *)       echo "FATAL: unknown spec '$SPEC' (use none|mtp|dflash2)"; exit 1 ;;
esac

[ "$VISION" = "1" ] && ARGS+=(--vision)

if [ "$THINKING" = "1" ]; then
    ARGS+=(--preserve-thinking)
    THINK_DESC="on / low effort (no budget cap)"
else
    ARGS+=(--no-thinking)
    THINK_DESC="off (fast, no reasoning tokens)"
fi

echo "=============================================="
echo " NInfer - Qwen3.8-27B NVFP4 on V100"
echo "=============================================="
echo " artifact : $MODEL"
echo " context  : $CTX  (NVFP4 ceiling; 262144 needs groupwise-int)"
echo " spec     : $SPEC (draft window $DRAFT)"
echo " vision   : $VISION"
echo " thinking : $THINK_DESC"
echo " out-limit: $DEFAULT_MAX_TOKENS tokens when the client omits max_tokens"
echo " kv-dtype : int8 (NVFP4/K8V4 KV is unavailable on Volta)"
echo " parallel : $CONCURRENCY at once; queue up to $MAX_PENDING_REQUESTS"
echo " API      : http://127.0.0.1:$PORT/v1"
echo "=============================================="
echo

cd "$NINFER_HOME" || exit 1
exec "$BIN" "$MODEL" "${ARGS[@]}" ${EXTRA_ARGS:-}
