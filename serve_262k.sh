#!/usr/bin/env bash
# ============================================================================
#  NInfer server launcher - NVFP4 at the FULL 262,144 context on a V100 (sm_70).
#
#  This is the "262K start option". Plain NVFP4 caps at 212,992 context because
#  the 20.0 GiB weights leave too little runtime budget. At 262,144 the engine
#  needs ~12.0 GiB of runtime reservation but only ~10.55 GiB is free, so it
#  refuses to load. This launcher applies five memory-fit levers that make the
#  full 262,144 context fit TEXT-ONLY (measured 2026-09-20: runtime 10.6 GiB,
#  ~190 MiB free, MTP intact):
#
#    (1) --kv-capacity $CTX        explicit KV drops the 1 GiB auto headroom
#    (2) --device-state-slots 0    1 state slot instead of 2 (prefix reuse ok)
#    (3) --no-cuda-graph           drops the CUDA-graph allowance
#    (4) --prefill-chunk 1024      shrinks the prefill workspace
#    (5) VISION=0                  frees the ~0.3 GiB vision tower (text-only)
#
#  Concurrency is forced to 1 (NVFP4's fixed minimum reservation only fits one
#  device-state slot at this context). Vision at 262,144 would need --spec none
#  (slow) or prefill 512 (razor-thin), so it defaults OFF; pass VISION=1 and
#  keep ctx <= 212992 for a comfortable fit.
#
#  Usage: ./serve_262k.sh [CTXSIZE] [SPEC] [DRAFT] [VISION] [PORT] [THINKING] [ARTIFACT]
#    CTXSIZE      logical context ceiling   (default 262144)
#    SPEC         none | mtp | dflash2      (default mtp)
#    DRAFT        draft window 1-7 (mtp) / 1-15 (dflash2)   (default 3)
#    VISION       1 | 0                     (default 0 = text-only at 262144)
#    PORT         HTTP port                 (default 8084)
#    THINKING     1 | 0                     (default 1 = thinking ON, low effort)
#    ARTIFACT     nvfp4 | cf-nvfp4 | <path> (default nvfp4)
#                  nvfp4     -> official Qwen3.8-27B NVFP4 (~20 GiB, ~66 tok/s)
#                  cf-nvfp4  -> Cold-Fusion GAIN NVFP4 (~21.5 GiB, ~54 tok/s)
#
#  Env overrides: NINFER_HOME, NINFER_MODEL, EXTRA_ARGS, DEFAULT_MAX_TOKENS,
#                 MAX_PENDING_REQUESTS, PENDING_TIMEOUT_MS,
#                 NINFER_TCP_USER_TIMEOUT_MS (see the TCP patch).
# ============================================================================
set -uo pipefail

CTX="${1:-262144}"
SPEC="${2:-mtp}"
DRAFT="${3:-3}"
VISION="${4:-0}"
PORT="${5:-8084}"
THINKING="${6:-1}"
ARTIFACT="${7:-nvfp4}"

DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-32768}"
MAX_PENDING_REQUESTS="${MAX_PENDING_REQUESTS:-16}"
PENDING_TIMEOUT_MS="${PENDING_TIMEOUT_MS:-600000}"

NINFER_HOME="${NINFER_HOME:-$HOME/ninfer}"

# --- resolve the weight artifact -------------------------------------------
case "${ARTIFACT,,}" in
    nvfp4|nv|n4)
        MODEL="$NINFER_HOME/models/qwen3_8_27b_nvfp4.ninfer" ;;
    cf-nvfp4|cf-nv|cfnvfp4|coldfusion-nvfp4|cold-nvfp4|gain-nvfp4)
        MODEL="$NINFER_HOME/models/qwen3_8_27b_coldfusion_nvfp4.ninfer" ;;
    /*)
        MODEL="$ARTIFACT" ;;
    *)
        MODEL="$NINFER_HOME/models/$ARTIFACT" ;;
esac
MODEL="${NINFER_MODEL:-$MODEL}"

BIN="$NINFER_HOME/build-v100/apps/ninfer-serve"

if [ ! -x "$BIN" ]; then
    echo "FATAL: $BIN not found or not executable."
    echo "Build it first:  bash scripts/build_ninfer.sh"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "FATAL: artifact not found: $MODEL"
    echo "Official NVFP4:  bash scripts/download_model.sh"
    echo "Cold-Fusion NVFP4: bash scripts/download_coldfusion.sh && bash scripts/build_coldfusion_nvfp4.sh"
    exit 1
fi

# --- 262,144 memory-fit levers (see header) ---------------------------------
CONCURRENCY=1
PREFILL_CHUNK=1024
KVCAP="$CTX"
DEVICE_STATE_SLOTS=0
USE_CUDA_GRAPH=0

ARGS=(
    --host 0.0.0.0
    --port "$PORT"
    --max-context "$CTX"
    --prefill-chunk "$PREFILL_CHUNK"
    --kv-capacity "$KVCAP"
    --max-concurrency "$CONCURRENCY"
    --device-state-slots "$DEVICE_STATE_SLOTS"
    --max-pending-requests "$MAX_PENDING_REQUESTS"
    --pending-timeout-ms "$PENDING_TIMEOUT_MS"
    --kv-dtype int8
    --host-state-slots 8
    --host-kv-mib 8192
    --default-max-tokens "$DEFAULT_MAX_TOKENS"
    --no-cuda-graph
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
echo " NInfer - Qwen3.8-27B NVFP4 @ 262,144 ctx (V100)"
echo "=============================================="
echo " artifact : $MODEL"
echo " context  : $CTX  (full 262K via memory-fit levers)"
echo " spec     : $SPEC (draft window $DRAFT)"
echo " vision   : $VISION  (text-only at 262144)"
echo " thinking : $THINK_DESC"
echo " out-limit: $DEFAULT_MAX_TOKENS tokens when the client omits max_tokens"
echo " kv       : $KVCAP (int8, explicit - no auto headroom)"
echo " prefill  : chunk $PREFILL_CHUNK"
echo " graph    : off"
echo " parallel : $CONCURRENCY (forced; NVFP4 min reservation)"
echo " API      : http://127.0.0.1:$PORT/v1"
echo " LAN      : http://0.0.0.0:$PORT/v1"
echo "=============================================="
echo

cd "$NINFER_HOME" || exit 1
exec "$BIN" "$MODEL" "${ARGS[@]}" ${EXTRA_ARGS:-}
