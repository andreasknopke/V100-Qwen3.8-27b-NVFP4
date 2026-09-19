#!/usr/bin/env bash
# ============================================================================
#  One-shot installer: Qwen3.8-27B NVFP4 on a V100 (sm_70) with NInfer.
#
#  Runs the full recipe end to end on any 64-bit Linux (native Ubuntu 24.04 or
#  Ubuntu 24.04 under WSL2 with a Tesla V100 visible):
#
#    1. system + CUDA 12.8 prerequisites
#    2. clone + build NInfer for sm_70
#    3. apply the two source patches (TCP backpressure fix, low-effort default)
#    4. rebuild the two binaries
#    5. download the official NVFP4 .ninfer artifact (SHA256-verified)
#
#  After this finishes, start the server with:  ./serve.sh
#
#  Idempotent: every phase skips work that is already done. Safe to re-run.
#
#  Environment overrides (all optional):
#    NINFER_HOME   install prefix            (default $HOME/ninfer)
#    JOBS          parallel build jobs       (default 8)
#    SKIP_DOWNLOAD set 1 to skip the model download
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NINFER_HOME="${NINFER_HOME:-$HOME/ninfer}"

log() { printf '\n\033[1;36m########## %s ##########\033[0m\n' "$*"; }

echo "=============================================================="
echo " Qwen3.8-27B NVFP4 on V100 - NInfer installer"
echo "=============================================================="
echo " install prefix : $NINFER_HOME"
echo " source tree    : $HERE"
echo "=============================================================="

# --- 0. environment report --------------------------------------------------
log "PHASE 0  environment"
bash "$HERE/scripts/check_env.sh" || true

# --- 1. prerequisites -------------------------------------------------------
log "PHASE 1  system + CUDA 12.8 prerequisites"
if [ "$(id -u)" -eq 0 ]; then
    bash "$HERE/scripts/install_deps.sh"
else
    bash "$HERE/scripts/install_deps.sh" || {
        echo "install_deps.sh needs root. Re-run as:  sudo ./install.sh"
        echo "(or: sudo bash $HERE/scripts/install_deps.sh  then re-run ./install.sh)"
        exit 1
    }
fi

# --- 2. build ---------------------------------------------------------------
log "PHASE 2  clone + build NInfer (sm_70)"
bash "$HERE/scripts/build_ninfer.sh"

# --- 3. patches -------------------------------------------------------------
log "PHASE 3  source patches"
bash "$HERE/scripts/patch_tcp_user_timeout.sh"
bash "$HERE/scripts/patch_default_low_effort.sh"

# --- 4. rebuild with patches -----------------------------------------------
log "PHASE 4  rebuild binaries with patches"
bash "$HERE/scripts/build_ninfer.sh"

# --- 5. model ---------------------------------------------------------------
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    log "PHASE 5  model download SKIPPED (SKIP_DOWNLOAD=1)"
else
    log "PHASE 5  download official NVFP4 artifact"
    bash "$HERE/scripts/download_model.sh"
fi

log "DONE"
cat <<EOF
Everything is in place. To start the OpenAI-compatible server:

    ./serve.sh

Then test it:

    curl http://127.0.0.1:8084/v1/models
    curl http://127.0.0.1:8084/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'

See README.md for context-size, thinking and LAN details.
EOF
