#!/usr/bin/env bash
# ============================================================================
#  Clone + build NInfer (ninfer-v100) for sm_70 on the V100.
#
#  Layout:
#    $NINFER_HOME/            repo root
#    $NINFER_HOME/build-v100/ build tree (apps/ninfer, apps/ninfer-serve)
#    $NINFER_HOME/models/     .ninfer artifacts
#
#  Upstream has NO install target: the binaries are run from the build tree.
#  Idempotent: re-running fetches latest master and rebuilds incrementally.
# ============================================================================
set -uo pipefail

REPO_URL="https://github.com/geoffwatts/ninfer-v100.git"
NINFER_HOME="${NINFER_HOME:-$HOME/ninfer}"
SRC_DIR="$NINFER_HOME"
BUILD_DIR="$NINFER_HOME/build-v100"
LOG="$NINFER_HOME/build.log"
JOBS="${JOBS:-8}"            # nvcc uses several GB per job; keep modest on 31 GB RAM

log() { printf '\n===== %s =====\n' "$*"; }

log "PHASE 1/4 source"
mkdir -p "$SRC_DIR/models"
if [ -d "$SRC_DIR/.git" ]; then
    echo "repo present, fetching latest master"
    git -C "$SRC_DIR" fetch --depth 1 origin master && git -C "$SRC_DIR" reset --hard origin/master
    # reset --hard discards the source patches; install.sh re-applies them and
    # rebuilds. If you run this script standalone, re-run the patch scripts too.
else
    git clone --depth 1 "$REPO_URL" "$SRC_DIR" || { echo "FATAL: clone failed"; exit 1; }
fi
git -C "$SRC_DIR" log -1 --format='commit %h  %ad  %s' --date=short

log "PHASE 2/4 CUDA sanity"
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
nvcc --version | tail -1 || { echo "FATAL: nvcc 12.8 not found - run install_deps.sh first"; exit 1; }
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "warning: no GPU visible"

log "PHASE 3/4 configure (Release, CUDA arch 70)"
# BUILD_TESTING must be forced OFF on the command line: FetchContent(cutlass)
# includes CTest and defines the BUILD_TESTING cache variable before the
# project's own option() call, which would otherwise be a no-op. Leaving tests
# on makes them build and fail to link.
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DBUILD_TESTING=OFF \
    -DNINFER_BUILD_BENCHMARKS=OFF \
    > "$SRC_DIR/configure.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "FATAL: cmake configure failed (rc=$rc). Tail of configure.log:"
    tail -40 "$SRC_DIR/configure.log"
    exit 1
fi

log "PHASE 4/4 build apps only (-j $JOBS, full log: $LOG)"
start=$(date +%s)
cmake --build "$BUILD_DIR" --target ninfer ninfer-serve -j "$JOBS" > "$LOG" 2>&1
rc=$?
end=$(date +%s)
printf 'build rc=%s  elapsed=%ss\n' "$rc" "$((end-start))"

if [ $rc -ne 0 ]; then
    echo "--- errors (first 60) ---"
    grep -nE 'error:|Error|FAILED' "$LOG" | head -60
    echo "--- tail ---"; tail -30 "$LOG"
    exit 1
fi

echo "--- produced binaries ---"
ls -lh "$BUILD_DIR/apps/" 2>/dev/null | grep -E 'ninfer' | head
echo "--- sm_70 cubins present? ---"
if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --list-elf "$BUILD_DIR/apps/ninfer-serve" 2>/dev/null | grep -c 'sm_70' | sed 's/^/sm_70 elf entries: /'
fi
echo "BUILD_DONE"
