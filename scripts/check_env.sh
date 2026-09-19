#!/usr/bin/env bash
# ============================================================================
#  Environment probe: report whether this host can build + run NInfer.
#  Non-fatal by design - install.sh calls it with `|| true`.
# ============================================================================
set -uo pipefail

echo "--- OS ---"
if [ -f /etc/os-release ]; then . /etc/os-release; echo "$PRETTY_NAME"; fi
echo "--- kernel ---"; uname -r
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "WSL2: yes (NInfer uses the Windows host GPU driver; do NOT install a Linux driver)"
else
    echo "WSL2: no (native Linux)"
fi

echo "--- GPU ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
else
    echo "nvidia-smi: MISSING (is the NVIDIA driver installed / visible?)"
fi

echo "--- required tools ---"
for t in nvcc cmake ninja g++ git curl pkg-config aria2c python3; do
    p="$(command -v "$t" 2>/dev/null)"; [ -z "$p" ] && p="MISSING"
    printf '%-10s %s\n' "$t" "$p"
done

echo "--- CUDA 12.8 ---"
if [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
    /usr/local/cuda-12.8/bin/nvcc --version | tail -1
else
    echo "/usr/local/cuda-12.8: not installed yet (PHASE 1 installs it)"
fi

echo "--- dev libs (min: avformat>=60 avcodec>=60 avutil>=58 swscale>=7 curl>=7.85) ---"
for m in libavformat libavcodec libavutil libswscale libcurl; do
    if pkg-config --exists "$m" 2>/dev/null; then
        printf '%-12s %s\n' "$m" "$(pkg-config --modversion "$m")"
    else
        printf '%-12s MISSING\n' "$m"
    fi
done

echo "--- disk (need ~50 GiB free for build + 22 GiB model) ---"
df -h "${NINFER_HOME:-$HOME}" 2>/dev/null | tail -1
echo "--- RAM ---"; free -g 2>/dev/null | head -2
