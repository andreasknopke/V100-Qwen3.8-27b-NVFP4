#!/usr/bin/env bash
# ============================================================================
#  NInfer build prerequisites for a V100 (sm_70) host.
#
#  Installs: build tools + FFmpeg/libcurl dev headers + aria2 + CUDA Toolkit
#  12.8. CUDA 12.8 is mandatory: CUDA 13 dropped offline compilation for Volta.
#
#  Works on BOTH:
#    * native Ubuntu 24.04  -> CUDA repo ubuntu2404/x86_64
#    * Ubuntu 24.04 in WSL2 -> CUDA repo wsl-ubuntu/x86_64, and a pin file keeps
#      any NVIDIA *driver* package out of the distro (WSL uses the Windows host
#      driver; installing a Linux driver here breaks GPU access).
#
#  Run as root, or with sudo available.
# ============================================================================
set -uo pipefail

log() { printf '\n===== %s =====\n' "$*"; }

CUDA_BASE="https://developer.download.nvidia.com/compute/cuda/repos"
if grep -qi microsoft /proc/version 2>/dev/null; then
    CUDA_REPO="$CUDA_BASE/wsl-ubuntu/x86_64"
    IS_WSL=1
else
    CUDA_REPO="$CUDA_BASE/ubuntu2404/x86_64"
    IS_WSL=0
fi

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
    log "privileges: root"
else
    SUDO="sudo"
    log "privileges: sudo"
    sudo -v || { echo "FATAL: sudo authentication failed"; exit 1; }
    ( while true; do sudo -n -v >/dev/null 2>&1 || true; sleep 45; done ) &
    KEEPALIVE=$!
    trap 'kill "$KEEPALIVE" 2>/dev/null' EXIT
fi

log "PHASE 1/4 apt update (repo: $(basename "$(dirname "$CUDA_REPO")"))"
$SUDO apt-get update || { echo "FATAL: apt-get update failed"; exit 1; }

log "PHASE 2/4 build dependencies"
$SUDO apt-get install -y --no-install-recommends \
    cmake ninja-build pkg-config build-essential git curl ca-certificates \
    libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
    libcurl4-openssl-dev aria2 \
    || { echo "FATAL: build dependency install failed"; exit 1; }

log "PHASE 3/4 CUDA 12.8 repository + toolkit (large download, be patient)"
if [ ! -f /etc/apt/sources.list.d/cuda-*-x86_64.list ]; then
    curl -fsSL -o /tmp/cuda-keyring.deb "$CUDA_REPO/cuda-keyring_1.1-1_all.deb" \
        || { echo "FATAL: keyring download failed"; exit 1; }
    $SUDO dpkg -i /tmp/cuda-keyring.deb || { echo "FATAL: keyring install failed"; exit 1; }
    printf 'Package: *\nPin: origin developer.download.nvidia.com\nPin-Priority: 600\n' \
        | $SUDO tee /etc/apt/preferences.d/cuda-repository-pin-600 >/dev/null
    $SUDO apt-get update || { echo "FATAL: apt-get update after keyring failed"; exit 1; }
else
    echo "CUDA repo already configured, skipping keyring step"
fi
$SUDO apt-get install -y cuda-toolkit-12-8 || { echo "FATAL: cuda-toolkit-12-8 install failed"; exit 1; }

log "PHASE 4/4 verification"
if [ "$IS_WSL" = "1" ]; then
    n=$(dpkg -l 2>/dev/null | grep -c 'nvidia-driver-[0-9]')
    echo "Linux driver packages found (must be 0 in WSL): $n"
    [ "$n" != "0" ] && echo "WARNING: a Linux NVIDIA driver is installed in WSL - remove it."
fi
echo "--- nvcc ---"
/usr/local/cuda-12.8/bin/nvcc --version | tail -1
echo "--- toolchain ---"
cmake --version | head -1; ninja --version; pkg-config --version
echo "--- dev libs ---"
for m in libavformat libavcodec libavutil libswscale libcurl; do
    printf '%-12s %s\n' "$m" "$(pkg-config --modversion "$m" 2>/dev/null || echo MISSING)"
done
echo "disk free:"; df -h / | tail -1
echo "PREREQUISITES_DONE"
